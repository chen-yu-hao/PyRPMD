"""Persistent i-PI force server with concurrent, dynamically assigned clients.

Each connection owns at most one in-flight geometry.  Results are returned only
when the complete batch has succeeded, in input order, using the standard i-PI
atomic-unit wire protocol (one structure per POSDATA).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from queue import Empty, Queue
import socket
import struct
import threading
import time

import numpy as np

from .base import Potential, PESResult, PotentialError, coordinates, validate_result
from ..geometry import validate_cell_pbc


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise PotentialError("i-PI driver force timeout")
    return remaining


class _Driver:
    """Connection-local protocol state; never shared by simultaneous requests."""

    def __init__(self, connection):
        self.connection = connection

    def recv(self, n, deadline):
        data = bytearray()
        while len(data) < n:
            self.connection.settimeout(_remaining(deadline))
            block = self.connection.recv(n - len(data))
            if not block:
                raise PotentialError("i-PI driver disconnected")
            data.extend(block)
        return bytes(data)

    def send(self, data, deadline):
        self.connection.settimeout(_remaining(deadline))
        self.connection.sendall(data)

    def header(self, name, deadline):
        self.send(name.encode("ascii").ljust(12, b" "), deadline)

    def status(self, deadline):
        self.header("STATUS", deadline)
        return self.recv(12, deadline).strip().decode("ascii")

    def evaluate(self, bead, frame, cell_payload, init, timeout):
        # An absolute per-geometry deadline also bounds clients that trickle
        # partial replies or continuously answer READY without producing data.
        deadline = time.monotonic() + timeout
        status = self.status(deadline)
        if status == "NEEDINIT":
            self.header("INIT", deadline)
            self.send(struct.pack("=ii", bead, len(init)) + init, deadline)
            status = self.status(deadline)
        if status != "READY":
            raise PotentialError(f"i-PI expected READY, received {status}")
        self.header("POSDATA", deadline)
        self.send(cell_payload + struct.pack("=i", len(frame))
                  + frame.astype("=f8").tobytes(), deadline)
        while True:
            status = self.status(deadline)
            if status == "HAVEDATA":
                break
            if status not in ("BUSY", "READY"):
                raise PotentialError(f"unexpected i-PI status {status}")
            time.sleep(min(0.001, _remaining(deadline)))
        self.header("GETFORCE", deadline)
        if self.recv(12, deadline).strip() != b"FORCEREADY":
            raise PotentialError("invalid i-PI force header")
        energy = struct.unpack("=d", self.recv(8, deadline))[0]
        count = struct.unpack("=i", self.recv(4, deadline))[0]
        if count != len(frame):
            raise PotentialError("i-PI driver atom count mismatch")
        force = np.frombuffer(self.recv(count * 24, deadline), dtype="=f8").reshape(count, 3).copy()
        # Match the existing i-PI/ASE tensor wire convention, W=-V*stress.
        virial = np.frombuffer(self.recv(72, deadline), dtype="=f8").reshape(3, 3).T.copy()
        extra = struct.unpack("=i", self.recv(4, deadline))[0]
        if not 0 <= extra <= 16 * 1024 * 1024:
            raise PotentialError("invalid i-PI extra payload size")
        self.recv(extra, deadline)
        return energy, force, virial

    def close(self, graceful):
        if graceful:
            try:
                self.connection.settimeout(0.1)
                self.connection.sendall(b"EXIT".ljust(12))
            except OSError:
                pass
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.connection.close()


class IPIPotential(Potential):
    metadata = {
        "provider": "ipi",
        "thread_safety": "context_serial",
        "execution_device": "external",
        "requires_bead_affinity": True,
        "periodic": True,
        "supports_pbc": True,
    }

    def __init__(
        self, host="127.0.0.1", port=31415, *, unix=None, timeout=60.0,
        init=None, cell=None, pbc=None, clients=1,
    ):
        if isinstance(timeout, (bool, np.bool_)) or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("socket timeout must be positive")
        if isinstance(clients, (bool, np.bool_)) or not isinstance(clients, (int, np.integer)) or clients < 1:
            raise ValueError("i-PI clients must be a positive integer")
        self.timeout = float(timeout)
        self.clients = int(clients)
        self.unix = unix
        self.closed = False
        self._lifecycle_lock = threading.Lock()
        self._drivers = []
        self._parameters = dict(init or {})
        self._set_cell(cell, pbc)
        self.init = json.dumps(self._parameters).encode()
        self._evaluation_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=self.clients, thread_name_prefix="pyrpmd-ipi")
        self.server = socket.socket(socket.AF_UNIX if unix else socket.AF_INET, socket.SOCK_STREAM)
        self._owns_unix = False
        try:
            self.server.settimeout(self.timeout)
            if unix:
                self.server.bind(str(unix))
                self._owns_unix = True
            else:
                self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server.bind((host, int(port)))
            self.server.listen(self.clients)
            self.address = self.server.getsockname()
        except BaseException:
            self.close()
            raise
        self.metadata = {**type(self).metadata, "clients": self.clients,
                         "batch_parallel": self.clients > 1}

    @property
    def connection(self):
        """The first driver, for compatibility with single-client callers."""
        return self._drivers[0].connection if self._drivers else None

    def _set_cell(self, cell, pbc):
        cell, pbc = validate_cell_pbc(cell, pbc)
        if "pbc" in self._parameters or (cell is not None and not pbc.all()):
            declared = np.asarray(self._parameters.get("pbc"))
            if declared.shape != (3,) or declared.dtype.kind != "b" or not np.array_equal(declared, pbc):
                raise ValueError(
                    "i-PI POSDATA has no PBC flags: init.pbc must match System flags; "
                    "partial/nonperiodic cells require an explicit declaration "
                    "and a driver configured to honor it"
                )
        if self._drivers and not np.array_equal(self.pbc, pbc):
            raise PotentialError("Cannot change i-PI PBC flags after driver initialization")
        self.cell, self.pbc = cell, pbc
        # i-PI h stores lattice vectors as columns; ih is its actual inverse.
        self._wire_cell = np.eye(3) * 1e6 if cell is None else cell.T.copy()
        self._wire_inverse = np.linalg.inv(self._wire_cell)
        self._cell_payload = (self._wire_cell.astype("=f8").tobytes()
                              + self._wire_inverse.astype("=f8").tobytes())

    def bind_system(self, system):
        if not self._evaluation_lock.acquire(blocking=False):
            raise PotentialError("Cannot bind an i-PI system during an active evaluation")
        try:
            with self._lifecycle_lock:
                if self.closed:
                    raise PotentialError("i-PI provider is closed")
                get = system.get if isinstance(system, dict) else lambda key: getattr(system, key, None)
                self._set_cell(get("cell"), get("pbc"))
        finally:
            self._evaluation_lock.release()

    def _accept_clients(self):
        deadline = time.monotonic() + self.timeout
        while True:
            with self._lifecycle_lock:
                if self.closed:
                    raise PotentialError("i-PI provider is closed")
                if len(self._drivers) >= self.clients:
                    return
            self.server.settimeout(_remaining(deadline))
            try:
                connection, _ = self.server.accept()
            except TimeoutError as exc:
                raise PotentialError(
                    f"i-PI timed out waiting for clients ({len(self._drivers)}/{self.clients} connected)"
                ) from exc
            # close() may run while accept() blocks. Never append an unowned
            # connection after the shutdown path has taken its snapshot.
            with self._lifecycle_lock:
                if self.closed:
                    connection.close()
                    raise PotentialError("i-PI provider closed while accepting a client")
                self._drivers.append(_Driver(connection))

    def evaluate(self, x):
        if self.closed:
            raise PotentialError("i-PI provider is closed")
        x = coordinates(x)
        if not self._evaluation_lock.acquire(blocking=False):
            raise PotentialError("Concurrent evaluate calls on one i-PI provider are unsupported")
        try:
            with self._lifecycle_lock:
                if self.closed:
                    raise PotentialError("i-PI provider is closed")
            self._accept_clients()
            cell_payload, init = self._cell_payload, self.init
            energy = np.empty(len(x))
            force = np.empty_like(x)
            virial = np.empty((len(x), 3, 3))
            active = min(self.clients, len(x))
            pending = Queue()
            for bead in range(active, len(x)):
                pending.put(bead)

            def work(client):
                driver = self._drivers[client]
                bead = client
                while True:
                    try:
                        result = driver.evaluate(bead, x[bead], cell_payload,
                                                 init, self.timeout)
                        energy[bead], force[bead], virial[bead] = result
                    except Exception as exc:
                        raise PotentialError(f"i-PI client {client}, frame {bead}: {exc}") from exc
                    try:
                        bead = pending.get_nowait()
                    except Empty:
                        return

            futures = [self._pool.submit(work, client) for client in range(active)]
            for future in as_completed(futures):
                future.result()
            return validate_result(PESResult(energy, force, virial=virial), x.shape)
        except BaseException as exc:
            # Poison the context after any incomplete batch: shutdown wakes
            # blocked peers before joining threads and prevents stale responses
            # from being admitted on a later evaluate call.
            self._close(graceful=False)
            if isinstance(exc, (OSError, RuntimeError)) and not isinstance(exc, PotentialError):
                raise PotentialError(f"i-PI communication failed: {exc}") from exc
            raise
        finally:
            self._evaluation_lock.release()

    def _close(self, *, graceful):
        with self._lifecycle_lock:
            if self.closed:
                return
            self.closed = True
            drivers = tuple(self._drivers)
        for driver in drivers:
            driver.close(graceful)
        self.server.close()
        self._pool.shutdown(wait=True, cancel_futures=True)
        if self.unix and self._owns_unix:
            from pathlib import Path
            Path(self.unix).unlink(missing_ok=True)

    def close(self):
        self._close(graceful=True)
