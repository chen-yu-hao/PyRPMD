"""Synchronous i-PI force-server adapter for existing driver executables.

One initialized driver per context; parallel bead workers use distinct ports or
Unix addresses. Coordinates/forces use the i-PI atomic-unit wire protocol.
"""

import json
import socket
import struct
import time
import numpy as np
from .base import Potential, PESResult, PotentialError, coordinates, validate_result


class IPIPotential(Potential):
    metadata = {
        "provider": "ipi",
        "thread_safety": "context_serial",
        "execution_device": "external",
        "requires_bead_affinity": True,
    }

    def __init__(
        self, host="127.0.0.1", port=31415, *, unix=None, timeout=60.0, init=None
    ):
        if timeout <= 0:
            raise ValueError("socket timeout must be positive")
        self.timeout = float(timeout)
        self.unix = unix
        self.connection = None
        self.closed = False
        self.init = json.dumps(init or {}).encode()
        self.server = socket.socket(
            socket.AF_UNIX if unix else socket.AF_INET, socket.SOCK_STREAM
        )
        self.server.settimeout(self.timeout)
        if unix:
            self.server.bind(str(unix))
        else:
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind((host, int(port)))
        self.server.listen(1)
        self.address = self.server.getsockname()

    def _recv(self, n):
        out = bytearray()
        while len(out) < n:
            chunk = self.connection.recv(n - len(out))
            if not chunk:
                raise PotentialError("i-PI driver disconnected")
            out.extend(chunk)
        return bytes(out)

    def _send_header(self, name):
        self.connection.sendall(name.encode().ljust(12, b" "))

    def _status(self):
        self._send_header("STATUS")
        return self._recv(12).strip().decode("ascii")

    def evaluate(self, x):
        if self.closed:
            raise PotentialError("i-PI provider is closed")
        x = coordinates(x)
        try:
            if self.connection is None:
                self.connection, _ = self.server.accept()
                self.connection.settimeout(self.timeout)
            es = []
            fs = []
            for bead, frame in enumerate(x):
                status = self._status()
                if status == "NEEDINIT":
                    self._send_header("INIT")
                    self.connection.sendall(
                        struct.pack("=ii", bead, len(self.init)) + self.init
                    )
                    status = self._status()
                if status != "READY":
                    raise PotentialError(f"i-PI expected READY, received {status}")
                self._send_header("POSDATA")
                # The molecule is nonperiodic. Driver must use the nonperiodic
                # setting supplied in init; these invertible cells are transport.
                cell = np.eye(3) * 1e6
                inverse = np.eye(3) * 1e-6
                self.connection.sendall(
                    cell.astype("=f8").tobytes()
                    + inverse.astype("=f8").tobytes()
                    + struct.pack("=i", len(frame))
                    + frame.astype("=f8").tobytes()
                )
                deadline = time.monotonic() + self.timeout
                while True:
                    status = self._status()
                    if status == "HAVEDATA":
                        break
                    if status not in ("BUSY", "READY"):
                        raise PotentialError(f"unexpected i-PI status {status}")
                    if time.monotonic() > deadline:
                        raise PotentialError("i-PI driver force timeout")
                    time.sleep(0.001)
                self._send_header("GETFORCE")
                if self._recv(12).strip() != b"FORCEREADY":
                    raise PotentialError("invalid i-PI force header")
                energy = struct.unpack("=d", self._recv(8))[0]
                n = struct.unpack("=i", self._recv(4))[0]
                if n != len(frame):
                    raise PotentialError("i-PI driver atom count mismatch")
                force = (
                    np.frombuffer(self._recv(n * 3 * 8), dtype="=f8")
                    .reshape(n, 3)
                    .copy()
                )
                self._recv(9 * 8)
                extra = struct.unpack("=i", self._recv(4))[0]
                if not 0 <= extra <= 16 * 1024 * 1024:
                    raise PotentialError("invalid i-PI extra payload size")
                self._recv(extra)
                es.append(energy)
                fs.append(force)
            return validate_result(PESResult(es, fs), x.shape)
        except (OSError, UnicodeError, struct.error) as exc:
            raise PotentialError(f"i-PI communication failed: {exc}") from exc

    def close(self):
        if self.closed:
            return
        if self.connection:
            try:
                self._send_header("EXIT")
            except OSError:
                pass
            self.connection.close()
        self.server.close()
        if self.unix:
            from pathlib import Path

            Path(self.unix).unlink(missing_ok=True)
        self.closed = True
