"""Protocol-level tests with real loopback clients, not mocked futures."""
from concurrent.futures import ThreadPoolExecutor
import json
import socket
import struct
import threading
import time

import numpy as np
import pytest

from pyrpmd.potentials import PotentialError, make_potential
from pyrpmd.potentials.socket import IPIPotential


def _recv(connection, length):
    data = bytearray()
    while len(data) < length:
        part = connection.recv(length - len(data))
        if not part:
            raise EOFError()
        data.extend(part)
    return bytes(data)


def _client(pes, *, before_force=None, fragment=False, repeat_init=False,
            fail=None, seen=None):
    with socket.create_connection(pes.address, timeout=3) as connection:
        state = b"NEEDINIT"
        frame = None
        while True:
            try:
                header = _recv(connection, 12).strip()
            except (EOFError, ConnectionResetError):
                return
            if header == b"STATUS":
                connection.sendall(state.ljust(12))
            elif header == b"INIT":
                bead, length = struct.unpack("=ii", _recv(connection, 8))
                parameters = json.loads(_recv(connection, length))
                if seen is not None:
                    seen.append(("init", bead, parameters))
                state = b"READY"
            elif header == b"POSDATA":
                cell = np.frombuffer(_recv(connection, 72), dtype="=f8").reshape(3, 3)
                inverse = np.frombuffer(_recv(connection, 72), dtype="=f8").reshape(3, 3)
                np.testing.assert_allclose(cell @ inverse, np.eye(3), atol=1e-14)
                count = struct.unpack("=i", _recv(connection, 4))[0]
                frame = np.frombuffer(_recv(connection, count * 24), dtype="=f8").reshape(count, 3)
                if seen is not None:
                    seen.append(("frame", frame.copy(), cell.copy()))
                if before_force is not None:
                    before_force(frame)
                if fail == "disconnect":
                    return
                state = b"HAVEDATA"
            elif header == b"GETFORCE":
                if fail == "header":
                    connection.sendall(b"BADHEADER".ljust(12))
                    continue
                if fail == "truncate":
                    connection.sendall(b"FORCEREADY".ljust(12) + b"\0")
                    # Wait for the server's deadline/abort instead of returning
                    # a complete float or voluntarily disconnecting.
                    try:
                        _recv(connection, 12)
                    except (EOFError, ConnectionResetError):
                        return
                    continue
                energy = .5 * np.sum(frame * frame)
                force = -frame
                virial = np.arange(9.).reshape(3, 3) + frame[0, 0]
                payload = (b"FORCEREADY".ljust(12) + struct.pack("=di", energy, len(frame))
                           + force.astype("=f8").tobytes() + virial.T.astype("=f8").tobytes()
                           + struct.pack("=i", 3) + b"{} ")
                try:
                    if fragment:
                        for begin in range(0, len(payload), 7):
                            connection.sendall(payload[begin:begin + 7])
                    else:
                        connection.sendall(payload)
                except (BrokenPipeError, ConnectionResetError):
                    return
                state = b"NEEDINIT" if repeat_init else b"READY"
            elif header == b"EXIT":
                return
            else:
                raise AssertionError(header)


def test_multiple_clients_are_concurrent_ordered_and_persistent():
    cell = np.array([[8., 0., 0.], [1., 9., 0.], [.3, .4, 10.]])
    pes = make_potential({"provider": "ipi", "params": {"port": 0, "clients": 2, "timeout": 3}},
                         {"cell": cell, "pbc": [True] * 3})
    barrier = threading.Barrier(2)
    first = threading.local()

    def concurrent(frame):
        if not getattr(first, "entered", False):
            first.entered = True
            # A serial bead loop cannot satisfy this barrier.
            barrier.wait(timeout=2)

    seen = [[], []]
    with ThreadPoolExecutor(2) as drivers:
        futures = [drivers.submit(_client, pes, before_force=concurrent,
                                  fragment=True, repeat_init=True, seen=record) for record in seen]
        try:
            for offset in (0., 100.):
                q = np.arange(60.).reshape(10, 2, 3) + offset
                result = pes.evaluate(q)
                np.testing.assert_array_equal(result.forces, -q)
                np.testing.assert_allclose(result.energy, .5 * np.sum(q * q, axis=(1, 2)))
                np.testing.assert_array_equal(result.virial,
                    np.arange(9.).reshape(1, 3, 3) + q[:, 0, 0, None, None])
            assert len(pes._drivers) == 2
        finally:
            pes.close()
        for future in futures:
            future.result(timeout=4)
    assert all(any(item[0] == "frame" for item in record) for record in seen)
    for record in seen:
        for kind, _, payload in record:
            if kind == "frame":
                np.testing.assert_array_equal(payload, cell.T)
    with pytest.raises(PotentialError, match="closed"):
        pes.evaluate(np.zeros((1, 2, 3)))


def test_fast_client_takes_more_work_than_slow_client():
    pes = IPIPotential(port=0, clients=2, timeout=3)
    barrier = threading.Barrier(2)
    fast_finished_second = threading.Event()
    counts = [0, 0]

    def slow(frame):
        counts[0] += 1
        if counts[0] == 1:
            barrier.wait(timeout=2)
            assert fast_finished_second.wait(2)

    def fast(frame):
        counts[1] += 1
        if counts[1] == 1:
            barrier.wait(timeout=2)
        if counts[1] == 2:
            fast_finished_second.set()

    with ThreadPoolExecutor(2) as drivers:
        futures = [drivers.submit(_client, pes, before_force=slow),
                   drivers.submit(_client, pes, before_force=fast)]
        try:
            q = np.arange(24.).reshape(8, 1, 3)
            np.testing.assert_array_equal(pes.evaluate(q).forces, -q)
        finally:
            pes.close()
        for future in futures:
            future.result(timeout=4)
    assert counts[1] >= 2
    assert sum(counts) == len(q)


@pytest.mark.parametrize("failure", ["header", "disconnect", "truncate"])
def test_bad_client_closes_entire_pool_and_cannot_return_stale_forces(failure):
    pes = IPIPotential(port=0, clients=2, timeout=.4)
    barrier = threading.Barrier(2)

    def started(frame):
        barrier.wait(timeout=2)

    with ThreadPoolExecutor(2) as drivers:
        futures = [drivers.submit(_client, pes, before_force=started, fail=failure),
                   drivers.submit(_client, pes, before_force=started)]
        try:
            with pytest.raises(PotentialError):
                pes.evaluate(np.ones((2, 1, 3)))
            assert pes.closed
            assert all(driver.connection.fileno() == -1 for driver in pes._drivers)
            with pytest.raises(PotentialError, match="closed"):
                pes.evaluate(np.ones((2, 1, 3)))
        finally:
            pes.close()
        for future in futures:
            future.result(timeout=3)


def test_missing_clients_timeout_and_close_listener():
    pes = IPIPotential(port=0, clients=2, timeout=.1)
    with socket.create_connection(pes.address):
        with pytest.raises(PotentialError, match="waiting for clients"):
            pes.evaluate(np.ones((1, 1, 3)))
    assert pes.closed
    assert pes.server.fileno() == -1


@pytest.mark.parametrize("clients", [0, -1, True, 1.5, "2"])
def test_invalid_client_counts_rejected(clients):
    with pytest.raises(ValueError, match="positive integer"):
        IPIPotential(port=0, clients=clients)


def test_existing_unix_address_is_not_removed_on_bind_failure(tmp_path):
    address = tmp_path / "driver.sock"
    pes = IPIPotential(unix=address)
    try:
        with pytest.raises(OSError):
            IPIPotential(unix=address)
        assert address.exists()
    finally:
        pes.close()
    assert not address.exists()


@pytest.mark.parametrize("cell,declared", [(np.eye(3), [False] * 3), (None, [True] * 3)])
def test_contradictory_init_pbc_rejected_before_opening_server(cell, declared):
    from unittest.mock import patch

    with patch("pyrpmd.potentials.socket.socket.socket") as listener:
        with pytest.raises(ValueError, match="PBC flags"):
            IPIPotential(port=0, cell=cell, init={"pbc": declared})
        listener.assert_not_called()


def test_close_while_accept_completes_does_not_leak_the_new_connection():
    """Deterministically place close between accept entry and its return."""
    pes = IPIPotential(port=0, timeout=1)
    entered, release = threading.Event(), threading.Event()
    accepted, peer = socket.socketpair()

    class DelayedListener:
        def settimeout(self, timeout):
            pass

        def accept(self):
            entered.set()
            assert release.wait(3)
            return accepted, None

        def close(self):
            # An accept already completing may still hand back a connection.
            pass

    pes.server.close()
    pes.server = DelayedListener()
    errors = []

    def evaluate():
        try:
            pes.evaluate(np.zeros((1, 1, 3)))
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=evaluate)
    worker.start()
    try:
        assert entered.wait(3)
        pes.close()
        release.set()
        worker.join(3)
        assert not worker.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], PotentialError)
        assert "closed" in str(errors[0])
        assert accepted.fileno() == -1
        assert pes.closed
    finally:
        release.set()
        worker.join(3)
        accepted.close()
        peer.close()
        pes.close()


def test_rebind_cannot_mix_cells_inside_one_batch_and_rejects_closed():
    cell = np.eye(3) * 8
    pes = IPIPotential(port=0, cell=cell)
    entered, release = threading.Event(), threading.Event()
    seen = []

    class BlockingDriver:
        def evaluate(self, bead, frame, cell_payload, init, timeout):
            seen.append(np.frombuffer(cell_payload[:72], dtype="=f8").reshape(3, 3).copy())
            if bead == 0:
                entered.set()
                assert release.wait(3)
            return 0., np.zeros_like(frame), np.zeros((3, 3))

        def close(self, graceful):
            pass

    pes._drivers.append(BlockingDriver())
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(pes.evaluate, np.ones((3, 1, 3)))
        try:
            assert entered.wait(3)
            with pytest.raises(PotentialError, match="active evaluation"):
                pes.bind_system({"cell": 2 * cell, "pbc": [True] * 3})
            release.set()
            future.result(timeout=3)
            assert len(seen) == 3
            for actual in seen:
                np.testing.assert_array_equal(actual, cell.T)
            pes.bind_system({"cell": 2 * cell, "pbc": [True] * 3})
            np.testing.assert_array_equal(pes.cell, 2 * cell)
        finally:
            release.set()
            pes.close()
    with pytest.raises(PotentialError, match="closed"):
        pes.bind_system({"cell": cell, "pbc": [True] * 3})


def test_evaluate_rechecks_close_after_coordinate_preparation(monkeypatch):
    from pyrpmd.potentials import socket as socket_provider

    pes = IPIPotential(port=0)
    original = socket_provider.coordinates

    def prepare_and_close(value):
        result = original(value)
        pes.close()
        return result

    monkeypatch.setattr(socket_provider, "coordinates", prepare_and_close)
    try:
        with pytest.raises(PotentialError, match="closed"):
            pes.evaluate(np.zeros((1, 1, 3)))
        assert pes.server.fileno() == -1
        assert not pes._drivers
    finally:
        pes.close()


def test_accept_oserror_closes_provider_and_uses_potential_error():
    pes = IPIPotential(port=0)

    class FailedListener:
        closed = False

        def settimeout(self, timeout):
            pass

        def accept(self):
            raise OSError("listener failed")

        def close(self):
            self.closed = True

    pes.server.close()
    pes.server = FailedListener()
    try:
        with pytest.raises(PotentialError, match="communication failed.*listener failed"):
            pes.evaluate(np.zeros((1, 1, 3)))
        assert pes.closed and pes.server.closed
    finally:
        pes.close()
