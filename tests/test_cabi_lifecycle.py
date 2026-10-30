"""Exercise context shutdown interleavings without calling a freed C pointer."""

import ctypes
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import pyrpmd
from pyrpmd.potentials import cabi


def _fake_potential():
    """Only context lifecycle is relevant; no native library is needed."""
    pes = cabi.CABIPotential.__new__(cabi.CABIPotential)
    pes.closed = False
    pes.natoms = 1
    pes.metadata = {"capabilities": {"stateful": False}}
    pes._context = ctypes.c_void_p(0x1234)
    pes._lock = threading.RLock()
    destroyed = []
    pes._destroy = lambda context: destroyed.append(context.value)
    pes._evaluate = ctypes.c_void_p(1)
    return pes, destroyed


def test_exported_binding_rejects_context_as_soon_as_close_unlocks(monkeypatch):
    released, resume = threading.Event(), threading.Event()

    class PauseAfterRelease:
        """Stop the closer exactly where another thread can acquire the lock."""

        def __init__(self):
            self.lock = threading.RLock()

        def __enter__(self):
            self.lock.acquire()
            return self

        def __exit__(self, *_):
            self.lock.release()
            if threading.current_thread() is closer:
                released.set()
                if not resume.wait(5):
                    raise RuntimeError("test closer was not resumed")

    pes, destroyed = _fake_potential()
    pes._lock = PauseAfterRelease()
    binding = pes.native_binding()
    bound = []
    monkeypatch.setattr(
        pyrpmd, "_native",
        SimpleNamespace(bind_pes=lambda *args: bound.append(args) or object()),
        raising=False,
    )
    errors = []

    def close():
        try:
            pes.close()
        except BaseException as exc:
            errors.append(exc)

    closer = threading.Thread(target=close)
    closer.start()
    try:
        assert released.wait(5)
        assert destroyed == [0x1234]
        with pytest.raises(RuntimeError, match="closed"):
            with binding.acquire():
                pass
        assert not bound
    finally:
        resume.set()
        closer.join(5)
    assert not closer.is_alive()
    assert not errors
    pes.close()
    assert destroyed == [0x1234]


def test_evaluation_rechecks_lifetime_after_input_preparation(monkeypatch):
    pes, destroyed = _fake_potential()
    original_coordinates = cabi.coordinates
    calls = []

    def prepare_and_close(q):
        result = original_coordinates(q)
        # Deterministically model close() winning the lock while evaluate()
        # prepares its arguments, after evaluate's preliminary closed check.
        pes.close()
        return result

    def unexpected_evaluate(*args):
        calls.append(args)
        raise AssertionError("native evaluation ran after context destruction")

    pes._evaluate = unexpected_evaluate
    monkeypatch.setattr(cabi, "coordinates", prepare_and_close)
    with pytest.raises(cabi.PotentialError, match="closed"):
        pes.evaluate(np.zeros((1, 1, 3)))
    assert destroyed == [0x1234]
    assert not calls


def test_batched_messages_copy_buffer_once_preserving_order_and_failed_frames(monkeypatch):
    """Fixed-width decoding must be linear in the batch's message storage."""
    pes, _ = _fake_potential()
    payloads = [b"", b"frame one", "核距离".encode(), b"bad \xff byte",
                b"x"*511, b"", b"last frame"]
    n = len(payloads)
    copies = []
    original = ctypes.create_string_buffer

    def tracked_buffer(init, size=None):
        buffer = original(init) if size is None else original(init, size)
        if init == n*512 and size is None:
            class Tracked(type(buffer)):
                pass
            def read_raw(self):
                copies.append(len(self))
                return bytes(self)
            # ctypes installs its raw descriptor while constructing the type.
            Tracked.raw = property(read_raw)
            return Tracked()
        return buffer

    def evaluate(context, input_pointer, output_pointer, error_pointer):
        out = ctypes.cast(output_pointer, ctypes.POINTER(cabi.Output)).contents
        for i, message in enumerate(payloads):
            failed = i in (2, 3)
            out.energies[i] = np.nan if failed else i+.25
            for j in range(3):
                out.forces[3*i+j] = np.nan if failed else -float(i)
            out.structure_status[i] = 2 if failed else 0
            ctypes.memmove(out.structure_messages+i*out.structure_message_stride, message+b"\0", len(message)+1)
        return 0

    pes._evaluate = evaluate
    monkeypatch.setattr(ctypes, "create_string_buffer", tracked_buffer)
    try:
        result = pes.evaluate(np.zeros((n, 1, 3)))
    finally:
        pes.close()
    assert copies == [n*512]
    assert result.messages == tuple(message.decode(errors="replace") for message in payloads)
    np.testing.assert_array_equal(result.status, [0, 0, 2, 2, 0, 0, 0])
    assert np.isnan(result.energy[[2, 3]]).all()
    np.testing.assert_array_equal(result.energy[[0, 1, 4, 5, 6]], [.25, 1.25, 4.25, 5.25, 6.25])
