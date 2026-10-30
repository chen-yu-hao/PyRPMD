"""Both native APIs preserve Python overrides and their distinct failure contracts."""
from pathlib import Path
import ctypes
import shutil
import subprocess
import threading

import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics, CayleyIntegrator, State, native_capabilities
from pyrpmd.cv import RPMDrateCoordinate, Cartesian, CVStack
from pyrpmd.bias import UmbrellaBias, CompositeBias
from pyrpmd.potentials import CABIPotential, HarmonicPotential, PythonPotential
from pyrpmd.potentials.native import NativePESBinding as DescriptorBinding, descriptor_binding
from pyrpmd.propagation import (
    NativePESBinding as CapsuleBinding, NativeUnavailableError, PropagationError,
    _atomic_binding, cv_descriptor,
)
from pyrpmd.block_propagation import cv_descriptor as block_cv_descriptor


@pytest.fixture
def cabi_pes(tmp_path):
    if not native_capabilities()["available"] or shutil.which("cc") is None:
        pytest.skip("compiled native kernel and C compiler required")
    root = Path(__file__).resolve().parents[1]
    library = tmp_path / "harmonic.so"
    subprocess.run([
        "cc", "-shared", "-fPIC", "-I", str(root / "native/include"),
        str(root / "native/examples/harmonic_pes.c"), "-o", str(library),
    ], check=True)
    pes = CABIPotential(library)
    yield pes
    pes.close()


def _state():
    return State([[[0.2, 0.0, 0.0]]], [[[0.1, 0.0, 0.0]]])


def _advance(dynamics, state, api, **kwargs):
    return getattr(dynamics, api)(state, 3, **kwargs)


@pytest.mark.parametrize("api", ["propagate", "propagate_block"])
@pytest.mark.parametrize("override", ["subclass", "instance", "class"])
def test_both_apis_preserve_custom_dynamics(cabi_pes, api, override, monkeypatch):
    calls = []

    class ChangedDynamics(Dynamics):
        def _force(self, q, bias=None):
            calls.append(True)
            return 0.25 * super()._force(q, bias)

    cls = ChangedDynamics if override == "subclass" else Dynamics
    dyn = cls(cabi_pes, [1.0], 1.0, 1, 0.01)
    if override == "instance":
        original = dyn._force

        def altered(q, bias=None):
            calls.append(True)
            return 0.25 * original(q, bias)

        dyn._force = altered
    elif override == "class":
        original = Dynamics._force

        def altered(self, q, bias=None):
            calls.append(True)
            return 0.25 * original(self, q, bias)

        monkeypatch.setattr(Dynamics, "_force", altered)
    actual, reference = _state(), _state()
    result = _advance(dyn, actual, api)
    assert result.backend == "python" and calls
    _advance(dyn, reference, api, backend="python")
    np.testing.assert_array_equal(actual.q, reference.q)
    np.testing.assert_array_equal(actual.p, reference.p)


@pytest.mark.parametrize("api", ["propagate", "propagate_block"])
@pytest.mark.parametrize("hook", ["integrator", "cv", "stack", "bias", "composite"])
def test_both_apis_preserve_class_mathematics(cabi_pes, api, hook, monkeypatch):
    dyn = Dynamics(cabi_pes, [1.0], 1.0, 1, 0.01)
    cv = Cartesian(0, 0)
    calls = []
    kwargs = {}
    if hook == "integrator":
        original = CayleyIntegrator.kick

        def changed(self, p, force, dt):
            calls.append(True)
            return original(self, p, force * 0.25, dt)

        monkeypatch.setattr(CayleyIntegrator, "kick", changed)
    elif hook == "cv":
        original = Cartesian.centroid_value_gradient

        def changed(self, q):
            calls.append(True)
            value, gradient = original(self, q)
            return value * 3 + 7, gradient * 3

        monkeypatch.setattr(Cartesian, "centroid_value_gradient", changed)
        kwargs["observe" if api == "propagate" else "cv"] = cv
    elif hook == "stack":
        cv = CVStack([cv])
        original = CVStack.value

        def changed(self, q):
            calls.append(True)
            return original(self, q) + 7

        monkeypatch.setattr(CVStack, "value", changed)
        kwargs["observe" if api == "propagate" else "cv"] = cv
    else:
        bias = UmbrellaBias(cv, [0.0], 0.2)
        cls = UmbrellaBias
        if hook == "composite":
            bias = CompositeBias(bias)
            cls = CompositeBias
        original = cls.energy_force

        def changed(self, q):
            calls.append(True)
            energy, force = original(self, q)
            return energy * 2, force * 2

        monkeypatch.setattr(cls, "energy_force", changed)
        kwargs["bias"] = bias
    actual, reference = _state(), _state()
    result = _advance(dyn, actual, api, **kwargs)
    assert result.backend == "python" and calls
    expected = _advance(dyn, reference, api, backend="python", **kwargs)
    np.testing.assert_array_equal(actual.q, reference.q)
    np.testing.assert_array_equal(actual.p, reference.p)
    if hook in ("cv", "stack"):
        np.testing.assert_array_equal(result.values, expected.values)
        assert np.all(result.values > 7)
    untouched = _state()
    with pytest.raises(RuntimeError, match="custom|overridden"):
        _advance(dyn, untouched, api, backend="native", **kwargs)
    np.testing.assert_array_equal(untouched.q, _state().q)
    np.testing.assert_array_equal(untouched.p, _state().p)


@pytest.mark.parametrize("api", ["propagate", "propagate_block"])
def test_both_apis_preserve_changed_python_callable(cabi_pes, api):
    pes = PythonPotential(cabi_pes)
    original = pes.fn
    calls = []

    def altered(q):
        calls.append(True)
        result = original(q)
        result.forces *= 0.5
        return result

    pes.fn = altered
    dyn = Dynamics(pes, [1.0], 1.0, 1, 0.01)
    actual, reference = _state(), _state()
    result = _advance(dyn, actual, api)
    assert result.backend == "python" and calls
    _advance(dyn, reference, api, backend="python")
    np.testing.assert_array_equal(actual.q, reference.q)
    np.testing.assert_array_equal(actual.p, reference.p)


@pytest.mark.parametrize("descriptor", [cv_descriptor, block_cv_descriptor])
@pytest.mark.parametrize("override", ["surfaces", "nested_com"])
@pytest.mark.parametrize("mode", ["instance", "class"])
def test_both_descriptors_reject_nested_rate_overrides(descriptor, override, mode, monkeypatch):
    coordinate = RPMDrateCoordinate(
        [[0], [1, 2]], [1.0, 1.0, 1.0],
        [[-1.0, 0, 0], [0, 0, 0], [1.0, 0, 0]],
        [[0, 1]], [[1, 2]], 10.0,
    )
    target = coordinate if override == "surfaces" else coordinate.com
    method = "surfaces" if override == "surfaces" else "centroid_value_gradient_hessian"
    monkeypatch.setattr(type(target) if mode == "class" else target, method, lambda *args: None)
    with pytest.raises(NativeUnavailableError, match="overridden"):
        descriptor(coordinate.dividing_surface(1.0), 3)


@pytest.mark.parametrize("api", ["propagate", "propagate_block"])
def test_invalid_python_cv_shape_is_rejected_before_state_commit(api):
    class InvalidCV:
        dimension = 2

        def value(self, q):
            return np.array([7.0])

    dyn = Dynamics(HarmonicPotential(), [1.0], 1.0, 1, 0.01, native=False)
    state = _state()
    original = state.copy()
    options = {"observe" if api == "propagate" else "cv": InvalidCV()}
    error = ValueError if api == "propagate" else PropagationError
    with pytest.raises(error, match="CV returned invalid observation"):
        _advance(dyn, state, api, backend="python", **options)
    np.testing.assert_array_equal(state.q, original.q)
    np.testing.assert_array_equal(state.p, original.p)
    assert state.step == 0 and dyn.propagation_stats["python_steps"] == 0


def test_atomic_and_valid_prefix_failure_contracts_remain_distinct():
    class DomainPotential:
        def evaluate(self, q):
            return np.zeros(len(q)), np.zeros_like(q), (q[:, 0, 0] > 0.25).astype(int)

    dyn = Dynamics(DomainPotential(), [1.0], 1.0, 1, 1.0, native=False)
    initial = State([[[0.0, 0.0, 0.0]]], [[[0.1, 0.0, 0.0]]])
    atomic, prefix = initial.copy(), initial.copy()
    with pytest.raises(RuntimeError):
        dyn.propagate(atomic, 4, backend="python")
    with pytest.raises(PropagationError) as caught:
        dyn.propagate_block(prefix, 4, backend="python")
    np.testing.assert_array_equal(atomic.q, initial.q)
    assert atomic.step == 0
    assert caught.value.result.completed == prefix.step == 2
    np.testing.assert_allclose(prefix.q[0, 0], [0.2, 0.0, 0.0])


def test_both_binding_bridges_preserve_explicit_open_checks(cabi_pes):
    opened = [True]
    descriptor = DescriptorBinding(
        "cabi-v1", cabi_pes._evaluate, 1, cabi_pes, cabi_pes._lock,
        cabi_pes._context, lambda: opened[0],
    )
    capsule = CapsuleBinding(
        cabi_pes, cabi_pes._evaluate, cabi_pes._lock, 1,
        context=cabi_pes._context, is_open=lambda: opened[0],
    )
    for bridge in (_atomic_binding(descriptor), descriptor_binding(capsule)):
        assert bridge.owner is cabi_pes and bridge.lock is cabi_pes._lock
        with bridge.acquire():
            pass
        opened[0] = False
        with pytest.raises(RuntimeError, match="closed"):
            with bridge.acquire():
                pytest.fail("closed binding was admitted")
        opened[0] = True


def test_explicit_python_callbacks_are_not_native_symbols():
    function = ctypes.CFUNCTYPE(None)(lambda: None)
    binding = DescriptorBinding("fortran-batch-v1", function, 1, function, threading.RLock())
    with pytest.raises(RuntimeError, match="Python callbacks"):
        with binding.acquire():
            pytest.fail("Python callback admitted to native propagation")


def test_loop_stats_tracks_replaced_counter_dictionary():
    dyn = Dynamics(HarmonicPotential(), [1.0], 1.0, 1, 0.01, native=False)
    old = dyn.loop_stats
    dyn.propagation_stats = {"native_steps": 0, "python_steps": 5, "fallback_reasons": {}}
    dyn.propagate_block(_state(), 2, backend="python")
    assert dyn.loop_stats is dyn.propagation_stats
    assert dyn.loop_stats["python_steps"] == 7 and old["python_steps"] == 0
