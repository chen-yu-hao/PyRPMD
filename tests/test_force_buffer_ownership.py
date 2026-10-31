"""Force composition and propagation must preserve provider-owned buffers."""

from pathlib import Path
import shutil
import subprocess
import sys
from types import MethodType

import numpy as np
import pytest

from pyrpmd.dynamics import CayleyIntegrator, Dynamics, State, native_capabilities
from pyrpmd.potentials import CABIPotential, PESResult


class ConstantProvider:
    def __init__(self, *, readonly=True, scale=1.):
        self.forces = scale * np.arange(1., 7.).reshape(1, 2, 3)
        self.forces.flags.writeable = not readonly

    def evaluate(self, q):
        return PESResult(np.zeros(1), self.forces)


class ConstantBias:
    def energy_force(self, q):
        return 0., np.full_like(q, 0.5)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("biased,fixed", [(False, False), (True, False), (False, True), (True, True)])
def test_steps_preserve_reused_readonly_provider_forces(native, biased, fixed):
    if native and not native_capabilities()["available"]:
        pytest.skip("native extension unavailable")
    provider = ConstantProvider()
    expected_force = provider.forces.copy()
    if biased:
        expected_force += 0.5
    if fixed:
        expected_force[:, 0] = 0.
    dynamics = Dynamics(
        provider, [2., 2.], beta=1., nbeads=1, dt=0.2, native=native,
        fixed_atoms=(0,) if fixed else (), fixed_positions=np.zeros((2, 3)),
    )
    state = State(np.zeros((1, 2, 3)), np.zeros((1, 2, 3)))
    original = provider.forces.copy()
    for _ in range(2):
        dynamics.step(state, ConstantBias() if biased else None)
    np.testing.assert_array_equal(provider.forces, original)
    assert not provider.forces.flags.writeable
    np.testing.assert_allclose(state.p, 0.4 * expected_force, rtol=1e-14)
    np.testing.assert_allclose(state.q, 0.04 * expected_force, rtol=1e-14)


def _dynamics(provider, native, operation, cls=Dynamics):
    return cls(
        provider, [2., 2.], beta=1., nbeads=1, dt=0.2, native=native,
        loop_backend="python" if operation == "sample_python" else "auto",
    )


def _advance_twice(dynamics, state, operation):
    for _ in range(2):
        if operation == "step":
            dynamics.step(state)
        else:
            dynamics.sample_step(state)


@pytest.mark.parametrize("readonly", [False, True])
def test_force_returns_an_independent_writable_buffer(readonly):
    provider = ConstantProvider(readonly=readonly)
    original = provider.forces.copy()
    dynamics = _dynamics(provider, False, "step")
    force = dynamics._force(np.zeros((1, 2, 3)))
    assert force.flags.writeable
    assert not np.shares_memory(force, provider.forces)
    force *= 0.5
    np.testing.assert_array_equal(provider.forces, original)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("readonly", [False, True])
@pytest.mark.parametrize("operation", ["step", "sample_python", "sample_auto"])
@pytest.mark.parametrize("override", ["subclass", "class", "instance"])
@pytest.mark.parametrize("target", ["force", "kick"])
def test_inplace_overrides_do_not_mutate_reused_provider_forces(
    native, readonly, operation, override, target, monkeypatch,
):
    if native and not native_capabilities()["available"]:
        pytest.skip("native extension unavailable")
    # Establish the independent, analytically scaled force before patching a
    # class. This reference also checks the sampler's state and RNG sequence.
    expected = _dynamics(ConstantProvider(readonly=readonly, scale=0.5), native, operation)
    reference = State(np.zeros((1, 2, 3)), np.zeros((1, 2, 3)))
    _advance_twice(expected, reference, operation)
    provider = ConstantProvider(readonly=readonly)
    original_buffer = provider.forces.copy()
    calls = []

    def scale(force):
        calls.append(True)
        force *= 0.5
        return force

    if target == "force":
        class ScaledDynamics(Dynamics):
            def _force(self, q, bias=None):
                return scale(super()._force(q, bias))

        dynamics = _dynamics(
            provider, native, operation, ScaledDynamics if override == "subclass" else Dynamics,
        )
        original = Dynamics._force

        def changed(self, q, bias=None):
            return scale(original(self, q, bias))

        if override == "class":
            monkeypatch.setattr(Dynamics, "_force", changed)
        elif override == "instance":
            monkeypatch.setattr(dynamics, "_force", MethodType(changed, dynamics))
    else:
        dynamics = _dynamics(provider, native, operation)

        class ScaledIntegrator(CayleyIntegrator):
            def kick(self, p, force, h):
                return super().kick(p, scale(force), h)

        original = CayleyIntegrator.kick

        def changed(self, p, force, h):
            return original(self, p, scale(force), h)

        if override == "subclass":
            dynamics.integrator = ScaledIntegrator([2., 2.], 1., 1, 0.2, native=native)
        elif override == "class":
            monkeypatch.setattr(CayleyIntegrator, "kick", changed)
        else:
            monkeypatch.setattr(dynamics.integrator, "kick", MethodType(changed, dynamics.integrator))

    actual = State(np.zeros((1, 2, 3)), np.zeros((1, 2, 3)))
    _advance_twice(dynamics, actual, operation)
    assert len(calls) == 4
    np.testing.assert_array_equal(provider.forces, original_buffer)
    assert provider.forces.flags.writeable == (not readonly)
    np.testing.assert_allclose(actual.q, reference.q, rtol=1e-14, atol=2e-14)
    np.testing.assert_allclose(actual.p, reference.p, rtol=1e-14, atol=2e-14)
    assert (actual.step, actual.time) == (reference.step, reference.time)
    assert dynamics.rng.bit_generator.state == expected.rng.bit_generator.state


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("readonly", [False, True])
@pytest.mark.parametrize("operation", ["step", "sample_python", "sample_auto"])
def test_builtin_kicks_borrow_provider_force_without_copying(native, readonly, operation):
    if native and not native_capabilities()["available"]:
        pytest.skip("native extension unavailable")
    provider = ConstantProvider(readonly=readonly)
    dynamics = _dynamics(provider, native, operation)
    state = State(np.zeros((1, 2, 3)), np.zeros((1, 2, 3)))
    original = provider.forces.copy()
    observed = []
    kick_code = CayleyIntegrator.kick.__code__

    def profile(frame, event, arg):
        if event == "call" and frame.f_code is kick_code:
            observed.append(frame.f_locals["force"])

    # Observe the actual arguments without replacing a method: a monkeypatched
    # kick must deliberately take the ownership-preserving compatibility path.
    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        _advance_twice(dynamics, state, operation)
    finally:
        sys.setprofile(previous)
    assert len(observed) == 4
    assert all(force is provider.forces for force in observed)
    np.testing.assert_array_equal(provider.forces, original)


@pytest.fixture(scope="module")
def native_pes_library(tmp_path_factory):
    compiler = shutil.which("cc") or shutil.which("gcc")
    if not native_capabilities()["available"] or compiler is None:
        pytest.skip("native extension and C compiler required")
    root = Path(__file__).resolve().parents[1]
    library = tmp_path_factory.mktemp("force-ownership-pes") / "harmonic.so"
    subprocess.run([
        compiler, "-shared", "-fPIC", "-I", str(root / "native/include"),
        str(root / "native/examples/harmonic_pes.c"), "-o", str(library),
    ], check=True)
    return library


@pytest.mark.parametrize("api", ["propagate", "propagate_block"])
@pytest.mark.parametrize("override", ["subclass", "class", "instance"])
def test_native_blocks_do_not_bypass_custom_half_kicks(
    api, override, native_pes_library, monkeypatch,
):
    original = Dynamics._kick
    calls = []

    def changed(self, q, p, bias=None):
        calls.append(True)
        original(self, q, p, bias)
        p += 0.0123

    class ChangedDynamics(Dynamics):
        _kick = changed

    with CABIPotential(native_pes_library) as provider:
        ordinary = Dynamics(provider, [1.], 1., 1, 0.01)
        assert ordinary.can_fuse()
        dynamics = (ChangedDynamics(provider, [1.], 1., 1, 0.01)
                    if override == "subclass" else ordinary)
        if override == "class":
            monkeypatch.setattr(Dynamics, "_kick", changed)
        elif override == "instance":
            monkeypatch.setattr(dynamics, "_kick", MethodType(changed, dynamics))
        assert not dynamics.can_fuse()
        actual = State([[[0.2, 0., 0.]]], [[[0.1, 0., 0.]]])
        reference = actual.copy()
        untouched = actual.copy()
        result = getattr(dynamics, api)(actual, 3, backend="auto")
        expected = getattr(dynamics, api)(reference, 3, backend="python")
        assert result.backend == "python" and result.fallback_reason
        assert len(calls) == 12
        np.testing.assert_array_equal(actual.q, reference.q)
        np.testing.assert_array_equal(actual.p, reference.p)
        np.testing.assert_array_equal(result.energies, expected.energies)
        assert (actual.step, actual.time) == (reference.step, reference.time)
        with pytest.raises(RuntimeError, match="custom"):
            getattr(dynamics, api)(untouched, 3, backend="native")
        assert len(calls) == 12
        np.testing.assert_array_equal(untouched.q, [[[0.2, 0., 0.]]])
        np.testing.assert_array_equal(untouched.p, [[[0.1, 0., 0.]]])
        assert (untouched.step, untouched.time) == (0, 0.)
