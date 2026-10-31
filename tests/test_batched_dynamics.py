"""Independent scalar maps and fault-injected providers verify batch semantics."""

import numpy as np
import pytest

from pyrpmd.batched import (BatchedDynamics, BatchedState, BatchedPropagationError,
                            validate_batch_potential)
from pyrpmd.bias import UmbrellaBias
from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.potentials import FreePotential, HarmonicPotential, PESResult, PotentialError
from pyrpmd.potentials.python import PythonPotential


NATIVE_CHOICES = [False] + (["required"] if native_capabilities()["available"] else [])


@pytest.mark.parametrize("native", NATIVE_CHOICES)
@pytest.mark.parametrize("scheme", ["cayley-bcb", "normal-mode-bab", "velocity-verlet"])
@pytest.mark.parametrize("beads", [1, 3, 8])
def test_independent_scalar_trajectories(native, scheme, beads):
    rng = np.random.default_rng(101)
    q = rng.normal(size=(4, beads, 2, 3))
    p = rng.normal(size=q.shape)
    state = BatchedState(q, p, [0, 2, 5, 8], [0., .2, .5, .8])
    initial = state.copy()
    cv = Cartesian(0, 1)
    batch = BatchedDynamics(HarmonicPotential(.2), [2., 3.], 4., beads, .005,
                            native=native, integrator=scheme)
    result = batch.propagate_block(state, 17, observe=cv)
    assert result.success
    assert result.pes_calls == 18
    assert result.pes_frames == 18*4*beads
    for lane in range(4):
        scalar = initial.to_states()[lane]
        dynamics = Dynamics(HarmonicPotential(.2), [2., 3.], 4., beads, .005,
                            native=native, integrator=scheme)
        assert result.initial_energy[lane] == pytest.approx(dynamics.energy(scalar), abs=2e-12)
        values = []
        for _ in range(17):
            dynamics.step(scalar)
            values.append(cv.value(scalar.q))
        np.testing.assert_allclose(state.q[lane], scalar.q, atol=3e-12, rtol=2e-13)
        np.testing.assert_allclose(state.p[lane], scalar.p, atol=3e-12, rtol=2e-13)
        np.testing.assert_allclose(result.values[:, lane], values, atol=3e-12, rtol=2e-13)
        assert result.energies[-1, lane] == pytest.approx(dynamics.energy(scalar), abs=4e-12)
        assert state.step[lane] == scalar.step
        assert state.time[lane] == scalar.time


@pytest.mark.parametrize("native", NATIVE_CHOICES)
def test_no_springs_between_trajectories_and_reversibility(native):
    q = np.repeat(np.array([[[[1., 0., 0.]]], [[[100., 2., 3.]]]]), 7, axis=1)
    state = BatchedState(q, np.zeros_like(q))
    dynamics = BatchedDynamics(FreePotential(), [2.], 3., 7, .01, native=native)
    dynamics.propagate_block(state, 9)
    np.testing.assert_allclose(state.q, q, atol=2e-12)
    np.testing.assert_allclose(state.p, 0, atol=2e-12)
    rng = np.random.default_rng(13)
    state = BatchedState(rng.normal(size=(5, 7, 1, 3)), rng.normal(size=(5, 7, 1, 3)))
    old = state.copy()
    dynamics = BatchedDynamics(HarmonicPotential(.1), [2.], 3., 7, .01, native=native)
    dynamics.propagate_block(state, 25)
    state.p *= -1
    dynamics.propagate_block(state, 25)
    np.testing.assert_allclose(state.q, old.q, atol=3e-12)
    np.testing.assert_allclose(state.p, -old.p, atol=3e-12)


class CountedFree:
    metadata = {"trajectory_batch_safe": True}

    def __init__(self):
        self.shapes = []

    def evaluate(self, q):
        self.shapes.append(q.shape)
        return PESResult(np.zeros(len(q)), np.zeros_like(q))


class FrameFailure(CountedFree):
    def evaluate(self, q):
        result = super().evaluate(q)
        bad = np.isclose(q[:, 0, 1], 1) & (q[:, 0, 0] > 1.5)
        result.status[bad] = 7
        result.energy[bad] = np.nan
        result.forces[bad] = np.nan
        result.messages = tuple("domain" if bad[i] else "" for i in range(len(q)))
        return result


def linear_states(beads=3):
    q = np.zeros((3, beads, 1, 3))
    q[:, :, 0, 1] = np.arange(3)[:, None]
    p = np.zeros_like(q)
    p[:, :, 0, 0] = np.array([1., 2., 3.])[:, None]
    return BatchedState(q, p)


def test_frame_failure_stop_masks_and_valid_prefixes():
    state = linear_states()
    potential = FrameFailure()
    dynamics = BatchedDynamics(potential, [1.], 5., 3, .5, native=False)
    result = dynamics.propagate_block(
        state, 4, observe=Cartesian(0, 0), stop=lambda i, s: i == 2 and s.step == 2,
        sample_interval=3, raise_on_error=False,
    )
    np.testing.assert_array_equal(result.completed, [4, 1, 2])
    np.testing.assert_array_equal(state.step, [4, 1, 2])
    np.testing.assert_array_equal(result.failed, [False, True, False])
    np.testing.assert_array_equal(result.stopped, [False, False, True])
    np.testing.assert_allclose(state.q[:, :, 0, 0], [[2.]*3, [1.]*3, [3.]*3])
    assert "bead 0" in result.errors[1] and "status=7: domain" in result.errors[1]
    assert result.pes_calls == 5
    assert potential.shapes == [(9, 1, 3)]*3 + [(3, 1, 3)]*2
    np.testing.assert_array_equal(result.offsets, [2, 3, 4])
    np.testing.assert_array_equal(result.valid[0], [True, False, True])
    assert np.isnan(result.values[1:, 2]).all()
    assert not result.success


def test_default_error_exposes_result_without_losing_successful_lanes():
    state = linear_states()
    potential = FrameFailure()
    dynamics = BatchedDynamics(potential, [1.], 5., 3, .5, native=False)
    with pytest.raises(BatchedPropagationError) as caught:
        dynamics.propagate_block(state, 4)
    np.testing.assert_array_equal(caught.value.result.completed, [2, 1, 2])
    np.testing.assert_array_equal(state.step, [2, 1, 2])
    assert potential.shapes == [(9, 1, 3)]*3


def test_initial_failure_isolated_without_stepping_failed_lane():
    state = linear_states()
    state.q[1, :, 0, 0] = 2.
    initial = state.copy()
    potential = FrameFailure()
    dynamics = BatchedDynamics(potential, [1.], 5., 3, .5, native=False)
    result = dynamics.propagate_block(state, 2, raise_on_error=False)
    np.testing.assert_array_equal(result.completed, [2, 0, 2])
    np.testing.assert_array_equal(state.q[1], initial.q[1])
    np.testing.assert_array_equal(state.p[1], initial.p[1])
    assert potential.shapes == [(9, 1, 3), (6, 1, 3), (6, 1, 3)]


def test_whole_call_failure_has_no_retry_or_half_step():
    class Broken(CountedFree):
        def evaluate(self, q):
            if len(self.shapes) == 2:
                self.shapes.append(q.shape)
                raise PotentialError("failed all frames")
            return super().evaluate(q)

    state = linear_states()
    potential = Broken()
    dynamics = BatchedDynamics(potential, [1.], 5., 3, .5, native=False)
    result = dynamics.propagate_block(state, 5, raise_on_error=False)
    assert result.failed.all()
    np.testing.assert_array_equal(state.step, [1, 1, 1])
    np.testing.assert_allclose(state.q[:, :, 0, 0], [[.5]*3, [1.]*3, [1.5]*3])
    assert result.pes_calls == 3


def test_callback_error_rolls_back_only_its_trial_step_and_cannot_mutate_input():
    state = linear_states()

    def callback(index, snapshot):
        if index == 1 and snapshot.step == 2:
            raise ValueError("bad classifier")
        snapshot.q[:] = 99  # A snapshot is never a writable view of the live state.
        return index == 2 and snapshot.step == 1

    result = BatchedDynamics(FreePotential(), [1.], 5., 3, .5, native=False).propagate_block(
        state, 4, stop=callback, raise_on_error=False)
    np.testing.assert_array_equal(result.completed, [4, 1, 1])
    np.testing.assert_allclose(state.q[:, :, 0, 0], [[2.]*3, [1.]*3, [1.5]*3])
    assert "bad classifier" in result.errors[1]


def test_active_mask_excludes_lanes_and_empty_work_does_not_call_pes():
    state = linear_states()
    original = state.copy()
    potential = CountedFree()
    dynamics = BatchedDynamics(potential, [1.], 5., 3, .5, native=False)
    empty = dynamics.propagate_block(state, 3, active=np.zeros(3, dtype=bool))
    assert empty.success and empty.pes_calls == 0
    assert empty.values is None and empty.energies.shape == (0, 3)
    zero = dynamics.propagate_block(state, 0)
    assert zero.success and zero.pes_calls == 0
    mask = np.array([False, True, False])
    result = dynamics.propagate_block(state, 3, active=mask)
    np.testing.assert_array_equal(mask, [False, True, False])
    np.testing.assert_array_equal(state.q[[0, 2]], original.q[[0, 2]])
    np.testing.assert_array_equal(result.completed, [0, 3, 0])
    assert result.success and all(shape == (3, 1, 3) for shape in potential.shapes)


def test_scalar_state_conversion_owns_arrays_and_preserves_clocks():
    scalar = [State(np.zeros((2, 1, 3)), np.ones((2, 1, 3)), 2, .4),
              State(np.ones((2, 1, 3)), np.zeros((2, 1, 3)), 7, .8)]
    batch = BatchedState.from_states(scalar)
    output = batch.to_states()
    output[0].q[:] = 5
    scalar[1].q[:] = 8
    assert batch.q[0, 0, 0, 0] == 0
    assert batch.q[1, 0, 0, 0] == 1
    np.testing.assert_array_equal(batch.step, [2, 7])
    np.testing.assert_array_equal(batch.time, [.4, .8])


@pytest.mark.parametrize("native", NATIVE_CHOICES)
def test_per_trajectory_umbrella_forces_and_fixed_atoms(native):
    rng = np.random.default_rng(52)
    q, p = rng.normal(size=(3, 5, 2, 3)), rng.normal(size=(3, 5, 2, 3))
    reference = np.array([[1., 2., 3.], [0., 0., 0.]])
    q[:, :, 0, :], p[:, :, 0, :] = reference[0], 0
    state = BatchedState(q, p)
    biases = [UmbrellaBias(Cartesian(1, 0), [center], .3) for center in (-1., 0., 1.)]
    batch = BatchedDynamics(HarmonicPotential(.1), [2., 3.], 4., 5, .02,
                            native=native, fixed_atoms=(0,), fixed_positions=reference)
    batch.propagate_block(state, 10, bias=biases)
    for i in range(3):
        single = State(q[i], p[i])
        dynamics = Dynamics(HarmonicPotential(.1), [2., 3.], 4., 5, .02,
                            native=native, fixed_atoms=(0,), fixed_positions=reference)
        for _ in range(10):
            dynamics.step(single, bias=biases[i])
        np.testing.assert_allclose(state.q[i], single.q, atol=1e-12)
        np.testing.assert_allclose(state.p[i], single.p, atol=1e-12)
    np.testing.assert_array_equal(state.q[:, :, 0], np.broadcast_to(reference[0], (3, 5, 3)))
    np.testing.assert_array_equal(state.p[:, :, 0], 0)


def test_reused_pes_force_buffer_is_not_mutated_or_retained():
    class Scratch:
        metadata = {"trajectory_batch_safe": True}

        def evaluate(self, q):
            if not hasattr(self, "forces") or self.forces.shape != q.shape:
                self.forces = np.empty_like(q)
            self.forces[:] = -.3*q
            return PESResult(.15*np.sum(q*q, axis=(1, 2)), self.forces)

    rng = np.random.default_rng(8)
    q = rng.normal(size=(3, 4, 2, 3)); p = rng.normal(size=q.shape)
    a, b = BatchedState(q, p), BatchedState(q, p)
    scratch = Scratch()
    BatchedDynamics(scratch, [2, 3], 5, 4, .01, native=False).propagate_block(a, 18)
    BatchedDynamics(HarmonicPotential(.3), [2, 3], 5, 4, .01, native=False).propagate_block(b, 18)
    np.testing.assert_array_equal(a.q, b.q)
    np.testing.assert_array_equal(a.p, b.p)
    np.testing.assert_array_equal(scratch.forces, -.3*a.q.reshape(12, 2, 3))


@pytest.mark.parametrize("metadata", [
    {}, {"trajectory_batch_safe": False}, {"batch_independent": False, "trajectory_batch_safe": True},
    {"ring_polymer_coupled": True, "trajectory_batch_safe": True},
    {"stateful": True, "trajectory_batch_safe": True},
    {"capabilities": {"stateful": True}, "trajectory_batch_safe": True},
])
def test_unsafe_and_undeclared_custom_provider_is_rejected(metadata):
    provider = CountedFree()
    provider.metadata = metadata
    with pytest.raises(ValueError):
        validate_batch_potential(provider)


def test_python_adapter_preserves_unit_conversion_and_declaration():
    class Model(CountedFree):
        def evaluate(self, q):
            return PESResult(.5*np.sum(q*q, axis=(1, 2)), q)

    wrapped = PythonPotential(Model(), units={"positions": "angstrom", "energy": "eV"},
                              derivative="gradient")
    validate_batch_potential(wrapped)
    q = np.random.default_rng(8).normal(size=(3, 4, 1, 3))
    p = np.zeros_like(q)
    batch = BatchedState(q, p)
    BatchedDynamics(wrapped, [2], 5, 4, .01, native=False).propagate_block(batch, 4)
    for i in range(3):
        scalar = State(q[i], p[i])
        dynamics = Dynamics(wrapped, [2], 5, 4, .01, native=False)
        for _ in range(4):
            dynamics.step(scalar)
        np.testing.assert_allclose(batch.q[i], scalar.q, atol=1e-13)


@pytest.mark.parametrize("bad", [True, -1, 1.5, [0, 1], [0, -1, 0]])
def test_invalid_state_clock(bad):
    with pytest.raises(ValueError):
        BatchedState(np.zeros((3, 2, 1, 3)), np.zeros((3, 2, 1, 3)), step=bad)


def test_invalid_active_mask_and_parameter_mutation():
    dynamics = BatchedDynamics(FreePotential(), [1], 3, 3, .1, native=False)
    state = linear_states()
    for mask in ([0, 1, 0], [True, False], True):
        with pytest.raises(ValueError):
            dynamics.propagate_block(state, 2, active=mask)
    dynamics.dt = .2
    with pytest.raises(ValueError, match="immutable"):
        dynamics.step(state)


def test_unknown_subclass_is_not_silently_treated_as_builtin():
    class Shared(FreePotential):
        def evaluate(self, q):
            return PESResult(np.full(len(q), np.mean(q)), np.ones_like(q))

    with pytest.raises(ValueError, match="trajectory_batch_safe"):
        validate_batch_potential(Shared())


def test_class_level_builtin_override_does_not_acquire_implicit_batch_safety(monkeypatch):
    def coupled(self, q):
        return PESResult(np.full(len(q), q.mean()), np.full_like(q, q.mean()))

    monkeypatch.setattr(HarmonicPotential, "evaluate", coupled)
    with pytest.raises(ValueError, match="trajectory_batch_safe"):
        validate_batch_potential(HarmonicPotential())


def test_python_wrapper_cannot_hide_explicitly_stateful_inner_capability():
    model = CountedFree()
    model.metadata = {"capabilities": {"stateful": True}}
    wrapped = PythonPotential(model)
    wrapped.metadata = {"trajectory_batch_safe": True}
    with pytest.raises(ValueError, match="coupled/stateful"):
        validate_batch_potential(wrapped)


@pytest.mark.parametrize("delegate", [False, True])
def test_torch_adapter_preserves_explicit_batch_and_reuse_declarations(monkeypatch, delegate):
    # Metadata routing has no tensor/GPU dependency; both factory routes must
    # preserve a model's lifecycle opt-out even when PyTorch is unavailable.
    import sys
    from types import SimpleNamespace
    from pyrpmd.potentials import optional
    from pyrpmd.parallel.execution import _can_reuse_potential

    class Module:
        pass

    fake_torch = SimpleNamespace(device=lambda name: SimpleNamespace(type="cpu", index=None),
                                 nn=SimpleNamespace(Module=Module), float64=object())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    model = CountedFree() if delegate else Module()
    model.metadata = {"trajectory_batch_safe": True, "reuse_across_tasks": False,
                      "capabilities": {"stateful": False}}
    monkeypatch.setattr(optional, "import_object", lambda path: lambda **kwargs: model)
    provider = optional.TorchPotential(factory="example:factory", device="cpu")
    assert provider.metadata["trajectory_batch_safe"] is True
    assert provider.metadata["reuse_across_tasks"] is False
    assert not _can_reuse_potential(provider)
    validate_batch_potential(provider)
