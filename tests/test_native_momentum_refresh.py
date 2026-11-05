"""Pre-drawn Andersen refreshes stay bitwise aligned across block backends."""

import copy

import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.block_propagation import PropagationError
from pyrpmd.momentum_refresh import MAX_PREDRAWN_BYTES, plan_refresh
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.workflows.sampling import AndersenSchedule


pytestmark = pytest.mark.skipif(
    not native_capabilities()["available"], reason="native build required"
)


def _dynamics(*, backend, seed=421, beads=4):
    return Dynamics(
        HarmonicPotential(), [1.0, 2.0], 1.0, beads, 0.01,
        seed=seed, native=True, loop_backend=backend,
    )


def _state(beads=4):
    rng = np.random.default_rng(8102)
    return State(rng.normal(size=(beads, 2, 3)), rng.normal(size=(beads, 2, 3)))


def _reference(dynamics, state, schedule, steps):
    for _ in range(steps):
        if schedule.due(state.step):
            state.p[:] = dynamics.sample_momenta()
            schedule.refreshed(state.step, dynamics.rng)
        dynamics.propagate_block(state, 1, backend="python")


@pytest.mark.parametrize("kind", ["fixed", "geometric"])
def test_native_refresh_matches_scalar_reference(kind):
    native = _dynamics(backend="native")
    reference = _dynamics(backend="python")
    actual, expected = _state(), _state()
    schedule = AndersenSchedule(3, kind)
    reference_schedule = AndersenSchedule(3, kind)
    original_rng = copy.deepcopy(native.rng.bit_generator.state)
    original_deadline = schedule.next_collision_step

    plan = plan_refresh(native, schedule, actual.step, 24)
    assert native.rng.bit_generator.state == original_rng
    assert schedule.next_collision_step == original_deadline

    result = native.propagate_block(actual, plan.steps, backend="native", refresh=plan)
    _reference(reference, expected, reference_schedule, plan.steps)

    assert result.backend == "native"
    assert result.refreshes_completed == len(plan.offsets)
    np.testing.assert_allclose(actual.q, expected.q, atol=2e-14, rtol=0)
    np.testing.assert_allclose(actual.p, expected.p, atol=2e-14, rtol=0)
    assert native.rng.bit_generator.state == reference.rng.bit_generator.state
    assert schedule.next_collision_step == reference_schedule.next_collision_step


def test_python_refresh_uses_the_same_transaction_and_rng_replay():
    dynamics = _dynamics(backend="python")
    actual, expected = _state(), _state()
    schedule = AndersenSchedule(2, "geometric")
    reference_schedule = AndersenSchedule(2, "geometric")
    plan = plan_refresh(dynamics, schedule, actual.step, 18)

    result = dynamics.propagate_block(actual, plan.steps, backend="python", refresh=plan)
    reference = _dynamics(backend="python")
    _reference(reference, expected, reference_schedule, plan.steps)

    assert result.refreshes_completed == len(plan.offsets)
    np.testing.assert_array_equal(actual.q, expected.q)
    np.testing.assert_array_equal(actual.p, expected.p)
    assert dynamics.rng.bit_generator.state == reference.rng.bit_generator.state
    assert schedule.next_collision_step == reference_schedule.next_collision_step


def test_planner_shortens_at_the_predraw_budget():
    dynamics = _dynamics(backend="python", beads=8)
    schedule = AndersenSchedule(1, "fixed")
    bytes_per_refresh = dynamics.integrator.shape[0] * dynamics.integrator.shape[1] * 3 * 8 + 8
    plan = plan_refresh(dynamics, schedule, 0, 100, budget=bytes_per_refresh * 2)
    assert plan.steps == 2
    np.testing.assert_array_equal(plan.offsets, [0, 1])
    assert plan.momenta.nbytes <= MAX_PREDRAWN_BYTES


def test_fragment_projection_matches_python_with_native_roundoff():
    native = _dynamics(backend="native")
    reference = _dynamics(backend="python")
    actual, expected = _state(), _state()
    fragments = native._fragment_data(actual, [[0], [1]])
    reference_fragments = reference._fragment_data(expected, [[0], [1]])
    native._fix_fragments(actual, fragments)
    reference._fix_fragments(expected, reference_fragments)
    schedule = AndersenSchedule(2, "fixed")
    plan = plan_refresh(native, schedule, actual.step, 8, projection="fragments")

    got = native.propagate_block(actual, plan.steps, backend="native", refresh=plan,
                                 fragment_data=fragments)
    reference.propagate_block(expected, plan.steps, backend="python", refresh=plan,
                              fragment_data=reference_fragments)
    assert got.refreshes_completed == len(plan.offsets)
    np.testing.assert_allclose(actual.q, expected.q, atol=3e-11, rtol=3e-11)
    np.testing.assert_allclose(actual.p, expected.p, atol=3e-10, rtol=3e-10)


def test_failed_step_rolls_back_its_refresh_and_rng_prefix():
    class FailingHarmonic(HarmonicPotential):
        def __init__(self, fail_on):
            super().__init__()
            self.calls = 0
            self.fail_on = fail_on

        def evaluate(self, coordinates):
            self.calls += 1
            if self.calls == self.fail_on:
                raise RuntimeError("injected PES failure")
            return super().evaluate(coordinates)

    dynamics = Dynamics(FailingHarmonic(4), [1.0, 2.0], 1.0, 4, 0.01,
                        seed=421, native=False, loop_backend="python")
    actual = _state()
    schedule = AndersenSchedule(2, "fixed")
    plan = plan_refresh(dynamics, schedule, actual.step, 5)
    with pytest.raises(PropagationError, match="injected PES failure") as caught:
        dynamics.propagate_block(actual, plan.steps, backend="python", refresh=plan)
    result = caught.value.result
    assert result.completed == 2
    assert result.refreshes_completed == 1

    reference = _dynamics(backend="python")
    expected = _state()
    ref_schedule = AndersenSchedule(2, "fixed")
    _reference(reference, expected, ref_schedule, 2)
    np.testing.assert_allclose(actual.q, expected.q, atol=2e-14, rtol=0)
    np.testing.assert_allclose(actual.p, expected.p, atol=2e-14, rtol=0)
    assert dynamics.rng.bit_generator.state == reference.rng.bit_generator.state
