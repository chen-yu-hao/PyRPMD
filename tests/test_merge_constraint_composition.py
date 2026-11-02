"""RATTLE keeps its supported map when combined with operator composition."""
import copy

import numpy as np
import pytest

from pyrpmd import Dynamics, ForceSplitting, State
from pyrpmd.potentials import FreePotential
from pyrpmd.potentials.contraction import SplitPotential
from pyrpmd.propagation import NativeUnavailableError


class CountingFreePotential(FreePotential):
    def __init__(self):
        self.calls = 0

    def evaluate(self, q):
        self.calls += 1
        return super().evaluate(q)


def _dynamics(potential=None, **options):
    kwargs = dict(integrator="velocity-verlet", native=False, seed=71,
                  bond_constraints=[(0, 1, 1.)], potential_identity="constrained-free")
    kwargs.update(options)
    return Dynamics(potential or FreePotential(), [2., 3.], 5., 3, .003, **kwargs)


def _state(dynamics):
    q = np.zeros((3, 2, 3))
    q[:, 1, 0] = 1.
    return State(q, dynamics.sample_momenta(q), step=3, time=.07)


@pytest.mark.parametrize("operator", ["pile-l", "nhc", "andersen", "barostat", "mts"])
def test_unsupported_constrained_plans_reject_before_force_or_rng(operator):
    provider = CountingFreePotential()
    potential = provider
    rng = np.random.default_rng(81)
    before = copy.deepcopy(rng.bit_generator.state)
    options = {"seed": rng}
    if operator == "barostat":
        options.update(barostat={"mode": "bzp", "pressure": 0.},
                       cell=np.eye(3) * 10, pbc=[True] * 3)
    elif operator == "mts":
        potential = SplitPotential(provider, FreePotential(), 3, 1)
        options["force_splitting"] = ForceSplitting(2)
    else:
        options["thermostat"] = operator
    with pytest.raises(ValueError, match="bond constraints.*legacy Andersen"):
        _dynamics(potential, **options)
    assert provider.calls == 0
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("thermostat", ["pile-l", "svr", {"mode": "nhc"}, {"mode": "andersen"}])
def test_constrained_sample_step_cannot_enter_a_new_unconstrained_bath(thermostat):
    provider = CountingFreePotential()
    dynamics = _dynamics(provider)
    state = _state(dynamics)
    before = state.copy()
    operators = dynamics.operator_state_dict()
    with pytest.raises(ValueError, match="bond constraints.*sample_step"):
        dynamics.sample_step(state, thermostat=thermostat)
    assert provider.calls == 0
    assert dynamics.operator_state_dict() == operators
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert (state.step, state.time) == (before.step, before.time)


def test_legacy_constrained_andersen_preserves_segmented_trajectory_and_rng():
    complete, segmented = _dynamics(), _dynamics()
    actual, expected = _state(complete), _state(segmented)
    complete.thermalize(actual, 6, interval=2, resample=False)
    segmented.thermalize(expected, 2, interval=2, resample=False)
    segmented.thermalize(expected, 4, interval=2, resample=False)
    np.testing.assert_array_equal(actual.q, expected.q)
    np.testing.assert_array_equal(actual.p, expected.p)
    assert (actual.step, actual.time) == (expected.step, expected.time)
    assert complete.operator_state_dict() == segmented.operator_state_dict()
    complete.constraints.check(actual.q, actual.p)


def test_inactive_common_executor_retains_rattle_and_refuses_strict_native():
    actual_dynamics, reference = _dynamics(), _dynamics()
    actual, expected = _state(actual_dynamics), _state(reference)
    operators = actual_dynamics.operator_state_dict()
    with pytest.raises(NativeUnavailableError):
        actual_dynamics._composition.step(actual, backend="native")
    assert actual_dynamics.operator_state_dict() == operators
    np.testing.assert_array_equal(actual.q, expected.q)
    np.testing.assert_array_equal(actual.p, expected.p)
    for _ in range(5):
        actual_dynamics._composition.step(actual, backend="auto")
        reference.step(expected)
    np.testing.assert_array_equal(actual.q, expected.q)
    np.testing.assert_array_equal(actual.p, expected.p)
    assert (actual.step, actual.time) == (expected.step, expected.time)
    actual_dynamics.constraints.check(actual.q, actual.p)


@pytest.mark.parametrize("changed", [
    {"bond_constraints": [(0, 1, 1.1)]},
    {"bond_constraints": []},
    {"constraint_tolerance": 1e-10},
    {"constraint_max_iterations": 50},
    {"constraint_cell": np.eye(3) * 8, "constraint_pbc": [True] * 3},
])
def test_operator_checkpoint_rejects_changed_constraint_physics(changed):
    source = _dynamics()
    target = _dynamics(**changed)
    before = target.operator_state_dict()
    with pytest.raises(ValueError, match="checkpoint parameters differ"):
        target.load_operator_state_dict(source.operator_state_dict())
    assert target.operator_state_dict() == before


def test_shared_cell_defines_periodic_rattle_geometry():
    dynamics = _dynamics(cell=np.eye(3) * 8, pbc=[True] * 3)
    q = np.zeros((3, 2, 3))
    q[:, 0, 0], q[:, 1, 0] = 7.5, .5
    state = State(q, dynamics.sample_momenta(q))
    dynamics.step(state)
    dynamics.constraints.check(state.q, state.p)


def test_conflicting_constraint_and_dynamics_cells_are_rejected():
    with pytest.raises(ValueError, match="constraint geometry must match"):
        _dynamics(cell=np.eye(3) * 8, pbc=[True] * 3,
                  constraint_cell=np.eye(3) * 9, constraint_pbc=[True] * 3)


def test_unconstrained_operator_identity_remains_compatible():
    dynamics = _dynamics(bond_constraints=[])
    saved = dynamics.operator_state_dict()
    assert "bond_constraints" not in saved["composition"]["identity"]
    dynamics.load_operator_state_dict(saved)
    assert dynamics.operator_state_dict() == saved
