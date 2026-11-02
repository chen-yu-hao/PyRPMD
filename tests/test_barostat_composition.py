"""Pressure plans must validate before mutating a particle/piston trajectory."""

import copy

import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics
from pyrpmd.engine import ForceSplitting, PropagationPlan, normalize_barostat
from pyrpmd.potentials import FreePotential
from pyrpmd.potentials.contraction import SplitPotential

from test_barostat import make


@pytest.mark.parametrize("spec", [
    {"pressure": .1}, {"mode": "none", "pressure": .1},
    {"mode": "bzp"}, {"mode": "bzp", "pressure": True},
    {"mode": "bzp", "pressure": np.inf},
    {"mode": "bzp", "pressure": .1, "gamma": -.1},
    {"mode": "bzp", "pressure": .1, "tau": 0},
    {"mode": "bzp", "pressure": .1, "cell_mask": np.eye(3)},
    {"mode": "scr", "pressure": .1},
    {"mode": "scr", "pressure": .1, "compressibility": 0},
    {"mode": "rgb", "stress": np.eye(3)},
    {"mode": "rgb", "stress": [[0, 1, 0], [0, 0, 0], [0, 0, 0]], "reference_cell": np.eye(3)},
    {"mode": "mttk", "pressure": .1, "cell_mask": np.zeros((3, 3))},
    {"mode": "mttk", "pressure": .1, "cell_mask": np.ones((3, 3))},
    {"mode": "mttk", "pressure": .1, "cell_mask": np.eye(3) * .5},
    {"mode": 7}, {"mode": "unknown"},
])
def test_invalid_pressure_specification_cannot_be_silently_ignored(spec):
    with pytest.raises(ValueError):
        PropagationPlan(barostat=spec)


def test_pressure_aliases_and_matrices_are_canonical_and_detached():
    cell = np.diag([3., 4., 5.])
    stress = np.diag([.1, .2, -.3])
    spec = {"mode": "ANISOTROPIC", "stress": stress,
            "reference_cell": cell, "cell_mask": np.eye(3, dtype=bool)}
    plan = PropagationPlan(barostat=spec)
    stress[0, 0], cell[0, 0] = 99, 99
    assert plan.barostat_spec == {
        "mode": "rgb", "stress": [[.1, 0., 0.], [0., .2, 0.], [0., 0., -.3]],
        "reference_cell": [[3., 0., 0.], [0., 4., 0.], [0., 0., 5.]],
        "cell_mask": np.eye(3).tolist(),
    }
    assert normalize_barostat({"mode": "stochastic_rescaling", "pressure": -.2,
                               "compressibility": 1.})["mode"] == "scr"
    assert not PropagationPlan(barostat="NONE").active


def _attach(barostat, original, *, integrator="normal-mode-bab", **changes):
    options = dict(potential=original.potential, masses=original.masses,
                   beta=original.beta, nbeads=original.nbeads, dt=original.dt,
                   native=False, loop_backend="python", integrator=integrator,
                   barostat=barostat)
    options.update(changes)
    dynamics = Dynamics(**options)
    dynamics.rng.bit_generator.state = copy.deepcopy(original.rng.bit_generator.state)
    return dynamics


@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab", "velocity-verlet"])
def test_composed_external_bzp_matches_established_piston_map(integrator):
    left, source, expected, _, _ = make(dt=.005, friction=.4)
    right, other, actual, _, _ = make(dt=.005, friction=.4)
    reference = _attach(None, source, integrator=integrator)
    composed = _attach(right, other, integrator=integrator)
    for _ in range(17):
        left.step(expected, reference, thermostat=None)
        composed.step(actual)
    np.testing.assert_array_equal(actual.q, expected.q)
    np.testing.assert_array_equal(actual.p, expected.p)
    assert (actual.step, actual.time) == (expected.step, expected.time)
    assert right.snapshot() == left.snapshot()
    assert composed.rng.bit_generator.state == reference.rng.bit_generator.state
    np.testing.assert_array_equal(composed.cell, right.cell)


@pytest.mark.parametrize("changes,match", [
    ({"potential": FreePotential()}, "must share"),
    ({"masses": [4., 3.]}, "must share"),
    ({"beta": 5.}, "must share"),
    ({"nbeads": 8}, "must share"),
    ({"dt": -.005}, "timestep"),
    ({"fixed_atoms": (0,), "fixed_positions": np.zeros((2, 3))}, "all atoms mobile"),
    ({"pbc": [True, True, False]}, "fully periodic"),
])
def test_external_piston_binding_errors_leave_every_operator_unchanged(changes, match):
    piston, source, state, _, model = make()
    before, saved = state.copy(), source.operator_state_dict()
    saved_piston = piston.snapshot()
    count = model.calls
    with pytest.raises(ValueError, match=match):
        _attach(piston, source, **changes)
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert (state.step, state.time) == (before.step, before.time)
    assert source.operator_state_dict() == saved
    assert piston.snapshot() == saved_piston
    assert model.calls == count


def test_split_pressure_requires_a_physical_contracted_virial():
    split = SplitPotential(FreePotential(), FreePotential(), 4, 2)
    with pytest.raises(ValueError, match="split/RPC physical virial"):
        Dynamics(split, [2.], 4., 4, .01, native=False,
                 force_splitting=ForceSplitting(2),
                 barostat={"mode": "bzp", "pressure": .1}, cell=np.eye(3))


@pytest.mark.parametrize("mode", ["bzp", "mttk", "rgb", "scr"])
def test_pressure_requires_three_periodic_directions_before_force_calls(mode):
    piston, original, state, system, model = make(beads=1)
    spec = {"mode": mode, "pressure": .1}
    if mode == "rgb":
        spec = {"mode": mode, "stress": np.eye(3) * .1, "reference_cell": system.cell}
    if mode == "scr":
        spec["compressibility"] = 1.
    saved = original.operator_state_dict()
    count = model.calls
    with pytest.raises(ValueError, match="fully periodic"):
        _attach(spec, original, cell=system.cell, pbc=[True, False, True])
    assert original.operator_state_dict() == saved
    assert model.calls == count


def test_external_piston_failure_rolls_back_cell_rng_and_physical_state():
    piston, original, state, _, model = make(friction=.3)
    dynamics = _attach(piston, original, thermostat={"mode": "nhc", "tau": .3})
    dynamics.step(state)
    saved, before = dynamics.operator_state_dict(), state.copy()
    model.fail_at = model.calls + 2
    with pytest.raises(RuntimeError, match="injected force failure"):
        dynamics.step(state)
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert (state.step, state.time) == (before.step, before.time)
    assert dynamics.operator_state_dict() == saved
    np.testing.assert_array_equal(dynamics.cell, piston.cell)


@pytest.mark.parametrize("parameter", ["pressure_au", "tau_au", "friction_au", "mass"])
def test_external_piston_parameters_cannot_change_during_a_bound_trajectory(parameter):
    piston, original, state, _, model = make()
    dynamics = _attach(piston, original)
    saved, before = dynamics.operator_state_dict(), state.copy()
    setattr(piston, parameter, getattr(piston, parameter) + .1)
    count = model.calls
    with pytest.raises(ValueError, match="barostat parameters changed"):
        dynamics.step(state)
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert dynamics.operator_state_dict() == saved
    assert model.calls == count


def test_external_piston_restart_rejects_different_pressure():
    piston, original, _, _, _ = make()
    first = _attach(piston, original)
    other_piston, other, _, _, _ = make()
    other_piston.pressure_au = .03
    second = _attach(other_piston, other)
    saved = second.operator_state_dict()
    with pytest.raises(ValueError, match="parameters differ"):
        second.load_operator_state_dict(first.operator_state_dict())
    assert second.operator_state_dict() == saved


@pytest.mark.parametrize("location", ["cell", "piston"])
def test_external_piston_checkpoint_requires_one_consistent_cell(location):
    piston, original, _, _, model = make()
    dynamics = _attach(piston, original)
    saved = dynamics.operator_state_dict()
    broken = copy.deepcopy(saved)
    if location == "cell":
        broken["composition"]["cell"][0][0] *= 1.2
    else:
        broken["composition"]["piston"]["cell"] = (piston.cell * 1.2).tolist()
    count = model.calls
    with pytest.raises(ValueError, match="cell differs"):
        dynamics.load_operator_state_dict(broken)
    assert dynamics.operator_state_dict() == saved
    np.testing.assert_array_equal(dynamics.cell, piston.cell)
    assert model.calls == count


def test_fixed_cell_operator_checkpoint_cannot_change_the_physical_cell():
    dynamics = Dynamics(FreePotential(), [2.], 4., 4, .01, cell=np.eye(3))
    saved = dynamics.operator_state_dict()
    broken = copy.deepcopy(saved)
    broken["composition"]["cell"][1][1] = 1.1
    with pytest.raises(ValueError, match="fixed-cell checkpoint"):
        dynamics.load_operator_state_dict(broken)
    assert dynamics.operator_state_dict() == saved


def test_native_pressure_checkpoint_requires_the_same_executor_cell():
    _, original, state, system, _ = make()
    dynamics = _attach({"mode": "bzp", "pressure": .02, "tau": 2., "gamma": 0.},
                       original, cell=system.cell, potential_identity="fractional-well-v1",
                       thermostat={"mode": "nhc", "tau": .3})
    dynamics.step(state)
    saved = dynamics.operator_state_dict()
    assert saved["composition"]["native"] is not None
    broken = copy.deepcopy(saved)
    broken["composition"]["cell"][2][2] *= 1.1
    with pytest.raises(ValueError, match="native executor cell"):
        dynamics.load_operator_state_dict(broken)
    assert dynamics.operator_state_dict() == saved
