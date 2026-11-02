"""Regression checks for fixed-full-ring RPMD committor sampling."""
import copy
import json

import numpy as np
import pytest
from scipy.stats import binomtest

from pyrpmd import CommittorSampler, CommittorResult, committor_statistics
from pyrpmd.cv import Cartesian, CVStack
from pyrpmd.dynamics import Dynamics, native_capabilities
from pyrpmd.engine import ForceSplitting, PropagationPlan
from pyrpmd.potentials import FreePotential, PESResult
from pyrpmd.potentials.contraction import SplitPotential


def sampler(beads=1, seed=15, dt=0.05, max_steps=8, *, integrator="cayley-bcb",
            native=False, potential=None):
    dynamics = Dynamics(FreePotential() if potential is None else potential,
                        [1.0], 4.0, beads, dt, native=native, integrator=integrator)
    return CommittorSampler(dynamics, Cartesian(0, "x"), basin_a=-0.1,
                           basin_b=0.1, max_steps=max_steps, seed=seed)


@pytest.mark.parametrize("counts", [(0, 10, 0), (10, 0, 0), (3, 7, 0),
                                     (0, 0, 10), (3, 2, 5)])
def test_censored_statistics_and_exact_limits(counts):
    a, b, u = counts
    n = a + b + u
    result = committor_statistics(a, b, u)
    assert result["p_B"] == (b / n if u == 0 else None)
    assert result["empirical_p_B_bounds"] == [b / n, (b + u) / n]
    assert result["p_B_confidence_envelope"] == [
        binomtest(b, n).proportion_ci().low,
        binomtest(b + u, n).proportion_ci().high,
    ]
    assert not result["converged"]
    json.dumps(result, allow_nan=False)
    if a + b == 0:
        assert result["p_B_given_resolved"] is None
        assert result["p_B_confidence_envelope"] == [0.0, 1.0]


@pytest.mark.parametrize("values", [(0, 0, 0), (-1, 2, 0), (True, 2, 0),
                                    (1.0, 2, 0)])
def test_statistics_reject_invalid_counts(values):
    with pytest.raises(ValueError):
        committor_statistics(*values)


@pytest.mark.parametrize("confidence", [0, 1, -1, np.nan, True, "0.95"])
def test_statistics_reject_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        committor_statistics(1, 1, 0, confidence_level=confidence)


@pytest.mark.parametrize("beads", [1, 4, 8])
@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab", "velocity-verlet"])
def test_free_ring_shots_match_analytical_centroid_motion(beads, integrator):
    s = sampler(beads=beads, integrator=integrator)
    q = np.zeros((beads, 1, 3))
    q[:, 0, 1] = np.arange(beads) * 0.01
    before = q.copy()
    dyn_rng = copy.deepcopy(s.dynamics.rng.bit_generator.state)
    result = s.sample(q, n_shots=40)
    p = np.random.default_rng(15).normal(size=(40, beads, 1, 3)) * np.sqrt(beads / 4)
    velocity = p[:, :, 0, 0].mean(axis=1)
    endpoint = velocity * (8 * 0.05)
    expected = np.where(endpoint <= -0.1, -1, np.where(endpoint >= 0.1, 1, 0))
    expected_steps = np.minimum(np.ceil(0.1 / (np.abs(velocity) * 0.05)), 8).astype(int)
    np.testing.assert_array_equal(result.outcomes, expected)
    np.testing.assert_array_equal(result.steps, expected_steps)
    np.testing.assert_allclose(result.final_values, velocity * expected_steps * 0.05,
                               atol=1e-12)
    np.testing.assert_array_equal(result.configuration, before)
    np.testing.assert_array_equal(q, before)
    assert s.dynamics.rng.bit_generator.state == dyn_rng
    assert not result.configuration.flags.writeable
    assert not result.outcomes.flags.writeable
    assert np.any(expected == 0)
    assert result.summary()["p_B"] is None


def test_maxwell_momenta_and_fixed_atoms():
    q = np.zeros((4, 2, 3))
    q[:, 1] = [2., 1., 0.]
    dyn = Dynamics(FreePotential(), [2., 5.], 3., 4, 0.01, native=False,
                   fixed_atoms=[1], fixed_positions=q[0])
    s = CommittorSampler(dyn, Cartesian(0), basin_a=-1, basin_b=1,
                         max_steps=2, seed=23)
    expected = np.random.default_rng(23).normal(size=q.shape) * np.sqrt(
        np.array([2., 5.])[None, :, None] * 4 / 3)
    expected[:, 1] = 0
    np.testing.assert_array_equal(s._momenta(), expected)
    shot = s.shoot(q)
    np.testing.assert_array_equal(shot.final_state.q[:, 1], q[:, 1])
    np.testing.assert_array_equal(shot.final_state.p[:, 1], np.zeros((4, 3)))
    bad = q.copy()
    bad[0, 1, 0] += 0.1
    with pytest.raises(ValueError, match="fixed atoms"):
        s.shoot(bad)


@pytest.mark.parametrize("x,basin", [(-0.1, "A"), (0.1, "B"),
                                    (-2., "A"), (2., "B")])
def test_initial_basin_is_exact_and_does_not_consume_rng_or_pes(x, basin):
    class NoPotential:
        def evaluate(self, q):
            raise AssertionError("A boundary shot should not evaluate PES")
    s = sampler(potential=NoPotential())
    before = copy.deepcopy(s.rng.bit_generator.state)
    q = np.array([[[x, 0., 0.]]])
    r = s.sample(q, n_shots=3)
    p = float(basin == "B")
    assert r.initial_basin == basin
    assert r.summary()["p_B_confidence_envelope"] == [p, p]
    assert np.all(r.steps == 0)
    assert s.rng.bit_generator.state == before


def test_exact_rng_continuation_matches_unsplit_sampling():
    q = np.zeros((4, 1, 3))
    full = sampler(beads=4).sample(q, n_shots=30)
    first = sampler(beads=4).sample(q, n_shots=12)
    resumed = sampler(beads=4, seed=999)
    resumed.rng.bit_generator.state = first.rng_state
    second = resumed.sample(q, n_shots=18)
    for key in ("outcomes", "steps", "final_values"):
        np.testing.assert_array_equal(
            np.r_[getattr(first, key), getattr(second, key)], getattr(full, key))
    assert second.rng_state == full.rng_state


def test_callback_receives_owned_terminal_states():
    q = np.zeros((1, 1, 3))
    expected = sampler().sample(q, n_shots=20)
    seen = []
    def observe(i, shot):
        seen.append((i, shot.outcome))
        shot.final_state.q[:] = 123
    actual = sampler().sample(q, n_shots=20, on_shot=observe)
    np.testing.assert_array_equal(actual.outcomes, expected.outcomes)
    assert any(outcome == "unresolved" for _, outcome in seen)
    assert [i for i, _ in seen] == list(range(20))
    assert np.all(q == 0)


@pytest.mark.parametrize("mode", ["potential", "coordinate"])
def test_numerical_failure_propagates_and_restores_shot_rng(mode):
    class BadPotential:
        def evaluate(self, q):
            raise ArithmeticError("failed force")
    s = sampler(potential=BadPotential() if mode == "potential" else None)
    if mode == "coordinate":
        def bad_coordinate(q):
            if np.any(q):
                raise ArithmeticError("failed coordinate")
            return 0.
        s._coordinate = bad_coordinate
    q = np.zeros((1, 1, 3))
    before = copy.deepcopy(s.rng.bit_generator.state)
    with pytest.raises(ArithmeticError):
        s.shoot(q)
    assert s.rng.bit_generator.state == before
    assert np.all(q == 0)


@pytest.mark.parametrize("key", ["history_dependent", "time_dependent", "stateful"])
@pytest.mark.parametrize("location", ["attribute", "metadata", "capabilities"])
def test_reject_declared_time_dependent_potential(key, location):
    p = FreePotential()
    if location == "attribute":
        setattr(p, key, True)
    else:
        p.metadata = {key: True} if location == "metadata" else {"capabilities": {key: True}}
    with pytest.raises(ValueError, match="time-independent"):
        sampler(potential=p)


def test_reject_patched_or_changed_dynamics():
    s = sampler()
    s.dynamics.step = lambda state: state
    with pytest.raises(TypeError, match="patched"):
        s.shoot(np.zeros((1, 1, 3)))
    s = sampler()
    s.dynamics.potential = FreePotential()
    with pytest.raises(ValueError, match="changed"):
        s.shoot(np.zeros((1, 1, 3)))


@pytest.mark.parametrize("mode", ["andersen", "langevin", "nhc", "barostat", "force-splitting"])
def test_reject_composed_dynamics_before_rng_or_pes_use(mode):
    class UncalledPotential(FreePotential):
        def evaluate(self, q):
            raise AssertionError("unsupported committor must not evaluate the PES")

        def set_cell(self, cell):
            raise AssertionError("unsupported committor must not update the cell")

    potential = UncalledPotential()
    options = {}
    if mode == "barostat":
        options.update(barostat={"mode": "bzp", "pressure": 0.1},
                       cell=np.eye(3) * 4, pbc=(True, True, True))
    elif mode == "force-splitting":
        potential = SplitPotential(potential, UncalledPotential(), 1)
        options["force_splitting"] = ForceSplitting(2)
    else:
        options["thermostat"] = mode
    d = Dynamics(potential, [1.], 4., 1, 0.05, native=False, **options)
    before = copy.deepcopy(d.operator_state_dict())
    shot_rng = np.random.default_rng(37)
    saved_shot_rng = copy.deepcopy(shot_rng.bit_generator.state)
    with pytest.raises(ValueError, match="no thermostat, barostat or force splitting"):
        CommittorSampler(d, Cartesian(0), basin_a=-0.1, basin_b=0.1,
                         max_steps=2, seed=shot_rng)
    assert d.operator_state_dict() == before
    assert shot_rng.bit_generator.state == saved_shot_rng


def test_reject_bond_constraints_before_sampling_rng_use():
    d = Dynamics(FreePotential(), [1., 1.], 4., 1, 0.05, native=False,
                 integrator="velocity-verlet", bond_constraints=[(0, 1, 1.)])
    before = copy.deepcopy(d.operator_state_dict())
    shot_rng = np.random.default_rng(37)
    saved_shot_rng = copy.deepcopy(shot_rng.bit_generator.state)
    with pytest.raises(ValueError, match="bond constraints"):
        CommittorSampler(d, Cartesian(0), basin_a=-0.1, basin_b=0.1,
                         seed=shot_rng)
    assert d.operator_state_dict() == before
    assert shot_rng.bit_generator.state == saved_shot_rng


@pytest.mark.parametrize("initial_x", [0., -0.2])
def test_recheck_composed_plan_before_each_shot(initial_x):
    s = sampler()
    # A later operator change must also be rejected for step-zero basin hits.
    s.dynamics.plan = PropagationPlan(thermostat="andersen")
    before = copy.deepcopy(s.rng.bit_generator.state)
    dyn_before = copy.deepcopy(s.dynamics.rng.bit_generator.state)
    with pytest.raises(ValueError, match="no thermostat, barostat or force splitting"):
        s.shoot(np.array([[[initial_x, 0., 0.]]]))
    assert s.rng.bit_generator.state == before
    assert s.dynamics.rng.bit_generator.state == dyn_before


@pytest.mark.parametrize("q", [np.zeros((1, 3)), np.zeros((2, 1, 3)),
                                np.full((1, 1, 3), np.nan),
                                np.ones((1, 1, 3), complex)])
def test_invalid_full_ring_input_rejected(q):
    with pytest.raises(ValueError):
        sampler().shoot(q)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_shot_count_rejected(count):
    with pytest.raises(ValueError):
        sampler().sample(np.zeros((1, 1, 3)), n_shots=count)


def test_periodic_and_vector_coordinates_rejected():
    d = sampler().dynamics
    c = Cartesian(0)
    c.periods = (2.,)
    with pytest.raises(ValueError, match="nonperiodic"):
        CommittorSampler(d, c, basin_a=-1, basin_b=1)
    with pytest.raises(ValueError, match="scalar"):
        CommittorSampler(d, CVStack([Cartesian(0), Cartesian(0, 1)]),
                         basin_a=-1, basin_b=1)


def test_invalid_coordinate_values_and_attempted_mutation():
    q = np.zeros((1, 1, 3))
    for value in (np.nan, True, 1j, [1, 2]):
        s = sampler()
        s._coordinate = lambda q, value=value: value
        with pytest.raises(ValueError, match="finite real"):
            s.shoot(q)
    s = sampler()
    def mutate(q):
        q[:] = 1
        return 0
    s._coordinate = mutate
    with pytest.raises(ValueError):
        s.shoot(q)
    assert np.all(q == 0)


def test_result_rejects_censored_misclassification():
    with pytest.raises(ValueError, match="termination"):
        CommittorResult(np.zeros((1, 1, 3)), np.array([1]), np.array([8]),
                         np.array([0.]), {}, 4., 0.05, 8, -0.1, 0.1, None)


def test_first_hit_at_exact_horizon_and_early_hit():
    for horizon in (4, 8):
        s = sampler(dt=0.125, max_steps=horizon)
        s.basin_a, s.basin_b = -0.5, 0.5
        s._momenta = lambda: np.array([[[1., 0., 0.]]])
        shot = s.shoot(np.zeros((1, 1, 3)))
        assert shot.outcome == "B" and shot.steps == 4
        assert shot.final_value == pytest.approx(0.5)


def test_classical_double_well_symmetry():
    class DoubleWell:
        def evaluate(self, q):
            e = np.sum((q[:, :, 0] ** 2 - 1) ** 2, axis=1)
            e += 0.5 * np.sum(q[:, :, 1:] ** 2, axis=(1, 2))
            force = -q.copy()
            force[:, :, 0] = -4 * q[:, :, 0] * (q[:, :, 0] ** 2 - 1)
            return PESResult(e, force)
    s = sampler(potential=DoubleWell(), max_steps=500, dt=0.01)
    r = s.sample(np.zeros((1, 1, 3)), n_shots=24)
    p = np.random.default_rng(15).normal(size=(24, 1, 1, 3))
    np.testing.assert_array_equal(r.outcomes, np.sign(p[:, 0, 0, 0]).astype(int))
    assert r.summary()["n_unresolved"] == 0


@pytest.mark.skipif(not native_capabilities()["available"],
                    reason="native extension not installed")
def test_native_and_numpy_propagation_agree():
    q = np.zeros((4, 1, 3))
    expected = sampler(beads=4, native=False).sample(q, n_shots=20)
    actual = sampler(beads=4, native="required").sample(q, n_shots=20)
    np.testing.assert_array_equal(actual.outcomes, expected.outcomes)
    np.testing.assert_array_equal(actual.steps, expected.steps)
    np.testing.assert_allclose(actual.final_values, expected.final_values, atol=1e-12)
