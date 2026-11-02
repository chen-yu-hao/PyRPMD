"""RATTLE, conditional-measure and restart checks independent of real-time QCMD."""

import copy
import json

import numpy as np
import pytest

from pyrpmd._qcmd_geometry import TriatomicQuasiCentroid
from pyrpmd._qcmd_sampling import ConstrainedQuasiCentroidSampler
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.dynamics import State
from pyrpmd.potentials import FreePotential, HarmonicPotential


def setup(beads=4, *, seed=87, dt=.015, **kwargs):
    geometry = TriatomicQuasiCentroid([[0, 1, 2]], [16., 1., 2.])
    reference = np.array([[0., 0., 0.], [1.8, 0., 0.], [-.5, 1.8, 0.]])
    q = reference[None] + np.random.default_rng(711).normal(scale=.035, size=(beads, 3, 3))
    sampler = ConstrainedQuasiCentroidSampler(HarmonicPotential(.2), geometry,
                geometry.mean_coordinates(q), beta=4., nbeads=beads, dt=dt,
                trajectory_steps=3, tolerance=1e-12, seed=seed, **kwargs)
    return sampler, State(q, np.zeros_like(q))


def test_projected_maxwell_has_exact_tangent_covariance():
    sampler, state = setup(beads=1)
    jacobian = sampler.constraint.local_jacobian(state.q)[0].reshape(3, 9)
    mass = np.repeat(sampler.geometry.masses, 3)
    gram = (jacobian / mass) @ jacobian.T
    # Integrating a Gaussian with linear tangent constraints gives this
    # covariance; the reference does not call project_momenta.
    expected = (np.diag(mass) - jacobian.T @ np.linalg.solve(gram, jacobian)) / sampler.beta
    values = np.stack([sampler.sample_momenta(state.q).ravel() for _ in range(6000)])
    normalized = values / np.sqrt(mass)[None]
    actual = normalized.T @ normalized / len(values)
    np.testing.assert_allclose(actual, expected / np.sqrt(mass[:, None] * mass[None]), atol=.018)
    np.testing.assert_allclose((values / mass) @ jacobian.T, 0, atol=2e-15)


@pytest.mark.parametrize("beads", [1, 2, 5])
def test_ring_hamiltonian_and_force_match_direct_spring_formula(beads):
    sampler, state = setup(beads)
    energy, force = sampler._energy_force(state.q)
    physical = .5 * .2 * np.sum(state.q ** 2)
    springs = .5 * (beads / sampler.beta) ** 2 * sum(
        np.sum(sampler.geometry.masses[:, None] * (state.q[b] - state.q[(b+1) % beads]) ** 2)
        for b in range(beads))
    assert energy == pytest.approx(physical + springs, abs=1e-14)
    numerical = np.empty_like(force)
    epsilon = 1e-6
    for bead in range(beads):
        for atom in range(3):
            for axis in range(3):
                plus, minus = state.q.copy(), state.q.copy()
                plus[bead, atom, axis] += epsilon
                minus[bead, atom, axis] -= epsilon
                numerical[bead, atom, axis] = -(sampler._energy_force(plus)[0] - sampler._energy_force(minus)[0]) / (2 * epsilon)
    np.testing.assert_allclose(force, numerical, atol=8e-10)


def test_rattle_reversibility_and_constraint_satisfaction():
    sampler, state = setup()
    p = sampler.sample_momenta(state.q)
    q1, p1, _ = sampler._trajectory(state.q, p)
    q2, p2, _ = sampler._trajectory(q1, -p1)
    np.testing.assert_allclose(q2, state.q, atol=2e-12)
    np.testing.assert_allclose(p2, -p, atol=3e-10)
    np.testing.assert_allclose(sampler.constraint.values(q1), 0., atol=1e-12)
    np.testing.assert_allclose(sampler.constraint.velocity_residual(q1, p1), 0., atol=2e-15)


def test_free_molecule_com_drift_is_exact_independent_cartesian_flow():
    sampler, state = setup(beads=5)
    sampler.potential = FreePotential()
    p = sampler.sample_momenta(state.q)
    duration = sampler.dt * sampler.trajectory_steps
    mass = sampler.geometry.masses
    initial_com = (state.q * mass[None, :, None]).sum(axis=(0, 1)) / (len(state.q) * mass.sum())
    velocity = p.sum(axis=(0, 1)) / (len(state.q) * mass.sum())
    q1, p1, _ = sampler._trajectory(state.q, p)
    actual_com = (q1 * mass[None, :, None]).sum(axis=(0, 1)) / (len(state.q) * mass.sum())
    np.testing.assert_allclose(actual_com, initial_com + duration * velocity, atol=2e-15)
    np.testing.assert_allclose(p1.sum(axis=(0, 1)), p.sum(axis=(0, 1)), atol=3e-13)


def test_metropolis_ratio_contains_conditional_measure_fixman_factor():
    sampler, state = setup()
    reference, reference_state = setup()
    p0 = reference.sample_momenta(reference_state.q)
    q1, p1, _ = reference._trajectory(reference_state.q, p0)
    beta_p = sampler.beta / sampler.nbeads

    def hamiltonian(q, p):
        physical = .5 * .2 * np.sum(q ** 2)
        difference = q - np.roll(q, -1, axis=0)
        spring = .5 * (len(q) / sampler.beta) ** 2 * np.sum(sampler.geometry.masses[None, :, None] * difference ** 2)
        kinetic = .5 * np.sum(p * p / sampler.geometry.masses[None, :, None])
        return physical + spring + kinetic

    def independent_logdet(q):
        # Independent Cartesian finite differences of the bead-averaged
        # constraints verify both their 1/P scaling and the metric measure.
        epsilon = 2e-6
        c = np.empty((3, q.size))
        for j in range(q.size):
            plus, minus = q.copy(), q.copy()
            plus.flat[j] += epsilon
            minus.flat[j] -= epsilon
            c[:, j] = ((reference.geometry.mean_coordinates(plus)
                        - reference.geometry.mean_coordinates(minus)) / (2 * epsilon)).ravel()
        invmass = np.tile(np.repeat(1 / sampler.geometry.masses, 3), len(q))
        return np.linalg.slogdet((c * invmass) @ c.T)[1]

    expected = (-beta_p * (hamiltonian(q1, p1) - hamiltonian(state.q, p0))
                - .5 * (independent_logdet(q1) - independent_logdet(state.q)))
    sampler.step(state)
    assert sampler.last_log_acceptance == pytest.approx(expected, abs=3e-9)
    assert sampler.attempted == state.step == 1
    assert sampler.metadata()["real_time_dynamics"] is False
    np.testing.assert_allclose(sampler.constraint.values(state.q), 0., atol=1e-12)


def test_checkpoint_restores_rng_state_and_complete_markov_chain(tmp_path):
    sampler, state = setup()
    for _ in range(9):
        sampler.step(state)
    path = tmp_path / "sampler.npz"
    sampler.save_checkpoint(path, state, potential_identity="harmonic:k=0.2")
    continued, _ = setup(seed=999)
    loaded = continued.load_checkpoint(path, potential_identity="harmonic:k=0.2")
    for _ in range(12):
        sampler.step(state)
        continued.step(loaded)
        np.testing.assert_array_equal(loaded.q, state.q)
        np.testing.assert_array_equal(loaded.p, state.p)
    assert (loaded.step, loaded.time) == (state.step, state.time)
    assert (continued.attempted, continued.accepted) == (sampler.attempted, sampler.accepted)
    assert continued.rng.bit_generator.state == sampler.rng.bit_generator.state
    with pytest.raises(ValueError, match="potential identity changed"):
        continued.load_checkpoint(path, potential_identity="harmonic:k=0.3")
    changed, _ = setup(dt=.01)
    with pytest.raises(ValueError, match="physics"):
        changed.load_checkpoint(path, potential_identity="harmonic:k=0.2")
    with pytest.raises(ValueError, match="potential_identity"):
        continued.save_checkpoint(path, loaded, potential_identity="")


def test_checkpoint_invalid_rng_and_counters_do_not_mutate_sampler(tmp_path):
    sampler, state = setup()
    sampler.step(state)
    path = tmp_path / "sampler.npz"
    sampler.save_checkpoint(path, state, potential_identity="harmonic:k=0.2")
    with load_checkpoint(path) as data:
        arrays = {key: data[key].copy() for key in data.files}
    before_rng = copy.deepcopy(sampler.rng.bit_generator.state)
    before_counters = sampler.attempted, sampler.accepted, sampler.last_log_acceptance
    arrays["rng"] = json.dumps({"bit_generator": "bad"})
    atomic_savez(path, kind="PyRPMD.quasicentroid-sampler.v1", **arrays)
    with pytest.raises(ValueError):
        sampler.load_checkpoint(path, potential_identity="harmonic:k=0.2")
    assert sampler.rng.bit_generator.state == before_rng
    assert (sampler.attempted, sampler.accepted, sampler.last_log_acceptance) == before_counters
    arrays["accepted"] = -1
    atomic_savez(path, kind="PyRPMD.quasicentroid-sampler.v1", **arrays)
    with pytest.raises(ValueError, match="accepted"):
        sampler.load_checkpoint(path, potential_identity="harmonic:k=0.2")


def test_potential_failure_rolls_back_rng_state_clock_and_counters():
    sampler, state = setup()
    before = state.copy()
    rng_before = copy.deepcopy(sampler.rng.bit_generator.state)
    base = sampler.potential

    class Failing:
        calls = 0

        def evaluate(self, q):
            self.calls += 1
            if self.calls == 4:
                raise RuntimeError("injected constrained sampling failure")
            return base.evaluate(q)

    sampler.potential = Failing()
    with pytest.raises(RuntimeError, match="injected"):
        sampler.step(state)
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert sampler.rng.bit_generator.state == rng_before
    assert sampler.attempted == sampler.accepted == state.step == 0
    assert sampler.last_log_acceptance is None
    sampler.potential = base
    sampler.step(state)
    other, expected = setup()
    other.step(expected)
    np.testing.assert_array_equal(state.q, expected.q)
    np.testing.assert_array_equal(state.p, expected.p)


def test_reversibility_failure_is_explicit_and_transactional(monkeypatch):
    sampler, state = setup()
    original = sampler._trajectory
    count = 0

    def broken(q, p):
        nonlocal count
        count += 1
        result_q, result_p, energy = original(q, p)
        if count == 2:
            result_q += .01
        return result_q, result_p, energy

    monkeypatch.setattr(sampler, "_trajectory", broken)
    before = state.copy()
    rng = copy.deepcopy(sampler.rng.bit_generator.state)
    with pytest.raises(ValueError, match="reversibility"):
        sampler.step(state)
    np.testing.assert_array_equal(before.q, state.q)
    assert sampler.rng.bit_generator.state == rng
    assert state.step == sampler.attempted == 0


@pytest.mark.parametrize("field,value", [("beta", 0), ("nbeads", True), ("dt", np.inf),
                                         ("trajectory_steps", 0), ("reversibility_tolerance", True)])
def test_sampler_rejects_invalid_scales(field, value):
    geometry = TriatomicQuasiCentroid([[0, 1, 2]], [16., 1., 1.])
    options = dict(beta=4., nbeads=4, dt=.01, trajectory_steps=1)
    options[field] = value
    with pytest.raises(ValueError):
        ConstrainedQuasiCentroidSampler(FreePotential(), geometry, [[1.8, 1.8, 1.8]], **options)


def test_sampler_rejects_unconstrained_state_mutation_and_periodic_potential():
    sampler, state = setup()
    state.q[:, 1] *= 1.1
    with pytest.raises(ValueError, match="project it explicitly"):
        sampler.step(state)
    sampler, state = setup()
    sampler.dt *= 2
    with pytest.raises(ValueError, match="immutable"):
        sampler.step(state)
    potential = FreePotential()
    potential.pbc = np.array([True, True, True])
    with pytest.raises(ValueError, match="nonperiodic"):
        ConstrainedQuasiCentroidSampler(potential, sampler.geometry, sampler.constraint.target,
                                        beta=4., nbeads=4, dt=.01)


def test_periodic_capability_does_not_imply_active_periodic_boundaries():
    geometry = TriatomicQuasiCentroid([[0, 1, 2]], [16., 1., 1.])
    sampler = ConstrainedQuasiCentroidSampler(FreePotential(), geometry, [[1.8, 1.8, 1.8]],
                                              beta=4., nbeads=4, dt=.01)
    assert sampler.metadata()["constraint"]["geometry"]["periodic"] is False
