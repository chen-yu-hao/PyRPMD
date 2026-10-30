"""Independent equilibrium checks for shared-number electrode MC/HMC."""

from copy import deepcopy
from contextlib import contextmanager
import math

import numpy as np
import pytest

from pyrpmd.bias import CompositeBias, UmbrellaBias
from pyrpmd.cv import Cartesian
from pyrpmd.electrochemistry import ElectronicPESResult, ElectronReservoir
from pyrpmd.electrochemistry.sampling import GrandCanonicalSampler, GrandCanonicalState


class QuadraticElectrode:
    """Positive Gaussian q/N model with analytically integrable centroids."""

    def __init__(self, k=0.5, a=0.7, coupling=0.2):
        self.k, self.a, self.coupling = k, a, coupling
        self.calls = 0
        self.fail_on_call = None

    def evaluate(self, q, electron_numbers):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("deliberate electronic model failure")
        number = np.asarray(electron_numbers)
        energy = 0.5 * self.k * np.sum(q * q, axis=(1, 2))
        energy += 0.5 * self.a * number**2 - self.coupling * number * q[:, 0, 0]
        force = -self.k * q.copy()
        force[:, 0, 0] += self.coupling * number
        return ElectronicPESResult(energy, force, self.a * number - self.coupling * q[:, 0, 0])


def make_sampler(beads=4, *, model=None, mu=0.0, **options):
    defaults = dict(
        model=QuadraticElectrode() if model is None else model,
        masses=[2.0], beta=3.0, nbeads=beads, dt=0.4,
        reservoir=ElectronReservoir(0.0, 0.0, mu, "model calibration"),
        electron_bounds=(-8.0, 8.0), proposal_width=1.5,
        seed=8162, native=False,
    )
    defaults.update(options)
    return GrandCanonicalSampler(**defaults)


@pytest.mark.parametrize("beads", [1, 4])
def test_joint_hmc_mc_matches_exact_centroid_number_covariance(beads):
    sampler = make_sampler(beads)
    state = GrandCanonicalState(np.zeros((beads, 1, 3)), 0.0)
    samples = []
    for sweep in range(5700):
        sampler.step(state, nuclear_steps=8, electron_moves=2)
        if sweep >= 700:
            samples.append([state.q[:, 0, 0].mean(), state.electron_number])
    samples = np.asarray(samples)
    # Integrating every non-centroid ring mode leaves exactly this 2D Gaussian,
    # independent of bead count. This reference uses no sampling implementation.
    inverse_precision = np.linalg.inv(np.array([[0.5, -0.2], [-0.2, 0.7]])) / 3.0
    np.testing.assert_allclose(samples.mean(axis=0), 0.0, atol=0.065)
    np.testing.assert_allclose(np.cov(samples.T), inverse_precision, rtol=0.12, atol=0.035)
    assert sampler.statistics["nuclear_acceptance"] > 0.75
    assert 0.35 < sampler.statistics["electron_acceptance"] < 0.85
    assert state.sweeps == 5700
    assert not hasattr(state, "time") and not hasattr(state, "p")


def test_conditional_electron_chain_uses_physical_beta_and_shared_mean():
    beta, stiffness, mu, width = 3.0, 0.7, 0.12, 0.85
    seed = 213
    expected_rng = np.random.default_rng(seed)
    expected = 0.0
    chains = [make_sampler(
        beads, model=QuadraticElectrode(k=0, a=stiffness, coupling=0),
        beta=beta, mu=mu, proposal_width=width, seed=seed,
        electron_bounds=(-1.0, 1.0),
    ) for beads in (1, 4)]
    states = [GrandCanonicalState(np.zeros((beads, 1, 3)), 0.0) for beads in (1, 4)]
    for _ in range(150):
        candidate = expected + expected_rng.uniform(-width, width)
        if -1 <= candidate <= 1:
            energy_difference = 0.5 * stiffness * (candidate**2 - expected**2) - mu * (candidate - expected)
            if math.log(expected_rng.random()) < min(0.0, -beta * energy_difference):
                expected = candidate
        for sampler, state in zip(chains, states):
            result = sampler.step(state, nuclear_steps=0)
            assert state.electron_number == pytest.approx(expected, abs=1e-14)
            assert result["nuclear_accepted"] is None
            assert result["physical_grand_energy_au"] == pytest.approx(
                0.5 * stiffness * expected**2 - mu * expected,
            )
            np.testing.assert_array_equal(state.q, 0.0)
    assert chains[0].statistics == chains[1].statistics


def test_rejecting_out_of_bounds_moves_preserves_uniform_target():
    sampler = make_sampler(
        model=QuadraticElectrode(k=0, a=0, coupling=0),
        electron_bounds=(-1.0, 1.0), proposal_width=1.6, seed=882,
    )
    state = GrandCanonicalState(np.zeros((4, 1, 3)), -1.0)
    samples = []
    for sweep in range(8500):
        sampler.step(state, nuclear_steps=0)
        if sweep >= 500:
            samples.append(state.electron_number)
    samples = np.array(samples)
    assert abs(samples.mean()) < 0.045
    assert samples.var() == pytest.approx(1 / 3, abs=0.035)
    np.testing.assert_allclose(np.histogram(samples, bins=np.linspace(-1, 1, 11))[0] / len(samples), 0.1, atol=0.023)
    assert np.all(np.abs(samples) < 1.0)  # clipping would create endpoint atoms
    stats = sampler.statistics
    assert stats["electron_out_of_bounds"] > 1500
    assert stats["electron_accepted"] + stats["electron_out_of_bounds"] == 8500
    assert stats["nuclear_acceptance"] is None


def test_static_umbrella_reuses_physical_bias_scaling_and_fixed_substrate():
    beads, beta, stiffness, center = 4, 2.0, 1.2, 0.7
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    sampler = make_sampler(
        beads, model=QuadraticElectrode(coupling=0.0), masses=[2.0, 5.0], beta=beta,
        dt=0.3, fixed_atoms=(1,), fixed_positions=positions, seed=41,
    )
    state = GrandCanonicalState(np.repeat(positions[None], beads, axis=0), 0.09)
    bias = UmbrellaBias(Cartesian(0, 0), [center], stiffness)
    centroids = []
    for sweep in range(3200):
        sampler.step(state, nuclear_steps=5, electron_moves=0, bias=bias)
        if sweep >= 400:
            centroids.append(state.q[:, 0, 0].mean())
    expected_mean = stiffness * center / (0.5 + stiffness)
    expected_variance = 1 / (beta * (0.5 + stiffness))
    assert np.mean(centroids) == pytest.approx(expected_mean, abs=0.055)
    assert np.var(centroids) == pytest.approx(expected_variance, abs=0.045)
    np.testing.assert_array_equal(state.q[:, 1], np.repeat(positions[None, 1], beads, axis=0))
    assert state.electron_number == 0.09
    assert sampler.statistics["electron_acceptance"] is None


@pytest.mark.parametrize("failure_call", [3, 5])
def test_failed_proposal_rolls_back_configuration_random_stream_and_counters(failure_call):
    model = QuadraticElectrode()
    sampler = make_sampler(model=model, seed=14)
    reference = make_sampler(seed=14)
    state = GrandCanonicalState(np.full((4, 1, 3), 0.2), 0.1)
    before = state.copy()
    rng = deepcopy(sampler.rng.bit_generator.state)
    stats = sampler.statistics
    model.fail_on_call = failure_call
    with pytest.raises(RuntimeError, match="deliberate"):
        sampler.step(state, nuclear_steps=2, electron_moves=2)
    np.testing.assert_array_equal(state.q, before.q)
    assert state.electron_number == before.electron_number
    assert state.sweeps == before.sweeps
    assert sampler.rng.bit_generator.state == rng
    assert sampler.statistics == stats
    model.fail_on_call = None
    actual = sampler.step(state, nuclear_steps=2, electron_moves=2)
    expected = reference.step(before, nuclear_steps=2, electron_moves=2)
    np.testing.assert_array_equal(state.q, before.q)
    assert actual == expected
    assert sampler.statistics == reference.statistics


def test_rejected_hmc_preserves_configuration_and_counts_the_attempt():
    sampler = make_sampler(
        1, model=QuadraticElectrode(k=20.0, coupling=0.0), beta=10.0, dt=1.5,
    )
    state = GrandCanonicalState(np.ones((1, 1, 3)), 0.0)
    result = sampler.step(state, nuclear_steps=1, electron_moves=0)
    assert result["nuclear_accepted"] is False
    np.testing.assert_array_equal(state.q, 1.0)
    assert state.sweeps == 1
    assert sampler.statistics["nuclear_attempted"] == 1
    assert sampler.statistics["nuclear_accepted"] == 0


class PassthroughCache:
    """Uncached control with the cache's public context/evaluation interface."""

    def __init__(self, provider):
        self.provider = provider

    def __getattr__(self, name):
        return getattr(self.provider, name)

    @contextmanager
    def step(self):
        yield self

    def evaluate(self, q):
        return self.provider.evaluate(q)


def test_fixed_number_hmc_reuses_each_energy_force_endpoint_once(monkeypatch):
    from pyrpmd.electrochemistry import sampling

    models = [QuadraticElectrode(), QuadraticElectrode()]
    cached = make_sampler(model=models[0], dt=0.01)
    monkeypatch.setattr(sampling, "StepEvaluationCache", PassthroughCache)
    uncached = make_sampler(model=models[1], dt=0.01)
    states = [GrandCanonicalState(np.zeros((4, 1, 3)), 0.2) for _ in range(2)]
    steps = 5
    actual = cached.step(states[0], nuclear_steps=steps, electron_moves=0)
    reference = uncached.step(states[1], nuclear_steps=steps, electron_moves=0)
    # Both paths additionally evaluate the post-HMC physical grand energy once
    # outside the transaction. HMC itself needs L+1 rather than 2*L+2 calls.
    assert models[0].calls - 1 == steps + 1
    assert models[1].calls - 1 == 2 * steps + 2
    np.testing.assert_array_equal(states[0].q, states[1].q)
    assert actual == reference
    assert cached.rng.bit_generator.state == uncached.rng.bit_generator.state


def test_cached_hmc_matches_uncached_across_number_changes_and_sweeps(monkeypatch):
    from pyrpmd.electrochemistry import sampling

    cached = make_sampler(seed=77, mu=0.1)
    monkeypatch.setattr(sampling, "StepEvaluationCache", PassthroughCache)
    uncached = make_sampler(seed=77, mu=0.1)
    states = [GrandCanonicalState(np.full((4, 1, 3), 0.2), 0.1) for _ in range(2)]
    previous_number = states[0].electron_number
    number_changed = False
    for _ in range(3):
        actual = cached.step(states[0], nuclear_steps=4, electron_moves=2)
        reference = uncached.step(states[1], nuclear_steps=4, electron_moves=2)
        np.testing.assert_array_equal(states[0].q, states[1].q)
        assert states[0].electron_number == states[1].electron_number
        assert actual == reference
        assert cached.statistics == uncached.statistics
        assert cached.rng.bit_generator.state == uncached.rng.bit_generator.state
        number_changed |= states[0].electron_number != previous_number
        previous_number = states[0].electron_number
    assert number_changed


@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab", "velocity-verlet"])
def test_existing_native_and_numpy_free_ring_maps_agree(integrator):
    states = [GrandCanonicalState(np.arange(12).reshape(4, 1, 3) / 30, 0.1) for _ in range(2)]
    samplers = [make_sampler(native=native, integrator=integrator, dt=0.04, seed=5) for native in (False, True)]
    results = [sampler.step(state, nuclear_steps=7, electron_moves=2) for sampler, state in zip(samplers, states)]
    np.testing.assert_allclose(states[0].q, states[1].q, rtol=1e-13, atol=1e-14)
    assert states[0].electron_number == states[1].electron_number
    assert results[0]["nuclear_accepted"] == results[1]["nuclear_accepted"]
    assert samplers[0].statistics == samplers[1].statistics


@pytest.mark.parametrize("nested", [False, True])
def test_history_dependent_bias_is_rejected_before_random_draws(nested):
    class ChangingBias:
        history_dependent = True

        def energy_force(self, q):
            raise AssertionError("must reject before calling a changing bias")

    bias = CompositeBias(ChangingBias()) if nested else ChangingBias()
    sampler = make_sampler()
    state = GrandCanonicalState(np.zeros((4, 1, 3)), 0)
    rng = deepcopy(sampler.rng.bit_generator.state)
    with pytest.raises(ValueError, match="static bias"):
        sampler.step(state, bias=bias)
    assert sampler.rng.bit_generator.state == rng
    assert state.sweeps == 0


@pytest.mark.parametrize("options", [
    {"beta": 0}, {"beta": True}, {"dt": -0.1}, {"dt": np.nan},
    {"electron_bounds": (0, 0)}, {"electron_bounds": (0, np.inf)},
    {"electron_bounds": (-1, 0, 1)}, {"proposal_width": 0},
    {"proposal_width": True}, {"reservoir": {}},
])
def test_invalid_sampler_inputs_are_rejected(options):
    with pytest.raises((ValueError, TypeError)):
        make_sampler(**options)


@pytest.mark.parametrize("options", [
    {"nuclear_steps": 0, "electron_moves": 0}, {"nuclear_steps": True},
    {"electron_moves": 1.5}, {"nuclear_steps": -1},
])
def test_invalid_move_counts_are_rejected(options):
    sampler = make_sampler()
    state = GrandCanonicalState(np.zeros((4, 1, 3)), 0)
    with pytest.raises(ValueError):
        sampler.step(state, **options)


def test_state_owns_coordinates_and_statistics_are_detached():
    q = np.zeros((4, 1, 3))
    state = GrandCanonicalState(q, 0.0)
    q[:] = 7.0
    np.testing.assert_array_equal(state.q, 0)
    sampler = make_sampler()
    statistics = sampler.statistics
    statistics["sweeps"] = 45
    assert sampler.statistics["sweeps"] == 0
    state.electron_number = 9.0
    with pytest.raises(ValueError, match="outside"):
        sampler.step(state)


def test_inconsistent_fixed_atom_configuration_is_rejected():
    sampler = make_sampler(fixed_atoms=(0,), fixed_positions=np.zeros((1, 3)))
    state = GrandCanonicalState(np.ones((4, 1, 3)), 0)
    with pytest.raises(ValueError, match="fixed atoms"):
        sampler.step(state)
