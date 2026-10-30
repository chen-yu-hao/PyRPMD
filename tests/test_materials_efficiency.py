"""Reuse physical PES results without changing accepted states or statistics."""

import numpy as np
import pytest

from pyrpmd.analysis.diagnostics import correlated_mean_diagnostics, nve_energy_diagnostics
from pyrpmd.analysis.diffusion import analyze_diffusion
from pyrpmd.constants import FS_TO_AU
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import PESResult
from pyrpmd.potentials.builtin import HarmonicPotential
from pyrpmd.potentials.cache import StepEvaluationCache
from pyrpmd.workflows.materials import quantum_observables


class CountingHarmonic(HarmonicPotential):
    def __init__(self):
        super().__init__(k=0.4)
        self.calls = 0
        self.fail_at = None

    def evaluate(self, q):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("endpoint PES failure")
        return super().evaluate(q)


@pytest.mark.parametrize("scheme", ["cayley-bcb", "normal-mode-bab", "velocity-verlet"])
def test_step_and_quantum_observation_share_endpoint_without_changing_trajectory(scheme):
    raw = CountingHarmonic()
    cached = StepEvaluationCache(raw)
    kwargs = dict(masses=[2.0, 5.0], beta=3.1, nbeads=4, dt=0.03,
                  native=False, loop_backend="python", integrator=scheme)
    dynamics = Dynamics(cached, **kwargs)
    reference = Dynamics(HarmonicPotential(k=0.4), **kwargs)
    rng = np.random.default_rng(89)
    state = State(rng.normal(size=(4, 2, 3)), rng.normal(size=(4, 2, 3)))
    expected = state.copy()
    for _ in range(3):
        with cached.step():
            dynamics.propagate(state, 1, record_centroids=False)
            values, radii = quantum_observables(state, cached, dynamics.masses, dynamics.beta)
        reference.step(expected)
        expected_values, expected_radii = quantum_observables(
            expected, reference.potential, reference.masses, reference.beta)
        np.testing.assert_array_equal(state.q, expected.q)
        np.testing.assert_array_equal(state.p, expected.p)
        np.testing.assert_array_equal(values, expected_values)
        np.testing.assert_array_equal(radii, expected_radii)
    # Both physical force evaluations are retained. Energy and observations
    # reuse them instead of adding three provider calls per saved frame.
    assert raw.calls == 6


@pytest.mark.parametrize("scheme", ["cayley-bcb", "normal-mode-bab"])
def test_sampling_observation_reuses_endpoint_and_preserves_rng(scheme):
    raw = CountingHarmonic()
    cached = StepEvaluationCache(raw)
    kwargs = dict(masses=[2.0], beta=3.1, nbeads=3, dt=0.03,
                  seed=82, native=False, loop_backend="python", integrator=scheme)
    dynamics = Dynamics(cached, **kwargs)
    reference = Dynamics(HarmonicPotential(k=0.4), **kwargs)
    state = State(np.ones((3, 1, 3)), np.ones((3, 1, 3)))
    expected = state.copy()
    with cached.step():
        dynamics.sample_step(state, gamma_centroid=0.6)
        quantum_observables(state, cached, dynamics.masses, dynamics.beta)
    reference.sample_step(expected, gamma_centroid=0.6)
    np.testing.assert_array_equal(state.q, expected.q)
    np.testing.assert_array_equal(state.p, expected.p)
    assert dynamics.rng.bit_generator.state == reference.rng.bit_generator.state
    assert raw.calls == 2


def test_failed_step_discards_cached_results_and_retry_reconsults_provider():
    raw = CountingHarmonic()
    raw.fail_at = 2
    cached = StepEvaluationCache(raw)
    dynamics = Dynamics(cached, [2.0], 3.1, 3, 0.03, native=False, loop_backend="python")
    state = State(np.ones((3, 1, 3)), np.ones((3, 1, 3)))
    initial = state.copy()
    with pytest.raises(RuntimeError, match="endpoint PES failure"):
        with cached.step():
            dynamics.propagate(state, 1)
    np.testing.assert_array_equal(state.q, initial.q)
    np.testing.assert_array_equal(state.p, initial.p)
    assert state.step == initial.step and state.time == initial.time
    with cached.step():
        dynamics.propagate(state, 1)
    assert raw.calls == 4
    reference = Dynamics(HarmonicPotential(k=0.4), [2.0], 3.1, 3, 0.03, native=False)
    reference.step(initial)
    np.testing.assert_array_equal(state.q, initial.q)
    np.testing.assert_array_equal(state.p, initial.p)


def test_cache_owns_reused_provider_buffers_and_detects_in_place_coordinate_changes():
    class ReusedBuffers:
        def __init__(self):
            self.calls = 0
            self.forces = np.zeros((1, 1, 3))
            self.energy = np.zeros(1)

        def evaluate(self, q):
            self.calls += 1
            self.forces[:] = -q
            self.energy[:] = np.sum(q * q) / 2
            return PESResult(self.energy, self.forces)

    provider = ReusedBuffers()
    cached = StepEvaluationCache(provider)
    q = np.ones((1, 1, 3))
    with cached.step():
        first = cached.evaluate(q)
        cached.evaluate(q.copy())
        assert provider.calls == 1
        q *= 2
        second = cached.evaluate(q)
        np.testing.assert_array_equal(first.forces, -np.ones_like(q))
        np.testing.assert_array_equal(second.forces, -q)
    provider.energy[:] = 99
    assert first.energy[0] == 1.5
    with cached.step():
        cached.evaluate(q)
    assert provider.calls == 3


def test_nve_drift_is_independent_of_energy_zero_and_uses_per_bead_thermal_scale():
    times = np.arange(9) * FS_TO_AU
    energies = np.array([4 + np.arange(9) / 64, -7 - np.arange(9) / 32])
    result = nve_energy_diagnostics(energies, times, beads=4, beta=2)
    shifted = nve_energy_diagnostics(energies + 128, times, beads=4, beta=2)
    assert result == shifted
    assert result["per_replica"][0]["linear_drift_kbt_per_bead_per_fs"] == pytest.approx(1 / 128)
    assert result["per_replica"][1]["endpoint_error_kbt_per_bead"] == pytest.approx(-0.125)
    assert result["per_replica"][0]["within_tolerance"] is True
    assert result["per_replica"][1]["within_tolerance"] is False
    assert result["within_tolerance"] is False


def test_conditional_batch_error_distinguishes_long_independent_series_from_missing_support():
    values = np.random.default_rng(742).normal(size=16384)
    result = correlated_mean_diagnostics(values)
    assert result["conditional_standard_error"] == pytest.approx(1 / np.sqrt(len(values)), rel=0.15)
    assert result["reason"] is None
    assert correlated_mean_diagnostics(values[:3])["conditional_standard_error"] is None
    assert correlated_mean_diagnostics(np.ones(20))["conditional_standard_error"] is None


def test_diffusion_selection_accepts_readonly_arrays_without_modifying_reference_frame_inputs():
    rng = np.random.default_rng(718)
    positions = rng.normal(size=(2, 65, 8, 3)).cumsum(axis=1)
    velocities = rng.normal(size=positions.shape)
    references = positions[:, :, 4:].copy()
    reference_velocities = velocities[:, :, 4:].copy()
    arrays = (positions, velocities, references, reference_velocities)
    original = [array.copy() for array in arrays]
    for array in arrays:
        array.flags.writeable = False
    result = analyze_diffusion(
        positions, np.arange(65), velocities=velocities, atom_indices=[1, 0], axes=[2, 0],
        reference_positions=references, reference_velocities=reference_velocities,
        reference_masses=[2, 3, 4, 5], max_lag=16, bootstrap=0,
    )
    assert result["input"]["atom_indices"] == [1, 0]
    for array, expected in zip(arrays, original):
        np.testing.assert_array_equal(array, expected)
