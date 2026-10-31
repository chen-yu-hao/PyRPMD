"""Quantum estimators checked against independently integrated Gaussian paths."""

import numpy as np
import pytest
from scipy.special import ndtri

from pyrpmd.constants import KB_AU
from pyrpmd.path_integral import OpenPathAction, OpenPathSamples, PathIntegralAction, PathIntegralSamples
from pyrpmd.potentials.builtin import HarmonicPotential
from pyrpmd.quantum_statistics import (
    energy_heat_capacity, fractionation_mass_integration, harmonic_log_partition,
    isotope_fractionation, isotope_substitution_free_energy,
    momentum_distribution_from_open_paths, quantum_kinetic_energy,
)


def harmonic_samples(method, *, samples=70000, beads=8, beta=2.0, stiffness=0.64, mass=1.0, seed=10):
    """Exact finite-P Gaussian draws from the discretized harmonic action.

    This independent construction does not invoke either Monte Carlo sampler.
    """
    action = PathIntegralAction(HarmonicPotential(stiffness), [mass], 1 / (KB_AU * beta), method=method)
    modes = np.fft.rfft(np.random.default_rng(seed).normal(size=(samples, beads, 1, 3)), axis=1, norm="ortho")
    eig = 4 * np.sin(np.pi * np.arange(beads // 2 + 1) / beads)**2
    precision = mass * beads / beta * eig + beta * stiffness / beads
    if method == "takahashi_imada":
        precision += beta**3 * stiffness**2 / (12 * mass * beads**3)
    modes /= np.sqrt(precision)[None, :, None, None]
    q = np.fft.irfft(modes, n=beads, axis=1, norm="ortho")
    return PathIntegralSamples(q, action, action.evaluate_batch(q), {}, 0, 0, 1)


@pytest.mark.parametrize("method", ["primitive", "takahashi_imada"])
def test_energy_and_heat_capacity_agree_with_harmonic_log_partition_derivatives(method):
    samples = harmonic_samples(method)
    beta, eps = samples.action.beta, 2e-4
    result = energy_heat_capacity(samples)
    values = [harmonic_log_partition(b, [0.8] * 3, beads=8, method=method)
              for b in (beta - eps, beta, beta + eps)]
    expected_energy = -(values[2] - values[0]) / (2 * eps)
    expected_heat = beta**2 * (values[2] - 2 * values[1] + values[0]) / eps**2
    assert result["internal_energy_hartree"] == pytest.approx(expected_energy, abs=5 * result["internal_energy_standard_error_hartree"])
    assert result["heat_capacity_over_kB"] == pytest.approx(expected_heat, abs=5 * result["heat_capacity_standard_error_over_kB"])
    assert result["physical_converged"] is False


def test_fourth_order_trace_convergence_and_bead_savings():
    beta, frequency = 8.0, [1.0]
    exact = harmonic_log_partition(beta, frequency)
    primitive = [abs(harmonic_log_partition(beta, frequency, beads=p) - exact) for p in (16, 32, 64)]
    ti = [abs(harmonic_log_partition(beta, frequency, beads=p, method="ti") - exact) for p in (16, 32, 64)]
    assert 3.8 < primitive[0] / primitive[1] < 4.1
    assert 15.0 < ti[0] / ti[1] < 16.5
    assert ti[0] < primitive[-1]
    assert harmonic_log_partition(1e4, [1]) == pytest.approx(-5000)


@pytest.mark.parametrize("method", ["primitive", "takahashi_imada"])
def test_scaled_heat_estimator_matches_independent_gaussian_trace_and_reduces_variance(method):
    samples = harmonic_samples(method, samples=25000, beads=16, beta=8, stiffness=1)
    direct = energy_heat_capacity(samples)
    scaled = energy_heat_capacity(samples, estimator="scaled_coordinates")
    beta, h = samples.action.beta, samples.action.beta * 1e-4
    logz = [harmonic_log_partition(b, [1] * 3, beads=16, method=method)
            for b in (beta - h, beta, beta + h)]
    energy = -(logz[2] - logz[0]) / (2 * h)
    heat = beta**2 * (logz[2] - 2 * logz[1] + logz[0]) / h**2
    assert scaled["internal_energy_hartree"] == pytest.approx(energy, abs=5 * scaled["internal_energy_standard_error_hartree"])
    assert scaled["heat_capacity_over_kB"] == pytest.approx(heat, abs=5 * scaled["heat_capacity_standard_error_over_kB"])
    assert scaled["heat_capacity_standard_error_over_kB"] < direct["heat_capacity_standard_error_over_kB"]
    assert scaled["finite_difference"]["relative_beta_step_used"] == 5e-4
    assert abs(scaled["finite_difference"]["heat_capacity_over_kB_coarse_minus_fine"]) < 1e-4


def test_scaled_temperature_finite_difference_sensitivity_and_classical_limit():
    samples = harmonic_samples("takahashi_imada", samples=200, beads=8, beta=8, stiffness=1)
    coarse = energy_heat_capacity(samples, estimator="scaled_coordinates", finite_difference_step=0.02)
    fine = energy_heat_capacity(samples.action, samples.positions, estimator="scaled_coordinates", finite_difference_step=0.01)
    # The harmonic TI transformed action is quartic in beta': central finite
    # differences therefore have a known leading h² sensitivity.
    delta_coarse = abs(coarse["finite_difference"]["energy_hartree_coarse_minus_fine"])
    delta_fine = abs(fine["finite_difference"]["energy_hartree_coarse_minus_fine"])
    assert delta_coarse / delta_fine == pytest.approx(4, rel=1e-5)
    classical = harmonic_samples("primitive", samples=1000, beads=1)
    direct = energy_heat_capacity(classical)
    scaled = energy_heat_capacity(classical, estimator="scaled_coordinates")
    assert scaled["internal_energy_hartree"] == pytest.approx(direct["internal_energy_hartree"], abs=1e-10)
    assert scaled["heat_capacity_over_kB"] == pytest.approx(direct["heat_capacity_over_kB"], abs=1e-7)
    for step in (0, -1, 1, 1e-30, float("nan")):
        with pytest.raises(ValueError, match="finite_difference_step"):
            energy_heat_capacity(classical, estimator="scaled_coordinates", finite_difference_step=step)


@pytest.mark.parametrize("method", ["primitive", "takahashi_imada"])
def test_scaled_mass_reweighting_matches_independent_partition_ratio(method):
    samples = harmonic_samples(method, samples=25000)
    result = isotope_substitution_free_energy(samples, [2.0])
    expected = harmonic_log_partition(2, [np.sqrt(0.64 / 2)] * 3, beads=8, method=method) - harmonic_log_partition(2, [0.8] * 3, beads=8, method=method)
    error = result["log_partition_ratio_standard_error"]
    assert result["log_partition_ratio_new_over_old"] == pytest.approx(expected, abs=5 * error)
    assert result["diagnostics"]["reweighting_ess_fraction"] > 0.95
    same = isotope_fractionation(result, result)
    assert same["alpha_A_over_B"] == 1


def test_classical_mass_prefactor_and_fractionation_cancel_exactly():
    samples = harmonic_samples("primitive", samples=100, beads=1)
    for estimator in ("scaled_coordinates", "thermodynamic"):
        result = isotope_substitution_free_energy(samples, [2], estimator=estimator)
        assert result["log_partition_ratio_new_over_old"] == pytest.approx(1.5 * np.log(2))
        assert result["diagnostics"]["reweighting_ess"] == pytest.approx(100)
        assert isotope_fractionation(result, result)["log_alpha_A_over_B"] == 0


def test_poor_overlap_is_reported_and_fractionation_rejects_it_by_default():
    samples = harmonic_samples("primitive", samples=100)
    result = isotope_substitution_free_energy(samples, [2], minimum_effective_samples=1000)
    assert not result["diagnostics"]["overlap_adequate"]
    with pytest.raises(ValueError, match="poor reweighting overlap"):
        isotope_fractionation(result, result)
    assert not isotope_fractionation(result, result, allow_poor_overlap=True)["overlap_adequate"]


def test_mass_thermodynamic_integration_sign_and_units():
    masses = np.geomspace(1, 2, 11)
    beta, difference = 3.0, 0.2
    result = fractionation_mass_integration(masses, np.full(11, 0.7), np.full(11, 0.5), 1 / (KB_AU * beta))
    assert result["log_alpha_A_over_B"] == pytest.approx(beta * difference * np.log(2))


@pytest.mark.parametrize("method", ["primitive", "takahashi_imada"])
def test_physical_kinetic_energy_is_mass_derivative_not_fictitious_momenta(method):
    samples = harmonic_samples(method, samples=30000)
    result = quantum_kinetic_energy(samples)
    eps = 1e-4
    # d log Z / d log m = beta*K; omega scales as m^(-1/2).
    plus = harmonic_log_partition(2, [0.8 * np.exp(-eps / 2)] * 3, beads=8, method=method)
    minus = harmonic_log_partition(2, [0.8 * np.exp(eps / 2)] * 3, beads=8, method=method)
    expected = (plus - minus) / (4 * eps)
    assert result["kinetic_energy_per_atom_hartree"][0] == pytest.approx(expected, abs=5 * result["standard_error_per_atom_hartree"][0])
    if method == "takahashi_imada":
        with pytest.raises(ValueError, match="not valid for TI"):
            quantum_kinetic_energy(samples, estimator="centroid_virial")


def test_momentum_fourier_normalization_matches_analytic_gaussian_and_keeps_grid_tails():
    # Stratified exact Gaussian quantiles remove stochastic reference noise.
    n, variance, bandwidth = 20000, 1.7, 0.05
    displacement = np.sqrt(variance) * ndtri((np.arange(n) + 0.5) / n)
    positions = np.zeros((n, 3, 1, 3))
    positions[:, -1, 0, 0] = displacement
    action = OpenPathAction(HarmonicPotential(), [1], 300)
    samples = OpenPathSamples(positions, displacement, action, {}, 0, 0, 1)
    grid = np.linspace(-4, 4, 161)
    result = momentum_distribution_from_open_paths(samples, grid, bandwidth_bohr=bandwidth, block_size=100)
    broadened = variance + bandwidth**2
    expected = np.sqrt(broadened / (2 * np.pi)) * np.exp(-0.5 * broadened * grid**2)
    np.testing.assert_allclose(result["density_per_atomic_momentum"], expected, atol=8e-5)
    assert result["integral_on_supplied_momentum_grid"] == pytest.approx(1, abs=2e-4)
    short = momentum_distribution_from_open_paths(samples, np.linspace(-0.3, 0.3, 21), bandwidth_bohr=bandwidth, block_size=100)
    assert short["integral_on_supplied_momentum_grid"] < 0.4
    assert result["physical_converged"] is False


def test_closed_positions_and_fictitious_momenta_are_rejected_as_momentum_input():
    with pytest.raises(TypeError, match="OpenPathSamples"):
        momentum_distribution_from_open_paths(np.ones((30, 4, 1, 3)), [-1, 0, 1])
    with pytest.raises(TypeError, match="closed-path"):
        energy_heat_capacity(np.ones(100))


def test_low_sample_heat_uncertainty_is_explicitly_unavailable():
    result = energy_heat_capacity(harmonic_samples("primitive", samples=3), block_size=1)
    assert result["heat_capacity_standard_error_over_kB"] is None
    assert result["diagnostics"]["reason"] == "fewer_than_four_complete_blocks"
