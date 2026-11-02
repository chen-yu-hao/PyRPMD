"""Harmonic references for generated baths; no external matrix fixtures."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.linalg import expm, solve_continuous_lyapunov

from pyrpmd._gle_fitting import fit_optimal_gle, fit_piglet, _potential_tau


def _independent_energies(spec, frequency, *, probe_time=None):
    """Physical one-dimensional K_cv and V from the complete harmonic SDE."""
    beta, beads = spec["beta"], spec["beads"]
    labels = [0]
    for k in range(1, (beads + 1) // 2):
        labels.extend([k, k])
    if beads % 2 == 0:
        labels.append(beads // 2)
    widths = []
    for label, a, c in zip(labels, np.asarray(spec["A"]), np.asarray(spec["C"])):
        spring = 2 * beads / beta * np.sin(np.pi * label / beads)
        drift = np.zeros((len(a) + 1, len(a) + 1))
        drift[0, 1] = -1
        drift[1, 0] = frequency**2 + spring**2
        drift[1:, 1:] = a
        diffusion = np.zeros_like(drift)
        diffusion[1:, 1:] = beads / beta * (a @ c + c @ a.T)
        covariance = solve_continuous_lyapunov(drift, diffusion)
        np.testing.assert_allclose(
            drift @ covariance + covariance @ drift.T, diffusion,
            atol=1e-9, rtol=2e-11)
        widths.append(covariance[0, 0])
        if label and probe_time is not None:
            # Static moments alone admit an almost undamped auxiliary mode.
            # Evolve both exactly stated initialization covariances independently
            # of the fitter's dimensionless transient implementation.
            assert np.linalg.eigvals(drift).real.min() >= .27 / beta
            transition = expm(-probe_time * drift)
            initial = np.zeros_like(drift)
            initial[1:, 1:] = beads / beta * c
            free_difference = transition @ (initial - covariance) @ transition.T
            cold_difference = transition @ covariance @ transition.T
            assert abs(free_difference[0, 0] / covariance[0, 0]) < .005
            assert abs(cold_difference[0, 0] / covariance[0, 0]) < .005
    potential = frequency**2 * np.mean(widths) / 2
    kinetic = 1 / (2 * beta) + frequency**2 * np.sum(widths[1:]) / (2 * beads)
    return potential, kinetic


@pytest.mark.parametrize("beads", [2, 3, 4, 8])
def test_piglet_fit_quantum_harmonic_reference_diffusion_and_relaxation(beads):
    beta = 8.
    spec = fit_piglet([.5 / beta, 12 / beta], beta=beta, beads=beads)
    a, c = np.asarray(spec["A"]), np.asarray(spec["C"])
    assert a.shape == c.shape == (beads, 3, 3)
    for drift, covariance in zip(a, c):
        assert np.linalg.eigvalsh((drift + drift.T) / 2).min() > 0
        assert np.linalg.eigvalsh(covariance).min() > 0
        diffusion = drift @ covariance + covariance @ drift.T
        assert np.linalg.eigvalsh(diffusion).min() > 0
        transition = expm(-.07 * drift)
        assert np.linalg.eigvalsh(covariance - transition @ covariance @ transition.T).min() > 0
    np.testing.assert_array_equal(c[0], np.eye(3))
    # Frequencies deliberately distinct from the log training grid.
    for beta_omega in [.7, 1.7, 5.3, 11.3]:
        frequency = beta_omega / beta
        exact = frequency / (4 * np.tanh(beta_omega / 2))
        potential, kinetic = _independent_energies(spec, frequency, probe_time=100.)
        np.testing.assert_allclose([potential, kinetic], exact, rtol=.004, atol=1e-10)
        assert potential == pytest.approx(kinetic, abs=2e-11)
    metadata = spec["parameter_metadata"]
    assert metadata["fit_converged"] is True
    assert metadata["physical_converged"] is False
    assert metadata["maximum_relative_error"] < .004
    assert metadata["lyapunov_relative_residual"] < 1e-11
    assert metadata["solver"]["success"] is True
    assert len(metadata["validation_frequency_grid_au"]) > len(metadata["frequency_grid_au"])
    assert metadata["maximum_relative_error"] == max(abs(x) for x in metadata["validation_relative_errors"])
    assert metadata["solver"]["numpy_version"] == np.__version__
    assert metadata["implementation"].endswith("relaxation-v2")
    assert metadata["relaxation_controls"]["probe_time_au"] == 100.
    assert metadata["relaxation_controls"]["scope"].startswith("internal harmonic modes only")
    assert metadata["relaxation_diagnostics"]["minimum_decay_beta"] >= .27
    assert metadata["relaxation_diagnostics"]["maximum_free_bath_relative_deviation"] < .005
    assert metadata["relaxation_diagnostics"]["maximum_cold_relative_deviation"] < .005
    json.dumps(spec, allow_nan=False)


def test_piglet_deterministic_generation_and_temperature_scaling():
    args = dict(beta=8., beads=4, naux=1)
    original = fit_piglet([.5 / 8, 12 / 8], **args)
    repeated = fit_piglet([.5 / 8, 12 / 8], **args)
    assert repeated == original
    scaled = fit_piglet([.5 / 16, 12 / 16], beta=16., beads=4, naux=1)
    np.testing.assert_array_equal(np.asarray(original["A"]) / 2, scaled["A"])
    np.testing.assert_array_equal(original["C"], scaled["C"])
    assert original["parameter_metadata"]["maximum_relative_error"] == scaled["parameter_metadata"]["maximum_relative_error"]


@pytest.mark.parametrize("band", [None, [], [1], [1, 2, 3], [[1, 2]],
                                  [True, 2], [False, 1], ["1", 2],
                                  [0, 1], [-1, 1], [2, 1], [1, 1],
                                  [1, np.inf], [np.nan, 1], [1j, 2]])
def test_fit_rejects_invalid_frequency_ranges(band):
    with pytest.raises(ValueError, match="frequency_range_au"):
        fit_piglet(band, beta=8., beads=4)
    with pytest.raises(ValueError, match="frequency_range_au"):
        fit_optimal_gle(band)


@pytest.mark.parametrize("field,value", [
    ("beta", True), ("beta", 0), ("beta", np.nan), ("beta", np.inf),
    ("beads", True), ("beads", 1), ("beads", 4.), ("beads", np.nan),
    ("naux", True), ("naux", 0), ("naux", 4), ("naux", 1.),
    ("grid_size", True), ("grid_size", 7), ("grid_size", 257),
    ("max_nfev", 0), ("max_nfev", True), ("max_nfev", 1.5),
    ("tolerance", True), ("tolerance", 0), ("tolerance", np.nan),
    ("tolerance", 1),
])
def test_piglet_fit_rejects_invalid_controls(field, value):
    args = dict(beta=8., beads=4)
    args[field] = value
    with pytest.raises(ValueError, match=field):
        fit_piglet([.1, 1.5], **args)


def test_piglet_fit_never_silently_accepts_failed_or_inaccurate_fit():
    with pytest.raises(ValueError, match="optimizer_success=False"):
        fit_piglet([.5 / 8, 12 / 8], beta=8., beads=4, max_nfev=1)
    with pytest.raises(ValueError, match="maximum_relative_error"):
        fit_piglet([.5 / 8, 12 / 8], beta=8., beads=4, naux=1, tolerance=1e-8)


def test_scalar_potential_autocorrelation_matches_closed_form():
    frequencies = np.geomspace(.1, 10, 51)
    for gamma in [.2, 1., 5.]:
        exact = 1 / (2 * gamma) + gamma / (2 * frequencies**2)
        np.testing.assert_allclose(_potential_tau(np.array([[gamma]]), frequencies), exact,
                                   rtol=2e-14, atol=1e-12)


def test_optimal_gle_canonical_equilibrium_and_sampling_efficiency():
    spec = fit_optimal_gle([.1, 10.], naux=2)
    a, c = np.asarray(spec["A"]), np.asarray(spec["C"])
    np.testing.assert_array_equal(c, np.eye(3))
    assert np.linalg.eigvalsh(a + a.T).min() > 0
    for omega in [.13, .7, 2.3, 8.9]:
        drift = np.zeros((4, 4))
        drift[0, 1] = -1
        drift[1, 0] = omega**2
        drift[1:, 1:] = a
        diffusion = np.zeros_like(drift)
        diffusion[1:, 1:] = a + a.T
        stationary = solve_continuous_lyapunov(drift, diffusion)
        np.testing.assert_allclose(stationary, np.diag([1 / omega**2, 1, 1, 1]),
                                   rtol=1e-9, atol=1e-10)
    metadata = spec["parameter_metadata"]
    # Scalar minimax gamma=1 gives kappa_min=2/(10+0.1) on this band.
    assert metadata["minimum_efficiency"] > 2 / (10 + .1)
    assert metadata["minimum_efficiency"] > .4
    assert metadata["fit_converged"] is True
    assert metadata["physical_converged"] is False
    assert metadata["solver"]["success"] is True
    assert len(metadata["validation_tau_V_au"]) == len(metadata["validation_frequency_grid_au"])
    json.dumps(spec, allow_nan=False)


def test_optimal_gle_fit_failure_is_explicit():
    with pytest.raises(ValueError, match="did not converge"):
        fit_optimal_gle([.1, 10.], max_nfev=1)
