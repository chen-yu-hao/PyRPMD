"""Amplitude correlation in pilot and production FES uncertainty diagnostics."""

import numpy as np
import pytest

from pyrpmd.analysis.fes import (
    cv_statistical_inefficiency, mbar_rate_pmf, reconstruct_fes,
    statistical_inefficiency,
)
from pyrpmd.analysis.surface import surface_rate_pmf


@pytest.fixture(scope="module")
def oscillator_stream():
    # Exact stationary Gaussian dynamics in harmonic phase space: rotate,
    # weakly damp, and add matching thermal noise. Its fast coordinate changes
    # sign before the oscillator's energy/amplitude has decorrelated.
    rng = np.random.default_rng(741)
    n, rho, angle = 12000, .999, 1.0
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    phase = np.empty((n, 2))
    phase[0] = rng.normal(size=2)
    noise = np.sqrt(1 - rho**2) * rng.normal(size=(n - 1, 2))
    for i in range(1, n):
        phase[i] = rho * (rotation @ phase[i - 1]) + noise[i - 1]
    return phase[:, :1]


def test_production_fes_detects_slow_amplitude_and_uses_longer_bootstrap_blocks(oscillator_stream):
    samples = oscillator_stream
    shuffled = np.random.default_rng(821).permutation(samples)
    options = dict(centers=[[0.]], stiffness=[[1e-12]], beta=1.,
                   bins=[[-4., -2., -1., 0., 1., 2., 4.]], bootstrap=4, seed=81)
    ordered = reconstruct_fes([samples], **options)
    independent_order = reconstruct_fes([shuffled], **options)
    linear_g = statistical_inefficiency(samples)
    assert linear_g < 2
    assert ordered["statistical_inefficiency"][0] > 100
    assert ordered["effective_samples_per_stream"][0] < 100
    assert independent_order["effective_samples_per_stream"][0] > 1000
    plan = ordered["bootstrap_block_plan"]
    assert plan["block_lengths"][0] > 500
    assert plan["block_lengths"][0] > 20 * independent_order["bootstrap_block_plan"]["block_lengths"][0]
    assert plan["status"] == "limited_blocks"
    assert ordered["bootstrap_attempted"] == 4
    # Correlation changes uncertainty and ESS, not the density estimator.
    np.testing.assert_allclose(ordered["free_energy"], independent_order["free_energy"], atol=1e-13)
    np.testing.assert_allclose(np.sort(ordered["weights"]), np.sort(independent_order["weights"]), atol=1e-16)


def test_rate_pmf_and_ui_bootstrap_include_amplitude_correlation(oscillator_stream):
    sample = oscillator_stream[:, 0]
    result = mbar_rate_pmf(
        [sample], [0.], 1e-12, 1., bins=[-4., -2., -1., 0., 1., 2., 4.],
        reference=-1.5, xi_star=1.5, compare_ui=True, bootstrap=4, seed=33,
    )
    assert result["ready"] and result["ui_comparison"]["ready"]
    assert result["effective_samples_per_stream"][0] < 100
    assert result["bootstrap"]["block_lengths"][0] > 500
    assert result["window_moments"][0]["estimated_effective_samples"] < 100
    assert result["bootstrap"]["block_plan"]["status"] == "limited_blocks"
    assert result["bootstrap"]["attempted"] == 4
    assert result["ui_comparison"]["paired_ui_bootstrap_barriers"]


def test_surface_pmf_uses_the_same_amplitude_correlation(oscillator_stream):
    result = surface_rate_pmf(
        [oscillator_stream], [[0.]], 1e-12, 1.,
        basin=(-1.5, -.5), dividing_surface=1.5, mass_metric=1.,
        bins=[-4., -2., -1., 0., 1., 2., 4.], bootstrap=4, seed=33,
    )
    assert result["ready"]
    assert result["effective_samples_per_stream"][0] < 100
    assert result["bootstrap"]["block_lengths"][0] > 100
    assert result["bootstrap"]["attempted"] == 4
    assert result["statistical_inefficiency"][0] == cv_statistical_inefficiency(oscillator_stream)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("scale,offset", [(1., 0.), (.1, 0.), (.1, 2.)])
def test_constant_derived_observable_does_not_invalidate_independent_signs(periodic, scale, offset):
    signs = np.repeat([-1., 1.], 2000)
    np.random.default_rng(9).shuffle(signs)
    values = (np.pi * (signs > 0)) if periodic else offset + scale * signs
    g = cv_statistical_inefficiency(values, [2 * np.pi] if periodic else None)
    assert g < 3  # centered square / second harmonic is constant, not slow


@pytest.mark.parametrize("n", [3, 100])
def test_truly_constant_cvs_retain_unresolved_variation_semantics(n):
    expected = statistical_inefficiency(np.zeros(n))
    assert cv_statistical_inefficiency(np.full(n, 3.)) == expected
    assert cv_statistical_inefficiency(np.full(n, 3.), [2 * np.pi]) == expected
    # Changing the branch label does not create physical circular variation.
    assert cv_statistical_inefficiency(2 * np.pi * (np.arange(n) % 2), [2 * np.pi]) == expected


def test_periodic_production_ess_tracks_slow_axis_and_is_branch_invariant():
    n = 4000
    signs = np.random.default_rng(73).choice([-1., 1.], n)
    values = (.7 * np.sin(2 * np.pi * np.arange(n) / 2000) + np.pi * (signs > 0))[:, None]
    options = dict(centers=[[0.]], stiffness=[[1e-12]], beta=1., periods=[2 * np.pi],
                   bins=8, ranges=[[-np.pi, np.pi]], bootstrap=0)
    result = reconstruct_fes([values], **options)
    shifted = reconstruct_fes([values + 2 * np.pi * (np.arange(n) % 3)[:, None]], **options)
    assert result["effective_samples_per_stream"][0] < 100
    np.testing.assert_allclose(result["statistical_inefficiency"], shifted["statistical_inefficiency"], rtol=1e-12)
    np.testing.assert_allclose(result["free_energy"], shifted["free_energy"], atol=1e-12)
