"""Surface basin normalization against analytical distributions and quadrature."""

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid, quad

from pyrpmd.analysis.surface import basin_qtst, surface_rate_pmf
from pyrpmd.constants import ATOMIC_TIME_S


def test_flat_basin_first_order_units_and_free_energy_gauge():
    grid = np.linspace(-2, 2, 17)
    beta, metric = 3.2, 0.07
    basin, dividing_surface = (-1.3, -0.2), 0.6
    expected = np.sqrt(metric / (2 * np.pi * beta)) / 1.1
    for offset in (0.0, -1e6, 1e6):
        actual = basin_qtst(
            grid, np.full_like(grid, offset), beta, metric, basin, dividing_surface
        )
        assert actual == pytest.approx(expected, rel=3e-15)


def test_harmonic_basin_matches_continuum_quadrature_and_affine_cv_change():
    grid = np.linspace(-2, 2, 8001)
    beta, metric, spring = 2.4, 0.3, 1.7
    free = 0.5 * spring * grid**2
    basin, dividing_surface = (-1.25, 0.35), 0.9
    integral = quad(lambda x: np.exp(-beta * spring * x * x / 2), *basin)[0]
    expected = (
        np.sqrt(metric / (2 * np.pi * beta))
        * np.exp(-beta * spring * dividing_surface**2 / 2)
        / integral
    )
    actual = basin_qtst(grid, free, beta, metric, basin, dividing_surface)
    assert actual == pytest.approx(expected, rel=2e-7)
    scale, offset = 3.7, 8.2
    transformed = basin_qtst(
        scale * grid + offset,
        free + np.log(scale) / beta,
        beta,
        metric * scale**2,
        tuple(scale * np.asarray(basin) + offset),
        scale * dividing_surface + offset,
    )
    assert transformed == pytest.approx(actual, rel=1e-13)
    # Product may also lie to the left; reversing the coordinate preserves k.
    reflected = basin_qtst(
        -grid[::-1], free[::-1], beta, metric, (-basin[1], -basin[0]), -dividing_surface
    )
    assert reflected == pytest.approx(actual, rel=1e-13)


def test_large_barrier_is_integrated_without_overflow_or_gauge_dependence():
    grid = np.array([-1.0, 0.0, 1.0])
    free = np.array([0.0, 0.0, 600.0])
    expected = np.exp(-600) / np.sqrt(2 * np.pi)
    with np.errstate(over="raise", invalid="raise"):
        for offset in (0.0, 1e6, -1e6):
            rate = basin_qtst(grid, free + offset, 1.0, 1.0, (-1.0, 0.0), 1.0)
            assert rate > 0
            assert rate == pytest.approx(expected, rel=1e-12, abs=0)


def test_linear_pmf_integral_matches_analytic_exponential_on_uneven_grid():
    grid = np.array([-2.0, -0.7, 0.0, 0.13, 0.8, 1.6, 2.0])
    beta, metric, slope = 1.4, 0.2, 4.2
    basin, dividing_surface = (-0.4, 0.9), 1.5
    integral = (
        np.exp(-beta * slope * basin[0]) - np.exp(-beta * slope * basin[1])
    ) / (beta * slope)
    expected = np.sqrt(metric / (2 * np.pi * beta)) * np.exp(
        -beta * slope * dividing_surface
    ) / integral
    assert basin_qtst(grid, slope * grid, beta, metric, basin, dividing_surface) == pytest.approx(
        expected, rel=5e-15
    )


def test_missing_support_and_invalid_basins_are_rejected():
    grid = np.arange(6.0)
    free = np.zeros(6)
    for mask in ([False, False, True, False, False, False], None):
        damaged = free.copy()
        if mask is None:
            damaged[2] = np.nan
        with pytest.raises(ValueError, match="masked"):
            basin_qtst(grid, damaged, 1.0, 1.0, (0.0, 1.0), 4.0, mask)
    with pytest.raises(ValueError, match="outside_histogram_centers"):
        basin_qtst(grid, free, 1.0, 1.0, (-0.1, 1.0), 4.0)
    with pytest.raises(ValueError, match="one side"):
        basin_qtst(grid, free, 1.0, 1.0, (0.0, 4.0), 2.0)
    with pytest.raises(ValueError, match="mass_metric"):
        basin_qtst(grid, free, 1.0, 0.0, (0.0, 1.0), 4.0)
    # An unrelated unsampled part of the global FES is harmless.
    free[-1] = np.nan
    assert np.isfinite(basin_qtst(grid, free, 1.0, 1.0, (0.0, 1.0), 4.0))


def test_nonuniform_histogram_uses_density_and_preserves_paired_bootstrap():
    n = 12000
    sample = ((np.arange(n) + 0.5) / n)[:, None]
    np.random.default_rng(81).shuffle(sample)
    beta, metric = 2.0, 0.04
    result = surface_rate_pmf(
        [sample], [[0.5]], 1e-12, beta,
        basin=(0.05, 0.45), dividing_surface=0.8, mass_metric=metric,
        bins=[0.0, 0.1, 0.3, 0.6, 1.0], bootstrap=60, seed=73,
    )
    assert result["ready"]
    assert result["measure"] == "cv_density"
    assert result["method"] == "mbar_surface_density"
    np.testing.assert_allclose(result["density"], 1.0, atol=1e-11)
    expected = np.sqrt(metric / (2 * np.pi * beta)) / 0.4
    assert result["k_qtst_au"] == pytest.approx(expected, rel=1e-11)
    assert result["k_qtst_s_inv"] == pytest.approx(expected / ATOMIC_TIME_S, rel=1e-11)
    assert result["bootstrap"]["valid"] == 60
    assert result["qtst_standard_error_s_inv"] > 0
    paired = result["bootstrap"]["log_density_integral_pairs"]
    log_rates = np.log(np.asarray(result["pmf_bootstrap_qtst"]) * ATOMIC_TIME_S)
    np.testing.assert_allclose(
        log_rates,
        0.5 * np.log(metric / (2 * np.pi * beta)) + paired[:, 0] - paired[:, 1],
        rtol=1e-13, atol=1e-13,
    )
    contrast = np.array([1.0, -1.0])
    variance = contrast @ result["bootstrap"]["log_density_integral_covariance"] @ contrast
    assert variance == pytest.approx(np.var(log_rates, ddof=1), rel=1e-12)


def test_umbrella_mbar_recovers_known_anharmonic_basin_rate():
    beta, stiffness, metric = 1.3, 8.0, 0.2
    centers = np.repeat(np.linspace(-0.2, 2.2, 9), 2)
    grid = np.linspace(-1.0, 3.0, 60001)
    potential = lambda x: 1.2 * x**2 * (x - 2)**2
    physical = potential(grid)
    rng = np.random.default_rng(7189)
    samples = []
    for center in centers:
        density = np.exp(-beta * (physical + 0.5 * stiffness * (grid - center)**2))
        cdf = cumulative_trapezoid(density, grid, initial=0)
        cdf /= cdf[-1]
        samples.append(np.interp(rng.random(1700), cdf, grid)[:, None])
    basin, dividing_surface = (-0.4, 1.0), 1.0
    result = surface_rate_pmf(
        samples, centers[:, None], stiffness, beta,
        basin=basin, dividing_surface=dividing_surface, mass_metric=metric,
        bins=64, ranges=(-0.6, 2.6), bootstrap=12, seed=941, min_count=10,
    )
    expected = np.sqrt(metric / (2 * np.pi * beta)) * np.exp(
        -beta * potential(dividing_surface)
    ) / quad(lambda x: np.exp(-beta * potential(x)), *basin)[0]
    assert result["ready"]
    assert result["k_qtst_au"] == pytest.approx(expected, rel=0.1)
    assert result["bootstrap"]["valid"] >= 10
    assert len(result["bootstrap"]["block_lengths"]) == len(samples)
    assert result["solver"]["details"]["unique_states"] == 9


def test_unsupported_histogram_and_disconnected_overlap_do_not_return_rates():
    sample = np.r_[np.full(100, -1.0), np.full(100, 1.0)][:, None]
    gap = surface_rate_pmf(
        [sample], [[0.0]], 1e-12, 1.0,
        basin=(-1.0, -0.8), dividing_surface=1.0, mass_metric=1.0,
        bins=np.linspace(-1.2, 1.2, 13), bootstrap=4, min_count=1,
    )
    assert not gap["ready"]
    assert "masked" in gap["status"]
    assert gap["k_qtst_s_inv"] is None
    assert gap["qtst_standard_error_s_inv"] is None
    assert gap["bootstrap"]["attempted"] == 0
    separate = surface_rate_pmf(
        [np.full((100, 1), -10.0), np.full((100, 1), 10.0)],
        [[-10.0], [10.0]], 100.0, 1.0,
        basin=(-10.0, -9.0), dividing_surface=10.0, mass_metric=1.0,
        bins=20, ranges=(-11.0, 11.0), bootstrap=0,
    )
    assert separate["status"] == "disconnected_overlap"
    assert separate["mask"].all()
    assert separate["k_qtst_au"] is None


def test_failed_bootstrap_draws_retain_reasons_and_are_not_zero_rates():
    sample = np.r_[np.full(120, 0.5), 1.5, np.full(120, 2.5)][:, None]
    np.random.default_rng(7).shuffle(sample)
    result = surface_rate_pmf(
        [sample], [[1.5]], 1e-12, 1.0,
        basin=(0.5, 1.5), dividing_surface=2.5, mass_metric=1.0,
        bins=[0.0, 1.0, 2.0, 3.0], bootstrap=80, seed=52, min_count=1,
    )
    assert result["ready"]
    info = result["bootstrap"]
    assert info["attempted"] == 80
    assert 2 <= info["valid"] < 80
    assert len(info["failures"]) == 80 - info["valid"]
    assert all(failure["reason"] for failure in info["failures"])
    assert len(result["pmf_bootstrap_qtst"]) == info["valid"]
    assert np.all(np.asarray(result["pmf_bootstrap_qtst"]) > 0)


@pytest.mark.parametrize("option,value", [("bootstrap", 1), ("min_count", 0), ("bins", True)])
def test_invalid_histogram_options_fail_early(option, value):
    options = dict(bootstrap=0, min_count=1, bins=10)
    options[option] = value
    with pytest.raises(ValueError):
        surface_rate_pmf(
            [np.linspace(-1, 1, 100)[:, None]], [[0.0]], 1.0, 1.0,
            basin=(-0.8, 0.0), dividing_surface=0.5, mass_metric=1.0, **options,
        )
