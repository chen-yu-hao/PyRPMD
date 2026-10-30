"""Independent 1D PMF oracles: quadrature, coordinate measures and UI moments.

The inverse-CDF streams below integrate known Boltzmann densities. They test
reweighting without depending on the MD sampler. Gaussian UI uses deliberately
exact empirical moments to separate its formula from finite-sample error.
"""

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid, quad
from scipy.special import ndtri
from scipy.signal import lfilter

from pyrpmd.analysis.fes import (
    _block_resample, _bootstrap_block_plan, mbar_rate_pmf, reconstruct_fes,
    statistical_inefficiency, umbrella_integration,
)
from pyrpmd.bias import FluxMetricBias


def _normal_with_exact_moments(n):
    values = ndtri((np.arange(n) + 0.5) / n)
    return (values - values.mean()) / values.std()


def _quantile_streams(grid, log_density, centers, stiffness, beta, counts):
    """Deterministic midpoint quadrature for each biased distribution."""
    streams = []
    for i, (center, kappa, count) in enumerate(zip(centers, stiffness, counts)):
        log_biased = log_density - 0.5 * beta * kappa * (grid - center) ** 2
        cdf = cumulative_trapezoid(np.exp(log_biased - log_biased.max()), grid, initial=0)
        cdf /= cdf[-1]
        stream = np.interp((np.arange(count) + 0.5) / count, cdf, grid)
        # Order is immaterial to quadrature, but sorted data would intentionally
        # appear highly correlated to the time-series diagnostic.
        np.random.default_rng(384 + i).shuffle(stream)
        streams.append(stream)
    return streams


@pytest.mark.parametrize("beta", [0.3, 1.7, 7.0])
def test_ui_recovers_exact_gaussian_free_energy_and_gradient(beta):
    curvature, minimum = 1.8, 0.35
    centers = np.array([-1.5, -0.4, 0.2, 1.0, 1.8])
    kappa = np.array([0.9, 2.0, 4.0, 8.0, 13.0])
    counts = [19, 101, 73, 201, 51]
    streams = [
        (curvature * minimum + k * c) / (curvature + k)
        + _normal_with_exact_moments(n) / np.sqrt(beta * (curvature + k))
        for c, k, n in zip(centers, kappa, counts)
    ]
    grid = np.linspace(-1.2, 1.7, 351)
    free, derivative = umbrella_integration(streams, centers, kappa, beta, grid)
    exact = 0.5 * curvature * (grid - minimum) ** 2
    exact -= exact.min()
    np.testing.assert_allclose(derivative, curvature * (grid - minimum), atol=2e-13)
    np.testing.assert_allclose(free, exact, atol=2e-13)


def test_ui_is_invariant_under_linear_cv_unit_change():
    centers = np.array([-0.6, 0.4, 1.1])
    stiffness = np.array([2.0, 4.0, 8.0])
    beta = 1.6
    streams = [
        k * c / (1 + k) + _normal_with_exact_moments(101) / np.sqrt(beta * (1 + k))
        for c, k in zip(centers, stiffness)
    ]
    grid = np.linspace(-1, 1.5, 101)
    free, derivative = umbrella_integration(streams, centers, stiffness, beta, grid)
    scale = 1e-12
    changed, gradient = umbrella_integration(
        [scale * a for a in streams], scale * centers,
        stiffness / scale**2, beta, scale * grid,
    )
    np.testing.assert_allclose(changed, free, atol=3e-14)
    np.testing.assert_allclose(gradient * scale, derivative, atol=3e-14)


def test_ui_common_scalar_stiffness_matches_per_stream_stiffness():
    streams = [np.linspace(-1, 1, 101), np.linspace(-0.5, 1.5, 71)]
    grid = np.linspace(-1, 1, 11)
    scalar = umbrella_integration(streams, [0, 0.5], 2.0, 1.0, grid)
    vector = umbrella_integration(streams, [0, 0.5], [2.0, 2.0], 1.0, grid)
    for actual, expected in zip(scalar, vector):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("replacement", [
    {"samples": []},
    {"samples": [[]]},
    {"samples": [[0.0]]},
    {"samples": [[0.0, np.nan]]},
    {"samples": [[0.0, np.inf]]},
    {"samples": [np.ones((3, 2))]},
    {"samples": [np.ones(3)]},
    {"centers": [np.nan]},
    {"centers": [[0.0, 1.0]]},
    {"kappa": 0.0},
    {"kappa": -1.0},
    {"kappa": np.nan},
    {"kappa": [1.0, 2.0]},
    {"beta": 0.0},
    {"beta": -1.0},
    {"beta": np.nan},
    {"beta": np.inf},
    {"grid": [0.0]},
    {"grid": [0.0, np.nan]},
    {"grid": [1.0, 0.0]},
    {"grid": [[0.0, 1.0]]},
])
def test_ui_rejects_invalid_scientific_inputs(replacement):
    options = dict(samples=[[-1.0, 0.0, 1.0]], centers=[0.0], kappa=[1.0],
                   beta=1.0, grid=np.array([-1.0, 0.0, 1.0]))
    options.update(replacement)
    with pytest.raises(ValueError):
        umbrella_integration(**options)


def quartic_quadrature_comparison():
    """Return bin-averaged oracle errors for report generation and the test."""
    beta = 1.7
    centers = np.array([-0.25, 0.1, 0.45, 0.8, 1.15, 1.55, 1.9, 2.25])
    kappa = np.array([5, 12, 7, 15, 9, 18, 6, 10], dtype=float)
    counts = [6000, 7000, 8000, 9000, 7500, 6500, 8500, 9500]
    grid = np.linspace(-1.5, 3.5, 100001)

    def physical(x):
        return 1.2 * x**2 * (x - 2) ** 2 + 0.1 * x

    streams = _quantile_streams(grid, -beta * physical(grid), centers, kappa, beta, counts)
    edges = np.array([-0.8, -0.45, -0.2, -0.05, 0.05, 0.25, 0.6, 0.8,
                      0.9, 1.1, 1.25, 1.5, 1.8, 1.95, 2.05, 2.3, 2.7])
    estimate = mbar_rate_pmf(streams, centers, kappa, beta, bins=edges,
                             xi_star=1.0, reference=0.0, bootstrap=0, min_count=10)
    partition = quad(lambda x: np.exp(-beta * physical(x)), -np.inf, np.inf,
                     epsabs=1e-12, epsrel=1e-12)[0]
    density = np.array([
        quad(lambda x: np.exp(-beta * physical(x)), lo, hi,
             epsabs=1e-12, epsrel=1e-12)[0] / ((hi - lo) * partition)
        for lo, hi in zip(edges[:-1], edges[1:])
    ])
    exact = -np.log(density) / beta
    centers_bin = (edges[:-1] + edges[1:]) / 2
    reference = int(np.flatnonzero(np.isclose(centers_bin, 0.0))[0])
    surface = int(np.flatnonzero(np.isclose(centers_bin, 1.0))[0])
    exact -= exact[reference]
    delta = estimate["free_energy"] - exact
    return estimate, {
        "samples": sum(counts),
        "beta": beta,
        "maximum_free_energy_error_kbt": float(beta * np.max(np.abs(delta))),
        "rms_free_energy_error_kbt": float(beta * np.sqrt(np.mean(delta**2))),
        "barrier_hartree": estimate["free_energy_barrier_hartree"],
        "bin_averaged_oracle_barrier_hartree": float(exact[surface]),
        "point_potential_barrier_hartree": float(physical(1.0) - physical(0.0)),
        "solver_residual": estimate["solver"]["residual"],
    }


def test_mbar_unequal_windows_against_quartic_partition_quadrature():
    estimate, diagnostics = quartic_quadrature_comparison()
    assert estimate["ready"]
    assert not estimate["mask"].any()
    assert diagnostics["maximum_free_energy_error_kbt"] < 0.015
    assert diagnostics["rms_free_energy_error_kbt"] < 0.004
    assert abs(diagnostics["barrier_hartree"]
               - diagnostics["bin_averaged_oracle_barrier_hartree"]) < 0.003


def test_generic_radial_density_retains_the_three_dimensional_jacobian():
    beta, curvature = 1.8, 0.7
    grid = np.linspace(1e-6, 10, 100001)
    centers = np.array([0.3, 0.8, 1.4, 2.0, 2.6])
    stiffness = np.array([3.0, 4.0, 2.0, 5.0, 2.5])
    log_density = 2 * np.log(grid) - beta * 0.5 * curvature * grid**2
    streams = _quantile_streams(grid, log_density, centers, stiffness, beta,
                                [8000] * len(centers))
    edges = np.linspace(0.2, 3.1, 30)
    result = reconstruct_fes([a[:, None] for a in streams], centers[:, None],
                             stiffness[:, None, None], beta, bins=[edges], bootstrap=0,
                             min_count=10)
    # Integrate r^2 dr explicitly. A default -2*kBT*log(r) "correction" would
    # change the requested density measure and fail this oracle.
    mass = np.array([
        quad(lambda r: r**2 * np.exp(-beta * 0.5 * curvature * r*r), lo, hi,
             epsabs=1e-12)[0]
        for lo, hi in zip(edges[:-1], edges[1:])
    ])
    exact = -np.log(mass / np.diff(edges)) / beta
    valid = ~result["mask"]
    difference = result["free_energy"][valid] - exact[valid]
    difference -= np.mean(difference)
    assert result["connected"]
    assert np.max(np.abs(beta * difference)) < 0.02


class _SquaredCentroidCoordinate:
    estimator = "centroid"

    def centroid_value_gradient_hessian(self, q):
        x = q[0, 0]
        gradient = np.zeros_like(q)
        gradient[0, 0] = 2 * x
        hessian = np.zeros(q.shape + q.shape)
        hessian[0, 0, 0, 0] = 2
        return x*x, gradient, hessian


@pytest.mark.parametrize("beads", [1, 3, 16])
def test_nonlinear_flux_measure_has_exact_metric_and_bead_scaling(beads):
    beta, mass = 2.3, 4.1
    q = np.zeros((beads, 1, 3))
    q[:, 0, 0] = 1.4 + np.linspace(-0.2, 0.2, beads)
    centroid = q[:, 0, 0].mean()
    bias = FluxMetricBias(_SquaredCentroidCoordinate(), [mass], beta)
    energy, force = bias.energy_force(q)
    expected_flux = 2 * centroid / np.sqrt(2 * np.pi * beta * mass)
    assert energy == pytest.approx(-beads * np.log(expected_flux) / beta, abs=1e-14)
    np.testing.assert_allclose(force[:, 0, 0], 1 / (beta * centroid), atol=1e-14)
    np.testing.assert_array_equal(force[:, 0, 1:], 0)
    # For s=x^2 on x>0, rho_s ds = rho_x dx. Multiplying the sampled
    # density by f_s cancels 1/(2*x), leaving a coordinate-invariant flux.
    density_jacobian = 1 / (2 * centroid)
    actual_measure = density_jacobian * np.exp(-beta * energy / beads)
    assert actual_measure == pytest.approx(1 / np.sqrt(2 * np.pi * beta * mass))


def ar1_bootstrap_comparison(draws=500):
    """PMF-error oracle from the known covariance of a stationary Gaussian AR(1)."""
    n, rho, variance = 50000, 0.95, 0.5
    rng = np.random.default_rng(29031)
    stream = lfilter([np.sqrt(variance * (1 - rho*rho))], [1, -rho],
                     rng.normal(size=n + 2000))[2000:]
    g = statistical_inefficiency(stream)
    lags = np.arange(1, n)
    exact_g = 1 + 2 * np.sum((1 - lags / n) * rho**lags)
    # For a zero-mean Gaussian, UI gives F(1)-F(-1)=-2*mean/variance
    # at beta=1. The delta-method variance uses the exact finite-N AR mean
    # variance. Mean/variance cross-covariance vanishes by Gaussian symmetry.
    analytic = np.sqrt(4 * exact_g / (variance * n))
    plan = _bootstrap_block_plan(np.array([n]), np.array([g]))
    records = {}
    for name, block in [("legacy", int(np.ceil(g))),
                        ("revised", int(plan["block_lengths"][0]))]:
        rngb = np.random.default_rng(639)
        values = []
        for _ in range(draws):
            sample = _block_resample(stream, block, rngb)
            curve, _ = umbrella_integration([sample], [0.0], 1.0, 1.0,
                                           np.array([-1.0, 0.0, 1.0]))
            values.append(curve[-1] - curve[0])
        error = float(np.std(values, ddof=1))
        records[name] = {"block_length": block, "bootstrap_standard_error": error,
                         "ratio_to_analytic": error / analytic}
    return {
        "rho": rho, "samples": n, "draws": draws, "seed": 29031,
        "bootstrap_seed": 639, "estimated_g": g, "finite_N_analytic_g": float(exact_g),
        "analytic_standard_error": float(analytic), "block_plan": plan,
        **records,
    }


def test_correlated_pmf_bootstrap_against_ar1_covariance_oracle():
    report = ar1_bootstrap_comparison()
    assert report["legacy"]["ratio_to_analytic"] < 0.85
    assert abs(report["revised"]["ratio_to_analytic"] - 1) < 0.15
    assert report["block_plan"]["status"] == "finite_block_estimate"


def test_short_stream_cap_does_not_resample_one_whole_circular_block():
    plan = _bootstrap_block_plan(np.array([100]), np.array([80.0]))
    np.testing.assert_array_equal(plan["requested_block_lengths"], [400])
    np.testing.assert_array_equal(plan["block_lengths"], [50])
    assert plan["block_length_capped"][0]
    assert plan["status"] == "limited_blocks"
    assert plan["warnings"]
    # A whole-length circular block only rotates x and gives exactly the
    # same histogram/moments. Half-length blocks do not force this artifact.
    rng = np.random.default_rng(773)
    means = [_block_resample(np.arange(100.0), 50, rng).mean() for _ in range(20)]
    assert np.std(means) > 1


def test_unresolved_constant_stream_has_unavailable_uncertainty():
    plan = _bootstrap_block_plan(np.array([100]), np.array([100.0]))
    assert plan["status"] == "unavailable"
    assert plan["warnings"]


def test_three_point_pmf_does_not_report_a_resolved_error_bar():
    result = mbar_rate_pmf([[0.5, 1.5, 2.5]], [1.5], 1e-12, 1.0,
                           bins=[0, 1, 2, 3], reference=0.5, xi_star=2.5,
                           bootstrap=20, min_count=1, seed=55)
    assert result["ready"]
    assert result["bootstrap"]["block_plan"]["status"] == "unavailable"
    assert result["barrier_standard_error_hartree"] is None
    assert result["barrier_percentile_95_ci_hartree"] is None
    assert np.isnan(result["standard_error"]).all()
