"""Matrix-valued flux-side uncertainty and convergence diagnostics."""

import numpy as np

from pyrpmd.analysis.flux_side_correction import flux_side_rate


def _matrix_observables(replicas=5, points=6):
    time = np.linspace(0.0, 1.0, points)
    seed = np.array([[1.0, 0.15], [0.15, 2.0]])
    flux = np.empty((replicas, points, 2, 2), dtype=float)
    for replica in range(replicas):
        flux[replica] = (1.0 + 0.08 * replica) * seed
    side = np.full_like(flux, 0.75)
    population = np.linspace(1.0, 1.4, replicas)
    return time, flux, side, population


def test_matrix_flux_side_preserves_shapes_and_reports_bootstrap_ci():
    time, flux, side, population = _matrix_observables()
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = flux_side_rate(
        time,
        flux,
        side,
        population,
        sample_weight=weights,
        bootstrap=192,
        confidence_level=0.9,
        rng=np.random.default_rng(12),
    )

    expected = np.average(flux * side, axis=0, weights=weights)
    expected /= np.average(population, weights=weights)
    np.testing.assert_allclose(result.rate, expected)
    assert result.rate.shape == (len(time), 2, 2)
    assert result.correlation.shape == result.rate.shape
    assert result.standard_error.shape == result.rate.shape
    assert result.bootstrap_rate.shape == (192, len(time), 2, 2)
    assert result.confidence_interval.shape == (2, len(time), 2, 2)
    assert np.all(result.confidence_interval[0] <= result.confidence_interval[1])
    assert result.diagnostics["confidence_interval_available"] is True
    assert result.diagnostics["confidence_level"] == 0.9
    assert result.diagnostics["bootstrap_valid"] == 192
    expected_neff = weights.sum() ** 2 / np.sum(weights**2)
    assert result.diagnostics["effective_sample_size"] == expected_neff
    bootstrap_neff = result.diagnostics["bootstrap_effective_sample_size"]
    assert bootstrap_neff.shape == (192,)
    assert np.all(bootstrap_neff >= 1.0)
    assert np.all(bootstrap_neff <= len(weights))
    assert result.physical_converged is False
    assert result.diagnostics["physical_converged"] is False

    # A scalar side indicator is broadcast over matrix elements, which is the
    # common flux-side convention for a matrix-valued response operator.
    scalar_side = np.full(flux.shape[:2], 0.75)
    scalar_side_result = flux_side_rate(
        time,
        flux,
        scalar_side,
        population,
        sample_weight=weights,
        bootstrap=0,
    )
    np.testing.assert_allclose(scalar_side_result.rate, result.rate)


def test_matrix_bootstrap_is_seed_reproducible_and_whole_replica_valued():
    time, flux, side, population = _matrix_observables(replicas=4, points=4)
    kwargs = dict(
        time=time,
        flux=flux,
        side=side,
        reactant_population=population,
        bootstrap=128,
    )
    first = flux_side_rate(rng=np.random.default_rng(31), **kwargs)
    second = flux_side_rate(rng=np.random.default_rng(31), **kwargs)
    np.testing.assert_array_equal(first.bootstrap_rate, second.bootstrap_rate)
    for draw in first.bootstrap_rate:
        assert np.allclose(draw, draw[0][None, ...])


def test_matrix_plateau_gate_uses_matrix_norm_and_requires_bootstrap():
    time, flux, side, population = _matrix_observables(replicas=4, points=6)
    flat = flux_side_rate(
        time,
        flux,
        side,
        population,
        bootstrap=64,
        rng=np.random.default_rng(7),
        plateau_start=0.2,
        plateau_stop=0.8,
        relative_tolerance=1.0e-12,
    )
    assert flat.diagnostics["plateau_detected"] is True
    assert flat.diagnostics["plateau_diagnostics"]["shape_stable"] is True
    assert np.asarray(flat.diagnostics["candidate_rate"]).shape == (2, 2)
    assert flat.diagnostics["plateau_diagnostics"]["maximum_relative_excursion"] == 0.0

    sloped_flux = flux * (1.0 + 0.25 * time[None, :, None, None])
    sloped = flux_side_rate(
        time,
        sloped_flux,
        side,
        population,
        bootstrap=64,
        rng=np.random.default_rng(7),
        plateau_start=0.2,
        plateau_stop=0.8,
        relative_tolerance=0.05,
    )
    assert sloped.diagnostics["plateau_detected"] is False
    assert sloped.diagnostics["plateau_diagnostics"]["shape_stable"] is False
    assert sloped.diagnostics["plateau_diagnostics"]["maximum_relative_excursion"] > 0.05

    no_bootstrap = flux_side_rate(
        time,
        flux,
        side,
        population,
        bootstrap=0,
        plateau_start=0.2,
        plateau_stop=0.8,
    )
    assert no_bootstrap.diagnostics["plateau_diagnostics"]["shape_stable"] is True
    assert no_bootstrap.diagnostics["plateau_detected"] is False
    assert no_bootstrap.confidence_interval is None


def test_matrix_inputs_reject_nonsquare_and_invalid_confidence_level():
    time = np.linspace(0.0, 1.0, 3)
    nonsquare = np.zeros((2, len(time), 2, 3))
    try:
        flux_side_rate(time, nonsquare, nonsquare, 1.0)
    except ValueError as exc:
        assert "square" in str(exc)
    else:
        raise AssertionError("non-square matrix flux must be rejected")

    square = np.zeros((2, len(time), 2, 2))
    with np.testing.assert_raises_regex(ValueError, "confidence_level"):
        flux_side_rate(time, square, square, 1.0, confidence_level=1.0)


def test_zero_weight_replicas_are_excluded_from_bootstrap_support():
    time, flux, side, population = _matrix_observables(replicas=4, points=4)
    weights = np.array([1.0, 0.0, 3.0, 0.0])
    result = flux_side_rate(
        time,
        flux,
        side,
        population,
        sample_weight=weights,
        bootstrap=32,
        rng=np.random.default_rng(4),
    )
    assert result.diagnostics["bootstrap_support_size"] == 2
    assert np.all(result.diagnostics["bootstrap_effective_sample_size"] >= 1.0)
    assert np.isfinite(result.rate).all()
