"""Regression tests for flux--side uncertainty and correction diagnostics."""

import numpy as np
import pytest

from pyrpmd.analysis.flux_side_correction import (
    FluxSideResult,
    flux_side_rate,
    hybrid_flux_side_correction,
)


def test_bootstrap_resamples_complete_replicas_and_is_seed_reproducible():
    """A bootstrap draw must retain each replica's complete time curve."""
    time = np.linspace(0.0, 1.0, 5)
    # The replicas have different *time-dependent* responses.  A time-frame
    # bootstrap would create curves that are not equal to either source curve.
    flux = np.array([[1.0, 2.0, 4.0, 8.0, 16.0],
                     [3.0, 5.0, 7.0, 11.0, 13.0]])
    side = np.ones_like(flux)
    kwargs = dict(
        time=time,
        flux=flux,
        side=side,
        reactant_population=1.0,
        bootstrap=128,
    )
    first = flux_side_rate(rng=np.random.default_rng(17), **kwargs)
    second = flux_side_rate(rng=np.random.default_rng(17), **kwargs)
    np.testing.assert_array_equal(first.bootstrap_rate, second.bootstrap_rate)
    source_curves = [flux[0], flux[1], flux.mean(axis=0)]
    for draw in first.bootstrap_rate:
        assert any(np.array_equal(draw, source) for source in source_curves)


def test_plateau_diagnostics_distinguish_flat_and_sloped_curves():
    time = np.linspace(0.0, 1.0, 6)
    flat = flux_side_rate(
        time,
        flux=np.ones((3, len(time))),
        side=np.ones((3, len(time))),
        reactant_population=1.0,
        bootstrap=16,
        rng=np.random.default_rng(3),
        plateau_start=0.2,
        plateau_stop=0.8,
    )
    assert flat.diagnostics["plateau_detected"]
    assert flat.diagnostics["plateau_diagnostics"]["shape_stable"]

    # A steadily changing response should not be labelled a plateau even when
    # the bootstrap count is large enough to report uncertainty.
    sloped = flux_side_rate(
        time,
        flux=np.tile(time, (3, 1)),
        side=np.ones((3, len(time))),
        reactant_population=1.0,
        bootstrap=16,
        rng=np.random.default_rng(3),
        plateau_start=0.2,
        plateau_stop=0.8,
        relative_tolerance=0.1,
    )
    assert not sloped.diagnostics["plateau_detected"]
    assert not sloped.diagnostics["plateau_diagnostics"]["shape_stable"]
    assert sloped.diagnostics["candidate_rate"] == pytest.approx(0.5)


def test_plateau_detection_requires_replicate_uncertainty_estimate():
    time = np.linspace(0.0, 1.0, 4)
    result = flux_side_rate(
        time,
        flux=np.ones((2, len(time))),
        side=np.ones((2, len(time))),
        reactant_population=1.0,
        bootstrap=0,
        plateau_start=0.0,
        plateau_stop=1.0,
    )
    assert result.standard_error is None
    assert not result.diagnostics["plateau_detected"]
    assert result.diagnostics["plateau_diagnostics"]["shape_stable"]


def test_matrix_inverse_defect_preserves_quantum_curve_when_full_equals_subspace():
    time = np.linspace(0.0, 1.0, 3)
    full = np.array([
        [[2.0, 0.0], [0.0, 1.0]],
        [[2.5, 0.0], [0.0, 1.5]],
        [[3.0, 0.0], [0.0, 2.0]],
    ])
    quantum = np.array([
        [[1.5, 0.0], [0.0, 0.75]],
        [[1.75, 0.0], [0.0, 1.0]],
        [[2.25, 0.0], [0.0, 1.25]],
    ])
    result = hybrid_flux_side_correction(
        time, full, full, quantum, mode="inverse_defect", ridge=1.0e-8
    )
    np.testing.assert_allclose(result["rate"], quantum, rtol=1e-7, atol=1e-8)
    assert result["diagnostics"]["exact_general_rate_claim"] is False


def test_matrix_rate_and_result_validation_reject_non_square_inputs():
    time = np.linspace(0.0, 1.0, 3)
    nonsquare = np.zeros((len(time), 2, 3))
    with pytest.raises(ValueError, match="must be square"):
        hybrid_flux_side_correction(time, nonsquare, nonsquare, nonsquare)

    with pytest.raises(ValueError, match="bootstrap_rate matrix samples must be square"):
        FluxSideResult(
            time=time,
            correlation=np.zeros((len(time), 2, 2)),
            rate=np.zeros((len(time), 2, 2)),
            standard_error=None,
            bootstrap_rate=np.zeros((2, len(time), 2, 3)),
            diagnostics={},
        )


def test_flux_side_input_tolerance_and_weight_validation():
    time = np.linspace(0.0, 1.0, 3)
    flux = np.ones((2, len(time)))
    side = np.ones_like(flux)
    with pytest.raises(ValueError, match="relative_tolerance"):
        flux_side_rate(time, flux, side, 1.0, relative_tolerance=0.0)
    with pytest.raises(ValueError, match="sample_weight"):
        flux_side_rate(time, flux, side, 1.0, sample_weight=[0.0, 0.0])
