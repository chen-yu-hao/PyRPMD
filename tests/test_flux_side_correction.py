import numpy as np
import pytest

from pyrpmd.analysis.flux_side_correction import (
    flux_side_rate,
    hybrid_flux_side_correction,
)


def test_flux_side_rate_keeps_reactant_normalization_and_bootstraps_replicas():
    time = np.linspace(0.0, 1.0, 6)
    flux = np.array([[2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                     [4.0, 4.0, 4.0, 4.0, 4.0, 4.0]])
    side = np.array([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                     [0.0, 0.5, 0.5, 0.5, 0.5, 0.5]])
    result = flux_side_rate(
        time, flux, side, reactant_population=np.array([2.0, 4.0]),
        bootstrap=32, rng=np.random.default_rng(4),
        plateau_start=0.2, plateau_stop=0.8,
    )
    # weighted mean(F h) / weighted mean(h_R), preserving replica-level
    # covariance in the bootstrap rather than resampling time frames.
    np.testing.assert_allclose(result.rate[1:], 2.0 / 3.0)
    assert result.bootstrap_rate.shape == (32, len(time))
    assert result.standard_error is not None
    assert result.diagnostics["plateau_detected"]
    assert result.physical_converged is False


def test_flux_side_correction_has_zero_defect_and_full_subspace_limits():
    time = np.linspace(0.0, 2.0, 5)
    full = np.linspace(0.1, 0.5, len(time))
    zero = hybrid_flux_side_correction(time, full, full, full)
    np.testing.assert_allclose(zero["rate"], full)
    quantum = full + 0.2
    full_limit = hybrid_flux_side_correction(time, full, full, quantum)
    np.testing.assert_allclose(full_limit["rate"], quantum)
    assert full_limit["diagnostics"]["exact_general_rate_claim"] is False


def test_inverse_flux_side_mode_requires_ridge():
    time = np.linspace(0.0, 1.0, 4)
    curve = np.ones(4)
    with pytest.raises(ValueError, match="positive ridge"):
        hybrid_flux_side_correction(time, curve, curve, curve, mode="inverse_defect")
    result = hybrid_flux_side_correction(
        time, curve, curve, curve + 0.1, mode="inverse_defect", ridge=0.05
    )
    assert np.all(np.isfinite(result["rate"]))
