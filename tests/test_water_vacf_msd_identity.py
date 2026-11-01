"""Excluded analytical checks of the additive, offline identity diagnostic."""

import numpy as np
import pytest

from benchmarks.water import vacf_msd_identity as diagnostic


def test_ballistic_velocity_dot_correlation_has_factor_two_and_one_third():
    lag = np.array([0., .02, .07, .19, .4])
    result = diagnostic.stationary_curves(lag, np.full_like(lag, 2.4))
    np.testing.assert_allclose(result["MSD_from_VACF"], 2.4*lag**2, atol=1e-16)
    np.testing.assert_allclose(result["D_VACF"], 2.4*lag/3, atol=1e-16)


def test_linear_signed_correlation_is_exact_on_nonuniform_grid():
    lag = np.array([0., .02, .11, .6, 1.4, 2.])
    result = diagnostic.stationary_curves(lag, 4-3*lag)
    np.testing.assert_allclose(result["MSD_from_VACF"], 4*lag**2-lag**3, atol=2e-15)
    np.testing.assert_allclose(result["D_VACF"], (4*lag-1.5*lag**2)/3, atol=2e-15)
    assert not np.allclose(result["MSD_from_VACF"], result["MSD_double_trapezoid"])


@pytest.mark.parametrize("lag,vacf", [([0., .1, .1], [1., 2., 3.]),
                                      ([.1, .2, .3], [1., 2., 3.]),
                                      ([0., .1, .2], [1., float("nan"), 3.])])
def test_invalid_or_nonfinite_saved_curve_is_not_silently_excluded(lag, vacf):
    with pytest.raises(ValueError, match="finite matching"):
        diagnostic.stationary_curves(lag, vacf)


def test_positive_intercept_preserves_linear_diffusion_slope():
    lag = np.linspace(0, 2, 401)
    msd = 6*.21*lag+.18
    assert diagnostic.slope_D(lag, msd, (.5, 1)) == pytest.approx(.21)
    assert diagnostic.slope_D(lag, msd, (1, 2)) == pytest.approx(.21)
    with pytest.raises(ValueError, match="complete original fit window"):
        diagnostic.slope_D(lag[:100], msd[:100], (.5, 1))


def test_uncertainty_uses_eight_paired_chain_differences():
    shared = np.arange(8)*100.
    differences = np.arange(1, 9)*.002
    summary = diagnostic.paired_chain_summary(shared+differences, shared)
    result = summary["observed_minus_reconstructed"]
    assert result["chains"] == 8 and result["degrees_of_freedom"] == 7
    assert result["mean"] == pytest.approx(differences.mean())
    assert result["standard_error"] == pytest.approx(differences.std(ddof=1)/np.sqrt(8))
    with pytest.raises(ValueError, match="exactly eight"):
        diagnostic.paired_chain_summary(np.zeros(200), np.zeros(200))
