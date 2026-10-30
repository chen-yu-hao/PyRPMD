"""Numerical edge cases found while exercising the real H3 postprocessor."""

import numpy as np
import pytest

from pyrpmd.analysis.rate_postprocess import flux_slice_rate_2d


def test_decimal_slice_does_not_borrow_masked_ulp_neighbor():
    # The physical slice 1.0 and its histogram centre differ by one ULP.
    edges = [np.arange(-0.025, 1.125, 0.05), np.array([0.0, 0.3, 1.0])]
    centres = (edges[0][:-1] + edges[0][1:]) / 2
    target = int(np.argmin(abs(centres-1)))
    f = np.zeros((len(centres), 2))
    mask = np.zeros_like(f, dtype=bool)
    mask[target+1, 0] = True
    result = flux_slice_rate_2d(f, edges, beta=1, surface=1.0, reference=0.0,
                               reference_rate_au=2.0, mask=mask)
    assert result["k_tst_au"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="masked|unsupported"):
        flux_slice_rate_2d(f, edges, beta=1, surface=1.0+1e-6, reference=0.0,
                           reference_rate_au=2.0, mask=mask)


def test_masked_exact_slice_itself_still_fails():
    edges = [np.linspace(0, 2, 5), np.array([0.0, 1.0])]
    f = np.zeros((4, 1))
    mask = np.zeros_like(f, dtype=bool)
    mask[2, 0] = True
    with pytest.raises(ValueError, match="masked|unsupported"):
        flux_slice_rate_2d(f, edges, beta=1, surface=1.25, reference=0.25,
                           reference_rate_au=1.0, mask=mask)


@pytest.mark.parametrize("axis", [True, 0.0, 1])
def test_flux_axis_is_the_explicit_integer_zero(axis):
    with pytest.raises(ValueError, match="flux_axis"):
        flux_slice_rate_2d(np.zeros((1, 1)), [[0., 1.], [0., 1.]], beta=1,
                           surface=.5, reference=.5, reference_rate_au=1,
                           flux_axis=axis)
