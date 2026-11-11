"""MBAR's finite-array fast path retains the equation and weight oracles."""

import numpy as np
import pytest
from scipy.special import logsumexp

from pyrpmd.analysis import mbar


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_finite_reduction_matches_scipy_over_extreme_ranges(axis):
    rng = np.random.default_rng(481)
    values = rng.normal(size=(9, 41)) * 400
    values[0, :4] = [0., -1000., 1000., 1e6]
    np.testing.assert_allclose(mbar._finite_logsumexp(values, axis=axis),
                               logsumexp(values, axis=axis), rtol=2e-15, atol=2e-13)
    with np.errstate(invalid="ignore", over="ignore"):
        values[0, 0] = np.inf
        np.testing.assert_allclose(mbar._finite_logsumexp(values, axis=axis),
                                   logsumexp(values, axis=axis), equal_nan=True)


def test_heterogeneous_umbrella_solution_matches_scipy_and_legacy(monkeypatch):
    rng = np.random.default_rng(214)
    centers = np.linspace(-1., 1., 7)
    springs = np.array([2., 5., 8., 30., 12., 4., 2.])
    x = np.concatenate([rng.normal(c, 1/np.sqrt(k), 250) for c, k in zip(centers, springs)])
    u = .5 * springs[:, None] * (x - centers[:, None])**2
    counts = np.full(7, 250)
    fast = mbar.solve_mbar(u, counts, tolerance=1e-11)
    warm = mbar.solve_mbar(u, counts, tolerance=1e-11,
                          initial_free_energies=fast.free_energies + np.linspace(0., .1, 7))
    legacy = mbar.solve_mbar(u, counts, method="legacy", tolerance=1e-11)
    monkeypatch.setattr("pyrpmd.analysis.mbar_streaming._finite_logsumexp", logsumexp)
    reference = mbar.solve_mbar(u, counts, tolerance=1e-11, backend="numpy")
    for result in (fast, warm, legacy, reference):
        assert result.converged and result.residual < 1e-10
        np.testing.assert_allclose(result.free_energies, reference.free_energies, atol=2e-9)
        np.testing.assert_allclose(result.weights, reference.weights, rtol=2e-9, atol=1e-12)
        np.testing.assert_allclose(result.overlap, reference.overlap, atol=2e-9)
