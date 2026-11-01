import numpy as np
import pytest

from pyrpmd.adaptive_string import adaptive_mean_force_string


def test_straight_valley_and_pinned_endpoints():
    start = np.column_stack((np.linspace(-1, 1, 9), np.linspace(0, .4, 9)))
    def gradient(x):
        return np.column_stack((4*x[:, 0]*(x[:, 0]**2-1), 20*x[:, 1]))
    result = adaptive_mean_force_string(start, gradient, step=.02,
                                        tolerance=1e-5, max_iterations=500)
    assert result.converged
    np.testing.assert_array_equal(result.images[0], start[0])
    np.testing.assert_array_equal(result.images[-1], start[-1])
    assert np.max(np.abs(result.images[1:-1, 1])) < 1e-4


def test_bad_oracle_rejected():
    path = np.column_stack((np.linspace(-1, 1, 5), np.zeros(5)))
    with pytest.raises(ValueError, match="gradient"):
        adaptive_mean_force_string(path, lambda x: np.zeros((2, 2)))


def test_not_falsely_converged():
    path = np.column_stack((np.linspace(-1, 1, 5), np.linspace(0, .5, 5)))
    result = adaptive_mean_force_string(path, lambda x: np.column_stack(
        (np.zeros(len(x)), x[:, 1] * 10)), step=.001,
        tolerance=1e-12, max_iterations=1)
    assert not result.converged
    assert result.maximum_perpendicular_force > 0
