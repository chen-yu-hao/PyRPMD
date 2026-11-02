import numpy as np
import pytest

from pyrpmd.adaptive_string import adaptive_mean_force_string


def test_straight_valley_and_pinned_endpoints():
    start = np.column_stack((np.linspace(-1, 1, 9), 0.15*np.sin(np.linspace(0, np.pi, 9))))
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


def test_budget_residual_matches_returned_geometry():
    path = np.column_stack((np.linspace(-1, 1, 5), np.linspace(0, .5, 5)))
    def gradient(x):
        return np.column_stack((np.zeros(len(x)), 10*x[:, 1]))
    result = adaptive_mean_force_string(path, gradient, step=.001,
                                        tolerance=1e-12, max_iterations=1)
    tangent = result.images[2:] - result.images[:-2]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    force = gradient(result.images)[1:-1]
    force -= np.sum(force*tangent, axis=1)[:, None]*tangent
    expected = np.max(np.linalg.norm(force, axis=1))
    assert result.maximum_perpendicular_force == pytest.approx(expected, abs=1e-14)
    assert result.history[-1] == result.maximum_perpendicular_force
    assert len(result.history) == result.iterations + 1


def test_last_update_can_reach_tolerance():
    path = np.array([[-1., 0.], [0., 1.], [1., 0.]])
    result = adaptive_mean_force_string(path, lambda x: np.column_stack(
        (np.zeros(len(x)), x[:, 1])), step=1., tolerance=1e-12,
        max_iterations=1)
    assert result.converged
    assert result.maximum_perpendicular_force == 0
    assert result.iterations == 1


@pytest.mark.parametrize("option", [
    {"step": True}, {"tolerance": False}, {"max_iterations": True},
    {"max_images": np.bool_(True)}, {"adapt_every": True},
    {"step": [1, 2]}, {"tolerance": 1j}, {"insertion_threshold": np.inf},
])
def test_reject_invalid_controls(option):
    path = np.array([[-1., 0.], [0., 1.], [1., 0.]])
    with pytest.raises(ValueError):
        adaptive_mean_force_string(path, lambda x: x, **option)


@pytest.mark.parametrize("value", [complex(1, 1), np.nan, np.inf, True])
def test_reject_invalid_oracle_without_silent_complex_cast(value):
    path = np.array([[-1., 0.], [0., 1.], [1., 0.]])
    with pytest.raises(ValueError):
        adaptive_mean_force_string(path, lambda x: np.full(x.shape, value))


def test_adaptation_inserts_without_exceeding_cap_or_mutating_input():
    path = np.array([[-1., 0.], [-.5, .8], [0., 1.], [.5, .8], [1., 0.]])
    original = path.copy()
    result = adaptive_mean_force_string(path, lambda x: x, step=.001,
        tolerance=1e-14, adapt_every=1, insertion_threshold=.01,
        max_iterations=4, max_images=7)
    assert len(result.images) == 7
    np.testing.assert_array_equal(path, original)
    np.testing.assert_array_equal(result.images[[0, -1]], original[[0, -1]])
    assert not np.shares_memory(result.images, path)


def test_degenerate_move_is_rejected():
    path = np.array([[-1., 0.], [0., 0.], [-1., 0.]])
    with pytest.raises(ValueError, match="tangent"):
        adaptive_mean_force_string(path, lambda x: x)
