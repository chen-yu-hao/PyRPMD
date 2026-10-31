"""Exchange-window samples share clock draws, unlike independent windows."""
import numpy as np
import pytest
from pyrpmd.analysis.fes import (_bootstrap_groups, _grouped_resample_indices,
                                 _block_resample, reconstruct_fes)


def test_synchronized_indices_preserve_pairing_and_independent_ladders():
    counts = np.array([80] * 4)
    groups = _bootstrap_groups([0, 1, 0, 1], counts)
    indices = _grouped_resample_indices(counts, np.array([9] * 4), groups, np.random.default_rng(2))
    clocks = indices.reshape(4, 80) - np.arange(4)[:, None] * 80
    np.testing.assert_array_equal(clocks[0], clocks[2])
    np.testing.assert_array_equal(clocks[1], clocks[3])
    assert not np.array_equal(clocks[0], clocks[1])
    signal = np.random.default_rng(4).normal(size=80)
    np.testing.assert_array_equal(signal[clocks[0]], signal[clocks[2]])


def test_default_resampling_preserves_previous_draws_and_rng_exactly():
    counts, blocks = np.array([20, 41, 37]), np.array([7, 13, 9])
    offsets = np.r_[0, np.cumsum(counts)]
    a, b = np.random.default_rng(72), np.random.default_rng(72)
    expected = np.concatenate([_block_resample(np.arange(lo, hi), length, a)
                               for lo, hi, length in zip(offsets[:-1], offsets[1:], blocks)])
    actual = _grouped_resample_indices(counts, blocks, _bootstrap_groups(None, counts), b)
    np.testing.assert_array_equal(expected, actual)
    assert a.bit_generator.state == b.bit_generator.state


@pytest.mark.parametrize("labels,counts", [
    ([0], [20, 20]), ([0., 0.], [20, 20]), ([True, False], [20, 20]),
    ([-1, 0], [20, 20]), ([0, 0], [20, 21]), ([[0], [1]], [20, 20]),
])
def test_invalid_groups_are_rejected(labels, counts):
    with pytest.raises(ValueError, match="bootstrap"):
        _bootstrap_groups(labels, np.array(counts))


def test_grouping_keeps_point_estimate_and_uses_group_max_block():
    rng = np.random.default_rng(94)
    slow = np.zeros(500)
    for i in range(1, len(slow)):
        slow[i] = .97 * slow[i - 1] + rng.normal(scale=.15)
    arrays = [rng.normal(size=(500, 1)), slow[:, None], rng.normal(size=(500, 1))]
    kwargs = dict(centers=[[0], [0], [0]], stiffness=[[.1]], beta=1., bins=8,
                  ranges=[[-2., 2.]], bootstrap=6, seed=81)
    independent = reconstruct_fes(arrays, **kwargs)
    grouped = reconstruct_fes(arrays, **kwargs, bootstrap_groups=[7, 7, 9])
    for key in ("free_energy", "weights", "raw_count", "density", "mask", "window_free_energies"):
        np.testing.assert_array_equal(independent[key], grouped[key])
    plan = grouped["bootstrap_block_plan"]
    assert plan["resampling"] == "synchronous_within_group"
    assert plan["block_lengths"][0] == plan["block_lengths"][1]
    assert plan["group_max_statistical_inefficiency"][0] == max(grouped["statistical_inefficiency"][:2])
    assert plan["mixing_established"] is False
