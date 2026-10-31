"""Observed support, not the number of retained draws, qualifies FES errors."""

import numpy as np

from pyrpmd.analysis.fes import reconstruct_fes


def shuffled(values):
    values = np.asarray(values, float)[:, None]
    np.random.default_rng(7).shuffle(values)
    return values


def test_local_missing_support_withholds_only_the_affected_primary_error():
    sample = shuffled(np.r_[np.full(120, .5), 1.5, np.full(120, 2.5)])
    options = dict(centers=[[1.5]], stiffness=[[1e-12]], beta=2.,
                   bins=[[0., 1., 2., 3.]], seed=52)
    point = reconstruct_fes([sample], bootstrap=0, **options)
    result = reconstruct_fes([sample], bootstrap=40, **options)
    np.testing.assert_array_equal(result["free_energy"], point["free_energy"])
    np.testing.assert_array_equal(result["mask"], point["mask"])
    assert not result["mask"].any()
    assert result["bootstrap_replicates"] == 40
    valid = result["bootstrap_valid_per_bin"]
    assert valid[0] == valid[2] == 40
    assert 2 <= valid[1] < 40
    assert np.isnan(result["standard_error"][1])
    assert result["supported_draw_standard_error"][1] > 0
    assert np.isfinite(result["standard_error"][[0, 2]]).all()
    np.testing.assert_array_equal(
        result["standard_error"][[0, 2]], result["supported_draw_standard_error"][[0, 2]],
    )
    assert "conditional on" in result["supported_draw_condition"]


def test_missing_reference_draws_withhold_all_primary_errors_but_keep_point_fes():
    # A rare, strongly reweighted frame defines the original minimum bin.
    # Draws missing it cannot use another reference without changing the target.
    sample = shuffled(np.r_[np.zeros(120), np.ones(120), 2.])
    options = dict(centers=[[0.]], stiffness=[[10.]], beta=1.,
                   bins=[[-.5, .5, 1.5, 2.5]], seed=52)
    point = reconstruct_fes([sample], bootstrap=0, **options)
    result = reconstruct_fes([sample], bootstrap=40, **options)
    assert result["bootstrap_reference_bin"].tolist() == [2]
    assert 2 <= result["bootstrap_replicates"] < 40
    failures = [draw for draw in result["bootstrap_diagnostics"] if not draw["valid"]]
    assert {draw["reason"] for draw in failures} == {"reference_bin_masked"}
    np.testing.assert_array_equal(result["free_energy"], point["free_energy"])
    assert np.isfinite(result["free_energy"]).all()
    assert np.isnan(result["standard_error"]).all()
    assert np.isfinite(result["supported_draw_standard_error"]).all()
    assert (result["supported_draw_standard_error"][:2] > 0).all()


def test_originally_masked_bins_do_not_gain_conditional_error_bars():
    sample = shuffled(np.r_[np.full(120, .5), np.full(120, 1.5),
                            np.full(120, 2.5), 3.5])
    result = reconstruct_fes(
        [sample], [[1.5]], [[1e-12]], 2., bins=[[0., 1., 2., 3., 4.]],
        bootstrap=40, seed=52, min_count=2,
    )
    np.testing.assert_array_equal(result["mask"], [False, False, False, True])
    assert np.isnan(result["standard_error"][-1])
    assert np.isnan(result["supported_draw_standard_error"][-1])
    assert result["bootstrap_valid_per_bin"][-1] == 0
    assert np.isfinite(result["standard_error"][:3]).all()


def test_unresolved_streams_cannot_recover_errors_under_the_conditional_label():
    samples = [np.full((100, 1), center) for center in (.5, 1.5, 2.5)]
    result = reconstruct_fes(
        samples, [[.5], [1.5], [2.5]], [[1e-12]], 2.,
        bins=[[0., 1., 2., 3.]], bootstrap=8, seed=52,
    )
    assert result["connected"] and not result["mask"].any()
    assert result["bootstrap_block_plan"]["status"] == "unavailable"
    assert result["bootstrap_replicates"] == 8
    assert np.isfinite(result["free_energy"]).all()
    assert np.isnan(result["standard_error"]).all()
    assert np.isnan(result["supported_draw_standard_error"]).all()


def test_disabled_bootstrap_keeps_both_errors_unavailable():
    result = reconstruct_fes(
        [shuffled(np.linspace(0., 1., 100))], [[.5]], [[1.]], 2.,
        bins=5, ranges=[[0., 1.]], bootstrap=0,
    )
    assert np.isnan(result["standard_error"]).all()
    assert np.isnan(result["supported_draw_standard_error"]).all()
    assert not result["bootstrap_valid_per_bin"].any()
