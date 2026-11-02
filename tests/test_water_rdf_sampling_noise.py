"""Analytic excluded-fixture checks; no trajectory or gate tuning from diffusion."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ive

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.water import analyze_rdf_sampling_noise as noise


def test_unordered_finite_pair_normalization_over_sphere():
    exposure = noise.unordered_pair_exposure(np.array([0., 1., 2.]), molecules=4,
                                             box_angstrom=4., independent_frames=5)
    # Six unordered pairs, five frames, sphere of radius L/2 fills pi/6 of box.
    assert exposure.sum() == pytest.approx(6 * 5 * np.pi / 6)
    assert exposure[1] / exposure[0] == pytest.approx(7)


def test_two_window_variance_and_strict_integer_tail():
    result = noise.unchanged_count_noise([2.], g=.25, margin=.5, chains=1)
    assert result["difference_standard_deviation"][0] == pytest.approx(.5)
    # X,Y~Pois(0.5): the strict |X-Y|>1 tail excludes -1,0,+1.
    expected = 1 - ive(0, 1.) - 2 * ive(1, 1.)
    assert result["single_bin_false_alarm_probability"][0] == pytest.approx(expected, abs=1e-14)


def test_multiple_bins_and_chains_are_explicit_additional_assumptions():
    result = noise.unchanged_count_noise([200., 300.], chains=8)
    per_bin = result["single_bin_false_alarm_probability"]
    assert result["one_chain_independent_bins_false_alarm_probability"] == pytest.approx(1-np.prod(1-per_bin))
    assert result["all_chains_independent_bins_and_chains_false_alarm_probability"] == pytest.approx(1-np.prod(1-per_bin)**8)
    assert result["one_chain_union_bound"] >= result["one_chain_independent_bins_false_alarm_probability"]
    assert result["all_chains_union_bound"] >= result["all_chains_independent_bins_and_chains_false_alarm_probability"]


def test_unoccupied_shell_has_no_count_noise():
    result = noise.unchanged_count_noise([10., 20.], g=0)
    assert np.array_equal(result["difference_standard_deviation"], [0, 0])
    assert result["all_chains_independent_bins_and_chains_false_alarm_probability"] == 0


@pytest.mark.parametrize("kwargs", [{"molecules": 1}, {"independent_frames": 0}, {"box_angstrom": 3.9}])
def test_invalid_exposure_design_is_rejected(kwargs):
    arguments = dict(molecules=216, independent_frames=200, box_angstrom=18.6)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        noise.unordered_pair_exposure([0, 1, 2], **arguments)


def test_registered_185_bin_uniform_null_demonstrates_multiplicity():
    exposure = noise.unordered_pair_exposure(np.linspace(2.3, 6., 186), molecules=216,
                                             independent_frames=200, box_angstrom=18.642822871774577)
    result = noise.unchanged_count_noise(exposure)
    assert .41 < result["one_chain_independent_bins_false_alarm_probability"] < .43
    assert .98 < result["all_chains_independent_bins_and_chains_false_alarm_probability"] < .99
    assert result["single_bin_false_alarm_probability"].max() < .03
