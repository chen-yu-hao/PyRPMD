"""Independent configurations, not bootstrap draw counts, support rate errors."""

import json

import numpy as np
import pytest

from pyrpmd.analysis.recrossing import summarize_recrossing
from pyrpmd.workflows import rate, site_rate


def child(chain, parent, pair, *, denominator=1.0, remains_product=True):
    # Each record is a physically allowed paired flux: v[h(q+) - h(q-)].
    # Both members initially depart the surface, then the returning pair's
    # indicator difference becomes zero. No fractional per-pair indicators.
    return {
        "chain": chain, "parent": parent, "pair": pair,
        "numerator": denominator * np.r_[np.ones(10), np.full(30, float(remains_product))],
        "denominator": denominator,
        "energy_drift": 0.0, "endpoint_energy_error_hartree_per_bead": 0.0,
        "endpoint_energy_error_kbt": 0.0, "surface": 0.6, "direction": 1,
    }


def aggregate(task, children, output, *, bootstrap=40):
    options = {
        "child_steps": 40, "plateau_fraction": 0.5,
        "plateau_tolerance": 0.1, "bootstrap": bootstrap,
    }
    config = {
        "task": task, "seed": 13 if task == "site_rate" else 18,
        "beta": 2.0, "dt_au": 0.05, "sampling": {},
        "recrossing": dict(options, equilibration_steps=0, parent_interval_steps=1),
    }
    if task == "site_rate":
        pmf = {
            "ready": True, "status": "ready", "warnings": [], "k_qtst_s": 12.0,
            "_bootstrap_qtst_s": [9.0, 10.0, 13.0, 15.0],
            "pmf_uncertainty": {"available": True, "reasons": []},
            "preparation": {"direction": 1}, "dividing_surface": 0.6,
        }
        return site_rate._aggregate(children, pmf, config, output), "s"
    if task == "surface_rate":
        pmf = {
            "ready": True, "method": "mbar_surface_density", "warnings": [],
            "k_qtst_s_inv": 12.0, "pmf_bootstrap_qtst": [9.0, 10.0, 13.0, 15.0],
            "bootstrap": {"requested": 4, "attempted": 4, "valid": 4},
        }
        unit = "s_inv"
    else:
        pmf = {
            "ready": True, "warnings": [], "k_qtst_cm3_s": 12.0,
            "free_energy_barrier_hartree": 0.01,
            "pmf_bootstrap_barriers": [0.005, 0.015, 0.02, 0.025],
        }
        unit = "cm3_s"
    return rate._aggregate(children, pmf, config, output, options), unit


@pytest.mark.parametrize("task", ["rate", "surface_rate", "site_rate"])
def test_one_parent_does_not_gain_error_support_from_many_bootstrap_draws(task, tmp_path):
    records = [child(0, 0, 0), child(0, 0, 1, remains_product=False)]
    result, unit = aggregate(task, records, tmp_path, bootstrap=100)
    assert result["kappa"] == 0.5
    assert result["parent_configurations"] == 1
    assert result["child_pairs"] == 2
    assert result["kappa_standard_error"] is None
    assert result["rate_standard_error_" + unit] is None
    assert result["rate_percentile_95_ci_" + unit] is None
    assert result["conditional_rate_standard_error_" + unit] is None
    assert not result["rate_uncertainty"]["available"]
    support = result["rate_uncertainty"]["recrossing_support"]
    assert support["bootstrap_valid"] == 100
    assert support["nonzero_flux_parent_configurations"] == 1
    assert "fewer_than_two_nonzero_flux_parent_configurations" in support["reasons"]
    with np.load(tmp_path / "recrossing.npz") as saved:
        # Diagnostic draws survive even though their degeneracy cannot prove
        # zero sampling error for the unsampled parent distribution.
        np.testing.assert_array_equal(saved["bootstrap_kappa"], np.full(100, 0.5))


@pytest.mark.parametrize("task", ["rate", "surface_rate", "site_rate"])
def test_zero_flux_parents_and_chains_do_not_inflate_sample_support(task, tmp_path):
    records = [child(0, 0, 0)]
    records += [child(0, parent, 0, denominator=0.0) for parent in range(1, 10)]
    records += [child(1, parent, 0, denominator=0.0) for parent in range(10)]
    result, unit = aggregate(task, records, tmp_path)
    support = result["rate_uncertainty"]["recrossing_support"]
    assert support["independent_parent_chains"] == 1
    assert support["nonzero_flux_parent_configurations"] == 1
    assert not support["available"]
    assert result["rate_standard_error_" + unit] is None
    assert result["conditional_rate_standard_error_" + unit] is None


@pytest.mark.parametrize("task", ["rate", "surface_rate", "site_rate"])
def test_single_chain_without_two_correlation_blocks_withholds_errors(task, tmp_path):
    records = [child(0, parent, pair, remains_product=pair == 0)
               for parent in range(12) for pair in range(2)]
    result, unit = aggregate(task, records, tmp_path)
    support = result["rate_uncertainty"]["recrossing_support"]
    assert support["nonzero_flux_parent_configurations"] == 12
    assert support["effective_parent_blocks"] == {"0": 1.0}
    assert "fewer_than_two_effective_parent_blocks" in support["reasons"]
    assert result["rate_standard_error_" + unit] is None
    assert result["conditional_rate_standard_error_" + unit] is None


@pytest.mark.parametrize("task", ["rate", "surface_rate", "site_rate"])
def test_long_single_chain_errors_are_explicitly_conditional(task, tmp_path):
    records = [child(0, parent, 0, remains_product=parent % 2 == 0) for parent in range(64)]
    result, unit = aggregate(task, records, tmp_path, bootstrap=80)
    support = result["rate_uncertainty"]["recrossing_support"]
    assert support["available"]
    assert support["effective_parent_blocks"]["0"] == 64
    assert support["scope"] == "conditional_on_single_chain_mixing"
    assert "independent-chain agreement has not been established" in support["condition"]
    assert result["conditional_rate_standard_error_" + unit] > 0
    if task == "site_rate":
        # Preserve the site's stronger primary policy while retaining the
        # useful single-chain diagnostic under its explicit conditional name.
        assert not result["rate_uncertainty"]["available"]
        assert result["rate_standard_error_s"] is None
        assert result["rate_percentile_95_ci_s"] is None
        assert result["conditional_kappa_standard_error"] > 0
    else:
        assert result["rate_uncertainty"]["available"]
        assert result["rate_standard_error_" + unit] > 0
        interval = result["rate_percentile_95_ci_" + unit]
        assert interval[1] > interval[0]


@pytest.mark.parametrize("task", ["rate", "surface_rate", "site_rate"])
def test_zero_flux_resamples_withhold_primary_error_even_with_multiple_chains(task, tmp_path):
    records = [child(0, 0, 0), child(1, 0, 0, remains_product=False),
               child(2, 0, 0, denominator=0.0)]
    result, unit = aggregate(task, records, tmp_path, bootstrap=200)
    support = result["rate_uncertainty"]["recrossing_support"]
    assert support["independent_parent_chains"] == 2
    assert 2 <= support["bootstrap_valid"] < 200
    assert support["bootstrap_rejected"] == 200 - support["bootstrap_valid"]
    assert "recrossing_bootstrap_contains_zero_flux_draws" in support["reasons"]
    assert not result["rate_uncertainty"]["available"]
    assert result["rate_standard_error_" + unit] is None
    assert result["rate_percentile_95_ci_" + unit] is None


@pytest.mark.parametrize("task", ["rate", "surface_rate", "site_rate"])
def test_recrossing_diagnostics_preserve_result_identity_after_json_roundtrip(task, tmp_path):
    records = [child(0, 0, 0), child(1, 0, 0, remains_product=False)]
    result, _ = aggregate(task, records, tmp_path)
    diagnostics = result["rate_uncertainty"]
    assert set(diagnostics["parent_block_lengths"]) == {"0", "1"}
    assert set(diagnostics["recrossing_support"]["effective_parent_blocks"]) == {"0", "1"}
    assert json.loads(json.dumps(result)) == result


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_shared_recrossing_options_reject_nonfinite_plateau_tolerance(value):
    config = {
        "dt_au": 0.05, "sampling": {},
        "recrossing": {"equilibration_steps": 0, "parent_interval_steps": 1,
                       "child_steps": 40, "plateau_tolerance": value},
    }
    with pytest.raises(ValueError, match="plateau"):
        rate._recrossing_options(config)


def test_single_frame_cannot_define_two_plateau_halves(recwarn):
    record = child(0, 0, 0)
    record["numerator"] = np.ones(1)
    with pytest.raises(ValueError, match="child_steps must be at least 2"):
        summarize_recrossing(
            [record], child_steps=1, plateau_fraction=1.0, plateau_tolerance=0.1,
            bootstrap=2, rng=np.random.default_rng(82),
        )
    assert not recwarn


@pytest.mark.parametrize("task, expected_error, expected_draws", [
    ("surface_rate", 2.1898652271619694, [
        0.7398373983739839, 0.5, 0.43700787401574803, 0.5916666666666667,
        0.5916666666666667, 0.9193548387096773, 0.6621621621621621,
        0.6666666666666665, 0.5696721311475411, 0.8400000000000001,
        0.7831858407079646, 0.6145038167938933,
    ]),
    ("site_rate", 1.9395801167287188, [
        0.2991452991452991, 0.7644628099173552, 0.723214285714286,
        0.6045454545454547, 0.5916666666666667, 0.49576271186440674,
        0.415966386554622, 0.6666666666666666, 0.5504587155963302,
        0.6015625, 0.5598290598290598, 0.47881355932203407,
    ]),
])
def test_shared_core_preserves_supported_workflow_random_streams(task, expected_error, expected_draws, tmp_path):
    # Frozen from the pre-refactor workflow at af90f20d using these admissible
    # weighted pairs. This checks draw order and subsequent PMF combination,
    # rather than duplicating the implementation in a second test function.
    records = []
    for i in range(6):
        for pair, denominator in enumerate([1.0 + 0.2 * i, 2.5]):
            keep = i % 2 == 0 if pair == 0 else i % 3 != 0
            records.append(child(i // 3, i % 3, pair, denominator=denominator, remains_product=keep))
    result, unit = aggregate(task, records, tmp_path, bootstrap=12)
    assert result["rate_uncertainty"]["available"]
    assert result["rate_uncertainty"]["recrossing_scope"] == "independent_parent_chains"
    assert result["rate_standard_error_" + unit] == expected_error
    with np.load(tmp_path / "recrossing.npz") as saved:
        np.testing.assert_array_equal(saved["bootstrap_kappa"], expected_draws)
