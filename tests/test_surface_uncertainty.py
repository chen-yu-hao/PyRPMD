"""Do not label support-conditioned surface draws as joint uncertainty."""

import copy

import numpy as np
import pytest

from pyrpmd.workflows.rate import _aggregate


def weighted_children():
    children = []
    for i, transmission in enumerate([0.4, 0.65, 0.85, 0.7, 0.3, 0.9]):
        for pair, denominator in enumerate([1.0 + 0.2 * i, 2.5]):
            children.append({
                "chain": i // 3, "parent": i % 3, "pair": pair,
                "numerator": np.full(40, transmission * denominator),
                "denominator": denominator,
                "energy_drift": 0.0,
                "endpoint_energy_error_hartree_per_bead": 0.0,
            })
    return children


def aggregate(pmf, output, task="surface_rate"):
    config = {"task": task, "seed": 18, "beta": 2.0, "dt_au": 0.05}
    options = {"child_steps": 40, "plateau_fraction": 0.5,
               "plateau_tolerance": 0.1, "bootstrap": 50}
    return _aggregate(weighted_children(), pmf, config, output, options)


def pmf_surface():
    return {
        "ready": True, "method": "mbar_surface_density", "warnings": [],
        "k_qtst_s_inv": 12.0, "pmf_bootstrap_qtst": [9.0, 10.0, 13.0, 15.0],
        "bootstrap": {"requested": 4, "attempted": 4, "valid": 4},
    }


def test_rejected_pmf_draws_withhold_joint_interval_but_preserve_conditioned_estimate(tmp_path):
    supported = pmf_surface()
    complete = aggregate(supported, tmp_path / "complete")
    incomplete = copy.deepcopy(supported)
    incomplete["bootstrap"].update(requested=6, attempted=6, valid=4)
    conditioned = aggregate(incomplete, tmp_path / "conditioned")
    expected_kappa = (
        sum(c["numerator"][0] for c in weighted_children())
        / sum(c["denominator"] for c in weighted_children())
    )
    assert conditioned["kappa"] == pytest.approx(expected_kappa)
    assert conditioned["rate_estimate_s_inv"] == pytest.approx(12 * expected_kappa)
    assert conditioned["rate_estimate_s_inv"] == complete["rate_estimate_s_inv"]
    assert complete["rate_uncertainty"]["available"] is True
    assert complete["rate_standard_error_s_inv"] > 0
    assert complete["rate_percentile_95_ci_s_inv"][0] < complete["rate_percentile_95_ci_s_inv"][1]
    assert conditioned["rate_standard_error_s_inv"] is None
    assert conditioned["rate_percentile_95_ci_s_inv"] is None
    assert conditioned["rate_uncertainty"]["available"] is False
    assert "pmf_bootstrap_contains_unsupported_draws" in conditioned["rate_uncertainty"]["reasons"]
    assert conditioned["rate_uncertainty"]["pmf_bootstrap_rejected"] == 2
    assert conditioned["rate_uncertainty"]["pmf_bootstrap_attempted"] == 6
    assert conditioned["rate_uncertainty"]["pmf_bootstrap_valid"] == 4
    assert "conditional on" in conditioned["rate_uncertainty"]["supported_draw_condition"]
    assert conditioned["supported_draw_rate_standard_error_s_inv"] == complete["rate_standard_error_s_inv"]
    assert conditioned["supported_draw_rate_percentile_95_ci_s_inv"] == complete["rate_percentile_95_ci_s_inv"]
    assert conditioned["conditional_rate_standard_error_s_inv"] == complete["conditional_rate_standard_error_s_inv"] > 0


@pytest.mark.parametrize("draws", [[], [10.0]])
def test_too_few_supported_draws_do_not_create_any_joint_error(tmp_path, draws):
    pmf = pmf_surface()
    pmf["pmf_bootstrap_qtst"] = draws
    pmf["bootstrap"].update(requested=6, attempted=6, valid=len(draws))
    result = aggregate(pmf, tmp_path)
    assert result["rate_standard_error_s_inv"] is None
    assert result["rate_percentile_95_ci_s_inv"] is None
    assert result["supported_draw_rate_standard_error_s_inv"] is None
    assert result["supported_draw_rate_percentile_95_ci_s_inv"] is None
    assert result["conditional_rate_standard_error_s_inv"] > 0
    assert "fewer_than_two_valid_pmf_bootstrap_draws" in result["rate_uncertainty"]["reasons"]
    assert result["rate_uncertainty"]["pmf_bootstrap_rejected"] == 6 - len(draws)


def test_surface_support_policy_does_not_change_existing_gas_phase_uncertainty(tmp_path):
    pmf = {
        "ready": True, "method": "mbar", "warnings": [],
        "k_qtst_cm3_s": 12.0, "free_energy_barrier_hartree": 0.2,
        "pmf_bootstrap_barriers": [0.12, 0.18, 0.23, 0.27],
        "bootstrap": {"requested": 4, "attempted": 4, "valid": 4},
    }
    unchanged = aggregate(pmf, tmp_path / "gas_supported", task="rate")
    pmf["bootstrap"].update(requested=6, attempted=6, valid=4)
    rejected_metadata = aggregate(pmf, tmp_path / "gas_metadata", task="rate")
    for key in ("rate_estimate_cm3_s", "rate_standard_error_cm3_s", "rate_percentile_95_ci_cm3_s",
                "conditional_rate_standard_error_cm3_s", "rate_uncertainty"):
        assert unchanged[key] == rejected_metadata[key]
    assert rejected_metadata["rate_uncertainty"]["available"] is True
    assert rejected_metadata["rate_standard_error_cm3_s"] > 0
    assert not any(key.startswith("supported_draw_") for key in rejected_metadata)
    assert "pmf_bootstrap_contains_unsupported_draws" not in rejected_metadata["rate_uncertainty"]["reasons"]
