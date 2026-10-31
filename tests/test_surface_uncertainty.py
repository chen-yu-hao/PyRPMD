"""Do not label support-conditioned surface draws as joint uncertainty."""

import copy
import json

import numpy as np
import pytest

from pyrpmd.analysis.surface import surface_rate_pmf
from pyrpmd.analysis.fes import mbar_rate_pmf
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


def test_failed_pmf_draws_are_conditioned_for_gas_phase_rates_too(tmp_path):
    pmf = {
        "ready": True, "method": "mbar", "warnings": [],
        "k_qtst_cm3_s": 12.0, "free_energy_barrier_hartree": 0.2,
        "pmf_bootstrap_barriers": [0.12, 0.18, 0.23, 0.27],
        "bootstrap": {"requested": 4, "attempted": 4, "valid": 4},
    }
    unchanged = aggregate(pmf, tmp_path / "gas_supported", task="rate")
    pmf["bootstrap"].update(requested=6, attempted=6, valid=4)
    rejected_metadata = aggregate(pmf, tmp_path / "gas_metadata", task="rate")
    for key in ("rate_estimate_cm3_s", "conditional_rate_standard_error_cm3_s"):
        assert unchanged[key] == rejected_metadata[key]
    assert rejected_metadata["rate_uncertainty"]["available"] is False
    assert rejected_metadata["rate_standard_error_cm3_s"] is None
    assert rejected_metadata["rate_percentile_95_ci_cm3_s"] is None
    assert rejected_metadata["supported_draw_rate_standard_error_cm3_s"] == unchanged["rate_standard_error_cm3_s"]
    assert rejected_metadata["supported_draw_rate_percentile_95_ci_cm3_s"] == unchanged["rate_percentile_95_ci_cm3_s"]
    assert "pmf_bootstrap_contains_unsupported_draws" in rejected_metadata["rate_uncertainty"]["reasons"]
    assert rejected_metadata["rate_uncertainty"]["pmf_bootstrap_rejected"] == 2
    assert "conditional on" in rejected_metadata["rate_uncertainty"]["supported_draw_condition"]


@pytest.mark.parametrize("block_status", ["finite_block_estimate", "limited_blocks", "unavailable"])
@pytest.mark.parametrize("task", ["rate", "surface_rate"])
@pytest.mark.parametrize("rejected_support", [False, True])
def test_block_support_gates_all_combined_errors_and_preserves_conditional_recrossing(
    tmp_path, block_status, task, rejected_support,
):
    if task == "surface_rate":
        pmf = pmf_surface()
        unit = "s_inv"
    else:
        pmf = {
            "ready": True, "method": "mbar", "warnings": [],
            "k_qtst_cm3_s": 12.0, "free_energy_barrier_hartree": 0.2,
            "pmf_bootstrap_barriers": [0.12, 0.18, 0.23, 0.27],
            "bootstrap": {"requested": 4, "attempted": 4, "valid": 4},
        }
        unit = "cm3_s"
    pmf["bootstrap"]["block_plan"] = {"status": block_status}
    if rejected_support:
        pmf["bootstrap"].update(requested=6, attempted=6)
    result = aggregate(pmf, tmp_path, task=task)
    supported_uncertainty = block_status != "unavailable"
    joint_uncertainty = supported_uncertainty and not rejected_support
    assert result["rate_uncertainty"]["available"] is joint_uncertainty
    assert (result["rate_standard_error_" + unit] is not None) is joint_uncertainty
    assert (result["rate_percentile_95_ci_" + unit] is not None) is joint_uncertainty
    assert result["rate_uncertainty"]["pmf_block_status"] == block_status
    assert result["conditional_rate_standard_error_" + unit] > 0
    if rejected_support:
        assert (result["supported_draw_rate_standard_error_" + unit] is not None) is supported_uncertainty
        assert (result["supported_draw_rate_percentile_95_ci_" + unit] is not None) is supported_uncertainty
    if block_status != "finite_block_estimate":
        assert "pmf_bootstrap_" + block_status in result["rate_uncertainty"]["reasons"]
    if block_status == "limited_blocks" and joint_uncertainty:
        assert any("Complete rate uncertainty has limited support" in w for w in result["warnings"])
        assert not any("Complete rate uncertainty is unavailable" in w for w in result["warnings"])


def test_actual_unmixed_surface_pmf_withholds_qtst_and_published_joint_uncertainty(tmp_path):
    # Every histogram bin has support, but none of the five time streams has
    # independent variation. Previously, whole-stream circular blocks yielded
    # identical draws and a spuriously available zero-width QTST interval.
    centers = np.arange(0.1, 1.0, 0.2)[:, None]
    samples = [np.full((100, 1), center) for center in centers[:, 0]]
    options = dict(basin=(0.1, 0.5), dividing_surface=0.7, mass_metric=1.0,
                   bins=np.linspace(0, 1, 6), seed=7, min_count=1)
    point = surface_rate_pmf(samples, centers, 1e-12, 2.0, bootstrap=0, **options)
    pmf = surface_rate_pmf(samples, centers, 1e-12, 2.0, bootstrap=8, **options)
    assert pmf["ready"] and not pmf["mask"].any()
    assert pmf["k_qtst_s_inv"] == point["k_qtst_s_inv"] > 0
    plan = pmf["bootstrap"]["block_plan"]
    assert plan["status"] == "unavailable"
    np.testing.assert_array_equal(plan["block_lengths"], [50] * 5)
    np.testing.assert_array_equal(pmf["bootstrap"]["block_lengths"], plan["block_lengths"])
    assert pmf["qtst_standard_error_s_inv"] is None
    assert pmf["qtst_percentile_95_ci_s_inv"] is None
    assert pmf["bootstrap"]["valid"] == 8
    assert pmf["bootstrap"]["failures"] == []
    assert pmf["bootstrap"]["log_density_integral_pairs"].shape == (8, 2)
    assert pmf["bootstrap"]["log_density_integral_covariance"].shape == (2, 2)
    assert any("unavailable" in warning for warning in pmf["warnings"])
    summary = aggregate(pmf, tmp_path)
    assert summary["rate_estimate_s_inv"] == pytest.approx(pmf["k_qtst_s_inv"] * summary["kappa"])
    assert summary["converged"] is False
    assert summary["rate_uncertainty"]["available"] is False
    assert summary["rate_uncertainty"]["pmf_block_status"] == "unavailable"
    assert "pmf_bootstrap_unavailable" in summary["rate_uncertainty"]["reasons"]
    assert summary["rate_standard_error_s_inv"] is None
    assert summary["rate_percentile_95_ci_s_inv"] is None
    assert summary["conditional_rate_standard_error_s_inv"] > 0
    public = json.loads((tmp_path / "summary.json").read_text())
    assert public["rate_uncertainty"]["available"] is False
    assert public["pmf"]["qtst_percentile_95_ci_s_inv"] is None


def test_actual_short_surface_streams_keep_point_estimates_but_not_resolved_errors(tmp_path):
    samples = [np.array([0.1, 0.3, 0.5])[:, None], np.array([0.5, 0.7, 0.9])[:, None]]
    pmf = surface_rate_pmf(samples, [[0.3], [0.7]], 1e-12, 2.0,
                           basin=(0.1, 0.5), dividing_surface=0.7, mass_metric=1.0,
                           bins=np.linspace(0, 1, 6), bootstrap=40, seed=55, min_count=1)
    assert pmf["ready"] and pmf["k_qtst_s_inv"] > 0
    assert pmf["bootstrap"]["block_plan"]["status"] == "unavailable"
    assert pmf["qtst_standard_error_s_inv"] is None
    assert pmf["qtst_percentile_95_ci_s_inv"] is None
    assert pmf["bootstrap"]["valid"] >= 2
    summary = aggregate(pmf, tmp_path)
    assert summary["rate_uncertainty"]["available"] is False
    assert summary["rate_standard_error_s_inv"] is None
    assert summary["rate_percentile_95_ci_s_inv"] is None
    if pmf["bootstrap"]["failures"]:
        assert summary["supported_draw_rate_standard_error_s_inv"] is None
        assert summary["supported_draw_rate_percentile_95_ci_s_inv"] is None


@pytest.mark.parametrize("n,status", [(80, "limited_blocks"), (1000, "finite_block_estimate")])
def test_actual_surface_block_diagnostics_propagate_without_dropping_paired_covariance(tmp_path, n, status):
    sample = ((np.arange(n) + 0.5) / n)[:, None]
    np.random.default_rng(81).shuffle(sample)
    beta, metric = 2.0, 0.04
    pmf = surface_rate_pmf([sample], [[0.5]], 1e-12, beta,
                           basin=(0.1, 0.5), dividing_surface=0.7, mass_metric=metric,
                           bins=np.linspace(0, 1, 6), bootstrap=20, seed=73, min_count=1)
    assert pmf["ready"] and pmf["bootstrap"]["valid"] == 20
    plan = pmf["bootstrap"]["block_plan"]
    assert plan["status"] == status
    np.testing.assert_array_equal(plan["requested_block_lengths"],
                                   np.ceil(5 * pmf["statistical_inefficiency"]).astype(int))
    assert pmf["qtst_standard_error_s_inv"] > 0
    pairs = pmf["bootstrap"]["log_density_integral_pairs"]
    log_rates = np.asarray([draw["log_k_qtst_au"] for draw in pmf["bootstrap"]["draws"]])
    np.testing.assert_allclose(log_rates, 0.5 * np.log(metric / (2 * np.pi * beta))
                               + pairs[:, 0] - pairs[:, 1], rtol=1e-13, atol=1e-13)
    contrast = np.array([1.0, -1.0])
    covariance = pmf["bootstrap"]["log_density_integral_covariance"]
    assert contrast @ covariance @ contrast == pytest.approx(np.var(log_rates, ddof=1), rel=1e-12)
    summary = aggregate(pmf, tmp_path)
    assert summary["rate_uncertainty"]["available"] is True
    assert summary["rate_uncertainty"]["pmf_block_status"] == status
    if status == "limited_blocks":
        assert "pmf_bootstrap_limited_blocks" in summary["rate_uncertainty"]["reasons"]


def test_actual_surface_failed_support_draws_remain_explicit_and_conditioned(tmp_path):
    sample = np.r_[np.full(120, 0.5), 1.5, np.full(120, 2.5)][:, None]
    np.random.default_rng(7).shuffle(sample)
    pmf = surface_rate_pmf([sample], [[1.5]], 1e-12, 2.0,
                           basin=(0.5, 1.5), dividing_surface=2.5, mass_metric=1.0,
                           bins=[0.0, 1.0, 2.0, 3.0], bootstrap=40, seed=52, min_count=1)
    info = pmf["bootstrap"]
    assert pmf["ready"] and info["block_plan"]["status"] != "unavailable"
    assert 2 <= info["valid"] < info["attempted"] == 40
    assert len(info["failures"]) + info["valid"] == 40
    assert sum(info["failure_counts"].values()) == len(info["failures"])
    assert all(failure["reason"] for failure in info["failures"])
    assert info["log_density_integral_pairs"].shape == (info["valid"], 2)
    assert pmf["qtst_standard_error_s_inv"] is None
    assert pmf["qtst_percentile_95_ci_s_inv"] is None
    assert pmf["supported_draw_qtst_standard_error_s_inv"] > 0
    assert pmf["supported_draw_qtst_percentile_95_ci_s_inv"] is not None
    assert not pmf["qtst_uncertainty"]["available"]
    assert "conditional on" in pmf["qtst_uncertainty"]["supported_draw_condition"]
    summary = aggregate(pmf, tmp_path)
    assert summary["rate_uncertainty"]["available"] is False
    assert "pmf_bootstrap_contains_unsupported_draws" in summary["rate_uncertainty"]["reasons"]
    assert summary["rate_standard_error_s_inv"] is None
    assert summary["rate_percentile_95_ci_s_inv"] is None
    assert summary["supported_draw_rate_standard_error_s_inv"] > 0
    assert summary["supported_draw_rate_percentile_95_ci_s_inv"] is not None


def test_actual_gas_pmf_failed_support_draws_remain_conditioned_in_the_final_rate(tmp_path):
    sample = np.r_[np.full(120, .5), 1.5, np.full(120, 2.5)]
    np.random.default_rng(7).shuffle(sample)
    estimate = mbar_rate_pmf(
        [sample], [1.5], 1e-12, 2., reference=.5, xi_star=2.5,
        bins=[0., 1., 2., 3.], bootstrap=40, seed=52, min_count=1,
    )
    info = estimate["bootstrap"]
    assert estimate["ready"] and info["block_plan"]["status"] == "finite_block_estimate"
    assert 2 <= info["valid"] < info["attempted"] == 40
    pmf = {key: estimate[key] for key in (
        "ready", "warnings", "free_energy_barrier_hartree",
        "pmf_bootstrap_barriers", "bootstrap", "barrier_uncertainty",
    )}
    pmf.update(method="mbar", k_qtst_cm3_s=12.)
    summary = aggregate(pmf, tmp_path, task="rate")
    assert summary["rate_estimate_cm3_s"] == pytest.approx(12 * summary["kappa"])
    assert summary["rate_standard_error_cm3_s"] is None
    assert summary["rate_percentile_95_ci_cm3_s"] is None
    assert summary["supported_draw_rate_standard_error_cm3_s"] > 0
    assert summary["supported_draw_rate_percentile_95_ci_cm3_s"] is not None
    assert summary["conditional_rate_standard_error_cm3_s"] > 0
    assert summary["rate_uncertainty"]["pmf_bootstrap_rejected"] == len(info["failures"])
    public = json.loads((tmp_path / "summary.json").read_text())
    assert public["rate_uncertainty"]["available"] is False
    assert "conditional on" in public["rate_uncertainty"]["supported_draw_condition"]


def quality_result(tmp_path, *, block_status="finite_block_estimate", draws=None,
                   parents=40, chains=4, target=.2):
    from test_recrossing_statistics import child
    children = [child(i % chains, i // chains, 0, remains_product=i % 3 != 0)
                for i in range(parents)]
    pmf = pmf_surface()
    pmf["bootstrap"]["block_plan"] = {"status": block_status}
    pmf["pmf_bootstrap_qtst"] = [11.9, 12., 12.1, 12.2] if draws is None else draws
    config = {"task": "surface_rate", "seed": 18, "beta": 2., "dt_au": .05,
              "surface": {"max_relative_standard_error": target}}
    options = {"child_steps": 40, "plateau_fraction": .5,
               "plateau_tolerance": .1, "bootstrap": 200}
    return _aggregate(children, pmf, config, tmp_path, options)


def test_surface_rate_is_reported_when_statistical_prerequisites_pass(tmp_path):
    result = quality_result(tmp_path)
    assert result["rate_quality"]["passed"]
    assert result["rate_quality"]["reasons"] == []
    assert result["rate_s_inv"] == result["rate_estimate_s_inv"] > 0
    assert result["rate_quality"]["relative_standard_error"] <= .2
    assert not result["converged"]
    assert not result["rate_quality"]["physical_convergence_established"]


@pytest.mark.parametrize("status", ["limited_blocks", "unavailable", None])
def test_a_flat_surface_transmission_does_not_override_missing_pmf_block_support(tmp_path, status):
    result = quality_result(tmp_path, block_status=status)
    assert result["plateau_detected"]
    assert result["rate_estimate_s_inv"] > 0
    assert result["rate_s_inv"] is None
    assert result["status"] == "pilot_requires_convergence"
    assert "pmf_correlation_blocks_not_resolved" in result["rate_quality"]["reasons"]


def test_user_precision_target_gates_reporting_without_changing_sampling_estimates(tmp_path):
    loose = quality_result(tmp_path / "loose", draws=[1., 5., 30., 100.], target=10.)
    strict = quality_result(tmp_path / "strict", draws=[1., 5., 30., 100.], target=.2)
    assert loose["rate_quality"]["passed"]
    assert strict["rate_estimate_s_inv"] == loose["rate_estimate_s_inv"]
    assert strict["rate_standard_error_s_inv"] == loose["rate_standard_error_s_inv"]
    assert strict["rate_s_inv"] is None
    assert "relative_standard_error_exceeds_target" in strict["rate_quality"]["reasons"]


@pytest.mark.parametrize("parents,chains,reason", [
    (12, 4, "fewer_than_twenty_nonzero_flux_parent_configurations"),
    (80, 1, "fewer_than_two_independent_parent_chains"),
])
def test_surface_reporting_requires_independent_parent_support(tmp_path, parents, chains, reason):
    result = quality_result(tmp_path, parents=parents, chains=chains)
    assert result["rate_estimate_s_inv"] > 0
    assert result["rate_s_inv"] is None
    assert reason in result["rate_quality"]["reasons"]
