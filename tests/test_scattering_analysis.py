"""Independent statistical references for specified-state scattering summaries."""

import copy
import json
import math

import numpy as np
import pytest

from pyrpmd.analysis.scattering import summarize_fixed_energy_scattering


AREA = 4.0 * math.pi
ENERGY = 0.02
STATE = {"method": "ne_rpmd", "vibrational_quanta": [[], [0]], "rotation": "j0"}


def records(statuses, *, parents=None, state=STATE):
    result = []
    for i, status in enumerate(statuses):
        result.append({
            "trajectory_id": i,
            "parent_id": 0 if parents is None else parents[i],
            "status": status,
            "channel": "OH+OH" if status == "product" else None,
            "initial_state": copy.deepcopy(state),
            "launch": {"sampling": "fixed_energy", "collision_energy_au": ENERGY,
                       "area_au": AREA},
        })
    return result


def summarize(data, **kwargs):
    options = {"area_bohr2": AREA, "initialization_mode": "independent"}
    options.update(kwargs)
    return summarize_fixed_energy_scattering(data, **options)


def test_unknown_outcome_bounds_keep_censored_and_failed_launches_in_denominator():
    data = records(["product", "product", "nonreactive", "unresolved", "failed"])
    data[1]["channel"] = "H2+O2"
    result = summarize(data, channels=["OH+OH", "H2+O2", "H2O+O"])
    total = result["total_reaction"]
    assert total["count"] == 2
    assert total["event_fraction"] == pytest.approx(2 / 5)
    assert total["cross_section_bohr2"] is None
    assert total["observed_lower_bohr2"] == pytest.approx(AREA * 2 / 5)
    assert total["possible_upper_bohr2"] == pytest.approx(AREA * 4 / 5)
    assert total["iid_wilson95_bohr2"] is None
    assert total["uncertainty"]["reason"] == "unresolved_or_failed_outcomes"
    assert result["unknown_outcomes"] == {
        "unresolved": 1, "failed": 1, "included_in_denominator": True,
        "treated_as_nonreactive": False,
    }
    for channel in ("OH+OH", "H2+O2"):
        assert result["channels"][channel]["observed_lower_bohr2"] == pytest.approx(AREA / 5)
        assert result["channels"][channel]["possible_upper_bohr2"] == pytest.approx(AREA * 3 / 5)
    assert result["channels"]["H2O+O"]["observed_lower_bohr2"] == 0
    assert result["channels"]["H2O+O"]["possible_upper_bohr2"] == pytest.approx(AREA * 2 / 5)
    json.dumps(result, allow_nan=False)


def test_iid_wilson_matches_score_interval_reference_including_zero_events():
    data = records(["product"] * 20 + ["nonreactive"] * 80)
    result = summarize(data, channels=["unseen"])
    entry = result["total_reaction"]
    assert entry["cross_section_bohr2"] == pytest.approx(AREA * 0.2)
    # The 95% binomial score interval for 20 successes in 100 independent trials.
    np.testing.assert_allclose(np.array(entry["iid_wilson95_bohr2"]) / AREA,
                               [0.1333669333310325, 0.2888291655931589])
    zero = result["channels"]["unseen"]
    assert zero["iid_wilson95_bohr2"][0] == pytest.approx(0, abs=1e-14)
    assert zero["iid_wilson95_bohr2"][1] > 0
    assert entry["parent_cluster_bootstrap95_bohr2"] is None
    assert entry["uncertainty"]["method"] == "iid_wilson"
    assert entry["uncertainty"]["assumed_independent_units"] == 100


def test_complete_initialization_independence_must_be_explicit_not_inferred_from_ids():
    data = records(["product", "nonreactive"] * 10, parents=list(range(20)))
    result = summarize(data, initialization_mode="unknown")
    entry = result["total_reaction"]
    assert entry["cross_section_bohr2"] == pytest.approx(AREA / 2)
    assert entry["iid_wilson95_bohr2"] is None
    assert entry["parent_cluster_bootstrap95_bohr2"] is None
    assert entry["uncertainty"]["reason"] == "initialization_independence_not_established"


def test_parent_chain_intervals_require_enough_chains_and_do_not_fall_back_to_iid():
    data = records(["product", "nonreactive"] * 20, parents=[0] * 20 + [1] * 20)
    entry = summarize(data, initialization_mode="parent_chain")["total_reaction"]
    assert entry["cross_section_bohr2"] == pytest.approx(AREA / 2)
    assert entry["iid_wilson95_bohr2"] is None
    assert entry["parent_cluster_bootstrap95_bohr2"] is None
    assert entry["uncertainty"]["reason"] == "too_few_independent_parent_clusters"
    assert entry["uncertainty"]["assumed_independent_units"] == 2


def test_cluster_resampling_keeps_whole_unequal_size_chains_and_ratio_denominator():
    # Completely correlated outcomes within eight independent parents. Averaging
    # parent means would give 1/2, while the launch-weighted estimate is 26/36.
    sizes = np.arange(1, 9)
    statuses = [status for parent, size in enumerate(sizes)
                for status in (["nonreactive"] if parent < 4 else ["product"]) * size]
    parents = np.repeat(np.arange(8), sizes).tolist()
    data = records(statuses, parents=parents)
    kwargs = {"initialization_mode": "parent_chain", "bootstrap_seed": 53,
              "bootstrap_replicates": 400}
    result = summarize(data, **kwargs)
    entry = result["total_reaction"]
    assert entry["cross_section_bohr2"] == pytest.approx(AREA * 26 / 36)
    # Independent reference enumerates a bootstrap draw by expanding its actual
    # child outcomes, instead of using the implementation's cluster count matrix.
    rng = np.random.default_rng(np.random.SeedSequence([53, 19, 0]))
    groups = [np.flatnonzero(np.array(parents) == parent) for parent in range(8)]
    reactive = np.array([status == "product" for status in statuses])
    draws = []
    for _ in range(400):
        sample = np.concatenate([groups[parent] for parent in rng.integers(0, 8, 8)])
        draws.append(AREA * reactive[sample].mean())
    np.testing.assert_allclose(entry["parent_cluster_bootstrap95_bohr2"],
                               np.quantile(draws, [0.025, 0.975]))
    assert entry["iid_wilson95_bohr2"] is None
    assert entry["uncertainty"]["method"] == "parent_cluster_percentile_bootstrap"
    assert result == summarize(data, **kwargs)
    assert (entry["parent_cluster_bootstrap95_bohr2"]
            == result["initial_state_groups"][0]["total_reaction"]["parent_cluster_bootstrap95_bohr2"])


@pytest.mark.parametrize("statuses", [["nonreactive"] * 16, ["product", "nonreactive"] * 8])
def test_degenerate_cluster_bootstrap_is_not_a_false_zero_width_error_bar(statuses):
    data = records(statuses, parents=np.repeat(np.arange(8), 2).tolist())
    entry = summarize(data, initialization_mode="parent_chain")["total_reaction"]
    assert entry["parent_cluster_bootstrap95_bohr2"] is None
    assert entry["uncertainty"]["reason"] == "no_between_parent_fraction_variation"


def test_missing_parent_identity_is_explicit_and_never_assumed_independent():
    data = records(["product", "nonreactive"] * 8, parents=list(range(16)))
    del data[0]["parent_id"]
    entry = summarize(data, initialization_mode="parent_chain")["total_reaction"]
    assert entry["parent_cluster_bootstrap95_bohr2"] is None
    assert "missing_parent_ids" in entry["uncertainty"]["unavailability_reasons"]
    assert entry["uncertainty"]["assumed_independent_units"] is None


def test_canonical_state_grouping_preserves_conditional_denominators_and_mixture_metadata():
    data = records(["product", "nonreactive", "product", "unresolved"])
    # Key order must not split the same state into different groups.
    data[1]["initial_state"] = dict(reversed(list(STATE.items())))
    excited = {**STATE, "vibrational_quanta": [[], [1]]}
    data[2]["initial_state"] = excited
    data[3]["initial_state"] = excited
    result = summarize(data)
    groups = {group["initial_state"]["vibrational_quanta"][1][0]: group
              for group in result["initial_state_groups"]}
    assert len(groups) == 2
    assert groups[0]["trajectories"] == groups[1]["trajectories"] == 2
    assert groups[0]["sampling_fraction"] == groups[1]["sampling_fraction"] == 0.5
    assert groups[0]["total_reaction"]["cross_section_bohr2"] == pytest.approx(AREA / 2)
    assert groups[0]["total_reaction"]["iid_wilson95_bohr2"] is not None
    assert groups[1]["total_reaction"]["cross_section_bohr2"] is None
    assert groups[1]["total_reaction"]["possible_upper_bohr2"] == pytest.approx(AREA)
    assert result["initial_state_scope"] == "sampled_initial_state_mixture"
    assert result["initial_state"] is None
    assert groups[0]["initial_state_id"] != groups[1]["initial_state_id"]
    assert "mixed_initial_states_require_stratified_uncertainty" in (
        result["total_reaction"]["uncertainty"]["unavailability_reasons"])


def test_complete_mixture_does_not_claim_identically_distributed_bernoulli_trials():
    data = records(["product", "nonreactive"])
    data[1]["initial_state"]["vibrational_quanta"][1] = [1]
    result = summarize(data)
    assert result["total_reaction"]["cross_section_bohr2"] == pytest.approx(AREA / 2)
    assert result["total_reaction"]["iid_wilson95_bohr2"] is None
    assert result["total_reaction"]["uncertainty"]["reason"] == (
        "mixed_initial_states_require_stratified_uncertainty")


def test_classifier_and_cutoff_gates_distinguish_candidate_counts_from_disk_bounds():
    data = records(["product", "nonreactive"])
    candidate = summarize(data, classifier_validated=False)["total_reaction"]
    assert candidate["count"] == 1
    assert candidate["candidate_only"] is True
    assert candidate["observed_lower_bohr2"] is None
    assert candidate["possible_upper_bohr2"] is None
    assert candidate["cross_section_bohr2"] is None
    disk = summarize(data, cutoff_validated=False)
    assert disk["bounds_scope"] == "sampled_impact_parameter_disk"
    assert disk["total_reaction"]["observed_lower_bohr2"] == pytest.approx(AREA / 2)
    assert disk["total_reaction"]["possible_upper_bohr2"] == pytest.approx(AREA / 2)
    assert disk["total_reaction"]["cross_section_bohr2"] is None
    assert disk["total_reaction"]["iid_wilson95_bohr2"] is None
    data[0]["classification_unvalidated"] = True
    assert summarize(data)["total_reaction"]["observed_lower_bohr2"] is None


def test_common_positive_weights_cancel_and_do_not_rescale_the_cross_section():
    data = records(["product", "nonreactive"])
    assert summarize(data, weights=[7, 7]) == summarize(data)


@pytest.mark.parametrize("kwargs,match", [
    ({"initialization_mode": "thermal"}, "initialization_mode"),
    ({"impact_sampling": "uniform_b"}, "uniform_area"),
    ({"weights": [1, 2]}, "equal"),
    ({"weights": [0, 0]}, "positive"),
    ({"weights": [1, np.nan]}, "finite"),
    ({"weights": [1]}, "weights"),
    ({"area_bohr2": np.inf}, "area_bohr2"),
    ({"collision_energy_au": -1}, "collision_energy_au"),
    ({"channels": [None]}, "channel"),
    ({"channels": "OH+OH"}, "channel"),
    ({"classifier_validated": "yes"}, "boolean"),
    ({"bootstrap_replicates": 1}, "at least 100"),
    ({"min_parent_clusters": 1}, "min_parent_clusters"),
])
def test_unsupported_or_invalid_estimator_inputs_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        summarize(records(["product", "nonreactive"]), **kwargs)


@pytest.mark.parametrize("change,match", [
    (lambda r: r.update(status="timeout"), "status"),
    (lambda r: r.update(channel=None), "channel"),
    (lambda r: r.update(channel=""), "channel"),
    (lambda r: r["launch"].update(sampling="thermal_flux"), "thermal"),
    (lambda r: r["launch"].update(collision_energy_au=2 * ENERGY), "common fixed"),
    (lambda r: r["launch"].update(area_au=2 * AREA), "common impact"),
    (lambda r: r.update(initial_state={"energy": np.nan}), "JSON-safe"),
    (lambda r: r.update(parent_id=[]), "parent_id"),
])
def test_invalid_records_and_mixed_sampling_measures_are_rejected(change, match):
    data = records(["product", "product"])
    change(data[1])
    with pytest.raises(ValueError, match=match):
        summarize(data)


def test_duplicate_launches_are_not_silently_counted_twice():
    data = records(["product", "nonreactive"])
    data[1]["trajectory_id"] = data[0]["trajectory_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(data)


def test_explicit_energy_and_state_fallback_support_minimal_offline_records():
    data = [{"status": "product", "channel": "A+B"}, {"status": "failed"}]
    with pytest.raises(ValueError, match="collision_energy_au"):
        summarize(data)
    result = summarize(data, collision_energy_au=ENERGY, initial_state=STATE)
    assert result["initial_state"] == STATE
    assert result["collision_energy_au"] == ENERGY
    assert result["total_reaction"]["possible_upper_bohr2"] == pytest.approx(AREA)


def test_empty_ensemble_has_no_scattering_estimate():
    with pytest.raises(ValueError, match="zero trajectories"):
        summarize([])
