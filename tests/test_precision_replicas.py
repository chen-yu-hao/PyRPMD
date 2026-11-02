"""Acceptance boundaries for frozen complete-replica rate evidence."""

from dataclasses import replace
import json
import math

import numpy as np
import pytest

from pyrpmd.precision import (
    CheckSchedule,
    NumericalRequirement,
    PrecisionSpec,
    PrecisionTarget,
    RateEvidence,
    SamplingDesign,
    ScientificIdentity,
    assess_precision,
    evidence_from_replicas,
    freeze_production,
)


def _spec(*, checks=(2, 4), max_epochs=2, numerical=None):
    if numerical is None:
        numerical = (NumericalRequirement("beads", 16, 0.5),)
    return PrecisionSpec(
        ScientificIdentity("thermal-model-and-estimator", 8, 0.2, "frozen-bias"),
        PrecisionTarget(0.5), CheckSchedule(checks, max_epochs), 1e6,
        numerical=numerical, seed=7281,
    )


def _sampling():
    return SamplingDesign(2, 1000, 2, 20, 10, 100,
                          umbrella_burnin_steps=8, parent_burnin_steps=4,
                          parent_interval_steps=2)


def _frozen(spec, **kwargs):
    return freeze_production(spec, _sampling(), **kwargs)


def _records(frozen, *, n=4, rates=None, qtst=None):
    rates = {} if rates is None else rates
    qtst = {} if qtst is None else qtst
    default_rates = [1.0, 1.002, 0.999, 1.001, 1.0005, 0.9995]
    records = []
    for job in frozen.jobs:
        index = job["replica_index"]
        if index >= n:
            continue
        rate = rates.get(job["group"], default_rates)[index]
        q = qtst.get(job["group"], [2.0] * n)[index]
        diagnostics = RateEvidence(
            job["scientific_identity"], pmf_ready=True, overlap_connected=True,
            minimum_umbrella_ess=100, independent_parent_chains=2,
            effective_parent_blocks=20, plateau_established=True,
            energy_complete=True, maximum_energy_error_kbt=0.001,
        )
        records.append({
            "job_id": job["job_id"], "source_id": "source:" + job["job_id"],
            "source_family_id": "family:" + job["job_id"],
            "epoch_id": frozen.epoch_id, "design_id": frozen.design_id,
            "seed": job["seed"], "sampling_design_id": job["sampling_design_id"],
            "independent_initialization": True,
            "log_qtst": math.log(q), "log_kappa": math.log(rate / q),
            "diagnostics": diagnostics,
        })
    return records


def _job_map(frozen):
    return {j["job_id"]: j for j in frozen.jobs}


def test_full_frozen_records_can_establish_statistical_and_numerical_targets():
    spec = _spec()
    frozen = _frozen(spec)
    evidence = evidence_from_replicas(spec, frozen, _records(frozen))
    result = assess_precision(spec, evidence, frozen=frozen)
    assert result["accepted"]
    assert result["statistical_target_met"]
    assert result["numerical_convergence_established"]
    assert result["diagnostics_passed"]
    json.dumps(result, allow_nan=False)


def test_replica_rate_and_numerical_ratio_use_arithmetic_means():
    spec = _spec(checks=(2,))
    frozen = _frozen(spec)
    requirement = spec.numerical[0].requirement_id
    records = _records(frozen, n=2, rates={"baseline": [0.5, 1.5], requirement: [1., 1.]})
    evidence = evidence_from_replicas(spec, frozen, records)
    assert evidence.rate == pytest.approx(1.0)
    assert evidence.rate != pytest.approx(math.sqrt(0.75))
    assert evidence.numerical[0].log_ratio == pytest.approx(0.0)
    assert evidence.numerical[0].standard_error == pytest.approx(0.5)
    assert evidence.degrees_of_freedom == 1
    # A zero estimated difference with broad uncertainty is not equivalence.
    assessment = assess_precision(spec, evidence, frozen=frozen)
    assert not assessment["numerical_convergence_established"]


def test_replica_covariance_and_contrast_match_direct_arithmetic_statistics():
    spec = _spec()
    frozen = _frozen(spec)
    requirement = spec.numerical[0].requirement_id
    baseline = np.array([0.4, 0.6, 0.6, 0.4])
    candidate = np.array([0.45, 0.55, 0.65, 0.35])
    evidence = evidence_from_replicas(
        spec, frozen, _records(frozen,
            rates={"baseline": baseline, requirement: candidate},
            qtst={"baseline": [1., 2., 3., 4.]}),
    )
    direct_se = np.std(baseline, ddof=1) / math.sqrt(len(baseline))
    relative_variance = (evidence.pmf_log_standard_error ** 2
                         + evidence.recrossing_log_standard_error ** 2
                         + 2 * evidence.covariance)
    assert evidence.covariance < 0
    assert relative_variance == pytest.approx((direct_se / np.mean(baseline)) ** 2)
    influence = candidate / np.mean(candidate) - baseline / np.mean(baseline)
    assert evidence.numerical[0].standard_error == pytest.approx(np.std(influence, ddof=1) / 2)
    assert evidence.numerical[0].log_ratio == pytest.approx(math.log(np.mean(candidate) / np.mean(baseline)))
    assert evidence.numerical[0].degrees_of_freedom == 3


@pytest.mark.parametrize("field,value", [
    ("independent_parent_chains", None), ("independent_parent_chains", 0),
    ("effective_parent_blocks", None), ("effective_parent_blocks", 0),
])
def test_candidate_parent_evidence_cannot_be_replaced_by_baseline_sampling(field, value):
    spec = _spec()
    frozen = _frozen(spec)
    jobs = _job_map(frozen)
    records = _records(frozen)
    for record in records:
        if jobs[record["job_id"]]["group"] != "baseline":
            record["diagnostics"] = replace(record["diagnostics"], **{field: value})
    evidence = evidence_from_replicas(spec, frozen, records)
    assert not evidence.numerical[0].diagnostics_passed
    assessment = assess_precision(spec, evidence, frozen=frozen)
    assert not assessment["accepted"]
    assert not assessment["numerical_convergence_established"]
    assert not assessment["diagnostics_passed"]


@pytest.mark.parametrize("field,value", [
    ("seed", 123), ("sampling_design_id", "stale-sampling"),
    ("epoch_id", "stale-epoch"), ("design_id", "stale-design"),
    ("independent_initialization", False),
])
def test_receipt_must_match_frozen_seed_sampling_and_epoch(field, value):
    spec = _spec()
    frozen = _frozen(spec)
    records = _records(frozen)
    records[0][field] = value
    with pytest.raises(ValueError, match="frozen independent design"):
        evidence_from_replicas(spec, frozen, records)


@pytest.mark.parametrize("case", ["duplicate_source", "duplicate_family", "pilot_source",
                                  "pilot_family", "family_alias_of_pilot_source"])
def test_production_sources_and_families_cannot_reuse_pilot_or_replica_data(case):
    spec = _spec()
    frozen = _frozen(spec, pilot_source_ids=("pilot-source",),
                     pilot_source_family_ids=("pilot-family",))
    records = _records(frozen)
    if case == "duplicate_source":
        records[1]["source_id"] = records[0]["source_id"]
    elif case == "duplicate_family":
        records[1]["source_family_id"] = records[0]["source_family_id"]
    elif case == "pilot_source":
        records[0]["source_id"] = "pilot-source"
    elif case == "pilot_family":
        records[0]["source_family_id"] = "pilot-family"
    else:
        records[0]["source_family_id"] = "pilot-source"
    with pytest.raises(ValueError, match="overlapping or pilot"):
        evidence_from_replicas(spec, frozen, records)


@pytest.mark.parametrize("group", ["baseline", "candidate"])
def test_missing_interior_replica_cannot_be_filtered_out(group):
    spec = _spec()
    frozen = _frozen(spec)
    selected = "baseline" if group == "baseline" else spec.numerical[0].requirement_id
    jobs = _job_map(frozen)
    records = [r for r in _records(frozen)
               if not (jobs[r["job_id"]]["group"] == selected and
                       jobs[r["job_id"]]["replica_index"] == 1)]
    with pytest.raises(ValueError, match="unfiltered prefix"):
        evidence_from_replicas(spec, frozen, records)


def test_failed_diagnostics_are_retained_instead_of_selecting_supported_replicas():
    spec = _spec()
    frozen = _frozen(spec)
    records = _records(frozen)
    records[1]["diagnostics"] = replace(records[1]["diagnostics"], pmf_ready=False)
    evidence = evidence_from_replicas(spec, frozen, records)
    assert len(evidence.replica_ids) == 4
    assert evidence.pmf_ready is False
    assert not assess_precision(spec, evidence, frozen=frozen)["accepted"]
    records[1]["log_kappa"] = None
    with pytest.raises(ValueError, match="finite"):
        evidence_from_replicas(spec, frozen, records)


@pytest.mark.parametrize("field,value", [("plateau_established", False),
                                        ("maximum_energy_error_kbt", 1.0),
                                        ("minimum_umbrella_ess", None)])
def test_numerical_convergence_requires_a_valid_baseline_as_well_as_candidate(field, value):
    spec = _spec()
    frozen = _frozen(spec)
    records = _records(frozen)
    records[0]["diagnostics"] = replace(records[0]["diagnostics"], **{field: value})
    evidence = evidence_from_replicas(spec, frozen, records)
    result = assess_precision(spec, evidence, frozen=frozen)
    assert not result["accepted"]
    assert not result["numerical_convergence_established"]
    assert "numerical_convergence_not_established" in result["conditions"]


def test_arrival_order_does_not_change_frozen_replica_prefix_or_statistics():
    spec = _spec()
    frozen = _frozen(spec)
    records = _records(frozen)
    ordered = evidence_from_replicas(spec, frozen, records)
    reversed_records = evidence_from_replicas(spec, frozen, list(reversed(records)))
    assert ordered.to_dict() == reversed_records.to_dict()


@pytest.mark.parametrize("field,value", [
    ("epoch_id", "previous-production-epoch"), ("design_id", "previous-frozen-design"),
    ("replica_ids", ()), ("source_ids", ()), ("diagnostics_passed", None),
    ("degrees_of_freedom", 1000),
])
def test_numerical_evidence_must_belong_to_same_frozen_look(field, value):
    spec = _spec()
    frozen = _frozen(spec)
    evidence = evidence_from_replicas(spec, frozen, _records(frozen))
    comparison = replace(evidence.numerical[0], **{field: value})
    changed = replace(evidence, numerical=(comparison,))
    result = assess_precision(spec, changed, frozen=frozen)
    assert not result["accepted"]
    assert not result["numerical_convergence_established"]
    assert not result["numerical"][0]["estimable"]


def test_different_numerical_replica_count_is_not_an_equivalence_look():
    spec = _spec()
    frozen = _frozen(spec)
    jobs = _job_map(frozen)
    records = [r for r in _records(frozen)
               if jobs[r["job_id"]]["group"] != "baseline" or jobs[r["job_id"]]["replica_index"] < 2]
    evidence = evidence_from_replicas(spec, frozen, records)
    assert len(evidence.replica_ids) == 2
    assert not evidence.numerical
    assert not assess_precision(spec, evidence, frozen=frozen)["numerical_convergence_established"]


def test_unscheduled_complete_replica_count_does_not_expose_inferential_interval():
    spec = _spec(checks=(2, 4))
    frozen = _frozen(spec)
    evidence = evidence_from_replicas(spec, frozen, _records(frozen, n=3))
    result = assess_precision(spec, evidence, frozen=frozen)
    assert result["status"] == "check_not_due"
    assert not result["accepted"]
    assert not result["statistical_target_met"]
    assert result["interval"]["interval"] is None
    assert not result["numerical_convergence_established"]


def test_alpha_is_allocated_across_all_epochs_looks_and_numerical_claims():
    spec = _spec(checks=(2, 4, 6), max_epochs=3,
                 numerical=(NumericalRequirement("beads", 16, 0.5),
                            NumericalRequirement("timestep", 0.1, 0.5)))
    assert spec.alpha_per_claim == pytest.approx(0.05 / (3 * 3 * 3))
    epochs = [_frozen(spec, epoch_index=i) for i in range(3)]
    assert len({j["seed"] for epoch in epochs for j in epoch.jobs}) == sum(len(e.jobs) for e in epochs)
    for epoch in epochs:
        evidence = evidence_from_replicas(spec, epoch, _records(epoch, n=2))
        result = assess_precision(spec, evidence, frozen=epoch)
        assert result["alpha_per_claim"] == pytest.approx(0.05 / 27)
        assert result["interval"]["confidence_level"] == pytest.approx(1 - 0.05 / 27)
    with pytest.raises(ValueError, match="alpha allocation"):
        _frozen(spec, epoch_index=3)


def test_timestep_comparison_keeps_physical_durations_fixed():
    spec = _spec(numerical=(NumericalRequirement("timestep", 0.1, 0.5),))
    frozen = _frozen(spec)
    base = next(j for j in frozen.jobs if j["group"] == "baseline")
    other = next(j for j in frozen.jobs if j["group"] != "baseline")
    for field in ("umbrella_steps", "child_steps", "umbrella_burnin_steps",
                  "parent_burnin_steps", "parent_interval_steps"):
        assert base["sampling"][field] * base["identity"]["timestep_au"] == pytest.approx(
            other["sampling"][field] * other["identity"]["timestep_au"],
        )
    assert base["sampling_design_id"] != other["sampling_design_id"]
    assert base["scientific_identity"] != other["scientific_identity"]
    evidence_from_replicas(spec, frozen, _records(frozen))


@pytest.mark.parametrize("change", ["epoch", "seed_collision", "sampling"])
def test_forged_or_stale_frozen_design_cannot_certify_production(change):
    spec = _spec()
    frozen = _frozen(spec)
    if change == "epoch":
        changed = replace(frozen, epoch_index=spec.checks.max_epochs)
    elif change == "sampling":
        changed = replace(frozen, sampling=replace(frozen.sampling, umbrella_steps=2000))
    else:
        jobs = frozen.jobs
        for job in jobs:
            job["seed"] = 7
        changed = replace(frozen, jobs_json=json.dumps(jobs, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError):
        evidence_from_replicas(spec, changed, _records(changed))
