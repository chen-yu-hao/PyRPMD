"""Scientific acceptance cases for thermal RPMD precision control."""

from dataclasses import replace

import pytest

from pyrpmd.precision import (
    CheckSchedule,
    NumericalEvidence,
    NumericalRequirement,
    PrecisionSpec,
    PrecisionTarget,
    RateEvidence,
    SamplingDesign,
    ScientificIdentity,
    assess_precision,
    freeze_production,
)
from pyrpmd.precision_planning import PrecisionController, recommend_supplement


@pytest.fixture
def identity():
    return ScientificIdentity("fixed-PES-T300-CV-normalization", 32, 1.0, "frozen-bias-0")


@pytest.fixture
def spec(identity):
    return PrecisionSpec(
        identity=identity,
        target=PrecisionTarget(relative_half_width=0.1),
        checks=CheckSchedule((4, 8), max_epochs=2),
        max_cost=1000,
        numerical=(NumericalRequirement("beads", 64, 0.2),),
        seed=412,
    )


@pytest.fixture
def sampling():
    return SamplingDesign(
        umbrella_streams=4, umbrella_steps=100,
        parent_chains=4, parent_samples=20,
        child_pairs=8, child_steps=100,
        umbrella_burnin_steps=20, parent_burnin_steps=20,
        parent_interval_steps=5,
    )


@pytest.fixture
def pilot(identity):
    return RateEvidence(
        scientific_identity=identity.identity_id,
        source_ids=("pilot-source",),
        rate=1.0,
        pmf_log_standard_error=0.01,
        recrossing_log_standard_error=0.02,
        components_independent=True,
        degrees_of_freedom=19,
        pmf_ready=True,
        overlap_connected=True,
        minimum_umbrella_ess=100,
        independent_parent_chains=4,
        effective_parent_blocks=8,
        parent_log_variance=0.0003,
        child_log_variance=0.0001,
        variance_attribution_validated=True,
        plateau_established=True,
        energy_complete=True,
        maximum_energy_error_kbt=0.01,
    )


@pytest.fixture
def costs():
    return {
        "add_umbrella_streams": 20,
        "add_parent_chains": 30,
        "add_children": 10,
        "extend_child_horizon": 15,
        "compare_beads": 100,
        "compare_timestep": 80,
    }


def production(spec, frozen, pilot, count=4, numerical_error=0.001):
    ids = tuple(job["job_id"] for job in frozen.jobs if job["group"] == "baseline")[:count]
    base_sources = tuple(f"production-{frozen.epoch_id}-{index}" for index in range(count))
    comparisons = tuple(
        NumericalEvidence(
            requirement_id=requirement.requirement_id,
            baseline_identity=spec.identity.identity_id,
            candidate_identity=spec.identity.refined(requirement.axis, requirement.value).identity_id,
            log_ratio=0.0,
            standard_error=numerical_error,
            degrees_of_freedom=count - 1,
            covariance_accounted=True,
            epoch_id=frozen.epoch_id,
            design_id=frozen.design_id,
            replica_ids=tuple(job["job_id"] for job in frozen.jobs
                              if job["group"] == requirement.requirement_id)[:count],
            source_ids=tuple(f"comparison-{frozen.epoch_id}-{requirement.requirement_id}-{index}"
                             for index in range(count)),
            diagnostics_passed=True,
        )
        for requirement in spec.numerical
    )
    return replace(
        pilot,
        phase="production",
        source_ids=base_sources + tuple(source for comparison in comparisons for source in comparison.source_ids),
        source_family_ids=tuple("family-" + source for source in (
            base_sources + tuple(source for comparison in comparisons for source in comparison.source_ids))),
        pmf_log_standard_error=0.001,
        recrossing_log_standard_error=0.001,
        degrees_of_freedom=count - 1,
        epoch_id=frozen.epoch_id,
        design_id=frozen.design_id,
        replica_ids=ids,
        independent_from_pilot=True,
        numerical=comparisons,
    )


def recommendation(spec, pilot, sampling, costs, remaining=1000):
    return recommend_supplement(spec, pilot, sampling, costs=costs, remaining_cost=remaining)


def test_pmf_limited_case_adds_independent_umbrella_sampling(spec, pilot, sampling, costs):
    limited = replace(pilot, pmf_log_standard_error=0.3,
                      recrossing_log_standard_error=0.01)
    proposed = recommendation(spec, limited, sampling, costs)
    assert proposed["actions"][0]["kind"] == "add_umbrella_streams"
    assert proposed["estimated_cost"] == costs["add_umbrella_streams"]
    assert proposed["scientific_identity"] == spec.identity.identity_id


@pytest.mark.parametrize("missing", [False, True])
def test_parent_limited_case_cannot_be_cured_by_enormous_children(
        spec, pilot, sampling, costs, missing):
    limited = replace(
        pilot,
        independent_parent_chains=None if missing else 1,
        effective_parent_blocks=1,
        recrossing_log_standard_error=0.3,
        parent_log_variance=0.089,
        child_log_variance=0.001,
    )
    enormous_children = replace(sampling, parent_chains=1, child_pairs=1_000_000)
    proposed = recommendation(spec, limited, enormous_children, costs)
    assert proposed["actions"][0]["kind"] == "add_parent_chains"


def test_child_horizon_limited_case_extends_child_time(spec, pilot, sampling, costs):
    limited = replace(pilot, plateau_established=False, plateau_failure="horizon")
    proposed = recommendation(spec, limited, sampling, costs)
    assert proposed["actions"][0]["kind"] == "extend_child_horizon"


@pytest.mark.parametrize("cause", ["sampling", "unknown"])
def test_noisy_or_unknown_plateau_does_not_blindly_extend_time(spec, pilot, sampling, costs, cause):
    limited = replace(pilot, plateau_established=False, plateau_failure=cause)
    proposed = recommendation(spec, limited, sampling, costs)
    assert all(action["kind"] != "extend_child_horizon" for action in proposed["actions"])


def test_child_limited_case_requires_valid_variance_attribution(spec, pilot, sampling, costs):
    limited = replace(
        pilot,
        pmf_log_standard_error=0.001,
        recrossing_log_standard_error=0.2,
        parent_log_variance=0.001,
        child_log_variance=0.039,
        variance_attribution_validated=True,
    )
    proposed = recommendation(spec, limited, sampling, costs)
    assert proposed["actions"][0]["kind"] == "add_children"
    unvalidated = recommendation(spec, replace(limited, variance_attribution_validated=False),
                                 sampling, costs)
    assert unvalidated["actions"][0]["kind"] == "add_parent_chains"


@pytest.mark.parametrize("diagnostic", ["overlap", "ess"])
def test_weak_overlap_or_effective_sampling_targets_umbrella(spec, pilot, sampling, costs, diagnostic):
    limited = replace(pilot, overlap_connected=False) if diagnostic == "overlap" else replace(
        pilot, minimum_umbrella_ess=1)
    assert recommendation(spec, limited, sampling, costs)["actions"][0]["kind"] == "add_umbrella_streams"


def test_energy_failure_requests_controlled_timestep_identity(spec, pilot, sampling, costs):
    spec = replace(spec, numerical=(NumericalRequirement("timestep", 0.5, 0.2),))
    unstable = replace(pilot, maximum_energy_error_kbt=1.0)
    proposed = recommendation(spec, unstable, sampling, costs)
    assert proposed["actions"][0]["kind"] == "compare_timestep"


def test_unaffordable_supplement_preserves_unestablished_convergence(spec, pilot, sampling, costs):
    proposed = recommendation(spec, replace(pilot, pmf_log_standard_error=0.3), sampling, costs, remaining=0)
    assert proposed["actions"] == []
    assert proposed["status"] == "budget_exhausted"
    assert proposed["reasons"]


def test_unknown_errors_are_not_zero_or_a_justification_for_blind_children(spec, pilot, sampling, costs):
    unknown = replace(pilot, pmf_log_standard_error=None, recrossing_log_standard_error=None,
                      parent_log_variance=None, child_log_variance=None,
                      variance_attribution_validated=False)
    report = assess_precision(spec, unknown)
    assert report["uncertainty"]["log_rate_standard_error"] is None
    assert report["interval"]["interval"] is None
    assert report["accepted"] is False
    assert "error_unestimable" in report["conditions"]
    assert recommendation(spec, unknown, sampling, costs)["actions"] == []


def test_missing_pmf_or_correlation_cannot_certify_production(spec, pilot, sampling):
    frozen = freeze_production(spec, sampling, pilot_source_ids=pilot.source_ids)
    evidence = production(spec, frozen, pilot)
    for changed in (
        replace(evidence, pmf_log_standard_error=None),
        replace(evidence, recrossing_log_standard_error=None),
        replace(evidence, covariance=None, components_independent=False),
        replace(evidence, degrees_of_freedom=None),
    ):
        report = assess_precision(spec, changed, frozen=frozen)
        assert report["statistical_target_met"] is False
        assert report["accepted"] is False
        assert "error_unestimable" in report["conditions"]


def test_nonsignificant_noisy_difference_is_not_numerical_equivalence(spec, pilot, sampling):
    frozen = freeze_production(spec, sampling, pilot_source_ids=pilot.source_ids)
    # Exactly zero estimated difference still fails when its interval is wide.
    evidence = production(spec, frozen, pilot, numerical_error=0.5)
    report = assess_precision(spec, evidence, frozen=frozen)
    assert report["statistical_target_met"] is True
    assert report["numerical_convergence_established"] is False
    assert report["accepted"] is False
    assert "numerical_convergence_not_established" in report["conditions"]


def test_numerical_acceptance_requires_matching_independent_production_provenance(spec, pilot, sampling):
    frozen = freeze_production(spec, sampling, pilot_source_ids=pilot.source_ids)
    evidence = production(spec, frozen, pilot)
    original = evidence.numerical[0]
    for changed in (
        replace(original, epoch_id=None),
        replace(original, design_id="different-design"),
        replace(original, replica_ids=()),
        replace(original, source_ids=("unaccounted-source",)),
        replace(original, diagnostics_passed=False),
    ):
        report = assess_precision(spec, replace(evidence, numerical=(changed,)), frozen=frozen)
        assert report["numerical_convergence_established"] is False
        assert report["accepted"] is False


def test_precise_pilot_cannot_satisfy_production_target(spec, pilot):
    excellent_pilot = replace(pilot, pmf_log_standard_error=0.0001, recrossing_log_standard_error=0.0001)
    report = assess_precision(spec, excellent_pilot)
    assert report["statistical_target_met"] is False
    assert report["accepted"] is False
    assert "pilot_requires_independent_production" in report["conditions"]


def test_production_planner_does_not_adapt_a_frozen_design(spec, pilot, sampling, costs):
    frozen = freeze_production(spec, sampling, pilot_source_ids=pilot.source_ids)
    evidence = replace(production(spec, frozen, pilot), pmf_log_standard_error=0.3)
    proposed = recommendation(spec, evidence, sampling, costs)
    assert proposed["actions"] == []
    assert proposed["status"] == "fixed_design_requires_new_pilot"


def test_only_scheduled_independent_production_check_can_accept(spec, pilot, sampling):
    frozen = freeze_production(spec, sampling, pilot_source_ids=pilot.source_ids)
    scheduled = production(spec, frozen, pilot, count=4)
    report = assess_precision(spec, scheduled, frozen=frozen)
    assert report["accepted"] is True
    assert report["status"] == "statistical_target_met"
    assert report["alpha_per_claim"] == pytest.approx(0.05 / (2 * 2 * 2))
    assert "Bonferroni" in report["repeated_check_method"]

    off_schedule = production(spec, frozen, pilot, count=3)
    before_check = assess_precision(spec, off_schedule, frozen=frozen)
    assert before_check["status"] == "check_not_due"
    assert before_check["interval"]["interval"] is None
    assert before_check["statistical_target_met"] is False


def test_budget_exhaustion_retains_unestablished_numerical_convergence(spec, pilot, sampling):
    frozen = freeze_production(spec, sampling, pilot_source_ids=pilot.source_ids)
    evidence = production(spec, frozen, pilot, numerical_error=0.5)
    report = assess_precision(spec, evidence, frozen=frozen, remaining_cost=0)
    assert report["status"] == "budget_exhausted"
    assert report["statistical_target_met"] is True
    assert report["numerical_convergence_established"] is False
    assert report["accepted"] is False
    assert "numerical_convergence_not_established" in report["conditions"]
    assert "budget_exhausted" in report["conditions"]


def test_no_numerical_checks_cannot_be_mislabeled_as_established_convergence(spec, pilot, sampling):
    spec = replace(spec, numerical=())
    frozen = freeze_production(spec, sampling, pilot_source_ids=pilot.source_ids)
    report = assess_precision(spec, production(spec, frozen, pilot), frozen=frozen)
    assert report["statistical_target_met"] is True
    assert report["numerical_convergence_established"] is False


def test_production_rejects_pilot_reuse_or_changed_frozen_design(spec, pilot, sampling):
    frozen = freeze_production(spec, sampling, pilot_source_ids=pilot.source_ids)
    evidence = production(spec, frozen, pilot)
    for changed in (
        replace(evidence, source_ids=("pilot-source", *evidence.source_ids)),
        replace(evidence, independent_from_pilot=False),
        replace(evidence, design_id="changed-child-horizon"),
        replace(evidence, replica_ids=tuple(reversed(evidence.replica_ids))),
    ):
        report = assess_precision(spec, changed, frozen=frozen)
        assert report["status"] == "production_design_invalid"
        assert report["accepted"] is False
        assert report["interval"]["interval"] is None


def test_beads_timestep_and_bias_changes_create_new_scientific_identities(identity, spec, pilot):
    alternatives = (
        identity.refined("beads", 64),
        identity.refined("timestep", 0.5),
        replace(identity, bias_id="refitted-bias-1"),
    )
    assert len({identity.identity_id, *(other.identity_id for other in alternatives)}) == 4
    for other in alternatives:
        with pytest.raises(ValueError, match="scientific identities"):
            assess_precision(spec, replace(pilot, scientific_identity=other.identity_id))


def test_frozen_jobs_have_distinct_streams_across_replicas_identities_and_epochs(spec, sampling):
    first = freeze_production(spec, sampling, epoch_index=0, pilot_source_ids=("pilot",))
    again = freeze_production(spec, sampling, epoch_index=0, pilot_source_ids=("pilot",))
    second = freeze_production(spec, sampling, epoch_index=1, pilot_source_ids=("pilot",))
    assert first == again
    assert first.epoch_id != second.epoch_id
    assert len({job["seed"] for job in first.jobs + second.jobs}) == len(first.jobs + second.jobs)
    assert len({job["job_id"] for job in first.jobs + second.jobs}) == len(first.jobs + second.jobs)
    with pytest.raises(ValueError, match="alpha allocation"):
        freeze_production(spec, sampling, epoch_index=2)


def test_controller_interrupt_recovery_deduplicates_plans_receipts_and_checks(tmp_path, spec, pilot, sampling):
    controller = PrecisionController(tmp_path, spec)
    frozen = controller.freeze(sampling, pilot_source_ids=pilot.source_ids)
    proposed = controller.plan_production(frozen, through_replicas=4, estimated_cost=100)
    controller.ledger.start(proposed["plan_id"], execution_id="production-0-4")

    recovered = PrecisionController(tmp_path, spec)
    refrozen = recovered.freeze(sampling, pilot_source_ids=pilot.source_ids)
    replayed = recovered.plan_production(refrozen, through_replicas=4, estimated_cost=100)
    assert replayed == proposed
    assert recovered.ledger.snapshot()["budget"]["pending_reconciliation"] == [proposed["plan_id"]]
    evidence = production(spec, refrozen, pilot)
    receipt = dict(execution_id="production-0-4", actual_cost=90,
                   result_ids=list(evidence.source_ids),
                   result={"job_ids": [job["job_id"] for job in proposed["jobs"]],
                           "evidence": evidence.to_dict()})
    recovered.ledger.complete(proposed["plan_id"], **receipt)
    first = recovered.check(evidence, frozen=refrozen)
    snapshot = recovered.ledger.snapshot()

    resumed = PrecisionController(tmp_path, spec)
    resumed.ledger.complete(proposed["plan_id"], **receipt)
    second = resumed.check(evidence, frozen=refrozen)
    assert first == second
    assert resumed.ledger.snapshot() == snapshot
    assert snapshot["budget"]["spent_cost"] == 90
    assert len(snapshot["results"]) == len(evidence.source_ids)


def test_production_progression_only_dispatches_new_frozen_replicas(tmp_path, spec, pilot, sampling):
    controller = PrecisionController(tmp_path, spec)
    frozen = controller.freeze(sampling, pilot_source_ids=pilot.source_ids)
    first = controller.plan_production(frozen, through_replicas=4, estimated_cost=100)
    first_ids = {job["job_id"] for job in first["jobs"]}
    controller.ledger.start(first["plan_id"], execution_id="first-look")
    controller.ledger.complete(
        first["plan_id"], execution_id="first-look", actual_cost=90,
        result_ids=["source-" + job_id for job_id in sorted(first_ids)],
        result={"job_ids": sorted(first_ids)},
    )
    next_plan = controller.plan_production(frozen, through_replicas=8, estimated_cost=100)
    next_ids = {job["job_id"] for job in next_plan["jobs"]}
    assert first_ids.isdisjoint(next_ids)
    assert next_ids == {job["job_id"] for job in frozen.jobs if 4 <= job["replica_index"] < 8}
    assert first_ids | next_ids == {job["job_id"] for job in frozen.jobs}
    assert controller.ledger.snapshot()["budget"]["spent_cost"] == 90
    assert controller.ledger.snapshot()["budget"]["reserved_cost"] == 100


def test_controller_recovers_stop_after_check_was_published(tmp_path, spec, pilot, sampling, monkeypatch):
    controller = PrecisionController(tmp_path, spec)
    frozen = controller.freeze(sampling, pilot_source_ids=pilot.source_ids)
    proposed = controller.plan_production(frozen, through_replicas=4, estimated_cost=100)
    controller.ledger.start(proposed["plan_id"], execution_id="production-0-4")
    evidence = production(spec, frozen, pilot)
    controller.ledger.complete(
        proposed["plan_id"], execution_id="production-0-4", actual_cost=90,
        result_ids=list(evidence.source_ids),
        result={"job_ids": [job["job_id"] for job in proposed["jobs"]]},
    )

    def interrupt_stop(*args, **kwargs):
        raise OSError("interrupted after check before stop")

    with monkeypatch.context() as patch:
        patch.setattr(controller.ledger, "stop", interrupt_stop)
        with pytest.raises(OSError, match="interrupted after check"):
            controller.check(evidence, frozen=frozen)
    interrupted = controller.ledger.snapshot()
    assert interrupted["stop"] is None
    assert any(check.get("assessment", {}).get("accepted") for check in interrupted["checks"].values())

    recovered = PrecisionController(tmp_path, spec)
    report = recovered.check(evidence, frozen=frozen)
    saved = recovered.ledger.snapshot()
    assert report["accepted"] is True
    assert saved["stop"]["status"] == "statistical_target_met"
    assert saved["checks"] == interrupted["checks"]
    assert saved["budget"]["spent_cost"] == 90


def test_controller_does_not_reuse_source_families_between_epochs(tmp_path, spec, pilot, sampling):
    controller = PrecisionController(tmp_path, spec)
    first_evidence = None
    for epoch in (0, 1):
        frozen = controller.freeze(sampling, epoch_index=epoch, pilot_source_ids=pilot.source_ids)
        proposed = controller.plan_production(frozen, through_replicas=4, estimated_cost=100)
        execution = f"production-{epoch}-4"
        controller.ledger.start(proposed["plan_id"], execution_id=execution)
        evidence = production(spec, frozen, pilot, numerical_error=0.5)
        if epoch:
            evidence = replace(evidence, source_family_ids=first_evidence.source_family_ids)
        controller.ledger.complete(
            proposed["plan_id"], execution_id=execution, actual_cost=90,
            result_ids=list(evidence.source_ids),
            result={"job_ids": [job["job_id"] for job in proposed["jobs"]]},
        )
        if epoch == 0:
            controller.check(evidence, frozen=frozen)
            first_evidence = evidence
        else:
            with pytest.raises(ValueError, match="reuse samples"):
                controller.check(evidence, frozen=frozen)


def test_replayed_production_check_still_requires_its_frozen_design(tmp_path, spec, pilot, sampling):
    controller = PrecisionController(tmp_path, spec)
    frozen = controller.freeze(sampling, pilot_source_ids=pilot.source_ids)
    proposed = controller.plan_production(frozen, through_replicas=4, estimated_cost=100)
    controller.ledger.start(proposed["plan_id"], execution_id="first-look")
    evidence = replace(production(spec, frozen, pilot), plateau_established=False)
    controller.ledger.complete(
        proposed["plan_id"], execution_id="first-look", actual_cost=90,
        result_ids=list(evidence.source_ids),
        result={"job_ids": [job["job_id"] for job in proposed["jobs"]]},
    )
    controller.check(evidence, frozen=frozen)
    assert controller.ledger.snapshot()["stop"] is None
    with pytest.raises(ValueError):
        controller.check(evidence)
    assert controller.ledger.snapshot()["stop"] is None


def test_replaying_old_check_cannot_replace_a_later_epoch_terminal_stop(tmp_path, spec, pilot, sampling):
    controller = PrecisionController(tmp_path, spec)
    old = None
    for epoch, cost in ((0, 90), (1, 910)):
        frozen = controller.freeze(sampling, epoch_index=epoch, pilot_source_ids=pilot.source_ids)
        proposed = controller.plan_production(frozen, through_replicas=4, estimated_cost=cost)
        execution = f"production-{epoch}-4"
        controller.ledger.start(proposed["plan_id"], execution_id=execution)
        evidence = production(spec, frozen, pilot)
        if epoch == 0:
            evidence = replace(evidence, plateau_established=False)
        controller.ledger.complete(
            proposed["plan_id"], execution_id=execution, actual_cost=cost,
            result_ids=list(evidence.source_ids),
            result={"job_ids": [job["job_id"] for job in proposed["jobs"]]},
        )
        report = controller.check(evidence, frozen=frozen)
        if epoch == 0:
            old = (evidence, frozen, report)
    saved = controller.ledger.snapshot()
    assert saved["stop"]["status"] == "statistical_target_met"
    assert saved["budget"]["available_cost"] == 0
    assert controller.check(old[0], frozen=old[1]) == old[2]
    assert controller.ledger.snapshot() == saved


def test_controller_rejects_changed_specification_and_frozen_epoch(tmp_path, spec, pilot, sampling):
    controller = PrecisionController(tmp_path, spec)
    controller.freeze(sampling, pilot_source_ids=pilot.source_ids)
    changed = replace(spec, target=replace(spec.target, relative_half_width=0.01))
    with pytest.raises(ValueError):
        PrecisionController(tmp_path, changed)
    with pytest.raises(ValueError):
        controller.freeze(replace(sampling, child_steps=200), pilot_source_ids=pilot.source_ids)


def test_controller_pilot_plan_is_idempotent(tmp_path, spec, pilot, sampling, costs):
    controller = PrecisionController(tmp_path, spec)
    evidence = replace(pilot, pmf_log_standard_error=0.3)
    first = controller.plan_pilot(evidence, sampling, costs=costs)
    snapshot = controller.ledger.snapshot()
    second = PrecisionController(tmp_path, spec).plan_pilot(evidence, sampling, costs=costs)
    assert first == second
    assert controller.ledger.snapshot() == snapshot


def test_reserved_budget_does_not_stop_work_that_has_not_started(tmp_path, spec, pilot, sampling, costs):
    spec = replace(spec, max_cost=costs["add_umbrella_streams"])
    controller = PrecisionController(tmp_path, spec)
    evidence = replace(pilot, pmf_log_standard_error=0.3)
    proposed = controller.plan_pilot(evidence, sampling, costs=costs)
    assert controller.ledger.snapshot()["budget"]["available_cost"] == 0
    controller.check(evidence)
    assert controller.ledger.snapshot()["stop"] is None
    controller.ledger.start(proposed["plan_id"], execution_id="reserved-pilot-work")
    assert controller.ledger.snapshot()["plans"][proposed["plan_id"]]["state"] == "running"


def test_unaffordable_production_increment_stops_without_erasing_scientific_conclusions(
        tmp_path, spec, pilot, sampling):
    spec = replace(spec, max_cost=100)
    controller = PrecisionController(tmp_path, spec)
    frozen = controller.freeze(sampling, pilot_source_ids=pilot.source_ids)
    first = controller.plan_production(frozen, through_replicas=4, estimated_cost=90)
    controller.ledger.start(first["plan_id"], execution_id="first-look")
    evidence = production(spec, frozen, pilot, numerical_error=0.5)
    controller.ledger.complete(
        first["plan_id"], execution_id="first-look", actual_cost=80,
        result_ids=list(evidence.source_ids),
        result={"job_ids": [job["job_id"] for job in first["jobs"]]},
    )
    previous = controller.check(evidence, frozen=frozen)
    assert previous["statistical_target_met"] is True
    assert previous["numerical_convergence_established"] is False
    checks = controller.ledger.snapshot()["checks"]

    rejected = controller.plan_production(frozen, through_replicas=8, estimated_cost=30)
    assert rejected["status"] == "budget_exhausted"
    assert rejected["actions"] == []
    saved = controller.ledger.snapshot()
    assert saved["budget"]["available_cost"] == 20
    assert saved["budget"]["spent_cost"] == 80
    assert saved["budget"]["reserved_cost"] == 0
    assert len(saved["plans"]) == 1
    assert all(saved["checks"][key] == value for key, value in checks.items())
    assert saved["stop"]["status"] == "budget_exhausted"
    assert saved["stop"]["conclusions"]["statistical_target_met"] is True
    assert saved["stop"]["conclusions"]["numerical_convergence_established"] is False
    assert saved["stop"]["conclusions"]["accepted"] is False

    recovered = PrecisionController(tmp_path, spec)
    assert recovered.plan_production(frozen, through_replicas=8, estimated_cost=30) == rejected
    assert recovered.check(evidence, frozen=frozen) == previous
    assert recovered.ledger.snapshot() == saved


def test_unaffordable_pilot_action_records_budget_stop_with_positive_balance(
        tmp_path, spec, pilot, sampling, costs):
    spec = replace(spec, max_cost=10)
    controller = PrecisionController(tmp_path, spec)
    evidence = replace(pilot, pmf_log_standard_error=0.3)
    previous = controller.check(evidence)
    proposed = controller.plan_pilot(evidence, sampling, costs=costs)
    assert proposed["status"] == "budget_exhausted"
    assert proposed["actions"] == []
    saved = controller.ledger.snapshot()
    assert saved["budget"]["available_cost"] == 10
    assert saved["budget"]["spent_cost"] == 0
    assert saved["plans"] == {}
    assert saved["stop"]["status"] == "budget_exhausted"
    assert saved["stop"]["conclusions"]["statistical_target_met"] is False
    assert saved["stop"]["conclusions"]["numerical_convergence_established"] is False
    assert saved["stop"]["conclusions"]["accepted"] is False
    recovered = PrecisionController(tmp_path, spec)
    assert recovered.plan_pilot(evidence, sampling, costs=costs) == proposed
    assert recovered.check(evidence) == previous
    assert recovered.ledger.snapshot() == saved


def test_resume_codecs_preserve_frozen_jobs_and_missing_errors(tmp_path, spec, pilot, sampling):
    controller = PrecisionController(tmp_path, spec)
    frozen = controller.freeze(sampling, epoch_index=1, pilot_source_ids=pilot.source_ids,
                               pilot_source_family_ids=("pilot-family",))
    snapshot = controller.ledger.snapshot()
    recovered = PrecisionController(tmp_path, spec)
    restored = recovered.restore_frozen(epoch_index=1)
    assert restored == frozen
    assert restored.to_dict() == frozen.to_dict()
    assert restored.jobs == frozen.jobs
    assert recovered.ledger.snapshot() == snapshot
    with pytest.raises(ValueError):
        recovered.restore_frozen(epoch_index=0)

    evidence = replace(production(spec, restored, pilot), pmf_log_standard_error=None,
                       notes=("PMF error is unavailable",))
    decoded = RateEvidence.from_dict(evidence.to_dict())
    assert decoded == evidence
    assert isinstance(decoded.numerical[0], NumericalEvidence)
    assert decoded.pmf_log_standard_error is None
    assert decoded.numerical[0].replica_ids == evidence.numerical[0].replica_ids
