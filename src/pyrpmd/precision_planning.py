"""Diagnostic pilot plans and a durable handoff to the rate execution layer.

The execution layer owns scheduling, checkpoints and actual resource limits.
It must use a plan's durable ID as its idempotency key, reconcile ``running``
work after interruption, and report actual costs and immutable artifact IDs.
No dynamics are launched here and no production design adapts in place.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import math

from .precision import (
    FrozenProduction, _digest, _number, assess_precision, freeze_production, validate_frozen_production,
)
from .precision_ledger import PrecisionBudgetError, PrecisionLedger, PrecisionLedgerError


def recommend_supplement(spec, evidence, sampling, *, costs, remaining_cost):
    """Choose one scientifically justified pilot action within its quote.

    ``costs`` contains execution-layer quotes for a *complete* action in
    ``spec.cost_unit``. Quotes must include any replay/recomputation necessary
    to produce a new estimate. They are not guesses about marginal CPU time.
    A weak/unknown parent-child decomposition never justifies adding children.
    Production work always follows the frozen independent-replica design.
    """
    if evidence.scientific_identity != spec.identity.identity_id:
        raise ValueError("Pilot evidence belongs to another scientific identity")
    _number(remaining_cost, "remaining_cost")
    if not isinstance(costs, dict):
        raise ValueError("costs must map action kinds to complete execution quotes")
    for kind, cost in costs.items():
        _number(cost, f"costs.{kind}")
    empty = {"format": "PyRPMD.precision.plan.v1", "actions": [], "estimated_cost": 0.,
             "scientific_identity": spec.identity.identity_id,
             "epoch_id": "pilot:" + spec.specification_id, "reasons": [], "status": "no_plan"}
    if evidence.phase != "pilot":
        return {**empty, "status": "fixed_design_requires_new_pilot",
                "reasons": ["Production allocation and time horizon are frozen; use another predeclared independent epoch after a new pilot."]}
    target, kind, reason, comparison = sampling, None, None, None
    t = spec.target
    energy_bad = (evidence.energy_complete is False or
                  (evidence.maximum_energy_error_kbt is not None and
                   evidence.maximum_energy_error_kbt > t.maximum_energy_error_kbt))
    if energy_bad:
        candidates = [r for r in spec.numerical if r.axis == "timestep"
                      and r.value < spec.identity.timestep_au]
        if not candidates:
            return {**empty, "status": "diagnostic_failed", "reasons": [
                "Energy diagnostics failed; a predeclared finer-timestep comparison is required."]}
        comparison = min(candidates, key=lambda r: r.value)
        kind, reason = "compare_timestep", "energy_error_requires_controlled_timestep_comparison"
    elif evidence.plateau_established is False and evidence.plateau_failure == "horizon":
        kind, reason = "extend_child_horizon", "resolved_tail_time_dependence"
        target = replace(sampling, child_steps=2 * sampling.child_steps)
    elif (evidence.pmf_ready is False or evidence.overlap_connected is False
          or (evidence.minimum_umbrella_ess is not None
              and evidence.minimum_umbrella_ess < t.minimum_umbrella_ess)):
        kind, reason = "add_umbrella_streams", "pmf_support_or_effective_sampling_limited"
        target = replace(sampling, umbrella_streams=2 * sampling.umbrella_streams)
    elif (evidence.independent_parent_chains is None
          or evidence.independent_parent_chains < t.minimum_parent_chains
          or evidence.effective_parent_blocks is None
          or evidence.effective_parent_blocks < t.minimum_effective_parent_blocks):
        kind, reason = "add_parent_chains", "independent_parent_sampling_limited"
        target = replace(sampling, parent_chains=max(2 * sampling.parent_chains, t.minimum_parent_chains))
    elif evidence.plateau_established is False:
        kind, reason = "add_parent_chains", "plateau_uncertainty_not_resolved_as_horizon_error"
        target = replace(sampling, parent_chains=2 * sampling.parent_chains)
    elif (evidence.pmf_log_standard_error is None or evidence.recrossing_log_standard_error is None
          or evidence.minimum_umbrella_ess is None):
        return {**empty, "status": "error_unestimable", "reasons": [
            "Missing component uncertainty or effective-sample diagnostics cannot identify a sampling remedy."]}
    else:
        report = assess_precision(spec, evidence)
        width = report["interval"]["relative_half_width"]
        needs_precision = width is None or width > t.relative_half_width
        if needs_precision:
            pmf = evidence.pmf_log_standard_error ** 2
            rec = (None if evidence.recrossing_log_standard_error is None
                   else evidence.recrossing_log_standard_error ** 2)
            if rec is not None and pmf >= rec:
                kind, reason = "add_umbrella_streams", "pmf_uncertainty_dominates"
                target = replace(sampling, umbrella_streams=2 * sampling.umbrella_streams)
            elif evidence.variance_attribution_validated:
                parent, child = evidence.parent_log_variance, evidence.child_log_variance
                if (parent is None or child is None or rec is None
                        or not math.isclose(parent + child, rec, rel_tol=1e-6, abs_tol=1e-15)):
                    raise ValueError("Validated parent/child variance components must sum to recrossing variance")
                if child > parent:
                    kind, reason = "add_children", "conditional_child_variance_dominates"
                    target = replace(sampling, child_pairs=2 * sampling.child_pairs)
                else:
                    kind, reason = "add_parent_chains", "between_parent_variance_dominates"
                    target = replace(sampling, parent_chains=2 * sampling.parent_chains)
            elif rec is not None:
                kind, reason = "add_parent_chains", "recrossing_variance_attribution_unavailable"
                target = replace(sampling, parent_chains=2 * sampling.parent_chains)
            else:
                return {**empty, "status": "error_unestimable", "reasons": [
                    "Recrossing error and independent variance attribution are unavailable."]}
        else:
            for requirement, check in zip(spec.numerical, report["numerical"]):
                if not check["established"]:
                    comparison = requirement
                    kind, reason = "compare_" + requirement.axis, "numerical_equivalence_not_established"
                    break
    if kind is None:
        return {**empty, "status": "ready_to_freeze_production", "reasons": [
            "Pilot is diagnostic only; freeze a fresh independent production design."]}
    if kind not in costs:
        return {**empty, "status": "execution_quote_required", "reasons": [f"Missing complete cost quote for {kind}."]}
    action = {"kind": kind, "reason": reason, "target_sampling": asdict(target),
              "scientific_identity": spec.identity.identity_id,
              "new_independent_parent_chains": (target.parent_chains - sampling.parent_chains
                                                if kind == "add_parent_chains" else 0),
              "phase": "pilot", "requires_fresh_production_after_pilot": True}
    if kind == "add_umbrella_streams":
        action["new_independent_streams_per_window"] = target.umbrella_streams - sampling.umbrella_streams
        action["bias_may_not_change_in_place"] = True
    if kind == "add_children":
        action["keep_parent_cluster_identity"] = True
    if kind == "extend_child_horizon":
        action["replace_short_horizon_estimate"] = True
        action["do_not_pool_overlapping_child_prefixes"] = True
    if comparison is not None:
        candidate = spec.identity.refined(comparison.axis, comparison.value)
        action.update(requirement_id=comparison.requirement_id,
                      candidate_identity=candidate.to_dict(),
                      controlled_axis=comparison.axis, preserve_physical_durations=True,
                      pool_with_baseline=False)
    plan = {**empty, "actions": [action], "estimated_cost": float(costs[kind]),
            "cost_unit": spec.cost_unit, "status": "planned", "reasons": [reason],
            "execution_budget": {"cost_unit": spec.cost_unit, "max_cost": float(costs[kind]),
                                 "enforce_in_executor": True},
            "based_on_source_ids": list(evidence.source_ids),
            "sampling_design_id": sampling.design_id}
    # The available balance is deliberately excluded: retries must find the
    # identical plan after its own reservation has reduced that balance.
    plan["plan_id"] = _digest(plan)
    if plan["estimated_cost"] > remaining_cost:
        return {**empty, "status": "budget_exhausted", "reasons": [
            f"The justified {kind} action exceeds the remaining budget."],
            "unfunded_action": action, "required_cost": plan["estimated_cost"]}
    return plan


class PrecisionController:
    """Concrete handoff API for Agent F; every mutation is durable and replayable."""

    def __init__(self, directory, spec):
        self.spec = spec
        self.ledger = PrecisionLedger(directory, specification=spec.to_dict(), max_cost=spec.max_cost)

    def freeze(self, sampling, *, epoch_index=0, pilot_source_ids=(), pilot_source_family_ids=()):
        frozen = freeze_production(self.spec, sampling, epoch_index=epoch_index,
                                   pilot_source_ids=pilot_source_ids,
                                   pilot_source_family_ids=pilot_source_family_ids)
        self.ledger.record_check("freeze:" + frozen.epoch_id, frozen.to_dict())
        return frozen

    def restore_frozen(self, epoch_index=0):
        """Recover a persisted design without choosing new seeds or budgets."""
        for value in self.ledger.snapshot()["checks"].values():
            if (value.get("format") == "PyRPMD.precision.frozen-production.v1"
                    and value.get("epoch_index") == epoch_index):
                return FrozenProduction.from_dict(self.spec, value)
        raise PrecisionLedgerError("Requested production epoch has not been frozen")

    def _require_frozen(self, frozen, state):
        validate_frozen_production(self.spec, frozen)
        if state["checks"].get("freeze:" + frozen.epoch_id) != frozen.to_dict():
            raise PrecisionLedgerError("Persist the exact production freeze before planning or checking")

    def plan_pilot(self, evidence, sampling, *, costs):
        state = self.ledger.snapshot()
        # Compute the deterministic candidate before checking its reservation.
        quoted = recommend_supplement(self.spec, evidence, sampling, costs=costs,
                                      remaining_cost=self.spec.max_cost)
        if quoted.get("plan_id") in state["plans"]:
            entry = state["plans"][quoted["plan_id"]]
            if entry["plan"] != quoted:
                raise PrecisionLedgerError("Conflicting pilot plan replay")
            return quoted
        plan = recommend_supplement(self.spec, evidence, sampling, costs=costs,
                                    remaining_cost=state["budget"]["available_cost"])
        if plan["actions"]:
            self.ledger.register_plan(plan)
        elif plan["status"] == "budget_exhausted":
            report = assess_precision(self.spec, evidence,
                                      remaining_cost=state["budget"]["available_cost"])
            plan = self._budget_stop(plan, state, report)
        return plan

    def plan_production(self, frozen, *, through_replicas, estimated_cost):
        """Reserve only the next unexecuted suffix through a predeclared look."""
        _number(estimated_cost, "estimated_cost")
        if through_replicas not in self.spec.checks.replicas or type(through_replicas) is not int:
            raise ValueError("Production target must be a predeclared check")
        state = self.ledger.snapshot()
        self._require_frozen(frozen, state)
        existing = [entry for entry in state["plans"].values()
                    if entry["plan"].get("production_design_id") == frozen.design_id]
        for entry in existing:
            if entry["plan"].get("through_replicas") == through_replicas:
                if entry["plan"]["estimated_cost"] != float(estimated_cost):
                    raise PrecisionLedgerError("Conflicting production cost quote replay")
                return entry["plan"]
        if any(entry["state"] != "completed" for entry in existing):
            raise PrecisionLedgerError("Reconcile pending or failed production before another check allocation")
        previous = max((e["plan"]["through_replicas"] for e in existing), default=0)
        next_count = next((n for n in self.spec.checks.replicas if n > previous), None)
        if through_replicas != next_count:
            raise ValueError("Production checks must be executed in their declared order")
        jobs = [job for job in frozen.jobs if previous <= job["replica_index"] < through_replicas]
        plan = {"format": "PyRPMD.precision.plan.v1", "status": "planned",
                "scientific_identity": self.spec.identity.identity_id, "epoch_id": frozen.epoch_id,
                "production_design_id": frozen.design_id, "through_replicas": through_replicas,
                "from_replicas": previous, "estimated_cost": float(estimated_cost),
                "execution_budget": {"cost_unit": self.spec.cost_unit, "max_cost": float(estimated_cost),
                                     "enforce_in_executor": True},
                "cost_unit": self.spec.cost_unit, "jobs": jobs,
                "actions": [{"kind": "collect_frozen_production", "job_ids": [j["job_id"] for j in jobs]}],
                "reasons": ["Predeclared independent production increment; no adaptive reallocation."]}
        plan["plan_id"] = _digest(plan)
        for value in state["checks"].values():
            decision = value.get("budget_decision", {})
            if decision.get("unfunded_plan", {}).get("plan_id") == plan["plan_id"]:
                # Reconcile a crash after publishing the budget decision but
                # before its stop, then return exactly the saved decision.
                if state["stop"] is None:
                    self.ledger.stop("budget_exhausted", decision["reasons"], decision["conclusions"])
                return decision
        try:
            self.ledger.register_plan(plan)
        except PrecisionBudgetError:
            latest = None
            for event in reversed(state["events"]):
                if event["kind"] != "check_recorded":
                    continue
                value = state["checks"].get(event["check_id"], {})
                if ("assessment" in value and value.get("evidence", {}).get("design_id") == frozen.design_id):
                    latest = value["assessment"]
                    break
            return self._budget_stop({
                "format": "PyRPMD.precision.plan.v1", "status": "budget_exhausted",
                "actions": [], "estimated_cost": 0., "required_cost": float(estimated_cost),
                "unfunded_plan": plan,
                "reasons": ["The next predeclared production increment exceeds the remaining budget."]},
                state, latest)
        return plan

    def _budget_stop(self, plan, state, previous=None):
        """Keep unmet scientific conclusions when the next valid action cannot fit."""
        conclusions = {name: bool((previous or {}).get(name, False)) for name in (
            "statistical_target_met", "numerical_convergence_established", "accepted")}
        # Affording an action is independent of whether an earlier interval
        # met its target; a budget stop never manufactures a convergence claim.
        result = {**plan, **conclusions, "conclusions": conclusions}
        outstanding = any(entry["state"] in ("planned", "running") for entry in state["plans"].values())
        if not outstanding and state["stop"] is None:
            key = "budget:" + _digest(result)
            self.ledger.record_check(key, {"budget_decision": result})
            self.ledger.stop("budget_exhausted", result["reasons"], conclusions)
        return result

    def check(self, evidence, *, frozen=None):
        """Persist one decision; the same look can never consume evidence twice."""
        state = self.ledger.snapshot()
        remaining = max(0., state["budget"]["max_cost"] - state["budget"]["spent_cost"])
        if evidence.phase == "production":
            if frozen is None:
                raise PrecisionLedgerError("Production checking requires its persisted frozen design")
            self._require_frozen(frozen, state)
        key = ("production:" + str(evidence.epoch_id) + ":" + str(len(evidence.replica_ids))
               if evidence.phase == "production" else "pilot:" + _digest(evidence.to_dict()))
        if key in state["checks"]:
            record = state["checks"][key]
            if record["evidence"] != evidence.to_dict():
                raise PrecisionLedgerError("A checked look cannot be replaced by selected or revised evidence")
            self._finalize_stop(record["assessment"], evidence, frozen, state, key)
            return record["assessment"]
        if evidence.phase == "production":
            if len(evidence.replica_ids) not in self.spec.checks.replicas:
                # No unplanned inference is persisted or spent as a look.
                return assess_precision(self.spec, evidence, frozen=frozen,
                                        remaining_cost=remaining)
            owned_sources, completed_jobs = set(), set()
            for entry in state["plans"].values():
                if (entry["state"] == "completed" and
                        entry["plan"].get("production_design_id") == frozen.design_id):
                    receipt = entry["receipt"]
                    planned_jobs = {job["job_id"] for job in entry["plan"]["jobs"]}
                    actual_jobs = set(receipt["result"].get("job_ids", ()))
                    if actual_jobs != planned_jobs:
                        raise PrecisionLedgerError("Completed production receipt must identify every planned job exactly")
                    completed_jobs.update(actual_jobs)
                    owned_sources.update(receipt["result_ids"])
            expected_jobs = {job["job_id"] for job in frozen.jobs
                             if job["replica_index"] < len(evidence.replica_ids)}
            if not expected_jobs.issubset(completed_jobs) or set(evidence.source_ids) != owned_sources:
                raise PrecisionLedgerError("Production evidence must come from complete reconciled receipts without filtering")
            for previous in state["checks"].values():
                old = previous.get("evidence", {})
                if (old.get("phase") == "production" and old.get("epoch_id") != evidence.epoch_id
                        and (set(old.get("source_ids", ())) & set(evidence.source_ids)
                             or set(old.get("source_family_ids", ())) & set(evidence.source_family_ids))):
                    raise PrecisionLedgerError("Independent production epochs cannot reuse samples")
        report = assess_precision(self.spec, evidence, frozen=frozen,
                                  remaining_cost=remaining)
        record = {"evidence": evidence.to_dict(), "assessment": report}
        self.ledger.record_check(key, record)
        self._finalize_stop(report, evidence, frozen, state, key)
        return report

    def _finalize_stop(self, report, evidence, frozen, state, key):
        # Also called when replaying a persisted check: a crash between the
        # check publication and terminal stop must not lose the stop record.
        if state["stop"] is not None:
            return
        outstanding = any(entry["state"] in ("planned", "running") for entry in state["plans"].values())
        terminal = (report["accepted"] or
                    ("budget_exhausted" in report["conditions"] and not outstanding) or
                    (evidence.phase == "production" and ("diagnostic_failed" in report["conditions"]
                     or "error_unestimable" in report["conditions"] or "check_plan_exhausted" in report["conditions"])))
        # A later predeclared epoch is still possible after a new independent
        # pilot, with its reserved share of the original global alpha budget.
        if (terminal and not report["accepted"] and frozen is not None
                and frozen.epoch_index + 1 < self.spec.checks.max_epochs
                and state["budget"]["available_cost"] > 0):
            terminal = False
        if terminal:
            self.ledger.stop(report["status"], report["conditions"], {
                "statistical_target_met": report["statistical_target_met"],
                "numerical_convergence_established": report["numerical_convergence_established"],
                "accepted": report["accepted"], "assessment_check_id": key})
