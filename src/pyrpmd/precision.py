"""Precision acceptance for frozen, independently sampled thermal RPMD rates.

Pilot diagnostics may choose a design; only fresh production under a frozen
design can satisfy a precision target. A finite family of predeclared looks,
epochs and numerical comparisons shares one Bonferroni error budget. Coverage
inherits the approximate cluster-t/log-delta model of the supplied estimates;
Bonferroni does not make that model exact or validate adaptive production.

This module contains no scheduler or CLI. ``precision_planning`` produces the
work contract consumed by an execution layer, and ``precision_ledger`` records
its durable receipts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

from .precision_statistics import (
    combine_log_rate_uncertainty, log_rate_interval, numerical_equivalence,
    paired_rate_replica_statistics,
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _number(value, name, *, minimum=0., strict=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"{name} must be {'greater than' if strict else 'at least'} {minimum}")
    return float(value)


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value, name):
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _unique(values, name):
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{name} must be a sequence")
    for value in values:
        _text(value, name)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate identities")
    return tuple(values)


@dataclass(frozen=True)
class ScientificIdentity:
    """Identity of a physical model, estimator, bias and numerical settings.

    ``model_id`` pins the PES, geometry, masses, temperature, reaction coordinate,
    normalization and rate definition. ``bias_id`` pins the complete frozen PMF
    bias. These are content identifiers, not run-directory names. Sampling
    amounts and random streams belong to the production design, not this ID.
    """

    model_id: str
    beads: int
    timestep_au: float
    bias_id: str
    workflow: str = "thermal_rpmd_rate"

    def __post_init__(self):
        _text(self.model_id, "model_id")
        _text(self.bias_id, "bias_id")
        _integer(self.beads, "beads")
        _number(self.timestep_au, "timestep_au", strict=True)
        if self.workflow != "thermal_rpmd_rate":
            raise ValueError("Precision v1 supports thermal RPMD rate only")

    @property
    def identity_id(self):
        return _digest(asdict(self))

    def refined(self, axis, value):
        if axis not in ("beads", "timestep"):
            raise ValueError("Controlled numerical axes are beads and timestep")
        return replace(self, **{"timestep_au" if axis == "timestep" else axis: value})

    def to_dict(self):
        return {**asdict(self), "identity_id": self.identity_id}


@dataclass(frozen=True)
class SamplingDesign:
    """Fixed sampling amounts per independent complete rate replica."""

    umbrella_streams: int
    umbrella_steps: int
    parent_chains: int
    parent_samples: int
    child_pairs: int
    child_steps: int
    umbrella_burnin_steps: int = 0
    parent_burnin_steps: int = 0
    parent_interval_steps: int = 1
    plateau_fraction: float = .25

    def __post_init__(self):
        for name in ("umbrella_streams", "umbrella_steps", "parent_chains",
                     "parent_samples", "child_pairs", "parent_interval_steps"):
            _integer(getattr(self, name), name)
        _integer(self.child_steps, "child_steps", 2)
        _integer(self.umbrella_burnin_steps, "umbrella_burnin_steps", 0)
        _integer(self.parent_burnin_steps, "parent_burnin_steps", 0)
        _number(self.plateau_fraction, "plateau_fraction", strict=True)
        if self.plateau_fraction >= 1:
            raise ValueError("plateau_fraction must be less than one")

    @property
    def design_id(self):
        return _digest(asdict(self))


@dataclass(frozen=True)
class PrecisionTarget:
    relative_half_width: float
    confidence: float = .95
    minimum_umbrella_ess: float = 20.
    minimum_parent_chains: int = 2
    minimum_effective_parent_blocks: float = 2.
    maximum_energy_error_kbt: float = .1

    def __post_init__(self):
        for name in ("relative_half_width", "minimum_umbrella_ess",
                     "minimum_effective_parent_blocks", "maximum_energy_error_kbt"):
            _number(getattr(self, name), name, strict=True)
        _number(self.confidence, "confidence", strict=True)
        if self.confidence >= 1:
            raise ValueError("confidence must be below one")
        _integer(self.minimum_parent_chains, "minimum_parent_chains", 2)


@dataclass(frozen=True)
class CheckSchedule:
    """Cumulative complete-replica counts; no inference between these looks."""

    replicas: tuple[int, ...]
    max_epochs: int = 1

    def __post_init__(self):
        object.__setattr__(self, "replicas", tuple(self.replicas))
        if not self.replicas:
            raise ValueError("At least one production check must be declared")
        for count in self.replicas:
            _integer(count, "check replica count", 2)
        if tuple(sorted(set(self.replicas))) != self.replicas:
            raise ValueError("Check replica counts must strictly increase")
        _integer(self.max_epochs, "max_epochs")


@dataclass(frozen=True)
class NumericalRequirement:
    axis: str
    value: float | int
    relative_margin: float

    def __post_init__(self):
        if self.axis not in ("beads", "timestep"):
            raise ValueError("Numerical comparisons support beads or timestep")
        if self.axis == "beads":
            _integer(self.value, "comparison beads")
        else:
            _number(self.value, "comparison timestep", strict=True)
        _number(self.relative_margin, "relative_margin", strict=True)

    @property
    def log_margin(self):
        # Symmetric multiplicative equivalence: 1/(1+r) < k2/k1 < 1+r.
        return math.log1p(self.relative_margin)

    @property
    def requirement_id(self):
        return _digest(asdict(self))


@dataclass(frozen=True)
class PrecisionSpec:
    identity: ScientificIdentity
    target: PrecisionTarget
    checks: CheckSchedule
    max_cost: float
    numerical: tuple[NumericalRequirement, ...] = ()
    cost_unit: str = "force_evaluations"
    seed: int = 0

    def __post_init__(self):
        _number(self.max_cost, "max_cost")
        _text(self.cost_unit, "cost_unit")
        _integer(self.seed, "seed", 0)
        object.__setattr__(self, "numerical", tuple(self.numerical))
        ids = [r.requirement_id for r in self.numerical]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate numerical requirement")
        candidates = []
        for requirement in self.numerical:
            candidate = self.identity.refined(requirement.axis, requirement.value)
            if candidate.identity_id == self.identity.identity_id:
                raise ValueError("Numerical comparison must change its declared axis")
            candidates.append(candidate.identity_id)
        if len(set(candidates)) != len(candidates):
            raise ValueError("Each controlled comparison must have a distinct identity")

    @property
    def specification_id(self):
        return _digest(asdict(self))

    @property
    def alpha_per_claim(self):
        return (1 - self.target.confidence) / (
            self.checks.max_epochs * len(self.checks.replicas) * (1 + len(self.numerical)))

    def to_dict(self):
        # JSON round trip makes tuples portable for the strict ledger.
        return json.loads(json.dumps({**asdict(self), "specification_id": self.specification_id,
                                      "alpha_per_claim": self.alpha_per_claim}))


@dataclass(frozen=True)
class FrozenProduction:
    specification_id: str
    epoch_index: int
    sampling: SamplingDesign
    pilot_source_ids: tuple[str, ...]
    jobs_json: str
    pilot_source_family_ids: tuple[str, ...] = ()

    @property
    def epoch_id(self):
        return _digest({"specification": self.specification_id, "epoch": self.epoch_index})

    @property
    def design_id(self):
        return _digest(asdict(self))

    @property
    def jobs(self):
        return json.loads(self.jobs_json)

    def to_dict(self):
        return {"format": "PyRPMD.precision.frozen-production.v1",
                "specification_id": self.specification_id, "epoch_index": self.epoch_index,
                "epoch_id": self.epoch_id, "design_id": self.design_id,
                "sampling": asdict(self.sampling), "pilot_source_ids": list(self.pilot_source_ids),
                "pilot_source_family_ids": list(self.pilot_source_family_ids),
                "jobs": self.jobs, "independent_initialization_required": True}

    @classmethod
    def from_dict(cls, spec, value):
        """Restore only a canonical design from a previously persisted freeze."""
        frozen = freeze_production(spec, SamplingDesign(**value["sampling"]),
                                   epoch_index=value["epoch_index"],
                                   pilot_source_ids=value["pilot_source_ids"],
                                   pilot_source_family_ids=value.get("pilot_source_family_ids", ()))
        if value != frozen.to_dict():
            raise ValueError("Saved freeze does not match its predeclared specification")
        return frozen


def freeze_production(spec, sampling, *, epoch_index=0, pilot_source_ids=(), pilot_source_family_ids=()):
    """Freeze all seeds, sampling amounts and comparisons before drawing data.

    The caller must persist this with ``PrecisionController.freeze`` before
    submitting jobs. Each job requires fresh umbrella and parent initialization;
    sharing pilot configurations/checkpoints or an estimated source PMF is not
    supported by the independent-production contract.
    """
    _integer(epoch_index, "epoch_index", 0)
    if epoch_index >= spec.checks.max_epochs:
        raise ValueError("Production epoch exceeds the predeclared alpha allocation")
    pilots = _unique(pilot_source_ids, "pilot_source_ids")
    pilot_families = _unique(pilot_source_family_ids, "pilot_source_family_ids")
    settings = [("baseline", spec.identity)] + [
        (r.requirement_id, spec.identity.refined(r.axis, r.value)) for r in spec.numerical]
    jobs = []
    for label, identity in settings:
        values = asdict(sampling)
        ratio = spec.identity.timestep_au / identity.timestep_au
        for name in ("umbrella_steps", "child_steps", "umbrella_burnin_steps",
                     "parent_burnin_steps", "parent_interval_steps"):
            scaled = values[name] * ratio
            if not math.isfinite(scaled) or not math.isclose(scaled, round(scaled), rel_tol=1e-12, abs_tol=1e-10):
                raise ValueError(f"Controlled timestep comparison cannot preserve {name} exactly")
            values[name] = int(round(scaled))
        job_sampling = SamplingDesign(**values)
        for index in range(spec.checks.replicas[-1]):
            key = {"specification": spec.specification_id, "epoch": epoch_index,
                   "identity": identity.identity_id, "replica": index, "seed": spec.seed}
            digest = _digest(key)
            jobs.append({"job_id": digest, "group": label, "replica_index": index,
                         "scientific_identity": identity.identity_id,
                         "identity": identity.to_dict(), "seed": int(digest[:15], 16),
                         "sampling": asdict(job_sampling), "sampling_design_id": job_sampling.design_id})
    if len({j["seed"] for j in jobs}) != len(jobs):
        raise ValueError("Production random-stream collision")
    return FrozenProduction(spec.specification_id, epoch_index, sampling, pilots,
                            json.dumps(jobs, sort_keys=True, separators=(",", ":")), pilot_families)


def validate_frozen_production(spec, frozen):
    """Reject altered/restored schedules, out-of-family epochs or reused seeds."""
    if not isinstance(frozen, FrozenProduction):
        raise ValueError("Expected a FrozenProduction design")
    canonical = freeze_production(spec, frozen.sampling, epoch_index=frozen.epoch_index,
                                  pilot_source_ids=frozen.pilot_source_ids,
                                  pilot_source_family_ids=frozen.pilot_source_family_ids)
    if frozen != canonical:
        raise ValueError("Frozen production differs from its predeclared canonical design")
    return frozen


@dataclass(frozen=True)
class NumericalEvidence:
    requirement_id: str
    baseline_identity: str
    candidate_identity: str
    log_ratio: float | None
    standard_error: float | None
    degrees_of_freedom: float | None
    covariance_accounted: bool = False
    epoch_id: str | None = None
    design_id: str | None = None
    replica_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    diagnostics_passed: bool | None = None

    def __post_init__(self):
        for name in ("requirement_id", "baseline_identity", "candidate_identity"):
            _text(getattr(self, name), name)
        if type(self.covariance_accounted) is not bool:
            raise ValueError("covariance_accounted must be boolean")
        if self.diagnostics_passed is not None and type(self.diagnostics_passed) is not bool:
            raise ValueError("diagnostics_passed must be boolean or missing")
        for name in ("source_ids", "replica_ids"):
            object.__setattr__(self, name, _unique(getattr(self, name), name))


@dataclass(frozen=True)
class RateEvidence:
    scientific_identity: str
    phase: str = "pilot"
    source_ids: tuple[str, ...] = ()
    rate: float | None = None
    pmf_log_standard_error: float | None = None
    recrossing_log_standard_error: float | None = None
    covariance: float | None = None
    components_independent: bool = False
    degrees_of_freedom: float | None = None
    pmf_ready: bool | None = None
    overlap_connected: bool | None = None
    minimum_umbrella_ess: float | None = None
    independent_parent_chains: int | None = None
    effective_parent_blocks: float | None = None
    parent_log_variance: float | None = None
    child_log_variance: float | None = None
    variance_attribution_validated: bool = False
    plateau_established: bool | None = None
    energy_complete: bool | None = None
    maximum_energy_error_kbt: float | None = None
    epoch_id: str | None = None
    design_id: str | None = None
    replica_ids: tuple[str, ...] = ()
    independent_from_pilot: bool = False
    numerical: tuple[NumericalEvidence, ...] = ()
    notes: tuple[str, ...] = ()
    plateau_failure: str = "unknown"
    source_family_ids: tuple[str, ...] = ()

    def __post_init__(self):
        _text(self.scientific_identity, "scientific_identity")
        if self.phase not in ("pilot", "production"):
            raise ValueError("Evidence phase must be pilot or production")
        for name in ("source_ids", "replica_ids", "source_family_ids"):
            object.__setattr__(self, name, _unique(getattr(self, name), name))
        for name in ("pmf_log_standard_error", "recrossing_log_standard_error",
                     "minimum_umbrella_ess", "effective_parent_blocks", "parent_log_variance",
                     "child_log_variance", "maximum_energy_error_kbt"):
            value = getattr(self, name)
            if value is not None:
                _number(value, name)
        if self.rate is not None:
            _number(self.rate, "rate", strict=True)
        if self.independent_parent_chains is not None:
            _integer(self.independent_parent_chains, "independent_parent_chains", 0)
        for name in ("components_independent", "variance_attribution_validated", "independent_from_pilot"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("pmf_ready", "overlap_connected", "plateau_established", "energy_complete"):
            if getattr(self, name) is not None and type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean or missing")
        object.__setattr__(self, "numerical", tuple(self.numerical))
        object.__setattr__(self, "notes", tuple(self.notes))
        if self.plateau_failure not in ("unknown", "horizon", "sampling"):
            raise ValueError("plateau_failure must be unknown, horizon or sampling")

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, value):
        """Restore receipt evidence without converting missing values to zero."""
        value = dict(value)
        value["numerical"] = tuple(NumericalEvidence(**item) for item in value.get("numerical", ()))
        return cls(**value)


def evidence_from_diagnostics(diagnostics, identity):
    """Import legacy diagnostics for pilot planning, never certify production."""
    pmf, rec, energy = (diagnostics[name] for name in ("pmf", "recrossing", "energy"))
    config = diagnostics.get("metadata", {}).get("resolved_input", {})
    beta, barrier_se = config.get("beta"), pmf.get("barrier_standard_error_hartree")
    p_se = beta * barrier_se if beta is not None and barrier_se is not None else None
    kappa, k_se = rec.get("kappa"), rec.get("standard_error")
    r_se = k_se / kappa if k_se is not None and kappa is not None and kappa > 0 else None
    blocks = rec.get("effective_parent_blocks")
    if isinstance(blocks, dict):
        blocks = sum(blocks.values()) if all(type(v) in (int, float) and math.isfinite(v)
                                            and v >= 0 for v in blocks.values()) else None
    rate = diagnostics.get("rate_cm3_s") or diagnostics.get("rate_estimate_cm3_s")
    tail = rec.get("plateau_diagnostics", {})
    plateau_failure = "unknown"
    if rec.get("plateau_detected") is False:
        deviation, radius, limit = (tail.get(name) for name in (
            "maximum_tail_deviation", "bootstrap_95_envelope_radius", "absolute_tolerance"))
        if ("fewer_than_ten_tail_samples" in tail.get("reasons", [])
                or (all(type(v) in (int, float) and math.isfinite(v) for v in (deviation, radius, limit))
                    and deviation - radius > limit)):
            plateau_failure = "horizon"
        elif tail.get("uncertainty_available") is False:
            plateau_failure = "sampling"
    return RateEvidence(
        identity.identity_id, source_ids=tuple(diagnostics.get("source_ids", ())),
        rate=rate if rate is not None and rate > 0 else None,
        pmf_log_standard_error=p_se, recrossing_log_standard_error=r_se,
        pmf_ready=pmf.get("ready"), overlap_connected=pmf.get("overlap_connected"),
        minimum_umbrella_ess=pmf.get("minimum_effective_samples"),
        independent_parent_chains=rec.get("independent_parent_chains"),
        effective_parent_blocks=blocks, plateau_established=rec.get("plateau_detected"),
        energy_complete=energy.get("complete"),
        maximum_energy_error_kbt=energy.get("maximum_endpoint_error_kbt_per_bead"),
        plateau_failure=plateau_failure,
        source_family_ids=(diagnostics["source_family_id"],) if diagnostics.get("source_family_id") else (),
        notes=tuple(diagnostics.get("missing", ())) + ("Legacy snapshot is pilot evidence only.",))


def _production_errors(spec, evidence, frozen):
    errors = []
    if frozen is None:
        return ["production_design_not_frozen"]
    try:
        validate_frozen_production(spec, frozen)
    except (ValueError, TypeError):
        return ["frozen_production_design_invalid"]
    if frozen.specification_id != spec.specification_id:
        errors.append("frozen_specification_mismatch")
    if evidence.epoch_id != frozen.epoch_id or evidence.design_id != frozen.design_id:
        errors.append("production_design_mismatch")
    n = len(evidence.replica_ids)
    expected = tuple(j["job_id"] for j in frozen.jobs if j["group"] == "baseline")[:n]
    if evidence.replica_ids != expected:
        errors.append("production_replicas_not_frozen_prefix")
    if not evidence.independent_from_pilot or set(evidence.source_ids).intersection(frozen.pilot_source_ids):
        errors.append("production_not_independent_from_pilot")
    if set(evidence.source_family_ids).intersection(frozen.pilot_source_family_ids):
        errors.append("production_reuses_pilot_source_family")
    if len(evidence.source_ids) < n or not n:
        errors.append("production_source_provenance_incomplete")
    if len(evidence.source_family_ids) != len(evidence.source_ids):
        errors.append("production_source_family_provenance_incomplete")
    if evidence.degrees_of_freedom is not None and evidence.degrees_of_freedom > n - 1:
        errors.append("degrees_of_freedom_exceed_independent_replicas")
    return errors


def assess_precision(spec, evidence, *, frozen=None, remaining_cost=None):
    """Assess one prespecified look and retain every unmet conclusion.

    ``statistical_target_met`` is an inferential production claim; pilot interval
    widths are diagnostic only. ``accepted`` additionally requires all requested
    numerical equivalence checks. No checks requested means numerical convergence
    remains unestablished, rather than becoming vacuously true.
    """
    if evidence.scientific_identity != spec.identity.identity_id:
        raise ValueError("Cannot pool different scientific identities")
    if remaining_cost is not None:
        _number(remaining_cost, "remaining_cost")
    conditions, reasons = [], list(evidence.notes)
    production_errors = _production_errors(spec, evidence, frozen) if evidence.phase == "production" else []
    due = evidence.phase == "production" and len(evidence.replica_ids) in spec.checks.replicas
    # Do not expose an inferential interval at an unplanned intermediate look.
    infer = evidence.phase == "pilot" or (due and not production_errors)
    uncertainty = combine_log_rate_uncertainty(
        evidence.pmf_log_standard_error, evidence.recrossing_log_standard_error,
        covariance=evidence.covariance, independent=evidence.components_independent,
        degrees_of_freedom=evidence.degrees_of_freedom)
    interval = (log_rate_interval(evidence.rate, uncertainty["log_rate_standard_error"],
                                alpha=spec.alpha_per_claim,
                                degrees_of_freedom=evidence.degrees_of_freedom)
                if infer else {"estimable": False, "interval": None,
                               "relative_half_width": None, "reasons": ["check_not_due_or_design_invalid"]})
    t = spec.target
    diagnostic_failures, missing = [], []
    for name, value in (("pmf_not_ready", evidence.pmf_ready),
                        ("overlap_disconnected", evidence.overlap_connected),
                        ("recrossing_plateau_not_established", evidence.plateau_established),
                        ("energy_diagnostics_incomplete", evidence.energy_complete)):
        if value is False:
            diagnostic_failures.append(name)
        elif value is None:
            missing.append(name + "_unknown")
    for name, value, limit in (
            ("insufficient_umbrella_effective_samples", evidence.minimum_umbrella_ess, t.minimum_umbrella_ess),
            ("insufficient_independent_parent_chains", evidence.independent_parent_chains, t.minimum_parent_chains),
            ("insufficient_effective_parent_blocks", evidence.effective_parent_blocks, t.minimum_effective_parent_blocks)):
        if value is None:
            missing.append(name + "_unknown")
        elif value < limit:
            diagnostic_failures.append(name)
    if evidence.maximum_energy_error_kbt is None:
        missing.append("energy_error_unknown")
    elif evidence.maximum_energy_error_kbt > t.maximum_energy_error_kbt:
        diagnostic_failures.append("energy_error_exceeds_tolerance")
    diagnostics_passed = not diagnostic_failures and not missing
    numerical, seen = [], set()
    for item in evidence.numerical:
        if item.requirement_id in seen:
            raise ValueError("Duplicate numerical evidence")
        seen.add(item.requirement_id)
    required = {r.requirement_id for r in spec.numerical}
    if seen - required:
        raise ValueError("Unplanned numerical comparison")
    for requirement in spec.numerical:
        item = next((x for x in evidence.numerical if x.requirement_id == requirement.requirement_id), None)
        candidate = spec.identity.refined(requirement.axis, requirement.value)
        valid = (item is not None and item.baseline_identity == spec.identity.identity_id
                 and item.candidate_identity == candidate.identity_id and item.covariance_accounted)
        if valid and evidence.phase == "production":
            expected = (tuple(j["job_id"] for j in frozen.jobs
                              if j["group"] == requirement.requirement_id)[:len(evidence.replica_ids)]
                        if frozen is not None and not production_errors else ())
            valid = (item.epoch_id == evidence.epoch_id and item.design_id == evidence.design_id
                     and item.replica_ids == expected and len(item.source_ids) == len(expected)
                     and bool(expected) and set(item.source_ids).issubset(evidence.source_ids)
                     and item.diagnostics_passed is True
                     and (item.degrees_of_freedom is None or item.degrees_of_freedom <= len(expected) - 1))
        result = numerical_equivalence(
            item.log_ratio if valid and infer else None,
            item.standard_error if valid and infer else None,
            log_margin=requirement.log_margin, alpha=spec.alpha_per_claim,
            degrees_of_freedom=item.degrees_of_freedom if valid and infer else None)
        numerical.append({"requirement_id": requirement.requirement_id, "axis": requirement.axis,
                          "candidate_identity": candidate.identity_id, **result})
    numerical_ok = bool(numerical) and all(x["established"] for x in numerical)
    numerical_established = (numerical_ok and evidence.phase == "production" and due
                             and not production_errors and diagnostics_passed)
    statistical = (evidence.phase == "production" and due and not production_errors
                   and diagnostics_passed and interval["estimable"]
                   and interval["relative_half_width"] <= t.relative_half_width)
    if production_errors:
        conditions.append("production_design_invalid")
    if diagnostic_failures:
        conditions.append("diagnostic_failed")
    if missing or not uncertainty["estimable"] or (infer and not interval["estimable"]):
        conditions.append("error_unestimable")
    if not numerical_established:
        conditions.append("numerical_convergence_not_established")
    if evidence.phase == "pilot":
        conditions.append("pilot_requires_independent_production")
    elif not due:
        conditions.append("check_not_due")
    if statistical:
        conditions.append("statistical_target_met")
    accepted = statistical and numerical_established
    if remaining_cost == 0 and not accepted:
        conditions.append("budget_exhausted")
    if (evidence.phase == "production" and frozen is not None and due
            and len(evidence.replica_ids) == spec.checks.replicas[-1] and not accepted):
        conditions.append("check_plan_exhausted")
    priority = ("production_design_invalid", "budget_exhausted", "diagnostic_failed",
                "check_not_due", "error_unestimable", "pilot_requires_independent_production",
                "numerical_convergence_not_established" if statistical else "check_plan_exhausted")
    status = "statistical_target_met" if accepted else next((s for s in priority if s in conditions), "continue_sampling")
    return {"format": "PyRPMD.precision.assessment.v1", "status": status,
            "conditions": conditions, "accepted": bool(accepted),
            "statistical_target_met": bool(statistical),
            "numerical_convergence_established": bool(numerical_established),
            "diagnostics_passed": diagnostics_passed, "diagnostic_failures": diagnostic_failures,
            "missing": missing, "production_errors": production_errors,
            "uncertainty": uncertainty, "interval": interval, "numerical": numerical,
            "alpha_per_claim": spec.alpha_per_claim,
            "repeated_check_method": "Bonferroni over all declared epochs, looks and claims; approximate cluster-t/log-delta intervals",
            "remaining_cost": remaining_cost,
            "reasons": reasons + production_errors + diagnostic_failures + missing,
            "scope": "requested statistical target and tested numerical pairs only; not PES or physical-model validation"}


def evidence_from_replicas(spec, frozen, records):
    """Build joint PMF/recrossing evidence from complete independent replicas.

    Each execution receipt supplies ``job_id, source_id, source_family_id,
    epoch_id, design_id, seed, sampling_design_id, independent_initialization,
    log_qtst, log_kappa, diagnostics``. Diagnostics is a RateEvidence. Every
    supplied replicate is retained, including failed diagnostics; filtering
    unsupported replicas to manufacture an interval is prohibited.
    """
    if frozen.specification_id != spec.specification_id:
        raise ValueError("Frozen specification mismatch")
    validate_frozen_production(spec, frozen)
    jobs = {j["job_id"]: j for j in frozen.jobs}
    groups = {}
    sources, families, seen = set(), set(), set()
    for record in records:
        job_id = record["job_id"]
        if job_id not in jobs or job_id in seen:
            raise ValueError("Unexpected or duplicate production job")
        seen.add(job_id)
        job = jobs[job_id]
        for name in ("source_id", "source_family_id"):
            _text(record.get(name), name)
        source, family = record["source_id"], record["source_family_id"]
        if (source in sources or family in families or source in frozen.pilot_source_ids
                or family in frozen.pilot_source_ids or family in frozen.pilot_source_family_ids):
            raise ValueError("Duplicate, overlapping or pilot production source")
        sources.add(source)
        families.add(family)
        if (record.get("epoch_id") != frozen.epoch_id or record.get("design_id") != frozen.design_id
                or record.get("seed") != job["seed"]
                or record.get("sampling_design_id") != job["sampling_design_id"]
                or record.get("independent_initialization") is not True):
            raise ValueError("Production receipt does not match the frozen independent design")
        diagnostic = record["diagnostics"]
        if diagnostic.scientific_identity != job["scientific_identity"]:
            raise ValueError("Production result scientific identity mismatch")
        groups.setdefault(job["group"], []).append(record)
    for values in groups.values():
        values.sort(key=lambda r: jobs[r["job_id"]]["replica_index"])
        if [jobs[r["job_id"]]["replica_index"] for r in values] != list(range(len(values))):
            raise ValueError("Production results must be an unfiltered prefix of frozen jobs")
    baseline = groups.get("baseline", [])
    if len(baseline) < 2:
        raise ValueError("At least two independent complete rate replicas are required")
    joint = paired_rate_replica_statistics([r["log_qtst"] for r in baseline],
                                          [r["log_kappa"] for r in baseline])
    log_rates = [r["log_qtst"] + r["log_kappa"] for r in baseline]
    rate = joint["rate"]
    def normalized_rates(logs):
        scaled = [math.exp(v - max(logs)) for v in logs]
        mean = sum(scaled) / len(scaled)
        return [v / mean for v in scaled], max(logs) + math.log(mean)
    normalized_base, log_rate = normalized_rates(log_rates)
    def group_diagnostics_passed(values):
        diagnostics = [r["diagnostics"] for r in values]
        t = spec.target
        return (all(d.pmf_ready is True and d.overlap_connected is True
                    and d.plateau_established is True and d.energy_complete is True
                    and d.minimum_umbrella_ess is not None and d.minimum_umbrella_ess >= t.minimum_umbrella_ess
                    and d.maximum_energy_error_kbt is not None and d.maximum_energy_error_kbt <= t.maximum_energy_error_kbt
                    and d.independent_parent_chains is not None and d.effective_parent_blocks is not None
                    for d in diagnostics)
                and sum(d.independent_parent_chains for d in diagnostics) >= t.minimum_parent_chains
                and sum(d.effective_parent_blocks for d in diagnostics) >= t.minimum_effective_parent_blocks)
    numerical = []
    for requirement in spec.numerical:
        other = groups.get(requirement.requirement_id, [])
        if len(other) != len(baseline):
            continue
        normalized_other, other_log_rate = normalized_rates([r["log_qtst"] + r["log_kappa"] for r in other])
        contrast = [v - normalized_base[i] for i, v in enumerate(normalized_other)]
        mean = sum(contrast) / len(contrast)
        se = math.sqrt(sum((v - mean) ** 2 for v in contrast) / (len(contrast) * (len(contrast) - 1)))
        numerical.append(NumericalEvidence(requirement.requirement_id, spec.identity.identity_id,
                         spec.identity.refined(requirement.axis, requirement.value).identity_id,
                         other_log_rate - log_rate, se, len(contrast) - 1, True,
                         epoch_id=frozen.epoch_id, design_id=frozen.design_id,
                         replica_ids=tuple(r["job_id"] for r in other),
                         source_ids=tuple(r["source_id"] for r in other),
                         diagnostics_passed=group_diagnostics_passed(other)))
    # Diagnostics from numerical cases also gate acceptance. Numerical equality
    # between two unstable trajectories cannot certify an RPMD rate.
    all_diagnostics = [r["diagnostics"] for values in groups.values() for r in values]
    def flag(name):
        values = [getattr(x, name) for x in all_diagnostics]
        return False if False in values else (None if None in values else True)
    def aggregate(name, operation, values=all_diagnostics):
        raw = [getattr(x, name) for x in values]
        return None if None in raw else operation(raw)
    def minimum_group_total(name):
        totals = [aggregate(name, sum, [r["diagnostics"] for r in values]) for values in groups.values()]
        return None if None in totals else min(totals)
    return RateEvidence(spec.identity.identity_id, "production", tuple(sorted(sources)), rate,
        math.sqrt(joint["pmf_variance"]), math.sqrt(joint["recrossing_variance"]), joint["covariance"],
        False, len(baseline) - 1,
        pmf_ready=flag("pmf_ready"), overlap_connected=flag("overlap_connected"),
        minimum_umbrella_ess=aggregate("minimum_umbrella_ess", min),
        independent_parent_chains=minimum_group_total("independent_parent_chains"),
        effective_parent_blocks=minimum_group_total("effective_parent_blocks"),
        plateau_established=flag("plateau_established"), energy_complete=flag("energy_complete"),
        maximum_energy_error_kbt=aggregate("maximum_energy_error_kbt", max),
        epoch_id=frozen.epoch_id, design_id=frozen.design_id,
        replica_ids=tuple(r["job_id"] for r in baseline), independent_from_pilot=True,
        source_family_ids=tuple(sorted(families)),
        numerical=tuple(numerical), notes=(
            "Arithmetic mean of complete-replica rates; paired product fluctuations retain PMF/recrossing covariance.",
            "Log-delta inference is approximate and does not remove finite-sample bias within each replica.",))
