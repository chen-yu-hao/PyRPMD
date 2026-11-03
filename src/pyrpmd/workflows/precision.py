"""Plan, execute and reconcile fixed thermal RPMD precision campaigns.

The CLI freezes independent production once. All launches use the existing
rate workflow and its RNG checkpoints, with a serial Python PES admission gate.
No execution option silently changes a requested device or propagation backend.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import tomllib

from ..input import JobSpec, rate_pmf_stage
from ..io import jsonable, referenced_files
from ..precision import (
    CheckSchedule, FrozenProduction, NumericalRequirement, PrecisionSpec,
    PrecisionTarget, RateEvidence, SamplingDesign, ScientificIdentity,
    evidence_from_replicas, freeze_production,
)
from ..precision_execution import (
    JobMeter, MeteredPotential, execution_lock, inside, read_json, read_sealed,
    write_sealed,
)
from ..precision_jobs import identity_from_config, materialize_rate_job, record_from_rate_job
from ..precision_ledger import PrecisionLedgerError, _digest, _json_value
from ..precision_planning import PrecisionController
from ..system import System


WORKFLOW_FORMAT = "PyRPMD.precision.workflow.v1"
REQUEST_FORMAT = "PyRPMD.precision.request.v1"
_ACCOUNTING = "PES frames admitted before provider dispatch; crash charges are retained"


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value > 2**53 - 1:
        raise ValueError(f"{name} exceeds the exactly representable ledger integer range")
    return value


def _mapping(value, name):
    if type(value) is not dict:
        raise ValueError(f"{name} must be an object")
    _json_value(value, name)
    return deepcopy(value)


def _request(value, identity):
    if isinstance(value, (str, Path)):
        path = Path(value)
        value = (tomllib.loads(path.read_text(encoding="utf-8"))
                 if path.suffix.lower() in (".toml", ".inp") else read_json(path))
    raw = _mapping(value, "precision specification")
    allowed = {"format", "target", "checks", "sampling", "numerical", "seed",
               "max_cost", "job_max_cost", "cost_unit"}
    if set(raw) - allowed:
        raise ValueError(f"Unknown precision fields: {sorted(set(raw) - allowed)}")
    if raw.get("format", REQUEST_FORMAT) != REQUEST_FORMAT:
        raise ValueError("Unsupported precision request format")
    if raw.get("cost_unit", "force_evaluations") != "force_evaluations":
        raise ValueError("Precision CLI supports cost_unit='force_evaluations' (PES frames) only")
    numerical = raw.get("numerical", [])
    if type(numerical) is not list:
        raise ValueError("numerical must be a list of predeclared comparisons")
    comparisons = []
    for item in numerical:
        item = _mapping(item, "numerical comparison")
        if item.get("axis") == "timestep" and isinstance(item.get("value"), str):
            from ..units import parse_quantity
            item["value"] = parse_quantity(item["value"], "time")
        comparisons.append(NumericalRequirement(**item))
    checks = CheckSchedule(**_mapping(raw.get("checks"), "checks"))
    if checks.max_epochs != 1:
        raise ValueError("Precision CLI executes one frozen epoch; max_epochs must be 1")
    spec = PrecisionSpec(
        identity=identity,
        target=PrecisionTarget(**_mapping(raw.get("target"), "target")),
        checks=checks, max_cost=_integer(raw.get("max_cost"), "max_cost", 0),
        numerical=tuple(comparisons), seed=raw.get("seed", 0),
        cost_unit="force_evaluations",
    )
    sampling = SamplingDesign(**_mapping(raw.get("sampling"), "sampling"))
    return spec, sampling, _integer(raw.get("job_max_cost"), "job_max_cost")


def _restore_spec(value):
    identity = {k: v for k, v in value["identity"].items() if k != "identity_id"}
    spec = PrecisionSpec(
        ScientificIdentity(**identity), PrecisionTarget(**value["target"]),
        CheckSchedule(**value["checks"]), value["max_cost"],
        numerical=tuple(NumericalRequirement(**item) for item in value["numerical"]),
        cost_unit=value["cost_unit"], seed=value["seed"],
    )
    if spec.to_dict() != value:
        raise PrecisionLedgerError("Saved precision specification is not canonical")
    return spec


def _validate_execution(config, system):
    from ..parallel import execution_plan
    from ..parallel.execution import _optional_world
    from .prepare import prepare_rate

    world = _optional_world()
    if world is not None and world.Get_size() != 1:
        raise ValueError("Precision CLI requires a single process, not an MPI launch")
    for scope in (config, rate_pmf_stage(config, system)):
        if scope is None:
            continue
        plan = execution_plan(scope.get("parallel", {}))
        if (plan.backend != "serial" or plan.mode != "serial"
                or plan.trajectory_workers != 1 or plan.bead_workers != 1
                or plan.trajectory_groups != 1 or plan.pes_ranks != 1):
            raise ValueError("Precision force metering requires serial execution with one worker")
        if scope.get("dynamics", {}).get("loop_backend") != "python":
            raise ValueError("Precision force metering requires explicit dynamics.loop_backend='python'")
        if scope.get("parallel", {}).get("trajectory_batch_size", 1) != 1:
            raise ValueError("Precision force metering requires trajectory_batch_size=1")
        if scope.get("resume") or scope.get("_resume"):
            raise ValueError("Plan precision production from an unresumed input")
    rec = config.get("recrossing", config.get("rate", {}).get("recrossing", {}))
    if rec.get("child_batch_pairs", 1) != 1:
        raise ValueError("Precision force metering requires child_batch_pairs=1")
    prepared = prepare_rate(config, system, numerical=False)
    if not prepared["ready"]:
        raise ValueError("Complete numerical TS preparation and freeze the geometry before precision planning")


def _load(directory):
    root = Path(directory).resolve()
    path = inside(root, "precision", "workflow.json")
    if not path.is_file():
        raise ValueError(f"No precision campaign in {root}; use 'pyrpmd precision plan' first")
    saved = read_sealed(path)
    if saved.get("format") != WORKFLOW_FORMAT:
        raise PrecisionLedgerError("Unsupported precision workflow format")
    spec = _restore_spec(saved["specification"])
    frozen = FrozenProduction.from_dict(spec, saved["frozen"])
    if saved["execution"] != {
        "cost_unit": "force_evaluations", "job_max_cost": saved["execution"].get("job_max_cost"),
        "accounting": _ACCOUNTING, "executor": "serial-python-v1",
    }:
        raise PrecisionLedgerError("Unsupported precision execution contract")
    _integer(saved["execution"]["job_max_cost"], "job_max_cost")
    _integer(spec.max_cost, "max_cost", 0)
    if spec.cost_unit != "force_evaluations" or spec.checks.max_epochs != 1:
        raise PrecisionLedgerError("Unsupported saved precision budget or epoch schedule")
    return root, saved, spec, frozen, System.from_dict(saved["system"])


def _controller(root, saved, spec, frozen):
    ledger_path = inside(root, "precision", "ledger.json")
    if not ledger_path.exists() and inside(root, "precision", "jobs").exists():
        raise PrecisionLedgerError("Precision ledger is missing after execution began")
    controller = PrecisionController(root, spec)
    state = controller.ledger.snapshot()
    key = "freeze:" + frozen.epoch_id
    if key not in state["checks"]:
        if state["plans"] or state["stop"]:
            raise PrecisionLedgerError("Precision production freeze is missing")
        controller.freeze(frozen.sampling, epoch_index=frozen.epoch_index)
    elif state["checks"][key] != frozen.to_dict():
        raise PrecisionLedgerError("Ledger and workflow production freezes differ")
    return controller


def _next_plan(controller, frozen, cap):
    state = controller.ledger.snapshot()
    if state["stop"] is not None:
        return None
    entries = list(state["plans"].values())
    if any(e["plan"].get("production_design_id") != frozen.design_id for e in entries):
        raise PrecisionLedgerError("Precision CLI cannot execute foreign or pilot ledger plans")
    pending = [e for e in entries if e["state"] in ("planned", "running")]
    if len(pending) > 1:
        raise PrecisionLedgerError("Precision CLI requires one outstanding increment")
    if pending:
        plan = pending[0]["plan"]
        expected = len(plan["jobs"]) * cap
        if (plan["estimated_cost"] != expected or plan["execution_budget"] != {
                "cost_unit": "force_evaluations", "max_cost": float(expected),
                "enforce_in_executor": True}):
            raise PrecisionLedgerError("Saved plan budget differs from its frozen per-job caps")
        return plan
    if any(e["state"] == "failed" for e in entries):
        return None
    previous = max((e["plan"]["through_replicas"] for e in entries), default=0)
    through = next((n for n in controller.spec.checks.replicas if n > previous), None)
    if through is None:
        return None
    jobs = [j for j in frozen.jobs if previous <= j["replica_index"] < through]
    plan = controller.plan_production(frozen, through_replicas=through,
                                      estimated_cost=len(jobs) * cap)
    return plan if plan.get("status") == "planned" else None


def plan_precision(job, request, *, output):
    """Resolve a normal .inp/.toml/.py rate input and persist its first plan."""
    if not isinstance(job, JobSpec):
        job = (JobSpec.from_file(job) if isinstance(job, (str, Path))
               else JobSpec.from_mapping(job))
    config, system = job.resolve()
    return plan_precision_resolved(config, system, request, output=output)


def plan_precision_resolved(config, system, request, *, output):
    """Freeze and quote production without constructing or evaluating a PES."""
    _validate_execution(config, system)
    spec, sampling, cap = _request(request, identity_from_config(config, system))
    frozen = freeze_production(spec, sampling)
    # Validate every controlled clock and route before writing any campaign.
    for planned in frozen.jobs:
        job = materialize_rate_job(config, system, frozen, planned["job_id"])
        _validate_execution(job["config"], system)
    saved = jsonable({
        "format": WORKFLOW_FORMAT, "config": config, "system": system.to_dict(),
        "specification": spec.to_dict(), "frozen": frozen.to_dict(),
        "input_files": referenced_files(config),
        "execution": {"cost_unit": "force_evaluations", "job_max_cost": cap,
                      "executor": "serial-python-v1", "accounting": _ACCOUNTING},
    }, callables=False)
    _json_value(saved, "precision workflow")
    root = Path(output).resolve()
    if root.exists() and not inside(root, "precision", "workflow.json").exists():
        # An interrupted initial publication may have created only its lock.
        unexpected = [p for p in root.rglob("*") if p.is_file()
                      and p != root / "precision" / "execution.lock"]
        if unexpected:
            raise ValueError("Precision output directory is not empty or owned by this campaign")
    with execution_lock(root):
        path = inside(root, "precision", "workflow.json")
        if path.exists():
            if read_sealed(path) != saved:
                raise PrecisionLedgerError("Cannot change a frozen precision campaign")
        else:
            write_sealed(path, saved)
        controller = _controller(root, saved, spec, frozen)
        # Replaying plan after a completed receipt must consume its scheduled
        # look before any later increment can reserve or spend more budget.
        _check_completed(controller, frozen)
        _next_plan(controller, frozen, cap)
        return precision_status(root)


def _execution_identity(saved, plan, job):
    execution_id = "precision:" + plan["plan_id"]
    return {
        "workflow_id": _digest(saved), "plan_id": plan["plan_id"],
        "execution_id": execution_id, "job_id": job["job_id"],
        "physics_fingerprint": job["expected_physics_fingerprint"],
    }


def _journal_path(root, job_id):
    return inside(root, "precision", "jobs", job_id + ".json")


def _initialize_job(root, directory, config, system):
    """Publish complete initial metadata before any dynamics can start.

    An interrupted staging directory contains no trajectories and is never
    adopted as a job. The published input still records resume=False; normal
    run_resolved recovery thereafter preserves that fresh-production receipt.
    """
    from ..runner import _initialize_output

    if directory.exists():
        if not (directory / "manifest.json").is_file():
            raise PrecisionLedgerError("Owned precision job has no complete run manifest")
        return
    parent = inside(root, "jobs")
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".initializing-", dir=parent))
    config = deepcopy(config)
    config["output"]["directory"] = str(directory)
    config["resume"] = False
    _initialize_output(config, system, staging, resume=False)
    os.replace(staging, directory)


def _portable_record(directory, job, frozen):
    record = record_from_rate_job(directory, job, frozen)
    return {**record, "diagnostics": record["diagnostics"].to_dict()}


def _execute_job(root, saved, frozen, plan, planned, system):
    from ..parallel.execution import _task_potential_transform
    from ..runner import run_resolved

    job = materialize_rate_job(saved["config"], system, frozen, planned["job_id"])
    directory = inside(root, "jobs", job["job_id"])
    path = _journal_path(root, job["job_id"])
    if directory.exists() and not path.exists():
        raise PrecisionLedgerError("Refusing to adopt a rate directory without its execution identity")
    meter = JobMeter(path, identity=_execution_identity(saved, plan, job),
                     max_cost=saved["execution"]["job_max_cost"])
    if meter.data["state"] == "failed":
        return meter.data
    if meter.data["blocked_request"] is not None:
        meter.fail("budget_exhausted", "The next PES batch cannot fit the frozen job budget")
        return meter.data
    if meter.data["state"] == "completed":
        if _portable_record(directory, job, frozen) != meter.data["record"]:
            raise PrecisionLedgerError("Completed precision artifacts differ from their immutable receipt")
        return meter.data
    # Both summaries are atomic completion publications. Authentication failure
    # is a failure, never a reason to overwrite or relaunch a complete result.
    published = ((directory / "summary.json").is_file()
                 or (directory / "results" / "summary.json").is_file())
    if not published:
        _initialize_job(root, directory, job["config"], system)
        try:
            with _task_potential_transform(lambda provider: MeteredPotential(provider, meter)):
                run_resolved(job["config"], system, output=directory, resume=True)
        except (OSError, KeyboardInterrupt):
            # An interrupted filesystem/provider operation remains reconcilable.
            # In particular, never refund admissions to the previous checkpoint.
            raise
        except Exception as exc:
            status = "budget_exhausted" if meter.data["blocked_request"] else "execution_failed"
            meter.fail(status, "The next PES batch cannot fit the frozen job budget"
                       if status == "budget_exhausted" else str(exc) or type(exc).__name__)
            return meter.data
    try:
        record = _portable_record(directory, job, frozen)
    except (ValueError, KeyError, TypeError) as exc:
        meter.fail("diagnostic_failed", str(exc))
        return meter.data
    meter.complete(record)
    return meter.data


def _plan_jobs(root, plan):
    return [read_sealed(_journal_path(root, job["job_id"])) for job in plan["jobs"]
            if _journal_path(root, job["job_id"]).is_file()]


def _finish_failure(controller, plan, jobs):
    failed = next(j for j in jobs if j["state"] == "failed")
    records = [j["record"] for j in jobs if j["state"] == "completed"]
    failure = failed["failure"]
    result = {"job_ids": [r["job_id"] for r in records], "records": records,
              "failure": failure, "accounting": _ACCOUNTING}
    controller.ledger.fail(
        plan["plan_id"], execution_id="precision:" + plan["plan_id"],
        actual_cost=sum(j["charged_cost"] for j in jobs),
        result_ids=[r["source_id"] for r in records], result=result,
        reason=failure["reason"],
    )
    _stop_failed(controller, failure)


def _stop_failed(controller, failure):
    state = controller.ledger.snapshot()
    if state["stop"] is not None:
        return
    assessment = _latest_assessment(state)
    conclusions = {key: bool((assessment or {}).get(key, False)) for key in (
        "statistical_target_met", "numerical_convergence_established")}
    # A previous look cannot authorize discarding a later failed replica.
    conclusions["accepted"] = False
    controller.ledger.stop(failure["status"], [failure["reason"]], conclusions)


def _latest_assessment(state):
    for event in reversed(state["events"]):
        if event["kind"] == "check_recorded":
            value = state["checks"][event["check_id"]]
            if "assessment" in value:
                return value["assessment"]
    return None


def _check_completed(controller, frozen):
    state = controller.ledger.snapshot()
    records = []
    entries = list(state["plans"].values())
    if any(e["state"] != "completed" for e in entries):
        return
    for entry in entries:
        for record in entry["receipt"]["result"]["records"]:
            records.append({**record, "diagnostics": RateEvidence.from_dict(record["diagnostics"])})
    if records:
        controller.check(evidence_from_replicas(controller.spec, frozen, records), frozen=frozen)


def run_precision(directory, *, max_jobs=None):
    """Reconcile first, then execute successive predeclared looks until stopped.

    ``max_jobs`` is an execution-only cooperative pause after complete jobs. A
    hard interruption may instead resume the existing per-task RNG checkpoints.
    Neither kind of interruption creates a new independent scientific sample.
    """
    if max_jobs is not None:
        _integer(max_jobs, "max_jobs")
    root, saved, spec, frozen, system = _load(directory)
    _validate_execution(saved["config"], system)
    if referenced_files(saved["config"]) != saved["input_files"]:
        raise PrecisionLedgerError("Precision input/model file changed or is missing")
    if identity_from_config(saved["config"], system) != spec.identity:
        raise PrecisionLedgerError("Saved precision scientific identity differs")
    with execution_lock(root):
        controller = _controller(root, saved, spec, frozen)
        completed_now = 0
        while True:
            state = controller.ledger.snapshot()
            if state["stop"] is not None:
                return precision_status(root)
            failed = [e for e in state["plans"].values() if e["state"] == "failed"]
            if failed:
                _stop_failed(controller, failed[0]["receipt"]["result"]["failure"])
                return precision_status(root)
            # Also recovers a crash after the increment receipt, before check.
            _check_completed(controller, frozen)
            plan = _next_plan(controller, frozen, saved["execution"]["job_max_cost"])
            if plan is None:
                return precision_status(root)
            if max_jobs is not None and completed_now >= max_jobs:
                return {**precision_status(root), "status": "paused"}
            controller.ledger.start(plan["plan_id"], execution_id="precision:" + plan["plan_id"])
            for planned in plan["jobs"]:
                prior = _journal_path(root, planned["job_id"])
                was_complete = prior.is_file() and read_sealed(prior)["state"] == "completed"
                if not was_complete and max_jobs is not None and completed_now >= max_jobs:
                    return {**precision_status(root), "status": "paused"}
                data = _execute_job(root, saved, frozen, plan, planned, system)
                if data["state"] == "failed":
                    _finish_failure(controller, plan, _plan_jobs(root, plan))
                    return precision_status(root)
                if not was_complete:
                    completed_now += 1
            jobs = _plan_jobs(root, plan)
            records = [j["record"] for j in jobs]
            controller.ledger.complete(
                plan["plan_id"], execution_id="precision:" + plan["plan_id"],
                actual_cost=sum(j["charged_cost"] for j in jobs),
                result_ids=[r["source_id"] for r in records],
                result={"job_ids": [r["job_id"] for r in records], "records": records,
                        "accounting": _ACCOUNTING},
            )


def precision_status(directory):
    """Read durable progress and conclusions without launching or making plans."""
    root, saved, spec, frozen, _ = _load(directory)
    ledger = inside(root, "precision", "ledger.json")
    if not ledger.is_file():
        return {"status": "initialization_incomplete", "directory": str(root),
                "specification_id": spec.specification_id, "stop": None,
                "jobs": [], "budget": None, "latest_assessment": None}
    controller = PrecisionController(root, spec)
    state = controller.ledger.snapshot()
    jobs = []
    unsettled = 0
    planned_entries = {job["job_id"]: entry for entry in state["plans"].values()
                       for job in entry["plan"].get("jobs", [])}
    for planned in frozen.jobs:
        path = _journal_path(root, planned["job_id"])
        entry = planned_entries.get(planned["job_id"])
        # A writer can reserve the next look after our atomic ledger snapshot.
        # Describe that snapshot rather than mixing in its newer job journals.
        data = read_sealed(path) if entry is not None and path.is_file() else None
        jobs.append({"job_id": planned["job_id"], "group": planned["group"],
                     "replica_index": planned["replica_index"],
                     "directory": str(inside(root, "jobs", planned["job_id"])),
                     "state": "not_started" if data is None else data["state"],
                     "charged_cost": 0 if data is None else data["charged_cost"],
                     "evaluation_batches": 0 if data is None else data["evaluation_batches"]})
        if data is not None:
            if entry["plan"]["plan_id"] != data["identity"]["plan_id"]:
                raise PrecisionLedgerError("Job execution has no registered precision plan")
            if entry["state"] in ("planned", "running"):
                unsettled += data["charged_cost"]
    budget = {**state["budget"], "cost_unit": spec.cost_unit, "accounting": _ACCOUNTING,
              "unsettled_cost": unsettled,
              "charged_cost": state["budget"]["spent_cost"] + unsettled}
    stop = state["stop"]
    status = (stop["status"] if stop else
              "running" if state["budget"]["pending_reconciliation"] else "planned")
    return {"status": status, "directory": str(root),
            "specification_id": spec.specification_id, "design_id": frozen.design_id,
            "budget": budget, "jobs": jobs, "stop": stop,
            "latest_assessment": _latest_assessment(state),
            "completed_results": len(state["results"])}


def _command(args):
    from ..cli import _print

    try:
        if args.precision_command == "plan":
            result = plan_precision(args.input, args.spec, output=args.output)
        elif args.precision_command == "run":
            result = run_precision(args.directory, max_jobs=args.max_jobs)
        else:
            result = precision_status(args.directory)
    except KeyboardInterrupt:
        # All admissions and completed records were already committed. The
        # existing execution remains running/reserved until the next run.
        if args.precision_command != "run":
            raise
        _print({**precision_status(args.directory), "status": "interrupted"})
        return 130
    _print(result)
    return 0


def register_parser(commands):
    """Keep all precision arguments and handlers out of the shared CLI."""
    parser = commands.add_parser("precision", help="Plan, run and inspect precision-driven thermal rates")
    actions = parser.add_subparsers(dest="precision_command", required=True)
    plan = actions.add_parser("plan", help="Freeze and reserve production without dynamics")
    plan.add_argument("input", help="An existing thermal rate .inp/.toml or Python input")
    plan.add_argument("--spec", required=True, help="Precision targets, sampling and budgets (JSON/TOML)")
    plan.add_argument("--output", required=True, help="New or identical precision campaign directory")
    run = actions.add_parser("run", help="Execute or reconcile the saved campaign")
    run.add_argument("directory")
    run.add_argument("--max-jobs", type=int, help="Pause after this many newly completed jobs")
    status = actions.add_parser("status", help="Read saved budget, progress and stop conclusions")
    status.add_argument("directory")
    for action in actions.choices.values():
        action.set_defaults(_precision_handler=_command)
        action.add_argument("--debug", action="store_true", default=argparse.SUPPRESS,
                            help="Show a traceback when a command fails")
