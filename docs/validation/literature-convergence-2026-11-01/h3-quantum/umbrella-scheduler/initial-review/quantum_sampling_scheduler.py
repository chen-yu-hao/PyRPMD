"""Schedule the complete registered umbrella cohort without changing sampling.

Only this coordinator writes the global completion manifest. Workers call the
unchanged quantum_sampling_campaign.one_seed and own separate seed directories.
Prior-stage files, sample/diagnostic prefixes, sources and settings are guarded.
Failures stop admissions and drain workers; no partial cohort is called complete.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import signal
import sys
import tempfile
import threading
import time
import uuid


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (211017, 211019, 211021, 211023, 211027, 211031, 211033, 211039)
PREFIXES = (30000, 100000, 300000)
DRIVER = "benchmarks/h3/quantum_sampling_campaign.py"
STATE_KEYS = ("R", "r", "completed", "rng", "accepted", "proposed", "swaps", "lineage")
SNAPSHOT_SCHEMA = "pyrpmd.h3.quantum-umbrella-stage-snapshot.v1"
COMPLETION_SCHEMA = "pyrpmd.h3.quantum-umbrella-scheduler-completion.v1"
FAILURE_SCHEMA = "pyrpmd.h3.quantum-umbrella-scheduler-failure.v1"


class CampaignFailure(RuntimeError):
    def __init__(self, path):
        self.manifest_path = Path(path)
        super().__init__(f"No complete-cohort manifest was written; drained failure receipt: {path}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def byte_hash(value):
    return hashlib.sha256(value).hexdigest()


def record(path, base=None):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path.relative_to(base)) if base is not None else str(path),
            "sha256": byte_hash(data), "bytes": len(data)}


def relative_path(base, name):
    parsed = PurePosixPath(name)
    require(not parsed.is_absolute() and ".." not in parsed.parts and "\\" not in name,
            "unsafe registered relative path")
    path = Path(base) / name
    require(path.resolve().is_relative_to(Path(base).resolve()), "registered path escapes its root")
    return path


def read_bound(path, expected):
    data = Path(path).read_bytes()
    require(byte_hash(data) == expected, f"immutable input SHA differs: {path}")
    return json.loads(data)


def atomic_new_json(path, value):
    """Publish a complete new JSON file atomically; never replace an old one."""
    path = Path(path)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def campaign_lock(campaign):
    """An unreaped lock after a hard crash requires inspection, never auto-break."""
    lock = Path(campaign) / ".quantum-scheduler-lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"Campaign is already locked; do not run overlapping drivers: {lock}") from error
    try:
        atomic_new_json(lock / "owner.json", {"pid": os.getpid(), "created_utc": now()})
        yield
    finally:
        (lock / "owner.json").unlink(missing_ok=True)
        lock.rmdir()


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_guards(guards):
    for path, expected in guards.items():
        require(file_hash(path) == expected, f"registered source/input changed: {path}")


def array_description(value):
    import numpy as np
    require(not value.dtype.hasobject, "object arrays are forbidden")
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "sha256": byte_hash(np.ascontiguousarray(value).tobytes())}


def inspect_seed(campaign, prior, stage, protocol, protocol_sha, driver_sha):
    """Inspect one complete checkpoint inode and all immutable earlier prefixes."""
    import numpy as np
    campaign = Path(campaign)
    directory = relative_path(campaign, prior["directory"])
    settings_path = relative_path(campaign, prior["settings"]["path"])
    settings_bytes = settings_path.read_bytes()
    require(byte_hash(settings_bytes) == prior["settings"]["sha256"], "settings bytes changed")
    settings = json.loads(settings_bytes)
    require(settings["seed"] == prior["seed"] and settings["beads"] == protocol["beads_primary"]
            and settings["warmup"] == protocol["warmup_sweeps"]
            and settings["production"] == protocol["fixed_refinement_prefixes"][-1], "settings/protocol differ")
    checkpoint_path = directory / "checkpoint.npz"
    data = checkpoint_path.read_bytes()
    with np.load(io.BytesIO(data), allow_pickle=False) as saved:
        require(len(saved.files) == len(set(saved.files)) and set(saved.files) == set(prior["arrays"]),
                "checkpoint array inventory changed")
        arrays = {name: saved[name] for name in saved.files}
    descriptions = {name: array_description(value) for name, value in arrays.items()}
    count = len(arrays["samples"])
    previous_count = prior["arrays"]["samples"]["shape"][0]
    require(previous_count <= count <= stage, "checkpoint progress outside requested extension")
    require(int(arrays["completed"]) == settings["warmup"] + count, "checkpoint sweep count differs")
    n = len(settings["centers"]); interval = settings["diagnostic_interval"]
    require(arrays["samples"].shape == (count, n) and
            arrays["diagnostics"].shape == ((count + interval - 1) // interval, n, 7), "observation shape differs")
    require(arrays["R"].shape == arrays["r"].shape == (n, settings["beads"], 3), "ring geometry shape differs")
    require(np.isfinite(arrays["samples"]).all() and np.isfinite(arrays["diagnostics"]).all()
            and np.isfinite(arrays["R"]).all() and np.isfinite(arrays["r"]).all(), "nonfinite checkpoint data")
    json.loads(str(arrays["rng"].item()))
    stages = []
    prior_numbers = {item["production_sweeps"] for item in prior["stage_receipts"]}
    for old in prior["stage_receipts"]:
        receipt_path = relative_path(campaign, old["receipt"]["path"])
        state_path = relative_path(campaign, old["state"]["path"])
        receipt = read_bound(receipt_path, old["receipt"]["sha256"])
        require(file_hash(state_path) == old["state"]["sha256"], "prior restart state bytes changed")
        size = old["production_sweeps"]
        require(array_description(arrays["samples"][:size])["sha256"] == old["samples_prefix_sha256"],
                "prior sample prefix changed")
        require(array_description(arrays["diagnostics"][:(size + interval - 1) // interval])["sha256"]
                == old["diagnostics_prefix_sha256"], "prior diagnostic prefix changed")
        require(receipt["protocol_sha256"] == protocol_sha and receipt["driver_sha256"] == driver_sha,
                "prior receipt source identity differs")
        stages.append(old)
    actual_stage_dirs = {path.name for path in (directory / "stages").iterdir() if path.is_dir()}
    require(actual_stage_dirs <= {str(x) for x in prior_numbers | {stage}}, "unregistered stage directory")
    destination = directory / "stages" / str(stage)
    target_receipt = None
    if destination.exists() and stage not in prior_numbers:
        require(count == stage, "target receipt exists before its sample count")
        target_receipt = json.loads((destination / "stage.json").read_text())
        require(target_receipt["seed"] == prior["seed"] and target_receipt["production_sweeps"] == stage,
                "target receipt seed/stage differs")
        require(target_receipt["protocol_sha256"] == protocol_sha and target_receipt["driver_sha256"] == driver_sha,
                "target receipt source differs")
        require(target_receipt["checkpoint_sha256"] == byte_hash(data), "target checkpoint/receipt differs")
        require(target_receipt["state_sha256"] == file_hash(destination / "state.npz"), "target restart-state SHA differs")
        require(target_receipt["samples_prefix_sha256"] == descriptions["samples"]["sha256"] and
                target_receipt["diagnostics_prefix_sha256"] == descriptions["diagnostics"]["sha256"],
                "target sample/diagnostic receipt differs")
        with np.load(destination / "state.npz", allow_pickle=False) as state:
            require(set(state.files) == set(STATE_KEYS), "target state inventory differs")
            for key in STATE_KEYS:
                require(array_description(state[key]) == descriptions[key], "target restart/RNG state differs")
        stages.append({"production_sweeps": stage,
                       "receipt": record(destination / "stage.json", campaign),
                       "state": record(destination / "state.npz", campaign),
                       "samples_prefix_sha256": descriptions["samples"]["sha256"],
                       "diagnostics_prefix_sha256": descriptions["diagnostics"]["sha256"]})
    report_path = directory / "run-report.json"
    report = json.loads(report_path.read_text())
    if target_receipt is not None:
        require(report["checkpoint_sha256"] == byte_hash(data) and report["source_unchanged"]
                and report["completed_sweeps"] == settings["warmup"] + stage, "target sampler report differs")
    return {"seed": prior["seed"], "directory": prior["directory"],
            "checkpoint": {"path": str(checkpoint_path.relative_to(campaign)), "sha256": byte_hash(data), "bytes": len(data)},
            "settings": record(settings_path, campaign), "run_report": record(report_path, campaign),
            "arrays": descriptions, "completed_sweeps": int(arrays["completed"]),
            "stage_receipts": stages, "source_sha256": settings["source_sha256"],
            "legacy_NPZ_without_array_envelope": True}


def load_snapshot(path, expected_sha):
    value = read_bound(path, expected_sha)
    if value.get("schema") == COMPLETION_SCHEMA:
        value = value["snapshot"]
    require(value.get("schema") == SNAPSHOT_SCHEMA, "not a complete prior-stage snapshot")
    return value


def preflight(campaign, library, protocol_path, snapshot_path, snapshot_sha, decision_path,
              decision_sha, stage, *, resume_path=None, resume_sha=None, root=ROOT):
    """No worker or write occurs until the entire cohort has passed inspection."""
    campaign, library, root = Path(campaign).resolve(), Path(library).resolve(), Path(root).resolve()
    prior = load_snapshot(snapshot_path, snapshot_sha)
    protocol = read_bound(protocol_path, prior["protocol"]["sha256"])
    decision = read_bound(decision_path, decision_sha)
    require(tuple(protocol["independent_ladder_seeds"]) == tuple(prior["independent_seeds"]) == SEEDS,
            "the fixed complete eight-seed cohort is required")
    require(tuple(protocol["fixed_refinement_prefixes"]) == PREFIXES, "registered prefixes changed")
    require(stage in PREFIXES[1:] and prior["production_sweeps"] == PREFIXES[PREFIXES.index(stage) - 1],
            "only the next registered 100000/300000 prefix may be scheduled")
    require(decision["from_production_sweeps"] == prior["production_sweeps"]
            and decision["to_production_sweeps"] == stage
            and tuple(decision["independent_seeds"]) == SEEDS
            and decision["protocol_sha256"] == prior["protocol"]["sha256"], "continuation decision mismatch")
    require(decision["trigger"]["criterion"] == "statistical_precision"
            and decision["trigger"]["passed"] is False
            and decision["trigger"]["observed_primary_log_standard_error"] > decision["trigger"]["registered_limit"]
            and decision["trigger"]["registered_limit"] == protocol["gates"]["target_qtst_relative_standard_error"]
            and decision["point_agreement_does_not_trigger_or_stop_continuation"] is True,
            "a matching precision-only continuation decision is required")
    require(file_hash(decision["completed_cohort_report_path"]) == decision["completed_cohort_report_sha256"],
            "decision's completed cohort report changed")
    complete_report = json.loads(Path(decision["completed_cohort_report_path"]).read_text())
    require(tuple(complete_report["completed_independent_seeds"]) == tuple(complete_report["expected_independent_seeds"]) == SEEDS
            and complete_report["production_sweeps_per_ladder"] == prior["production_sweeps"]
            and complete_report["protocol_sha256"] == prior["protocol"]["sha256"]
            and complete_report["statistics"]["primary_log_standard_error"] == decision["trigger"]["observed_primary_log_standard_error"]
            and complete_report["criteria"]["statistical_precision"] is False, "decision/report evidence mismatch")
    guards = {str(relative_path(root, name)): value for name, value in prior["source_sha256"].items()}
    guards.update({str(library): prior["model_sha256"], str(Path(protocol_path).resolve()): prior["protocol"]["sha256"],
                   str(Path(snapshot_path).resolve()): snapshot_sha, str(Path(decision_path).resolve()): decision_sha,
                   str(relative_path(root, protocol["schedule_file"])): protocol["schedule_sha256"],
                   str(Path(__file__).resolve()): file_hash(__file__),
                   str(Path(decision["completed_cohort_report_path"]).resolve()): decision["completed_cohort_report_sha256"]})
    if "completed_cohort_execution_sha256" in decision:
        execution = Path(decision["completed_cohort_report_path"]).parent / "execution.json"
        guards[str(execution.resolve())] = decision["completed_cohort_execution_sha256"]
    require(guards[str(relative_path(root, protocol["sampler_file"]))] == protocol["sampler_sha256"],
            "sampler provenance differs")
    verify_guards(guards)
    if prior.get("archive"):
        require(file_hash(prior["archive"]["path"]) == prior["archive"]["sha256"], "prior raw archive changed or missing")
    require({path.name for path in campaign.glob("seed-*") if path.is_dir()} == {f"seed-{seed}" for seed in SEEDS},
            "campaign contains missing or extra seed directories")
    require([row["seed"] for row in prior["seeds"]] == list(SEEDS), "prior snapshot seed inventory differs")
    recovery = None
    if resume_path is not None:
        require(resume_sha is not None, "resumption requires the explicit failure-receipt SHA")
        recovery = read_bound(resume_path, resume_sha)
        require(recovery["schema"] == FAILURE_SCHEMA and recovery["stage"] == stage
                and recovery["prior_snapshot_sha256"] == snapshot_sha and recovery["decision_sha256"] == decision_sha
                and recovery["scheduler_source_sha256"] == file_hash(__file__), "recovery receipt identity differs")
        guards[str(Path(resume_path).resolve())] = resume_sha
    current = []
    for item in prior["seeds"]:
        row = inspect_seed(campaign, item, stage, protocol, prior["protocol"]["sha256"], prior["source_sha256"][DRIVER])
        expected = item if recovery is None else next(x for x in recovery["checkpoints_after_drain"] if x["seed"] == item["seed"])
        require("integrity_error" not in expected and row["checkpoint"] == expected["checkpoint"]
                and row["settings"] == expected["settings"] and row["run_report"] == expected["run_report"]
                and row["stage_receipts"] == expected["stage_receipts"],
                "unrecorded progress/tamper; resume only from a fully drained failure receipt")
        settings = json.loads(relative_path(campaign, row["settings"]["path"]).read_text())
        schedule = json.loads(relative_path(root, protocol["schedule_file"]).read_text())
        require(settings["centers"] == schedule["centers"] and settings["stiffness"] == schedule["stiffness"],
                "registered schedule/settings differ")
        for name, expected_sha in settings["source_sha256"].items():
            path = library if name == "bkmp2_library" else relative_path(root, name)
            require(guards.get(str(path)) == expected_sha, "per-seed source/model identity differs")
        current.append(row)
    return prior, protocol, guards, current


def _configure_worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _frozen_worker(job):
    from benchmarks.h3.quantum_sampling_campaign import one_seed
    return one_seed(job)


def verify_imported_sources(guards):
    """A matching file on disk must also be the module actually imported."""
    for name, expected in guards.items():
        path = Path(name)
        if path.suffix != ".py" or not path.is_relative_to(ROOT):
            continue
        relative = path.relative_to(ROOT)
        parts = relative.with_suffix("").parts
        if parts[0] == "src":
            parts = parts[1:]
        elif parts[0] != "benchmarks":
            continue
        module = importlib.import_module(".".join(parts))
        actual = Path(module.__file__).resolve()
        require(actual == path.resolve() and file_hash(actual) == expected,
                f"imported source differs from registered file: {name}")


def _execute_job(worker, job, guards):
    verify_guards(guards)
    if worker is _frozen_worker:
        verify_imported_sources(guards)
    result = worker(job)
    verify_guards(guards)
    return result


@contextmanager
def stop_signals():
    state = {"reason": None}
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda number, frame: state.update(reason=f"signal {number}; drain requested"))
    try:
        yield state
    finally:
        for signum, handler in previous.items(): signal.signal(signum, handler)


def worker_limit(workers, control):
    if control is None:
        return workers, None
    data = Path(control).read_bytes()
    value = json.loads(data)
    require(set(value) == {"active_workers"} and type(value["active_workers"]) is int
            and 1 <= value["active_workers"] <= 4, "worker control must contain only active_workers, an integer 1..4")
    return value["active_workers"], byte_hash(data)


def run_campaign(campaign, library, protocol_path, snapshot_path, snapshot_sha, decision_path,
                 decision_sha, stage, *, workers=2, control=None, resume_path=None,
                 resume_sha=None, check_only=False, root=ROOT, worker=_frozen_worker):
    require(type(workers) is int and 1 <= workers <= 4, "workers must be 1..4")
    campaign = Path(campaign).resolve()
    final_path = campaign / f"stage-{stage}.json"
    with campaign_lock(campaign):
        require(not final_path.exists(), "global stage manifest already exists; it will not be overwritten")
        prior, protocol, guards, initial = preflight(campaign, library, protocol_path, snapshot_path,
            snapshot_sha, decision_path, decision_sha, stage, resume_path=resume_path, resume_sha=resume_sha, root=root)
        initial_limit, _ = worker_limit(workers, control)
        if check_only:
            if worker is _frozen_worker:
                verify_imported_sources(guards)
            return {"preflight_passed": True, "stage": stage, "independent_seeds": list(SEEDS),
                    "initial_worker_limit": initial_limit, "workers_launched": 0,
                    "source_guard_count": len(guards), "prior_snapshot_sha256": snapshot_sha}
        attempt = campaign / "scheduler-attempts" / f"stage-{stage}-{uuid.uuid4().hex}"
        attempt.mkdir(parents=True, exist_ok=False)
        plan = {"schema": "pyrpmd.h3.quantum-umbrella-scheduler-plan.v1", "created_utc": now(),
                "stage": stage, "seeds": list(SEEDS), "prior_snapshot_sha256": snapshot_sha,
                "decision_sha256": decision_sha, "scheduler_source_sha256": file_hash(__file__),
                "source_guards": guards, "initial_limit": initial_limit,
                "worker_control": str(Path(control).resolve()) if control else None,
                "worker_control_semantics": "Scheduling only; decrease drains existing work; increase admits next registered seeds; ceiling4",
                "resumed_failure_receipt_sha256": resume_sha, "sampler_entrypoint": DRIVER + ":one_seed"}
        import numpy as np
        plan["runtime"] = {"python": sys.version, "numpy": np.__version__,
                           "process_start_method": "spawn", "worker_BLAS_OMP_threads": 1}
        atomic_new_json(attempt / "plan.json", plan)
        events = (attempt / "events.jsonl").open("x")

        def event(kind, **fields):
            events.write(json.dumps({"utc": now(), "event": kind, **fields}) + "\n"); events.flush()

        done_seeds = {row["seed"] for row in initial if any(x["production_sweeps"] == stage for x in row["stage_receipts"])}
        pending = deque(seed for seed in SEEDS if seed not in done_seeds)
        active, errors = {}, []
        prior_by_seed = {row["seed"]: row for row in prior["seeds"]}
        last_control = object()
        pool = None
        try:
            with stop_signals() as stop:
                pool = ProcessPoolExecutor(max_workers=4 if control else workers,
                    mp_context=multiprocessing.get_context("spawn"), initializer=_configure_worker)
                while active or (pending and not errors):
                    if stop["reason"] and not errors:
                        errors.append(stop["reason"]); event("stop_admissions", reason=stop["reason"])
                    if not errors:
                        try:
                            limit, control_sha = worker_limit(workers, control)
                            if (limit, control_sha) != last_control:
                                event("worker_limit", active_workers=limit, control_sha256=control_sha)
                                last_control = (limit, control_sha)
                            verify_guards(guards)
                            while pending and len(active) < limit:
                                seed = pending.popleft()
                                job = (campaign, Path(library).resolve(), Path(protocol_path).resolve(), seed, stage)
                                active[pool.submit(_execute_job, worker, job, guards)] = seed
                                event("submitted", seed=seed, active=len(active))
                        except Exception as error:
                            errors.append(f"{type(error).__name__}: {error}"); event("stop_admissions", reason=errors[-1])
                    if errors:
                        for future in list(active):
                            if future.cancel():
                                event("cancelled_before_start", seed=active.pop(future))
                    if not active:
                        break
                    finished, _ = wait(active, timeout=.2, return_when=FIRST_COMPLETED)
                    for future in sorted(finished, key=lambda item: SEEDS.index(active[item])):
                        seed = active.pop(future)
                        try:
                            result = future.result()
                            row = inspect_seed(campaign, prior_by_seed[seed], stage, protocol,
                                prior["protocol"]["sha256"], prior["source_sha256"][DRIVER])
                            receipt = json.loads((campaign / f"seed-{seed}/stages/{stage}/stage.json").read_text())
                            require(result == receipt and row["arrays"]["samples"]["shape"][0] == stage,
                                    "worker return/disk stage receipt differs")
                            done_seeds.add(seed); event("seed_complete", seed=seed)
                        except BaseException as error:
                            errors.append(f"seed {seed}: {type(error).__name__}: {error}")
                            event("seed_failed", seed=seed, reason=errors[-1])
        except BaseException as error:
            errors.append(f"coordinator: {type(error).__name__}: {error}")
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
            event("all_submitted_workers_drained")
        rows = []
        for seed in SEEDS:
            try:
                rows.append(inspect_seed(campaign, prior_by_seed[seed], stage, protocol,
                    prior["protocol"]["sha256"], prior["source_sha256"][DRIVER]))
            except Exception as error:
                rows.append({"seed": seed, "integrity_error": f"{type(error).__name__}: {error}"})
                errors.append(f"final integrity for seed {seed}: {error}")
        try:
            verify_guards(guards)
        except Exception as error:
            errors.append(str(error))
        if errors or done_seeds != set(SEEDS):
            failed = {"schema": FAILURE_SCHEMA, "created_utc": now(), "stage": stage,
                      "prior_snapshot_sha256": snapshot_sha, "decision_sha256": decision_sha,
                      "scheduler_source_sha256": plan["scheduler_source_sha256"], "errors": errors,
                      "completed_seeds": sorted(done_seeds), "all_submitted_workers_drained": True,
                      "checkpoints_after_drain": rows, "complete_cohort": False}
            atomic_new_json(attempt / "failure.json", failed)
            events.close()
            raise CampaignFailure(attempt / "failure.json")
        snapshot = {"schema": SNAPSHOT_SCHEMA, "created_utc": now(), "production_sweeps": stage,
                    "warmup_sweeps": protocol["warmup_sweeps"], "independent_seeds": list(SEEDS),
                    "campaign_directory": str(campaign), "protocol": prior["protocol"], "schedule": prior["schedule"],
                    "source_sha256": prior["source_sha256"], "model_sha256": prior["model_sha256"], "seeds": rows,
                    "initial_raw_archive": prior.get("archive", prior.get("initial_raw_archive")),
                    "prior_snapshot_sha256": snapshot_sha, "physical_converged": False}
        completion = {"schema": COMPLETION_SCHEMA, "created_utc": now(), "production": stage,
                      "requested_seeds": list(SEEDS), "completed_seeds": list(SEEDS),
                      "protocol_sha256": prior["protocol"]["sha256"], "decision_sha256": decision_sha,
                      "prior_snapshot_sha256": snapshot_sha, "scheduler_source_sha256": plan["scheduler_source_sha256"],
                      "attempt_directory": str(attempt), "all_submitted_workers_drained": True,
                      "prior_prefix_and_state_files_unchanged": True, "snapshot": snapshot,
                      "physical_converged": False, "analysis_recomputed": False}
        event("complete_cohort_verified", seeds=list(SEEDS))
        events.close()
        atomic_new_json(final_path, completion)
        return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--prior-snapshot", type=Path, required=True)
    parser.add_argument("--prior-snapshot-sha256", required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--decision-sha256", required=True)
    parser.add_argument("--stage", type=int, choices=PREFIXES[1:], required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--worker-control", type=Path)
    parser.add_argument("--resume-failure", type=Path)
    parser.add_argument("--resume-failure-sha256")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = run_campaign(args.campaign, args.library, args.protocol, args.prior_snapshot,
        args.prior_snapshot_sha256, args.decision, args.decision_sha256, args.stage,
        workers=args.workers, control=args.worker_control, resume_path=args.resume_failure,
        resume_sha=args.resume_failure_sha256, check_only=args.check_only)
    print(json.dumps({key: value for key, value in result.items() if key != "snapshot"}, indent=2))


if __name__ == "__main__":
    main()
