"""Fixed eight-chain P256 parent generation; scheduling and identity only.

The scientific sampler is unchanged. Historical chains must reproduce every
old non-snapshot payload byte; four new chains start independently. This tool
cannot generate recrossing children or accept a four-chain interim result.
"""
from __future__ import annotations

import time
ENTRY_MONOTONIC = time.monotonic()

import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True

DESIGN = ROOT / "docs/validation/literature-convergence-2026-11-01/h3-quantum/p256-fixed-eight-chain-design/proposal.json"
DESIGN_SHA256 = "c668c15fd89ca0a535afc4082521e3f42ba9e84f5736b5b481bc1d7c376d791d"
SEEDS = (825601, 825603, 825607, 825609, 825613, 825617, 825621, 825623)
HISTORICAL_SEEDS = SEEDS[:4]
OBSERVATION_ARRAYS = frozenset(("fingerprint", "endpoint_q", "endpoint_completed_sweeps"))
REPORT_OBSERVATION_FIELDS = frozenset(("elapsed_this_call_seconds", "created_utc", "checkpoint_sha256",
                                      "endpoint_snapshot_count", "endpoint_snapshot_interval"))
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
AUTH_SCHEMA = "pyrpmd.h3.p256-parent-execution-authorization.v1"
REGISTRATION_SCHEMA = "pyrpmd.h3.p256-eight-parent-registration.v1"
FINALIZATION_RESERVE_SECONDS = 15.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def read_json(path):
    return json.loads(Path(path).read_bytes())


def settings_fingerprint(value):
    # The frozen scientific producer uses this exact non-compact serialization.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_new_json(path, value):
    """Publish a complete new receipt, with no overwrite of an existing one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def input_path(name):
    path = Path(name)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def child_path(base, name):
    parsed = PurePosixPath(name)
    require(not parsed.is_absolute() and ".." not in parsed.parts and "\\" not in name,
            "unsafe registered output path")
    path = (Path(base) / name).resolve()
    require(path.is_relative_to(Path(base).resolve()), "output path escapes the campaign")
    return path


def check_identity(row):
    actual = identity(input_path(row["path"]))
    require(actual["sha256"] == row["sha256"] and actual["bytes"] == row["bytes"],
            f"bound input changed: {row['path']}")
    return actual


def verify_guards(guards):
    for name, expected in guards.items():
        require(sha(name) == expected, f"source or immutable input changed: {name}")


def current_runtime():
    import numpy as np
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": platform.platform()}


def load_design(path=DESIGN):
    path = Path(path).resolve()
    require(sha(path) == DESIGN_SHA256, "the fixed eight-chain design changed")
    design = read_json(path)
    require(tuple(design["independent_P256_seeds"]) == SEEDS
            and tuple(design["historical_replay_seeds"]) == HISTORICAL_SEEDS
            and tuple(design["new_independent_seeds"]) == SEEDS[4:], "fixed cohort changed")
    parent = design["parent_generation"]
    require(tuple(row["seed"] for row in parent["cohorts"]) == SEEDS
            and parent["independent_uncertainty_units"] == 8
            and parent["snapshots_per_seed"] == 200 and parent["total_snapshots"] == 1600,
            "all eight complete parent chains are required")
    require(parent["snapshot_completed_sweeps"] == list(range(2100, 22001, 100)),
            "fixed parent observation schedule changed")
    for row in parent["cohorts"]:
        expected = row["expected_settings"]
        require(settings_fingerprint(expected) == row["expected_settings_fingerprint"],
                "design settings fingerprint differs")
        require(expected["beads"] == 256 and expected["warmup"] == 2000
                and expected["production"] == 20000 and expected["snapshot_interval"] == 100
                and expected["seed"] == row["seed"], "unregistered physical or statistical counts")
    return design


def inspect_payload(path, expected_sha=None):
    """Hash and decode the same immutable bytes, including the integrity envelope."""
    from pyrpmd.checkpoint import load_checkpoint
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(expected_sha is None or digest == expected_sha, "checkpoint/report checksum mismatch")
    with load_checkpoint(io.BytesIO(raw), expected_kind="h3.surface-partition", require_integrity=True) as saved:
        arrays = {name: saved[name] for name in saved.files}
    return arrays, digest, len(raw)


def exact_array_equal(a, b):
    return a.dtype.str == b.dtype.str and a.shape == b.shape and a.tobytes(order="C") == b.tobytes(order="C")


def verify_chain(directory, row, sampling_flags):
    """Verify a completed new/replayed chain without changing any producer file."""
    import numpy as np
    directory = Path(directory)
    paths = {name: directory / name for name in
             ("settings.json", "report.json", "reference.json", "reference-domain.json", "checkpoint.npz")}
    before = {name: identity(path) for name, path in paths.items()}
    settings, report = read_json(paths["settings.json"]), read_json(paths["report.json"])
    require(settings == row["expected_settings"], "complete settings differ from the fixed design")
    require(settings_fingerprint(settings) == row["expected_settings_fingerprint"], "settings fingerprint differs")
    require(report["complete"] and report["source_unchanged"] and report["surface"] == "ts"
            and report["phase"] == "production" and report["seed"] == row["seed"],
            "complete unchanged production TS chain required")
    arrays, digest, size = inspect_payload(paths["checkpoint.npz"], report["checkpoint_sha256"])
    require(str(arrays["fingerprint"].item()) == row["expected_settings_fingerprint"], "checkpoint settings identity differs")
    require(str(arrays["domain_sha256"].item()) == before["reference-domain.json"]["sha256"],
            "checkpoint domain normalization identity differs")
    warmup, production, beads = (settings[name] for name in ("warmup", "production", "beads"))
    count = production // settings["snapshot_interval"]
    sweeps = warmup + np.arange(1, count + 1) * settings["snapshot_interval"]
    require(int(arrays["completed"]) == warmup + production, "incomplete production counts")
    require(arrays["delta_energy"].shape == ((production + settings["sample_interval"] - 1) // settings["sample_interval"], len(settings["lambdas"])),
            "energy observation count differs")
    require(arrays["diagnostics"].shape == ((production + settings["diagnostic_interval"] - 1) // settings["diagnostic_interval"], len(settings["lambdas"]), 8),
            "diagnostic observation count differs")
    q = arrays["endpoint_q"]
    require(q.shape == (count, beads, 3, 3) and np.isfinite(q).all()
            and np.array_equal(arrays["endpoint_completed_sweeps"], sweeps), "parent observation count/order differs")
    require(report["endpoint_snapshot_count"] == count
            and report["endpoint_snapshot_interval"] == settings["snapshot_interval"], "report parent count differs")
    center = q.mean(axis=1)
    distances = {name: np.linalg.norm(center[:, a] - center[:, b], axis=-1)
                 for name, a, b in (("01", 0, 1), ("02", 0, 2), ("12", 1, 2))}
    surface_error = float(np.max(abs(distances["01"] - distances["12"])))
    minimum = min(float(np.linalg.norm(q[:, :, a] - q[:, :, b], axis=-1).min())
                  for a, b in ((0, 1), (0, 2), (1, 2)))
    com_error = float(np.max(abs(q.mean(axis=2))))
    require(surface_error <= 1e-10 and np.all(distances["02"] >= distances["01"] - 1e-10),
            "parent is not on the same active centroid surface")
    require(np.all(distances["12"] <= settings["maximum_centroid_bond_bohr"] + 1e-12),
            "parent exceeds the registered centroid domain")
    require(minimum >= settings["minimum_bead_pair_bohr"] - 1e-12 and com_error <= 1e-10,
            "parent bead domain or zero-COM convention differs")
    require(np.isfinite(arrays["delta_energy"]).all() and np.isfinite(arrays["diagnostics"]).all(),
            "nonfinite saved equilibrium observations")
    replay = None
    if row["cohort_role"] == "historical_deterministic_replay":
        original = row["original_files"]
        for name, entry in original.items():
            check_identity(entry)
        old_settings = read_json(input_path(original["settings.json"]["path"]))
        require(settings == dict(old_settings, snapshot_interval=settings["snapshot_interval"])
                and old_settings["snapshot_interval"] == 0, "replay changed more than the observation interval")
        old_report = read_json(input_path(original["report.json"]["path"]))
        old, old_digest, _ = inspect_payload(input_path(original["checkpoint.npz"]["path"]), old_report["checkpoint_sha256"])
        require(set(old) == set(arrays), "replay payload key set changed")
        required = sorted(set(old) - OBSERVATION_ARRAYS)
        require(required == sorted(row["exact_original_payloads"]), "declared replay comparison omitted a payload")
        for name in required:
            require(exact_array_equal(old[name], arrays[name]), f"historical replay is not bitwise identical: {name}")
        for name in ("reference.json", "reference-domain.json"):
            require(paths[name].read_bytes() == input_path(original[name]["path"]).read_bytes(),
                    f"replay changed reference bytes: {name}")
        require(set(report) == set(old_report), "replay report schema changed")
        for name in set(report) - REPORT_OBSERVATION_FIELDS:
            require(report[name] == old_report[name], f"replay changed scientific report field: {name}")
        replay = {"original_checkpoint_sha256": old_digest, "bitwise_verified_arrays": required,
                  "bitwise_verified_array_count": len(required), "original_files": original,
                  "reference_bytes_equal": True, "scientific_report_equal": True}
    else:
        require(row["cohort_role"] == "new_independent_chain" and row["original_files"] is None,
                "unknown or falsely historical cohort role")
    analysis = report["analysis"]
    ess = [float(x) for x in analysis["diagnostic_energy_effective_samples"] if x is not None]
    minimum_ess = min(ess) if ess else float(len(arrays["delta_energy"]))
    half_difference = float(analysis["second_half_log_ratio"] - analysis["first_half_log_ratio"])
    overlap = float(analysis["minimum_BAR_overlap"])
    flags = (np.isfinite([minimum_ess, half_difference, overlap]).all()
             and minimum_ess >= sampling_flags["minimum_energy_ESS"]
             and overlap >= sampling_flags["minimum_BAR_overlap"]
             and abs(half_difference) <= sampling_flags["maximum_absolute_half_log_difference"])
    for name, path in paths.items():
        require(identity(path) == before[name], "producer file changed while its complete state was verified")
    return {"seed": row["seed"], "cohort_role": row["cohort_role"], "complete": True,
            "identity_and_geometry_verified": True, "sampling_flags_passed": bool(flags),
            "parents": count, "beads": beads, "completed_sweeps": warmup + production,
            "files": before, "checkpoint_sha256": digest, "checkpoint_bytes": size,
            "settings_fingerprint": row["expected_settings_fingerprint"], "historical_replay": replay,
            "minimum_bead_pair_bohr": minimum, "maximum_surface_error_bohr": surface_error,
            "maximum_COM_coordinate_bohr": com_error,
            "minimum_BAR_overlap": overlap, "minimum_energy_ESS": minimum_ess,
            "half_log_partition_difference": half_difference,
            "BAR_log_TS_quantum_classical_ratio": analysis["log_quantum_classical_ratio"],
            "uncertainty_unit": "one independent equilibrium chain", "recrossing_children_generated": 0}


def preflight(design_path=DESIGN, output=None):
    design = load_design(design_path)
    parent = design["parent_generation"]
    expected_output = Path(parent["output_root_proposed"]).resolve()
    output = expected_output if output is None else Path(output).resolve()
    require(output == expected_output, "output must be the separate registered parent root")
    require(current_runtime() == parent["scientific_source_identity"]["runtime"], "runtime differs from historical replay")
    guards = {str(Path(__file__).resolve()): sha(__file__), str(Path(design_path).resolve()): DESIGN_SHA256}
    for name, digest in parent["scientific_source_identity"]["source_sha256"].items():
        guards[str((ROOT / name).resolve())] = digest
    for key in ("external_library", "external_library_build_receipt"):
        actual = check_identity(parent[key]); guards[actual["path"]] = actual["sha256"]
    for key in ("frozen_four_chain_feasibility_inventory", "frozen_original_control_protocol", "frozen_original_four_chain_assessment"):
        actual = check_identity(design[key]); guards[actual["path"]] = actual["sha256"]
    for row in parent["cohorts"]:
        child_path(output, row["relative_directory"])
        if row["cohort_role"] == "historical_deterministic_replay":
            for entry in row["original_files"].values():
                actual = check_identity(entry); guards[actual["path"]] = actual["sha256"]
            old_report = read_json(input_path(row["original_files"]["report.json"]["path"]))
            arrays, _, _ = inspect_payload(input_path(row["original_files"]["checkpoint.npz"]["path"]), old_report["checkpoint_sha256"])
            require(int(arrays["completed"]) == 22000 and arrays["endpoint_q"].shape == (0, 256, 3, 3),
                    "historical control is not the registered complete no-snapshot state")
    verify_guards(guards)
    return design, output, guards


def load_authorization(path, guards):
    path = Path(path).resolve()
    authorization = read_json(path)
    require(authorization.get("schema") == AUTH_SCHEMA and authorization.get("parent_adoption") is True,
            "reviewed parent execution authorization required")
    require(authorization.get("design_sha256") == DESIGN_SHA256
            and authorization.get("driver_sha256") == sha(__file__), "authorization binds different design/source")
    workers = authorization.get("workers")
    cpus = authorization.get("cpu_affinity")
    require(type(workers) is int and 1 <= workers <= 2 and isinstance(cpus, list)
            and len(cpus) == workers and len(set(cpus)) == workers
            and all(type(cpu) is int and cpu in os.sched_getaffinity(0) for cpu in cpus),
            "one or two distinct available worker CPUs required")
    require(authorization.get("whole_campaign_wall_seconds") == 7200, "whole campaign budget changed")
    for key in ("synthetic_gate", "independent_review", "resource_decision"):
        actual = check_identity(authorization[key]); guards[actual["path"]] = actual["sha256"]
    resource_decision = read_json(input_path(authorization["resource_decision"]["path"]))
    require(resource_decision.get("safe_to_launch_parent_workers") is True
            and type(resource_decision.get("parent_workers")) is int
            and resource_decision["parent_workers"] == workers
            and type(resource_decision.get("declared_project_active_workers")) is int
            and workers <= resource_decision["declared_project_active_workers"] <= 8,
            "resource decision does not confirm this bounded allocation")
    require(resource_decision.get("prerequisite_queues_drained") is True
            and isinstance(resource_decision.get("evidence"), list) and resource_decision["evidence"],
            "actual queue-drain evidence is required, not a forecast")
    for entry in resource_decision["evidence"]:
        actual = check_identity(entry); guards[actual["path"]] = actual["sha256"]
    guards[str(path)] = sha(path)
    verify_guards(guards)
    return authorization


@contextmanager
def campaign_lock(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / ("." + output.name + "-p256-parent-lock")
    lock.mkdir()  # A stale/hard-crash lock is never broken automatically.
    lease = {"release": True, "path": str(lock)}
    try:
        atomic_new_json(lock / "owner.json", {"pid": os.getpid(), "created_utc": now(), "output": str(output)})
        yield lease
    finally:
        if lease["release"]:
            (lock / "owner.json").unlink(missing_ok=True)
            lock.rmdir()


@contextmanager
def deferred_stop_signals():
    state = {"reason": None}
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.getsignal(number)
            signal.signal(number, lambda signum, frame: state.update(reason=f"signal {signum}"))
    try:
        yield state
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def drain_processes(active):
    """Only return after all owned process groups have been stopped and reaped."""
    for process, _ in active.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # The process may have exited between poll and kill.
    deadline = time.monotonic() + 5.
    for process, _ in active.values():
        try:
            process.wait(timeout=max(.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5.)
    require(all(process.poll() is not None for process, _ in active.values()), "worker drain not confirmed")


def dispatch_jobs(seeds, command_for, verify_result, *, cpus, deadline, attempt, lease,
                  guard=lambda: None, event_hook=None):
    """Scheduling-only engine; bounded tests supply inert subprocess commands."""
    pending = deque(seeds)
    active, results, errors, logs, launched = {}, {}, [], {}, []
    attempt = Path(attempt)
    journal = (attempt / "events.jsonl").open("x")

    def event(name, **fields):
        row = {"utc": now(), "event": name, **fields}
        if event_hook:
            event_hook(row)
        journal.write(json.dumps(row, sort_keys=True) + "\n"); journal.flush()

    with deferred_stop_signals() as stop:
        try:
            while active or pending:
                if stop["reason"]:
                    errors.append(stop["reason"]); break
                if time.monotonic() >= deadline:
                    errors.append("whole campaign wall deadline"); break
                free = [cpu for cpu in cpus if cpu not in {slot for _, slot in active.values()}]
                if pending and free:
                    # Full original-state guards include ~10MB of frozen NPZ.
                    # Check them at admissions, not in a 20Hz polling loop.
                    # The frozen producer also checks its own sources at every
                    # checkpoint; final identity is checked after all workers.
                    guard()
                while pending and free:
                    reason = stop["reason"] or ("whole campaign wall deadline" if time.monotonic() >= deadline else None)
                    if reason:
                        errors.append(reason)
                        break
                    seed, cpu = pending.popleft(), free.pop(0)
                    log = (attempt / f"seed-{seed}.log").open("x")
                    logs[seed] = log
                    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **{name: "1" for name in THREAD_VARIABLES})
                    command = command_for(seed, cpu)
                    reason = stop["reason"] or ("whole campaign wall deadline" if time.monotonic() >= deadline else None)
                    if reason:
                        pending.appendleft(seed)
                        errors.append(reason)
                        break
                    lease["release"] = False
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               cwd=ROOT, env=environment, start_new_session=True)
                    active[seed] = (process, cpu); launched.append(seed)
                    event("worker_started", seed=seed, pid=process.pid, cpu=cpu)
                if errors:
                    break
                finished = [seed for seed in seeds if seed in active and active[seed][0].poll() is not None]
                for seed in finished:
                    process, cpu = active.pop(seed)
                    code = process.wait(); logs[seed].close()
                    event("worker_reaped", seed=seed, pid=process.pid, returncode=code)
                    require(code == 0, f"seed {seed} worker exited {code}")
                    results[seed] = verify_result(seed)
                    event("seed_verified", seed=seed)
                if active:
                    time.sleep(min(.05, max(0., deadline - time.monotonic())))
        except BaseException as error:
            errors.append(f"{type(error).__name__}: {error}")
        finally:
            # Keep deferred SIGINT/SIGTERM handlers installed until every owned
            # child has drained, including after a journal/verification error.
            try:
                drain_processes(active)
            except BaseException as error:
                journal.close()
                for log in logs.values():
                    log.close()
                raise RuntimeError("worker draining unconfirmed; exclusive lease retained") from error
            lease["release"] = True
            for log in logs.values():
                log.close()
            if stop["reason"] and stop["reason"] not in errors:
                errors.append(stop["reason"])
            try:
                event("all_owned_workers_drained")
            except BaseException as error:
                errors.append(f"drained journal: {type(error).__name__}: {error}")
            journal.close()
    return {"launched_seeds": launched, "results": results, "errors": errors,
            "all_workers_drained": True, "unstarted_seeds": [seed for seed in seeds if seed not in launched]}


def produce_one(directory, row, flags, library, *, resume=False, sampler=None, before_sampler=None):
    """A completed producer is inspected and reused without rewriting its bytes."""
    directory = Path(directory)
    if (directory / "report.json").is_file() and read_json(directory / "report.json").get("complete"):
        require(resume, "completed parent exists; explicit reviewed resume required")
        return verify_chain(directory, row, flags), False
    kwargs = dict(row["run_keyword_arguments"])
    require(kwargs.get("resume") is False and kwargs.get("stop_after") is None, "unregistered sampler dispatch")
    kwargs["resume"] = bool(resume and (directory / "settings.json").exists())
    if sampler is None:
        from benchmarks.h3.surface_partition import run as sampler
    if before_sampler is not None:
        before_sampler()
    sampler(directory, library, **kwargs)
    return verify_chain(directory, row, flags), True


def worker(registration_path, attempt_path, seed, cpu):
    started = time.monotonic()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    registration_path, attempt_path = Path(registration_path).resolve(), Path(attempt_path).resolve()
    registration = read_json(registration_path)
    require(registration.get("schema") == REGISTRATION_SCHEMA, "invalid parent registration")
    design, output, guards = preflight(registration["design_path"], registration["output"])
    require(registration["design_sha256"] == DESIGN_SHA256
            and registration["driver_sha256"] == sha(__file__), "registered driver/design changed")
    attempt = read_json(attempt_path / "plan.json")
    require(attempt["registration_sha256"] == sha(registration_path) and seed in SEEDS
            and cpu in attempt["cpu_affinity"], "worker differs from registered attempt")
    # Only the coordinator named in the attempt may launch this worker once.
    require(attempt["coordinator_pid"] == os.getppid(), "worker requires its owning coordinator")
    atomic_new_json(attempt_path / f"seed-{seed}-worker-claim.json",
                    {"seed": seed, "pid": os.getpid(), "parent_pid": os.getppid(), "created_utc": now()})
    guards.update(attempt["source_guards"]); verify_guards(guards)
    os.sched_setaffinity(0, {cpu})
    require(all(os.environ.get(name) == "1" for name in THREAD_VARIABLES), "worker threads must be one")
    row = next(value for value in design["parent_generation"]["cohorts"] if value["seed"] == seed)
    result, failure, invoked = None, None, False
    def mark_sampler():
        nonlocal invoked
        invoked = True
    try:
        result, _ = produce_one(child_path(output, row["relative_directory"]), row,
            design["parent_generation"]["sampling_flags"], design["parent_generation"]["external_library"]["path"],
            resume=attempt["resume"], before_sampler=mark_sampler)
        verify_guards(guards)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        atomic_new_json(attempt_path / f"seed-{seed}-execution.json", {
            "seed": seed, "pid": os.getpid(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
            "sampler_invoked": invoked, "sampler_entrypoint": "benchmarks.h3.surface_partition.run",
            "resume": attempt["resume"], "whole_worker_call_wall_seconds": time.monotonic() - started,
            "whole_process_user_CPU_seconds": usage.ru_utime, "whole_process_system_CPU_seconds": usage.ru_stime,
            "worker_call_user_CPU_seconds": usage.ru_utime - usage_before.ru_utime,
            "worker_call_system_CPU_seconds": usage.ru_stime - usage_before.ru_stime,
            "peak_RSS_KiB": usage.ru_maxrss, "result": result, "failure": failure,
            "created_utc": now(), "recrossing_children_generated": 0})
    return 0


def run_campaign(authorization_path, *, design_path=DESIGN, output=None, resume=False):
    started = ENTRY_MONOTONIC
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    design, output, guards = preflight(design_path, output)
    authorization = load_authorization(authorization_path, guards)
    deadline = started + authorization["whole_campaign_wall_seconds"]
    with campaign_lock(output) as lease:
        registration_path = output / "registration.json"
        expected = {"schema": REGISTRATION_SCHEMA, "design_path": str(Path(design_path).resolve()),
                    "design_sha256": DESIGN_SHA256, "driver_sha256": sha(__file__),
                    "output": str(output), "fixed_seeds": list(SEEDS), "runtime": current_runtime()}
        if resume:
            require(registration_path.is_file() and read_json(registration_path) == expected,
                    "explicit resume requires unchanged registered source/settings/runtime")
        else:
            require(not output.exists() or not any(output.iterdir()), "new parent root must be empty")
            atomic_new_json(registration_path, expected)
        require(not (output / "parents-complete.json").exists(), "all-eight manifest already exists; never overwrite it")
        attempt = output / "attempts" / uuid.uuid4().hex
        attempt.mkdir(parents=True, exist_ok=False)
        plan = {"schema": "pyrpmd.h3.p256-parent-attempt.v1", "created_utc": now(),
                "coordinator_pid": os.getpid(), "registration_sha256": sha(registration_path),
                "authorization": identity(authorization_path), "source_guards": guards,
                "cpu_affinity": authorization["cpu_affinity"], "workers": authorization["workers"],
                "whole_campaign_wall_seconds": authorization["whole_campaign_wall_seconds"], "resume": bool(resume),
                "worker_stop_before_whole_campaign_deadline_seconds": FINALIZATION_RESERVE_SECONDS,
                "seed_order": list(SEEDS), "script_argv": sys.argv}
        atomic_new_json(attempt / "plan.json", plan)
        rows = {row["seed"]: row for row in design["parent_generation"]["cohorts"]}

        def command(seed, cpu):
            return [sys.executable, str(Path(__file__).resolve()), "_worker", "--registration", str(registration_path),
                    "--attempt", str(attempt), "--seed", str(seed), "--cpu", str(cpu)]

        def verify(seed):
            return verify_chain(child_path(output, rows[seed]["relative_directory"]), rows[seed],
                                design["parent_generation"]["sampling_flags"])

        outcome = dispatch_jobs(SEEDS, command, verify, cpus=authorization["cpu_affinity"],
                                deadline=deadline - FINALIZATION_RESERVE_SECONDS, attempt=attempt, lease=lease,
                                guard=lambda: verify_guards(guards))
        errors = list(outcome["errors"])
        final_rows = []
        if not errors:
            try:
                verify_guards(guards)
                require(set(outcome["results"]) == set(SEEDS), "full fixed cohort has not completed")
                final_rows = [verify(seed) for seed in SEEDS]
                require(time.monotonic() <= deadline, "whole campaign deadline includes final verification")
            except BaseException as error:
                errors.append(f"final audit: {type(error).__name__}: {error}")
        eligible = (not errors and len(final_rows) == 8
                    and all(row["sampling_flags_passed"] for row in final_rows))
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        result = {"schema": "pyrpmd.h3.p256-eight-parent-completion.v1", "created_utc": now(),
                  "all_eight_counts_complete_and_verified": not errors and len(final_rows) == 8,
                  "eligible_for_separately_authorized_children": bool(eligible),
                  "seed_order": list(SEEDS), "design_sha256": DESIGN_SHA256,
                  "registration_sha256": sha(registration_path), "attempt_plan": identity(attempt / "plan.json"),
                  "authorization": identity(authorization_path), "parents": final_rows,
                  "partial_verified_results": list(outcome["results"].values()) if errors else [],
                  "errors": errors, "all_owned_workers_drained": outcome["all_workers_drained"],
                  "launched_seeds": outcome["launched_seeds"], "unstarted_seeds": outcome["unstarted_seeds"],
                  "wall_seconds_since_module_entry": time.monotonic() - started,
                  "child_user_CPU_seconds": usage.ru_utime - before.ru_utime,
                  "child_system_CPU_seconds": usage.ru_stime - before.ru_stime,
                  "coordinator_user_CPU_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                  "coordinator_system_CPU_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                  "sampling_flags": design["parent_generation"]["sampling_flags"],
                  "recrossing_children_generated": 0, "dynamic_bead_convergence_demonstrated": False,
                  "independent_parent_chains": 8, "four_chain_interim_selection": False}
        atomic_new_json(attempt / "result.json", result)
        if eligible:
            atomic_new_json(output / "parents-complete.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    check = sub.add_parser("preflight", help="read-only guards; zero workers/PES calls")
    check.add_argument("--design", type=Path, default=DESIGN)
    check.add_argument("--output", type=Path)
    execute = sub.add_parser("run", help="reviewed all-eight parent campaign only")
    execute.add_argument("--authorization", type=Path, required=True)
    execute.add_argument("--design", type=Path, default=DESIGN)
    execute.add_argument("--output", type=Path)
    execute.add_argument("--resume", action="store_true")
    child = sub.add_parser("_worker", help=argparse.SUPPRESS)
    child.add_argument("--registration", type=Path, required=True)
    child.add_argument("--attempt", type=Path, required=True)
    child.add_argument("--seed", type=int, required=True)
    child.add_argument("--cpu", type=int, required=True)
    args = parser.parse_args()
    if args.action == "preflight":
        _, output, guards = preflight(args.design, args.output)
        print(json.dumps({"preflight_passed": True, "output": str(output), "seeds": list(SEEDS),
                          "source_guard_count": len(guards), "workers_launched": 0, "PES_calls": 0}, indent=2))
        return 0
    if args.action == "_worker":
        return worker(args.registration, args.attempt, args.seed, args.cpu)
    result = run_campaign(args.authorization, design_path=args.design, output=args.output, resume=args.resume)
    print(json.dumps({key: result[key] for key in ("all_eight_counts_complete_and_verified",
                      "eligible_for_separately_authorized_children", "errors", "launched_seeds")}, indent=2))
    return 0 if result["eligible_for_separately_authorized_children"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
