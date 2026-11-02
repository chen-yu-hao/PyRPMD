#!/usr/bin/env python3
"""Four fixed water queues with inherited leases and fail-closed dispatch.

This is only an orchestration layer around the frozen native_reference.py.
It never changes configurations, random states, trajectories, or seed order.
The migration registration must identify normally finished old native calls.
Both old queue locks are held for the complete invocation. Every native child
inherits those locks, the coordinator lock, and its own seed lease, so an
unexpected coordinator exit cannot release exclusion while a child runs.

Creating OUTPUT/STOP_AFTER_CURRENT stops dispatch after running children exit
normally. A detected error also leaves a durable FAILURE.json; subsequent
invocations refuse to run until that failure has been explicitly reviewed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.water.campaign import sha256
from benchmarks.water.native_campaign_schedule import committed_prefix
from benchmarks.water.native_reference import Config, source_identity

SCHEDULER_FILE = Path(__file__).resolve()
LEGACY_SCHEDULER = ROOT / "benchmarks/water/native_campaign_schedule.py"
NATIVE = "benchmarks/water/native_reference.py"
SCHEMA = "pyrpmd.water-four-queue-registration.v1"


def read_json(path):
    return json.loads(Path(path).read_text())


@contextmanager
def exclusive_lease(path):
    """Keep the flock open; pass its descriptor to the native child as well."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"campaign lease is already held: {path.name}") from error
        # Do not explicitly LOCK_UN here: an inherited descriptor may still
        # protect a native child after the parent closes its copy on an error.
        yield stream


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.run_id = uuid.uuid4().hex

    def __call__(self, event, **values):
        row = {"utc": datetime.now(timezone.utc).isoformat(), "run_id": self.run_id,
               "coordinator_pid": os.getpid(), "event": event, **values}
        with self.lock:
            with self.path.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(json.dumps(row, sort_keys=True), flush=True)


def load_registration(path):
    """Validate the complete physical/source identity before acquiring work."""
    path = Path(path).resolve()
    registration = read_json(path)
    if registration.get("schema") != SCHEMA:
        raise ValueError("unsupported four-queue registration schema")
    protocol_path = (ROOT / registration["protocol_path"]).resolve()
    protocol = read_json(protocol_path)
    legacy_path = protocol_path.parent / "native-queue-registration.json"
    legacy = read_json(legacy_path)
    revision_path = protocol_path.parent / "checkpoint-contract-v2.json"
    revision = read_json(revision_path)
    expected = dict(protocol["source_sha256"])
    if revision["from_source_sha256"] != expected[NATIVE]:
        raise ValueError("checkpoint revision does not identify the original registered source")
    expected[NATIVE] = revision["to_source_sha256"]
    if source_identity() != expected or legacy["native_source_sha256"] != expected:
        raise ValueError("native source differs from the frozen registered source")
    if (legacy["protocol_sha256"] != sha256(protocol_path)
            or legacy["checkpoint_revision_sha256"] != sha256(revision_path)
            or legacy["scheduler_source_sha256"] != sha256(LEGACY_SCHEDULER)
            or revision["patch_sha256"] != sha256(protocol_path.parent / "checkpoint-contract-v2.patch")):
        raise ValueError("original protocol/checkpoint/scheduler registration changed")
    identities = {"protocol_sha256": sha256(protocol_path),
                  "legacy_registration_sha256": sha256(legacy_path),
                  "checkpoint_revision_sha256": sha256(revision_path),
                  "legacy_scheduler_sha256": sha256(LEGACY_SCHEDULER),
                  "scheduler_source_sha256": sha256(SCHEDULER_FILE),
                  "native_source_sha256": expected}
    if any(registration.get(key) != value for key, value in identities.items()):
        raise ValueError("four-queue source or protocol registration mismatch")
    queues = registration["queues"]
    seeds = protocol["fixed_stage"]["seeds"]
    if (registration.get("max_concurrent_workers") != 4
            or len(queues) != 4 or any(not values for values in queues.values())
            or sorted(seed for values in queues.values() for seed in values) != sorted(seeds)
            or len(set(seeds)) != len(seeds)
            or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
            or registration.get("propagator_protocol_changes") is not False):
        raise ValueError("four queues must partition exactly the original fixed seeds without physical changes")
    receipt_path = (path.parent / registration["migration_receipt"]).resolve()
    receipt = read_json(receipt_path)
    if sha256(receipt_path) != registration["migration_receipt_sha256"]:
        raise ValueError("migration receipt checksum mismatch")
    retired = receipt["old_dispatchers"]
    retired_pids = [row[key] for row in retired for key in ("dispatcher_pid", "native_child_pid")]
    if (receipt.get("children_normally_exited") is not True
            or receipt.get("dispatchers_retired") is not True
            or len(retired) != 2 or {row["queue"] for row in retired} != {"odd", "even"}
            or len(set(retired_pids)) != 4
            or any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in retired_pids)
            or set(receipt["baseline_prefixes"]) != {str(seed) for seed in seeds}
            or any(row["native_exit_code"] != 0 for row in retired)):
        raise ValueError("migration requires normal child exits, retired dispatchers and every seed prefix")
    configs = {}
    for seed in seeds:
        config_path = protocol_path.parent / "configs" / f"classical-seed-{seed}.json"
        config = Config(**read_json(config_path)).validate()
        if (config.seed != seed or config.threads != 1 or config.platform != "CPU"
                or sha256(config_path) != legacy["config_sha256"][str(seed)]
                or not config.initial_checkpoint
                or sha256(ROOT / config.initial_checkpoint) != legacy["initial_checkpoint_sha256"]):
            raise ValueError("registered seed, single-thread configuration or initial state mismatch")
        configs[seed] = (config_path, config)
    return {"path": path, "sha256": sha256(path), "registration": registration,
            "protocol_path": protocol_path, "expected": expected, "configs": configs,
            "legacy": legacy, "receipt": receipt, "receipt_path": receipt_path}


def verify_runtime(identity):
    """Recheck immutable identities before each invocation, never during MD."""
    current = load_registration(identity["path"])
    if current["sha256"] != identity["sha256"]:
        raise RuntimeError("four-queue registration changed during the invocation")


def registered_prefix(output, identity, seed):
    directory = output / "runs" / f"classical-{seed}"
    state, chunks = committed_prefix(directory)
    baseline = identity["receipt"]["baseline_prefixes"][str(seed)]
    if chunks[:len(baseline["chunks"])] != baseline["chunks"]:
        raise RuntimeError("pre-migration committed trajectory prefix was changed")
    if state["committed_steps"] < baseline["committed"]["committed_steps"]:
        raise RuntimeError("committed state moved backwards from the migration receipt")
    path = directory / "manifest.json"
    if path.exists():
        manifest = read_json(path)
        if (manifest["config"] != asdict(identity["configs"][seed][1])
                or manifest["source_sha256"] != identity["expected"]
                or sha256(path) != state["manifest_sha256"]):
            raise ValueError("existing trajectory differs from the registered source/configuration")
    return state, chunks


def run_campaign(output, registration_path, *, wall_seconds=7200, invocation_seconds=1800, workers=4):
    if (not math.isfinite(wall_seconds) or not 0 < wall_seconds <= 7200
            or not math.isfinite(invocation_seconds) or not 0 < invocation_seconds <= 1800
            or workers not in (2, 4)):
        raise ValueError("require 2 or 4 workers, outer budget in (0,7200] and native call in (0,1800] seconds")
    identity = load_registration(registration_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    control = output / "four-worker-control"
    control.mkdir(exist_ok=True)
    failure_path = control / "FAILURE.json"
    stop_path = control / "STOP_AFTER_CURRENT"
    if failure_path.exists():
        raise RuntimeError("a prior campaign failure requires explicit review; no seed will be replaced")
    journal = Journal(control / "events.jsonl")
    stopped = threading.Event()
    launch_lock = threading.RLock()
    errors = []
    started = time.perf_counter()

    def latch(error, queue, seed):
        with launch_lock:
            stopped.set()
            if not any(previous is error for previous in errors):
                errors.append(error)
            if not failure_path.exists():
                with failure_path.open("x") as stream:
                    json.dump({"utc": datetime.now(timezone.utc).isoformat(),
                               "registration_sha256": identity["sha256"], "queue": queue,
                               "seed": seed, "error_type": type(error).__name__, "message": str(error)}, stream)
                    stream.write("\n"); stream.flush(); os.fsync(stream.fileno())

    def fail(error, queue, seed):
        latch(error, queue, seed)
        journal("campaign_stopped_on_error", queue=queue, seed=seed,
                error_type=type(error).__name__, message=str(error),
                running_children_allowed_to_finish_normally=True)

    with ExitStack() as stack:
        # Retain these locks in every child too; default close_fds would lose
        # crash protection. Old odd/even launchers therefore stay excluded.
        global_leases = [stack.enter_context(exclusive_lease(path)) for path in (
            control / "coordinator.lock", output / "scheduler-odd.lock", output / "scheduler-even.lock")]
        if failure_path.exists():
            raise RuntimeError("a prior campaign failure requires explicit review")
        seeds = [seed for values in identity["registration"]["queues"].values() for seed in values]
        seed_leases = {seed: stack.enter_context(exclusive_lease(control / f"seed-{seed}.lock"))
                       for seed in seeds}
        for seed in seeds:
            registered_prefix(output, identity, seed)
        journal("coordinator_started", registration_sha256=identity["sha256"],
                queues=identity["registration"]["queues"], outer_wall_seconds=wall_seconds,
                invocation_wall_seconds=invocation_seconds, inherited_legacy_and_seed_leases=True,
                active_worker_limit=workers, propagator_protocol_changes=False)

        def worker(queue, queue_seeds):
            seed = None
            try:
                for seed in queue_seeds:
                    while True:
                        # The lock makes the failure/stop check and process
                        # creation one operation with respect to other queues.
                        with ExitStack() as invocation:
                            with launch_lock:
                                try:
                                    remaining = wall_seconds - (time.perf_counter() - started)
                                    if stopped.is_set() or stop_path.exists() or remaining <= 0:
                                        journal("queue_dispatch_stopped", queue=queue, seed=seed,
                                                reason="failure" if stopped.is_set() else
                                                "operator_request" if stop_path.exists() else "outer_budget")
                                        return
                                    verify_runtime(identity)
                                    before, prefix = registered_prefix(output, identity, seed)
                                    if before["complete"]:
                                        journal("chain_complete", queue=queue, seed=seed, committed=before)
                                        break
                                    config_path, _ = identity["configs"][seed]
                                    directory = output / "runs" / f"classical-{seed}"
                                    command = [sys.executable, str(ROOT / NATIVE), "--output", str(directory),
                                               "--wall-seconds", str(min(invocation_seconds, remaining))]
                                    command += ["--resume"] if (directory / "manifest.json").exists() else ["--config", str(config_path)]
                                    logs = output / "logs"
                                    logs.mkdir(exist_ok=True)
                                    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                                       OPENMM_CPU_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
                                    inherited = tuple(lease.fileno() for lease in global_leases + [seed_leases[seed]])
                                    journal("invocation_prepared", queue=queue, seed=seed, committed=before,
                                            config_sha256=identity["legacy"]["config_sha256"][str(seed)],
                                            command=command, inherited_lock_count=len(inherited))
                                    log = invocation.enter_context((logs / f"classical-{seed}.log").open("a"))
                                    child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                                             stderr=subprocess.STDOUT, pass_fds=inherited)
                                    # Register draining immediately after successful
                                    # spawn, before even a journal write can fail.
                                    invocation.callback(child.wait)
                                    journal("invocation_started", queue=queue, seed=seed, native_child_pid=child.pid)
                                except BaseException as error:
                                    # Latch while still owning launch_lock: another
                                    # queue cannot launch after this known failure.
                                    latch(error, queue, seed)
                                    raise
                            code = child.wait()
                        if code != 0:
                            # Block new dispatch immediately, before hashing a
                            # possibly large final manifest after a failed call.
                            with launch_lock:
                                stopped.set()
                        after, updated = registered_prefix(output, identity, seed)
                        if updated[:len(prefix)] != prefix:
                            raise RuntimeError("previously committed trajectory prefix was changed")
                        journal("invocation_finished", queue=queue, seed=seed, native_child_pid=child.pid,
                                exit_code=code, committed=after)
                        if code != 0:
                            raise RuntimeError(f"native seed {seed} failed with exit {code}; no replacement")
                        if after["committed_steps"] <= before["committed_steps"]:
                            raise RuntimeError("successful native child made no progress")
                journal("queue_complete", queue=queue)
            except BaseException as error:
                fail(error, queue, seed)

        # Exiting the pool waits for every already-running native call. No
        # timeout, cancellation, terminate(), or kill() is applied to a child.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="water-queue") as pool:
            futures = [pool.submit(worker, queue, values)
                       for queue, values in identity["registration"]["queues"].items()]
            for future in futures:
                future.result()
        journal("coordinator_finished", failed=bool(errors), elapsed_wall_seconds=time.perf_counter()-started)
    if errors:
        raise RuntimeError("water campaign stopped after an error; existing children finished normally") from errors[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=7200)
    parser.add_argument("--invocation-seconds", type=float, default=1800)
    parser.add_argument("--workers", type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    run_campaign(args.output, args.registration, wall_seconds=args.wall_seconds,
                 invocation_seconds=args.invocation_seconds, workers=args.workers)


if __name__ == "__main__":
    main()
