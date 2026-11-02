"""Wait for root-observed P256 completion, verify every raw hash, analyze once.

Only completion metadata is polled. The fixed analyzer is not imported, invoked
or replaced until the whole registered cohort and owned-worker drain are proved.
The execution directory is a permanent one-attempt claim, including on failure.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "docs/validation/literature-convergence-2026-11-01/h3-quantum/p256-completion-tool/policy.json"
POLICY_SHA = "ff5be96fe6412a4d98f03019d5c7107bf7bbd7a7dfcb2426455412562bb30858"


class UnstableRead(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def stable_bytes(path):
    path = Path(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read()
        after = os.fstat(stream.fileno())
    current = path.stat()
    fields = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if fields(before) != fields(after) or fields(after) != fields(current):
        raise UnstableRead(f"file changed while reading: {path}")
    return raw


def identity(path):
    path = Path(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
    current = path.stat()
    fields = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if fields(before) != fields(after) or fields(after) != fields(current):
        raise UnstableRead(f"file changed while hashing: {path}")
    return {"path": str(path.resolve()), "bytes": current.st_size, "sha256": digest}


def read(path):
    return json.loads(stable_bytes(path))


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def check_identity(expected):
    actual = identity(expected["path"])
    require(actual == expected, f"bound file identity differs: {expected['path']}")
    return actual


def exact_integer(value, expected):
    return type(value) is int and value == expected


def checkpoint_budget(deadline, stop):
    if stop["reason"]:
        raise InterruptedError(stop["reason"])
    if time.monotonic() >= deadline:
        raise TimeoutError("registered analysis/wait budget exhausted")


@contextmanager
def deferred_signals():
    stop, old = {"reason": None}, {}
    for number in (signal.SIGINT, signal.SIGTERM):
        old[number] = signal.getsignal(number)
        signal.signal(number, lambda signum, frame: stop.update(reason=f"signal {signum}"))
    try:
        yield stop
    finally:
        for number, handler in old.items():
            signal.signal(number, handler)


def load_authorization(path):
    require(identity(POLICY)["sha256"] == POLICY_SHA, "fixed completion policy changed")
    policy = read(POLICY)
    adoption = read(path)
    require(adoption.get("schema") == "pyrpmd.h3.p256-complete-analysis-adoption.v1"
            and adoption.get("adopted") is True and adoption.get("policy_sha256") == POLICY_SHA
            and adoption.get("source_sha256") == identity(__file__)["sha256"]
            and adoption.get("analysis_resources") == policy["analysis_resources"],
            "exact reviewed root adoption required")
    check_identity(adoption["independent_review"])
    review = read(adoption["independent_review"]["path"])
    require(review.get("approved_for_complete_analysis_once") is True
            and review.get("policy_sha256") == POLICY_SHA
            and review.get("source_sha256") == identity(__file__)["sha256"],
            "independent wrapper review differs")
    return policy, adoption


def claim(policy, adoption_path, adoption):
    paths = policy["paths"]
    require(not Path(paths["analysis_output"]).exists(), "analysis output already exists; never replace it")
    directory = Path(paths["execution"])
    directory.mkdir(parents=True, exist_ok=False)
    registration = {"schema": "pyrpmd.h3.p256-complete-analysis-registration.v1", "created_utc": now(),
        "policy": identity(POLICY), "source": identity(__file__), "adoption": identity(adoption_path),
        "independent_review": adoption["independent_review"], "PID": os.getpid(),
        "analysis_output": paths["analysis_output"], "analyzer_invocations_maximum": 1,
        "automatic_retry_resume_extension": False}
    write_new(directory / "RUN_ONCE.json", registration)
    return directory, registration


def pending_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    require(not path.is_symlink(), "completion receipt must not be a symlink")
    try:
        raw = stable_bytes(path)
        return json.loads(raw), {"path": str(path.resolve()), "bytes": len(raw),
                                 "sha256": hashlib.sha256(raw).hexdigest()}
    except (json.JSONDecodeError, UnicodeDecodeError, UnstableRead):
        # The old root launcher publishes small JSON receipts via open('x').
        # A not-yet-complete write is pending, never an admission signal.
        return None


def readiness(policy):
    """Metadata only: never open a branch report, raw chunk or rate analysis."""
    paths, contract = policy["paths"], policy["root_exit_contract"]
    completion_snapshot = pending_json(paths["root_completion"])
    completion = None if completion_snapshot is None else completion_snapshot[0]
    if completion is not None:
        require(exact_integer(completion.get("exit_code"), 0)
                and completion.get("sources_unchanged") is True,
                "root launch ended unsuccessfully or changed its source")
    observed_snapshot = pending_json(paths["root_observed_exit"])
    observed = None if observed_snapshot is None else observed_snapshot[0]
    if observed is not None:
        require(observed.get("schema") == contract["schema"]
                and exact_integer(observed.get("exec_session_id"), contract["exec_session_id"])
                and exact_integer(observed.get("exit_code"), 0)
                and observed.get("all_workers_drained") is True
                and exact_integer(observed.get("complete_registered_child_seeds"), 8)
                and exact_integer(observed.get("tool_result", {}).get("exit_code"), 0)
                and observed.get("driver_sha256") == contract["driver_sha256"]
                and isinstance(observed.get("observed_utc"), str) and bool(observed["observed_utc"]),
                "actual root-observed normal exit is not established")
    manifest_snapshot = pending_json(paths["children_complete"])
    manifest = None if manifest_snapshot is None else manifest_snapshot[0]
    if completion is None or observed is None or manifest is None or Path(paths["child_lock"]).exists():
        return None
    require(Path(paths["root_tool_observation"]).is_file(), "root tool observation is missing")
    c = policy["cohort"]
    require(manifest.get("schema") == c["complete_schema"]
            and manifest.get("complete_registered_cohort") is True
            and manifest.get("eligible_for_separate_full_cohort_analysis") is True
            and manifest.get("all_owned_workers_drained") is True
            and manifest.get("rate_analysis_performed") is False
            and manifest.get("seed_order") == manifest.get("launched_seeds") == c["seeds"]
            and manifest.get("branch_order") == c["branch_order"]
            and manifest.get("errors") == [] and manifest.get("unstarted_seeds") == []
            and manifest.get("partial_verified_results") == []
            and manifest.get("automatic_resume") is False
            and manifest.get("scientific_convergence_claimed") is False
            and exact_integer(manifest.get("expected_branch_count"), 32)
            and exact_integer(manifest.get("expected_chunk_count"), 800)
            and exact_integer(manifest.get("expected_native_steps"), 76_800_000)
            and exact_integer(manifest.get("native_steps"), 76_800_000),
            "complete drained fixed 8/32/800 cohort required")
    for key, value in policy["child_launch_value"].items():
        require(completion.get(key) == value, "root completion belongs to another launch")
    snapshots = {"children_complete": manifest_snapshot[1], "root_completion": completion_snapshot[1],
                 "root_observed_exit": observed_snapshot[1],
                 "child_registration": identity(paths["child_registration"]),
                 "root_tool_observation": identity(paths["root_tool_observation"])}
    links = {"children_complete_sha256": "children_complete", "registration_sha256": "child_registration",
             "root_completion_sha256": "root_completion", "actual_tool_observation_sha256": "root_tool_observation"}
    for key, name in links.items():
        require(observed.get(key) == snapshots[name]["sha256"], f"observed-exit identity differs: {key}")
    require(completion.get("log_sha256") == identity(paths["root_log"])["sha256"], "completed root log changed")
    for key, name in (("protocol", "child_protocol"), ("registration", "registration"),
                      ("attempt_plan", "attempt_plan"), ("authorization", "child_authorization"),
                      ("parent_manifest", "parent_manifest")):
        require(manifest.get(key) == policy[name], f"global completion route/identity differs: {key}")
    require(manifest.get("P128_baseline_sha256") == policy["baseline"]["sha256"]
            and manifest.get("source_files") == policy["source_files"]
            and manifest.get("runtime") == policy["scientific_runtime"], "registered baseline/source/runtime differs")
    return {"manifest": manifest, "completion": completion, "observed": observed, "snapshots": snapshots}


def wait_ready(policy, stop, *, clock=time.monotonic, sleep=time.sleep):
    deadline = clock() + policy["waiting"]["wall_seconds"]
    while True:
        if stop["reason"]:
            raise InterruptedError(stop["reason"])
        if clock() >= deadline:
            raise TimeoutError("complete root-observed cohort did not arrive within waiting budget")
        result = readiness(policy)
        if result is not None:
            return result
        sleep(min(policy["waiting"]["poll_seconds"], max(0.0, deadline-clock())))


def dependency(row, name):
    check_identity(row)
    spec = importlib.util.spec_from_file_location(name, row["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == Path(row["path"]).resolve(), "metadata verifier import route differs")
    check_identity(row)
    return module


def verify_final_inputs(policy, metadata, deadline, stop):
    """After the root gate, hash every final report/settings/chunk; infer no rate."""
    checkpoint_budget(deadline, stop)
    fresh = readiness(policy)
    require(fresh is not None and fresh["snapshots"] == metadata["snapshots"],
            "completion eligibility or exact receipt identity changed")
    manifest = metadata["manifest"]
    rows = []
    def retain(row, *, raw=False):
        checkpoint_budget(deadline, stop)
        if raw:
            path = Path(row["path"])
            require(not path.is_symlink() and path.resolve().is_relative_to(Path(policy["paths"]["data"]).resolve()),
                    "raw member route leaves registered cohort")
        actual = check_identity({key: row[key] for key in ("path", "bytes", "sha256")})
        rows.append(actual)
        return actual
    for row in policy["static_guards"]:
        retain(row)
    validators = policy["pure_metadata_verifiers"]
    coordinator = dependency(validators["cohort_module"], "_p256_completed_cohort_metadata")
    journal_helper = dependency(validators["journal_helper"], "_p256_completed_worker_journal")
    require(coordinator.runtime_identity() == policy["scientific_runtime"], "analysis process runtime differs from registered child runtime")
    require(identity(policy["command"][0]) == policy["scientific_runtime"]["python_executable"],
            "analyzer executable route or bytes differ")
    counts = coordinator.verify_complete_cohort(manifest["seeds"], policy["paths"]["data"])
    require(counts == {"branches": 32, "chunks": 800, "native_steps": 76_800_000}, "cohort verifier count differs")
    for seed in manifest["seeds"]:
        for branch in seed["branches"]:
            retain(branch["settings"], raw=True)
            retain(branch["report"], raw=True)
            report = read(branch["report"]["path"])
            require(report.get("complete") is True and report.get("source_unchanged") is True
                    and report.get("settings_sha256") == branch["settings"]["sha256"], "final branch report is incomplete")
            require([(item["file"], item["sha256"]) for item in report["chunks"]]
                    == [(Path(item["path"]).name, item["sha256"]) for item in branch["chunks"]],
                    "branch/global raw inventory differs")
            require({p.name for p in Path(branch["report"]["path"]).parent.glob("chunk-*.npz")}
                    == {Path(row["path"]).name for row in branch["chunks"]}, "extra or missing chunk file")
            for chunk in branch["chunks"]:
                retain(chunk, raw=True)
    attempt = Path(policy["paths"]["attempt"])
    events_path = attempt / "events.jsonl"
    events = [json.loads(line) for line in stable_bytes(events_path).splitlines()]
    require(all(type(row.get("returncode")) is int for row in events if row["event"] == "worker_reaped"),
            "worker reap code must be an integer")
    starts, journal = journal_helper.audit_journal(events, policy["cohort"]["seeds"], 2)
    plan = read(policy["attempt_plan"]["path"])
    execution_rows = manifest["execution_receipts"]
    require(len(execution_rows) == 8, "eight worker execution receipts required")
    for seed, row, cohort in zip(policy["cohort"]["seeds"], execution_rows, manifest["seeds"]):
        require(Path(row["path"]).resolve() == attempt / f"seed-{seed}-execution.json", "worker receipt route differs")
        retain(row)
        receipt = read(row["path"])
        require(receipt.get("seed") == seed and receipt.get("pid") == starts[seed]["pid"]
                and receipt.get("complete") is True and receipt.get("failure") is None
                and receipt.get("rate_analysis_performed") is False
                and exact_integer(receipt.get("native_steps"), 9_600_000)
                and receipt.get("CPU_affinity") == [starts[seed]["cpu"]]
                and receipt.get("thread_environment") == {name: "1" for name in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                "successful matching worker execution required")
        require([[branch["batch"], branch["timestep_fs"]] for branch in receipt["completed_branches"]]
                == policy["cohort"]["branch_order"], "worker branch order differs")
        require(all(a["report"] == b["report"] and a["native_steps"] == b["native_steps"]
                    for a, b in zip(receipt["completed_branches"], cohort["branches"])),
                "worker/global branch report identities differ")
        for label in ("whole_worker_call_wall_seconds", "whole_process_user_CPU_seconds", "whole_process_system_CPU_seconds"):
            require(isinstance(receipt.get(label), (float, int)) and not isinstance(receipt[label], bool)
                    and math.isfinite(receipt[label]) and receipt[label] >= 0, "worker resource observation is invalid")
        claim_path = attempt / f"seed-{seed}-worker-claim.json"
        worker_claim = read(claim_path)
        require(worker_claim.get("seed") == seed and worker_claim.get("pid") == receipt["pid"]
                and worker_claim.get("parent_pid") == plan["coordinator_pid"]
                and worker_claim.get("coordinator_process_token") == plan["coordinator_process_token"],
                "worker owning-coordinator claim differs")
        retain(identity(claim_path))
    require(stable_bytes(attempt / "result.json") == stable_bytes(policy["paths"]["children_complete"]),
            "attempt/global completion bytes differ")
    for path in (events_path, attempt / "result.json"):
        retain(identity(path))
    for name in ("children_complete", "root_completion", "root_observed_exit", "root_tool_observation", "root_log"):
        retain(identity(policy["paths"][name]))
    unique = {row["path"]: row for row in rows}
    require(len([row for row in unique if row.endswith(".npz") and Path(row).is_relative_to(Path(policy["paths"]["data"]))]) == 800,
            "exactly 800 unique child raw identities required")
    checkpoint_budget(deadline, stop)
    return {"counts": counts, "journal": journal, "input_files": list(unique.values()),
            "all_final_raw_hashes_verified": True, "partial_analysis_performed": False}


def stop_and_drain(process, grace=5, forced=5):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=forced)
    require(process.poll() is not None, "owned analyzer drain is unconfirmed; permanent claim remains")
    return process.returncode


def execute_analyzer(policy, directory, before, deadline, stop, *, popen=subprocess.Popen, lifecycle=None):
    lifecycle = {} if lifecycle is None else lifecycle
    checkpoint_budget(deadline, stop)
    require(not Path(policy["paths"]["analysis_output"]).exists(), "analysis output appeared before invocation")
    for row in before["input_files"]:
        check_identity(row)
    checkpoint_budget(deadline, stop)
    env = dict(os.environ, **policy["analysis_resources"]["selected_environment"])
    launch = {"created_utc": now(), "command": policy["command"], "cwd": policy["working_directory"],
              "selected_environment": policy["analysis_resources"]["selected_environment"],
              "verified_input_manifest_sha256": identity(directory / "verified-inputs.json")["sha256"],
              "one_analyzer_invocation": True}
    write_new(directory / "analyzer-launch.json", launch)
    process = None
    before_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
    entered = time.monotonic()
    try:
        with (directory / "analyzer.log").open("x") as log:
            checkpoint_budget(deadline, stop)
            require(not Path(policy["paths"]["child_lock"]).exists(), "child campaign lease reappeared before invocation")
            process = popen(policy["command"], cwd=policy["working_directory"], env=env,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            lifecycle.update(analyzer_invocations=1, owned_analyzer_drained=False, analyzer_PID=process.pid)
            write_new(directory / "analyzer-started.json", {"PID": process.pid, "created_utc": now()})
            while process.poll() is None:
                checkpoint_budget(deadline, stop)
                time.sleep(min(.1, max(0., deadline-time.monotonic())))
            checkpoint_budget(deadline, stop)
            require(exact_integer(process.returncode, 0), "frozen analyzer exited unsuccessfully")
    finally:
        if process is not None:
            stop_and_drain(process, policy["analysis_resources"]["termination_grace_seconds"],
                           policy["analysis_resources"]["forced_reap_seconds"])
            lifecycle["owned_analyzer_drained"] = True
            after_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
            write_new(directory / "analyzer-exit.json", {"PID": process.pid, "exit_code": process.returncode,
                "owned_analyzer_drained": True, "observed_utc": now(), "wall_seconds": time.monotonic()-entered,
                "child_CPU_seconds": after_cpu.ru_utime+after_cpu.ru_stime-before_cpu.ru_utime-before_cpu.ru_stime,
                "child_peak_RSS_KiB": after_cpu.ru_maxrss})
    for row in before["input_files"]:
        checkpoint_budget(deadline, stop)
        check_identity(row)
    require(not Path(policy["paths"]["child_lock"]).exists(), "child campaign lease reappeared")
    report_path = Path(policy["paths"]["analysis_output"]) / "bead-comparison.json"
    report = read(report_path)
    require(report.get("schema") == "pyrpmd.h3.complete-P256-dynamic-bead-comparison.v1"
            and report.get("completed_seeds") == report.get("expected_seeds") == policy["cohort"]["seeds"]
            and exact_integer(report.get("beads"), 256) and exact_integer(report.get("horizon_fs"), 200)
            and report.get("pair_batches") == [0, 1]
            and exact_integer(report.get("native_steps_both_timesteps"), 76_800_000)
            and report.get("protocol_sha256") == policy["child_protocol"]["sha256"]
            and report.get("analysis_source_sha256") == policy["analyzer"]["sha256"]
            and report.get("P128_baseline_sha256") == policy["baseline"]["sha256"]
            and report.get("parent_manifest_sha256") == policy["parent_manifest"]["sha256"]
            and isinstance(report.get("registered_full_rate_bead_control_passed"), bool),
            "analyzer report is not the registered complete comparison")
    checkpoint_budget(deadline, stop)
    return {"complete_cohort_analysis_executed_once": True, "analyzer_invocations": 1,
            "all_input_hashes_unchanged_after_analysis": True, "owned_analyzer_drained": True,
            "report": identity(report_path),
            "scientific_control_passed": report["registered_full_rate_bead_control_passed"],
            "no_extension_or_retry_launched": True}


def run(adoption_path):
    policy, adoption = load_authorization(adoption_path)
    for row in policy["static_guards"]:
        check_identity(row)
    directory, registration = claim(policy, adoption_path, adoption)
    own_guards = [registration[name] for name in ("policy", "source", "adoption", "independent_review")]
    result = {"complete_cohort_analysis_executed_once": False, "analyzer_invocations": 0}
    lifecycle = {"analyzer_invocations": 0, "owned_analyzer_drained": True}
    error = None
    entered = time.monotonic()
    with deferred_signals() as stop:
        try:
            metadata = wait_ready(policy, stop)
            write_new(directory / "root-completion-gate.json", {"observed_utc": now(),
                      "root_exit": identity(policy["paths"]["root_observed_exit"]),
                      "global_completion": identity(policy["paths"]["children_complete"]),
                      "no_raw_branch_reads_before_this_gate": True})
            os.sched_setaffinity(0, set(policy["analysis_resources"]["CPU_affinity"]))
            deadline = time.monotonic() + policy["analysis_resources"]["complete_verification_analysis_and_finalization_wall_seconds"]
            for row in own_guards:
                check_identity(row)
            before = verify_final_inputs(policy, metadata, deadline, stop)
            before["input_files"].extend(own_guards)
            write_new(directory / "verified-inputs.json", before)
            result = execute_analyzer(policy, directory, before, deadline-10, stop, lifecycle=lifecycle)
        except BaseException as caught:
            error = caught
            result["error"] = {"type": type(caught).__name__, "message": str(caught)}
            # A launched, failed or interrupted process still consumes the one
            # permitted invocation; never imply that it may be retried.
            result.update(lifecycle)
            result["permanent_attempt_claim_retained"] = True
        result.update({"schema": "pyrpmd.h3.p256-complete-analysis-execution.v1", "created_utc": now(),
                       "registration": registration, "elapsed_wall_seconds": time.monotonic()-entered,
                       "new_physical_evaluations": 0, "automatic_retry_resume_extension": False})
        write_new(directory / "execution.json", result)
    if error is not None:
        raise error
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.authorization)
    print(json.dumps({name: result[name] for name in
                     ("complete_cohort_analysis_executed_once", "analyzer_invocations", "scientific_control_passed")}))


if __name__ == "__main__":
    main()
