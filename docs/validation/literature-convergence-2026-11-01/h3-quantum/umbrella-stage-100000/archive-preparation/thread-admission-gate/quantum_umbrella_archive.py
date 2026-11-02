"""Preserve the completed fixed100000 umbrella cohort and replay its analysis.

This is an additive local archive operation, never a sampling/continuation
driver. Root's actual sampling and analysis exits are required before raw reads.
The existing frozen scheduler inspector and scientific analyzer are reused.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import resource
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
PLAN = ROOT / "docs/validation/literature-convergence-2026-11-01/h3-quantum/umbrella-stage-100000/archive-preparation/plan.json"
PLAN_SHA = "d06a07fed6d6e69f2ddd09c614d9fc17b336c669c5f4013d3cf65ac45e7d2fa4"
HELPER_SHA = "67e4e3eb1b7a7cc875fd14c74829c71be265a8171c516b7b10af5dca461d7b7e"
SEEDS = [211017, 211019, 211021, 211023, 211027, 211031, 211033, 211039]
SCHEMA = "pyrpmd.h3.umbrella100000-archive-adoption.v1"
STAGE = 100000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def json_bytes(raw):
    return json.loads(raw)


def record(path, check=lambda: None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "expected a regular non-symlink file")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        while True:
            check()
            block = stream.read(1024*1024)
            if not block:
                break
            value.update(block)
        after = os.fstat(stream.fileno())
    fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(fields(before) == fields(after) == fields(path.stat()), "file changed while hashing")
    return {"path": str(path.resolve()), "bytes": after.st_size, "sha256": value.hexdigest()}


def bound(row, check=lambda: None):
    require(record(row["path"], check) == row, f"bound input changed: {row['path']}")
    return Path(row["path"])


def load_bound(row, check=lambda: None):
    path = bound(row, check)
    raw = path.read_bytes()
    require(len(raw) == row["bytes"] and digest(raw) == row["sha256"], "JSON changed after hashing")
    return json_bytes(raw)


def strict_zero(value):
    return type(value) is int and value == 0


def root_exit(row, session):
    require(type(row.get("exec_session_id")) is int and row["exec_session_id"] == session
            and strict_zero(row.get("exit_code")) and row.get("all_owned_workers_drained") is True
            and isinstance(row.get("observed_utc"), str) and bool(row["observed_utc"]),
            "actual root-observed zero exit and drain required")
    bound(row["actual_tool_observation"])


def validate_complete_metadata(plan, adoption, completion, execution, comparison):
    """Pure metadata gate, before a checkpoint or numerical array is opened."""
    paths = plan["expected_paths"]
    require(completion.get("schema") == "pyrpmd.h3.quantum-umbrella-scheduler-completion.v1"
            and completion.get("production") == STAGE
            and completion.get("requested_seeds") == completion.get("completed_seeds") == SEEDS
            and completion.get("all_submitted_workers_drained") is True
            and completion.get("prior_prefix_and_state_files_unchanged") is True
            and completion.get("attempt_directory") == paths["scheduler_attempt"]
            and completion.get("analysis_recomputed") is False,
            "exact drained complete100000 scheduler cohort required")
    fixed = plan["fixed_inputs"]
    require(completion.get("protocol_sha256") == fixed["protocol"]["sha256"]
            and completion.get("prior_snapshot_sha256") == fixed["prior_snapshot"]["sha256"]
            and completion.get("decision_sha256") == fixed["precision_continuation_decision"]["sha256"],
            "registered continuation identities differ")
    source_map = {row["path"]: row["sha256"] for row in plan["protected_source_files"]}
    require(completion.get("scheduler_source_sha256") == source_map[str(ROOT/"benchmarks/h3/quantum_sampling_scheduler.py")],
            "frozen scheduler identity differs")
    require(execution.get("schema") == "pyrpmd.h3.complete-umbrella-analysis-execution.v1"
            and strict_zero(execution.get("returncode")) and execution.get("stage") == STAGE
            and execution.get("independent_seeds") == SEEDS
            and execution.get("complete_cohort_required") is True
            and execution.get("available_only_used") is False
            and execution.get("sources_unchanged") is True
            and execution.get("completion_unchanged") is True
            and execution.get("all_submitted_workers_drained_and_lock_released_required") is True
            and execution.get("no_refinement_launched") is True
            and execution.get("completion_manifest") == paths["global_completion"]
            and execution.get("completion_sha256") == adoption["global_completion"]["sha256"],
            "complete unchanged original analysis execution required")
    expected_command = [str(ROOT/".venv/bin/python"), str(ROOT/"benchmarks/h3/quantum_sampling_analyze.py"),
        "--campaign", paths["campaign"], "--protocol", fixed["protocol"]["path"],
        "--output", paths["analysis"], "--stage", "100000"]
    require(execution.get("command") == expected_command, "original analysis command changed")
    require(comparison.get("schema") == "pyrpmd.h3.quantum-ladder-comparison.v1"
            and comparison.get("production_sweeps_per_ladder") == STAGE
            and comparison.get("expected_independent_seeds") == comparison.get("completed_independent_seeds") == SEEDS
            and comparison.get("protocol_sha256") == fixed["protocol"]["sha256"]
            and comparison.get("criteria", {}).get("complete_registered_cohort") is True,
            "complete original scientific comparison required; partial estimates are excluded")
    # A failed precision/agreement/other scientific gate is valid complete data.
    require(type(comparison.get("registered_primary_qtst_gates_passed")) is bool,
            "original scientific gate result must be explicit")


def authorize(path):
    require(record(PLAN)["sha256"] == PLAN_SHA, "fixed archive plan changed")
    plan, adoption = json.loads(PLAN.read_text()), json.loads(Path(path).read_text())
    require(adoption.get("schema") == SCHEMA and adoption.get("adopted") is True
            and adoption.get("plan_sha256") == PLAN_SHA
            and adoption.get("source_sha256") == record(__file__)["sha256"]
            and adoption.get("no_pending_or_running_300000") is True,
            "exact root adoption and no-extension attestation required")
    affinity = adoption.get("CPU_affinity")
    require(isinstance(affinity, list) and len(affinity) == 1 and type(affinity[0]) is int,
            "root must allocate one actual CPU")
    root_exit(adoption["sampling_exit"], 57942)
    root_exit(adoption["analysis_exit"], 1094)
    paths = plan["expected_paths"]
    for name, expected in (("global_completion", paths["global_completion"]),
                           ("analysis_execution", str(Path(paths["analysis"])/"execution.json")),
                           ("analysis_report", str(Path(paths["analysis"])/"campaign-comparison.json"))):
        require(adoption[name]["path"] == expected, "final input route differs")
    require(not Path(paths["scheduler_lease"]).exists(), "sampling lease still exists")
    completion = load_bound(adoption["global_completion"])
    execution = load_bound(adoption["analysis_execution"])
    comparison = load_bound(adoption["analysis_report"])
    validate_complete_metadata(plan, adoption, completion, execution, comparison)
    return plan, adoption, completion, execution, comparison


class Budget:
    def __init__(self, limits, owned, *, started=None):
        self.limits, self.owned = limits, owned
        self.started = time.monotonic() if started is None else started
        self.deadline = self.started + limits["whole_call_wall_seconds"] - 10
        self.max_combined_RSS_bytes = 0

    def observations(self):
        parent = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = parent.ru_utime+parent.ru_stime+children.ru_utime+children.ru_stime
        rss = parent.ru_maxrss*1024
        process = self.owned.get("process")
        if process is not None and process.poll() is None:
            try:
                stat = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
                cpu += (int(stat[11])+int(stat[12]))/os.sysconf("SC_CLK_TCK")
                rss += int(stat[21])*os.sysconf("SC_PAGE_SIZE")
            except FileNotFoundError:
                process.poll()
                children = resource.getrusage(resource.RUSAGE_CHILDREN)
                cpu = parent.ru_utime+parent.ru_stime+children.ru_utime+children.ru_stime
        self.max_combined_RSS_bytes = max(self.max_combined_RSS_bytes, rss)
        return {"whole_parent_and_owned_child_CPU_seconds": cpu,
                "elapsed_wall_seconds": time.monotonic()-self.started,
                "maximum_observed_parent_peak_plus_active_child_RSS_bytes": self.max_combined_RSS_bytes}

    def check(self):
        value = self.observations()
        require(value["whole_parent_and_owned_child_CPU_seconds"] < self.limits["whole_call_CPU_seconds"],
                "registered aggregate CPU budget exhausted")
        require(value["maximum_observed_parent_peak_plus_active_child_RSS_bytes"] <= self.limits["peak_working_memory_target_bytes"],
                "registered observed working-memory budget exhausted")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("archive/replay work deadline exhausted; drain reserve remains")


def audit_seed_rows(campaign, prior, completion, protocol, inspector, check=lambda: None):
    rows = []
    require([row["seed"] for row in completion["snapshot"]["seeds"]] == SEEDS, "snapshot seed order differs")
    require([row["seed"] for row in prior["seeds"]] == SEEDS, "original prior seed order differs")
    for expected, old in zip(completion["snapshot"]["seeds"], prior["seeds"]):
        check()
        actual = inspector(campaign, old, STAGE, protocol, prior["protocol"]["sha256"],
                           prior["source_sha256"]["benchmarks/h3/quantum_sampling_campaign.py"])
        require(actual == expected, "checkpoint/state/RNG/prefix differs from complete global snapshot")
        require(actual["completed_sweeps"] == 105000
                and actual["arrays"]["samples"]["shape"] == [100000, 48]
                and actual["arrays"]["diagnostics"]["shape"] == [10000, 48, 7],
                "registered100000 observation shape differs")
        rows.append(actual)
    return rows


def member_path(name):
    parsed = PurePosixPath(name)
    require(not parsed.is_absolute() and ".." not in parsed.parts and "\\" not in name
            and parsed.as_posix() == name and name not in ("", "."), "unsafe archive member")
    return parsed


def archive_entries(plan, adoption, adoption_path, completion, check):
    paths = plan["expected_paths"]; campaign, analysis = Path(paths["campaign"]), Path(paths["analysis"])
    pairs = []
    def add(path, name, category):
        member_path(name)
        pairs.append({"source": record(path, check), "member": name, "category": category})
    for seed in SEEDS:
        base = campaign/f"seed-{seed}"
        for name in ("checkpoint.npz", "settings.json", "progress.json", "run-report.json"):
            add(base/name, f"data/campaign-run/seed-{seed}/{name}", "complete100000_checkpoint_or_metadata")
        for stage in (30000, 100000):
            for name in ("state.npz", "stage.json"):
                add(base/f"stages/{stage}"/name, f"data/campaign-run/seed-{seed}/stages/{stage}/{name}",
                    "historical_prefix_state" if stage == 30000 else "complete100000_restart_state")
        add(campaign/f"seed-{seed}-sampling.log", f"data/campaign-run/seed-{seed}-sampling.log", "full_retained_sampling_stdout")
    for name in ("stage-30000.json", "stage-100000.json"):
        add(campaign/name, "data/campaign-run/"+name, "global_stage_manifest")
    attempt = Path(paths["scheduler_attempt"])
    for name in ("plan.json", "events.jsonl"):
        add(attempt/name, "data/campaign-run/"+(attempt/name).relative_to(campaign).as_posix(), "scheduler_execution")
    attempt_plan = json.loads((attempt/"plan.json").read_text())
    if attempt_plan["worker_control"]:
        control = Path(attempt_plan["worker_control"])
        launch = load_bound(plan["fixed_inputs"]["launch_receipt"], check)
        require(str(control.resolve()) == launch["initial_worker_control"]["path"], "control differs from exact original launch route")
        add(control, "data/umbrella-worker-control.json", "final_scheduling_control")
    analysis_names = ["campaign-comparison.json", "pooled-wham.npz", "execution.json"]
    analysis_names += [f"seed-{seed}{suffix}" for seed in SEEDS for suffix in (".json", "-wham.npz")]
    require({p.name for p in analysis.iterdir() if p.is_file()} == set(analysis_names), "unexpected original analysis member")
    for name in analysis_names:
        add(analysis/name, "data/full-cohort-stage-100000/"+name, "complete_original_analysis")
    for key, row in plan["fixed_inputs"].items():
        if key in ("prior_archive_identity_from_frozen_snapshot", "prior_member_inventory"):
            continue
        add(row["path"], "reference/"+key+".json", "fixed_protocol_or_prior_identity")
    add(adoption_path, "execution/root-archive-adoption.json", "root_archive_adoption")
    for kind in ("sampling_exit", "analysis_exit"):
        add(adoption[kind]["actual_tool_observation"]["path"], f"execution/{kind}-tool-observation.json", "actual_root_exec_observation")
    for name in ("sampling_stdout", "analysis_stdout", "worker_control_change"):
        row = adoption[name]
        if row is None:
            require(name == "analysis_stdout" and isinstance(adoption.get("analysis_stdout_scope"), str)
                    and bool(adoption["analysis_stdout_scope"]), "missing optional analysis stdout must have an explicit retained-output scope")
            continue
        bound(row, check)
        require(Path(row["path"]).resolve().is_relative_to(WORK/"literature-convergence-quantum"),
                "root execution metadata route leaves registered workspace campaign")
        add(row["path"], "execution/"+name+(".json" if name == "worker_control_change" else ".log"), "root_execution_metadata")
    require(len({row["member"] for row in pairs}) == len(pairs), "duplicate archive member")
    limits = plan["proposed_resources_not_yet_adopted"]
    require(sum(row["source"]["bytes"] for row in pairs) <= limits["maximum_total_input_bytes"]
            and max(row["source"]["bytes"] for row in pairs) <= limits["maximum_single_file_bytes"], "archive size limit exceeded")
    require(sum(row["member"].endswith(".npz") for row in pairs) == 33, "exact33 primary raw NPZ files required")
    return sorted(pairs, key=lambda row: row["member"])


def pack_and_extract(entries, partial, restored, check):
    require(not Path(partial).exists() and not Path(restored).exists(), "never overwrite archive or restoration")
    Path(restored).mkdir()
    with Path(partial).open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for row in entries:
                    path = bound(row["source"], check)
                    info = tarfile.TarInfo(row["member"]); info.size = row["source"]["bytes"]
                    info.mode = 0o444; info.mtime = 0
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
                    check()
        raw.flush(); os.fsync(raw.fileno())
    expected = {row["member"]: row for row in entries}; seen = set()
    with tarfile.open(partial, "r|gz") as archive:
        for info in archive:
            check(); member_path(info.name)
            require(info.name in expected and info.name not in seen and info.isfile() and not info.issparse(),
                    "unexpected duplicate/nonregular archive member")
            row = expected[info.name]; require(info.size == row["source"]["bytes"], "archived size differs")
            target = Path(restored)/info.name; target.parent.mkdir(parents=True, exist_ok=True)
            value = hashlib.sha256(); length = 0
            with archive.extractfile(info) as source, target.open("xb") as output:
                while True:
                    check(); block = source.read(1024*1024)
                    if not block:
                        break
                    value.update(block); length += len(block); output.write(block)
            require(length == row["source"]["bytes"] and value.hexdigest() == row["source"]["sha256"],
                    "archived/extracted member bytes differ")
            copied = record(target, check)
            require(all(copied[key] == row["source"][key] for key in ("bytes", "sha256")), "extracted file hash differs")
            seen.add(info.name)
    require(seen == set(expected), "missing archived member")


def compare_replay(original, replay, check=lambda: None):
    import numpy as np
    original, replay = Path(original), Path(replay)
    rows = []
    for name in ["campaign-comparison.json", *[f"seed-{seed}.json" for seed in SEEDS]]:
        check(); first, second = [json.loads((p/name).read_text()) for p in (original, replay)]
        ignored = ["created_utc", "elapsed_seconds"] if name == "campaign-comparison.json" else []
        for key in ignored:
            first.pop(key); second.pop(key)
        require(json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(second, sort_keys=True, allow_nan=False),
                f"exact scientific JSON replay differs: {name}")
        rows.append({"file": name, "exact_scientific_JSON_equal": True, "ignored_top_level_fields": ignored})
    for name in ["pooled-wham.npz", *[f"seed-{seed}-wham.npz" for seed in SEEDS]]:
        check()
        with np.load(original/name, allow_pickle=False) as a, np.load(replay/name, allow_pickle=False) as b:
            require(set(a.files) == set(b.files) and len(a.files) == len(set(a.files))
                    and len(b.files) == len(set(b.files)), "WHAM array inventory differs")
            for key in a.files:
                check(); x, y = a[key], b[key]
                require(not x.dtype.hasobject and not y.dtype.hasobject and x.dtype == y.dtype and x.shape == y.shape
                        and digest(np.ascontiguousarray(x).tobytes()) == digest(np.ascontiguousarray(y).tobytes()),
                        f"exact WHAM array replay differs: {name}:{key}")
        rows.append({"file": name, "all_array_dtype_shape_bytes_equal": True})
    return rows


def main_run(adoption_path):
    started = time.monotonic()
    require(all(os.environ.get(name) == "1" for name in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")),
            "one-thread environment is required before admission or NumPy imports")
    plan, adoption, completion, execution, comparison = authorize(adoption_path)
    paths = plan["expected_paths"]
    output = Path(paths["new_evidence"]); output.mkdir(parents=True, exist_ok=False)
    require(not Path(paths["archive"]).exists() and not Path(paths["member_inventory"]).exists()
            and not Path(paths["fresh_replay"]).exists(), "final destinations already exist")
    write_new(output/"RUN_ONCE.json", {"created_utc": now(), "adoption": record(adoption_path),
        "source": record(__file__), "plan": record(PLAN), "automatic_retry_or_continuation": False})
    os.sched_setaffinity(0, set(adoption["CPU_affinity"])); os.nice(10)
    helper_path = ROOT/"benchmarks/h3/p256_complete_analysis.py"
    require(record(helper_path)["sha256"] == HELPER_SHA, "reviewed timeout helper changed")
    for value in (str(ROOT), str(ROOT/"src")):
        if value not in sys.path:
            sys.path.insert(0, value)
    helper = importlib.import_module("benchmarks.h3.p256_complete_analysis")
    require(Path(helper.__file__).resolve() == helper_path.resolve(), "timeout-helper import route differs")
    owned, result, error = {}, {"complete_archive": False, "replay_invocations": 0}, None
    budget = Budget(plan["proposed_resources_not_yet_adopted"], owned, started=started)
    partial = Path(paths["archive"]+".partial")
    with helper.deferred_signals() as stop:
        def check():
            helper.checkpoint_budget(budget.deadline, stop); budget.check()
        try:
            with helper.work_alarm(budget.deadline, owned):
                check()
                guards = [*plan["protected_source_files"], *plan["fixed_inputs"].values(),
                          *[adoption[name] for name in ("global_completion", "analysis_execution", "analysis_report")],
                          adoption["sampling_exit"]["actual_tool_observation"], adoption["analysis_exit"]["actual_tool_observation"],
                          record(helper_path), record(__file__), record(PLAN), record(adoption_path)]
                for row in guards:
                    bound(row, check)
                fixed = plan["fixed_inputs"]
                for row in fixed.values():
                    bound(row, check)
                protocol = load_bound(fixed["protocol"], check)
                prior = load_bound(fixed["prior_snapshot"], check)
                scheduler = importlib.import_module("benchmarks.h3.quantum_sampling_scheduler")
                require(Path(scheduler.__file__).resolve() == ROOT/"benchmarks/h3/quantum_sampling_scheduler.py",
                        "frozen inspector import route differs")
                inspector = scheduler.inspect_seed
                import numpy as np
                original_runtime = json.loads((Path(paths["scheduler_attempt"])/"plan.json").read_text())["runtime"]
                require(original_runtime["python"] == sys.version and original_runtime["numpy"] == np.__version__
                        and Path(sys.executable).resolve() == (ROOT/".venv/bin/python").resolve(),
                        "same original Python/NumPy runtime is required for exact replay")
                original_rows = audit_seed_rows(paths["campaign"], prior, completion, protocol, inspector, check)
                require(not Path(paths["scheduler_lease"]).exists()
                        and not (Path(paths["campaign"])/"stage-300000.json").exists(), "unexpected new sampling writer or300000 stage")
                entries = archive_entries(plan, adoption, adoption_path, completion, check)
                write_new(output/"registered-members.json", {"members": entries, "member_count": len(entries),
                    "raw_NPZ_count": 33, "sources": guards, "external_prior_archive": fixed["prior_archive_identity_from_frozen_snapshot"]})
                write_new(output/"stage-snapshot.json", {"global_completion": adoption["global_completion"],
                    "prior_snapshot": fixed["prior_snapshot"], "seeds": original_rows})
                pack_and_extract(entries, partial, paths["fresh_restore"], check)
                restored = Path(paths["fresh_restore"])
                extracted_rows = audit_seed_rows(restored/"data/campaign-run", prior, completion, protocol, inspector, check)
                require(extracted_rows == original_rows, "extracted state/prefix identities differ")
                env = dict(os.environ, PYTHONPATH=f"{ROOT}/src:{ROOT}", PYTHONDONTWRITEBYTECODE="1",
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
                command = [str(ROOT/".venv/bin/python"), str(ROOT/"benchmarks/h3/quantum_sampling_analyze.py"),
                    "--campaign", str(restored/"data/campaign-run"), "--protocol", str(restored/"reference/protocol.json"),
                    "--output", paths["fresh_replay"], "--stage", "100000"]
                write_new(output/"replay-launch.json", {"created_utc": now(), "command": command,
                    "original_execution": adoption["analysis_execution"], "one_analysis_replay_only": True})
                with (output/"replay.log").open("x") as log:
                    check(); mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
                    try:
                        from functools import partial as bind_arguments
                        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, preexec_fn=bind_arguments(helper.analyzer_alarm, budget.deadline-time.monotonic()))
                        owned["process"] = process; result["replay_invocations"] = 1
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
                    while process.poll() is None:
                        check(); time.sleep(.1)
                    require(strict_zero(process.returncode), "unchanged replay analyzer failed")
                result["replay_comparison"] = compare_replay(restored/"data/full-cohort-stage-100000", paths["fresh_replay"], check)
                for row in entries:
                    bound(row["source"], check)
                for row in guards:
                    bound(row, check)
                require(not Path(paths["scheduler_lease"]).exists(), "unexpected writer after archival")
                archive_identity = record(partial, check); archive_identity["path"] = paths["archive"]
                inventory = {"schema": "pyrpmd.h3.quantum-umbrella-archive-members.v2", "archive": archive_identity,
                    "members": entries, "member_count": len(entries), "member_bytes": sum(r["source"]["bytes"] for r in entries),
                    "category_counts": dict(Counter(r["category"] for r in entries)), "raw_NPZ_count": 33}
                check(); os.link(partial, paths["archive"])
                write_new(paths["member_inventory"], inventory); write_new(output/"member-inventory.json", inventory)
                result.update(complete_archive=True, archive=archive_identity, member_inventory=record(paths["member_inventory"]),
                    all_original_archived_extracted_bytes_verified=True, all8_state_RNG_and_prefixes_equal=True,
                    exact_scientific_replay=True, original_scientific_criteria=comparison["criteria"],
                    original_registered_primary_qtst_gates_passed=comparison["registered_primary_qtst_gates_passed"],
                    physical_rate_or_dynamic_bead_validation_claimed=False)
        except BaseException as caught:
            error = caught; result["failure"] = {"type": type(caught).__name__, "message": str(caught)}
        finally:
            process = owned.get("process")
            if process is not None:
                try:
                    helper.stop_and_drain(process, 5, 5); result["owned_replay_drained"] = True
                except BaseException as caught:
                    result["owned_replay_drained"] = False; error = caught
                    result["drain_failure"] = {"type": type(caught).__name__, "message": str(caught)}
        result.update(schema="pyrpmd.h3.quantum-umbrella100000-archive-execution.v1", created_utc=now(),
            resources=budget.observations(), source=record(__file__), plan=record(PLAN),
            original_data_modified=False, new_PES_or_dynamics_calls=0, automatic300000_continuation=False,
            permanent_attempt_claim_retained=True, upload_performed=False, Git_mutated=False,
            original_analysis_stdout_scope=adoption.get("analysis_stdout_scope", "full retained stdout file supplied"))
        write_new(output/"execution.json", result)
    if error is not None:
        raise error
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    result = main_run(args.authorization)
    print(json.dumps({key: result[key] for key in ("complete_archive", "archive", "exact_scientific_replay")}, indent=2))


if __name__ == "__main__":
    main()
