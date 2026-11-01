"""Read-only archive of the completed100000 cohort with a bound lease anomaly.

The predecessor, its28-case evidence and every scientific input remain frozen.
This tool does not retire a lease or admit sampling. It reuses the predecessor's
archive, checkpoint and exact scientific replay functions without monkeypatching.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time

from benchmarks.h3 import quantum_umbrella_archive as prior

ROOT, WORK = prior.ROOT, prior.WORK
POLICY = ROOT / "docs/validation/literature-convergence-2026-11-01/h3-quantum/umbrella-stage-100000/archive-preparation-v2/policy.json"
POLICY_SHA = "e73a940695fdfe86460c76b1d3ade84555867d85f4ca115fe103b44c233dbfe0"
PRIOR_SOURCE_SHA = "af1288ec403880899a42202ae7e5c1ab6c1f44c419dad6f1c73678d7ffb9738f"
SCHEMA = "pyrpmd.h3.umbrella100000-readonly-archive-adoption.v2"
ATTESTATIONS = (
    "complete_sampling_and_analysis_exits_observed",
    "all_owned_sampling_workers_drained",
    "no_existing_or_pending_new_legacy_or_direct_sampler",
    "no_300000_admitted",
    "no_writers_admitted_until_archive_attempt_finishes",
    "original_campaign_remains_unchanged",
    "future_300000_only_new_isolated_restore_from_verified100000_archive",
    "lease_anomaly_cause_not_assumed",
)
require, record, bound, load_bound = prior.require, prior.record, prior.bound, prior.load_bound
write_new, now = prior.write_new, prior.now


def metadata(path):
    value = Path(path).lstat()
    return {"device": value.st_dev, "inode": value.st_ino, "mode": value.st_mode,
            "size": value.st_size, "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns}


def observe_lease(path, check=lambda: None):
    """Record the exact present lease, without changing it or inferring liveness."""
    path = Path(path)
    require(path.resolve() == path and not path.is_symlink()
            and stat.S_ISDIR(path.lstat().st_mode), "lease route must be an exact non-symlink directory")
    before = metadata(path)
    require(sorted(item.name for item in path.iterdir()) == ["owner.json"], "lease membership changed")
    owner = path / "owner.json"
    owner_before = metadata(owner)
    require(stat.S_ISREG(owner_before["mode"]) and not owner.is_symlink(), "lease owner must be regular")
    identity = record(owner, check)
    require(metadata(owner) == owner_before and metadata(path) == before
            and sorted(item.name for item in path.iterdir()) == ["owner.json"], "lease changed during observation")
    return {"path": str(path), "directory": before, "owner": owner_before,
            "owner_identity": identity, "members": ["owner.json"]}


def guard_lease_anomaly(policy, plan, check=lambda: None):
    anomaly = policy["fixed_lease_anomaly"]
    require(anomaly.get("durable_lease_absence_claimed") is False
            and anomaly.get("PID_alone_proves_liveness") is False
            and anomaly.get("cause") == "unexplained", "anomaly scope changed")
    observations = {}
    for name in ("current_old_path", "quarantined_original"):
        check()
        expected = anomaly[name]
        actual = observe_lease(expected["path"], check)
        require(actual == expected, "bound anomalous lease identity changed")
        observations[name] = actual
    current, original = [observations[key] for key in ("current_old_path", "quarantined_original")]
    require(current["path"] == plan["expected_paths"]["scheduler_lease"]
            and current["directory"]["inode"] != original["directory"]["inode"]
            and current["owner"]["inode"] != original["owner"]["inode"]
            and current["owner_identity"]["sha256"] == original["owner_identity"]["sha256"]
            and current["owner_identity"]["bytes"] == original["owner_identity"]["bytes"] == 68,
            "recorded original/reappeared lease relationship differs")
    campaign = Path(plan["expected_paths"]["campaign"])
    future = [campaign/"stage-300000.json",
              *[campaign/f"seed-{seed}/stages/300000" for seed in prior.SEEDS]]
    require(not any(path.exists() or path.is_symlink() for path in future), "new300000 stage is forbidden")
    return observations


def validate_attestation(adoption):
    attestation = adoption.get("root_no_writer_attestation", {})
    require(set(attestation) == set(ATTESTATIONS)
            and all(attestation[key] is True for key in ATTESTATIONS),
            "explicit current and future no-writer root attestation required")
    require(adoption.get("lease_absence_claimed") is False
            and adoption.get("read_only_lease_anomaly_exception_adopted") is True
            and adoption.get("no_pending_or_running_300000") is True,
            "read-only anomaly exception must not assert lease absence")


def verify_actual_exit(row, expected, session, check=lambda: None):
    require(row.get("actual_tool_observation") == expected, "actual terminal observation route differs")
    observation = load_bound(expected, check)
    require(observation.get("schema") == "pyrpmd.root-actual-exec-observation.v1"
            and type(observation.get("exec_session_id")) is int
            and observation["exec_session_id"] == session
            and prior.strict_zero(observation.get("tool_result", {}).get("exit_code"))
            and bool(observation.get("tool_result", {}).get("chunk_id"))
            and isinstance(observation.get("observed_utc"), str)
            and bool(observation["observed_utc"]), "saved actual terminal observation is not a normal exact-session exit")
    require(type(row.get("exec_session_id")) is int and row["exec_session_id"] == session
            and prior.strict_zero(row.get("exit_code"))
            and row.get("all_owned_workers_drained") is True
            and row.get("observed_utc") == observation["observed_utc"],
            "root attestation must match its saved terminal observation")
    return observation


def verify_drain_journal(path):
    events = [json.loads(line) for line in Path(path).read_text().splitlines()]
    active, submitted, completed = set(), [], []
    drained, verified, limit = False, False, None
    for index, event in enumerate(events):
        kind = event.get("event")
        require(not verified, "events after final completion are forbidden")
        if kind == "worker_limit":
            require(not drained and event.get("active_workers") in (2, 4), "unexpected worker limit")
            limit = event["active_workers"]
        elif kind == "submitted":
            seed = event.get("seed")
            require(not drained and seed in prior.SEEDS and seed not in submitted and limit is not None,
                    "duplicate or late worker admission")
            active.add(seed); submitted.append(seed)
            require(event.get("active") == len(active) <= limit, "journal active count differs")
        elif kind == "seed_complete":
            seed = event.get("seed")
            require(not drained and seed in active, "worker completion has no matching admission")
            active.remove(seed); completed.append(seed)
        elif kind == "all_submitted_workers_drained":
            require(not drained and not active and submitted == prior.SEEDS
                    and sorted(completed) == prior.SEEDS, "drain precedes exact full-cohort completion")
            drained = True
        elif kind == "complete_cohort_verified":
            require(drained and event.get("seeds") == prior.SEEDS and index == len(events)-1,
                    "global verification must follow all-worker drain")
            verified = True
        else:
            raise ValueError("unknown/failure journal event cannot admit archival")
    require(drained and verified and not active, "complete all-worker drain journal required")
    return {"submitted": submitted, "completed": sorted(completed), "active_workers": 0,
            "all_owned_workers_drained": True, "events": len(events)}


def authorize(path):
    require(record(POLICY)["sha256"] == POLICY_SHA, "new read-only archive policy changed")
    policy = json.loads(POLICY.read_text())
    require(record(prior.__file__) == policy["frozen_v1_source"]
            and policy["frozen_v1_source"]["sha256"] == PRIOR_SOURCE_SHA,
            "frozen archive predecessor identity changed")
    require(policy["prior_plan"]["sha256"] == prior.PLAN_SHA, "predecessor plan identity differs")
    plan = deepcopy(load_bound(policy["prior_plan"]))
    require(set(policy["path_overrides"]) == {"new_evidence", "fresh_restore", "fresh_replay"},
            "only three new output directories may differ")
    plan["expected_paths"].update(policy["path_overrides"])
    adoption = json.loads(Path(path).read_text())
    require(adoption.get("schema") == SCHEMA and adoption.get("adopted") is True
            and adoption.get("policy_sha256") == POLICY_SHA
            and adoption.get("prior_plan_sha256") == prior.PLAN_SHA
            and adoption.get("source_sha256") == record(__file__)["sha256"],
            "exact new root adoption is required")
    validate_attestation(adoption)
    affinity = adoption.get("CPU_affinity")
    require(isinstance(affinity, list) and len(affinity) == 1 and type(affinity[0]) is int,
            "root must allocate one actual CPU")
    fixed = policy["fixed_completed_metadata"]
    for name, session in (("sampling", 57942), ("analysis", 1094)):
        verify_actual_exit(adoption[name+"_exit"], fixed["actual_"+name+"_tool_observation"], session)
    for name in ("global_completion", "analysis_execution", "analysis_report"):
        require(adoption.get(name) == fixed[name], "complete metadata identity differs from the reviewed policy")
    for row in fixed.values():
        bound(row)
    completion = load_bound(fixed["global_completion"])
    execution = load_bound(fixed["analysis_execution"])
    comparison = load_bound(fixed["analysis_report"])
    prior.validate_complete_metadata(plan, adoption, completion, execution, comparison)
    journal = verify_drain_journal(fixed["scheduler_drain_journal"]["path"])
    observed = guard_lease_anomaly(policy, plan)
    return policy, plan, adoption, completion, execution, comparison, journal, observed


def archive_entries(policy, plan, adoption, adoption_path, completion, check):
    entries = prior.archive_entries(plan, adoption, adoption_path, completion, check)
    anomaly = policy["fixed_lease_anomaly"]
    sources = {"policy": record(POLICY, check),
               "quarantined_original_owner": anomaly["quarantined_original"]["owner_identity"],
               "reappeared_owner": anomaly["current_old_path"]["owner_identity"],
               **policy["fixed_completed_metadata"]}
    for key, name in policy["whitelist_additions"].items():
        prior.member_path(name)
        require(name.startswith("execution/lease-anomaly/"), "anomaly evidence cannot create a restored campaign lease")
        bound(sources[key], check)
        entries.append({"source": sources[key], "member": name, "category": "read_only_lease_anomaly_evidence"})
    limits = plan["proposed_resources_not_yet_adopted"]
    require(len({row["member"] for row in entries}) == len(entries)
            and sum(row["source"]["bytes"] for row in entries) <= limits["maximum_total_input_bytes"]
            and max(row["source"]["bytes"] for row in entries) <= limits["maximum_single_file_bytes"]
            and sum(row["member"].endswith(".npz") for row in entries) == 33,
            "expanded exact archive whitelist or size budget differs")
    return sorted(entries, key=lambda row: row["member"])


def main_run(adoption_path):
    started = time.monotonic()
    require(all(os.environ.get(name) == "1" for name in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")),
            "one-thread environment is required before admission or NumPy imports")
    policy, plan, adoption, completion, execution, comparison, journal, first_observation = authorize(adoption_path)
    paths = plan["expected_paths"]
    output = Path(paths["new_evidence"]); output.mkdir(parents=True, exist_ok=False)
    require(not any(Path(paths[key]).exists() for key in
                    ("archive", "member_inventory", "fresh_restore", "fresh_replay")), "final destinations already exist")
    write_new(output/"RUN_ONCE.json", {"created_utc": now(), "adoption": record(adoption_path),
        "source": record(__file__), "policy": record(POLICY), "automatic_retry_or_continuation": False})
    os.sched_setaffinity(0, set(adoption["CPU_affinity"])); os.nice(10)
    helper_path = ROOT/"benchmarks/h3/p256_complete_analysis.py"
    require(record(helper_path)["sha256"] == prior.HELPER_SHA, "reviewed timeout helper changed")
    for value in (str(ROOT), str(ROOT/"src")):
        if value not in sys.path:
            sys.path.insert(0, value)
    helper = importlib.import_module("benchmarks.h3.p256_complete_analysis")
    require(Path(helper.__file__).resolve() == helper_path.resolve(), "timeout-helper import route differs")
    owned, result, error = {}, {"complete_archive": False, "replay_invocations": 0}, None
    budget = prior.Budget(plan["proposed_resources_not_yet_adopted"], owned, started=started)
    partial = Path(paths["archive"]+".partial")
    with helper.deferred_signals() as stop:
        def check():
            helper.checkpoint_budget(budget.deadline, stop); budget.check()
        try:
            with helper.work_alarm(budget.deadline, owned):
                check()
                guards = [*plan["protected_source_files"], *plan["fixed_inputs"].values(),
                          *policy["fixed_completed_metadata"].values(), policy["frozen_v1_source"],
                          policy["frozen_v1_review"], policy["prior_plan"],
                          record(helper_path), record(__file__), record(POLICY), record(adoption_path)]
                for row in guards:
                    bound(row, check)
                observations = [{"phase": "pre_raw", "leases": guard_lease_anomaly(policy, plan, check)}]
                require(observations[0]["leases"] == first_observation, "lease changed after admission")
                fixed = plan["fixed_inputs"]
                protocol = load_bound(fixed["protocol"], check)
                old_snapshot = load_bound(fixed["prior_snapshot"], check)
                scheduler = importlib.import_module("benchmarks.h3.quantum_sampling_scheduler")
                require(Path(scheduler.__file__).resolve() == ROOT/"benchmarks/h3/quantum_sampling_scheduler.py",
                        "frozen inspector import route differs")
                inspector = scheduler.inspect_seed
                import numpy as np
                original_runtime = json.loads((Path(paths["scheduler_attempt"])/"plan.json").read_text())["runtime"]
                require(original_runtime["python"] == sys.version and original_runtime["numpy"] == np.__version__
                        and Path(sys.executable).resolve() == (ROOT/".venv/bin/python").resolve(),
                        "same original Python/NumPy runtime is required for exact replay")
                original_rows = prior.audit_seed_rows(paths["campaign"], old_snapshot, completion, protocol, inspector, check)
                observations.append({"phase": "pre_pack", "leases": guard_lease_anomaly(policy, plan, check)})
                entries = archive_entries(policy, plan, adoption, adoption_path, completion, check)
                write_new(output/"registered-members.json", {"members": entries, "member_count": len(entries),
                    "raw_NPZ_count": 33, "sources": guards,
                    "external_prior_archive": fixed["prior_archive_identity_from_frozen_snapshot"]})
                write_new(output/"stage-snapshot.json", {"global_completion": adoption["global_completion"],
                    "prior_snapshot": fixed["prior_snapshot"], "seeds": original_rows})
                prior.pack_and_extract(entries, partial, paths["fresh_restore"], check)
                observations.append({"phase": "post_extract", "leases": guard_lease_anomaly(policy, plan, check)})
                restored = Path(paths["fresh_restore"])
                require(not (restored/"data/campaign-run/.quantum-scheduler-lock").exists(),
                        "restored data must not contain the original anomalous lease")
                extracted_rows = prior.audit_seed_rows(restored/"data/campaign-run", old_snapshot, completion, protocol, inspector, check)
                require(extracted_rows == original_rows, "extracted state/prefix identities differ")
                env = dict(os.environ, PYTHONPATH=f"{ROOT}/src:{ROOT}", PYTHONDONTWRITEBYTECODE="1",
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
                command = [str(ROOT/".venv/bin/python"), str(ROOT/"benchmarks/h3/quantum_sampling_analyze.py"),
                    "--campaign", str(restored/"data/campaign-run"), "--protocol", str(restored/"reference/protocol.json"),
                    "--output", paths["fresh_replay"], "--stage", "100000"]
                observations.append({"phase": "pre_replay", "leases": guard_lease_anomaly(policy, plan, check)})
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
                    require(prior.strict_zero(process.returncode), "unchanged replay analyzer failed")
                result["replay_comparison"] = prior.compare_replay(restored/"data/full-cohort-stage-100000", paths["fresh_replay"], check)
                for row in entries:
                    bound(row["source"], check)
                for row in guards:
                    bound(row, check)
                observations.append({"phase": "pre_publish", "leases": guard_lease_anomaly(policy, plan, check)})
                archive_identity = record(partial, check); archive_identity["path"] = paths["archive"]
                inventory = {"schema": "pyrpmd.h3.quantum-umbrella-archive-members.v3", "archive": archive_identity,
                    "members": entries, "member_count": len(entries), "member_bytes": sum(r["source"]["bytes"] for r in entries),
                    "category_counts": dict(Counter(r["category"] for r in entries)), "raw_NPZ_count": 33,
                    "read_only_lease_exception_policy": record(POLICY), "lease_absence_claimed": False}
                check(); os.link(partial, paths["archive"])
                write_new(paths["member_inventory"], inventory); write_new(output/"member-inventory.json", inventory)
                result.update(complete_archive=True, archive=archive_identity, member_inventory=record(paths["member_inventory"]),
                    all_original_archived_extracted_bytes_verified=True, all8_state_RNG_and_prefixes_equal=True,
                    exact_scientific_replay=True, original_scientific_criteria=comparison["criteria"],
                    original_registered_primary_qtst_gates_passed=comparison["registered_primary_qtst_gates_passed"],
                    original_lease_observations=observations, exact_drain_journal=journal,
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
        result.update(schema="pyrpmd.h3.quantum-umbrella100000-readonly-archive-execution.v2", created_utc=now(),
            resources=budget.observations(), source=record(__file__), policy=record(POLICY),
            frozen_predecessor=record(prior.__file__), original_data_modified=False,
            original_lease_modified=False, original_lease_absence_claimed=False, lease_anomaly_cause="unexplained",
            root_no_writer_attestation=adoption["root_no_writer_attestation"], new_PES_or_dynamics_calls=0,
            automatic300000_continuation=False, permanent_attempt_claim_retained=True, upload_performed=False,
            Git_mutated=False, original_analysis_stdout_scope=adoption.get("analysis_stdout_scope", "full retained stdout file supplied"))
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
