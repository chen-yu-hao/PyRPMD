"""Inert read-only lease-exception admission; no original cohort/PES is opened."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pytest

if sys.platform != "linux":
    pytest.skip("registered archive uses Linux ownership and signal guards", allow_module_level=True)

from benchmarks.h3 import quantum_umbrella_archive_v2 as tool


OWNER = b'{\n  "created_utc": "2026-11-01T18:34:51.123630+00:00",\n  "pid": 4\n}\n'


def put(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+"\n")
    return tool.record(path)


def leases(tmp_path):
    campaign = tmp_path/"campaign"
    original = tmp_path/"quarantine"
    current = campaign/".quantum-scheduler-lock"
    for path in (original, current):
        path.mkdir(parents=True)
        (path/"owner.json").write_bytes(OWNER)
    policy = {"fixed_lease_anomaly": {"current_old_path": tool.observe_lease(current),
        "quarantined_original": tool.observe_lease(original), "cause": "unexplained",
        "durable_lease_absence_claimed": False, "PID_alone_proves_liveness": False}}
    plan = {"expected_paths": {"campaign": str(campaign), "scheduler_lease": str(current)}}
    return policy, plan, current, original


def attestation():
    return {"root_no_writer_attestation": {key: True for key in tool.ATTESTATIONS},
            "lease_absence_claimed": False, "read_only_lease_anomaly_exception_adopted": True,
            "no_pending_or_running_300000": True}


def exit_fixture(tmp_path):
    observation = {"schema": "pyrpmd.root-actual-exec-observation.v1", "exec_session_id": 1094,
                   "observed_utc": "2026-11-01 21:28:28 UTC", "tool_result": {"exit_code": 0, "chunk_id": "inert"}}
    expected = put(tmp_path/"actual-terminal.json", observation)
    row = {"actual_tool_observation": expected, "exec_session_id": 1094, "exit_code": 0,
           "observed_utc": observation["observed_utc"], "all_owned_workers_drained": True}
    return observation, expected, row


def journal_fixture():
    events = [{"event": "worker_limit", "active_workers": 2}]
    for seed in tool.prior.SEEDS:
        events.extend([{"event": "submitted", "seed": seed, "active": 1},
                       {"event": "seed_complete", "seed": seed}])
    return events+[{"event": "all_submitted_workers_drained"},
                   {"event": "complete_cohort_verified", "seeds": tool.prior.SEEDS}]


def save_journal(path, events):
    path.write_text("".join(json.dumps(row)+"\n" for row in events))


def test_exact_present_anomaly_passes_without_mutation_or_absence_claim(tmp_path):
    policy, plan, current, original = leases(tmp_path)
    before = [tool.observe_lease(p) for p in (current, original)]
    actual = tool.guard_lease_anomaly(policy, plan)
    assert list(actual.values()) == before
    assert current.exists() and original.exists()
    assert [tool.observe_lease(p) for p in (current, original)] == before
    assert actual["current_old_path"]["owner_identity"]["sha256"] == actual["quarantined_original"]["owner_identity"]["sha256"]


@pytest.mark.parametrize("change", ["owner_bytes", "replacement_same_bytes", "extra_member", "owner_symlink",
                                    "missing_current", "quarantine_bytes", "new_global_stage", "new_seed_stage"])
def test_anomaly_change_refuses_admission_instead_of_retiring_or_ignoring_it(tmp_path, change):
    policy, plan, current, original = leases(tmp_path)
    if change == "owner_bytes":
        (current/"owner.json").write_bytes(OWNER.replace(b'"pid": 4', b'"pid": 9'))
    elif change == "replacement_same_bytes":
        replacement = tmp_path/"new-owner"; replacement.write_bytes(OWNER)
        os.replace(replacement, current/"owner.json")
    elif change == "extra_member":
        (current/"new-writer").write_text("not admitted")
    elif change == "owner_symlink":
        (current/"owner.json").unlink(); (current/"owner.json").symlink_to(original/"owner.json")
    elif change == "missing_current":
        (current/"owner.json").unlink(); current.rmdir()
    elif change == "quarantine_bytes":
        (original/"owner.json").write_bytes(b"changed")
    elif change == "new_global_stage":
        (current.parent/"stage-300000.json").write_text("{}")
    else:
        (current.parent/f"seed-{tool.prior.SEEDS[-1]}/stages/300000").mkdir(parents=True)
    with pytest.raises((ValueError, FileNotFoundError)):
        tool.guard_lease_anomaly(policy, plan)


@pytest.mark.parametrize("key", tool.ATTESTATIONS)
def test_every_root_no_writer_and_future_isolated_copy_condition_is_required(key):
    value = attestation(); tool.validate_attestation(value)
    value["root_no_writer_attestation"][key] = False
    with pytest.raises(ValueError, match="no-writer"):
        tool.validate_attestation(value)


def test_true_or_missing_absence_claim_is_not_a_lease_exception():
    value = attestation(); value["lease_absence_claimed"] = True
    with pytest.raises(ValueError, match="must not assert lease absence"):
        tool.validate_attestation(value)
    del value["lease_absence_claimed"]
    with pytest.raises(ValueError, match="must not assert lease absence"):
        tool.validate_attestation(value)


def test_saved_actual_terminal_semantics_and_matching_root_row(tmp_path):
    observation, expected, row = exit_fixture(tmp_path)
    assert tool.verify_actual_exit(row, expected, 1094) == observation


@pytest.mark.parametrize("change", ["wrong_session", "nonzero_exit", "bool_exit", "missing_chunk", "wrong_schema",
                                    "row_time", "row_exit", "row_drain"])
def test_rehashed_but_invalid_terminal_observation_is_rejected(tmp_path, change):
    observation, expected, row = exit_fixture(tmp_path)
    if change == "wrong_session": observation["exec_session_id"] = 57942
    elif change == "nonzero_exit": observation["tool_result"]["exit_code"] = 1
    elif change == "bool_exit": observation["tool_result"]["exit_code"] = False
    elif change == "missing_chunk": del observation["tool_result"]["chunk_id"]
    elif change == "wrong_schema": observation["schema"] = "declared-completion-only"
    elif change == "row_time": row["observed_utc"] = "different"
    elif change == "row_exit": row["exit_code"] = False
    else: row["all_owned_workers_drained"] = False
    expected = put(tmp_path/"actual-terminal.json", observation); row["actual_tool_observation"] = expected
    with pytest.raises(ValueError):
        tool.verify_actual_exit(row, expected, 1094)


def test_exact_complete_drain_journal(tmp_path):
    path = tmp_path/"journal.jsonl"; save_journal(path, journal_fixture())
    value = tool.verify_drain_journal(path)
    assert value["submitted"] == value["completed"] == tool.prior.SEEDS
    assert value["active_workers"] == 0 and value["all_owned_workers_drained"] is True


@pytest.mark.parametrize("change", ["early_drain", "duplicate_complete", "failure", "late_admission", "missing_seed", "wrong_active"])
def test_final_claim_does_not_override_failed_or_incomplete_worker_history(tmp_path, change):
    events = journal_fixture()
    if change == "early_drain": events.insert(2, {"event": "all_submitted_workers_drained"})
    elif change == "duplicate_complete": events.insert(3, deepcopy(events[2]))
    elif change == "failure": events[2]["event"] = "worker_failed"
    elif change == "late_admission": events.append({"event": "submitted", "seed": tool.prior.SEEDS[0], "active": 1})
    elif change == "missing_seed": del events[-4:-2]
    else: events[1]["active"] = 2
    path = tmp_path/"journal.jsonl"; save_journal(path, events)
    with pytest.raises(ValueError): tool.verify_drain_journal(path)


def test_anomaly_members_are_metadata_only_outside_restored_campaign(tmp_path, monkeypatch):
    policy, plan, current, original = leases(tmp_path)
    row = put(tmp_path/"metadata.json", {"inert": True})
    policy.update(fixed_completed_metadata={"postcondition_independent_review": row},
        whitelist_additions={"policy": "execution/lease-anomaly/policy.json",
            "quarantined_original_owner": "execution/lease-anomaly/original-owner.json",
            "reappeared_owner": "execution/lease-anomaly/reappeared-owner.json",
            "postcondition_independent_review": "execution/lease-anomaly/review.json"})
    plan["proposed_resources_not_yet_adopted"] = {"maximum_total_input_bytes": 10**6, "maximum_single_file_bytes": 10**5}
    monkeypatch.setattr(tool, "POLICY", Path(row["path"]))
    base = [{"source": row, "member": f"data/registered-{i}.npz", "category": "inert"} for i in range(33)]
    monkeypatch.setattr(tool.prior, "archive_entries", lambda *args: deepcopy(base))
    entries = tool.archive_entries(policy, plan, {}, tmp_path/"unused", {}, lambda: None)
    assert len(entries) == 37 and sum(x["member"].endswith(".npz") for x in entries) == 33
    assert all(not x["member"].startswith("data/campaign-run/.quantum-scheduler-lock") for x in entries)
    policy["whitelist_additions"]["reappeared_owner"] = "data/campaign-run/.quantum-scheduler-lock/owner.json"
    with pytest.raises(ValueError, match="cannot create a restored campaign lease"):
        tool.archive_entries(policy, plan, {}, tmp_path/"unused", {}, lambda: None)


def test_v2_multithread_environment_rejects_before_admission_or_raw_read(tmp_path, monkeypatch):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("NUMEXPR_NUM_THREADS", "2")
    monkeypatch.setattr(tool, "authorize", lambda *a: pytest.fail("admission must not run"))
    with pytest.raises(ValueError, match="one-thread"):
        tool.main_run(tmp_path/"absent")


def test_phase_guard_failure_retains_claim_and_failure_before_raw_or_analyzer(tmp_path, monkeypatch):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    inert = put(tmp_path/"inert.json", {"inert": True})
    adoption_path = Path(inert["path"])
    output = tmp_path/"evidence"
    paths = {"new_evidence": str(output), **{key: str(tmp_path/key) for key in
             ("archive", "member_inventory", "fresh_restore", "fresh_replay")}}
    policy = {"fixed_completed_metadata": {}, "frozen_v1_source": inert,
              "frozen_v1_review": inert, "prior_plan": inert}
    plan = {"expected_paths": paths, "protected_source_files": [], "fixed_inputs": {},
            "proposed_resources_not_yet_adopted": {"whole_call_CPU_seconds": 100000,
                "whole_call_wall_seconds": 10, "peak_working_memory_target_bytes": 2**40}}
    # Keep a positive work interval after the fixed10s drain reserve.
    plan["proposed_resources_not_yet_adopted"]["whole_call_wall_seconds"] = 30
    adoption = {"CPU_affinity": [0], "root_no_writer_attestation": attestation()["root_no_writer_attestation"]}
    monkeypatch.setattr(tool, "authorize", lambda path: (policy, plan, adoption, {}, {}, {}, {}, {}))
    monkeypatch.setattr(tool, "POLICY", adoption_path)
    monkeypatch.setattr(tool.os, "sched_setaffinity", lambda *args: None)
    monkeypatch.setattr(tool.os, "nice", lambda *args: None)
    def changed(*args): raise ValueError("bound anomalous lease identity changed")
    monkeypatch.setattr(tool, "guard_lease_anomaly", changed)
    monkeypatch.setattr(tool.prior, "audit_seed_rows", lambda *args: pytest.fail("no raw checkpoint admission"))
    monkeypatch.setattr(tool.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("no analyzer admission"))
    with pytest.raises(ValueError, match="lease identity changed"):
        tool.main_run(adoption_path)
    assert (output/"RUN_ONCE.json").exists()
    result = json.loads((output/"execution.json").read_text())
    assert result["complete_archive"] is False and result["replay_invocations"] == 0
    assert result["original_lease_modified"] is False and result["original_lease_absence_claimed"] is False
    with pytest.raises(FileExistsError):
        tool.main_run(adoption_path)
