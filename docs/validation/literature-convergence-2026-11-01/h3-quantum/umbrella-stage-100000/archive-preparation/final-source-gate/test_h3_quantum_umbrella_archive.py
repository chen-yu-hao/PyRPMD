"""Inert archive/provenance/replay guards; no active cohort or PES is read."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest

if sys.platform != "linux":
    pytest.skip("registered local archive uses Linux ownership/resource guards", allow_module_level=True)

from benchmarks.h3 import quantum_umbrella_archive as tool


def put(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+"\n")
    return tool.record(path)


def metadata():
    # Independent inert routes: never inspect a real final cohort, and remain
    # valid when this test is collected from a different isolated checkout.
    plan = {"expected_paths": {"campaign": "/inert/campaign", "analysis": "/inert/analysis",
        "global_completion": "/inert/campaign/stage-100000.json",
        "scheduler_attempt": "/inert/campaign/scheduler-attempts/fixed"},
        "fixed_inputs": {"protocol": {"path": "/inert/protocol.json", "sha256": "fixed-protocol"},
            "prior_snapshot": {"sha256": "fixed-prior"}, "precision_continuation_decision": {"sha256": "fixed-decision"}},
        "protected_source_files": [{"path": str(path), "sha256": "frozen-"+path.name} for path in
            (tool.ROOT/"benchmarks/h3/quantum_sampling_scheduler.py", tool.ROOT/"benchmarks/h3/quantum_sampling_analyze.py",
             tool.ROOT/"benchmarks/h3/mixing_analysis.py", tool.WORK/"run_complete_umbrella_100000_analysis.py")]}
    paths, fixed = plan["expected_paths"], plan["fixed_inputs"]
    source_map = {row["path"]: row["sha256"] for row in plan["protected_source_files"]}
    completion = {"schema": "pyrpmd.h3.quantum-umbrella-scheduler-completion.v1", "production": 100000,
        "requested_seeds": tool.SEEDS, "completed_seeds": tool.SEEDS,
        "all_submitted_workers_drained": True, "prior_prefix_and_state_files_unchanged": True,
        "attempt_directory": paths["scheduler_attempt"], "analysis_recomputed": False,
        "protocol_sha256": fixed["protocol"]["sha256"], "prior_snapshot_sha256": fixed["prior_snapshot"]["sha256"],
        "decision_sha256": fixed["precision_continuation_decision"]["sha256"],
        "scheduler_source_sha256": source_map[str(tool.ROOT/"benchmarks/h3/quantum_sampling_scheduler.py")]}
    adoption = {"global_completion": {"sha256": "fixture-complete-hash"}}
    execution = {"schema": "pyrpmd.h3.complete-umbrella-analysis-execution.v1", "returncode": 0,
        "stage": 100000, "independent_seeds": tool.SEEDS, "complete_cohort_required": True,
        "available_only_used": False, "sources_unchanged": True, "completion_unchanged": True,
        "all_submitted_workers_drained_and_lock_released_required": True, "no_refinement_launched": True,
        "completion_manifest": paths["global_completion"], "completion_sha256": "fixture-complete-hash",
        "source_sha256": {fixed["protocol"]["path"]: fixed["protocol"]["sha256"],
            **{str(path): source_map[str(path)] for path in
               (tool.ROOT/"benchmarks/h3/quantum_sampling_analyze.py", tool.ROOT/"benchmarks/h3/quantum_sampling_scheduler.py",
                tool.WORK/"run_complete_umbrella_100000_analysis.py")}},
        "command": [str(tool.ROOT/".venv/bin/python"), str(tool.ROOT/"benchmarks/h3/quantum_sampling_analyze.py"),
            "--campaign", paths["campaign"], "--protocol", fixed["protocol"]["path"],
            "--output", paths["analysis"], "--stage", "100000"]}
    comparison = {"schema": "pyrpmd.h3.quantum-ladder-comparison.v1", "production_sweeps_per_ladder": 100000,
        "expected_independent_seeds": tool.SEEDS, "completed_independent_seeds": tool.SEEDS,
        "protocol_sha256": fixed["protocol"]["sha256"],
        "source_sha256": {name: source_map[str(tool.ROOT/name)] for name in
                          ("benchmarks/h3/quantum_sampling_analyze.py", "benchmarks/h3/mixing_analysis.py")},
        "criteria": {"complete_registered_cohort": True, "statistical_precision": False},
        "registered_primary_qtst_gates_passed": False}
    return plan, adoption, completion, execution, comparison


def test_complete_metadata_accepts_preserved_scientific_failure_without_continuation():
    values = metadata()
    tool.validate_complete_metadata(*values)
    assert values[-1]["criteria"]["statistical_precision"] is False


@pytest.mark.parametrize("fault", ["seed", "stage", "false_exit", "source", "partial", "completion_hash",
                                  "execution_source_digest", "comparison_source_digest"])
def test_incomplete_or_mismatched_execution_never_admits_archival(fault):
    plan, adoption, completion, execution, comparison = metadata()
    if fault == "seed":
        completion["completed_seeds"] = tool.SEEDS[:-1]
    elif fault == "stage":
        completion["production"] = 300000
    elif fault == "false_exit":
        execution["returncode"] = False
    elif fault == "source":
        execution["sources_unchanged"] = False
    elif fault == "partial":
        execution["available_only_used"] = True
    elif fault == "execution_source_digest":
        execution["source_sha256"][plan["fixed_inputs"]["protocol"]["path"]] = "stale-but-rehashed-source"
    elif fault == "comparison_source_digest":
        comparison["source_sha256"]["benchmarks/h3/mixing_analysis.py"] = "stale-but-rehashed-source"
    else:
        execution["completion_sha256"] = "foreign"
    with pytest.raises(ValueError):
        tool.validate_complete_metadata(plan, adoption, completion, execution, comparison)


@pytest.mark.parametrize("fault", [None, "session", "exit", "drain", "hash"])
def test_root_exit_requires_actual_session_zero_drain_and_bound_observation(tmp_path, fault):
    observation = put(tmp_path/"tool-observation.json", {"inert_tool_result": "not a real session"})
    row = {"exec_session_id": 57942, "exit_code": 0, "all_owned_workers_drained": True,
           "observed_utc": "fixture timestamp", "actual_tool_observation": observation}
    if fault == "session": row["exec_session_id"] = 1094
    if fault == "exit": row["exit_code"] = False
    if fault == "drain": row["all_owned_workers_drained"] = False
    if fault == "hash": observation["sha256"] = "0"*64
    if fault is None:
        tool.root_exit(row, 57942)
    else:
        with pytest.raises(ValueError): tool.root_exit(row, 57942)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "safe/../../outside", "a\\b", "a//b"])
def test_archive_member_routes_reject_escape_and_ambiguous_paths(name):
    with pytest.raises(ValueError, match="unsafe archive member"):
        tool.member_path(name)


def test_original_npz_bytes_and_rng_state_survive_pack_extract_without_rewriting(tmp_path):
    original = tmp_path/"checkpoint.npz"
    np.savez_compressed(original, R=np.arange(18., dtype=np.float64).reshape(2, 3, 3),
                        rng=np.array('{"fixture_state":123}'), completed=np.array(105000),
                        samples=np.arange(12., dtype=np.float64).reshape(4, 3))
    expected = tool.record(original)
    entries = [{"source": expected, "member": "data/seed-fixture/checkpoint.npz", "category": "inert_fixture"}]
    archive, restore = tmp_path/"fixture.tar.gz.partial", tmp_path/"fresh-restore"
    tool.pack_and_extract(entries, archive, restore, lambda: None)
    copy = restore/entries[0]["member"]
    assert copy.read_bytes() == original.read_bytes()
    with np.load(copy, allow_pickle=False) as saved:
        assert saved["rng"].item() == '{"fixture_state":123}' and int(saved["completed"]) == 105000
    with tarfile.open(archive, "r:gz") as saved:
        assert len(saved.getmembers()) == 1 and saved.extractfile(entries[0]["member"]).read() == original.read_bytes()
    with pytest.raises(ValueError, match="never overwrite"):
        tool.pack_and_extract(entries, archive, restore, lambda: None)


def test_changed_source_before_packing_fails_and_preserves_partial_attempt(tmp_path):
    original = tmp_path/"raw.bin"; original.write_bytes(b"original")
    entries = [{"source": tool.record(original), "member": "data/raw.bin", "category": "inert"}]
    original.write_bytes(b"changed!")
    archive, restore = tmp_path/"fixture.partial", tmp_path/"fresh"
    with pytest.raises(ValueError, match="bound input changed"):
        tool.pack_and_extract(entries, archive, restore, lambda: None)
    assert archive.exists() and restore.exists()


def replay_fixture(tmp_path):
    original, replay = tmp_path/"original", tmp_path/"replayed"
    original.mkdir(); replay.mkdir()
    for directory, clock in ((original, "original time"), (replay, "replay time")):
        put(directory/"campaign-comparison.json", {"created_utc": clock, "elapsed_seconds": len(clock),
            "criteria": {"statistical_precision": False}, "result": [1., 2.]})
        for seed in tool.SEEDS:
            put(directory/f"seed-{seed}.json", {"seed": seed, "receipt": {"elapsed_seconds_this_call": 3.}, "estimate": 4.})
        for name in ["pooled-wham.npz", *[f"seed-{seed}-wham.npz" for seed in tool.SEEDS]]:
            np.savez_compressed(directory/name, density=np.array([1., 2., 3.]), occupied=np.array([True, False, True]))
    return original, replay


def test_replay_ignores_only_declared_report_clock_fields_and_preserves_failed_gate(tmp_path):
    original, replay = replay_fixture(tmp_path)
    rows = tool.compare_replay(original, replay)
    assert len(rows) == 18
    assert rows[0]["ignored_top_level_fields"] == ["created_utc", "elapsed_seconds"]
    assert all(row.get("ignored_top_level_fields", []) == [] for row in rows[1:])


@pytest.mark.parametrize("fault", ["gate", "nested_clock", "one_ulp"])
def test_replay_refuses_any_scientific_or_state_change_without_tolerance_widening(tmp_path, fault):
    original, replay = replay_fixture(tmp_path)
    if fault == "gate":
        path = replay/"campaign-comparison.json"; value = json.loads(path.read_text())
        value["criteria"]["statistical_precision"] = True; put(path, value)
    elif fault == "nested_clock":
        path = replay/f"seed-{tool.SEEDS[0]}.json"; value = json.loads(path.read_text())
        value["receipt"]["elapsed_seconds_this_call"] = 4.; put(path, value)
    else:
        np.savez_compressed(replay/"pooled-wham.npz", density=np.array([np.nextafter(1., 2.), 2., 3.]),
                            occupied=np.array([True, False, True]))
    with pytest.raises(ValueError, match="replay differs"):
        tool.compare_replay(original, replay)


def test_all8_checkpoint_inspections_must_equal_global_snapshot_and_keep_order():
    expected = [{"seed": seed, "completed_sweeps": 105000,
        "arrays": {"samples": {"shape": [100000, 48]}, "diagnostics": {"shape": [10000, 48, 7]}},
        "rng_hash": f"seed-{seed}-exact-RNG"} for seed in tool.SEEDS]
    prior = {"seeds": [{"seed": seed} for seed in tool.SEEDS], "protocol": {"sha256": "protocol"},
             "source_sha256": {"benchmarks/h3/quantum_sampling_campaign.py": "frozen driver"}}
    calls = []
    def inspector(campaign, old, stage, protocol, protocol_sha, driver_sha):
        assert stage == 100000 and protocol_sha == "protocol" and driver_sha == "frozen driver"
        calls.append(old["seed"])
        return deepcopy(expected[tool.SEEDS.index(old["seed"])])
    complete = {"snapshot": {"seeds": deepcopy(expected)}}
    assert tool.audit_seed_rows("inert", prior, complete, {}, inspector) == expected
    assert calls == tool.SEEDS
    complete["snapshot"]["seeds"][-1]["rng_hash"] = "changed"
    with pytest.raises(ValueError, match="checkpoint/state/RNG/prefix"):
        tool.audit_seed_rows("inert", prior, complete, {}, inspector)


def test_archive_cpu_budget_refuses_work_without_launching_any_process():
    limits = {"whole_call_wall_seconds": 900, "whole_call_CPU_seconds": 0,
              "peak_working_memory_target_bytes": 2**40}
    budget = tool.Budget(limits, {})
    with pytest.raises(ValueError, match="CPU budget"):
        budget.check()


def test_multithread_environment_is_rejected_before_actual_admission_or_raw_read(tmp_path, monkeypatch):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    def forbidden(*args):
        pytest.fail("real admission or raw-data path must not be entered")
    monkeypatch.setattr(tool, "authorize", forbidden)
    with pytest.raises(ValueError, match="one-thread environment"):
        tool.main_run(tmp_path/"absent-root-adoption.json")
