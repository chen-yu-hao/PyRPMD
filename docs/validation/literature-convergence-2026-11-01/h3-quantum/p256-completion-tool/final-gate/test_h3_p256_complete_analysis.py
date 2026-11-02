"""Inert completion/identity/one-invocation fixtures; no PES or real analyzer."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

if sys.platform != "linux":
    pytest.skip("registered completion wrapper uses Linux process groups and affinity", allow_module_level=True)

from benchmarks.h3 import p256_complete_analysis as tool
from benchmarks.h3 import p256_child_campaign as coordinator

SEEDS = list(coordinator.SEEDS)
BRANCHES = [[b, dt] for b, dt in coordinator.BRANCHES]


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return tool.identity(path)


def staged(path, value, materialize):
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    if materialize:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(raw)
    return {"path": str(Path(path).resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def fixture_policy(tmp_path, *, materialize=False):
    data, child, attempt = tmp_path / "data", tmp_path / "child", tmp_path / "data/coordinator-attempts/fixed"
    data.mkdir(); child.mkdir(); attempt.mkdir(parents=True)
    names = {"data": data, "execution": tmp_path / "analysis-execution", "analysis_output": tmp_path / "analysis",
             "child_lock": tmp_path / "child-lease", "child_execution": child,
             "root_completion": child / "completion.json", "root_observed_exit": child / "root-exec-exit.json",
             "root_tool_observation": child / "exec-observation.json", "root_log": child / "campaign.log",
             "children_complete": data / "children-complete.json", "attempt": attempt,
             "child_registration": data / "coordinator-registration.json", "child_protocol": data / "protocol.json"}
    paths = {k: str(v) for k, v in names.items()}
    registration = put(names["child_registration"], {"test": "registration"})
    protocol = put(names["child_protocol"], {"test": "protocol"})
    auth = put(child / "authorization.json", {"test": "authorization"})
    parent = put(tmp_path / "parents-complete.json", {"test": "parents"})
    plan = put(attempt / "plan.json", {"coordinator_pid": 31, "coordinator_process_token": "fixed-owner"})
    baseline = put(tmp_path / "baseline.json", {"test": "immutable complete P128 reference"})
    runtime = {"python_executable": tool.identity(sys.executable)}
    rows, executions, events = [], [], []
    for index, seed in enumerate(SEEDS):
        branches, calls = [], []
        pid = 100 + index
        events.extend([{"event": "worker_started", "seed": seed, "pid": pid, "cpu": 6},
                       {"event": "worker_reaped", "seed": seed, "pid": pid, "returncode": 0},
                       {"event": "seed_verified", "seed": seed}])
        for batch, dt in BRANCHES:
            directory = coordinator.branch_directory(data, seed, dt, batch)
            settings = staged(directory / "settings.json", {"seed": seed, "batch": batch, "dt": dt}, materialize)
            chunks = []
            for left in range(0, 200, 8):
                p = directory / f"chunk-{left:04d}.npz"
                raw = f"INERT RAW IDENTIFIER {seed} {batch} {dt} {left}".encode()
                if materialize:
                    p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
                chunks.append({"path": str(p.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
            count = 200*2*2*round(200/dt)
            report = staged(directory / "report.json", {"complete": True, "source_unchanged": True,
                "settings_sha256": settings["sha256"], "chunks": [{"file": Path(c["path"]).name,
                "sha256": c["sha256"]} for c in chunks]}, materialize)
            branches.append({"seed": seed, "pair_batch": batch, "timestep_fs": dt, "complete": True,
                "parents": 200, "pairs_per_parent": 2, "native_steps": count, "python_steps": 0,
                "settings": settings, "report": report, "chunks": chunks})
            calls.append({"batch": batch, "timestep_fs": dt, "native_steps": count, "report": report})
        rows.append({"seed": seed, "complete": True, "branches": branches,
                     "native_steps": 9_600_000, "rate_analysis_performed": False})
        executions.append(staged(attempt / f"seed-{seed}-execution.json", {"seed": seed, "pid": pid,
            "complete": True, "failure": None, "native_steps": 9_600_000, "CPU_affinity": [6],
            "thread_environment": {name: "1" for name in coordinator.THREAD_VARIABLES},
            "rate_analysis_performed": False, "completed_branches": calls,
            "whole_worker_call_wall_seconds": 1., "whole_process_user_CPU_seconds": .5,
            "whole_process_system_CPU_seconds": .1}, materialize))
        staged(attempt / f"seed-{seed}-worker-claim.json", {"seed": seed, "pid": pid,
            "parent_pid": 31, "coordinator_process_token": "fixed-owner"}, materialize)
    events.append({"event": "all_owned_workers_drained"})
    if materialize:
        (attempt / "events.jsonl").write_text("".join(json.dumps(x)+"\n" for x in events))
    manifest = {"schema": coordinator.COMPLETE_SCHEMA, "complete_registered_cohort": True,
        "eligible_for_separate_full_cohort_analysis": True, "all_owned_workers_drained": True,
        "rate_analysis_performed": False, "seed_order": SEEDS, "launched_seeds": SEEDS,
        "branch_order": BRANCHES, "errors": [], "unstarted_seeds": [], "partial_verified_results": [],
        "automatic_resume": False, "scientific_convergence_claimed": False,
        "expected_branch_count": 32, "expected_chunk_count": 800, "expected_native_steps": 76_800_000,
        "native_steps": 76_800_000, "protocol": protocol, "registration": registration,
        "attempt_plan": plan, "authorization": auth, "parent_manifest": parent,
        "P128_baseline_sha256": baseline["sha256"], "source_files": {}, "runtime": runtime,
        "seeds": rows, "execution_receipts": executions}
    put(names["children_complete"], manifest)
    if materialize:
        (attempt / "result.json").write_bytes(names["children_complete"].read_bytes())
    names["root_log"].write_text("inert complete coordinator\n")
    put(names["root_tool_observation"], {"exit_code": 0, "output": "inert normal exit"})
    launch = {"schema": "inert-root-launch", "command": ["inert-coordinator"]}
    put(names["root_completion"], dict(launch, exit_code=0, sources_unchanged=True,
                                      log_sha256=tool.identity(names["root_log"])["sha256"]))
    observed = {"schema": "pyrpmd.p256-child.root-observed-exec-exit.v1", "exec_session_id": 30588,
        "exit_code": 0, "all_workers_drained": True, "complete_registered_child_seeds": 8,
        "tool_result": {"exit_code": 0}, "driver_sha256": "a"*64, "observed_utc": "2026-11-01T22:00:00+00:00"}
    for key, name in (("children_complete_sha256", "children_complete"), ("registration_sha256", "child_registration"),
                      ("root_completion_sha256", "root_completion"), ("actual_tool_observation_sha256", "root_tool_observation")):
        observed[key] = tool.identity(names[name])["sha256"]
    put(names["root_observed_exit"], observed)
    helper = tool.ROOT / "docs/validation/literature-convergence-2026-11-01/h3-quantum/p256-parent-stage/archive_parents.py"
    policy = {"paths": paths, "root_exit_contract": {"schema": observed["schema"], "exec_session_id": 30588,
        "driver_sha256": "a"*64}, "cohort": {"complete_schema": coordinator.COMPLETE_SCHEMA,
        "seeds": SEEDS, "branch_order": BRANCHES}, "child_launch_value": launch, "child_protocol": protocol,
        "registration": registration, "attempt_plan": plan, "child_authorization": auth,
        "parent_manifest": parent, "baseline": baseline, "source_files": {}, "scientific_runtime": runtime,
        "static_guards": [protocol, registration, plan, auth, parent, baseline],
        "pure_metadata_verifiers": {"cohort_module": tool.identity(coordinator.__file__), "journal_helper": tool.identity(helper)},
        "waiting": {"wall_seconds": 1., "poll_seconds": .01},
        "command": [sys.executable, "INERT_ANALYZER"], "working_directory": str(tmp_path),
        "analysis_resources": {"selected_environment": {}, "termination_grace_seconds": 1, "forced_reap_seconds": 1}}
    return policy


def rebind_observed(policy):
    paths = policy["paths"]
    observed = tool.read(paths["root_observed_exit"])
    for key, name in (("children_complete_sha256", "children_complete"), ("root_completion_sha256", "root_completion")):
        observed[key] = tool.identity(paths[name])["sha256"]
    put(paths["root_observed_exit"], observed)


def test_ready_is_metadata_only_before_full_hash_gate(tmp_path, monkeypatch):
    policy = fixture_policy(tmp_path)
    old = tool.stable_bytes
    def guarded(path):
        assert Path(path).name not in ("settings.json", "report.json") and not str(path).endswith(".npz")
        return old(path)
    monkeypatch.setattr(tool, "stable_bytes", guarded)
    assert tool.readiness(policy) is not None


@pytest.mark.parametrize("name", ["children_complete", "root_completion", "root_observed_exit"])
def test_missing_completion_evidence_stays_pending(tmp_path, name):
    policy = fixture_policy(tmp_path)
    Path(policy["paths"][name]).unlink()
    assert tool.readiness(policy) is None


def test_released_child_lease_is_required(tmp_path):
    policy = fixture_policy(tmp_path)
    Path(policy["paths"]["child_lock"]).mkdir()
    assert tool.readiness(policy) is None


@pytest.mark.parametrize("key,value", [("complete_registered_cohort", False), ("all_owned_workers_drained", False),
    ("seed_order", SEEDS[:4]), ("expected_chunk_count", 799), ("native_steps", True),
    ("unstarted_seeds", [SEEDS[-1]]), ("errors", ["worker failure"])])
def test_declared_partial_or_failed_manifest_is_rejected(tmp_path, key, value):
    policy = fixture_policy(tmp_path)
    path = policy["paths"]["children_complete"]; data = tool.read(path); data[key] = value
    put(path, data); rebind_observed(policy)
    with pytest.raises(ValueError, match="complete drained"):
        tool.readiness(policy)


@pytest.mark.parametrize("key,value", [("exec_session_id", 97158), ("exit_code", False),
                                      ("all_workers_drained", False)])
def test_actual_root_observation_cannot_be_substituted(tmp_path, key, value):
    policy = fixture_policy(tmp_path); path = policy["paths"]["root_observed_exit"]
    data = tool.read(path); data[key] = value; put(path, data)
    with pytest.raises(ValueError, match="root-observed"):
        tool.readiness(policy)


@pytest.mark.parametrize("exit_code,unchanged", [(2, True), (0, False), (False, True)])
def test_failed_launch_stops_before_complete_cohort(tmp_path, exit_code, unchanged):
    policy = fixture_policy(tmp_path); path = policy["paths"]["root_completion"]
    data = tool.read(path); data.update(exit_code=exit_code, sources_unchanged=unchanged); put(path, data)
    Path(policy["paths"]["children_complete"]).unlink()
    with pytest.raises(ValueError, match="root launch ended"):
        tool.readiness(policy)


def test_unfinished_json_write_remains_pending(tmp_path):
    policy = fixture_policy(tmp_path)
    Path(policy["paths"]["root_completion"]).write_text('{"exit_code":')
    assert tool.readiness(policy) is None


def test_json_payload_and_hash_share_the_same_read(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"; put(path, {"version": 1})
    old = path.read_bytes()
    def replace_after_read(_path):
        put(path, {"version": 2})
        return old
    monkeypatch.setattr(tool, "stable_bytes", replace_after_read)
    value, identity = tool.pending_json(path)
    assert value == {"version": 1}
    assert identity["sha256"] == hashlib.sha256(old).hexdigest()
    assert identity["sha256"] != tool.identity(path)["sha256"]


def metadata_dependencies(policy, monkeypatch):
    old = tool.dependency
    def load(row, name):
        if name.endswith("cohort_metadata"):
            return SimpleNamespace(verify_complete_cohort=coordinator.verify_complete_cohort,
                                   runtime_identity=lambda: policy["scientific_runtime"])
        return old(row, name)
    monkeypatch.setattr(tool, "dependency", load)


def test_declared_complete_does_not_replace_all_raw_file_hashes(tmp_path, monkeypatch):
    policy = fixture_policy(tmp_path); metadata_dependencies(policy, monkeypatch)
    metadata = tool.readiness(policy)
    with pytest.raises(FileNotFoundError):
        tool.verify_final_inputs(policy, metadata, time.monotonic()+5, {"reason": None})


def test_all_800_raw_identities_and_worker_drain_are_verified(tmp_path, monkeypatch):
    policy = fixture_policy(tmp_path, materialize=True); metadata_dependencies(policy, monkeypatch)
    result = tool.verify_final_inputs(policy, tool.readiness(policy), time.monotonic()+10, {"reason": None})
    assert result["counts"] == {"branches": 32, "chunks": 800, "native_steps": 76_800_000}
    assert result["journal"]["successful_reaps"] == result["journal"]["verified_seeds"] == 8
    assert result["all_final_raw_hashes_verified"] is True and result["partial_analysis_performed"] is False
    chunks = [row for row in result["input_files"] if row["path"].endswith(".npz")]
    assert len(chunks) == 800
    Path(chunks[-1]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity differs"):
        tool.verify_final_inputs(policy, tool.readiness(policy), time.monotonic()+10, {"reason": None})


def test_stale_root_snapshot_is_rejected_before_raw_verification(tmp_path, monkeypatch):
    policy = fixture_policy(tmp_path); metadata = tool.readiness(policy)
    path = policy["paths"]["root_observed_exit"]; value = tool.read(path)
    value["observed_utc"] = "later observation"; put(path, value)
    with pytest.raises(ValueError, match="exact receipt identity changed"):
        tool.verify_final_inputs(policy, metadata, time.monotonic()+5, {"reason": None})


def test_bounded_wait_has_no_partial_fallback(tmp_path):
    policy = fixture_policy(tmp_path); Path(policy["paths"]["root_observed_exit"]).unlink()
    timer = [0.]
    def sleep(value): timer[0] += value
    with pytest.raises(TimeoutError):
        tool.wait_ready(policy, {"reason": None}, clock=lambda: timer[0], sleep=sleep)


def test_permanent_claim_excludes_concurrent_waiters(tmp_path, monkeypatch):
    policy = fixture_policy(tmp_path)
    config = put(tmp_path / "policy.json", {"test": "policy"})
    adoption = put(tmp_path / "adoption.json", {"test": "adoption"})
    review = put(tmp_path / "review.json", {"test": "review"})
    monkeypatch.setattr(tool, "POLICY", Path(config["path"]))
    def attempt():
        try:
            tool.claim(policy, adoption["path"], {"independent_review": review})
            return "claimed"
        except FileExistsError:
            return "rejected"
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: attempt(), range(2)))
    assert sorted(outcomes) == ["claimed", "rejected"]
    assert Path(policy["paths"]["execution"], "RUN_ONCE.json").is_file()


def analyzer_fixture(tmp_path, *, fail=False, scientific_pass=False, delay=False):
    output, execution = tmp_path / "analysis", tmp_path / "execution"
    execution.mkdir()
    script = tmp_path / "inert_analyzer.py"
    report = {"schema": "pyrpmd.h3.complete-P256-dynamic-bead-comparison.v1",
        "completed_seeds": SEEDS, "expected_seeds": SEEDS, "beads": 256, "horizon_fs": 200,
        "pair_batches": [0, 1], "native_steps_both_timesteps": 76_800_000,
        "protocol_sha256": "p", "analysis_source_sha256": "INJECT", "P128_baseline_sha256": "b",
        "parent_manifest_sha256": "m", "registered_full_rate_bead_control_passed": scientific_pass}
    report_template = tmp_path / "template.json"
    script.write_text("import sys,time,json,pathlib\n" + ("time.sleep(30)\n" if delay else "") +
        ("raise SystemExit(7)\n" if fail else
         f"p=pathlib.Path({str(output)!r});p.mkdir();(p/'bead-comparison.json').write_bytes(pathlib.Path({str(report_template)!r}).read_bytes())\n"))
    report["analysis_source_sha256"] = tool.identity(script)["sha256"]; put(report_template, report)
    policy = {"paths": {"analysis_output": str(output), "child_lock": str(tmp_path / "child-lock")},
        "command": [sys.executable, str(script)], "working_directory": str(tmp_path), "cohort": {"seeds": SEEDS},
        "analysis_resources": {"selected_environment": {}, "termination_grace_seconds": 1, "forced_reap_seconds": 1},
        "child_protocol": {"sha256": "p"}, "analyzer": tool.identity(script),
        "baseline": {"sha256": "b"}, "parent_manifest": {"sha256": "m"}}
    before = {"input_files": [tool.identity(script), tool.identity(report_template)]}
    put(execution / "verified-inputs.json", before)
    return policy, execution, before


def test_scientific_gate_failure_is_retained_after_exactly_one_analysis(tmp_path):
    policy, execution, before = analyzer_fixture(tmp_path, scientific_pass=False)
    lifecycle = {}
    result = tool.execute_analyzer(policy, execution, before, time.monotonic()+5, {"reason": None}, lifecycle=lifecycle)
    assert result["complete_cohort_analysis_executed_once"] is True
    assert result["scientific_control_passed"] is False
    assert lifecycle["analyzer_invocations"] == 1 and lifecycle["owned_analyzer_drained"] is True
    with pytest.raises(ValueError, match="output appeared"):
        tool.execute_analyzer(policy, execution, before, time.monotonic()+5, {"reason": None})


def test_nonzero_analyzer_exit_is_drained_without_retry(tmp_path):
    policy, execution, before = analyzer_fixture(tmp_path, fail=True)
    lifecycle = {}
    with pytest.raises(ValueError, match="unsuccessfully"):
        tool.execute_analyzer(policy, execution, before, time.monotonic()+5, {"reason": None}, lifecycle=lifecycle)
    assert lifecycle["analyzer_invocations"] == 1 and lifecycle["owned_analyzer_drained"] is True
    assert tool.read(execution / "analyzer-exit.json")["exit_code"] == 7


def test_timeout_terminates_only_owned_inert_process_and_drains(tmp_path):
    policy, execution, before = analyzer_fixture(tmp_path, delay=True)
    lifecycle = {}
    with pytest.raises(TimeoutError):
        tool.execute_analyzer(policy, execution, before, time.monotonic()+.2, {"reason": None}, lifecycle=lifecycle)
    assert lifecycle["analyzer_invocations"] == 1 and lifecycle["owned_analyzer_drained"] is True
    assert tool.read(execution / "analyzer-exit.json")["exit_code"] < 0


def test_journal_failure_after_popen_still_records_one_invocation_and_drains(tmp_path, monkeypatch):
    policy, execution, before = analyzer_fixture(tmp_path, delay=True)
    original = tool.write_new
    def fail_started(path, value):
        if Path(path).name == "analyzer-started.json":
            raise OSError("injected marker-write failure")
        original(path, value)
    monkeypatch.setattr(tool, "write_new", fail_started)
    lifecycle = {}
    with pytest.raises(OSError, match="injected"):
        tool.execute_analyzer(policy, execution, before, time.monotonic()+5, {"reason": None}, lifecycle=lifecycle)
    assert lifecycle["analyzer_invocations"] == 1 and lifecycle["owned_analyzer_drained"] is True


def test_signal_deferral_remains_active_during_drain(monkeypatch):
    import os
    previous = signal.getsignal(signal.SIGINT)
    with tool.deferred_signals() as stop:
        os.kill(os.getpid(), signal.SIGINT)
        assert stop["reason"] == f"signal {signal.SIGINT}"
        assert signal.getsignal(signal.SIGINT) is not previous
    assert signal.getsignal(signal.SIGINT) is previous


@pytest.mark.parametrize("bad", [None, "policy", "source", "review", "resources"])
def test_exact_adoption_and_independent_review_are_required(tmp_path, monkeypatch, bad):
    policy_path = tmp_path / "policy.json"
    policy = {"analysis_resources": {"CPUs": 1, "wall": 600}}
    put(policy_path, policy)
    monkeypatch.setattr(tool, "POLICY", policy_path)
    monkeypatch.setattr(tool, "POLICY_SHA", tool.identity(policy_path)["sha256"])
    review_path = tmp_path / "review.json"
    review = {"approved_for_complete_analysis_once": bad != "review",
              "policy_sha256": tool.POLICY_SHA, "source_sha256": tool.identity(tool.__file__)["sha256"]}
    review_identity = put(review_path, review)
    adoption = {"schema": "pyrpmd.h3.p256-complete-analysis-adoption.v1", "adopted": True,
        "policy_sha256": "0"*64 if bad == "policy" else tool.POLICY_SHA,
        "source_sha256": "0"*64 if bad == "source" else tool.identity(tool.__file__)["sha256"],
        "analysis_resources": {} if bad == "resources" else policy["analysis_resources"],
        "independent_review": review_identity}
    path = tmp_path / "adoption.json"; put(path, adoption)
    if bad is None:
        assert tool.load_authorization(path) == (policy, adoption)
    else:
        with pytest.raises(ValueError):
            tool.load_authorization(path)


def test_stop_request_prevents_any_analyzer_admission(tmp_path):
    policy, execution, before = analyzer_fixture(tmp_path)
    def forbidden(*_a, **_k):
        pytest.fail("Popen was called after a stop request")
    with pytest.raises(InterruptedError):
        tool.execute_analyzer(policy, execution, before, time.monotonic()+5,
                              {"reason": "signal 15"}, popen=forbidden)
    assert not (execution / "analyzer-launch.json").exists()


def test_changed_input_after_analyzer_is_failure_even_when_it_exits_zero(tmp_path):
    policy, execution, before = analyzer_fixture(tmp_path)
    victim = tmp_path / "frozen-input.json"; put(victim, {"v": 1})
    before["input_files"].append(tool.identity(victim))
    # This is an intentionally inert process; its only extra effect is the
    # injected corruption of a fixture input after pre-invocation hashing.
    script = Path(policy["command"][1])
    script.write_text(script.read_text() + f"pathlib.Path({str(victim)!r}).write_text('changed')\n")
    policy["analyzer"] = tool.identity(script); before["input_files"][0] = tool.identity(script)
    with pytest.raises(ValueError, match="identity differs"):
        tool.execute_analyzer(policy, execution, before, time.monotonic()+5, {"reason": None})
    assert tool.read(execution / "analyzer-exit.json")["exit_code"] == 0


def test_signal_during_journal_failure_drain_keeps_permanent_claim(tmp_path, monkeypatch):
    import os
    policy, unused_execution, before = analyzer_fixture(tmp_path, delay=True)
    policy["paths"].update(execution=str(tmp_path / "once-execution"),
                           root_observed_exit=str(tmp_path / "root-exit.json"),
                           children_complete=str(tmp_path / "children-complete.json"))
    put(policy["paths"]["root_observed_exit"], {"inert": True})
    put(policy["paths"]["children_complete"], {"inert": True})
    policy["static_guards"] = []
    policy["analysis_resources"].update(CPU_affinity=[6],
        complete_verification_analysis_and_finalization_wall_seconds=30)
    config = tmp_path / "policy.json"; put(config, {"inert": True})
    auth = tmp_path / "adoption.json"; put(auth, {"inert": True})
    review = put(tmp_path / "review.json", {"inert": True})
    monkeypatch.setattr(tool, "POLICY", config)
    monkeypatch.setattr(tool, "load_authorization", lambda _p: (policy, {"independent_review": review}))
    monkeypatch.setattr(tool, "wait_ready", lambda *_a: {})
    monkeypatch.setattr(tool, "verify_final_inputs", lambda *_a: deepcopy(before))
    monkeypatch.setattr(tool.os, "sched_setaffinity", lambda *_a: None)
    original_write, original_drain = tool.write_new, tool.stop_and_drain
    previous = signal.getsignal(signal.SIGINT)
    def fail_started(path, value):
        if Path(path).name == "analyzer-started.json":
            raise OSError("injected journal failure")
        original_write(path, value)
    def interrupt_while_draining(process, *args):
        assert signal.getsignal(signal.SIGINT) is not previous
        os.kill(os.getpid(), signal.SIGINT)
        return original_drain(process, *args)
    monkeypatch.setattr(tool, "write_new", fail_started)
    monkeypatch.setattr(tool, "stop_and_drain", interrupt_while_draining)
    with pytest.raises(OSError, match="journal failure"):
        tool.run(auth)
    execution = Path(policy["paths"]["execution"])
    assert (execution / "RUN_ONCE.json").is_file()
    report = tool.read(execution / "execution.json")
    assert report["analyzer_invocations"] == 1 and report["owned_analyzer_drained"] is True
    assert report["permanent_attempt_claim_retained"] is True
    assert signal.getsignal(signal.SIGINT) is previous
