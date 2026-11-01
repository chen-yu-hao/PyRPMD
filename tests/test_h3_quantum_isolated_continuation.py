"""No PES, raw cohort or sampling: exact-copy and inert process contracts."""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest

if sys.platform != "linux" or not hasattr(signal, "pthread_sigmask"):
    pytest.skip("This registered process owner is Linux-specific", allow_module_level=True)

from benchmarks.h3 import quantum_isolated_continuation as owner
from benchmarks.h3 import p256_parent_campaign as dispatch
from benchmarks.h3 import p256_complete_analysis as alarms
from benchmarks.h3 import quantum_sampling_scheduler as inspector

requires_pidfd = pytest.mark.skipif(
    not (hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")),
    reason="real owned-process checks require os.pidfd_open and signal.pidfd_send_signal",
)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return owner.identity(path)


def archive_fixture(tmp_path, *, members=None, kinds=None):
    members = members or [("data/campaign-run/state.bin", b"\x00\xff exact state\n"),
                          ("reference/protocol.json", b'{"fixed": true}\n')]
    kinds = kinds or {}
    path = tmp_path/"input.tar.gz"
    rows = []
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members:
            item = tarfile.TarInfo(name); item.size = len(data)
            if name in kinds:
                item.type = kinds[name]; item.linkname = "outside"
            archive.addfile(item, io.BytesIO(data))
            rows.append({"member": name, "source": {"path": "/old/"+name,
                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}})
    return owner.identity(path), {"member_count": len(rows), "members": rows}


def test_exact_archive_copy_uses_new_inodes_and_never_rewrites_source(tmp_path):
    archive, inventory = archive_fixture(tmp_path)
    before = owner.identity(archive["path"])
    copied = owner.extract_archive(archive, inventory, tmp_path/"fresh")
    assert len(copied) == 2 and owner.identity(archive["path"]) == before
    for row in copied:
        assert row["copy"]["sha256"] == row["archive_source"]["sha256"]
        assert Path(row["copy"]["path"]).stat().st_nlink == 1
    with pytest.raises(FileExistsError):
        owner.extract_archive(archive, inventory, tmp_path/"fresh")


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a\\b", "a/../b", "./a", ""])
def test_archive_paths_cannot_escape_or_alias(name):
    with pytest.raises(ValueError):
        owner.safe_member(name)


@pytest.mark.parametrize("fault", ["member_hash", "member_size", "extra", "duplicate", "symlink", "hardlink", "lease"])
def test_archive_whitelist_and_byte_integrity_fail_closed(tmp_path, fault):
    if fault in ("symlink", "hardlink"):
        archive, inventory = archive_fixture(tmp_path, kinds={"data/campaign-run/state.bin":
            tarfile.SYMTYPE if fault == "symlink" else tarfile.LNKTYPE})
    elif fault == "duplicate":
        archive, inventory = archive_fixture(tmp_path, members=[("a", b"1"), ("a", b"1")])
    elif fault == "lease":
        archive, inventory = archive_fixture(tmp_path,
            members=[("data/campaign-run/.quantum-scheduler-lock/owner.json", b"{}")])
    else:
        archive, inventory = archive_fixture(tmp_path)
        if fault == "member_hash": inventory["members"][0]["source"]["sha256"] = "0"*64
        if fault == "member_size": inventory["members"][0]["source"]["bytes"] += 1
        if fault == "extra": inventory["members"] = inventory["members"][:1]; inventory["member_count"] = 1
    with pytest.raises(ValueError):
        owner.extract_archive(archive, inventory, tmp_path/"fresh")


def test_hashing_checks_budget_between_blocks(tmp_path):
    path = tmp_path/"large"; path.write_bytes(b"x"*(3*1024*1024))
    calls = []
    def budget():
        calls.append(True)
        if len(calls) == 2: raise TimeoutError("inert deadline")
    with pytest.raises(TimeoutError): owner.identity(path, budget)
    assert len(calls) == 2


def test_symlink_identity_is_not_an_immutable_input(tmp_path):
    original = tmp_path/"original"; original.write_bytes(b"exact")
    link = tmp_path/"link"; link.symlink_to(original)
    with pytest.raises(ValueError): owner.identity(link)


@pytest.mark.parametrize("cpus", [[0], [0, 0], [0, 1, 2], [True, 1], [-1, 0]])
def test_fixed_slot_count_and_integer_identity(cpus):
    with pytest.raises(ValueError): owner.validate_cpus(cpus)


@pytest.mark.parametrize("fault", ["P256_missing", "bad_sum", "insufficient", "bool_bytes", "empty_basis"])
def test_space_admission_includes_other_live_growth_before_restore(fault):
    storage = {"mandatory_free_reserve_bytes": 100,
               "MC_new_working_future_archive_and_fresh_replay_allowance_bytes": 200}
    adoption = {"remaining_external_reservations": [
        {"name": "water", "remaining_bytes": 30, "basis": "fixed pending raw/archive"},
        {"name": "P256_children", "remaining_bytes": 20, "basis": "remaining branch files"}],
        "remaining_other_reserved_bytes": 50, "other_reservation_basis": "root observed full reservation"}
    free = 350
    if fault == "P256_missing": adoption["remaining_external_reservations"].pop()
    if fault == "bad_sum": adoption["remaining_other_reserved_bytes"] = 49
    if fault == "insufficient": free = 349
    if fault == "bool_bytes": adoption["remaining_external_reservations"][0]["remaining_bytes"] = True
    if fault == "empty_basis": adoption["remaining_external_reservations"][1]["basis"] = ""
    with pytest.raises(ValueError): owner.storage_admission(storage, adoption, free)


def test_global_free_reserve_is_counted_once_not_per_task():
    storage = {"mandatory_free_reserve_bytes": 100,
               "MC_new_working_future_archive_and_fresh_replay_allowance_bytes": 200}
    adoption = {"remaining_external_reservations": [
        {"name": "water", "remaining_bytes": 30, "basis": "fixed"},
        {"name": "P256_children", "remaining_bytes": 20, "basis": "fixed"}],
        "remaining_other_reserved_bytes": 50, "other_reservation_basis": "actual remaining quantities"}
    assert owner.storage_admission(storage, adoption, 350) == 350


@pytest.mark.parametrize("fault", ["session", "exit", "bool_exit", "missing_chunk"])
def test_real_archive_exit_identity_is_required(fault):
    value = {"schema": "pyrpmd.root-actual-exec-observation.v1", "exec_session_id": 46558,
             "observed_utc": "recorded", "tool_result": {"exit_code": 0, "chunk_id": "actual"}}
    if fault == "session": value["exec_session_id"] = 57942
    if fault == "exit": value["tool_result"]["exit_code"] = 137
    if fault == "bool_exit": value["tool_result"]["exit_code"] = False
    if fault == "missing_chunk": value["tool_result"].pop("chunk_id")
    with pytest.raises(ValueError): owner.validate_terminal(value)


def receipt_fixture(tmp_path):
    review = {"approved_for_fixed_isolated300000": True,
              "source_sha256": owner.identity(owner.__file__)["sha256"],
              "tests_sha256": owner.identity(owner.TESTS)["sha256"], "policy_sha256": owner.POLICY_SHA}
    gate = {"returncode": 0, "failures": 0, "errors": 0, "skipped": 0, "passed": 1,
            "source_sha256": review["source_sha256"], "tests_sha256": review["tests_sha256"]}
    return review, gate


@pytest.mark.parametrize("fault", ["unapproved", "stale_source", "stale_test", "skip", "no_tests"])
def test_rehashed_review_or_gate_must_approve_current_exact_source(tmp_path, fault):
    review, gate = receipt_fixture(tmp_path)
    if fault == "unapproved": review["approved_for_fixed_isolated300000"] = False
    if fault == "stale_source": review["source_sha256"] = "0"*64
    if fault == "stale_test": gate["tests_sha256"] = "0"*64
    if fault == "skip": gate["skipped"] = 1
    if fault == "no_tests": gate["passed"] = 0
    adoption = {"independent_review": dump(tmp_path/"review.json", review),
                "test_gate": dump(tmp_path/"gate.json", gate)}
    with pytest.raises(ValueError): owner.validate_receipts(adoption)


def complete_rows():
    return [{"seed": seed, "completed_sweeps": 305000,
             "arrays": {"samples": {"shape": [300000, 48]}, "diagnostics": {"shape": [30000, 48, 7]}},
             "stage_receipts": [{"production_sweeps": n} for n in (30000, 100000, 300000)]}
            for seed in owner.SEEDS]


@pytest.mark.parametrize("fault", ["missing_seed", "reorder", "count", "shape", "no_stage_receipt"])
def test_no_partial_or_checkpoint_only_global_completion(fault):
    rows = complete_rows()
    if fault == "missing_seed": rows.pop()
    if fault == "reorder": rows.reverse()
    if fault == "count": rows[0]["completed_sweeps"] -= 1
    if fault == "shape": rows[0]["arrays"]["samples"]["shape"][0] -= 1
    if fault == "no_stage_receipt": rows[0]["stage_receipts"].pop()
    with pytest.raises(ValueError): owner.verify_complete(rows)


def test_complete_counts_and_full_snapshot_schema_without_sampling(tmp_path):
    assert owner.verify_complete(complete_rows())["additional_sweeps_all8"] == 1600000
    compact = dump(tmp_path/"compact.json", {"global_completion": {}, "prior_snapshot": {}, "seeds": []})
    with pytest.raises(ValueError, match="not a complete"):
        inspector.load_snapshot(compact["path"], compact["sha256"])


def worker_claim_fixture(tmp_path, monkeypatch):
    output = tmp_path/"output"; output.mkdir()
    attempt = output/"attempt"; attempt.mkdir()
    adoption = dump(tmp_path/"adoption.json", {"CPU_affinity": [0, 1]})
    registration = {"schema": owner.REG_SCHEMA, "source": owner.identity(owner.__file__),
        "output": str(output), "campaign": str(output/"restored/data/campaign-run"),
        "seed_order": list(owner.SEEDS), "target_stage": 300000, "adoption": adoption}
    registration_path = output/"registration.json"; dump(registration_path, registration)
    plan = {"schema": owner.ATTEMPT_SCHEMA, "registration": owner.identity(registration_path),
        "source": owner.identity(owner.__file__), "CPU_affinity": [0, 1], "owner_PID": os.getppid(),
        "owner_process_token": "test-only-token", "seed_order": list(owner.SEEDS), "target_stage": 300000,
        "fixed_worker_count": 2, "mid_run_resize": False,
        "worker_CPU_hard_limit_seconds": 21600, "worker_wall_hard_limit_seconds": 21600,
        "work_deadline_monotonic": 1000., "whole_deadline_monotonic": 1120.}
    monkeypatch.setenv(owner.TOKEN_ENV, "test-only-token")
    bootstrap = {"installed_before_scientific_imports": True, "alarm_unblocked": True,
                 "RLIMIT_CPU": [21600, 21600], "RLIMIT_AS": [1073741824, 1073741824],
                 "owner_work_deadline": 1000.}
    return registration, registration_path, plan, attempt, owner.SEEDS[0], 0, bootstrap


@pytest.mark.parametrize("fault", ["parent", "token", "stage", "CPU", "limits", "AS_limit", "missing_bootstrap", "resize", "deadline"])
def test_private_worker_rejects_other_owner_or_resource_contract(tmp_path, monkeypatch, fault):
    args = list(worker_claim_fixture(tmp_path, monkeypatch))
    if fault == "parent": args[2]["owner_PID"] += 1
    if fault == "token": monkeypatch.delenv(owner.TOKEN_ENV)
    if fault == "stage": args[2]["target_stage"] = 1000000
    if fault == "CPU": args[5] = 2
    if fault == "limits": args[6]["RLIMIT_CPU"] = [1, 2]
    if fault == "AS_limit": args[6]["RLIMIT_AS"] = [1, 2]
    if fault == "missing_bootstrap": args[6] = None
    if fault == "resize": args[2]["mid_run_resize"] = True
    if fault == "deadline": args[2]["whole_deadline_monotonic"] += 1
    with pytest.raises(ValueError): owner.check_worker_claim(*args)


def test_exact_worker_claim_passes_without_importing_physics(tmp_path, monkeypatch):
    owner.check_worker_claim(*worker_claim_fixture(tmp_path, monkeypatch))


@pytest.mark.parametrize("mode", ["wall", "CPU"])
def test_independent_worker_kernel_caps_survive_blocked_owner_alarm(mode):
    # The isolated interpreter is inert. CPU case deliberately consumes one
    # CPU second; wall case proves the inherited blocked alarm is unblocked.
    body = (
        "import signal,time\n"
        "from benchmarks.h3.quantum_isolated_continuation import install_worker_limits\n"
        "signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM})\n"
        f"install_worker_limits(time.monotonic()+4,cpu_seconds=1,wall_seconds={'.12' if mode == 'wall' else '3'})\n"
        "while True: pass\n")
    result = subprocess.run([sys.executable, "-c", body], cwd=owner.ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=6)
    assert result.returncode == (-signal.SIGALRM if mode == "wall" else -signal.SIGKILL)


def test_inherited_smaller_CPU_ceiling_cannot_be_silently_claimed():
    body = (
        "import resource,time\n"
        "from benchmarks.h3.quantum_isolated_continuation import install_worker_limits\n"
        "resource.setrlimit(resource.RLIMIT_CPU,(1,1))\n"
        "try: install_worker_limits(time.monotonic()+2,cpu_seconds=2)\n"
        "except ValueError: pass\n"
        "else: raise SystemExit(3)\n")
    assert subprocess.run([sys.executable, "-c", body], cwd=owner.ROOT, timeout=4).returncode == 0


def test_real_address_space_cap_prevents_an_oversize_allocation():
    body = (
        "from benchmarks.h3.quantum_isolated_continuation import install_address_space_limit\n"
        "assert install_address_space_limit(128*1024**2)==[128*1024**2]*2\n"
        "try: a=bytearray(200*1024**2)\n"
        "except MemoryError: pass\n"
        "else: raise SystemExit(3)\n")
    assert subprocess.run([sys.executable, "-c", body], cwd=owner.ROOT, timeout=4).returncode == 0


class InertMonitor:
    def __init__(self, hook=lambda row: None): self.rows, self.hook, self.closed = [], hook, False
    def start(self): pass
    def event(self, row): self.rows.append(row); self.hook(row)
    def close(self): self.closed = True


def dispatch_inert(tmp_path, monkeypatch, *, popen_hook=None, event_hook=lambda row: None, guard=lambda: None):
    attempt = tmp_path/"attempt"; attempt.mkdir()
    monitor = InertMonitor(event_hook)
    original = dispatch.subprocess.Popen
    if popen_hook:
        def wrapped(*a, **k):
            process = original(*a, **k); popen_hook(process); return process
        monkeypatch.setattr(dispatch.subprocess, "Popen", wrapped)
    with owner.exclusive_owner(tmp_path/"new-root") as lease, dispatch.deferred_stop_signals():
        with alarms.work_alarm(time.monotonic()+3, {}):
            value = owner.dispatch_owned(dispatch,
                lambda seed, cpu: [sys.executable, "-c", "import time;time.sleep(.05)"],
                lambda seed: {"seed": seed}, cpus=[0, 1], deadline=time.monotonic()+2,
                attempt=attempt, lease=lease, guard=guard, monitor=monitor)
        assert lease["release"]
    return value, monitor


@requires_pidfd
def test_full_inert_cohort_preserves_order_once_and_two_worker_ceiling(tmp_path, monkeypatch):
    old = os.environ.get(owner.TOKEN_ENV)
    result, monitor = dispatch_inert(tmp_path, monkeypatch)
    assert result["launched_seeds"] == list(owner.SEEDS) and not result["errors"]
    assert result["all_workers_drained"] and set(result["results"]) == set(owner.SEEDS)
    active = peak = 0
    for row in monitor.rows:
        if row["event"] == "worker_started": active += 1; peak = max(peak, active)
        if row["event"] == "worker_reaped": active -= 1
    assert active == 0 and peak == 2 and monitor.closed
    assert os.environ.get(owner.TOKEN_ENV) == old


@requires_pidfd
def test_alarm_between_Popen_and_active_capture_drains_registered_child(tmp_path, monkeypatch):
    def pending_alarm(process): os.kill(os.getpid(), signal.SIGALRM)
    result, monitor = dispatch_inert(tmp_path, monkeypatch, popen_hook=pending_alarm)
    assert result["launched_seeds"] == [owner.SEEDS[0]]
    assert result["errors"] and result["all_workers_drained"] and monitor.closed
    assert signal.SIGALRM not in signal.pthread_sigmask(signal.SIG_BLOCK, set())


@pytest.mark.parametrize("failure", [pytest.param("signal", marks=requires_pidfd),
                                    pytest.param("journal", marks=requires_pidfd), "guard"])
def test_stop_or_event_error_never_admits_remaining_workers(tmp_path, monkeypatch, failure):
    def event(row):
        if row["event"] == "worker_started":
            if failure == "signal": os.kill(os.getpid(), signal.SIGTERM)
            if failure == "journal": raise OSError("inert journal error")
    def guard():
        if failure == "guard": raise ValueError("inert source changed")
    result, _ = dispatch_inert(tmp_path, monkeypatch, event_hook=event, guard=guard)
    assert result["errors"] and result["all_workers_drained"]
    assert result["launched_seeds"] == ([] if failure == "guard" else [owner.SEEDS[0]])


@requires_pidfd
def test_unconfirmed_drain_retains_exclusion_and_attempt(tmp_path, monkeypatch):
    real = dispatch.drain_processes
    def drain_then_fail(active):
        real(active)
        raise RuntimeError("inert confirmation failure after actual drain")
    monkeypatch.setattr(dispatch, "drain_processes", drain_then_fail)
    monkeypatch.setattr(owner.OwnedChildren, "drain", lambda *args:
                        (_ for _ in ()).throw(RuntimeError("independent owner drain unconfirmed")))
    with pytest.raises(RuntimeError, match="unconfirmed"):
        dispatch_inert(tmp_path, monkeypatch, event_hook=lambda row:
            os.kill(os.getpid(), signal.SIGTERM) if row["event"] == "worker_started" else None)
    assert (tmp_path/".new-root-continuation-lock/owner.json").is_file()
    assert (tmp_path/"attempt/events.jsonl").is_file()


@requires_pidfd
def test_alarm_during_frozen_exceptional_drain_is_independently_reaped(tmp_path, monkeypatch):
    seen = []
    def interrupted_drain(active):
        seen.extend(process.pid for process, _ in active.values())
        os.kill(os.getpid(), signal.SIGALRM)
    monkeypatch.setattr(dispatch, "drain_processes", interrupted_drain)
    def fail_event(row):
        if row["event"] == "worker_started": raise OSError("inert event failure")
    with pytest.raises(RuntimeError, match="unconfirmed"):
        dispatch_inert(tmp_path, monkeypatch, event_hook=fail_event)
    assert len(seen) == 1
    for pid in seen:
        with pytest.raises(ChildProcessError): os.waitpid(pid, os.WNOHANG)
    assert not (tmp_path/".new-root-continuation-lock").exists()


def resource_fixture(tmp_path, monkeypatch):
    policy = {"storage_revision": {"mandatory_free_reserve_bytes": 100,
              "per_active_checkpoint_temporary_buffer_allowance_bytes": 10},
              "sampling_envelope": {"planned_total_observed_RSS_limit_bytes": 10**10,
              "owner_observed_CPU_limit_seconds": 1e6, "metadata_watchdog_interval_seconds": 1,
              "worker_resource_heartbeat_stale_seconds": 10}}
    monitor = owner.ResourceWatchdog(policy, [0, 1])
    monitor.bind(tmp_path, {"sha256": "registered"})
    monitor.registry = SimpleNamespace(exited=lambda seed: False)
    monitor.owned = {1: {"pid": 1234, "cpu": 0, "first_pending": False}}
    monkeypatch.setattr(owner.shutil, "disk_usage", lambda path: SimpleNamespace(free=150))
    value = {"schema": "pyrpmd.h3.isolated300k-worker-resource.v1", "seed": 1, "PID": 1234,
        "CPU": 0, "owner_PID": os.getpid(), "owner_token": owner.PROCESS_TOKEN,
        "source_sha256": owner.identity(owner.__file__)["sha256"], "registration_sha256": "registered",
        "RLIMIT_AS": [1073741824, 1073741824], "peak_RSS_bytes": 100, "monotonic": time.monotonic()}
    dump(tmp_path/"seed-1-resource.json", value)
    return monitor, value


def test_resource_observer_uses_positive_self_reports_and_preserves_reserve(tmp_path, monkeypatch):
    monitor, _ = resource_fixture(tmp_path, monkeypatch)
    assert monitor.observe() is None and monitor.maximum_observed_RSS > 100
    monkeypatch.setattr(owner.shutil, "disk_usage", lambda path: SimpleNamespace(free=119))
    assert "free space" in monitor.observe()


@pytest.mark.parametrize("fault", ["missing", "stale", "zero", "wrong_PID", "changed_source"])
def test_unknown_live_worker_RSS_is_never_zero_compliance(tmp_path, monkeypatch, fault):
    monitor, value = resource_fixture(tmp_path, monkeypatch)
    if fault == "missing": (tmp_path/"seed-1-resource.json").unlink()
    else:
        if fault == "stale": value["monotonic"] -= 11
        if fault == "zero": value["peak_RSS_bytes"] = 0
        if fault == "wrong_PID": value["PID"] += 1
        if fault == "changed_source": value["source_sha256"] = "0"*64
        dump(tmp_path/"seed-1-resource.json", value)
    with pytest.raises(ValueError): monitor.observe()
    assert monitor.maximum_observed_RSS is None  # Unknown is not a zero-RSS observation.


def test_only_kernel_confirmed_exited_worker_can_be_excluded(tmp_path, monkeypatch):
    monitor, _ = resource_fixture(tmp_path, monkeypatch)
    (tmp_path/"seed-1-resource.json").unlink()
    monitor.registry = SimpleNamespace(exited=lambda seed: True)
    assert monitor.observe() is None and monitor.maximum_observed_RSS > 0


@requires_pidfd
def test_real_capped_worker_self_heartbeat_is_accepted_before_next_admission(tmp_path, monkeypatch):
    # The temporary child must import the checkout without pytest's sys.path or
    # a PYTHONPATH supplied by a local focused-test launcher.
    monkeypatch.delenv("PYTHONPATH", raising=False)
    policy = json.loads(owner.POLICY.read_text())
    monitor = owner.ResourceWatchdog(policy, [0, 1])
    registration = dump(tmp_path/"registration.json", {"fixture": "no physics"})
    monitor.bind(tmp_path, registration)
    script = tmp_path/"inert_resource.py"
    script.write_text(
        "import os,time,json,sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(owner.ROOT)!r})\n"
        "from benchmarks.h3.quantum_isolated_continuation import install_address_space_limit,WorkerResourceReporter,identity\n"
        "install_address_space_limit()\n"
        "with WorkerResourceReporter(Path(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),identity(Path(sys.argv[1])/'registration.json'),{'owner_process_token':os.environ['PYRPMD_ISOLATED300K_OWNER_TOKEN']}):time.sleep(.1)\n")
    attempt = tmp_path/"attempt"; attempt.mkdir()
    # Resource files and registration intentionally share the fixture root;
    # production binds the actual attempt path independently.
    with owner.exclusive_owner(tmp_path/"new-root") as lease, dispatch.deferred_stop_signals(), alarms.work_alarm(time.monotonic()+8, {}):
        result = owner.dispatch_owned(dispatch,
            lambda seed, cpu: [sys.executable, str(script), str(tmp_path), str(seed), str(cpu)],
            lambda seed: {"seed": seed}, cpus=[0, 1], deadline=time.monotonic()+6,
            attempt=attempt, lease=lease, guard=lambda: None, monitor=monitor)
    child_logs = ({path.name: path.read_text() for path in sorted(attempt.glob("seed-*.log"))}
                  if result["errors"] else {})
    assert not result["errors"] and result["all_workers_drained"], (result["errors"], child_logs)
    assert result["independent_owner_drain"]["all_captured_children_confirmed_reaped"]
    assert monitor.maximum_observed_RSS > 0 and monitor.last_observation_complete


@pytest.mark.parametrize("fault", ["missing_water", "negative", "too_large", "unknown_basis"])
def test_shared_memory_admission_keeps_external_forecasts_explicit(fault):
    policy = json.loads(owner.POLICY.read_text())
    adoption = {"external_memory_reservations": [
        {"name": "water", "bytes": 1280*1024**2, "basis": "independent cap proposal", "hard_limit": True},
        {"name": "P256_children", "bytes": 512*1024**2, "basis": "observed fixed-size worker evidence", "hard_limit": False},
        {"name": "other_diagnostics_and_runtime", "bytes": 512*1024**2, "basis": "reserved diagnostic space", "hard_limit": False}]}
    if fault == "missing_water": adoption["external_memory_reservations"].pop(0)
    if fault == "negative": adoption["external_memory_reservations"][0]["bytes"] = -1
    if fault == "too_large": adoption["external_memory_reservations"][0]["bytes"] = 8*1024**3
    if fault == "unknown_basis": adoption["external_memory_reservations"][0]["basis"] = ""
    with pytest.raises(ValueError): owner.memory_admission(policy, adoption, 4)


def test_frozen_helper_identities_and_policy_are_exact():
    for path, expected in ((owner.DISPATCH, owner.DISPATCH_SHA), (owner.ALARM, owner.ALARM_SHA),
                           (owner.INSPECTOR, owner.INSPECTOR_SHA), (owner.DRIVER, owner.DRIVER_SHA),
                           (owner.POLICY, owner.POLICY_SHA)):
        assert owner.identity(path)["sha256"] == expected
