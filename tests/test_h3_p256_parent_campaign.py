"""Small analytic samplers and inert processes; no BKMP2 or real dynamics."""
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

if sys.platform != "linux" or not hasattr(os, "sched_getaffinity"):
    pytest.skip("The fixed historical replay and process controls require Linux", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.h3 import p256_parent_campaign as campaign
from benchmarks.h3.surface_partition import run
from pyrpmd.checkpoint import atomic_savez
from pyrpmd.constants import KB_AU

FLAGS = {"minimum_BAR_overlap": 0., "minimum_energy_ESS": 0.,
         "maximum_absolute_half_log_difference": 100.}
REPLAY_ARRAYS = ("domain_sha256", "radius", "ratio", "eta_R", "eta_r", "completed",
                 "delta_energy", "diagnostics", "accepted", "proposed", "swap_attempts",
                 "swap_accepts", "lineage", "last_extreme", "endpoint_traversals", "rng")


class AnalyticBond:
    partition_identity = {"test_only": "P256 parent audit analytic bond", "stiffness": .3}

    def evaluate(self, q):
        difference = q[:, 0] - q[:, 1]
        return SimpleNamespace(energy=.15 * np.sum(difference * difference, axis=1),
                               status=np.zeros(len(q), dtype=int))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def sampled_pair(tmp_path):
    options = dict(surface="ts", beads=4, temperature=1 / (2 * KB_AU), mass=2.,
                   seed=91832, warmup=3, production=16, lambdas=[0., .25, 1.],
                   minimum=0., maximum=4., radial_order=16, angular_order=8,
                   domain_samples=16, checkpoint_interval=6, sample_interval=1,
                   diagnostic_interval=2, phase="production")
    old, new = tmp_path / "old", tmp_path / "replay"
    run(old, potential=AnalyticBond(), snapshot_interval=0, **options)
    run(new, potential=AnalyticBond(), snapshot_interval=4, **options)
    settings = campaign.read_json(new / "settings.json")
    old_files = {name: campaign.identity(old / name) for name in
                 ("checkpoint.npz", "settings.json", "report.json", "reference.json",
                  "reference-domain.json", "progress.json")}
    row = {"seed": options["seed"], "cohort_role": "historical_deterministic_replay",
           "original_files": old_files, "expected_settings": settings,
           "expected_settings_fingerprint": campaign.settings_fingerprint(settings),
           "exact_original_payloads": list(REPLAY_ARRAYS),
           "run_keyword_arguments": dict(options, snapshot_interval=4, resume=False, stop_after=None)}
    return old, new, row


def test_fixed_design_is_eight_predeclared_chains_without_loading_PES():
    design = campaign.load_design()
    assert tuple(design["independent_P256_seeds"]) == campaign.SEEDS
    assert design["parent_generation"]["total_snapshots"] == 1600
    assert [r["cohort_role"] for r in design["parent_generation"]["cohorts"]] == [
        "historical_deterministic_replay"] * 4 + ["new_independent_chain"] * 4


def test_changed_design_is_rejected_before_any_sampling(tmp_path):
    changed = json.loads(campaign.DESIGN.read_text())
    changed["independent_P256_seeds"] = changed["independent_P256_seeds"][:4]
    path = tmp_path / "changed.json"; write_json(path, changed)
    with pytest.raises(ValueError, match="design changed"):
        campaign.load_design(path)


def test_exact_observer_replay_binds_all_old_payloads_and_scientific_report(sampled_pair):
    old, new, row = sampled_pair
    before = campaign.identity(new / "checkpoint.npz")
    result = campaign.verify_chain(new, row, FLAGS)
    assert result["complete"] and result["identity_and_geometry_verified"]
    assert result["historical_replay"]["bitwise_verified_array_count"] == 16
    assert set(result["historical_replay"]["bitwise_verified_arrays"]) == set(REPLAY_ARRAYS)
    assert result["parents"] == 4 and result["beads"] == 4
    assert campaign.identity(new / "checkpoint.npz") == before


def test_new_independent_chain_does_not_invent_a_historical_identity(sampled_pair):
    _, new, row = sampled_pair
    row = dict(row, cohort_role="new_independent_chain", original_files=None)
    result = campaign.verify_chain(new, row, FLAGS)
    assert result["historical_replay"] is None and result["sampling_flags_passed"]


@pytest.mark.parametrize("name", REPLAY_ARRAYS)
def test_every_original_payload_is_bound_even_after_valid_reenveloping(sampled_pair, name):
    _, new, row = sampled_pair
    arrays, _, _ = campaign.inspect_payload(new / "checkpoint.npz")
    value = arrays[name].copy()
    if value.dtype.kind in "US":
        text = str(value.item())
        value = np.asarray(("0" if text[0] != "0" else "1") + text[1:])
    elif value.dtype.kind in "iu":
        value.flat[0] += 1
    else:
        value.flat[0] += 1e-12
    arrays[name] = value
    atomic_savez(new / "checkpoint.npz", kind="h3.surface-partition", **arrays)
    report = campaign.read_json(new / "report.json")
    report["checkpoint_sha256"] = campaign.sha(new / "checkpoint.npz")
    write_json(new / "report.json", report)
    with pytest.raises(ValueError):
        campaign.verify_chain(new, row, FLAGS)


def test_signed_zero_and_dtype_are_not_silently_treated_as_exact_replay():
    a, b = np.array([0.]), np.array([-0.])
    assert np.array_equal(a, b) and not campaign.exact_array_equal(a, b)
    assert not campaign.exact_array_equal(np.array([1], dtype=np.int64), np.array([1], dtype=np.int32))


def test_changed_domain_serialization_is_rejected_even_if_semantics_match(sampled_pair):
    _, new, row = sampled_pair
    path = new / "reference-domain.json"
    path.write_text(path.read_text() + " ")
    arrays, _, _ = campaign.inspect_payload(new / "checkpoint.npz")
    arrays["domain_sha256"] = campaign.sha(path)
    atomic_savez(new / "checkpoint.npz", kind="h3.surface-partition", **arrays)
    report = campaign.read_json(new / "report.json")
    report["checkpoint_sha256"] = campaign.sha(new / "checkpoint.npz")
    write_json(new / "report.json", report)
    with pytest.raises(ValueError):
        campaign.verify_chain(new, row, FLAGS)


def test_completed_resume_reuses_original_bytes_without_rerunning_producer(sampled_pair):
    _, new, row = sampled_pair
    before = {name: campaign.identity(new / name) for name in
              ("checkpoint.npz", "report.json", "settings.json", "reference-domain.json")}
    result, invoked = campaign.produce_one(new, row, FLAGS, None, resume=True,
        sampler=lambda *a, **k: pytest.fail("completed parent was rewritten"))
    assert not invoked and result["complete"]
    assert before == {name: campaign.identity(new / name) for name in before}
    with pytest.raises(ValueError, match="explicit reviewed resume"):
        campaign.produce_one(new, row, FLAGS, None)


def test_interrupted_observed_replay_resumes_to_exact_original_history(sampled_pair, tmp_path):
    _, _, row = sampled_pair
    path = tmp_path / "partial"
    options = dict(row["run_keyword_arguments"])
    options.pop("resume"); options.pop("stop_after")
    run(path, potential=AnalyticBond(), stop_after=10, **options)
    calls = []
    def synthetic(output, library, **kwargs):
        calls.append(kwargs["resume"])
        return run(output, potential=AnalyticBond(), **kwargs)
    result, invoked = campaign.produce_one(path, row, FLAGS, None, resume=True, sampler=synthetic)
    assert invoked and calls == [True]
    assert result["historical_replay"]["bitwise_verified_array_count"] == 16


def test_failed_sampling_flag_is_retained_as_a_complete_ineligible_chain(sampled_pair):
    _, new, row = sampled_pair
    result = campaign.verify_chain(new, row, dict(FLAGS, minimum_energy_ESS=1e20))
    assert result["complete"] and not result["sampling_flags_passed"]


@pytest.mark.parametrize("name", ["../escape", "/absolute", "path\\escape"])
def test_registered_output_cannot_escape_its_root(tmp_path, name):
    with pytest.raises(ValueError):
        campaign.child_path(tmp_path, name)


def fake_commands(tmp_path, *, fail_seed=0, delay=.03):
    script = tmp_path / "inert_worker.py"
    script.write_text(
        "import json,sys,time,os\nfrom pathlib import Path\n"
        "seed=int(sys.argv[1]); output=Path(sys.argv[2]); fail=int(sys.argv[3])\n"
        "time.sleep(float(sys.argv[4]))\n"
        "if seed==fail: raise SystemExit(3)\n"
        "(output/f'{seed}.json').write_text(json.dumps({'seed':seed,'pid':os.getpid()}))\n")
    def command(seed, cpu):
        return [sys.executable, str(script), str(seed), str(tmp_path), str(fail_seed), str(delay)]
    def verify(seed):
        return campaign.read_json(tmp_path / f"{seed}.json")
    return command, verify


def test_scheduler_fixed_full_cohort_never_exceeds_two_workers(tmp_path):
    command, verify = fake_commands(tmp_path)
    attempt = tmp_path / "attempt"; attempt.mkdir()
    events = []
    with campaign.campaign_lock(tmp_path / "campaign") as lease:
        result = campaign.dispatch_jobs(tuple(range(1, 9)), command, verify, cpus=[0, 1],
            deadline=time.monotonic() + 5, attempt=attempt, lease=lease, event_hook=events.append)
        assert lease["release"]
    assert result["launched_seeds"] == list(range(1, 9))
    assert set(result["results"]) == set(range(1, 9)) and not result["errors"]
    active, maximum = 0, 0
    for row in events:
        if row["event"] == "worker_started":
            active += 1; maximum = max(maximum, active)
        elif row["event"] == "worker_reaped":
            active -= 1
    assert maximum == 2 and active == 0


def test_worker_failure_stops_admissions_and_drains_all_owned_processes(tmp_path):
    command, verify = fake_commands(tmp_path, fail_seed=1, delay=.1)
    attempt = tmp_path / "attempt"; attempt.mkdir()
    with campaign.campaign_lock(tmp_path / "campaign") as lease:
        result = campaign.dispatch_jobs(tuple(range(1, 9)), command, verify, cpus=[0, 1],
            deadline=time.monotonic() + 5, attempt=attempt, lease=lease)
    assert result["errors"] and result["all_workers_drained"]
    assert result["launched_seeds"] == [1, 2]
    assert result["unstarted_seeds"] == list(range(3, 9))


def test_wall_deadline_preserves_failure_without_launching_remaining_seeds(tmp_path):
    command, verify = fake_commands(tmp_path, delay=5.)
    attempt = tmp_path / "attempt"; attempt.mkdir()
    with campaign.campaign_lock(tmp_path / "campaign") as lease:
        result = campaign.dispatch_jobs(tuple(range(1, 9)), command, verify, cpus=[0, 1],
            deadline=time.monotonic() + .12, attempt=attempt, lease=lease)
    assert "whole campaign wall deadline" in result["errors"]
    assert result["all_workers_drained"] and result["launched_seeds"] == [1, 2]


def test_signal_after_first_admission_does_not_launch_second_worker(tmp_path):
    command, verify = fake_commands(tmp_path, delay=5.)
    attempt = tmp_path / "attempt"; attempt.mkdir()
    def stop_on_first(row):
        if row["event"] == "worker_started":
            os.kill(os.getpid(), signal.SIGTERM)
    with campaign.campaign_lock(tmp_path / "campaign") as lease:
        result = campaign.dispatch_jobs(tuple(range(1, 9)), command, verify, cpus=[0, 1],
            deadline=time.monotonic() + 5, attempt=attempt, lease=lease, event_hook=stop_on_first)
    assert result["launched_seeds"] == [1]
    assert result["unstarted_seeds"] == list(range(2, 9))
    assert result["all_workers_drained"] and f"signal {signal.SIGTERM}" in result["errors"]


def test_guard_crossing_deadline_admits_no_worker(tmp_path):
    command, verify = fake_commands(tmp_path)
    attempt = tmp_path / "attempt"; attempt.mkdir()
    with campaign.campaign_lock(tmp_path / "campaign") as lease:
        result = campaign.dispatch_jobs(tuple(range(1, 9)), command, verify, cpus=[0, 1],
            deadline=time.monotonic() + .01, attempt=attempt, lease=lease, guard=lambda: time.sleep(.03))
    assert result["launched_seeds"] == [] and result["unstarted_seeds"] == list(range(1, 9))
    assert result["all_workers_drained"] and "whole campaign wall deadline" in result["errors"]


def test_stop_during_command_construction_keeps_seed_unstarted(tmp_path):
    command, verify = fake_commands(tmp_path)
    attempt = tmp_path / "attempt"; attempt.mkdir()
    def interrupted_command(seed, cpu):
        result = command(seed, cpu)
        os.kill(os.getpid(), signal.SIGINT)
        return result
    with campaign.campaign_lock(tmp_path / "campaign") as lease:
        result = campaign.dispatch_jobs(tuple(range(1, 9)), interrupted_command, verify, cpus=[0, 1],
            deadline=time.monotonic() + 5, attempt=attempt, lease=lease)
    assert result["launched_seeds"] == [] and result["unstarted_seeds"] == list(range(1, 9))
    assert result["all_workers_drained"] and f"signal {signal.SIGINT}" in result["errors"]


def test_failed_command_construction_keeps_complete_unstarted_identity(tmp_path):
    _, verify = fake_commands(tmp_path)
    attempt = tmp_path / "attempt"; attempt.mkdir()
    def bad_command(seed, cpu):
        raise OSError("injected command construction error")
    with campaign.campaign_lock(tmp_path / "campaign") as lease:
        result = campaign.dispatch_jobs(tuple(range(1, 9)), bad_command, verify, cpus=[0, 1],
            deadline=time.monotonic() + 5, attempt=attempt, lease=lease)
    assert result["launched_seeds"] == [] and result["unstarted_seeds"] == list(range(1, 9))
    assert result["all_workers_drained"] and any("command construction error" in e for e in result["errors"])


def test_exited_failed_worker_blocks_the_next_free_slot_admission(tmp_path):
    command, verify = fake_commands(tmp_path, fail_seed=1, delay=.01)
    attempt = tmp_path / "attempt"; attempt.mkdir()
    observed = []
    def wait_for_first_failure(row):
        if row["event"] == "worker_started" and row["seed"] == 1:
            # Observe the known inert child exit without reaping it. The
            # coordinator still owns and must drain/reap this exact process.
            result = os.waitid(os.P_PID, row["pid"], os.WEXITED | os.WNOWAIT)
            assert result.si_status == 3
            observed.append(row["pid"])
    with campaign.campaign_lock(tmp_path / "campaign") as lease:
        result = campaign.dispatch_jobs(tuple(range(1, 9)), command, verify, cpus=[0, 1],
            deadline=time.monotonic() + 5, attempt=attempt, lease=lease, event_hook=wait_for_first_failure)
    assert len(observed) == 1 and result["launched_seeds"] == [1]
    assert result["unstarted_seeds"] == list(range(2, 9))
    assert result["all_workers_drained"] and any("worker exited 3 before next admission" in e for e in result["errors"])
    with pytest.raises(ChildProcessError):
        os.waitpid(observed[0], os.WNOHANG)


def test_journal_failure_and_repeated_SIGINT_keep_exclusion_until_drained(tmp_path, monkeypatch):
    command, verify = fake_commands(tmp_path, delay=5.)
    attempt = tmp_path / "attempt"; attempt.mkdir()
    seen = []
    def journal(row):
        if row["event"] == "worker_started":
            seen.append(row["pid"])
            if len(seen) == 2:
                raise OSError("injected journal failure")
    original = campaign.drain_processes
    lockpath = tmp_path / ".campaign-p256-parent-lock"
    def interrupted_drain(active):
        assert lockpath.is_dir()
        os.kill(os.getpid(), signal.SIGINT)
        os.kill(os.getpid(), signal.SIGINT)
        assert lockpath.is_dir()
        original(active)
        assert lockpath.is_dir()
    monkeypatch.setattr(campaign, "drain_processes", interrupted_drain)
    with campaign.campaign_lock(tmp_path / "campaign") as lease:
        result = campaign.dispatch_jobs(tuple(range(1, 9)), command, verify, cpus=[0, 1],
            deadline=time.monotonic() + 5, attempt=attempt, lease=lease, event_hook=journal)
    assert len(seen) == 2 and not lockpath.exists()
    assert result["all_workers_drained"] and any("journal failure" in x for x in result["errors"])
    for pid in seen:
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)


def test_unconfirmed_drain_keeps_lease_even_after_coordinator_error(tmp_path, monkeypatch):
    command, verify = fake_commands(tmp_path, delay=5.)
    attempt = tmp_path / "attempt"; attempt.mkdir()
    original = campaign.drain_processes
    def unconfirmed(active):
        original(active)  # Test must not leave its inert processes alive.
        raise KeyboardInterrupt("injected uncertain confirmation")
    monkeypatch.setattr(campaign, "drain_processes", unconfirmed)
    with pytest.raises(RuntimeError, match="lease retained"):
        with campaign.campaign_lock(tmp_path / "campaign") as lease:
            campaign.dispatch_jobs((1, 2), command, verify, cpus=[0, 1],
                deadline=time.monotonic() + .1, attempt=attempt, lease=lease)
    assert (tmp_path / ".campaign-p256-parent-lock/owner.json").exists()


def test_concurrent_coordinator_and_stale_lock_are_rejected(tmp_path):
    with campaign.campaign_lock(tmp_path / "campaign"):
        with pytest.raises(FileExistsError):
            with campaign.campaign_lock(tmp_path / "campaign"):
                pytest.fail("second coordinator entered")


@pytest.mark.parametrize("completed,sampling_pass,published", [(4, True, False), (8, False, False), (8, True, True)])
def test_global_manifest_requires_all_eight_and_retains_sampling_failure(tmp_path, monkeypatch, completed, sampling_pass, published):
    output = tmp_path / "parents"
    rows = [{"seed": seed, "relative_directory": f"seed-{seed}"} for seed in campaign.SEEDS]
    design = {"parent_generation": {"cohorts": rows, "sampling_flags": FLAGS}}
    monkeypatch.setattr(campaign, "preflight", lambda *a, **k: (design, output, {}))
    monkeypatch.setattr(campaign, "load_authorization", lambda *a: {"whole_campaign_wall_seconds": 7200,
        "workers": 2, "cpu_affinity": [0, 1]})
    monkeypatch.setattr(campaign, "verify_chain", lambda path, row, flags: {
        "seed": row["seed"], "complete": True, "sampling_flags_passed": sampling_pass})
    def dispatch(*args, **kwargs):
        assert kwargs["deadline"] == campaign.ENTRY_MONOTONIC + 7200 - campaign.FINALIZATION_RESERVE_SECONDS
        return {"results": {seed: {} for seed in campaign.SEEDS[:completed]}, "errors": [],
                "launched_seeds": list(campaign.SEEDS[:completed]), "unstarted_seeds": list(campaign.SEEDS[completed:]),
                "all_workers_drained": True}
    monkeypatch.setattr(campaign, "dispatch_jobs", dispatch)
    authorization = tmp_path / "authorization.json"; write_json(authorization, {})
    result = campaign.run_campaign(authorization)
    assert result["eligible_for_separately_authorized_children"] is published
    assert (output / "parents-complete.json").exists() is published
    assert not result["dynamic_bead_convergence_demonstrated"]
    if published:
        before = campaign.sha(output / "parents-complete.json")
        with pytest.raises(ValueError, match="already exists"):
            campaign.run_campaign(authorization, resume=True)
        assert campaign.sha(output / "parents-complete.json") == before


def test_guard_change_stops_before_any_worker_is_spawned(tmp_path):
    command, verify = fake_commands(tmp_path)
    attempt = tmp_path / "attempt"; attempt.mkdir()
    guardfile = tmp_path / "guard"; guardfile.write_text("original")
    guards = {str(guardfile): campaign.sha(guardfile)}
    guardfile.write_text("changed")
    with campaign.campaign_lock(tmp_path / "campaign") as lease:
        result = campaign.dispatch_jobs(tuple(range(1, 9)), command, verify, cpus=[0, 1],
            deadline=time.monotonic() + 5, attempt=attempt, lease=lease,
            guard=lambda: campaign.verify_guards(guards))
    assert result["launched_seeds"] == [] and result["errors"]


@pytest.fixture
def authorization_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign.os, "sched_getaffinity", lambda pid: {0, 1})
    gate, review, evidence = (tmp_path / name for name in ("gate.json", "review.json", "queue-drain.json"))
    for path in (gate, review, evidence):
        write_json(path, {"synthetic_test_only": True})
    resource = {"safe_to_launch_parent_workers": True, "parent_workers": 2,
                "declared_project_active_workers": 8, "prerequisite_queues_drained": True,
                "evidence": [campaign.identity(evidence)]}
    resource_path = tmp_path / "resource.json"
    write_json(resource_path, resource)
    authorization = {"schema": campaign.AUTH_SCHEMA, "parent_adoption": True,
        "design_sha256": campaign.DESIGN_SHA256, "driver_sha256": campaign.sha(campaign.__file__),
        "workers": 2, "cpu_affinity": [0, 1], "whole_campaign_wall_seconds": 7200,
        "synthetic_gate": campaign.identity(gate), "independent_review": campaign.identity(review),
        "resource_decision": campaign.identity(resource_path)}
    authorization_path = tmp_path / "authorization.json"
    def save():
        write_json(resource_path, resource)
        authorization["resource_decision"] = campaign.identity(resource_path)
        write_json(authorization_path, authorization)
    save()
    return authorization, resource, authorization_path, evidence, save


def test_reviewed_resource_authorization_binds_actual_drain_receipts(authorization_fixture):
    authorization, resource, path, evidence, save = authorization_fixture
    guards = {}
    assert campaign.load_authorization(path, guards) == authorization
    assert guards[str(evidence)] == campaign.sha(evidence)
    assert guards[str(path)] == campaign.sha(path)
    evidence.write_text("changed queue evidence\n")
    with pytest.raises(ValueError, match="bound input changed"):
        campaign.load_authorization(path, {})


@pytest.mark.parametrize("change", [
    "unadopted", "schema", "source", "design", "three_workers", "boolean_workers",
    "duplicate_CPU", "unavailable_CPU", "wall_extension", "not_safe", "wrong_parent_count",
    "negative_project_count", "understated_project_count", "overcommitted", "no_actual_drain",
    "no_evidence", "changed_review_hash",
])
def test_unreviewed_or_overcommitted_authorization_never_admits_workers(authorization_fixture, change):
    authorization, resource, path, evidence, save = authorization_fixture
    if change == "unadopted": authorization["parent_adoption"] = False
    elif change == "schema": authorization["schema"] = "unreviewed"
    elif change == "source": authorization["driver_sha256"] = "0" * 64
    elif change == "design": authorization["design_sha256"] = "0" * 64
    elif change == "three_workers": authorization.update(workers=3, cpu_affinity=[0, 1, 2])
    elif change == "boolean_workers": authorization.update(workers=True, cpu_affinity=[0])
    elif change == "duplicate_CPU": authorization["cpu_affinity"] = [0, 0]
    elif change == "unavailable_CPU": authorization["cpu_affinity"] = [0, 2]
    elif change == "wall_extension": authorization["whole_campaign_wall_seconds"] = 7201
    elif change == "not_safe": resource["safe_to_launch_parent_workers"] = False
    elif change == "wrong_parent_count": resource["parent_workers"] = 1
    elif change == "negative_project_count": resource["declared_project_active_workers"] = -1
    elif change == "understated_project_count": resource["declared_project_active_workers"] = 1
    elif change == "overcommitted": resource["declared_project_active_workers"] = 9
    elif change == "no_actual_drain": resource["prerequisite_queues_drained"] = False
    elif change == "no_evidence": resource["evidence"] = []
    elif change == "changed_review_hash": authorization["independent_review"]["sha256"] = "0" * 64
    save()
    with pytest.raises(ValueError):
        campaign.load_authorization(path, {})
