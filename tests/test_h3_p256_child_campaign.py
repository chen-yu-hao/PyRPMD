"""Metadata, integrity and inert-process checks; no external PES or dynamics."""
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import numpy as np
import pytest

if sys.platform != "linux" or not hasattr(os, "sched_getaffinity"):
    pytest.skip("Frozen execution/affinity/process-group contract requires Linux", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.h3 import p256_child_campaign as campaign
from pyrpmd.checkpoint import atomic_savez
from pyrpmd.constants import FS_TO_AU


def put(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


@pytest.fixture
def fixture_context(tmp_path, monkeypatch):
    output = tmp_path/"children"
    monkeypatch.setattr(campaign, "OUTPUT", output)
    authorization_path = tmp_path/"authorization.json"
    manifest_path = tmp_path/"parents-complete.json"; put(manifest_path, {"synthetic": True})
    baseline_path = tmp_path/"complete-baseline.json"; put(baseline_path, {"synthetic": True})
    library_path = tmp_path/"inert-library-identity"; library_path.write_bytes(b"not an executable or PES")
    guard = tmp_path/"immutable-input"; guard.write_text("source guard fixture")
    rows = []
    for seed in campaign.SEEDS:
        rows.append({"seed": seed, "checkpoint_sha256": f"checkpoint-{seed}",
                     "files": {"settings.json": {"sha256": f"settings-{seed}"},
                               "report.json": {"sha256": f"report-{seed}"}},
                     "BAR_log_TS_quantum_classical_ratio": 0.})
    authorization = {"schema": campaign.AUTH_SCHEMA, "parent_adoption": True,
        "design_sha256": campaign.DESIGN_SHA, "P128_baseline_sha256": campaign.BASELINE_SHA,
        "source_files": {"synthetic-source": campaign.identity(guard)}, "runtime": {"fixture": True},
        "output": str(output), "seed_order": list(campaign.SEEDS),
        "branch_order": [[batch, dt] for batch, dt in campaign.BRANCHES],
        "whole_campaign_wall_seconds": 36000, "finalization_reserve_seconds": 120,
        "wall_budget_origin": "root explicit child-campaign choice; not the parent-generation proposal",
        "workers": 2, "cpu_affinity": [0, 1], "automatic_resume": False, "automatic_analysis": False,
        "parent_manifest": campaign.identity(manifest_path),
        "P128_complete_four_pair_report": {"path": str(baseline_path), "sha256": campaign.BASELINE_SHA, "bytes": baseline_path.stat().st_size},
        "library": campaign.identity(library_path)}
    put(authorization_path, authorization)
    context = {"authorization": authorization, "authorization_path": str(authorization_path),
        "specification": {}, "manifest": {"parents": rows, "registration_sha256": "parent-registration-fixture"},
        "guards": {str(guard): campaign.sha(guard), str(authorization_path): campaign.sha(authorization_path)},
        "physical_source_identity": {"synthetic-physical-source": "fixed"},
        "runtime": {"fixture": True}, "output": str(output)}
    return context


def protocol_for(context):
    return {"schema": campaign.PRODUCER_SCHEMA, "design_sha256": campaign.DESIGN_SHA,
        "seeds": list(campaign.SEEDS), "beads": 256, "parents_per_seed": 200, "horizon_fs": 200,
        "momentum_pairs_per_batch": 2, "registered_primary_pair_batches": [0, 1],
        "momentum_pair_indices": [0, 1, 2, 3], "children_per_pair": 2,
        "timestep_fs": [.1, .05], "primary_timestep_fs": .05,
        "parent_completed_sweeps": list(range(2100, 22001, 100)),
        "parent_manifest_sha256": context["authorization"]["parent_manifest"]["sha256"],
        "parent_registration_sha256": context["manifest"]["registration_sha256"],
        "source_sha256": context["physical_source_identity"],
        "P128_complete_four_pair_report": {"path": context["authorization"]["P128_complete_four_pair_report"]["path"],
                                           "sha256": campaign.BASELINE_SHA}}


def inert_registration(context):
    protocol = protocol_for(context)
    def register(output, library):
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        assert not any(output.iterdir())
        put(output/"protocol.json", dict(protocol, created_before_primary_children_utc="synthetic fixture"))
        return protocol
    return SimpleNamespace(register=register)


def branch_row(output, seed, dt, batch):
    directory = campaign.branch_directory(output, seed, dt, batch).resolve()
    count = 200*2*2*round(200/dt)
    return {"seed": seed, "pair_batch": batch, "timestep_fs": dt, "complete": True,
        "parents": 200, "pairs_per_parent": 2, "native_steps": count, "python_steps": 0,
        "settings": {"path": str(directory/"settings.json"), "sha256": "synthetic", "bytes": 1},
        "report": {"path": str(directory/"report.json"), "sha256": "synthetic", "bytes": 1},
        "chunks": [{"path": str(directory/f"chunk-{left:04d}.npz"), "sha256": "synthetic", "bytes": 1,
                    "initial_p_sha256": f"shared-p-{seed}-{batch}-{left}", "channel_sha256": f"shared-channel-{seed}-{left}"}
                   for left in range(0, 200, 8)]}


def seed_row(output, seed):
    return {"seed": seed, "complete": True, "branches": [branch_row(output, seed, dt, batch) for batch, dt in campaign.BRANCHES],
            "native_steps": 9_600_000, "rate_analysis_performed": False}


def test_fixed_child_counts_and_reused_dispatcher_are_explicit():
    assert campaign.sha(campaign.SCHEDULER_PATH) == campaign.SCHEDULER_SHA
    assert campaign.shared().dispatch_jobs.__module__ == "benchmarks.h3.p256_parent_campaign"
    assert campaign.PARENTS == 200 and campaign.BEADS == 256 and len(campaign.SEEDS) == 8
    assert campaign.BRANCHES == ((0, .1), (0, .05), (1, .1), (1, .05))
    assert sum(200*2*2*round(200/dt) for _, dt in campaign.BRANCHES)*8 == 76_800_000
    assert campaign.WALL_SECONDS == 36000 and campaign.FINALIZATION_RESERVE_SECONDS == 120


@pytest.mark.parametrize("change", [None, "adoption", "schema", "design", "baseline", "source", "runtime", "output",
    "seed_subset", "branch_subset", "branch_order", "parent_wall_budget", "reserve", "budget_origin", "three_workers",
    "boolean_workers", "duplicate_CPU", "unavailable_CPU", "resume", "analysis"])
def test_authorization_rejects_unreviewed_or_partial_contract(fixture_context, change):
    context = fixture_context
    authorization = json.loads(json.dumps(context["authorization"]))
    if change == "adoption": authorization["parent_adoption"] = False
    elif change == "schema": authorization["schema"] = "wrong"
    elif change == "design": authorization["design_sha256"] = "changed"
    elif change == "baseline": authorization["P128_baseline_sha256"] = "changed"
    elif change == "source": authorization["source_files"] = {}
    elif change == "runtime": authorization["runtime"] = {}
    elif change == "output": authorization["output"] += "-unregistered"
    elif change == "seed_subset": authorization["seed_order"] = authorization["seed_order"][:4]
    elif change == "branch_subset": authorization["branch_order"] = authorization["branch_order"][:2]
    elif change == "branch_order": authorization["branch_order"].reverse()
    elif change == "parent_wall_budget": authorization["whole_campaign_wall_seconds"] = 7200
    elif change == "reserve": authorization["finalization_reserve_seconds"] = 0
    elif change == "budget_origin": authorization["wall_budget_origin"] = "assumed from parent proposal"
    elif change == "three_workers": authorization.update(workers=3, cpu_affinity=[0, 1, 2])
    elif change == "boolean_workers": authorization.update(workers=True, cpu_affinity=[0])
    elif change == "duplicate_CPU": authorization["cpu_affinity"] = [0, 0]
    elif change == "unavailable_CPU": authorization["cpu_affinity"] = [0, 2]
    elif change == "resume": authorization["automatic_resume"] = True
    elif change == "analysis": authorization["automatic_analysis"] = True
    arguments = dict(actual_sources=context["authorization"]["source_files"], actual_runtime=context["runtime"], available_cpus={0, 1})
    if change is None:
        campaign.validate_authorization(authorization, **arguments)
    else:
        with pytest.raises(ValueError): campaign.validate_authorization(authorization, **arguments)


@pytest.mark.parametrize("field", ["parent_manifest_sha256", "parent_registration_sha256", "source_sha256", "P128_complete_four_pair_report",
                                  "beads", "parents_per_seed", "seeds", "registered_primary_pair_batches"])
def test_protocol_cannot_change_parent_baseline_source_or_counts(fixture_context, field):
    protocol = protocol_for(fixture_context)
    campaign.validate_protocol(protocol, fixture_context)
    protocol[field] = "changed"
    with pytest.raises(ValueError): campaign.validate_protocol(protocol, fixture_context)


def raw_arrays(indices, fingerprint="fixture", dt=.1, batch=0):
    n = len(indices)
    return {"fingerprint": np.asarray(fingerprint), "parent_index": np.asarray(indices),
        "parent_completed_sweeps": 2000+(np.asarray(indices)+1)*100, "channel": np.zeros(n, dtype=np.int64),
        "momentum_pair_index": np.arange(2*batch, 2*batch+2), "time_fs": np.arange(1, 201, dtype=float),
        "surface_values": np.zeros((n, 2, 2, 200)), "minimum_pair_at_1fs": np.ones((n, 2, 2, 201)),
        "energy_extrema": np.zeros((n, 2, 2, 3)), "initial_p": np.zeros((n, 2, 256, 3, 3)),
        "final_q": np.zeros((n, 2, 2, 256, 3, 3)), "final_p": np.zeros((n, 2, 2, 256, 3, 3)),
        "final_step": np.full((n, 2, 2), round(200/dt), dtype=np.int64),
        "final_time_au": np.full((n, 2, 2), 200*FS_TO_AU),
        "flux_metric": np.ones(n), "velocity": np.zeros((n, 2)), "numerator": np.zeros((n, 2, 200)),
        "denominator": np.zeros((n, 2)), "native_steps_this_stage": np.asarray(n*4*round(200/dt)),
        "previous_chunk_sha256": np.asarray("")}


@pytest.mark.parametrize("change", [None, "parent_index", "sweep", "pair", "time", "channel", "nonfinite", "shape",
                                  "native_count", "final_step", "final_clock", "fingerprint", "extension", "missing_payload"])
def test_raw_chunk_structural_accounting_without_rate_calculation(change):
    arrays = raw_arrays(np.arange(2))
    if change == "parent_index": arrays["parent_index"][0] = 99
    elif change == "sweep": arrays["parent_completed_sweeps"][0] += 100
    elif change == "pair": arrays["momentum_pair_index"] += 2
    elif change == "time": arrays["time_fs"][-1] = 199
    elif change == "channel": arrays["channel"][0] = 2
    elif change == "nonfinite": arrays["final_p"].flat[0] = np.nan
    elif change == "shape": arrays["initial_p"] = arrays["initial_p"][:, :, :-1]
    elif change == "native_count": arrays["native_steps_this_stage"] += 1
    elif change == "final_step": arrays["final_step"].flat[0] -= 1
    elif change == "final_clock": arrays["final_time_au"].fill(1.)
    elif change == "fingerprint": arrays["fingerprint"] = np.asarray("other")
    elif change == "extension": arrays["previous_chunk_sha256"] = np.asarray("unregistered predecessor")
    elif change == "missing_payload": arrays.pop("surface_values")
    arguments = dict(indices=np.arange(2), settings_fingerprint="fixture", timestep=.1, batch=0)
    if change is None:
        assert campaign.validate_chunk(arrays, **arguments)["native_steps"] == 16000
    else:
        with pytest.raises(ValueError): campaign.validate_chunk(arrays, **arguments)


@pytest.mark.parametrize("timestep", [.1, .05])
def test_floating_clock_recurrence_fits_derived_bound_without_dynamics(timestep):
    arrays = raw_arrays(np.arange(1), dt=timestep)
    clock = 0.
    for _ in range(round(200/timestep)):
        clock += timestep*FS_TO_AU
    arrays["final_time_au"].fill(clock)
    checked = campaign.validate_chunk(arrays, indices=np.arange(1), settings_fingerprint="fixture", timestep=timestep, batch=0)
    assert checked["native_steps"] == 4*round(200/timestep)
    arrays["final_time_au"] += FS_TO_AU
    with pytest.raises(ValueError, match="final clock"):
        campaign.validate_chunk(arrays, indices=np.arange(1), settings_fingerprint="fixture", timestep=timestep, batch=0)


def test_actual_checkpoint_envelope_is_required(tmp_path):
    arrays = raw_arrays(np.arange(1))
    path = tmp_path/"chunk.npz"
    atomic_savez(path, kind="h3.quantum-surface-children", **arrays)
    loaded = campaign.read_child_chunk(path.read_bytes())
    assert campaign.validate_chunk(loaded, indices=np.arange(1), settings_fingerprint="fixture", timestep=.1, batch=0)["native_steps"] == 8000
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError): campaign.read_child_chunk(path.read_bytes())


@pytest.fixture
def complete_branch(fixture_context, monkeypatch):
    context = fixture_context; output = Path(context["output"]); protocol = protocol_for(context)
    seed, dt, batch = campaign.SEEDS[0], .1, 0
    directory = campaign.branch_directory(output, seed, dt, batch)
    settings = {"schema": campaign.PRODUCER_SCHEMA, "phase": "primary", "seed": seed, "beads": 256,
        "parents": 200, "timestep_fs": dt, "horizon_fs": 200, "start_horizon_fs": 0,
        "pair_batch": batch, "momentum_pair_indices": [0, 1], "protocol_sha256": campaign.fingerprint(protocol),
        "parent_identity": campaign.parent_identity(context, seed), "source_sha256": context["physical_source_identity"], "chunk_parents": 8}
    put(directory/"settings.json", settings)
    inventory = []
    for left in range(0, 200, 8):
        path = directory/f"chunk-{left:04d}.npz"; path.write_text(str(left))
        inventory.append({"file": path.name, "sha256": campaign.sha(path)})
    report = {key: settings[key] for key in ("schema", "phase", "seed", "pair_batch", "timestep_fs", "horizon_fs", "momentum_pair_indices", "parent_identity")}
    report.update(complete=True, parents_completed=200, pairs_per_parent=2, source_unchanged=True,
        native_steps_this_stage=1600000, unique_native_steps_all_stages=1600000, python_steps=0,
        settings_sha256=campaign.sha(directory/"settings.json"), chunks=inventory,
        plateaus={"must_not_be_used_for_completion": "no scientific inference in this fixture"})
    put(directory/"report.json", report)
    monkeypatch.setattr(campaign, "read_child_chunk", lambda raw: raw_arrays(np.arange(int(raw), int(raw)+8), campaign.fingerprint(settings)))
    return context, protocol, directory, settings, report


@pytest.mark.parametrize("change", [None, "incomplete", "parents", "native", "python", "duplicate", "missing", "extra", "settings"])
def test_complete_branch_requires_exact_inventory_and_actual_counts(complete_branch, change):
    context, protocol, directory, settings, report = complete_branch
    if change == "incomplete": report["complete"] = False
    elif change == "parents": report["parents_completed"] = 199
    elif change == "native": report["native_steps_this_stage"] -= 1
    elif change == "python": report["python_steps"] = 1
    elif change == "duplicate": report["chunks"][-1] = report["chunks"][0]
    elif change == "missing": (directory/"chunk-0192.npz").unlink()
    elif change == "extra": (directory/"chunk-0200.npz").write_text("200")
    elif change == "settings": settings["seed"] = campaign.SEEDS[1]; put(directory/"settings.json", settings)
    put(directory/"report.json", report)
    args = (context["output"], context, protocol, campaign.SEEDS[0], .1, 0)
    if change is None:
        checked = campaign.verify_branch(*args)
        assert len(checked["chunks"]) == 25 and checked["native_steps"] == 1_600_000
        assert "plateaus" not in checked
    else:
        with pytest.raises((ValueError, FileNotFoundError)): campaign.verify_branch(*args)


def test_paired_timestep_momentum_identity_is_required(fixture_context, monkeypatch):
    context = fixture_context; output = context["output"]
    monkeypatch.setattr(campaign, "verify_branch", lambda root, ctx, protocol, seed, dt, batch: branch_row(root, seed, dt, batch))
    assert campaign.verify_seed(output, context, {}, campaign.SEEDS[0])["native_steps"] == 9_600_000
    def changed(root, ctx, protocol, seed, dt, batch):
        row = branch_row(root, seed, dt, batch)
        if dt == .05: row["chunks"][0]["initial_p_sha256"] = "different draw"
        return row
    monkeypatch.setattr(campaign, "verify_branch", changed)
    with pytest.raises(ValueError, match="Maxwell"): campaign.verify_seed(output, context, {}, campaign.SEEDS[0])


@pytest.mark.parametrize("change", [None, "four_seeds", "duplicate_seed", "one_batch", "wrong_dt", "foreign_chunk", "missing_chunk", "steps"])
def test_global_completion_requires_all_32_branches_and_800_unique_chunks(fixture_context, change):
    output = fixture_context["output"]
    rows = [seed_row(output, seed) for seed in campaign.SEEDS]
    if change == "four_seeds": rows = rows[:4]
    elif change == "duplicate_seed": rows[-1] = rows[0]
    elif change == "one_batch": rows[-1]["branches"] = rows[-1]["branches"][:2]
    elif change == "wrong_dt": rows[-1]["branches"][-1]["timestep_fs"] = .1
    elif change == "foreign_chunk": rows[-1]["branches"][-1]["chunks"][0]["path"] = rows[0]["branches"][0]["chunks"][0]["path"]
    elif change == "missing_chunk": rows[-1]["branches"][-1]["chunks"].pop()
    elif change == "steps": rows[-1]["native_steps"] += 1
    if change is None:
        assert campaign.verify_complete_cohort(rows, output) == {"branches": 32, "chunks": 800, "native_steps": 76_800_000}
    else:
        with pytest.raises(ValueError): campaign.verify_complete_cohort(rows, output)


def install_inert_campaign(context, tmp_path, monkeypatch, *, failure_seed=0, missing_receipt_seed=0):
    monkeypatch.setattr(campaign, "preflight", lambda path: context)
    monkeypatch.setattr(campaign, "producer", lambda: inert_registration(context))
    monkeypatch.setattr(campaign, "verify_seed", lambda output, ctx, protocol, seed: seed_row(output, seed))
    script = tmp_path/"inert-worker.py"
    script.write_text("""
import json,os,sys,time
from pathlib import Path
seed,cpu,fail,missing=map(int,sys.argv[1:5]);attempt=Path(sys.argv[5])
time.sleep(.025)
if seed==fail: raise SystemExit(3)
if seed!=missing:
    receipt={'seed':seed,'complete':True,'failure':None,'native_steps':9600000,
        'completed_branches':[{'batch':b,'timestep_fs':dt} for b,dt in ((0,.1),(0,.05),(1,.1),(1,.05))],
        'CPU_affinity':[cpu], 'thread_environment':{k:os.environ[k] for k in
          ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        'whole_worker_call_wall_seconds':.025,'whole_process_user_CPU_seconds':.001,'whole_process_system_CPU_seconds':.001}
    (attempt/f'seed-{seed}-execution.json').write_text(json.dumps(receipt))
""")
    monkeypatch.setattr(campaign, "worker_command", lambda registration, attempt, seed, cpu:
        [sys.executable, str(script), str(seed), str(cpu), str(failure_seed), str(missing_receipt_seed), str(attempt)])


def test_single_coordinator_completes_fixed_matrix_and_refuses_implicit_resume(fixture_context, tmp_path, monkeypatch):
    context = fixture_context
    install_inert_campaign(context, tmp_path, monkeypatch)
    result = campaign.run_campaign(context["authorization_path"])
    assert result["complete_registered_cohort"] and result["all_owned_workers_drained"]
    assert result["native_steps"] == 76_800_000 and not result["rate_analysis_performed"]
    assert len(result["seeds"]) == 8 and sum(len(x["branches"]) for x in result["seeds"]) == 32
    path = Path(context["output"])/"children-complete.json"; before = campaign.sha(path)
    with pytest.raises(ValueError, match="one-attempt"): campaign.run_campaign(context["authorization_path"])
    assert campaign.sha(path) == before


def test_worker_failure_preserves_attempt_and_stops_admission_without_global_result(fixture_context, tmp_path, monkeypatch):
    context = fixture_context
    install_inert_campaign(context, tmp_path, monkeypatch, failure_seed=campaign.SEEDS[0])
    result = campaign.run_campaign(context["authorization_path"])
    assert result["errors"] and not result["complete_registered_cohort"] and result["all_owned_workers_drained"]
    assert result["launched_seeds"] == list(campaign.SEEDS[:2])
    output = Path(context["output"])
    assert not (output/"children-complete.json").exists()
    assert len(list((output/"coordinator-attempts").glob("*/result.json"))) == 1
    with pytest.raises(ValueError): campaign.run_campaign(context["authorization_path"])


def test_missing_per_seed_accounting_cannot_enable_global_completion(fixture_context, tmp_path, monkeypatch):
    context = fixture_context
    install_inert_campaign(context, tmp_path, monkeypatch, missing_receipt_seed=campaign.SEEDS[-1])
    result = campaign.run_campaign(context["authorization_path"])
    assert not result["complete_registered_cohort"] and any("worker accounting" in x for x in result["errors"])
    assert not (Path(context["output"])/"children-complete.json").exists()


def test_guard_mutation_during_final_inventory_prevents_publication(fixture_context, tmp_path, monkeypatch):
    context = fixture_context
    install_inert_campaign(context, tmp_path, monkeypatch)
    original = campaign.verify_complete_cohort
    guard = next(Path(path) for path in context["guards"] if Path(path).name == "immutable-input")
    def changed_after_audit(rows, output):
        result = original(rows, output)
        guard.write_text("changed after final raw audit")
        return result
    monkeypatch.setattr(campaign, "verify_complete_cohort", changed_after_audit)
    result = campaign.run_campaign(context["authorization_path"])
    assert not result["complete_registered_cohort"] and result["all_owned_workers_drained"]
    assert any("final publication guard" in message for message in result["errors"])
    assert not (Path(context["output"])/"children-complete.json").exists()


def test_registration_failure_is_retained_and_not_automatically_retried(fixture_context, monkeypatch):
    context = fixture_context
    monkeypatch.setattr(campaign, "preflight", lambda path: context)
    def bad_register(output, library):
        raise ValueError("inert registration rejection")
    monkeypatch.setattr(campaign, "producer", lambda: SimpleNamespace(register=bad_register))
    with pytest.raises(ValueError, match="inert registration"): campaign.run_campaign(context["authorization_path"])
    failure = Path(context["output"])/"coordinator-pre-dispatch-failure.json"
    assert campaign.read_json(failure)["workers_launched"] == 0
    with pytest.raises(ValueError, match="one-attempt"): campaign.run_campaign(context["authorization_path"])


def test_unconfirmed_dispatch_drain_retains_reused_lease(fixture_context, monkeypatch):
    context = fixture_context
    monkeypatch.setattr(campaign, "preflight", lambda path: context)
    monkeypatch.setattr(campaign, "producer", lambda: inert_registration(context))
    def unconfirmed(*args, **kwargs):
        kwargs["lease"]["release"] = False
        raise RuntimeError("inert unconfirmed drain")
    monkeypatch.setattr(campaign.shared(), "dispatch_jobs", unconfirmed)
    with pytest.raises(RuntimeError, match="unconfirmed"): campaign.run_campaign(context["authorization_path"])
    output = Path(context["output"])
    assert (output.parent/("."+output.name+"-p256-parent-lock")/"owner.json").exists()
    assert not (output/"children-complete.json").exists()


def worker_fixture(context, monkeypatch):
    output = Path(context["output"]); inert_registration(context).register(output, None)
    registration = output/"coordinator-registration.json"
    put(registration, {"schema": campaign.REG_SCHEMA, "authorization_path": context["authorization_path"],
        "authorization_sha256": campaign.sha(context["authorization_path"]), "source_sha256": campaign.sha(campaign.__file__),
        "seed_order": list(campaign.SEEDS), "protocol_sha256": campaign.sha(output/"protocol.json")})
    attempt = output/"coordinator-attempts"/"fixture"; attempt.mkdir(parents=True)
    guards = dict(context["guards"], **{str(registration): campaign.sha(registration), str(output/"protocol.json"): campaign.sha(output/"protocol.json")})
    put(attempt/"plan.json", {"registration_sha256": campaign.sha(registration), "coordinator_pid": os.getppid(),
        "coordinator_process_token": "inert-owner-token", "cpu_affinity": [0, 1], "workers": 2,
        "whole_campaign_wall_seconds": 36000, "finalization_reserve_seconds": 120,
        "source_guards": guards, "seed_order": list(campaign.SEEDS),
        "branch_order": [[batch, dt] for batch, dt in campaign.BRANCHES]})
    monkeypatch.setenv(campaign.TOKEN_ENV, "inert-owner-token")
    for name in campaign.THREAD_VARIABLES: monkeypatch.setenv(name, "1")
    monkeypatch.setattr(campaign, "preflight", lambda path: context)
    monkeypatch.setattr(campaign.os, "sched_setaffinity", lambda *args: None)
    monkeypatch.setattr(campaign.os, "sched_getaffinity", lambda pid: {0})
    monkeypatch.setattr(campaign, "verify_branch", lambda root, ctx, protocol, seed, dt, batch: branch_row(root, seed, dt, batch))
    monkeypatch.setattr(campaign, "verify_seed", lambda root, ctx, protocol, seed: seed_row(root, seed))
    return registration, attempt


def test_worker_runs_exact_four_branch_order_without_analysis_or_resume(fixture_context, monkeypatch):
    context = fixture_context; registration, attempt = worker_fixture(context, monkeypatch)
    calls = []
    def run(output, library, seed, dt, **kwargs):
        calls.append((seed, dt, kwargs))
        campaign.branch_directory(output, seed, dt, kwargs["pair_batch"]).mkdir(parents=True)
        return {"ignored_partial_scientific_field": "must not be inspected"}
    monkeypatch.setattr(campaign, "producer", lambda: SimpleNamespace(run_seed=run))
    assert campaign.worker(registration, attempt, campaign.SEEDS[0], 0) == 0
    assert calls == [(campaign.SEEDS[0], dt, {"horizon": 200, "pair_batch": batch}) for batch, dt in campaign.BRANCHES]
    receipt = campaign.read_json(attempt/f"seed-{campaign.SEEDS[0]}-execution.json")
    assert receipt["complete"] and len(receipt["completed_branches"]) == 4
    assert not receipt["rate_analysis_performed"] and receipt["whole_process_user_CPU_seconds"] >= 0
    with pytest.raises(FileExistsError): campaign.worker(registration, attempt, campaign.SEEDS[0], 0)


def test_worker_signal_preserves_failure_receipt_and_restores_handlers(fixture_context, monkeypatch):
    context = fixture_context; registration, attempt = worker_fixture(context, monkeypatch)
    old_handler = signal.getsignal(signal.SIGTERM)
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(kwargs)
        os.kill(os.getpid(), signal.SIGTERM)
    monkeypatch.setattr(campaign, "producer", lambda: SimpleNamespace(run_seed=interrupted))
    with pytest.raises(InterruptedError): campaign.worker(registration, attempt, campaign.SEEDS[0], 0)
    assert len(calls) == 1 and signal.getsignal(signal.SIGTERM) == old_handler
    receipt = campaign.read_json(attempt/f"seed-{campaign.SEEDS[0]}-execution.json")
    assert not receipt["complete"] and receipt["failure"]["type"] == "InterruptedError"
    assert receipt["whole_worker_call_wall_seconds"] >= 0 and receipt["native_steps"] is None


def test_private_worker_requires_current_inherited_owner_token(fixture_context, monkeypatch):
    registration, attempt = worker_fixture(fixture_context, monkeypatch)
    monkeypatch.delenv(campaign.TOKEN_ENV)
    monkeypatch.setattr(campaign, "producer", lambda: pytest.fail("unowned worker reached producer"))
    with pytest.raises(ValueError, match="owning live coordinator"):
        campaign.worker(registration, attempt, campaign.SEEDS[0], 0)
    assert not list(attempt.glob("*-worker-claim.json"))
