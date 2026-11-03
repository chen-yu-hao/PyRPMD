"""Real-rate precision CLI, hard budgets, and durable interruption recovery."""

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pyrpmd.constants import KB_AU
from pyrpmd.precision import (
    CheckSchedule, FrozenProduction, NumericalRequirement, PrecisionSpec,
    PrecisionTarget, ScientificIdentity,
)
from pyrpmd.precision_jobs import materialize_rate_job
from pyrpmd.potentials import FreePotential
from pyrpmd.runner import run_resolved
from pyrpmd.workflows.precision import (
    plan_precision_resolved, precision_status, run_precision,
)

from test_precision_jobs import configuration, frozen_jobs


def request(config, system, **overrides):
    """Deliberately short physical trajectories cannot certify a real rate."""
    _, frozen = frozen_jobs(config, system)
    return {
        "target": {"relative_half_width": .2},
        "checks": {"replicas": [2], "max_epochs": 1},
        "sampling": asdict(frozen.sampling),
        "max_cost": 100_000,
        "job_max_cost": 10_000,
        "cost_unit": "force_evaluations",
        "numerical": [],
        "seed": 41,
        **overrides,
    }


def ledger(directory):
    return json.loads((directory / "precision" / "ledger.json").read_text(encoding="utf-8"))


def frozen_from_ledger(directory):
    saved = ledger(directory)
    value = saved["specification"]
    spec = PrecisionSpec(
        ScientificIdentity(**value["identity"]),
        PrecisionTarget(**value["target"]),
        CheckSchedule(**value["checks"]), value["max_cost"],
        numerical=tuple(NumericalRequirement(**item) for item in value["numerical"]),
        cost_unit=value["cost_unit"], seed=value["seed"],
    )
    frozen = next(item for item in saved["checks"].values()
                  if item.get("format") == "PyRPMD.precision.frozen-production.v1")
    return FrozenProduction.from_dict(spec, frozen)


def assert_same_trajectories(actual, reference):
    """Compare dynamics and RNG to the independent existing rate executor."""
    names = {"q", "p", "step", "time", "elapsed", "rng", "snapshots", "values",
             "energies", "numerator", "denominator", "energy_drift",
             "endpoint_energy_error_hartree_per_bead", "endpoint_energy_error_kbt"}
    checkpoints = sorted(reference.rglob("*.npz"))
    assert checkpoints
    compared = set()
    for source in checkpoints:
        target = actual / source.relative_to(reference)
        assert target.is_file()
        with np.load(source, allow_pickle=False) as first, np.load(target, allow_pickle=False) as second:
            for name in names.intersection(first.files):
                np.testing.assert_array_equal(second[name], first[name], err_msg=str(target) + ":" + name)
                compared.add(name)
    assert {"q", "p", "rng", "numerator", "denominator"}.issubset(compared)


@pytest.mark.parametrize("scoped", [False, True])
def test_real_rate_plan_pause_resume_stop_and_replay(tmp_path, monkeypatch, scoped):
    config, system = configuration(scoped=scoped)
    original = deepcopy(config)
    directory = tmp_path / "precision campaign"
    with monkeypatch.context() as patch:
        patch.setattr("pyrpmd.potentials.FreePotential.evaluate",
                      lambda *args, **kwargs: pytest.fail("planning evaluated the PES"))
        planned = plan_precision_resolved(config, system, request(config, system), output=directory)
    assert config == original
    assert planned["budget"]["spent_cost"] == 0
    assert planned["budget"]["charged_cost"] == 0
    assert len(planned["jobs"]) == 2
    assert all(job["charged_cost"] == 0 for job in planned["jobs"])
    initial_ledger = (directory / "precision" / "ledger.json").read_bytes()
    assert precision_status(directory)["specification_id"] == planned["specification_id"]
    replayed_plan = plan_precision_resolved(config, system, request(config, system), output=directory)
    assert replayed_plan["specification_id"] == planned["specification_id"]
    assert (directory / "precision" / "ledger.json").read_bytes() == initial_ledger

    evaluated_frames = []
    original_evaluate = FreePotential.evaluate

    def counted_evaluate(provider, q, **kwargs):
        evaluated_frames.append(len(q))
        return original_evaluate(provider, q, **kwargs)

    monkeypatch.setattr(FreePotential, "evaluate", counted_evaluate)
    paused = run_precision(directory, max_jobs=1)
    assert paused["status"] == "paused"
    assert len([job for job in paused["jobs"] if job["state"] == "completed"]) == 1
    assert 0 < paused["budget"]["charged_cost"] <= 10_000
    assert paused["budget"]["charged_cost"] == sum(evaluated_frames)
    assert paused["budget"]["spent_cost"] == 0  # An increment is committed as one receipt.
    assert len(ledger(directory)["results"]) == 0

    final = run_precision(directory)
    assert final["stop"]["status"] == "diagnostic_failed"
    assert final["latest_assessment"]["accepted"] is False
    assert final["latest_assessment"]["numerical_convergence_established"] is False
    assert all(job["state"] == "completed" for job in final["jobs"])
    assert final["budget"]["spent_cost"] == final["budget"]["charged_cost"]
    assert final["budget"]["charged_cost"] == sum(job["charged_cost"] for job in final["jobs"])
    assert final["budget"]["charged_cost"] == sum(evaluated_frames)
    assert final["budget"]["reserved_cost"] == 0
    saved = ledger(directory)
    assert len(saved["results"]) == 2
    assert len([value for value in saved["checks"].values() if "assessment" in value]) == 1
    committed = (directory / "precision" / "ledger.json").read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr("pyrpmd.potentials.FreePotential.evaluate",
                      lambda *args, **kwargs: pytest.fail("terminal replay evaluated the PES"))
        repeated = run_precision(directory)
    assert repeated["budget"] == final["budget"]
    assert repeated["stop"] == final["stop"]
    assert (directory / "precision" / "ledger.json").read_bytes() == committed

    monkeypatch.setattr(FreePotential, "evaluate", original_evaluate)
    frozen = frozen_from_ledger(directory)
    job = materialize_rate_job(config, system, frozen, frozen.jobs[0]["job_id"])
    reference = tmp_path / "unmetered-reference"
    run_resolved(job["config"], system, output=reference, **job["run_kwargs"])
    actual = Path(next(item for item in final["jobs"] if item["job_id"] == job["job_id"])["directory"])
    assert_same_trajectories(actual, reference)


def test_real_checkpoint_interruption_preserves_rng_and_job_identity(tmp_path, monkeypatch):
    from pyrpmd.workflows import fes

    config, system = configuration()
    directory = tmp_path / "interrupted"
    planned = plan_precision_resolved(config, system, request(config, system), output=directory)
    original_save = fes._atomic_npz
    committed = []

    def interrupt_after_checkpoint(path, **arrays):
        original_save(path, **arrays)
        if not committed:
            committed.append(Path(path))
            raise KeyboardInterrupt("interrupt after a durable umbrella checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(fes, "_atomic_npz", interrupt_after_checkpoint)
        with pytest.raises(KeyboardInterrupt, match="durable umbrella"):
            run_precision(directory)
    assert len(committed) == 1 and committed[0].is_file()
    interrupted = precision_status(directory)
    assert interrupted["budget"]["charged_cost"] > 0
    assert interrupted["budget"]["spent_cost"] == 0
    assert len(ledger(directory)["results"]) == 0
    running = [entry for entry in ledger(directory)["plans"].values() if entry["state"] == "running"]
    assert len(running) == 1
    execution_id = running[0]["execution_id"]

    final = run_precision(directory)
    assert final["stop"]["status"] == "diagnostic_failed"
    assert final["budget"]["charged_cost"] >= interrupted["budget"]["charged_cost"]
    assert final["specification_id"] == planned["specification_id"]
    saved = ledger(directory)
    assert len(saved["plans"]) == 1 and len(saved["results"]) == 2
    assert next(iter(saved["plans"].values()))["execution_id"] == execution_id
    frozen = frozen_from_ledger(directory)
    job = materialize_rate_job(config, system, frozen, frozen.jobs[0]["job_id"])
    reference = tmp_path / "reference"
    run_resolved(job["config"], system, output=reference, **job["run_kwargs"])
    actual = Path(next(item for item in final["jobs"] if item["job_id"] == job["job_id"])["directory"])
    assert_same_trajectories(actual, reference)


@pytest.mark.parametrize("receipt_written", [False, True])
def test_completed_artifact_reconciliation_never_recomputes_first_job(tmp_path, monkeypatch, receipt_written):
    from pyrpmd.precision_execution import JobMeter

    config, system = configuration()
    directory = tmp_path / "artifact-recovery"
    plan_precision_resolved(config, system, request(config, system), output=directory)
    original_complete = JobMeter.complete
    stopped = []

    def interrupt_receipt(meter, record):
        if receipt_written:
            original_complete(meter, record)
        if not stopped:
            stopped.append(record["job_id"])
            raise KeyboardInterrupt("interrupt at immutable job receipt")
        if not receipt_written:
            original_complete(meter, record)

    with monkeypatch.context() as patch:
        patch.setattr(JobMeter, "complete", interrupt_receipt)
        with pytest.raises(KeyboardInterrupt, match="immutable job receipt"):
            run_precision(directory)
    assert len(stopped) == 1
    before = precision_status(directory)
    first = next(job for job in before["jobs"] if job["job_id"] == stopped[0])
    assert (Path(first["directory"]) / "summary.json").is_file()
    assert first["charged_cost"] > 0
    assert first["state"] == ("completed" if receipt_written else "running")
    assert not ledger(directory)["results"]
    if not receipt_written:
        # Reconciliation of a published result counts as one completed job,
        # with no fresh PES evaluation or new statistical source.
        with monkeypatch.context() as patch:
            patch.setattr(FreePotential, "evaluate",
                          lambda *args, **kwargs: pytest.fail("published job was recomputed"))
            recovered = run_precision(directory, max_jobs=1)
        assert recovered["status"] == "paused"
        assert recovered["budget"]["charged_cost"] == before["budget"]["charged_cost"]
    final = run_precision(directory)
    after = next(job for job in final["jobs"] if job["job_id"] == stopped[0])
    assert after["charged_cost"] == first["charged_cost"]
    assert after["evaluation_batches"] == first["evaluation_batches"]
    assert final["stop"]["status"] == "diagnostic_failed"
    assert len(ledger(directory)["results"]) == 2


def test_increment_receipt_recovers_check_without_charging_or_sampling_twice(tmp_path, monkeypatch):
    from pyrpmd.precision_ledger import PrecisionLedger

    config, system = configuration()
    directory = tmp_path / "check-recovery"
    plan_precision_resolved(config, system, request(config, system), output=directory)
    original_complete = PrecisionLedger.complete

    def interrupt_after_complete(store, *args, **kwargs):
        original_complete(store, *args, **kwargs)
        raise KeyboardInterrupt("interrupt after durable increment receipt")

    with monkeypatch.context() as patch:
        patch.setattr(PrecisionLedger, "complete", interrupt_after_complete)
        with pytest.raises(KeyboardInterrupt, match="durable increment receipt"):
            run_precision(directory)
    before = precision_status(directory)
    assert before["latest_assessment"] is None and before["stop"] is None
    assert before["budget"]["spent_cost"] == before["budget"]["charged_cost"] > 0
    assert len(ledger(directory)["results"]) == 2
    with monkeypatch.context() as patch:
        patch.setattr(FreePotential, "evaluate",
                      lambda *args, **kwargs: pytest.fail("reconciled increment was recomputed"))
        final = run_precision(directory)
    assert final["stop"]["status"] == "diagnostic_failed"
    assert final["latest_assessment"]["accepted"] is False
    assert final["budget"]["charged_cost"] == before["budget"]["charged_cost"]
    assert len(ledger(directory)["results"]) == 2
    assert len([value for value in ledger(directory)["checks"].values() if "assessment" in value]) == 1
    assert run_precision(directory)["budget"] == final["budget"]


def test_replanning_after_increment_receipt_cannot_skip_a_due_check(tmp_path, monkeypatch):
    from pyrpmd.precision_ledger import PrecisionLedger

    config, system = configuration()
    directory = tmp_path / "replan-check-recovery"
    specification = request(config, system, checks={"replicas": [2, 4], "max_epochs": 1})
    plan_precision_resolved(config, system, specification, output=directory)
    original_complete = PrecisionLedger.complete

    def interrupt_after_complete(store, *args, **kwargs):
        original_complete(store, *args, **kwargs)
        raise KeyboardInterrupt("interrupt before the first scheduled check")

    with monkeypatch.context() as patch:
        patch.setattr(PrecisionLedger, "complete", interrupt_after_complete)
        with pytest.raises(KeyboardInterrupt, match="first scheduled check"):
            run_precision(directory)
    before = precision_status(directory)
    assert before["stop"] is None and before["latest_assessment"] is None
    assert len(ledger(directory)["plans"]) == 1
    assert len(ledger(directory)["results"]) == 2

    with monkeypatch.context() as patch:
        patch.setattr(FreePotential, "evaluate",
                      lambda *args, **kwargs: pytest.fail("replanning skipped the pending check"))
        final = plan_precision_resolved(config, system, specification, output=directory)
        repeated = run_precision(directory)
    assert final["stop"]["status"] == "diagnostic_failed"
    assert final["latest_assessment"]["accepted"] is False
    assert final["budget"]["charged_cost"] == before["budget"]["charged_cost"]
    assert repeated["budget"] == final["budget"]
    assert len(ledger(directory)["plans"]) == 1
    assert len(ledger(directory)["results"]) == 2
    assert len([value for value in ledger(directory)["checks"].values() if "assessment" in value]) == 1
    assert all(job["state"] == "not_started" and job["charged_cost"] == 0
               for job in final["jobs"] if job["replica_index"] >= 2)


def test_runtime_budget_rejects_first_oversized_force_batch(tmp_path, monkeypatch):
    config, system = configuration()
    directory = tmp_path / "too-small-job"
    plan_precision_resolved(config, system, request(config, system, job_max_cost=1), output=directory)
    with monkeypatch.context() as patch:
        patch.setattr("pyrpmd.potentials.FreePotential.evaluate",
                      lambda *args, **kwargs: pytest.fail("over-budget PES batch was evaluated"))
        status = run_precision(directory)
    assert status["stop"]["status"] == "budget_exhausted"
    assert status["budget"]["charged_cost"] == 0
    assert status["budget"]["spent_cost"] == 0
    assert not ledger(directory)["results"]
    assert any(job["state"] == "failed" for job in status["jobs"])
    assert run_precision(directory)["budget"] == status["budget"]


def test_runtime_budget_retains_partial_work_and_never_admits_an_overrun(tmp_path, monkeypatch):
    config, system = configuration()
    directory = tmp_path / "partial-budget"
    plan_precision_resolved(config, system, request(config, system, job_max_cost=100), output=directory)
    frames = []
    original_evaluate = FreePotential.evaluate

    def counted(provider, q, **kwargs):
        frames.append(len(q))
        assert sum(frames) <= 100, "an over-budget batch reached the provider"
        return original_evaluate(provider, q, **kwargs)

    monkeypatch.setattr(FreePotential, "evaluate", counted)
    status = run_precision(directory)
    assert status["stop"]["status"] == "budget_exhausted"
    assert 0 < sum(frames) <= 100
    assert status["budget"]["charged_cost"] == status["budget"]["spent_cost"] == sum(frames)
    assert status["budget"]["reserved_cost"] == 0
    assert len([job for job in status["jobs"] if job["state"] == "failed"]) == 1
    assert len([job for job in status["jobs"] if job["state"] == "not_started"]) == 1
    assert not ledger(directory)["results"]
    assert status["stop"]["conclusions"]["accepted"] is False
    before = sum(frames)
    assert run_precision(directory)["budget"] == status["budget"]
    assert sum(frames) == before


def test_first_complete_look_must_fit_global_budget_without_launch(tmp_path, monkeypatch):
    config, system = configuration()
    directory = tmp_path / "too-small-campaign"
    with monkeypatch.context() as patch:
        patch.setattr("pyrpmd.potentials.FreePotential.evaluate",
                      lambda *args, **kwargs: pytest.fail("unfunded plan evaluated the PES"))
        status = plan_precision_resolved(
            config, system, request(config, system, max_cost=19_999), output=directory)
        final = run_precision(directory)
    assert status["stop"]["status"] == "budget_exhausted"
    assert final["stop"] == status["stop"]
    assert final["budget"]["charged_cost"] == 0
    assert not ledger(directory)["plans"]
    assert not ledger(directory)["results"]
    assert final["stop"]["conclusions"]["accepted"] is False


@pytest.mark.parametrize("changes", [
    {"max_cost": True}, {"max_cost": -1}, {"max_cost": 1.5},
    {"job_max_cost": True}, {"job_max_cost": 0}, {"job_max_cost": 3.5},
    {"cost_unit": "seconds"}, {"unknown_precision_option": 1},
    {"checks": {"replicas": [2, 2]}}, {"target": {"relative_half_width": 0}},
])
def test_invalid_requests_fail_before_output_or_sampling(tmp_path, changes):
    config, system = configuration()
    directory = tmp_path / "invalid"
    with pytest.raises((ValueError, TypeError)):
        plan_precision_resolved(config, system, request(config, system, **changes), output=directory)
    assert not (directory / "precision" / "ledger.json").exists()


@pytest.mark.parametrize("field,value", [
    ("loop_backend", "native"), ("loop_backend", "auto"),
    ("backend", "process"), ("backend", "mpi"),
])
def test_unmetered_execution_paths_are_rejected(tmp_path, field, value):
    config, system = configuration()
    if field == "loop_backend":
        config["dynamics"][field] = value
    else:
        config["parallel"][field] = value
    with pytest.raises(ValueError):
        plan_precision_resolved(config, system, request(config, system), output=tmp_path / "unsupported")


def test_changed_frozen_request_cannot_reset_budget_or_random_streams(tmp_path):
    config, system = configuration()
    directory = tmp_path / "frozen"
    value = request(config, system)
    plan_precision_resolved(config, system, value, output=directory)
    before = (directory / "precision" / "ledger.json").read_bytes()
    for changed in ({**value, "seed": 42}, {**value, "max_cost": 100_001},
                    {**value, "job_max_cost": 10_001}):
        with pytest.raises(ValueError):
            plan_precision_resolved(config, system, changed, output=directory)
        assert (directory / "precision" / "ledger.json").read_bytes() == before


def test_status_does_not_create_an_unknown_campaign(tmp_path):
    missing = tmp_path / "unknown"
    with pytest.raises((ValueError, FileNotFoundError)):
        precision_status(missing)
    assert not missing.exists()


def test_corrupt_control_file_never_restarts_work_or_accounting(tmp_path, monkeypatch):
    config, system = configuration()
    directory = tmp_path / "corrupt"
    plan_precision_resolved(config, system, request(config, system), output=directory)
    path = directory / "precision" / "workflow.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["config"]["seed"] += 1
    path.write_text(json.dumps(saved), encoding="utf-8")
    before = (directory / "precision" / "ledger.json").read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(FreePotential, "evaluate",
                      lambda *args, **kwargs: pytest.fail("corrupt campaign evaluated the PES"))
        with pytest.raises(ValueError):
            precision_status(directory)
        with pytest.raises(ValueError):
            run_precision(directory)
    assert (directory / "precision" / "ledger.json").read_bytes() == before


def cli(*arguments):
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-m", "pyrpmd", *map(str, arguments)],
                            text=True, capture_output=True, env=environment, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_cli_json_plan_run_status_and_replay(tmp_path):
    config, system = configuration()
    raw = {
        "task": "rate", "temperature": 1 / (5 * KB_AU), "beads": 2,
        "native": False, "seed": 17,
        "system": {"symbols": list(system.symbols), "masses": system.masses.tolist(),
                   "geometry": system.geometry.tolist(), "fragments": [[0], [1, 2]]},
        "potential": config["potential"], "reaction": config["reaction"],
        "dynamics": {"timestep": .003, "loop_backend": "python"},
        "parallel": {"backend": "serial"},
        "sampling": {"equilibration_steps": 8, "production_steps": 40,
                     "interval_steps": 2, "andersen_interval": 4},
        "umbrella": config["umbrella"], "analysis": config["analysis"],
        "recrossing": config["recrossing"], "output": {"checkpoint_steps": 12},
    }
    source = tmp_path / "rate.py"
    source.write_text("import json\n\ndef build():\n    return json.loads(" +
                      repr(json.dumps(raw)) + ")\n", encoding="utf-8")
    specification = tmp_path / "precision.json"
    specification.write_text(json.dumps(request(config, system)), encoding="utf-8")
    directory = tmp_path / "cli run"
    planned = cli("precision", "plan", source, "--spec", specification, "--output", directory)
    assert planned["budget"]["charged_cost"] == 0
    paused = cli("precision", "run", directory, "--max-jobs", "1")
    assert paused["status"] == "paused"
    current = cli("precision", "status", directory)
    assert current["budget"] == paused["budget"]
    completed = cli("precision", "run", directory)
    assert completed["stop"]["status"] == "diagnostic_failed"
    assert completed["latest_assessment"]["accepted"] is False
    assert cli("precision", "run", directory)["budget"] == completed["budget"]


def test_two_cli_executors_share_one_campaign_without_duplicate_launches(tmp_path):
    config, system = configuration()
    directory = tmp_path / "concurrent"
    plan_precision_resolved(config, system, request(config, system), output=directory)
    arguments = [sys.executable, "-m", "pyrpmd", "precision", "run", str(directory),
                 "--max-jobs", "1"]
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    processes = []
    outcomes = []
    try:
        for _ in range(2):
            processes.append(subprocess.Popen(arguments, text=True, stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE, env=environment))
        for process in processes:
            stdout, stderr = process.communicate(timeout=90)
            assert process.returncode == 0, stdout + stderr
            outcomes.append(json.loads(stdout))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=20)
    assert sorted(value["status"] for value in outcomes) == ["diagnostic_failed", "paused"]
    final = precision_status(directory)
    assert len(ledger(directory)["results"]) == 2
    assert len(ledger(directory)["plans"]) == 1
    assert final["budget"]["charged_cost"] == sum(job["charged_cost"] for job in final["jobs"])
    assert all(job["state"] == "completed" for job in final["jobs"])
    assert run_precision(directory)["budget"] == final["budget"]
