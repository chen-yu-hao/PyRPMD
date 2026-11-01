"""Small fake-worker oracles for the scheduling layer; no PES or MC sampling."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.h3 import quantum_sampling_scheduler as scheduler


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fake_arrays(seed, count, warmup=2):
    n = 2
    samples = np.column_stack((np.arange(count, dtype=float), np.full(count, float(seed))))
    diagnostics = np.zeros(((count + 9) // 10, n, 7))
    diagnostics[:, :, 0] = np.arange(len(diagnostics))[:, None]
    return dict(R=np.full((n, 1, 3), count / 10.), r=np.full((n, 1, 3), seed / 10.),
                completed=np.array(warmup + count), samples=samples, diagnostics=diagnostics,
                accepted=np.full((6, n), count, dtype=np.int64),
                proposed=np.full((6, n), count + 1, dtype=np.int64),
                swaps=np.full((2, n - 1), count // 10, dtype=np.int64),
                lineage=np.arange(n), rng=np.array(json.dumps({"fake_draws": count, "seed": seed})))


def write_fake_stage(campaign, protocol_path, seed, stage):
    campaign, protocol_path = Path(campaign), Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    directory = campaign / f"seed-{seed}"
    settings = json.loads((directory / "settings.json").read_text())
    arrays = fake_arrays(seed, stage, settings["warmup"])
    np.savez_compressed(directory / "checkpoint.npz", **arrays)
    destination = directory / "stages" / str(stage)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "state.npz", **{key: arrays[key] for key in scheduler.STATE_KEYS})
    receipt = dict(seed=seed, production_sweeps=stage, warmup_sweeps=settings["warmup"],
                   samples_prefix_sha256=scheduler.array_description(arrays["samples"])["sha256"],
                   diagnostics_prefix_sha256=scheduler.array_description(arrays["diagnostics"])["sha256"],
                   checkpoint_sha256=scheduler.file_hash(directory / "checkpoint.npz"),
                   state_sha256=scheduler.file_hash(destination / "state.npz"),
                   protocol_sha256=scheduler.file_hash(protocol_path),
                   driver_sha256=settings["source_sha256"][scheduler.DRIVER])
    dump(destination / "stage.json", receipt)
    dump(directory / "run-report.json", {"seed": seed, "source_unchanged": True,
        "checkpoint_sha256": receipt["checkpoint_sha256"], "completed_sweeps": settings["warmup"] + stage})
    return receipt


def fake_worker(job):
    """Filesystem events independently expose admission order and draining."""
    campaign, library, protocol, seed, stage = job
    campaign = Path(campaign)
    behavior = json.loads((campaign / "fake-behavior.json").read_text())
    (campaign / f"started-{seed}").write_text(str(time.monotonic()))
    with np.load(campaign / f"seed-{seed}/checkpoint.npz", allow_pickle=False) as saved:
        input_count = len(saved["samples"])
    (campaign / f"input-count-{seed}-{input_count}").write_text("observed existing checkpoint")
    if behavior.get("barrier_first_four") and seed in scheduler.SEEDS[:4]:
        deadline = time.monotonic() + 30
        while sum((campaign / f"started-{s}").exists() for s in scheduler.SEEDS[:4]) < 4:
            if time.monotonic() > deadline:
                raise RuntimeError("fake worker barrier timed out")
            time.sleep(.02)
    if behavior.get("fail_first_once") and seed == scheduler.SEEDS[0] and not (campaign / "failure-consumed").exists():
        (campaign / "failure-consumed").write_text("failed before new state")
        time.sleep(.1)
        raise RuntimeError("intentional fake-worker failure")
    if behavior.get("fail_partial_once") and seed == scheduler.SEEDS[0] and not (campaign / "failure-consumed").exists():
        np.savez_compressed(campaign / f"seed-{seed}/checkpoint.npz", **fake_arrays(seed, 50000))
        (campaign / "failure-consumed").write_text("failed after committed partial progress")
        raise RuntimeError("intentional partial fake-worker failure")
    time.sleep(behavior.get("delay", 0.))
    result = write_fake_stage(campaign, protocol, seed, stage)
    (campaign / f"finished-{seed}").write_text(str(time.monotonic()))
    if behavior.get("corrupt_return") and seed == scheduler.SEEDS[0]:
        result = {**result, "seed": -1}
    return result


@pytest.fixture
def cohort(tmp_path):
    root = tmp_path / "repository"; root.mkdir()
    campaign = tmp_path / "campaign"; campaign.mkdir()
    source_names = [scheduler.DRIVER, "benchmarks/h3/mixing_sampling.py"]
    sources = {}
    for name in source_names:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("immutable fake source " + name)
        sources[name] = scheduler.file_hash(path)
    library = tmp_path / "model.identity"; library.write_text("fake model; never loaded")
    schedule = root / "schedule.json"; dump(schedule, {"centers": [0., 1.], "stiffness": [2., 30.]})
    protocol_path = root / "protocol.json"
    protocol = {"independent_ladder_seeds": list(scheduler.SEEDS), "fixed_refinement_prefixes": list(scheduler.PREFIXES),
                "beads_primary": 1, "warmup_sweeps": 2, "schedule_file": "schedule.json",
                "schedule_sha256": scheduler.file_hash(schedule), "sampler_file": source_names[1],
                "sampler_sha256": sources[source_names[1]], "gates": {"target_qtst_relative_standard_error": .03}}
    dump(protocol_path, protocol)
    rows = []
    for seed in scheduler.SEEDS:
        directory = campaign / f"seed-{seed}"; directory.mkdir()
        settings = {"seed": seed, "beads": 1, "warmup": 2, "production": 300000,
                    "centers": [0., 1.], "stiffness": [2., 30.], "diagnostic_interval": 10,
                    "source_sha256": {**sources, "bkmp2_library": scheduler.file_hash(library)}}
        dump(directory / "settings.json", settings)
        receipt = write_fake_stage(campaign, protocol_path, seed, 30000)
        stage_dir = directory / "stages/30000"
        row = {"seed": seed, "directory": directory.name, "checkpoint": scheduler.record(directory / "checkpoint.npz", campaign),
               "settings": scheduler.record(directory / "settings.json", campaign),
               "run_report": scheduler.record(directory / "run-report.json", campaign),
               "arrays": {name: scheduler.array_description(value) for name, value in fake_arrays(seed, 30000).items()},
               "stage_receipts": [{"production_sweeps": 30000,
                   "receipt": scheduler.record(stage_dir / "stage.json", campaign), "state": scheduler.record(stage_dir / "state.npz", campaign),
                   "samples_prefix_sha256": receipt["samples_prefix_sha256"],
                   "diagnostics_prefix_sha256": receipt["diagnostics_prefix_sha256"]}]}
        rows.append(row)
    snapshot_path = tmp_path / "snapshot.json"
    snapshot = {"schema": scheduler.SNAPSHOT_SCHEMA, "production_sweeps": 30000,
                "independent_seeds": list(scheduler.SEEDS), "warmup_sweeps": 2,
                "protocol": scheduler.record(protocol_path, root), "schedule": scheduler.record(schedule, root),
                "source_sha256": sources, "model_sha256": scheduler.file_hash(library), "seeds": rows}
    dump(snapshot_path, snapshot)
    report = tmp_path / "analysis.json"
    dump(report, {"completed_independent_seeds": list(scheduler.SEEDS), "expected_independent_seeds": list(scheduler.SEEDS),
                  "protocol_sha256": scheduler.file_hash(protocol_path),
                  "production_sweeps_per_ladder": 30000, "statistics": {"primary_log_standard_error": .08},
                  "criteria": {"statistical_precision": False}})
    decision_path = tmp_path / "decision.json"
    decision = {"from_production_sweeps": 30000, "to_production_sweeps": 100000,
                "independent_seeds": list(scheduler.SEEDS), "protocol_sha256": scheduler.file_hash(protocol_path),
                "trigger": {"criterion": "statistical_precision", "passed": False,
                            "observed_primary_log_standard_error": .08, "registered_limit": .03},
                "point_agreement_does_not_trigger_or_stop_continuation": True,
                "completed_cohort_report_path": str(report), "completed_cohort_report_sha256": scheduler.file_hash(report)}
    dump(decision_path, decision)
    dump(campaign / "fake-behavior.json", {})
    return dict(campaign=campaign, library=library, protocol_path=protocol_path,
                snapshot_path=snapshot_path, snapshot_sha=scheduler.file_hash(snapshot_path),
                decision_path=decision_path, decision_sha=scheduler.file_hash(decision_path), stage=100000,
                root=root, worker=fake_worker)


def test_preflight_launches_no_workers_or_changes_prior_files(cohort):
    before = {p: scheduler.file_hash(p) for p in cohort["campaign"].rglob("*") if p.is_file()}
    result = scheduler.run_campaign(**cohort, check_only=True)
    assert result["workers_launched"] == 0 and result["independent_seeds"] == list(scheduler.SEEDS)
    assert before == {p: scheduler.file_hash(p) for p in cohort["campaign"].rglob("*") if p.is_file()}


@pytest.mark.parametrize("target", ["checkpoint", "settings", "state", "source", "model", "diagnostic_prefix"])
def test_tampered_prior_or_source_never_submits(cohort, target):
    directory = cohort["campaign"] / f"seed-{scheduler.SEEDS[0]}"
    if target in ("checkpoint", "diagnostic_prefix"):
        path = directory / "checkpoint.npz"
        with np.load(path, allow_pickle=False) as saved: arrays = {key: saved[key] for key in saved.files}
        arrays["samples" if target == "checkpoint" else "diagnostics"].flat[0] += 1
        np.savez_compressed(path, **arrays)
    else:
        path = {"settings": directory / "settings.json", "state": directory / "stages/30000/state.npz",
                "source": cohort["root"] / "benchmarks/h3/mixing_sampling.py", "model": cohort["library"]}[target]
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError): scheduler.run_campaign(**cohort, check_only=True)
    assert not list(cohort["campaign"].glob("started-*"))


def test_concurrent_coordinator_and_existing_global_manifest_are_rejected(cohort):
    with scheduler.campaign_lock(cohort["campaign"]):
        with pytest.raises(RuntimeError, match="locked"):
            scheduler.run_campaign(**cohort, check_only=True)
    manifest = cohort["campaign"] / "stage-100000.json"; manifest.write_text("existing global result")
    with pytest.raises(ValueError, match="not be overwritten"):
        scheduler.run_campaign(**cohort, check_only=True)
    assert manifest.read_text() == "existing global result"


@pytest.mark.parametrize("field,value", [("workers", 5), ("stage", 300000)])
def test_worker_ceiling_and_skipped_prefix_are_rejected(cohort, field, value):
    args = {**cohort, field: value}
    with pytest.raises(ValueError): scheduler.run_campaign(**args, check_only=True)


def test_four_workers_complete_all_seeds_once_and_preserve_exact_prefixes(cohort):
    old = json.loads(cohort["snapshot_path"].read_text())
    result = scheduler.run_campaign(**cohort, workers=4)
    assert result["completed_seeds"] == list(scheduler.SEEDS)
    assert result["all_submitted_workers_drained"] and result["prior_prefix_and_state_files_unchanged"]
    manifest = cohort["campaign"] / "stage-100000.json"
    assert json.loads(manifest.read_text()) == result
    for row in old["seeds"]:
        for item in row["stage_receipts"]:
            assert scheduler.file_hash(cohort["campaign"] / item["receipt"]["path"]) == item["receipt"]["sha256"]
            assert scheduler.file_hash(cohort["campaign"] / item["state"]["path"]) == item["state"]["sha256"]
        with np.load(cohort["campaign"] / row["checkpoint"]["path"], allow_pickle=False) as saved:
            for key, expected in fake_arrays(row["seed"], 100000).items():
                np.testing.assert_array_equal(saved[key], expected)
    events = [json.loads(line) for line in (Path(result["attempt_directory"]) / "events.jsonl").read_text().splitlines()]
    submitted = [row for row in events if row["event"] == "submitted"]
    assert [row["seed"] for row in submitted] == list(scheduler.SEEDS)
    assert max(row["active"] for row in submitted) <= 4
    digest = scheduler.file_hash(manifest)
    with pytest.raises(ValueError): scheduler.run_campaign(**cohort, workers=4)
    assert scheduler.file_hash(manifest) == digest


def test_failure_stops_admission_drains_and_hash_bound_resume_reuses_completed_seed(cohort):
    campaign = cohort["campaign"]
    dump(campaign / "fake-behavior.json", {"fail_first_once": True, "delay": .4})
    with pytest.raises(scheduler.CampaignFailure) as caught:
        scheduler.run_campaign(**cohort, workers=2)
    failure = caught.value.manifest_path; report = json.loads(failure.read_text())
    assert report["all_submitted_workers_drained"] and not (campaign / "stage-100000.json").exists()
    started = {int(p.name.split("-")[1]) for p in campaign.glob("started-*")}
    assert started <= set(scheduler.SEEDS[:2])
    assert (campaign / f"finished-{scheduler.SEEDS[1]}").exists()
    completed_receipt = campaign / f"seed-{scheduler.SEEDS[1]}/stages/100000/stage.json"
    before = scheduler.file_hash(completed_receipt)
    with pytest.raises(ValueError, match="unrecorded progress"):
        scheduler.run_campaign(**cohort, check_only=True)
    result = scheduler.run_campaign(**cohort, workers=2, resume_path=failure, resume_sha=scheduler.file_hash(failure))
    assert result["completed_seeds"] == list(scheduler.SEEDS)
    assert scheduler.file_hash(completed_receipt) == before


def test_runtime_limit_increase_two_to_four_keeps_seed_order(cohort, tmp_path):
    campaign = cohort["campaign"]
    dump(campaign / "fake-behavior.json", {"barrier_first_four": True})
    control = tmp_path / "workers.json"; dump(control, {"active_workers": 2})
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(scheduler.run_campaign, **cohort, workers=2, control=control)
        deadline = time.monotonic() + 30
        while len(list(campaign.glob("started-*"))) < 2:
            assert time.monotonic() < deadline
            time.sleep(.02)
        assert len(list(campaign.glob("started-*"))) == 2
        replacement = tmp_path / "workers-next.json"; dump(replacement, {"active_workers": 4}); replacement.replace(control)
        result = task.result(timeout=40)
    events = [json.loads(line) for line in (Path(result["attempt_directory"]) / "events.jsonl").read_text().splitlines()]
    assert [row["active_workers"] for row in events if row["event"] == "worker_limit"] == [2, 4]
    assert [row["seed"] for row in events if row["event"] == "submitted"] == list(scheduler.SEEDS)
    assert result["completed_seeds"] == list(scheduler.SEEDS)


@pytest.mark.parametrize("value", [0, 5, True, "4"])
def test_invalid_mutable_worker_control_rejected_before_sampling(cohort, tmp_path, value):
    control = tmp_path / "workers.json"; dump(control, {"active_workers": value})
    with pytest.raises(ValueError, match="worker control"):
        scheduler.run_campaign(**cohort, control=control, check_only=True)


def test_worker_return_corruption_cannot_publish_complete_cohort(cohort):
    dump(cohort["campaign"] / "fake-behavior.json", {"corrupt_return": True})
    with pytest.raises(scheduler.CampaignFailure) as caught:
        scheduler.run_campaign(**cohort, workers=2)
    assert not (cohort["campaign"] / "stage-100000.json").exists()
    assert json.loads(caught.value.manifest_path.read_text())["all_submitted_workers_drained"]


def test_actual_entrypoint_forwards_unchanged_job(monkeypatch):
    from benchmarks.h3 import quantum_sampling_campaign
    job = ("campaign", "library", "protocol", 211017, 100000)
    received = []
    monkeypatch.setattr(quantum_sampling_campaign, "one_seed", lambda value: received.append(value) or {"result": 7})
    assert scheduler._frozen_worker(job) == {"result": 7}
    assert received == [job]


def test_next_300000_stage_uses_complete100000_manifest_and_preserves_both_prefixes(cohort):
    first = scheduler.run_campaign(**cohort, workers=4)
    first_path = cohort["campaign"] / "stage-100000.json"
    first_sha = scheduler.file_hash(first_path)
    report = cohort["decision_path"].with_name("analysis100000.json")
    dump(report, {"completed_independent_seeds": list(scheduler.SEEDS), "expected_independent_seeds": list(scheduler.SEEDS),
                  "protocol_sha256": scheduler.file_hash(cohort["protocol_path"]),
                  "production_sweeps_per_ladder": 100000, "statistics": {"primary_log_standard_error": .04},
                  "criteria": {"statistical_precision": False}})
    decision = json.loads(cohort["decision_path"].read_text())
    decision.update(from_production_sweeps=100000, to_production_sweeps=300000,
                    completed_cohort_report_path=str(report), completed_cohort_report_sha256=scheduler.file_hash(report))
    decision["trigger"]["observed_primary_log_standard_error"] = .04
    decision_path = cohort["decision_path"].with_name("decision300000.json"); dump(decision_path, decision)
    args = {**cohort, "snapshot_path": first_path, "snapshot_sha": first_sha,
            "decision_path": decision_path, "decision_sha": scheduler.file_hash(decision_path), "stage": 300000}
    result = scheduler.run_campaign(**args, workers=4)
    assert result["production"] == 300000 and result["completed_seeds"] == list(scheduler.SEEDS)
    assert scheduler.file_hash(first_path) == first_sha
    for row in result["snapshot"]["seeds"]:
        assert [item["production_sweeps"] for item in row["stage_receipts"]] == [30000, 100000, 300000]
        old = next(item for item in first["snapshot"]["seeds"] if item["seed"] == row["seed"])
        assert row["stage_receipts"][:2] == old["stage_receipts"]


def test_hash_bound_partial_resume_keeps_partial_checkpoint_rng_state(cohort):
    campaign = cohort["campaign"]
    dump(campaign / "fake-behavior.json", {"fail_partial_once": True})
    with pytest.raises(scheduler.CampaignFailure) as caught:
        scheduler.run_campaign(**cohort, workers=1)
    failure = caught.value.manifest_path
    row = json.loads(failure.read_text())["checkpoints_after_drain"][0]
    assert row["arrays"]["samples"]["shape"][0] == 50000
    assert row["arrays"]["rng"] == scheduler.array_description(fake_arrays(scheduler.SEEDS[0], 50000)["rng"])
    result = scheduler.run_campaign(**cohort, workers=2, resume_path=failure, resume_sha=scheduler.file_hash(failure))
    assert result["completed_seeds"] == list(scheduler.SEEDS)
    assert (campaign / f"input-count-{scheduler.SEEDS[0]}-50000").exists()


def test_paper_point_or_incomplete_cohort_cannot_authorize_extension(cohort):
    decision = json.loads(cohort["decision_path"].read_text())
    decision["trigger"]["criterion"] = "paper_point_agreement"
    dump(cohort["decision_path"], decision)
    args = {**cohort, "decision_sha": scheduler.file_hash(cohort["decision_path"])}
    with pytest.raises(ValueError, match="precision-only"):
        scheduler.run_campaign(**args, check_only=True)
    decision["trigger"]["criterion"] = "statistical_precision"
    report_path = Path(decision["completed_cohort_report_path"])
    report = json.loads(report_path.read_text()); report["completed_independent_seeds"] = list(scheduler.SEEDS[:-1])
    dump(report_path, report); decision["completed_cohort_report_sha256"] = scheduler.file_hash(report_path)
    dump(cohort["decision_path"], decision)
    args["decision_sha"] = scheduler.file_hash(cohort["decision_path"])
    with pytest.raises(ValueError, match="evidence mismatch"):
        scheduler.run_campaign(**args, check_only=True)


def test_import_path_must_be_the_guarded_source(monkeypatch, tmp_path):
    expected = scheduler.ROOT / "src/pyrpmd/constants.py"
    other = tmp_path / "constants.py"; other.write_bytes(expected.read_bytes())
    monkeypatch.setattr(scheduler.importlib, "import_module", lambda name: SimpleNamespace(__file__=str(other)))
    with pytest.raises(ValueError, match="imported source differs"):
        scheduler.verify_imported_sources({str(expected): scheduler.file_hash(expected)})


@pytest.mark.parametrize("mutation", ["control", "source"])
def test_runtime_bad_control_or_source_change_stops_admission_and_drains(cohort, tmp_path, mutation):
    campaign = cohort["campaign"]
    dump(campaign / "fake-behavior.json", {"delay": 1.})
    control = tmp_path / "workers.json"; dump(control, {"active_workers": 2})
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(scheduler.run_campaign, **cohort, workers=2, control=control)
        deadline = time.monotonic() + 30
        while len(list(campaign.glob("started-*"))) < 2:
            assert time.monotonic() < deadline
            time.sleep(.02)
        if mutation == "control":
            replacement = tmp_path / "bad-workers.json"; dump(replacement, {"active_workers": 5}); replacement.replace(control)
        else:
            path = cohort["root"] / "benchmarks/h3/mixing_sampling.py"
            path.write_bytes(path.read_bytes() + b" changed while fake workers run")
        with pytest.raises(scheduler.CampaignFailure) as caught:
            task.result(timeout=30)
    report = json.loads(caught.value.manifest_path.read_text())
    assert report["all_submitted_workers_drained"]
    assert len(list(campaign.glob("started-*"))) == 2
    assert len(list(campaign.glob("finished-*"))) == 2
    assert not (campaign / "stage-100000.json").exists()
