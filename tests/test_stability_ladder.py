"""Opt-in preflight ladders stop on every incomplete or failed history."""
from copy import deepcopy
import json

import pytest

from benchmarks.h3 import stability_ladder as ladder
from pyrpmd.input import InputError, JobSpec


def specification(method="metadynamics"):
    return {
        "task": "fes", "fes": {"method": method}, "temperature": "300 K", "beads": 2, "native": False,
        "system": {"symbols": ["H", "H"], "geometry": [[0., 0., 0.], [2., 0., 0.]]},
        "potential": {"provider": "free"}, "dynamics": {"timestep": "0.01 fs"},
        "sampling": {"equilibration_steps": 2, "production_steps": 12,
                     "sample_interval": 2, "andersen_interval": 2, "replicas": 2},
        "cvs": [{"name": "r", "kind": "distance", "atoms": [0, 1]}],
        method: {"sigma": .1, "pace": 2, "bias_factor": 8.,
                 "freeze_after_steps": 4, "freeze_equilibration_steps": 2,
                 **({"height": ".001 kcal/mol"} if method == "metadynamics"
                    else {"barrier": ".1 kcal/mol"})},
        "analysis": {"method": "reweighted_histogram", "bins": 4, "bootstrap": 0},
        "output": {"checkpoint_steps": 2},
    }


def mock_result(config, output):
    total = config["sampling"]["equilibration_steps"] + config["sampling"]["production_steps"]
    directory = output / "diagnostics"
    directory.mkdir(exist_ok=True)
    streams = []
    for replica in range(config["sampling"]["replicas"]):
        path = directory / f"{ladder.fes_method(config)}_{replica:04d}.pair-distances.jsonl"
        records = [{"start_step": max(0, step - 1), "end_step": step,
                    "minimum_bead_pair_distance": 1.2, "finite": True} for step in range(total + 1)]
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        streams.append({"replica": replica, "elapsed": total, "complete": True})
    return {"streams": streams}


@pytest.mark.parametrize("method", ["metadynamics", "opes"])
def test_ladder_preserves_freeze_contract_and_halves_time_step_without_mutating_input(tmp_path, monkeypatch, method):
    config, system = JobSpec.from_mapping(specification(method)).resolve()
    before = deepcopy(config)
    calls = []
    def run(stage, model, output):
        calls.append(deepcopy(stage))
        return mock_result(stage, output)
    monkeypatch.setattr(ladder, "_run_stage", run)
    report = ladder.run_ladder(config, system, tmp_path / "ladder", factors=[.1, 1.], production_steps=[12, 16])
    assert config == before
    assert report["passed"] and not report["production_started"]
    assert len(calls) == 3
    for stage in calls[:2]:
        assert stage[method]["freeze_after_steps"] == 4
        assert stage[method]["freeze_equilibration_steps"] == 2
        assert stage["sampling"]["monitor_pair_distances"] is True
    coarse, fine = calls[-2:]
    assert fine["dt_au"] == coarse["dt_au"] / 2
    assert fine["sampling"]["dt_au"] == coarse["sampling"]["dt_au"] / 2
    for key in ("production_steps", "equilibration_steps", "sample_interval", "andersen_interval"):
        assert fine["sampling"][key] == 2 * coarse["sampling"][key]
    for key in ("pace", "freeze_after_steps", "freeze_equilibration_steps"):
        assert fine[method][key] == 2 * coarse[method][key]
    assert all(len(stage["walkers"]) == 2 for stage in report["stages"])


@pytest.mark.parametrize("failure", ["exception", "missing_walker", "partial", "failure_record", "domain", "gap", "nan"])
def test_failed_or_incomplete_history_blocks_every_later_stage(tmp_path, monkeypatch, failure):
    config, system = JobSpec.from_mapping(specification()).resolve()
    calls = []
    def run(stage, model, output):
        calls.append(output)
        result = mock_result(stage, output)
        if len(calls) != 2:
            return result
        if failure == "exception":
            raise RuntimeError("PES domain failure")
        if failure == "missing_walker":
            result["streams"].pop()
        elif failure == "partial":
            result["streams"][1]["complete"] = False
        elif failure == "failure_record":
            (output / "failures").mkdir()
            (output / "failures/walker.json").write_text('{"status":"history_failed"}')
        else:
            path = output / "diagnostics/metadynamics_0001.pair-distances.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            if failure == "gap":
                del records[3]
            elif failure == "domain":
                records[3]["minimum_bead_pair_distance"] = .19
            else:
                records[3]["minimum_bead_pair_distance"] = None
                records[3]["finite"] = False
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return result
    monkeypatch.setattr(ladder, "_run_stage", run)
    output = tmp_path / "ladder"
    with pytest.raises((ValueError, RuntimeError)):
        ladder.run_ladder(config, system, output, factors=[.1, .3, 1.], production_steps=[12, 16, 20])
    assert len(calls) == 2
    report = json.loads((output / "ladder.json").read_text())
    assert not report["passed"] and report["failed_stage"] == "level-02"
    assert report["stages"][0]["status"] == "passed"
    assert report["stages"][1]["status"] == "failed"
    assert not (output / "level-03").exists()


def test_insufficient_frozen_budget_rejected_before_output_or_sampling(tmp_path, monkeypatch):
    config, system = JobSpec.from_mapping(specification()).resolve()
    monkeypatch.setattr(ladder, "_run_stage", lambda *args: pytest.fail("must not start trajectories"))
    output = tmp_path / "ladder"
    with pytest.raises(ValueError, match="frozen-bias samples"):
        ladder.run_ladder(config, system, output, factors=[.1, 1.], production_steps=[6, 16])
    assert not output.exists()


@pytest.mark.parametrize("value", ["true", 1, None])
def test_pair_monitor_requires_boolean_input(value):
    spec = specification()
    spec["sampling"]["monitor_pair_distances"] = value
    with pytest.raises(InputError, match="must be boolean"):
        JobSpec.from_mapping(spec).resolve()


@pytest.mark.parametrize("method", ["metadynamics", "opes"])
def test_pair_monitor_is_accepted_for_both_independent_workflows(method):
    spec = specification(method)
    spec["sampling"]["monitor_pair_distances"] = True
    config, _ = JobSpec.from_mapping(spec).resolve()
    assert config["sampling"]["monitor_pair_distances"] is True


def test_shared_histories_cannot_silently_ignore_pair_monitor():
    spec = specification()
    spec["sampling"]["monitor_pair_distances"] = True
    spec["metadynamics"]["multiple_walkers"] = True
    with pytest.raises(InputError, match="independent histories"):
        JobSpec.from_mapping(spec).resolve()


@pytest.mark.parametrize("method", ["metadynamics", "opes"])
def test_real_short_workflows_publish_every_accepted_step(tmp_path, method):
    config, system = JobSpec.from_mapping(specification(method)).resolve()
    report = ladder.run_ladder(config, system, tmp_path / method,
                               factors=[1.], production_steps=[12])
    assert report["passed"]
    for index, stage in enumerate(report["stages"]):
        expected = 14 if index == 0 else 28
        assert [w["accepted_steps"] for w in stage["walkers"]] == [expected, expected]
        assert all(w["minimum_bead_pair_distance"] >= .2 for w in stage["walkers"])
