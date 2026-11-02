"""Offline diagnostics for stable umbrella IDs and window-local sampling."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.analysis.report_sampling import collect_sampling
from pyrpmd.checkpoint import atomic_savez
from pyrpmd.dynamics import Dynamics
from pyrpmd.system import System
from pyrpmd.workflows import fes


def _run(path, config, *, task="fes", summary=None):
    return {
        "id": "test", "label": "saved windows", "path": str(path), "task": task,
        "_config": config, "_summary": summary or {}, "_plots": [], "_sources": {},
        "metrics": {}, "findings": [], "skipped": [],
    }


def _config():
    return {
        "schema_version": 2, "task": "fes", "beta": 1.7,
        "beads": 3, "dt_au": .03, "seed": 903, "native": False,
        "potential": {"provider": "harmonic", "k": .4},
        "sampling": {"equilibration_steps": 6, "production_steps": 40,
                     "sample_interval": 2, "andersen_interval": 3, "replicas": 1},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
        "umbrella": {"coordinate": "x", "force_constant": "1 hartree/bohr^2",
                     "windows": [
                         {"id": "zeta", "center": -.2},
                         {"id": "alpha", "center": .2,
                          "force_constant": "3 hartree/bohr^2",
                          "sampling": {"equilibration": {"steps": 9},
                                       "production": {"steps": 57},
                                       "sample_every": {"steps": 3},
                                       "chains_per_window": 2}},
                     ]},
        "analysis": {"bins": 5, "bootstrap": 0},
        "output": {"checkpoint_steps": 7},
    }


def _forbid_factories(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline report constructed a CV, PES, or dynamics object")

    monkeypatch.setattr(Dynamics, "__init__", forbidden)
    monkeypatch.setattr("pyrpmd.cv.make_cvs", forbidden)
    monkeypatch.setattr("pyrpmd.potentials.make_potential", forbidden)
    monkeypatch.setattr(fes, "resolve_window_plan", forbidden)


@pytest.mark.parametrize("keep_plan", [True, False])
def test_real_hashed_checkpoints_report_all_windows_without_hdf5_or_factories(tmp_path, monkeypatch, keep_plan):
    config = _config()
    system = System(("H",), np.array([2.3]), np.zeros((1, 3)))
    result = fes.run_fes(config, system, tmp_path, return_data=True)
    (tmp_path / "fes.h5").unlink(missing_ok=True)
    if not keep_plan:
        (tmp_path / "umbrella-plan.json").unlink()
    _forbid_factories(monkeypatch)
    config["cvs"] = [{"name": "x", "kind": "python", "factory": "never.import.this"}]
    run = _run(tmp_path, config)
    collect_sampling(run, tmp_path, result["summary"], [("x (bohr)", 1.)], "hartree")

    assert not run["skipped"]
    records = run["sampling_diagnostics"]["sampling"]
    assert len(records["streams"]) == 3
    rows = {row["window_id"]: row for row in records["windows"]}
    assert rows["zeta"]["center_atomic"] == [-.2]
    assert rows["alpha"]["center_atomic"] == [.2]
    assert rows["zeta"]["stiffness_atomic"] == [[1.]]
    assert rows["alpha"]["stiffness_atomic"] == [[3.]]
    assert rows["zeta"]["samples"] == 20
    assert rows["alpha"]["samples"] == 38
    assert rows["alpha"]["replicas"] == 2
    assert sum(row["samples"] for row in rows.values()) == sum(len(stream["samples"]) for stream in result["streams"])
    assert rows["zeta"]["window"] == (0 if keep_plan else 1)
    assert rows["alpha"]["window"] == (1 if keep_plan else 0)
    assert any("-stiffness-" in plot["id"] for plot in run["_plots"])
    assert any("-trace-" in plot["id"] for plot in run["_plots"])
    if keep_plan:
        assert str((tmp_path / "umbrella-plan.json").resolve()) in run["_sources"]
    else:
        assert any(item["code"] == "umbrella_plan_missing" for item in run["findings"])


def test_hashed_chunked_checkpoints_retain_committed_sample_history(tmp_path, monkeypatch):
    config = _config()
    config["umbrella"]["windows"][1].pop("sampling")
    system = System(("H",), np.array([2.3]), np.zeros((1, 3)))
    result = fes.run_fes(config, system, tmp_path, return_data=True, production_steps=61)
    (tmp_path / "fes.h5").unlink(missing_ok=True)
    _forbid_factories(monkeypatch)
    run = _run(tmp_path, config)
    collect_sampling(run, tmp_path, result["summary"], [("x (bohr)", 1.)], "hartree")
    rows = run["sampling_diagnostics"]["sampling"]["windows"]
    assert [row["samples"] for row in rows] == [30, 30]
    assert any(".chunks/" in Path(path).as_posix() for path in run["_sources"])
    assert not run["skipped"]


def _saved_rate(path, *, effective_production_steps=None):
    rows = [
        {"index": 0, "id": "global", "center_au": [0.], "stiffness_au": [[1.]],
         "chains": 1, "equilibration_steps": 2, "production_steps": 8,
         "sample_stride_steps": 2, "raw_samples_per_chain": 4},
        {"index": 1, "id": "local", "center_au": [1.], "stiffness_au": [[3.]],
         "chains": 1, "equilibration_steps": 5, "production_steps": 12,
         "sample_stride_steps": 3, "raw_samples_per_chain": 4},
    ]
    plan = {"dimension": 1, "cv_units": ["dimensionless"], "windows": rows}
    (path / "umbrella-plan.json").write_text(json.dumps(plan))
    checkpoints = path / "checkpoints"
    checkpoints.mkdir()
    files = {}
    for row in rows:
        production = effective_production_steps or row["production_steps"]
        count = production // row["sample_stride_steps"]
        file = checkpoints / f"window_{hashlib.sha256(row['id'].encode()).hexdigest()[:24]}_replica_0000.npz"
        atomic_savez(file, window_id=row["id"], center=row["center_au"], stiffness=row["stiffness_au"],
                     values=np.linspace(0., 1., count)[:, None],
                     elapsed=row["equilibration_steps"] + production)
        files[row["id"]] = file
    config = {
        "task": "rate", "sampling": {"equilibration_steps": 2, "production_steps": 8, "sample_interval": 2},
        "umbrella": {"windows": [
            {"id": "global", "center": 0.},
            {"id": "local", "center": 1., "sampling": {"production": {"steps": 12}}},
        ]},
    }
    summary = {"effective_production_steps": effective_production_steps} if effective_production_steps else {}
    return _run(path, config, task="rate", summary=summary), files, plan


@pytest.mark.parametrize("effective_production_steps", [None, 20])
def test_rate_report_uses_each_saved_window_clock_and_published_extension_budget(tmp_path, monkeypatch, effective_production_steps):
    run, _, _ = _saved_rate(tmp_path, effective_production_steps=effective_production_steps)
    _forbid_factories(monkeypatch)
    collect_sampling(run, tmp_path, {"task": "fes", "units": ["dimensionless"]},
                     [("xi (dimensionless)", 1.)], "hartree")
    rows = run["sampling_diagnostics"]["sampling"]["windows"]
    assert [row["samples"] for row in rows] == ([4, 4] if effective_production_steps is None else [10, 6])
    assert not run["skipped"]


def test_rate_report_never_substitutes_global_clocks_for_missing_local_plan(tmp_path):
    run, _, _ = _saved_rate(tmp_path)
    (tmp_path / "umbrella-plan.json").unlink()
    collect_sampling(run, tmp_path, {"task": "fes", "units": ["dimensionless"]},
                     [("xi (dimensionless)", 1.)], "hartree")
    assert not run["_plots"]
    assert any("Per-window rate sampling clocks" in item["reason"] for item in run["skipped"])


@pytest.mark.parametrize("corruption,reason", [
    ("window_id", "does not match its checkpoint filename"),
    ("center", "contradicts its plan"),
    ("clock", "advanced beyond the published rate result"),
    ("samples", "sample count differs from its checkpoint clock"),
])
def test_bad_saved_window_is_withheld_without_dropping_independent_valid_stream(tmp_path, corruption, reason):
    run, files, _ = _saved_rate(tmp_path)
    values = np.linspace(0., 1., 5 if corruption == "samples" else 4)[:, None]
    atomic_savez(files["local"], window_id="foreign" if corruption == "window_id" else "local",
                 center=[2.] if corruption == "center" else [1.], stiffness=[[3.]], values=values,
                 elapsed=20 if corruption == "clock" else 17)
    collect_sampling(run, tmp_path, {"task": "fes", "units": ["dimensionless"]},
                     [("xi (dimensionless)", 1.)], "hartree")
    rows = run["sampling_diagnostics"]["sampling"]["windows"]
    assert [row["window_id"] for row in rows] == ["global"]
    assert any(reason in item["reason"] for item in run["skipped"])


@pytest.mark.parametrize("corruption", ["duplicate_id", "units", "stride"])
def test_invalid_saved_window_plan_is_not_silently_replaced_by_input_guesses(tmp_path, corruption):
    run, _, plan = _saved_rate(tmp_path)
    if corruption == "duplicate_id":
        plan["windows"][1]["id"] = "global"
    elif corruption == "units":
        plan["cv_units"] = ["bohr"]
    else:
        plan["windows"][1]["sample_stride_steps"] = True
    (tmp_path / "umbrella-plan.json").write_text(json.dumps(plan))
    collect_sampling(run, tmp_path, {"task": "fes", "units": ["dimensionless"]},
                     [("xi (dimensionless)", 1.)], "hartree")
    assert not run["_plots"]
    assert any(item["figure"] == "saved umbrella window plan" for item in run["skipped"])


@pytest.mark.parametrize("plan_state", ["frozen", "missing", "corrupted", "not_frozen"])
def test_hashed_adaptive_checkpoints_still_require_checked_frozen_metadata(tmp_path, plan_state):
    run, files, plan = _saved_rate(tmp_path)
    run["task"] = "fes"
    run["_config"]["adaptive_umbrella"] = {"enabled": True}
    # Adaptive production uses one common stiffness, even with stable IDs.
    plan["windows"][1]["stiffness_au"] = [[1.]]
    (tmp_path / "umbrella-plan.json").write_text(json.dumps(plan))
    atomic_savez(files["local"], window_id="local", center=[1.], stiffness=[[1.]],
                 values=[[0.], [.3], [.6], [1.]], elapsed=17)
    if plan_state != "missing":
        adaptive = {
            "schema_version": 1, "status": "adapting" if plan_state == "not_frozen" else "frozen",
            "centers": [[0.], [1.]], "stiffness": [[1.]], "rounds": [],
        }
        adaptive["checksum"] = hashlib.sha256(json.dumps(
            adaptive, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        if plan_state == "corrupted":
            adaptive["centers"][1][0] = 2.
        (tmp_path / "adaptive-plan.json").write_text(json.dumps(adaptive))
    collect_sampling(run, tmp_path, {"task": "fes", "units": ["dimensionless"]},
                     [("xi (dimensionless)", 1.)], "hartree")
    rows = run["sampling_diagnostics"]["sampling"]["windows"]
    assert [row["samples"] for row in rows] == [4, 4]
    if plan_state == "frozen":
        assert [row["center_atomic"] for row in rows] == [[0.], [1.]]
        assert [row["stiffness_atomic"] for row in rows] == [[[1.]], [[1.]]]
    else:
        assert all(row["center_atomic"] is None for row in rows)
        assert all(row["stiffness_atomic"] is None for row in rows)
        assert any(item["code"] == "umbrella_metadata_missing" for item in run["findings"])
