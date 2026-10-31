"""Real short dynamics outputs must survive collection and numerical CSV export."""

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.analysis_reports.validate import (
    create_cases, short_fes_config, snapshot_files, validate_report, verify_fes_data,
)
from pyrpmd.cli import main
from pyrpmd.runner import resume_run, run_resolved
from pyrpmd.system import System


@pytest.fixture(scope="module")
def actual_cases(tmp_path_factory):
    pytest.importorskip("matplotlib")
    pytest.importorskip("h5py")
    return create_cases(tmp_path_factory.mktemp("actual-report-workflows"))


def test_actual_workflow_exports_match_archives_and_never_execute_dynamics(actual_cases, tmp_path, monkeypatch):
    import pyrpmd.potentials
    from pyrpmd.dynamics import Dynamics

    def unexpected_dynamics(*args, **kwargs):
        raise AssertionError("Saved-data reporting must not invoke a potential or advance dynamics")

    monkeypatch.setattr(pyrpmd.potentials, "make_potential", unexpected_dynamics)
    monkeypatch.setattr(Dynamics, "step", unexpected_dynamics)
    report, evidence = validate_report(actual_cases, tmp_path / "report", formats=("svg",), style="nature")
    assert evidence["passed"] and evidence["source_files_unchanged"]
    assert evidence["cases"]["fes-2d"]["shape"] == [5, 7]
    assert evidence["cases"]["fes-2d"]["masked_bins"] > 0
    assert evidence["cases"]["adaptive-1d"]["windows"] > 2
    assert evidence["cases"]["toy-rate"]["child_trajectories"] == 4
    assert evidence["cases"]["metadynamics-2d"]["hills_per_replica"] == [8, 8]
    assert evidence["cases"]["adaptive-incomplete"]["production_started"] is False
    saved = json.loads((Path(report["output"]) / "report.json").read_text(),
                       parse_constant=lambda value: pytest.fail(f"Non-standard JSON constant: {value}"))
    assert saved["settings"]["style"] == "nature"
    assert all((Path(report["output"]) / path).is_file() for path in saved["generated_files"])


def test_completed_adaptive_resume_retains_production_samples_for_reporting(actual_cases):
    """A report after resume must describe production once, with frozen centers."""
    from pyrpmd.analysis.reporting import collect_run

    path = actual_cases["adaptive-1d"]["path"]
    before = snapshot_files(path / "checkpoints")
    first = collect_run(path)
    result = resume_run(path)
    after = collect_run(path)
    assert snapshot_files(path / "checkpoints") == before
    assert result["samples"] == actual_cases["adaptive-1d"]["summary"]["samples"]
    assert after["sampling_diagnostics"] == first["sampling_diagnostics"]
    assert after["metrics"] == first["metrics"]


def test_cli_recompute_shortcut_reports_new_grid_without_resampling(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    pytest.importorskip("h5py")
    path, output = tmp_path / "run", tmp_path / "report"
    system = System(("H",), np.array([1.0]), np.zeros((1, 3)))
    run_resolved(short_fes_config(2), system, output=path)
    checkpoints = snapshot_files(path / "checkpoints")
    assert main(["analyze", str(path), "--recompute", "--bins", "7", "4",
                 "--bootstrap", "0", "--report", "--report-output", str(output),
                 "--report-style", "nature"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert "analysis_report" in summary
    assert snapshot_files(path / "checkpoints") == checkpoints
    with np.load(path / "fes.npz", allow_pickle=False) as archive:
        assert archive["free_energy"].shape == (7, 4)
    report = json.loads((output / "report.json").read_text())
    evidence = verify_fes_data({"path": path}, report, 1)
    assert evidence["shape"] == [7, 4]
    assert evidence["csv_matches_archive"]
