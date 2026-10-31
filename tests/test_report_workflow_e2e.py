"""Real short dynamics outputs must survive collection and numerical CSV export."""

import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from benchmarks.analysis_reports.validate import (
    create_cases, short_fes_config, snapshot_files, validate_report, verify_fes_data,
    figure, verify_metadynamics_data, verify_rate_data,
)
from pyrpmd.cli import main
from pyrpmd.runner import resume_run, run_resolved
from pyrpmd.system import System


@pytest.fixture(scope="module")
def actual_cases(tmp_path_factory):
    pytest.importorskip("matplotlib")
    pytest.importorskip("h5py")
    # An ancestor directory called adaptation is not a run's discarded pilots.
    return create_cases(tmp_path_factory.mktemp("actual-report-workflows") / "adaptation" / "runs")


@pytest.fixture(scope="module")
def actual_report(actual_cases, tmp_path_factory):
    import pyrpmd.potentials
    from pyrpmd.dynamics import Dynamics

    def unexpected_dynamics(*args, **kwargs):
        raise AssertionError("Saved-data reporting must not invoke a potential or advance dynamics")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pyrpmd.potentials, "make_potential", unexpected_dynamics)
        patch.setattr(Dynamics, "step", unexpected_dynamics)
        return validate_report(actual_cases, tmp_path_factory.mktemp("actual-report") / "report",
                               formats=("svg",), style="nature")


def test_actual_workflow_exports_match_archives_and_never_execute_dynamics(actual_report):
    report, evidence = actual_report
    assert evidence["passed"] and evidence["source_files_unchanged"]
    assert evidence["cases"]["fes-2d"]["shape"] == [5, 7]
    assert evidence["cases"]["fes-2d"]["masked_bins"] > 0
    assert evidence["cases"]["adaptive-1d"]["windows"] > 2
    assert evidence["cases"]["toy-rate"]["child_trajectories"] == 4
    assert evidence["cases"]["toy-rate"]["pmf_csv_matches_archive"]
    assert evidence["cases"]["toy-rate"]["thermal_rate_withheld"]
    assert evidence["cases"]["toy-rate"]["rate_uncertainty_available"] is False
    assert evidence["cases"]["metadynamics-2d"]["hills_per_replica"] == [8, 8]
    assert evidence["cases"]["metadynamics-2d"]["replica_labels_preserved"]
    assert evidence["cases"]["adaptive-incomplete"]["production_started"] is False
    saved = json.loads((Path(report["output"]) / "report.json").read_text(),
                       parse_constant=lambda value: pytest.fail(f"Non-standard JSON constant: {value}"))
    assert saved["settings"]["style"] == "nature"
    assert all((Path(report["output"]) / path).is_file() for path in saved["generated_files"])


@pytest.mark.parametrize("case_name,suffix,verify,corruption", [
    ("fes-2d", "fes", verify_fes_data, "precision"),
    ("toy-rate", "pmf", verify_rate_data, "pmf_value"),
    ("toy-rate", "pmf", verify_rate_data, "fabricated_uncertainty"),
    ("metadynamics-2d", "sampling-hill-height", verify_metadynamics_data, "replica_label"),
])
def test_numerical_oracles_reject_changed_exports(actual_cases, actual_report, tmp_path,
                                                  case_name, suffix, verify, corruption):
    """Faults in exported data must not produce a successful validation record."""
    report, _ = actual_report
    run_index = list(actual_cases).index(case_name) + 1
    for item in report["figures"]:
        target = tmp_path / item["data"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(report["output"]) / item["data"], target)
    copied = {**report, "output": str(tmp_path)}
    data = tmp_path / figure(copied, run_index, suffix)["data"]
    with data.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    if corruption == "replica_label":
        rows[-1]["series"] = rows[0]["series"]
    elif corruption == "fabricated_uncertainty":
        assert rows[0]["error_minus"] == ""
        rows[0]["error_minus"] = "0"
    else:
        key = "z" if corruption == "precision" else "y"
        row = next(row for row in rows if row[key] and float(row[key]) != 0)
        value = float(row[key])
        # This passes numpy's default 1e-7 relative tolerance, but it loses
        # scientifically reusable CSV precision and must fail our tight oracle.
        row[key] = str(value * (1 + 5e-8))
    with data.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(AssertionError):
        verify(actual_cases[case_name], copied, run_index)


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
