"""Report regressions for unavailable diagnostics and physical unit conversion."""

import hashlib
import json

import numpy as np
import pytest

from pyrpmd.analysis.reporting import collect_run, generate_report
from pyrpmd.checkpoint import atomic_savez
from pyrpmd.cli import main
from pyrpmd.constants import BOHR_ANGSTROM, FS_TO_AU


def saved_fes(path, *, dimension=2, **changes):
    path.mkdir(exist_ok=True)
    shape = (2,) if dimension == 1 else (2, 3)
    (path / "summary.json").write_text(json.dumps({
        "task": "fes", "units": ["bohr"] * dimension, "converged": False,
    }))
    data = dict(free_energy=np.arange(np.prod(shape), dtype=float).reshape(shape),
                mask=np.zeros(shape, dtype=bool), standard_error=np.ones(shape),
                raw_count=np.ones(shape, dtype=int), overlap=np.eye(2),
                effective_samples_per_stream=[8., 9.], edges_0=[0., 1., 3.])
    if dimension == 2:
        data["edges_1"] = [0., 1., 2., 5.]
    data.update(changes)
    atomic_savez(path / "fes.npz", **data)
    return path


def by_suffix(run, suffix):
    return next(plot for plot in run["_plots"] if plot["id"].endswith(suffix))


@pytest.mark.parametrize("dimension", [1, 2])
def test_fes_uncertainty_mask_wins_over_finite_saved_error(tmp_path, dimension):
    shape = (2,) if dimension == 1 else (2, 3)
    unavailable = np.zeros(shape, dtype=bool)
    unavailable.flat[1] = True
    path = saved_fes(tmp_path, dimension=dimension, standard_error_mask=unavailable)
    run = collect_run(path, energy_unit="hartree")
    if dimension == 1:
        curve = by_suffix(run, "-fes")["series"][0]
        assert np.isfinite(curve["y"]).all()
        assert curve["yerr"][0] == 1.0 and np.isnan(curve["yerr"][1])
    else:
        assert not by_suffix(run, "-fes")["mask"][0, 1]
        assert by_suffix(run, "-fes-error")["mask"][0, 1]
        assert np.isnan(by_suffix(run, "-fes-error")["z"][0, 1])


@pytest.mark.parametrize("changes, missing, retained", [
    ({"standard_error": [1., 2.]}, "FES uncertainty", "-fes"),
    ({"standard_error_mask": [False, True]}, "FES uncertainty", "-fes"),
    ({"raw_count": [[1., np.nan, 1.], [1., 1., 1.]]}, "FES sampling coverage", "-fes-overlap"),
    ({"overlap": [[1., np.nan], [0., 1.]]}, "MBAR overlap", "-fes-ess"),
    ({"overlap": [[1., -0.1], [0., 1.]]}, "MBAR overlap", "-fes-counts"),
    ({"overlap": [[1., 2.], [0., 1.]]}, "MBAR overlap", "-fes-counts"),
    ({"overlap": []}, "MBAR overlap", "-fes-ess"),
    ({"effective_samples_per_stream": [-1., 9.]}, "Effective sample counts", "-fes-overlap"),
    ({"effective_samples_per_stream": [np.nan, np.nan]}, "Effective sample counts", "-fes-counts"),
])
def test_bad_optional_fes_diagnostic_keeps_valid_figures(tmp_path, changes, missing, retained):
    run = collect_run(saved_fes(tmp_path, **changes), energy_unit="hartree")
    assert by_suffix(run, retained)
    np.testing.assert_array_equal(by_suffix(run, "-fes")["z"], np.arange(6).reshape(2, 3))
    assert any(item["figure"] == missing for item in run["skipped"])
    assert any(item["code"] == "artifact_unavailable" for item in run["findings"])
    if missing == "FES uncertainty":
        assert not any(plot["id"].endswith("-fes-error") for plot in run["_plots"])


def test_asymmetric_overlap_retains_stream_row_column_orientation(tmp_path):
    overlap = [[.8, .2], [.1, .9]]
    run = collect_run(saved_fes(tmp_path, overlap=overlap))
    plot = by_suffix(run, "-fes-overlap")
    assert plot["xlabel"] == "Sampling stream j" and plot["ylabel"] == "Sampling stream i"
    # Heatmaps use z[x, y]: O(i=0,j=1)=0.2 belongs at x=1, y=0.
    assert plot["z"][1, 0] == .2
    assert plot["z"][0, 1] == .1


@pytest.mark.parametrize("unit, factor", [("bohr", 1.), ("nm", BOHR_ANGSTROM / 10)])
def test_report_length_unit_reaches_distance_and_cross_section_intervals(tmp_path, unit, factor):
    (tmp_path / "trajectories").mkdir()
    summary = {
        "task": "drpmd", "sampling": "fixed_energy", "trajectories": 4,
        "counts": {"product": 1, "nonreactive": 3, "unresolved": 0, "failed": 0},
        "all_trajectories_resolved": True,
        "channels": {"exchange": {"cross_section_bohr2": 4., "parent_cluster_bootstrap95": [1., 9.]}},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    atomic_savez(tmp_path / "trajectories" / "trajectory_0000.npz",
                 time_au=[0., FS_TO_AU], energy_per_bead=[0., 0.],
                 centroid=[[[0., 0., 0.], [2., 0., 0.]], [[0., 0., 0.], [3., 0., 0.]]])
    run = collect_run(tmp_path, length_unit=unit)
    distance = by_suffix(run, "-centroid-distances-1")
    assert f"({unit})" in distance["ylabel"]
    np.testing.assert_allclose(distance["series"][0]["y"], np.array([2., 3.]) * factor)
    area = by_suffix(run, "-channel-estimates")
    assert f"({unit}²)" in area["ylabel"]
    np.testing.assert_allclose(area["y"], [4. * factor**2])
    np.testing.assert_allclose(area["yerr"], [[3. * factor**2], [5. * factor**2]])
    # Saved provenance quantities retain their explicitly named atomic units.
    assert run["metrics"]["channels"]["exchange"]["cross_section_bohr2"] == 4.


def test_length_unit_reaches_gyration_msd_and_vacf(tmp_path):
    (tmp_path / "samples").mkdir()
    (tmp_path / "summary.json").write_text(json.dumps({
        "task": "diffusion", "diffusion": {
            "lag_time_fs": [0., 1.], "msd_au": [0., 4.], "vacf_au": [1., .5],
            "diffusion_m2_s": None, "status": "not_diffusive",
        },
    }))
    atomic_savez(tmp_path / "samples" / "materials_0000.npz",
                 times=[0., FS_TO_AU], observables=np.zeros((2, 5)),
                 radii_gyration_squared=[[1., 9.], [4., 16.]])
    run = collect_run(tmp_path, length_unit="nm")
    factor = BOHR_ANGSTROM / 10
    np.testing.assert_allclose(by_suffix(run, "-gyration")["series"][0]["y"], np.sqrt([5., 10.]) * factor)
    np.testing.assert_allclose(by_suffix(run, "-msd")["series"][0]["y"], np.array([0., 4.]) * factor**2)
    np.testing.assert_allclose(by_suffix(run, "-vacf")["series"][0]["y"], np.array([1., .5]) * (factor * FS_TO_AU)**2)
    assert "nm²" in by_suffix(run, "-vacf")["ylabel"]
    assert run["metrics"]["diffusion"]["diffusion_m2_s"] is None


@pytest.mark.parametrize("command", ["run", "analyze"])
def test_report_style_requires_report_before_any_run_is_started(command, tmp_path, capsys):
    missing = tmp_path / "not-started"
    assert main([command, str(missing), "--report-style", "nature"]) == 2
    assert "require --report" in capsys.readouterr().err
    assert not missing.exists()


def test_invalid_style_does_not_publish_a_report(tmp_path):
    with pytest.raises(ValueError, match="Figure style"):
        generate_report(tmp_path / "not-read", tmp_path / "report", style="unsupported")
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("task, folder_name", [("fes", ""), ("rate", "umbrella"), ("site_rate", "pmf")])
@pytest.mark.parametrize("saved_fes_state", ["missing", "corrupt", "valid"])
def test_adaptive_progress_survives_missing_summary_and_bad_fes(tmp_path, task, folder_name, saved_fes_state):
    (tmp_path / "input").mkdir()
    (tmp_path / "input/resolved.json").write_text(json.dumps({"task": task}))
    folder = tmp_path / folder_name
    folder.mkdir(exist_ok=True)
    plan = {"schema_version": 2, "status": "adapting", "rounds": [
        {"round": 0, "centers": [[0.]], "effective_samples_per_window": [2.],
         "spanning_overlap": 1.},
    ]}
    plan["checksum"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (folder / "adaptive-plan.json").write_text(json.dumps(plan))
    if saved_fes_state == "corrupt":
        (folder / "fes.npz").write_bytes(b"interrupted archive write")
    elif saved_fes_state == "valid":
        atomic_savez(folder / "fes.npz", free_energy=[0., 1.], edges_0=[0., 1., 2.], cv_units=["bohr"])
    run = collect_run(tmp_path)
    progress = [plot for plot in run["_plots"] if "-adaptive-" in plot["id"]]
    assert len(progress) == 3 and len({plot["id"] for plot in progress}) == 3
    assert run["status"] == "summary_unavailable" and run["converged"] is None
    assert any(item["code"] == "summary_unavailable" for item in run["findings"])
    if saved_fes_state == "corrupt":
        assert any("archive" in item["reason"].lower() or "pickle" in item["reason"].lower()
                   or "numpy" in item["reason"].lower() for item in run["skipped"])


def test_partial_archive_discovers_both_nested_plans_without_guessing_task(tmp_path):
    for name in ("umbrella", "pmf"):
        folder = tmp_path / name
        folder.mkdir()
        plan = {"schema_version": 2, "status": "adapting", "rounds": [
            {"round": 0, "centers": [[0.]], "effective_samples_per_window": [2.],
             "spanning_overlap": 1.},
        ]}
        plan["checksum"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        (folder / "adaptive-plan.json").write_text(json.dumps(plan))
    run = collect_run(tmp_path)
    assert run["task"] == "unknown" and run["converged"] is None
    plots = [plot for plot in run["_plots"] if "-adaptive-" in plot["id"]]
    assert len(plots) == 6 and len({plot["id"] for plot in plots}) == 6
    assert set(run["sampling_diagnostics"]) == {"umbrella", "pmf"}
