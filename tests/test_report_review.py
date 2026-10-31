"""Regression cases from independent review of real workflow schemas."""

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from pyrpmd.analysis.report_compare import collect_comparisons
from pyrpmd.analysis.reporting import collect_run, generate_report
from pyrpmd.checkpoint import atomic_savez
from pyrpmd.constants import BOHR_ANGSTROM


def site_run(identifier, temperature=300):
    return {
        "id": identifier, "label": identifier, "task": "site_rate", "_plots": [],
        "_config": {"task": "site_rate", "potential": {"provider": "builtin", "name": "periodic_cosine"},
                    "site_rate": {"cv": "x", "basin": [-.5, .5], "dividing_surface": .5, "direction": 1},
                    "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": 0}],
                    "umbrella": {"cvs": ["x"]}},
        "_system": {"symbols": ["H"], "masses": [1837.], "geometry": [[0., 0., 0.]], "fragments": [],
                    "charge": 0, "multiplicity": 1, "cell": np.eye(3).tolist(), "pbc": [True, False, False]},
        "metadata": {"temperature_K": temperature, "beads": 8, "dt_fs": .1, "integrator": "cayley"},
        "_rate": {"value": 2., "estimate": 2., "standard_error": .2, "unit": "s^-1"},
    }


def test_distinct_basin_normalizations_cannot_form_one_rate_series():
    first, second = site_run("narrow"), site_run("wide", 400)
    second["_config"]["site_rate"]["basin"] = [-2., .5]
    result = collect_comparisons([first, second])
    assert len(result["metrics"]["series_members"]) == 2
    with pytest.raises(ValueError, match="incompatible"):
        collect_comparisons([first, second], series=["shared", "shared"])


def test_same_cv_name_with_different_atom_definition_is_not_overlaid():
    first, second = site_run("x"), site_run("y")
    for run in (first, second):
        run["task"] = "fes"
        run.pop("_rate")
        run["_plots"] = [{"_compare": "pmf", "xlabel": "x (angstrom)", "ylabel": "Free energy (hartree)",
                          "series": [{"label": run["id"], "x": [0, 1], "y": [0, 1]}]}]
    second["_config"]["cvs"][0]["axis"] = 1
    result = collect_comparisons([first, second])
    assert not result["plots"]


def test_kie_accepts_hydrogen_isotope_symbol_and_records_true_ratio():
    h, d = site_run("H"), site_run("D")
    d["_system"]["symbols"] = ["D"]
    d["_system"]["masses"] = [3670.]
    d["_rate"].update(value=1., standard_error=.1)
    result = collect_comparisons([h, d], series=["H", "D"], kie=["H", "D"])
    point, = result["metrics"]["kinetic_isotope_effect"]["data"]
    assert point["ratio"] == 2.
    assert point["standard_error"] == pytest.approx(2 * np.sqrt(.1**2 + .1**2))


def test_identical_isotopes_cannot_be_labelled_as_kinetic_isotope_effect():
    h1, h2 = site_run("H1"), site_run("H2")
    h2["_rate"]["value"] = 1.
    with pytest.raises(ValueError, match="isotope mass change"):
        collect_comparisons([h1, h2], series=["H1", "H2"], kie=["H1", "H2"])


def test_surface_csv_fallback_uses_actual_writer_columns_and_mask(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"task": "surface_rate", "pmf": {}, "warnings": []}))
    (tmp_path / "surface_pmf.csv").write_text(
        "coordinate_bohr,free_energy_hartree,masked\n0,0,0\n1,0.1,0\n2,0.2,1\n")
    run = collect_run(tmp_path, energy_unit="hartree", length_unit="angstrom")
    pmf = next(p for p in run["_plots"] if p["id"].endswith("-pmf"))
    series = pmf["series"][0]
    assert series["x"] == pytest.approx(np.array([0., 1., 2.]) * BOHR_ANGSTROM)
    assert series["y"][:2] == pytest.approx([0., .1])
    assert np.isnan(series["y"][2])


def test_rate_pmf_saved_sample_count_available_for_sampling_scan(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"task": "rate", "pmf": {}, "warnings": []}))
    folder = tmp_path / "umbrella"
    folder.mkdir()
    (folder / "summary.json").write_text(json.dumps({"task": "fes", "dimension": 1, "units": ["dimensionless"],
                                                     "samples": 12500, "measure": "rate_flux"}))
    atomic_savez(folder / "fes.npz", edges_0=[0., .5, 1.], free_energy=[0., .1], mask=[False, False])
    run = collect_run(tmp_path)
    assert run["metadata"]["samples"] == 12500


def test_reference_change_after_read_is_detected_before_report_publication(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import pyrpmd.analysis.report_compare as comparison
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"task": "unknown", "warnings": []}))
    reference = tmp_path / "reference.csv"
    reference.write_text("series,temperature_K,rate,rate_unit,source\nA,300,1e-12,cm3 molecule^-1 s^-1,experiment\n")
    original = comparison.read_reference

    def update_after_read(path):
        points = original(path)
        reference.write_text(reference.read_text().replace("1e-12", "9e-12"))
        return points

    monkeypatch.setattr(comparison, "read_reference", update_after_read)
    with pytest.raises((RuntimeError, ValueError), match="[Ss]ource changed"):
        generate_report(run, tmp_path / "report", reference=reference, formats=("png",), dpi=72)
    assert not (tmp_path / "report").exists()


def test_mapping_cv_definitions_and_single_name_selection_are_supported(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "resolved.json").write_text(json.dumps({
        "task": "fes", "cvs": {"reaction_distance": {"kind": "distance", "atoms": [0, 1]}},
        "umbrella": {"cvs": "reaction_distance"},
    }))
    (tmp_path / "summary.json").write_text(json.dumps({"task": "fes", "dimension": 1,
                                                      "units": ["bohr"], "measure": "cv_density"}))
    atomic_savez(tmp_path / "fes.npz", edges_0=[0., 1., 2.], free_energy=[0., .1], mask=[False, False])
    run = collect_run(tmp_path)
    figure = next(p for p in run["_plots"] if p["id"].endswith("-fes"))
    assert "reaction_distance" in figure["xlabel"]


def test_adaptive_checkpoint_metadata_comes_from_frozen_plan(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "input" / "resolved.json").write_text(json.dumps({
        "task": "fes", "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": 0}],
        "umbrella": {"cvs": ["x"], "centers": [[-1.], [1.]], "stiffness": 1.},
        "adaptive_umbrella": {"enabled": True},
    }))
    (tmp_path / "summary.json").write_text(json.dumps({"task": "fes", "dimension": 1,
                                                      "units": ["bohr"], "measure": "cv_density"}))
    plan = {
        "schema_version": 1, "status": "frozen", "centers": [[-1.], [0.], [1.]], "stiffness": [[1.]], "rounds": [],
    }
    plan["checksum"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    (tmp_path / "adaptive-plan.json").write_text(json.dumps(plan))
    atomic_savez(tmp_path / "fes.npz", edges_0=[-1.5, -.5, .5, 1.5], free_energy=[0., .1, 0.], mask=[False] * 3)
    for index, center in enumerate((-1., 0., 1.)):
        atomic_savez(tmp_path / "checkpoints" / f"window_{index:04d}_replica_0000.npz",
                     values=np.array([center - .1, center, center + .1])[:, None])
    run = collect_run(tmp_path)
    windows = run["sampling_diagnostics"]["sampling"]["windows"]
    assert [window["center_atomic"] for window in windows] == [[-1.], [0.], [1.]]


def test_shared_source_change_between_run_collections_is_rejected(tmp_path, monkeypatch):
    import pyrpmd.analysis.reporting as reporting
    run = tmp_path / "run"
    run.mkdir()
    summary = run / "summary.json"
    summary.write_text(json.dumps({"task": "unknown", "status": "before", "warnings": []}))
    original = reporting.collect_run
    calls = []

    def collect_and_change(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            summary.write_text(json.dumps({"task": "unknown", "status": "after", "warnings": []}))
        return result

    monkeypatch.setattr(reporting, "collect_run", collect_and_change)
    with pytest.raises((RuntimeError, ValueError), match="[Ss]ource changed"):
        generate_report([run, run], tmp_path / "report", formats=("png",), dpi=72)
    assert not (tmp_path / "report").exists()
