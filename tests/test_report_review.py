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


def prepared_rate(tmp_path, name, *, temperature=300, other_channel=False, mass=1837.):
    from test_reporting import _rate

    path = _rate(tmp_path, name=name, temperature=temperature, mass=mass)
    source = path / "input/resolved.json"
    config = json.loads(source.read_text())
    reaction = config.pop("reaction")
    config["system"] = {"transition_state": "/model/ts.xyz"}
    source.write_text(json.dumps(config))
    if other_channel:
        reaction.update(fragments=[[2], [0, 1]], forming_bonds=[[1, 2]], breaking_bonds=[[0, 1]])
    (path / "preparation.json").write_text(json.dumps({"ready": True, "reaction": reaction}))
    return path


def test_ts_only_actual_channels_are_not_combined_or_fitted(tmp_path):
    paths = [prepared_rate(tmp_path, "A300"), prepared_rate(tmp_path, "A400", temperature=400),
             prepared_rate(tmp_path, "B500", temperature=500, other_channel=True)]
    runs = [collect_run(path, prefix=f"run-{index}") for index, path in enumerate(paths)]
    result = collect_comparisons(runs, arrhenius_fit=True)
    assert sorted(map(len, result["metrics"]["series_members"].values())) == [1, 2]
    assert result["metrics"]["arrhenius_fits"] == []
    assert len([p for p in result["plots"] if p["id"].startswith("comparison-pmf")]) == 1
    assert runs[0]["metadata"]["rinf_bohr"] == 12.
    assert str(paths[0] / "preparation.json") in runs[0]["_sources"]
    with pytest.raises(ValueError, match="incompatible"):
        collect_comparisons(runs, series=["shared"] * 3)


def test_ts_only_incompatible_channel_cannot_become_kie(tmp_path):
    first = collect_run(prepared_rate(tmp_path, "H"))
    second = collect_run(prepared_rate(tmp_path, "D", other_channel=True, mass=3670.))
    with pytest.raises(ValueError, match="compatible saved reaction"):
        collect_comparisons([first, second], series=["H", "D"], kie=["H", "D"])


@pytest.mark.parametrize("preparation", [None, {"ready": False, "reaction": {}}, {"ready": True, "reaction": {}}])
def test_unavailable_prepared_channel_withholds_grouping(tmp_path, preparation):
    paths = [prepared_rate(tmp_path, name) for name in ("one", "two")]
    for path in paths:
        if preparation is None:
            (path / "preparation.json").unlink()
        else:
            (path / "preparation.json").write_text(json.dumps(preparation))
    runs = [collect_run(path) for path in paths]
    result = collect_comparisons(runs)
    assert len(result["metrics"]["series_members"]) == 2
    assert not any(p["id"].startswith("comparison-pmf") for p in result["plots"])
    assert all(any(f["code"] == "reaction_identity_unavailable" for f in run["findings"]) for run in runs)
    with pytest.raises(ValueError, match="unavailable"):
        collect_comparisons(runs, series=["same", "same"])


def test_model_file_revisions_are_part_of_comparison_identity(tmp_path):
    paths = [prepared_rate(tmp_path, name) for name in ("one", "two")]
    for index, path in enumerate(paths):
        config_path = path / "input/resolved.json"
        config = json.loads(config_path.read_text())
        config.update(potential={"provider": "cabi", "library": "/model/pes.so"}, _input_path=f"/inputs/{index}.inp")
        config_path.write_text(json.dumps(config))
        (path / "manifest.json").write_text(json.dumps({"schema": "pyrpmd.run", "files": {
            "/model/pes.so": str(index) * 64, config["_input_path"]: str(index + 1) * 64,
            "/model/ts.xyz": "f" * 64}}))
    runs = [collect_run(path) for path in paths]
    assert len(collect_comparisons(runs)["metrics"]["series_members"]) == 2
    assert set(runs[0]["_model_files"]) == {"/model/pes.so"}
    manifest = paths[1] / "manifest.json"
    saved = json.loads(manifest.read_text())
    saved["files"]["/model/pes.so"] = "0" * 64
    manifest.write_text(json.dumps(saved))
    runs[1] = collect_run(paths[1])
    assert len(collect_comparisons(runs)["metrics"]["series_members"]) == 1


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


def test_different_static_confinement_is_not_one_metadynamics_ensemble():
    first, second = site_run("confined"), site_run("wide")
    for run, upper in ((first, 1.), (second, 5.)):
        run["task"] = "metadynamics"
        run["metadata"]["measure"] = "cv_density"
        run.pop("_rate")
        run["_config"]["metadynamics"] = {"cvs": ["x"], "walls": [{"cv": "x", "upper": upper, "kappa": 1.}]}
        run["_plots"] = [{"_compare": "pmf", "xlabel": "x (angstrom)", "ylabel": "Free energy (hartree)",
                          "series": [{"label": run["id"], "x": [0, 1], "y": [0, 1]}]}]
    result = collect_comparisons([first, second])
    assert not result["plots"]
    assert len(result["metrics"]["series_members"]) == 2


def selected_fes_run(path, *, task="opes", selected=None, order=("x", "y"),
                     mapping=True, summary_task=True):
    definitions = {name: {"kind": "cartesian", "atom": 0, "axis": "xyz".index(name)} for name in order}
    config = {"task": task, "potential": {"provider": "harmonic"}, "cvs": definitions}
    if not mapping:
        config["cvs"] = [dict(spec, name=name) for name, spec in definitions.items()]
    section = task if task in ("opes", "metadynamics") else "umbrella"
    if selected is not None:
        config[section] = {"cvs": selected}
    dimension = len(order) if selected is None else 1 if isinstance(selected, str) else len(selected)
    (path / "input").mkdir(parents=True)
    (path / "input/resolved.json").write_text(json.dumps(config))
    (path / "input/system.json").write_text(json.dumps({"symbols": ["H"], "masses": [1837.]}))
    summary = {"measure": "cv_density", "units": ["bohr"] * dimension}
    if summary_task:
        summary["task"] = task
    (path / "summary.json").write_text(json.dumps(summary))
    shape = (2,) * dimension
    atomic_savez(path / "fes.npz", free_energy=np.arange(2 ** dimension).reshape(shape) * .1,
                 mask=np.zeros(shape, dtype=bool), measure="cv_density", cv_units=["bohr"] * dimension,
                 **{f"edges_{axis}": [0., 1., 2.] for axis in range(dimension)})
    return collect_run(path)


@pytest.mark.parametrize("selection_type", [str, lambda name: [name]])
def test_opes_selected_observables_with_same_units_are_not_combined(tmp_path, selection_type):
    runs = [selected_fes_run(tmp_path / name, selected=selection_type(name)) for name in ("x", "y")]
    for run, name in zip(runs, ("x", "y")):
        plot = next(plot for plot in run["_plots"] if plot["id"].endswith("-fes"))
        assert plot["xlabel"] == f"{name} (angstrom)"
        np.testing.assert_allclose(plot["series"][0]["x"], np.array([.5, 1.5]) * BOHR_ANGSTROM)
    comparison = collect_comparisons(runs)
    assert len(comparison["metrics"]["series_members"]) == 2
    assert not any(plot["id"].startswith("comparison-pmf") for plot in comparison["plots"])
    with pytest.raises(ValueError, match="incompatible"):
        collect_comparisons(runs, series=["same", "same"])

    same = selected_fes_run(tmp_path / "same-x", selected=selection_type("x"))
    comparison = collect_comparisons([runs[0], same])
    assert len(comparison["metrics"]["series_members"]) == 1
    assert len([plot for plot in comparison["plots"] if plot["id"].startswith("comparison-pmf")]) == 1


def test_opes_axis_order_and_missing_summary_task_use_saved_selection(tmp_path):
    run = selected_fes_run(tmp_path, selected=["y", "x"], summary_task=False)
    plot = next(plot for plot in run["_plots"] if plot["id"].endswith("-fes"))
    assert (plot["xlabel"], plot["ylabel"]) == ("y (angstrom)", "x (angstrom)")


@pytest.mark.parametrize("task", ["fes", "metadynamics", "opes"])
@pytest.mark.parametrize("mapping", [True, False])
def test_default_cv_selection_retains_definition_order(tmp_path, task, mapping):
    runs = [selected_fes_run(tmp_path / "".join(order), task=task, order=order, mapping=mapping)
            for order in (("x", "y"), ("y", "x"))]
    for run, names in zip(runs, (("x", "y"), ("y", "x"))):
        plot = next(plot for plot in run["_plots"] if plot["id"].endswith("-fes"))
        assert (plot["xlabel"], plot["ylabel"]) == tuple(f"{name} (angstrom)" for name in names)
    assert len(collect_comparisons(runs)["metrics"]["series_members"]) == 2
    with pytest.raises(ValueError, match="incompatible"):
        collect_comparisons(runs, series=["same", "same"])


def fes_measure_run(path, *, saved_measure="cv_density", summary_measure=None,
                    saved_units=("bohr",), summary_units=None):
    (path / "input").mkdir(parents=True)
    (path / "input/resolved.json").write_text(json.dumps({"task": "fes", "potential": {"provider": "harmonic"}}))
    (path / "input/system.json").write_text(json.dumps({"symbols": ["H"], "masses": [1837.]}))
    summary = {"task": "fes"}
    if summary_measure is not None:
        summary["measure"] = summary_measure
    if summary_units is not None:
        summary["units"] = summary_units
    (path / "summary.json").write_text(json.dumps(summary))
    data = {"edges_0": [0., 1., 2.], "free_energy": [0., .1], "mask": [False, False], "cv_units": list(saved_units)}
    if saved_measure is not None:
        data["measure"] = saved_measure
    atomic_savez(path / "fes.npz", **data)
    return collect_run(path)


def test_array_measure_separates_density_and_flux_curves(tmp_path):
    density = fes_measure_run(tmp_path / "density")
    flux = fes_measure_run(tmp_path / "flux", saved_measure="rate_flux")
    assert density["metadata"]["measure"] == "cv_density"
    assert flux["metadata"]["measure"] == "rate_flux"
    result = collect_comparisons([density, flux])
    assert not result["plots"]
    assert len(result["metrics"]["series_members"]) == 2


@pytest.mark.parametrize("options", [{"summary_measure": "rate_flux"}, {"summary_units": ["angstrom"]}])
def test_contradictory_summary_measure_or_units_withhold_fes(tmp_path, options):
    run = fes_measure_run(tmp_path, **options)
    assert not any(p["id"].endswith("-fes") for p in run["_plots"])
    assert run["_comparison_identity_available"] is False
    assert any("contradict" in item["reason"] for item in run["skipped"])


def test_unknown_fes_measure_does_not_create_comparison(tmp_path):
    runs = [fes_measure_run(tmp_path / name, saved_measure=None) for name in ("one", "two")]
    assert all(any(p["id"].endswith("-fes") for p in run["_plots"]) for run in runs)
    result = collect_comparisons(runs)
    assert not result["plots"] and len(result["metrics"]["series_members"]) == 2


@pytest.mark.parametrize("task,suffix", [("rate", "cm3_s"), ("site_rate", "s"), ("surface_rate", "s_inv")])
def test_thermal_rate_ci_and_uncertainty_scope_are_retained(tmp_path, task, suffix):
    (tmp_path / "summary.json").write_text(json.dumps({"task": task,
        "rate_" + suffix: 2., "rate_percentile_95_ci_" + suffix: [.5, 7.],
        "rate_uncertainty": {"available": True, "method": "joint_pmf_parent_bootstrap"},
        "pmf": {"barrier_percentile_95_ci_hartree": [.01, .04]}}))
    run = collect_run(tmp_path)
    assert run["metrics"]["rate_percentile_95_ci"] == [.5, 7.]
    assert run["metrics"]["rate_uncertainty"]["method"] == "joint_pmf_parent_bootstrap"
    assert run["metrics"]["barrier_percentile_95_ci_hartree"] == [.01, .04]
    assert run["metrics"]["rate_standard_error"] is None


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
