"""Independent saved-data contract checks for automatic sampling reports."""

import hashlib
import json

import numpy as np
import pytest

from pyrpmd.analysis import report_sampling
from pyrpmd.constants import ANGSTROM_TO_AU


def _run(path, config=None):
    return {"id": "test", "label": "sampling", "path": str(path), "_config": config or {},
            "_plots": [], "_sources": {}, "metrics": {}, "findings": [], "skipped": []}


def _write_plan(path, plan):
    plan = {key: value for key, value in plan.items() if key != "checksum"}
    plan["checksum"] = hashlib.sha256(json.dumps(
        plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    (path / "adaptive-plan.json").write_text(json.dumps(plan))


def _plan():
    return {
        "schema_version": 2, "status": "incomplete", "failure_reason": "max_rounds before coverage target",
        "centers": [[0., 0.], [1., 1.]], "stiffness": [[2., 0.], [0., 2.]],
        "options": {"target_overlap": .05, "min_effective_samples": 20.},
        "pilot_samples_used_in_production": 0,
        "rounds": [
            {"round": 0, "centers": [[0., 0.]], "spanning_overlap": 1.,
             "effective_samples_per_window": [10.], "pilot": {"production_steps": 100},
             "coverage": {"fraction": .25, "target": .8}},
            {"round": 1, "centers": [[0., 0.], [1., 1.]], "spanning_overlap": .03,
             "effective_samples_per_window": [30., 25.], "pilot": {"production_steps": 200},
             "coverage": {"fraction": .65, "target": .8}},
        ],
    }


def _stream_file(path, *, window=0, cv_units=None, measure=None):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path / "fes.h5", "w") as saved:
        saved.attrs["energy_unit"] = "hartree"
        if cv_units is not None:
            saved.attrs["cv_units"] = json.dumps(cv_units)
        if measure is not None:
            saved.attrs["measure"] = measure
        stream = saved.create_group("streams/0")
        stream.attrs.update(window=window, replica=0)
        stream["samples"] = [[10.], [20.]]
        stream["center"] = [15.]
        stream["stiffness"] = [[1.]]


def test_vector_python_cv_broadcasts_declared_numeric_center_unit():
    config = {"cvs": [{"name": "pair", "kind": "python", "factory": "never.import.this", "unit": "angstrom"}],
              "umbrella": {"cvs": ["pair"], "centers": [[1., 2.]], "stiffness": 3.}}
    centers, stiffness = report_sampling._canonical_metadata(config, {"units": ["bohr", "bohr"]}, 2)
    np.testing.assert_allclose(centers, [[ANGSTROM_TO_AU, 2 * ANGSTROM_TO_AU]])
    np.testing.assert_allclose(stiffness, np.eye(2) * 3.)


def test_explicit_quantity_pairs_do_not_receive_cv_unit_factor_twice():
    config = {"cvs": [{"name": "distance", "unit": "angstrom"}],
              "umbrella": {"centers": [[[1., "angstrom"]], [[2., "bohr"]]], "stiffness": 1.}}
    centers, _ = report_sampling._canonical_metadata(config, {"units": ["bohr"]}, 1)
    np.testing.assert_allclose(centers[:, 0], [ANGSTROM_TO_AU, 2.])


def test_vector_cv_without_input_units_preserves_each_canonical_dimension():
    config = {"cvs": [{"name": "mixed", "kind": "python", "factory": "never.import.this"}],
              "umbrella": {"centers": [[2., .5]], "stiffness": [[1., 0.], [0., 2.]]}}
    centers, _ = report_sampling._canonical_metadata(config, {"units": ["bohr", "radian"]}, 2)
    np.testing.assert_allclose(centers, [[2., .5]])


@pytest.mark.parametrize("cv_units,measure", [(["angstrom"], None), (["bohr", "bohr"], None), (None, "rate_flux")])
def test_conflicting_hdf5_metadata_is_not_used_with_fes_axes(tmp_path, cv_units, measure):
    _stream_file(tmp_path, cv_units=cv_units, measure=measure)
    run = _run(tmp_path)
    report_sampling.collect_sampling(run, tmp_path, {"task": "fes", "units": ["bohr"], "measure": "cv_density"},
                                     [("x (bohr)", 1.)], "hartree")
    assert not run["_plots"]
    assert any(item["figure"] == "raw HDF5 sampling" for item in run["skipped"])


def test_hdf5_unit_conflict_can_fall_back_to_independent_checkpoints(tmp_path):
    _stream_file(tmp_path, cv_units=["angstrom"])
    (tmp_path / "checkpoints").mkdir()
    np.savez(tmp_path / "checkpoints/window_00000_replica_0000.npz", values=[[1.], [2.]])
    config = {"umbrella": {"centers_au": [[1.5]], "stiffness_au": [[1.]]}}
    run = _run(tmp_path, config)
    report_sampling.collect_sampling(run, tmp_path, {"task": "fes", "units": ["bohr"]}, [("x (bohr)", 1.)], "hartree")
    trace = next(plot for plot in run["_plots"] if plot["id"].endswith("-trace-1"))
    np.testing.assert_array_equal(trace["series"][0]["y"], [1., 2.])


@pytest.mark.parametrize("bad_level", ["streams", "stream", "samples"])
def test_hdf5_wrong_object_type_is_a_diagnostic_not_an_uncaught_error(tmp_path, bad_level):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "fes.h5", "w") as saved:
        if bad_level == "streams":
            saved["streams"] = [0.]
        elif bad_level == "stream":
            saved["streams/0"] = [0.]
        else:
            saved.create_group("streams/0/samples")
    run = _run(tmp_path)
    report_sampling.collect_sampling(run, tmp_path, {"task": "fes"}, [("x (bohr)", 1.)], "hartree")
    assert not run["_plots"]
    assert any(item["figure"] == "raw HDF5 sampling" for item in run["skipped"])


@pytest.mark.parametrize("window", [1.5, -1, True])
def test_hdf5_window_identity_is_never_truncated_or_coerced(tmp_path, window):
    _stream_file(tmp_path, window=window)
    run = _run(tmp_path)
    report_sampling.collect_sampling(run, tmp_path, {"task": "fes"}, [("x (bohr)", 1.)], "hartree")
    assert "sampling_diagnostics" not in run
    assert any("nonnegative integers" in item["reason"] for item in run["skipped"])


def test_bad_hills_replica_does_not_leave_a_partial_height_curve(tmp_path):
    for replica, bias in (("0000", "nan"), ("0001", "0.3")):
        folder = tmp_path / "replicas" / replica
        folder.mkdir(parents=True)
        (folder / "HILLS.csv").write_text(f"time_au,height_hartree,bias_before_hartree\n10,0.2,0\n20,0.1,{bias}\n")
    run = _run(tmp_path)
    report_sampling.collect_sampling(run, tmp_path, {"task": "metadynamics"}, [("x (bohr)", 1.)], "hartree")
    curves = [plot for plot in run["_plots"] if "hill-" in plot["id"]]
    assert len(curves) == 2
    assert all([series["label"] for series in plot["series"]] == ["Replica 0001"] for plot in curves)
    assert any(item["figure"] == "replicas/0000/HILLS.csv" for item in run["skipped"])


def test_incomplete_pilot_plan_has_progress_without_production_or_raw_pilot_reads(tmp_path):
    _write_plan(tmp_path, _plan())
    pilot = tmp_path / "adaptation/round_000/checkpoints"
    pilot.mkdir(parents=True)
    (pilot / "window_00000_replica_0000.npz").write_bytes(b"this must never be opened")
    run = _run(tmp_path)
    report_sampling.collect_adaptive_progress(run, tmp_path)
    plots = {plot["id"]: plot for plot in run["_plots"]}
    assert len(plots) == 5
    assert run["metrics"]["sampling_adaptive_status"] == "incomplete"
    assert plots["test-sampling-adaptive-overlap"]["hlines"][0]["y"] == .05
    assert plots["test-sampling-adaptive-ess"]["hlines"][0]["y"] == 20.
    assert plots["test-sampling-adaptive-pilot-steps"]["series"][0]["y"] == [100., 200.]
    assert plots["test-sampling-adaptive-coverage"]["series"][0]["y"] == [.25, .65]
    assert plots["test-sampling-adaptive-coverage"]["series"][1]["y"] == [.8, .8]
    assert not any("/adaptation/" in source for source in run["_sources"])
    assert any(finding["code"] == "adaptation_incomplete" for finding in run["findings"])
    assert run["sampling_diagnostics"]["sampling"]["adaptive"]["scope"] == "discarded_adaptation_pilots"


@pytest.mark.parametrize("schema", [2, 3])
def test_frozen_centers_are_usable_without_hdf5(tmp_path, schema):
    plan = _plan()
    plan["status"] = "frozen"
    plan["schema_version"] = schema
    if schema == 3:
        plan["overlap_metric"] = "normalized_mbar_cosine"
    _write_plan(tmp_path, plan)
    (tmp_path / "checkpoints").mkdir()
    np.savez(tmp_path / "checkpoints/window_00001_replica_0000.npz", values=[[.9, .9], [1.1, 1.1]])
    run = _run(tmp_path, {"adaptive_umbrella": {"enabled": True}})
    report_sampling.collect_sampling(run, tmp_path, {"task": "fes", "units": ["bohr", "bohr"]},
                                     [("x (bohr)", 1.), ("y (bohr)", 1.)], "hartree")
    row = run["sampling_diagnostics"]["sampling"]["windows"][0]
    assert row["center_atomic"] == [1., 1.]
    assert row["stiffness_atomic"] == [[2., 0.], [0., 2.]]


def test_schema_three_report_names_the_normalized_overlap_metric(tmp_path):
    plan = _plan()
    plan.update(schema_version=3, overlap_metric="normalized_mbar_cosine")
    _write_plan(tmp_path, plan)
    run = _run(tmp_path)
    report_sampling.collect_adaptive_progress(run, tmp_path)
    overlap = next(plot for plot in run["_plots"] if plot["id"].endswith("-adaptive-overlap"))
    assert "normalized MBAR cosine".lower() in overlap["ylabel"].lower()
    assert "not directly comparable" in overlap["caption"]
    assert run["sampling_diagnostics"]["sampling"]["adaptive"]["overlap_metric"] == "normalized_mbar_cosine"


@pytest.mark.parametrize("metric", [None, "mbar_symmetric_transition", "unknown"])
def test_schema_three_report_rejects_an_ambiguous_overlap_metric(tmp_path, metric):
    plan = _plan()
    plan["schema_version"] = 3
    if metric is not None:
        plan["overlap_metric"] = metric
    _write_plan(tmp_path, plan)
    run = _run(tmp_path)
    report_sampling.collect_adaptive_progress(run, tmp_path)
    assert not run["_plots"]
    assert any("overlap metric" in item["reason"] for item in run["skipped"])


def test_legacy_plan_without_saved_schedule_does_not_invent_steps_or_coverage(tmp_path):
    plan = _plan()
    plan["schema_version"] = 1
    for row in plan["rounds"]:
        row.pop("pilot")
        row.pop("coverage")
    _write_plan(tmp_path, plan)
    run = _run(tmp_path)
    report_sampling.collect_adaptive_progress(run, tmp_path)
    assert len(run["_plots"]) == 3
    assert not any(plot["id"].endswith(("-pilot-steps", "-coverage")) for plot in run["_plots"])
    diagnostics = run["sampling_diagnostics"]["sampling"]["adaptive"]
    assert diagnostics["schema_version"] == 1 and diagnostics["status"] == "incomplete"
    assert all(row["pilot_steps_per_stream"] is None and row["coverage_fraction"] is None
               for row in diagnostics["rounds"])


def test_interrupted_plan_before_first_round_keeps_status_without_fabricating_progress(tmp_path):
    plan = _plan()
    plan.update(status="adapting", rounds=[])
    plan.pop("failure_reason")
    _write_plan(tmp_path, plan)
    run = _run(tmp_path)
    report_sampling.collect_adaptive_progress(run, tmp_path)
    assert run["metrics"]["sampling_adaptive_status"] == "adapting"
    assert not run["_plots"]
    assert any(item["figure"] == "adaptive progress" for item in run["skipped"])
    assert any(item["code"] == "adaptation_incomplete" for item in run["findings"])


@pytest.mark.parametrize("corruption", ["unknown_schema", "negative_ess", "ess_shape", "overlap", "coverage"])
def test_corrupt_pilot_diagnostics_are_not_rendered(tmp_path, corruption):
    plan = _plan()
    if corruption == "unknown_schema":
        plan["schema_version"] = 999
    elif corruption == "negative_ess":
        plan["rounds"][0]["effective_samples_per_window"] = [-5.]
    elif corruption == "ess_shape":
        plan["rounds"][1]["effective_samples_per_window"] = [10.]
    elif corruption == "overlap":
        plan["rounds"][0]["spanning_overlap"] = 1.5
    else:
        plan["rounds"][0]["coverage"]["fraction"] = -1.
    _write_plan(tmp_path, plan)
    run = _run(tmp_path)
    report_sampling.collect_adaptive_progress(run, tmp_path)
    assert not run["_plots"]
    assert any(item["figure"] == "adaptive sampling progress" for item in run["skipped"])
