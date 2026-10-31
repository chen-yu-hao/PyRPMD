"""Saved sampling diagnostics with independent unit and statistics checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.analysis.report_sampling import collect_sampling, _canonical_metadata
from pyrpmd.constants import ANGSTROM_TO_AU, AU_TIME_FS
from pyrpmd.units import unit_factor


def _run(path, config=None):
    return {"id": "test", "label": "example", "path": str(path), "_config": config or {},
            "_plots": [], "_sources": {}, "metrics": {}, "findings": [], "skipped": []}


def _hdf5(path, streams):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path / "fes.h5", "w") as file:
        file.attrs["energy_unit"] = "hartree"
        groups = file.create_group("streams")
        for i, stream in enumerate(streams):
            group = groups.create_group(str(i))
            group.attrs.update(window=stream["window"], replica=stream["replica"])
            for key in ("samples", "center", "stiffness"):
                group.create_dataset(key, data=stream[key])


def _plot(run, suffix):
    return next(plot for plot in run["_plots"] if plot["id"] == "test-" + suffix)


def test_all_window_statistics_retain_replica_boundaries_and_sd_scope(tmp_path):
    _hdf5(tmp_path, [
        {"window": 0, "replica": 0, "samples": [[0.], [2.]], "center": [2.], "stiffness": [[3.]]},
        {"window": 0, "replica": 1, "samples": [[4.], [6.]], "center": [2.], "stiffness": [[3.]]},
    ])
    run = _run(tmp_path)
    collect_sampling(run, tmp_path, {"task": "fes", "units": ["bohr"]}, [("x (angstrom)", 1/ANGSTROM_TO_AU)], "hartree")
    record = run["sampling_diagnostics"]["sampling"]
    assert len(record["streams"]) == 2
    assert all(stream["coordinates"][0]["conditional_standard_error"] is None for stream in record["streams"])
    assert record["windows"][0]["mean"][0] == pytest.approx(3/ANGSTROM_TO_AU)
    assert record["windows"][0]["sample_variance"][0] == pytest.approx((20/3)/ANGSTROM_TO_AU**2)
    offset = _plot(run, "sampling-offset-1")
    assert offset["series"][0]["y"][0] == pytest.approx(1/ANGSTROM_TO_AU)
    assert offset["series"][0]["yerr"][0] == pytest.approx(np.sqrt(20/3)/ANGSTROM_TO_AU)
    assert "not a standard error" in offset["caption"]
    stiffness = _plot(run, "sampling-stiffness-1")
    assert stiffness["series"][0]["y"][0] == pytest.approx(3*ANGSTROM_TO_AU**2)


def test_checkpoint_fallback_resolves_declared_units_without_cv_factory(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    np.savez(checkpoints / "window_00000_replica_0000.npz", values=np.array([[1., 2.], [3., 4.], [5., 6.]]))
    config = {"cvs": [{"name": "a", "kind": "python", "factory": "do.not.execute", "units": ["angstrom", "bohr"]}],
              "umbrella": {"cvs": ["a"], "centers": [[1, 2]], "stiffness": "2 hartree/bohr^2"}}
    run = _run(tmp_path, config)
    collect_sampling(run, tmp_path, {"task": "fes", "units": ["bohr", "bohr"]},
                     [("a (angstrom)", 1/ANGSTROM_TO_AU), ("b (angstrom)", 1/ANGSTROM_TO_AU)], "eV")
    row = run["sampling_diagnostics"]["sampling"]["windows"][0]
    assert row["center_atomic"] == pytest.approx([ANGSTROM_TO_AU, 2])
    assert row["stiffness_atomic"] == [[2, 0], [0, 2]]
    shown = _plot(run, "sampling-stiffness-1")["series"][0]["y"][0]
    assert shown == pytest.approx(2/unit_factor("eV", "energy")*ANGSTROM_TO_AU**2)
    assert _plot(run, "sampling-trace-1")["xlabel"] == "Saved sample index"
    assert any(source.get("integrity") for source in run["_sources"].values())


def test_checkpoint_without_metadata_still_provides_raw_traces(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    np.savez(checkpoints / "window_00001_replica_0000.npz", values=np.arange(5.)[:, None])
    run = _run(tmp_path)
    collect_sampling(run, tmp_path, {"task": "fes", "units": ["bohr"]}, [("x (bohr)", 1.)], "hartree")
    assert _plot(run, "sampling-trace-1")
    assert any(finding["code"] == "umbrella_metadata_missing" for finding in run["findings"])
    assert any("center/stiffness" in item["figure"] for item in run["skipped"])


def test_representative_distributions_are_explicit_but_all_window_statistics_remain(tmp_path):
    _hdf5(tmp_path, [{"window": i, "replica": 0, "samples": [[i-.1], [i+.1]], "center": [i], "stiffness": [[1.]]} for i in range(13)])
    run = _run(tmp_path)
    collect_sampling(run, tmp_path, {"task": "fes", "units": ["dimensionless"]}, [("xi (dimensionless)", 1.)], "hartree")
    hist = _plot(run, "sampling-distributions-1")
    assert len(hist["series"]) == 8
    assert "8 of 13" in hist["caption"]
    assert len(run["sampling_diagnostics"]["sampling"]["windows"]) == 13
    assert len(_plot(run, "sampling-offset-1")["series"][0]["x"]) == 13
    assert len(_plot(run, "sampling-trace-1")["series"]) == 6


def test_metadynamics_uses_saved_clock_and_does_not_claim_equilibrium_errors(tmp_path):
    replicas = tmp_path / "replicas" / "0000"
    replicas.mkdir(parents=True)
    (replicas / "HILLS.csv").write_text("step,time_au,height_hartree,bias_before_hartree\n2,10,0.2,0\n4,20,0.1,0.3\n")
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    np.savez(checkpoints / "metadynamics_0000.npz", samples=np.array([[1.], [2.], [3.]]), sample_steps=[2, 4, 6])
    run = _run(tmp_path, {"dt_au": 5})
    collect_sampling(run, tmp_path, {"task": "metadynamics"}, [("x (bohr)", 1.)], "hartree")
    trace = _plot(run, "sampling-trace-1")
    assert trace["series"][0]["x"] == pytest.approx(np.array([10, 20, 30])*AU_TIME_FS)
    assert "not reaction time" in trace["caption"]
    assert "sampling_diagnostics" not in run
    heights = _plot(run, "sampling-hill-height")
    assert heights["series"][0]["y"] == pytest.approx([.2, .1])
    assert "kinetic rate" in heights["caption"]


def test_external_hdf5_data_is_not_followed(tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "fes.h5", "w") as file:
        file["streams"] = h5py.ExternalLink("outside.h5", "/streams")
    run = _run(tmp_path)
    collect_sampling(run, tmp_path, {"task": "fes"}, [("x (bohr)", 1.)], "hartree")
    assert not run["_plots"]
    assert any("External HDF5" in finding["message"] for finding in run["findings"])


def test_duplicate_replica_identity_does_not_double_sampling_counts(tmp_path):
    stream = {"window": 0, "replica": 0, "samples": [[0.], [1.]], "center": [0.], "stiffness": [[1.]]}
    _hdf5(tmp_path, [stream, stream])
    run = _run(tmp_path)
    collect_sampling(run, tmp_path, {"task": "fes"}, [("x (bohr)", 1.)], "hartree")
    assert "sampling_diagnostics" not in run
    assert any("repeat a window/replica" in finding["message"] for finding in run["findings"])


def test_bad_adaptive_checksum_is_reported_instead_of_presented_as_progress(tmp_path):
    (tmp_path / "adaptive-plan.json").write_text(json.dumps({"checksum": "bad", "rounds": [{"round": 0}]}))
    run = _run(tmp_path)
    collect_sampling(run, tmp_path, {"task": "fes"}, [("x (bohr)", 1.)], "hartree")
    assert any("checksum mismatch" in finding["message"] for finding in run["findings"])
    assert not run["_plots"]


def test_saved_grid_order_and_mixed_unit_stiffness(tmp_path):
    config = {"cvs": [{"name": "x", "unit": "angstrom"}, {"name": "theta", "unit": "radian"}],
              "umbrella": {"cvs": ["x", "theta"], "grid": [{"start": 0, "stop": 1, "count": 2}, [0, .5]],
                           "stiffness": [["2 hartree/bohr^2", "0.1 hartree/bohr/radian"], ["0.1 hartree/bohr/radian", "3 hartree/radian^2"]]}}
    centers, stiffness = _canonical_metadata(config, {"units": ["bohr", "radian"]}, 2)
    np.testing.assert_allclose(centers, [[0, 0], [0, .5], [ANGSTROM_TO_AU, 0], [ANGSTROM_TO_AU, .5]])
    np.testing.assert_allclose(stiffness, [[2, .1], [.1, 3]])


def test_adaptive_checkpoint_fallback_uses_frozen_grid_and_excludes_pilots(tmp_path):
    import hashlib

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for window, center in enumerate((-1., 0., 1.)):
        np.savez(checkpoints / f"window_{window:05d}_replica_0000.npz", values=np.array([[center-.1], [center+.1]]))
    pilots = tmp_path / "adaptation" / "round_000" / "checkpoints"
    pilots.mkdir(parents=True)
    np.savez(pilots / "window_00001_replica_0000.npz", values=np.array([[999.], [1001.]]))
    plan = {"schema_version": 1, "status": "frozen", "centers": [[-1.], [0.], [1.]],
            "stiffness": [[4.]], "rounds": [], "pilot_samples_used_in_production": 0}
    plan["checksum"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (tmp_path / "adaptive-plan.json").write_text(json.dumps(plan))
    config = {"cvs": [{"name": "x", "unit": "bohr"}],
              "umbrella": {"centers": [[-1.], [1.]], "stiffness": 2.},
              "adaptive_umbrella": {"enabled": True}}
    run = _run(tmp_path, config)
    collect_sampling(run, tmp_path, {"task": "fes", "units": ["bohr"]}, [("x (bohr)", 1.)], "hartree")
    records = run["sampling_diagnostics"]["sampling"]["windows"]
    assert [row["center_atomic"] for row in records] == [[-1.], [0.], [1.]]
    assert [row["stiffness_atomic"] for row in records] == [[[4.]]] * 3
    assert sum(row["samples"] for row in records) == 6
    np.testing.assert_allclose(_plot(run, "sampling-offset-1")["series"][0]["y"], 0., atol=1e-15)
    assert not any("/adaptation/" in source for source in run["_sources"])
    assert any("adaptive-plan.json" in source for source in run["_sources"])


@pytest.mark.parametrize("plan_state", ["missing", "unchecked", "not_frozen", "corrupted"])
def test_adaptive_fallback_does_not_guess_centers_without_valid_frozen_plan(tmp_path, plan_state):
    import hashlib

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    np.savez(checkpoints / "window_00001_replica_0000.npz", values=np.array([[-.1], [.1]]))
    if plan_state != "missing":
        plan = {"schema_version": 1, "status": "adapting" if plan_state == "not_frozen" else "frozen",
                "centers": [[-1.], [0.], [1.]], "stiffness": [[4.]], "rounds": []}
        if plan_state != "unchecked":
            plan["checksum"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if plan_state == "corrupted":
            plan["centers"][1][0] = 0.5
        (tmp_path / "adaptive-plan.json").write_text(json.dumps(plan))
    run = _run(tmp_path, {"umbrella": {"centers": [[-1.], [1.]], "stiffness": 2.},
                          "adaptive_umbrella": {"enabled": True}})
    collect_sampling(run, tmp_path, {"task": "fes", "units": ["bohr"]}, [("x (bohr)", 1.)], "hartree")
    assert _plot(run, "sampling-trace-1")
    assert not any("-offset-" in plot["id"] or "-stiffness-" in plot["id"] for plot in run["_plots"])
    assert run["sampling_diagnostics"]["sampling"]["windows"][0]["center_atomic"] is None
    assert any(finding["code"] == "umbrella_metadata_missing" for finding in run["findings"])
