"""Opt-in real-PES integration; ordinary tests never download model data."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def pilot():
    path = Path(__file__).resolve().parents[1] / "benchmarks/materials/ni_h_eam.py"
    spec = importlib.util.spec_from_file_location("materials_eam_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_model_requires_explicit_download_and_bad_bytes_are_rejected(tmp_path, pilot):
    path = tmp_path / "model.eam.alloy"
    with pytest.raises(FileNotFoundError, match="explicitly"):
        pilot.obtain_model(path)
    path.write_bytes(b"incorrect model")
    with pytest.raises(ValueError, match="SHA256"):
        pilot.obtain_model(path)
    assert path.read_bytes() == b"incorrect model"


@pytest.mark.parametrize("storage", ["memory", "stream"])
def test_pilot_diagnostics_accept_legacy_and_streamed_exports(tmp_path, pilot, storage):
    from pyrpmd.trajectory import TrajectoryStream
    from pyrpmd.workflows.materials import OBSERVABLE_NAMES

    values = {
        "positions": np.arange(30, dtype=float).reshape(5, 2, 3),
        "velocities": np.zeros((5, 2, 3)),
        "observables": np.zeros((5, len(OBSERVABLE_NAMES))),
        "radii_gyration_squared": np.array([[0, radius] for radius in (0., .125, .25, .375, .5)]),
    }
    values["observables"][:, OBSERVABLE_NAMES.index("ring_polymer_hamiltonian")] = [10., 10.25, 9.5, 11., 10.5]
    if storage == "memory":
        path = tmp_path / "materials_0000.npz"
        np.savez(path, **values)
    else:
        schema = {name: {"dtype": "<f8", "shape": list(value.shape[1:])}
                  for name, value in values.items()}
        writer = TrajectoryStream(tmp_path, schema=schema, chunk_frames=2)
        writer.extend(values)
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"format": "PyRPMD.materials-trajectory-stream.v1",
                                    "trajectory": writer.prepare_commit()}))
    drift, radius, finite = pilot._trajectory_diagnostics(path)
    assert drift == 1.0
    assert radius == .25
    assert finite is True


def test_pilot_reads_only_manifest_committed_prefix(tmp_path, pilot):
    from pyrpmd.trajectory import TrajectoryStream
    from pyrpmd.workflows.materials import OBSERVABLE_NAMES

    schema = {"positions": {"dtype": "<f8", "shape": [1, 3]},
              "velocities": {"dtype": "<f8", "shape": [1, 3]},
              "observables": {"dtype": "<f8", "shape": [len(OBSERVABLE_NAMES)]},
              "radii_gyration_squared": {"dtype": "<f8", "shape": [1]}}
    writer = TrajectoryStream(tmp_path, schema=schema, chunk_frames=1)
    record = {name: np.zeros(spec["shape"]) for name, spec in schema.items()}
    record["radii_gyration_squared"][:] = .5
    writer.append(record)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"format": "PyRPMD.materials-trajectory-stream.v1",
                                "trajectory": writer.prepare_commit()}))
    # A later durable chunk is an orphan from this exported prefix's view.
    record["observables"][-1] = 100.
    record["radii_gyration_squared"][:] = 9.
    writer.append(record)
    assert pilot._trajectory_diagnostics(path) == (0.0, .5, True)


def test_real_nickel_hydrogen_eam_adapter_and_periodic_workflow(tmp_path, pilot):
    model = os.environ.get("PYRPMD_NIALH_POTENTIAL")
    if not model:
        pytest.skip("Set PYRPMD_NIALH_POTENTIAL to opt into the source-pinned real EAM check")
    pytest.importorskip("ase")
    from pyrpmd import run

    model = pilot.obtain_model(model)
    assert pilot.verify_adapter(model)["passed"]
    result = run(pilot.job(model, isotope="D", beads=2, quick=True), output=tmp_path / "run")
    assert result["status"] == "complete"
    assert result["pbc"] == [True, True, False]
    assert result["phase"] == "unthermostatted_rpmd_production"
    assert result["physical_converged"] is False
    assert result["diffusion"]["diffusion_m2_s"] is None
