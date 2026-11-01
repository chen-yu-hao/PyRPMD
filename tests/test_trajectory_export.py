"""External-reader roundtrips, unit checks and transactional export failures."""
import json

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez
from pyrpmd.constants import AMU_TO_AU, BOHR_ANGSTROM, FS_TO_AU, PS_TO_AU
from pyrpmd.dynamics import State
from pyrpmd.system import System
from pyrpmd.trajectory import TrajectoryStream, TrajectoryError
from pyrpmd.trajectory_export import (TrajectoryFrame, export_materials_trajectory,
                                     frames_from_states, write_trajectory)


def system(periodic=True):
    return System(("O", "D"), np.array([16., 2.014]) * AMU_TO_AU,
                  np.array([[0.,0.,0.],[2.,0.,0.]]),
                  cell=np.array([[12.,0.,0.],[2.,13.,0.],[1.,3.,14.]]) if periodic else None)


def frames():
    for index in range(3):
        yield TrajectoryFrame(np.array([[25. + index, 2., 3.], [4., 5., 6.]]),
                              7 + index * 4, index * 20., np.full((2, 3), .001 + index * .0001))


@pytest.mark.parametrize("periodic", [False, True])
def test_extxyz_ase_roundtrip_preserves_cell_isotopes_and_units(tmp_path, periodic):
    pytest.importorskip("ase")
    from ase.io import read
    model = system(periodic)
    path = tmp_path / "export.extxyz"
    result = write_trajectory(path, frames(), model, representation="centroid")
    assert result["frames"] == 3
    decoded = read(path, ":", format="extxyz")
    for expected, actual in zip(frames(), decoded):
        np.testing.assert_allclose(actual.positions, expected.positions * BOHR_ANGSTROM, atol=1e-14)
        np.testing.assert_allclose(actual.get_masses(), model.masses / AMU_TO_AU)
        np.testing.assert_allclose(actual.arrays["velocities"], expected.velocities * BOHR_ANGSTROM * FS_TO_AU)
        assert actual.get_chemical_symbols() == ["O", "H"]
        assert list(actual.arrays["pyrpmd_symbol"]) == ["O", "D"]
        assert actual.info["Time"] == pytest.approx(expected.time / FS_TO_AU)
        assert actual.info["Step"] == expected.step
        assert actual.info["pyrpmd_representation"] == "centroid"
        np.testing.assert_array_equal(actual.pbc, model.pbc)
        np.testing.assert_allclose(actual.cell, np.zeros((3,3)) if model.cell is None else model.cell * BOHR_ANGSTROM)


def test_ase_native_trajectory_preserves_physical_velocities_and_masses(tmp_path):
    pytest.importorskip("ase")
    from ase.io.trajectory import Trajectory
    from ase import units
    model = system()
    path = tmp_path / "export.traj"
    write_trajectory(path, frames(), model)
    with Trajectory(path) as saved:
        assert len(saved) == 3
        for expected, actual in zip(frames(), saved):
            np.testing.assert_allclose(actual.get_velocities() * units.fs,
                                       expected.velocities * BOHR_ANGSTROM * FS_TO_AU)
            np.testing.assert_allclose(actual.get_masses(), [16.,2.014])
            assert actual.info["pyrpmd_symbols"] == ["O","D"]


@pytest.mark.parametrize("periodic", [False, True])
def test_h5md_mdanalysis_roundtrip_and_hardlinked_clocks(tmp_path, periodic):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("MDAnalysis")
    from MDAnalysis.coordinates.H5MD import H5MDReader
    model = system(periodic)
    path = tmp_path / "export.h5md"
    write_trajectory(path, frames(), model, representation="centroid")
    with h5py.File(path) as saved:
        p = saved["particles/all"]
        assert p["position/step"].id == p["velocity/step"].id
        np.testing.assert_array_equal(p["species"], [8,1])
        np.testing.assert_allclose(p["mass"], [16.,2.014])
        if periodic:
            assert p["box/edges/time"].id == p["position/time"].id
        np.testing.assert_array_equal(saved["h5md"].attrs["version"], [1,0])
    with H5MDReader(path) as saved:
        assert len(saved) == 3
        for expected, actual in zip(frames(), saved):
            np.testing.assert_allclose(actual.positions, expected.positions * BOHR_ANGSTROM, rtol=1e-6)
            np.testing.assert_allclose(actual.velocities, expected.velocities * BOHR_ANGSTROM * PS_TO_AU, rtol=1e-6)
            assert actual.time == pytest.approx(expected.time / PS_TO_AU)
            if periodic:
                np.testing.assert_allclose(actual.triclinic_dimensions, model.cell * BOHR_ANGSTROM, atol=2e-6)
            else:
                assert actual.dimensions is None


def test_variable_cell_and_partial_pbc_are_preserved_in_h5md(tmp_path):
    h5py = pytest.importorskip("h5py")
    model = system()
    model.pbc = (True, False, True)
    values = list(frames())
    for index, frame in enumerate(values):
        frame.cell = model.cell * (1 + index * .02)
    path = tmp_path / "variable.h5md"
    write_trajectory(path, values, model)
    with h5py.File(path) as saved:
        np.testing.assert_array_equal(saved["particles/all/box"].attrs["boundary"], [b"periodic",b"none",b"periodic"])
        np.testing.assert_allclose(saved["particles/all/box/edges/value"],
                                   np.array([f.cell for f in values]) * BOHR_ANGSTROM)


@pytest.mark.parametrize("format", ["extxyz", "h5md", "traj"])
def test_failed_late_frame_leaves_existing_export_unchanged(tmp_path, format):
    if format == "traj": pytest.importorskip("ase")
    path = tmp_path / ("saved." + format)
    path.write_bytes(b"existing user data")
    values = list(frames())
    values[-1].positions[0,0] = np.nan
    with pytest.raises(ValueError, match="finite real"):
        write_trajectory(path, values, system(), overwrite=True)
    assert path.read_bytes() == b"existing user data"
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]
    with pytest.raises(FileExistsError):
        write_trajectory(path, frames(), system())


@pytest.mark.parametrize("case", ["clock", "velocities", "empty"])
def test_invalid_stream_never_publishes_output(tmp_path, case):
    values = list(frames())
    if case == "clock": values[-1].step = values[0].step
    if case == "velocities": values[-1].velocities = None
    if case == "empty": values = []
    path = tmp_path / "invalid.xyz"
    with pytest.raises(ValueError):
        write_trajectory(path, values, system())
    assert not path.exists()


def test_centroid_helper_does_not_wrap_beads_or_reuse_mutable_arrays():
    model = system()
    q = np.array([[[0.,0.,0.],[1.,0.,0.]],[[12.,0.,0.],[13.,0.,0.]]])
    state = State(q, np.ones_like(q))
    saved = next(frames_from_states([state], model))
    np.testing.assert_array_equal(saved.positions[:,0], [6.,7.])
    np.testing.assert_allclose(saved.velocities, 1 / model.masses[:,None] * np.ones((2,3)))
    state.q[:] = 999
    np.testing.assert_array_equal(saved.positions[:,0], [6.,7.])


def material_archive(tmp_path, storage):
    model = system()
    metadata = {"positions":"unwrapped_ring_polymer_centroids", "position_unit":"bohr",
                "velocity_unit":"bohr/atomic_time", "time_unit":"atomic_time",
                "production_timestep_au":5., "equilibration_steps":7, "system":model.to_dict()}
    data = list(frames())
    # Real run roots also have a run manifest, distinct from a trajectory.
    (tmp_path / "manifest.json").write_text(json.dumps({"format":"PyRPMD.run.v1"}))
    arrays = {"sample_steps": np.array([f.step for f in data],dtype=np.int64),
              "positions": np.array([f.positions for f in data]),
              "velocities":np.array([f.velocities for f in data])}
    if storage == "stream":
        directory = tmp_path / "samples/materials_0000"
        schema = {key:{"dtype":value.dtype.str, "shape":list(value.shape[1:])} for key,value in arrays.items()}
        writer = TrajectoryStream(directory, schema=schema, chunk_frames=2)
        writer.extend(arrays)
        descriptor = writer.prepare_commit()
        path = directory / "manifest.json"
        path.write_text(json.dumps({"format":"PyRPMD.materials-trajectory-stream.v1", "metadata":metadata,
                                    "trajectory":descriptor}))
        # An uncommitted later chunk must not appear in an export.
        writer.append({key:value[-1] for key,value in arrays.items()})
        writer.prepare_commit()
    else:
        path = tmp_path / "samples/materials_0000.npz"
        atomic_savez(path, kind="PyRPMD.materials-trajectory.v1", **arrays,
                     times=(arrays["sample_steps"]-7)*5., metadata=json.dumps(metadata))
    return path, arrays


@pytest.mark.parametrize("storage", ["stream", "memory"])
def test_export_saved_materials_prefix_and_cli(tmp_path, storage, capsys):
    pytest.importorskip("ase")
    from ase.io import read
    from pyrpmd.cli import main
    source, expected = material_archive(tmp_path, storage)
    output = tmp_path / "export.xyz"
    assert main(["export", str(tmp_path), str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["frames"] == 3 and report["representation"] == "centroid"
    saved = read(output, ":")
    np.testing.assert_allclose(np.array([f.positions for f in saved]), expected["positions"] * BOHR_ANGSTROM)
    assert saved[0].info["Time"] == 0.


def test_corrupted_source_chunk_cannot_replace_export(tmp_path):
    source, _ = material_archive(tmp_path, "stream")
    descriptor = json.loads(source.read_text())["trajectory"]
    (source.parent / (descriptor["head"] + ".npz")).write_bytes(b"corrupted")
    output = tmp_path / "output.extxyz"
    output.write_text("previous export")
    with pytest.raises(TrajectoryError, match="digest mismatch"):
        export_materials_trajectory(source, output, overwrite=True)
    assert output.read_text() == "previous export"
