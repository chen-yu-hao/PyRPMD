"""Actual propagation and streamed storage share one committed phase clock."""

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.system import System
from pyrpmd.trajectory import CHUNK_KIND
from pyrpmd.workflows.materials import (
    STREAM_CHECKPOINT_KIND, _sample_replica, _read_stream, run_materials,
)


def setup(task="pimd"):
    system = System(("H", "Ni", "H"), np.array([2., 20., 2.]),
                    np.array([[.1, 0., 0.], [1., 0., 0.], [.3, .4, 0.]]))
    config = {
        "task": task, "beta": 2., "beads": 3, "dt_au": .03,
        "seed": 732, "native": False,
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": .4}},
        "sampling": {"equilibration_steps": 2, "production_steps": 24, "sample_interval": 1,
                     "replicas": 1, "gamma_centroid": .6, "thermostat": "pile-l"},
        "dynamics": {"loop_backend": "python"},
        "output": {"checkpoint_steps": 4, "chunk_frames": 3},
    }
    if task == "diffusion":
        config["diffusion"] = {"atom_indices": [2], "reference_atoms": [1], "axes": [2, 0], "bootstrap": 0}
    return config, system


def run_replica(config, system, path, *, materialize=True):
    return _sample_replica({"config": config, "system": system, "output": str(path),
                            "replica": 0, "materialize": materialize})


def assert_same(left, right):
    for name in ("positions", "velocities", "sample_steps", "observables", "radii_gyration_squared", "times"):
        np.testing.assert_array_equal(left[name], right[name])
    for name in ("q", "p"):
        np.testing.assert_array_equal(getattr(left["state"], name), getattr(right["state"], name))
    assert left["state"].step == right["state"].step
    assert left["state"].time == right["state"].time
    assert left["rng"] == right["rng"]
    assert left["propagation"] == right["propagation"]


def test_chunk_publish_failure_restores_sample_state_and_rng_before_saving(tmp_path, monkeypatch):
    import pyrpmd.trajectory as storage

    config, system = setup()
    reference = run_replica(config, system, tmp_path / "reference")
    original = storage.atomic_savez
    def interrupted(path, **arrays):
        if arrays.get("kind") == CHUNK_KIND and arrays["sample_steps"][-1] == 7:
            raise OSError("injected chunk publish failure")
        return original(path, **arrays)
    directory = tmp_path / "resumed"
    with monkeypatch.context() as patch:
        patch.setattr(storage, "atomic_savez", interrupted)
        with pytest.raises(OSError, match="chunk publish failure"):
            run_replica(config, system, directory)
    saved = _read_stream(config, system, directory, 0)
    assert saved["elapsed"] == 6
    assert saved["sample_steps"][-1] == saved["state"].step == 6
    config["output"].update(chunk_frames=2, checkpoint_steps=7)
    resumed = run_replica(config, system, directory)
    assert_same(reference, resumed)


def test_checkpoint_replacement_failure_recovers_only_its_old_prefix(tmp_path, monkeypatch):
    import pyrpmd.workflows.materials as materials

    config, system = setup("diffusion")
    reference = run_replica(config, system, tmp_path / "reference")
    original = materials.atomic_savez
    def interrupted(path, **arrays):
        if arrays.get("kind") == STREAM_CHECKPOINT_KIND and arrays["elapsed"] == 8:
            raise OSError("injected checkpoint replacement failure")
        return original(path, **arrays)
    directory = tmp_path / "resumed"
    with monkeypatch.context() as patch:
        patch.setattr(materials, "atomic_savez", interrupted)
        with pytest.raises(OSError, match="checkpoint replacement failure"):
            run_replica(config, system, directory)
    committed = _read_stream(config, system, directory, 0, materialize=False)
    assert committed["elapsed"] == 4 and committed["frames"] == 3
    assert len(list((directory / "samples/materials_0000").glob("*.npz"))) > committed["trajectory"]["chunks"]
    config["output"].update(chunk_frames=2, checkpoint_steps=7)
    resumed = run_replica(config, system, directory)
    assert_same(reference, resumed)


def test_compact_worker_results_do_not_materialize_any_history(tmp_path, monkeypatch):
    import pyrpmd.workflows.materials as materials

    config, system = setup()
    monkeypatch.setattr(materials, "materialize_stream", lambda *a, **k: pytest.fail("worker materialized trajectory"))
    stream = run_replica(config, system, tmp_path, materialize=False)
    assert stream["frames"] == stream["trajectory"]["frames"] == 25
    for name in ("positions", "velocities", "sample_steps", "observables", "radii_gyration_squared", "times"):
        assert name not in stream
    # Completed resume validates saved data but also stays compact.
    completed = run_replica(config, system, tmp_path, materialize=False)
    assert completed["trajectory"] == stream["trajectory"]


def test_migration_of_completed_prefix_does_not_reconstruct_a_potential(tmp_path, monkeypatch):
    import pyrpmd.workflows.materials as materials

    config, system = setup()
    config["output"]["trajectory_storage"] = "memory"
    reference = run_replica(config, system, tmp_path)
    checkpoint = tmp_path / "checkpoints/materials_0000.npz"
    config["output"]["trajectory_storage"] = "stream"
    monkeypatch.setattr(materials, "make_task_potential", lambda *a, **k: pytest.fail("storage migration constructed PES"))
    migrated = run_replica(config, system, tmp_path)
    with load_checkpoint(checkpoint, require_integrity=True) as saved:
        assert saved.envelope["kind"] == STREAM_CHECKPOINT_KIND
        assert "positions" not in saved and "trajectory" in saved
    assert_same(reference, migrated)


def test_failed_legacy_migration_keeps_the_original_complete_checkpoint(tmp_path, monkeypatch):
    import pyrpmd.workflows.materials as materials

    config, system = setup()
    config["output"]["trajectory_storage"] = "memory"
    reference = run_replica(config, system, tmp_path)
    checkpoint = tmp_path / "checkpoints/materials_0000.npz"
    original_bytes = checkpoint.read_bytes()
    original = materials.atomic_savez
    def interrupted(path, **arrays):
        if arrays.get("kind") == STREAM_CHECKPOINT_KIND:
            raise OSError("interrupted storage migration")
        return original(path, **arrays)
    config["output"]["trajectory_storage"] = "stream"
    with monkeypatch.context() as patch:
        patch.setattr(materials, "atomic_savez", interrupted)
        with pytest.raises(OSError, match="interrupted storage migration"):
            run_replica(config, system, tmp_path)
    assert checkpoint.read_bytes() == original_bytes
    assert_same(reference, run_replica(config, system, tmp_path))


def test_selected_atom_offline_analysis_preserves_original_atom_identity_and_values(tmp_path):
    config, system = setup("diffusion")
    config["output"]["trajectory_storage"] = "memory"
    memory = run_materials(config, system, tmp_path / "memory")
    config["output"]["trajectory_storage"] = "stream"
    streamed = run_materials(config, system, tmp_path / "stream")
    assert streamed["diffusion"]["input"]["atom_indices"] == [2]
    assert streamed["diffusion"]["input"]["axes"] == [2, 0]
    assert streamed["diffusion"] == memory["diffusion"]
    assert streamed["nve_energy_diagnostics"] == memory["nve_energy_diagnostics"]
