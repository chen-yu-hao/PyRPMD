"""One complete validation traversal supplies tensor materials analysis."""

from collections import Counter
import builtins
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from pyrpmd.checkpoint import CheckpointError, atomic_savez, load_checkpoint
from pyrpmd.system import System
from pyrpmd.trajectory import CHUNK_KIND
from pyrpmd.workflows import torch_materials as tm
from pyrpmd.workflows.materials import run_materials


def _case(task="diffusion"):
    system = System(("H", "He", "Ne", "Ar"), np.array([2., 5., 12., 22.]),
                    np.array([[.1, .2, -.1], [.3, -.1, .2], [-.2, .4, .1], [.5, .2, .3]]))
    config = {"task": task, "beta": 2., "beads": 4, "dt_au": .04, "seed": 41,
              "potential": {"provider": "harmonic", "params": {"k": .7}},
              "sampling": {"engine": "torch", "device": "cpu", "integrator": "bcocb",
                           "thermostat": "pile-l", "gamma_centroid": .7,
                           "equilibration_steps": 5, "production_steps": 24,
                           "sample_interval": 2, "replicas": 2},
              "dynamics": {"integrator": "cayley-bcb", "block_steps": 7},
              "output": {"checkpoint_steps": 4, "chunk_frames": 3}}
    if task == "diffusion":
        # Preserve measured order and include a separate reference atom;
        # atom 3 must not acquire a full position/velocity history by default.
        config["diffusion"] = {"atom_indices": [2, 0], "reference_atoms": [1],
                               "axes": [2, 0], "max_lag": 1, "bootstrap": 0}
    return config, system


def _count_reads(monkeypatch):
    import pyrpmd.trajectory as trajectory
    original = trajectory._load_chunk
    reads = Counter()

    def counted(directory, digest, *args, **kwargs):
        reads[(str(directory), digest)] += 1
        return original(directory, digest, *args, **kwargs)

    monkeypatch.setattr(trajectory, "_load_chunk", counted)
    return reads


def _forbid_execution(monkeypatch):
    import pyrpmd.potentials as potentials

    def unexpected(*args, **kwargs):
        raise AssertionError("analysis must not construct a provider, execute Torch, or save a checkpoint")

    monkeypatch.setattr(potentials, "make_potential", unexpected)
    monkeypatch.setattr(tm, "atomic_savez", unexpected)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            unexpected()
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def _assert_once(reads, descriptors):
    assert len(reads) == sum(descriptor["chunks"] for descriptor in descriptors)
    assert set(reads.values()) == {1}


def _assert_analysis_equal(actual, reference, task):
    if task == "diffusion":
        assert actual["diffusion"] == reference["diffusion"]
        assert actual["nve_energy_diagnostics"] == reference["nve_energy_diagnostics"]
    else:
        assert actual["estimators"] == reference["estimators"]
        np.testing.assert_allclose(actual["radius_of_gyration_squared_bohr2"],
                                   reference["radius_of_gyration_squared_bohr2"], rtol=2e-15)
    assert actual["status"] == reference["status"]


@pytest.mark.parametrize("task", ["pimd", "diffusion"])
@pytest.mark.parametrize("entrypoint", ["replay", "offline"])
@pytest.mark.parametrize("return_data", [False, True])
def test_complete_analysis_reads_each_chunk_once_without_torch_or_provider(
        tmp_path, monkeypatch, task, entrypoint, return_data):
    config, system = _case(task)
    reference = run_materials(config, system, tmp_path, return_data=True)
    descriptors = [stream["trajectory"] for stream in reference["streams"]]
    committed = tm._path(tmp_path).read_bytes()
    reads = _count_reads(monkeypatch)
    _forbid_execution(monkeypatch)

    analyze = run_materials if entrypoint == "replay" else tm.analyze_torch_materials
    result = analyze(config, system, tmp_path, return_data=return_data)

    _assert_once(reads, descriptors)
    assert tm._path(tmp_path).read_bytes() == committed
    summary = result["summary"] if return_data else result
    _assert_analysis_equal(summary, reference["summary"], task)
    if return_data:
        for actual, expected in zip(result["streams"], reference["streams"]):
            for name in (*actual["trajectory"]["schema"], "times"):
                np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize("task", ["pimd", "diffusion"])
@pytest.mark.parametrize("materialize", [False, True])
def test_collected_histories_are_full_only_when_explicitly_requested(
        tmp_path, task, materialize):
    config, system = _case(task)
    reference = run_materials(config, system, tmp_path, return_data=True)
    saved = tm._read_batch(config, system, tmp_path, analysis=True, materialize=materialize)

    for data, expected in zip(saved["analysis_data"], reference["streams"]):
        if materialize:
            assert set(data) == set(expected["trajectory"]["schema"])
            for name in data:
                np.testing.assert_array_equal(data[name], expected[name])
        else:
            fields = {"sample_steps", "observables", "radii_gyration_squared"}
            if task == "diffusion":
                fields |= {"positions", "velocities", "_analysis_atoms"}
                assert data["_analysis_atoms"] == [0, 1, 2]
                for name in ("positions", "velocities"):
                    assert data[name].shape[1] == 3 < system.natoms
                    np.testing.assert_array_equal(data[name], expected[name][:, [0, 1, 2]])
            assert set(data) == fields
            assert data["radii_gyration_squared"].shape == (1, system.natoms)
            np.testing.assert_allclose(data["radii_gyration_squared"][0],
                                       expected["radii_gyration_squared"].mean(axis=0), rtol=2e-15)


def _interrupt_after_commit(config, system, output, monkeypatch, step=20):
    original = tm._save

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        if args[3].step == step:
            raise RuntimeError("injected interruption after committed production")

    with monkeypatch.context() as patch:
        patch.setattr(tm, "_save", interrupted)
        with pytest.raises(RuntimeError, match="injected interruption"):
            run_materials(config, system, output)
    return tm._read_batch(config, system, output)


def test_partial_propagation_restart_only_validates_the_existing_prefix(tmp_path, monkeypatch):
    from pyrpmd.workflows import materials
    config, system = _case()
    saved = _interrupt_after_commit(config, system, tmp_path, monkeypatch)
    reads = _count_reads(monkeypatch)

    def unexpected(*args, **kwargs):
        raise AssertionError("partial propagation restart must not retain old analysis histories")

    def stop_before_propagation(*args, saved, **kwargs):
        assert not saved["complete"] and saved["step"] == 20
        assert "analysis_data" not in saved
        _assert_once(reads, saved["trajectories"])
        raise RuntimeError("validated partial restart boundary")

    monkeypatch.setattr(materials, "_collect_frames", unexpected)
    monkeypatch.setattr(tm, "_run_batch", stop_before_propagation)
    with pytest.raises(RuntimeError, match="validated partial restart boundary"):
        run_materials(config, system, tmp_path, return_data=True)


def test_partial_checkpoint_validation_does_not_import_torch(tmp_path, monkeypatch):
    config, system = _case()
    original = _interrupt_after_commit(config, system, tmp_path, monkeypatch)
    reads = _count_reads(monkeypatch)
    _forbid_execution(monkeypatch)

    saved = tm._read_batch(config, system, tmp_path, analysis=True,
                           materialize=True, only_complete=True)

    _assert_once(reads, original["trajectories"])
    assert not saved["complete"] and "analysis_data" not in saved
    assert saved["step"] == original["step"]
    np.testing.assert_array_equal(saved["q"], original["q"])
    np.testing.assert_array_equal(saved["p"], original["p"])


@pytest.mark.parametrize("return_data", [False, True])
def test_partial_offline_analysis_collects_once_without_advancing(tmp_path, monkeypatch, return_data):
    config, system = _case()
    saved = _interrupt_after_commit(config, system, tmp_path, monkeypatch)
    reference = tm.analyze_torch_materials(config, system, tmp_path, return_data=True)
    committed = tm._path(tmp_path).read_bytes()
    reads = _count_reads(monkeypatch)
    _forbid_execution(monkeypatch)

    result = tm.analyze_torch_materials(config, system, tmp_path, return_data=return_data)

    _assert_once(reads, saved["trajectories"])
    assert tm._path(tmp_path).read_bytes() == committed
    summary = result["summary"] if return_data else result
    assert summary["status"] == "partial"
    assert summary["samples_per_replica"] == [8, 8]
    _assert_analysis_equal(summary, reference["summary"], "diffusion")
    if return_data:
        for actual, expected in zip(result["streams"], reference["streams"]):
            for name in (*actual["trajectory"]["schema"], "times"):
                np.testing.assert_array_equal(actual[name], expected[name])


def _corrupt_head(output, field):
    """Keep both content hashes and NPZ integrity valid, but violate semantics."""
    with load_checkpoint(tm._path(output)) as saved:
        checkpoint = {name: saved[name] for name in saved.files}
    descriptors = json.loads(str(checkpoint["trajectories"]))
    directory = Path(output) / "samples/materials_0001"
    with load_checkpoint(directory / f"{descriptors[1]['head']}.npz") as saved:
        chunk = {name: saved[name] for name in saved.files}
    if field == "sample_steps":
        chunk[field][0] += 1
    elif field == "radii_gyration_squared":
        chunk[field][0, 3] = np.nan
    else:
        chunk[field][0, 3, 0] = np.nan  # A deliberately unselected atom.
    temporary = directory / "invalid-semantics.npz"
    atomic_savez(temporary, kind=CHUNK_KIND, **chunk)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.replace(directory / f"{digest}.npz")
    descriptors[1]["head"] = digest
    checkpoint["trajectories"] = json.dumps(descriptors)
    atomic_savez(tm._path(output), kind=tm.KIND, **checkpoint)


@pytest.mark.parametrize("task", ["pimd", "diffusion"])
@pytest.mark.parametrize("field", ["sample_steps", "positions", "velocities", "radii_gyration_squared"])
def test_reduced_analysis_still_rejects_invalid_dropped_fields_before_publishing(
        tmp_path, task, field):
    config, system = _case(task)
    run_materials(config, system, tmp_path)
    _corrupt_head(tmp_path, field)
    published = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    committed = tm._path(tmp_path).read_bytes()

    with pytest.raises(CheckpointError, match="trajectory clock|invalid trajectory chunk values"):
        tm.analyze_torch_materials(config, system, tmp_path)

    assert tm._path(tmp_path).read_bytes() == committed
    assert {path: path.read_bytes() for path in tmp_path.rglob("*.json")} == published


@pytest.mark.parametrize("return_data", [False, True])
def test_analysis_checks_older_chunk_integrity_before_replacing_results(tmp_path, return_data):
    config, system = _case()
    reference = run_materials(config, system, tmp_path, return_data=True)
    descriptor = reference["streams"][0]["trajectory"]
    directory = tmp_path / "samples/materials_0000"
    digest = descriptor["head"]
    while True:
        path = directory / f"{digest}.npz"
        with load_checkpoint(path) as saved:
            previous = json.loads(str(saved["metadata"]))["previous"]
        if previous is None:
            break
        digest = previous
    assert digest != descriptor["head"]
    path.write_bytes(b"corrupt older trajectory chunk")
    published = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    committed = tm._path(tmp_path).read_bytes()

    with pytest.raises(CheckpointError, match="digest mismatch"):
        tm.analyze_torch_materials(config, system, tmp_path, return_data=return_data)

    assert tm._path(tmp_path).read_bytes() == committed
    assert {path: path.read_bytes() for path in tmp_path.rglob("*.json")} == published


def test_analysis_selection_can_change_without_resampling_or_changing_checkpoint(tmp_path, monkeypatch):
    config, system = _case()
    run_materials(config, system, tmp_path)
    changed = copy.deepcopy(config)
    changed["diffusion"].update(atom_indices=[3], reference_atoms=[1, 0], axes=[1, 2])
    reference = tm.analyze_torch_materials(changed, system, tmp_path, return_data=True)
    committed = tm._path(tmp_path).read_bytes()
    reads = _count_reads(monkeypatch)
    _forbid_execution(monkeypatch)

    reduced = tm.analyze_torch_materials(changed, system, tmp_path)

    _assert_once(reads, [stream["trajectory"] for stream in reference["streams"]])
    assert reduced["diffusion"]["input"]["atom_indices"] == [3]
    _assert_analysis_equal(reduced, reference["summary"], "diffusion")
    assert tm._path(tmp_path).read_bytes() == committed


@pytest.mark.parametrize("task", ["pimd", "diffusion"])
@pytest.mark.parametrize("dt", [.04, 1e-20])
def test_resolved_workflow_rejects_different_sampling_timestep(task, dt):
    config, system = _case(task)
    config["dt_au"] = dt
    config["sampling"]["dt_au"] = 2 * dt
    with pytest.raises(ValueError, match="timesteps must match"):
        tm.resolve_torch_materials(config, system)
