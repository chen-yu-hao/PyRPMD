"""Offline PIMD inspection, reversible storage migration and exact continuation."""
from __future__ import annotations

import builtins
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.checkpoint import CheckpointError, atomic_savez, load_checkpoint
from pyrpmd.dynamics import Dynamics
from pyrpmd.pimd_checkpoint import inspect_pimd_checkpoint, main, migrate_legacy_checkpoint
from pyrpmd.system import System
from pyrpmd.workflows import materials


def _case(*, npt=False):
    system = System(("H", "He"), [2., 5.], [[.1, .2, -.1], [.4, -.2, .3]],
                    cell=np.diag([4., 5., 6.]) if npt else None,
                    pbc=(True, True, True) if npt else None)
    config = {"task": "pimd", "beta": 2., "beads": 3, "dt_au": .02,
              "seed": 713, "native": False,
              "potential": {"provider": "free"} if npt else {"provider": "harmonic", "params": {"k": .4}},
              "sampling": {"equilibration_steps": 4, "production_steps": 12,
                           "sample_interval": 2, "replicas": 1,
                           "thermostat": "pile-l", "gamma_centroid": .6},
              "dynamics": {"loop_backend": "python"},
              "output": {"checkpoint_steps": 2, "chunk_frames": 2, "trajectory_storage": "memory"}}
    if npt:
        config["barostat"] = {"pressure": .01, "time_constant": 2., "friction": .5}
    return config, system


def _sample(config, system, directory):
    return materials._sample_replica({"config": config, "system": system,
                                      "output": str(directory), "replica": 0})


def _checkpoint(directory):
    return Path(directory) / "checkpoints" / "materials_0000.npz"


def _payload(path):
    with load_checkpoint(path) as data:
        return data.envelope.get("kind"), {name: data[name] for name in data.files}


def _same_payload(left, right):
    assert set(left) == set(right)
    for name in left:
        assert left[name].dtype == right[name].dtype, name
        assert left[name].shape == right[name].shape, name
        assert left[name].tobytes(order="C") == right[name].tobytes(order="C"), name


def _same_trajectory(left, right):
    for name in ("positions", "velocities", "observables", "radii_gyration_squared", "sample_steps", "times"):
        np.testing.assert_array_equal(left[name], right[name])
    for name in ("q", "p"):
        np.testing.assert_array_equal(getattr(left["state"], name), getattr(right["state"], name))
    assert left["state"].step == right["state"].step
    assert left["state"].time == right["state"].time
    assert left["rng"] == right["rng"]
    assert left["propagation"] == right["propagation"]
    assert left["barostat"] == right["barostat"]
    for name in ("cells", "pressures", "volumes"):
        if name in left:
            np.testing.assert_array_equal(left[name], right[name])


def _interrupt(config, system, directory, monkeypatch, stop):
    original = materials.atomic_savez

    def interrupted(path, **arrays):
        original(path, **arrays)
        if arrays.get("elapsed") == stop:
            raise RuntimeError("stopped after a committed checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(materials, "atomic_savez", interrupted)
        with pytest.raises(RuntimeError, match="committed checkpoint"):
            _sample(config, system, directory)


@pytest.mark.parametrize("npt", [False, True])
@pytest.mark.parametrize("stop", [None, 2, 8])
def test_memory_stream_roundtrip_preserves_every_state_byte_and_continues_exactly(
        tmp_path, monkeypatch, npt, stop):
    config, system = _case(npt=npt)
    reference = _sample(config, system, tmp_path / "reference")
    directory = tmp_path / "migrated"
    if stop is None:
        _sample(config, system, directory)
    else:
        _interrupt(config, system, directory, monkeypatch, stop)
    path = _checkpoint(directory)
    kind, original = _payload(path)
    # Metadata formatting and unknown auxiliaries must survive without being
    # regenerated from a reduced workflow view. BZP embeds a second RNG here.
    for name in ("rng", "barostat", "propagation"):
        if name in original:
            original[name] = np.asarray(json.dumps(json.loads(str(original[name])), indent=2) + "\n")
    original["future_auxiliary"] = np.array([0, 7, 2**32 - 1], dtype=np.uint32)
    atomic_savez(path, kind=kind, **original)
    original_config = copy.deepcopy(config)

    def unavailable(*args, **kwargs):
        pytest.fail("storage migration constructed a PES or execution engine")

    with monkeypatch.context() as patch:
        patch.setattr(materials, "make_task_potential", unavailable)
        patch.setattr(Dynamics, "__init__", unavailable)
        streamed = migrate_legacy_checkpoint(config, system, directory, storage="stream")
        assert streamed["storage"] == "stream"
        assert streamed["elapsed"] == (16 if stop is None else stop)
        assert streamed["frames"] == (0 if stop == 2 else 3 if stop == 8 else 7)
        migrated_kind, migrated = _payload(path)
        assert migrated_kind == materials.STREAM_CHECKPOINT_KIND
        history = set(materials._trajectory_schema(system, materials.resolve_materials(config, system)))
        _same_payload({name: value for name, value in original.items() if name not in history},
                      {name: value for name, value in migrated.items() if name != "trajectory"})
        committed = path.read_bytes()
        migrate_legacy_checkpoint(config, system, directory, storage="stream")
        assert path.read_bytes() == committed
        restored = migrate_legacy_checkpoint(config, system, directory, storage="memory")
        assert restored["storage"] == "memory"
    assert config == original_config
    restored_kind, restored_payload = _payload(path)
    assert restored_kind == kind
    _same_payload(original, restored_payload)
    # The uninterrupted independently executed trajectory is the continuation
    # reference; matching only a serializer round trip would not test RNG use.
    _same_trajectory(reference, _sample(config, system, directory))


def test_failed_migration_leaves_original_checkpoint_and_resume_intact(tmp_path, monkeypatch):
    config, system = _case()
    reference = _sample(config, system, tmp_path / "reference")
    directory = tmp_path / "partial"
    _interrupt(config, system, directory, monkeypatch, 8)
    path = _checkpoint(directory)
    original = path.read_bytes()
    writer = materials.atomic_savez

    def fail_publish(path, **arrays):
        if arrays.get("kind") == materials.STREAM_CHECKPOINT_KIND:
            raise OSError("failed final checkpoint publication")
        return writer(path, **arrays)

    with monkeypatch.context() as patch:
        patch.setattr(materials, "atomic_savez", fail_publish)
        with pytest.raises(OSError, match="final checkpoint publication"):
            migrate_legacy_checkpoint(config, system, directory, storage="stream")
    assert path.read_bytes() == original
    # Any orphan chunks are invisible to the original authoritative memory file.
    _same_trajectory(reference, _sample(config, system, directory))


@pytest.mark.parametrize("damage", ["rng", "q_precision", "q_shape", "clock", "fingerprint", "history"])
def test_invalid_legacy_state_is_rejected_before_replacement(tmp_path, damage):
    config, system = _case()
    _sample(config, system, tmp_path)
    path = _checkpoint(tmp_path)
    kind, payload = _payload(path)
    if damage == "rng":
        rng = json.loads(str(payload["rng"]))
        rng["state"]["inc"] = 0
        payload["rng"] = np.asarray(json.dumps(rng))
    elif damage == "q_precision":
        payload["q"] = payload["q"].astype(np.float32)
    elif damage == "q_shape":
        payload["q"] = payload["q"][..., :2]
    elif damage == "clock":
        payload["elapsed"] = np.asarray(True)
    elif damage == "fingerprint":
        payload["fingerprint"] = np.asarray("0" * 64)
    else:
        payload["positions"][0, 0, 0] = np.nan
    atomic_savez(path, kind=kind, **payload)
    original = path.read_bytes()
    with pytest.raises((CheckpointError, ValueError)):
        migrate_legacy_checkpoint(config, system, tmp_path, storage="stream")
    assert path.read_bytes() == original
    assert not (tmp_path / "samples" / "materials_0000").exists()


def test_corrupt_committed_stream_prefix_is_rejected_before_materializing(tmp_path):
    config, system = _case()
    _sample(config, system, tmp_path)
    saved = migrate_legacy_checkpoint(config, system, tmp_path, storage="stream")
    path = _checkpoint(tmp_path)
    original = path.read_bytes()
    chunk = Path(saved["trajectory_directory"]) / f"{saved['trajectory']['head']}.npz"
    chunk.write_bytes(b"damaged trajectory chunk")
    # Inspection reports its local scope honestly; migration must authenticate
    # external data and fail before publishing the target archive.
    report = inspect_pimd_checkpoint(path)
    assert report["external_trajectory_verified"] is False
    with pytest.raises(CheckpointError, match="digest mismatch"):
        migrate_legacy_checkpoint(config, system, tmp_path, storage="memory")
    assert path.read_bytes() == original


@pytest.mark.parametrize("engine", ["native", "torch"])
def test_migration_never_changes_engine_or_recreates_an_output(tmp_path, engine):
    config, system = _case()
    config["sampling"]["engine"] = engine
    directory = tmp_path / "absent"
    with pytest.raises(ValueError, match="cannot change engines"):
        migrate_legacy_checkpoint(config, system, directory, storage="stream")
    assert not directory.exists()


def test_inspection_reports_legacy_integrity_without_coordinates_or_rng_values(tmp_path, capsys):
    config, system = _case()
    _sample(config, system, tmp_path)
    path = _checkpoint(tmp_path)
    original = path.read_bytes()
    report = inspect_pimd_checkpoint(path)
    assert report["engine"] == "legacy"
    assert report["container_integrity"] == "verified"
    assert report["resume_validated"] is False
    assert report["rng"] == {"algorithm": "PCG64", "generators": 1}
    assert report["arrays"]["q"]["shape"] == [3, 2, 3]
    assert path.read_bytes() == original
    assert main(["inspect", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == report
    _, arrays = _payload(path)
    legacy = tmp_path / "unenveloped.npz"
    np.savez(legacy, **arrays)
    assert inspect_pimd_checkpoint(legacy)["container_integrity"] == "unverified_legacy"


def test_native_inspector_accepts_old_and_new_containers_without_runtime_import(tmp_path, monkeypatch):
    pytest.importorskip("pyrpmd._thermal")
    from pyrpmd.native_ensemble import NativeEnsemble

    engine = NativeEnsemble(np.zeros((3, 2, 3)), [2., 5.], 2., .02,
                            potential={"kind": "harmonic", "k": .4}, seed=17)
    engine.run(5)
    current, old = tmp_path / "current.npz", tmp_path / "old.npz"
    engine.save(current)
    _, payload = _payload(current)
    np.savez(old, **payload)
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch.") or name.endswith("_thermal"):
            pytest.fail("inspection imported an execution runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    for path in (old, current):
        report = inspect_pimd_checkpoint(path)
        assert report["engine"] == "native"
        assert report["step"] == 5
        assert report["state_digest_verified"] is True
    payload["q"][0, 0, 0] += .1
    np.savez(old, **payload)
    with pytest.raises(CheckpointError, match="digest differs"):
        inspect_pimd_checkpoint(old)


def test_torch_inspector_does_not_require_torch_or_external_chunks(tmp_path, monkeypatch):
    from pyrpmd.trajectory import TrajectoryStream

    config, system = _case()
    schema = materials._trajectory_schema(system, materials.resolve_materials(config, system))
    descriptor = TrajectoryStream(tmp_path / "missing", schema=schema).prepare_commit()
    runtime = {"torch": "2.5.1", "device": "cpu", "dtype": "float64", "platform": "Linux",
               "machine": "x86_64", "threads": 1, "interop_threads": 1,
               "normal_mode_layout": "rfft", "deterministic_algorithms": True, "cuda": None}
    path = tmp_path / "tensor.npz"
    atomic_savez(path, kind="PyRPMD.materials-torch.v1", fingerprint="1" * 64,
                 sources="{}", runtime=json.dumps(runtime), q=np.zeros((1, 3, 2, 3)),
                 p=np.zeros((1, 3, 2, 3)), step=0, time=0., production_started=False,
                 rng_sampling=np.arange(4, dtype=np.uint8), rng_production=np.arange(5, dtype=np.uint8),
                 trajectories=json.dumps([descriptor]))
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("inspection imported Torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    report = inspect_pimd_checkpoint(path)
    assert report["engine"] == "torch"
    assert report["rng"]["generators"] == 2
    assert report["external_trajectory_verified"] is False
    assert report["resume_validated"] is False


def test_cli_partial_migration_preserves_saved_inputs_and_resumes_exactly(tmp_path, monkeypatch, capsys):
    from pyrpmd.runner import _initialize_output, resume_run

    config, system = _case()
    reference = _sample(config, system, tmp_path / "reference")
    directory = tmp_path / "public"
    _initialize_output(config, system, directory)
    _interrupt(config, system, directory, monkeypatch, 8)
    input_files = [directory / "manifest.json", *sorted((directory / "input").iterdir())]
    originals = {path: path.read_bytes() for path in input_files}
    assert main(["migrate", str(directory), "--storage", "stream"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["step"] == 8 and report["complete"] is False
    assert report["configured_resume_storage"] == "memory"
    assert originals == {path: path.read_bytes() for path in input_files}
    # Normal resume honors the unchanged input's storage preference and must
    # continue the original stochastic trajectory even after offline migration.
    resume_run(directory)
    _same_trajectory(reference, materials._read_stream(config, system, directory, 0))


def test_cli_rejects_changed_manifest_before_migration(tmp_path, monkeypatch, capsys):
    from pyrpmd.runner import _initialize_output

    config, system = _case()
    _initialize_output(config, system, tmp_path)
    _interrupt(config, system, tmp_path, monkeypatch, 8)
    path = _checkpoint(tmp_path)
    original = path.read_bytes()
    manifest = tmp_path / "manifest.json"
    value = json.loads(manifest.read_text())
    value["physics_fingerprint"] = "0" * 64
    manifest.write_text(json.dumps(value))
    with pytest.raises(SystemExit) as error:
        main(["migrate", str(tmp_path), "--storage", "stream"])
    assert error.value.code == 2
    assert "changed scientific settings" in capsys.readouterr().err
    assert path.read_bytes() == original
