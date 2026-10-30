"""Checkpoint integrity, atomic replacement and public resume compatibility."""

import copy
from collections import Counter
import json
from unittest import mock
import weakref

import numpy as np
import pytest

from pyrpmd.checkpoint import (
    CheckpointError, ENVELOPE, atomic_savez, inspect_checkpoint, load_checkpoint,
)
from pyrpmd.constants import KB_AU
from pyrpmd.input import InputError, JobSpec
from pyrpmd.io import fingerprint
from pyrpmd.runner import _initialize_output, resume_run, run


def collision_job():
    return {
        "task": "drpmd", "temperature": 1 / KB_AU, "beads": 1,
        "seed": 11, "native": False,
        "system": {
            "symbols": ["H", "H"], "masses": [1.0, 1.0],
            "geometry": [[0, 0, -2], [0, 0, 2]], "fragments": [[0], [1]],
        },
        "potential": {"provider": "free"},
        "dynamics": {"timestep": 0.1, "loop_backend": "python", "block_steps": 3},
        "sampling": {
            "equilibration_steps": 4, "production_steps": 4,
            "interval_steps": 2, "andersen_interval": 2, "parents": 1,
        },
        "collision": {"bmax": 0.5, "trajectories": 2, "max_steps": 200},
        "outcomes": {"bond_cutoffs": {"H-H": 0.3}, "persistence_steps": 2},
        "output": {"checkpoint_steps": 10},
    }


def raw_payload(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def test_roundtrip_verifies_all_arrays_before_exposing_payload(tmp_path):
    path = tmp_path / "state.npz"
    q = np.arange(18, dtype=np.float64).reshape(2, 3, 3)
    atomic_savez(path, kind="test.state.v1", q=q, step=np.int64(9),
                 rng_state=json.dumps(np.random.default_rng(3).bit_generator.state))
    with load_checkpoint(path, expected_kind="test.state.v1", require_integrity=True) as saved:
        assert saved.envelope["integrity"] == "verified"
        assert ENVELOPE not in saved.files
        np.testing.assert_array_equal(saved["q"], q)
        assert saved["step"].item() == 9
    report = inspect_checkpoint(path)
    assert report["schema_version"] == 2
    assert report["arrays"]["q"] == {"shape": [2, 3, 3], "dtype": "<f8"}
    assert "rng_state" in report["arrays"]
    assert "bit_generator" not in json.dumps(report)
    with pytest.raises(CheckpointError, match="Expected checkpoint kind"):
        with load_checkpoint(path, expected_kind="other.state.v1"):
            pytest.fail("wrong payload kind was admitted")


@pytest.mark.parametrize("legacy", [False, True])
def test_validated_arrays_are_decoded_once_and_reads_remain_independent(tmp_path, monkeypatch, legacy):
    path = tmp_path / "cached.npz"
    kind = "PyRPMD.test.v1"
    arrays = {"q": np.asfortranarray(np.arange(18.0).reshape(2, 3, 3)),
              "step": np.asarray(7), "history": np.arange(128.0),
              "metadata": np.asarray(json.dumps({"format": kind}))}
    if legacy:
        np.savez_compressed(path, **arrays)
    else:
        atomic_savez(path, kind=kind, **arrays)
    with np.load(path, allow_pickle=False) as archive:
        archive_type = type(archive)
    original = archive_type.__getitem__
    reads, materialized = Counter(), {}

    def counted(archive, key):
        value = original(archive, key)
        reads[key] += 1
        materialized[key] = weakref.ref(value)
        return value

    monkeypatch.setattr(archive_type, "__getitem__", counted)
    with load_checkpoint(path, expected_kind=kind) as saved:
        expected = Counter({name: 1 for name in arrays})
        if not legacy:
            expected[ENVELOPE] = 1
        # Even a member that the consumer never requests was fully validated.
        assert reads == expected
        first = saved["q"]
        assert first.flags.writeable and first.flags.f_contiguous
        first[:] = -1
        np.testing.assert_array_equal(saved["q"], arrays["q"])
        np.testing.assert_array_equal(saved.get("q"), arrays["q"])
        assert saved.get("absent", "default") == "default"
        assert list(saved) == saved.keys() == saved.files
        assert len(saved) == len(arrays)
        for name, value in saved.items():
            np.testing.assert_array_equal(value, arrays[name])
        with pytest.raises(KeyError):
            saved[ENVELOPE]
        with pytest.raises(KeyError):
            saved["absent"]
        assert reads == expected
    # Retaining the lightweight payload view must not retain the decompressed
    # history. Arrays handed to callers remain valid and independent of it.
    assert all(reference() is None for reference in materialized.values())
    np.testing.assert_array_equal(first, -np.ones_like(arrays["q"]))


def test_checkpoint_cache_is_per_open_and_still_rejects_unread_corruption(tmp_path):
    path = tmp_path / "state.npz"
    atomic_savez(path, q=np.ones(3), history=np.arange(20.0))
    with load_checkpoint(path) as saved:
        np.testing.assert_array_equal(saved["q"], np.ones(3))
    arrays = raw_payload(path)
    arrays["history"][7] += 1
    np.savez_compressed(path, **arrays)
    with pytest.raises(CheckpointError, match="integrity check failed for array 'history'"):
        with load_checkpoint(path):
            pytest.fail("an unread corrupt member bypassed validation")


@pytest.mark.parametrize("damage", ["value", "shape", "dtype", "missing", "extra"])
def test_tampered_payload_is_rejected_even_with_valid_zip_crc(tmp_path, damage):
    path = tmp_path / "state.npz"
    atomic_savez(path, q=np.arange(6, dtype=float), step=9)
    arrays = raw_payload(path)
    if damage == "value":
        arrays["q"][2] += 1
    elif damage == "shape":
        arrays["q"] = arrays["q"].reshape(2, 3)
    elif damage == "dtype":
        arrays["q"] = arrays["q"].astype("float32")
    elif damage == "missing":
        del arrays["q"]
    else:
        arrays["extra"] = np.array(1)
    np.savez_compressed(path, **arrays)
    with pytest.raises(CheckpointError, match="integrity|inventory"):
        with load_checkpoint(path):
            pytest.fail("corrupted state reached propagation")


@pytest.mark.parametrize("schema_version", [3, True, "2"])
def test_unknown_schema_is_not_silently_interpreted_as_current(tmp_path, schema_version):
    path = tmp_path / "state.npz"
    atomic_savez(path, q=np.zeros((1, 1, 3)))
    arrays = raw_payload(path)
    envelope = json.loads(str(arrays[ENVELOPE].item()))
    envelope["schema_version"] = schema_version
    arrays[ENVELOPE] = np.asarray(json.dumps(envelope))
    np.savez_compressed(path, **arrays)
    with pytest.raises(CheckpointError, match="Unsupported checkpoint schema"):
        with load_checkpoint(path):
            pass


def test_interrupted_write_preserves_previous_complete_checkpoint(tmp_path):
    path = tmp_path / "state.npz"
    atomic_savez(path, q=np.array([1.0]))
    previous = path.read_bytes()

    def interrupted(stream, **arrays):
        stream.write(b"interrupted replacement")
        raise OSError("simulated filesystem interruption")

    with mock.patch("pyrpmd.checkpoint.np.savez_compressed", side_effect=interrupted):
        with pytest.raises(OSError, match="interruption"):
            atomic_savez(path, q=np.array([2.0]))
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]
    with load_checkpoint(path, require_integrity=True) as saved:
        np.testing.assert_array_equal(saved["q"], [1.0])


def test_no_pickle_on_either_write_or_read_and_legacy_is_explicit(tmp_path):
    path = tmp_path / "state.npz"
    with pytest.raises(CheckpointError, match="object arrays"):
        atomic_savez(path, q=np.array([{"unsafe": "object"}], dtype=object))
    assert not path.exists()
    np.savez(path, q=np.array([{"unsafe": "object"}], dtype=object))
    with pytest.raises(CheckpointError, match="Object arrays"):
        with load_checkpoint(path):
            pass
    np.savez(path, q=np.ones((1, 1, 3)), metadata=json.dumps({"format": "PyRPMD.test.v1"}))
    assert inspect_checkpoint(path)["integrity"] == "unverified_legacy"
    with pytest.raises(CheckpointError, match="Legacy checkpoint"):
        inspect_checkpoint(path, require_integrity=True)
    np.savez(path, q=np.ones((1, 1, 3)), metadata=json.dumps({"format": "PyRPMD.test.v9"}))
    with pytest.raises(CheckpointError, match="Unsupported legacy"):
        inspect_checkpoint(path)


def test_public_resume_accepts_execution_changes_and_checks_state_integrity(tmp_path):
    directory = tmp_path / "run"
    first = run(collision_job(), output=directory)
    second = resume_run(directory, execution={"loop_backend": "auto", "block_steps": 7})
    assert first["records"] == second["records"]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["schema"] == "pyrpmd.run"
    assert manifest["schema_version"] == manifest["checkpoint_schema_version"] == 2
    path = next((directory / "checkpoints").glob("parent-*.npz"))
    assert inspect_checkpoint(path)["integrity"] == "verified"
    arrays = raw_payload(path)
    arrays["q"].flat[0] += 0.5
    np.savez_compressed(path, **arrays)
    with pytest.raises(CheckpointError, match="integrity check failed"):
        resume_run(directory)


def test_resume_rejects_scientific_changes_and_unknown_execution_overrides(tmp_path):
    config, system = JobSpec.from_mapping(collision_job()).resolve()
    _initialize_output(config, system, tmp_path)
    changed = copy.deepcopy(config)
    changed["dynamics"]["integrator"] = "normal-mode-bab"
    with pytest.raises(InputError, match="changed scientific settings"):
        _initialize_output(changed, system, tmp_path, resume=True)
    with pytest.raises(InputError, match="only loop_backend and block_steps"):
        resume_run(tmp_path, execution={"integrator": "normal-mode-bab"})
    with pytest.raises(InputError, match="integer"):
        resume_run(tmp_path, execution={"block_steps": True})


def test_verified_legacy_manifest_migrates_without_changing_payload(tmp_path):
    config, system = JobSpec.from_mapping(collision_job()).resolve()
    _initialize_output(config, system, tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.pop("schema")
    manifest["schema_version"] = 1
    manifest["physics_fingerprint"] = fingerprint(config, system)
    legacy_fingerprint = manifest["physics_fingerprint"]
    path.write_text(json.dumps(manifest))
    changed = copy.deepcopy(config)
    changed["dynamics"].update(loop_backend="auto", block_steps=32)
    _initialize_output(changed, system, tmp_path, resume=True)
    migrated = json.loads(path.read_text())
    assert migrated["schema_version"] == 2
    assert migrated["legacy_physics_fingerprint"] == legacy_fingerprint
    assert json.loads((tmp_path / "input" / "resolved.json").read_text())["dynamics"]["block_steps"] == 3


def test_modified_legacy_input_and_future_manifest_are_rejected(tmp_path):
    config, system = JobSpec.from_mapping(collision_job()).resolve()
    _initialize_output(config, system, tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema_version"] = 1
    manifest["physics_fingerprint"] = fingerprint(config, system)
    path.write_text(json.dumps(manifest))
    altered = copy.deepcopy(config)
    altered["temperature_K"] *= 2
    (tmp_path / "input" / "resolved.json").write_text(json.dumps(altered))
    with pytest.raises(InputError, match="Legacy saved input"):
        _initialize_output(config, system, tmp_path, resume=True)
    manifest["schema_version"] = 999
    path.write_text(json.dumps(manifest))
    with pytest.raises(InputError, match="Unsupported run manifest"):
        _initialize_output(config, system, tmp_path, resume=True)


@pytest.mark.parametrize("name, canonical", [
    ("cayley", "cayley-bcb"),
    ("normal_mode", "normal-mode-bab"),
    ("velocity_verlet", "velocity-verlet"),
])
def test_public_input_keeps_sampler_and_propagator_consistent(name, canonical):
    job = collision_job()
    job["dynamics"]["integrator"] = name
    config, _ = JobSpec.from_mapping(job).resolve()
    assert config["dynamics"]["integrator"] == canonical
    assert config["sampling"]["integrator"] == canonical + "-andersen"
    job["sampling"]["integrator"] = "unrelated-andersen"
    with pytest.raises(InputError, match="sampling.integrator"):
        JobSpec.from_mapping(job).resolve()


@pytest.mark.parametrize("integrator", ["normal-mode-bab", "velocity-verlet"])
def test_public_multibead_drpmd_resumes_selected_integrator_exactly(tmp_path, monkeypatch, integrator):
    from pyrpmd.workflows import drpmd

    job = collision_job()
    job["beads"] = 4
    job["dynamics"]["integrator"] = integrator
    job["output"]["checkpoint_steps"] = 2
    reference = run(job, output=tmp_path / "full")
    original_save = drpmd._atomic_npz

    def interrupt_after_child_checkpoint(path, metadata, **arrays):
        original_save(path, metadata, **arrays)
        if (metadata["format"] == "PyRPMD.drpmd.trajectory-checkpoint.v1"
                and metadata["done"] > 0):
            raise KeyboardInterrupt("interrupted after a persisted child step")

    directory = tmp_path / "resume"
    with monkeypatch.context() as patch:
        patch.setattr(drpmd, "_atomic_npz", interrupt_after_child_checkpoint)
        with pytest.raises(KeyboardInterrupt, match="persisted child"):
            run(job, output=directory)

    # A valid envelope must not turn a future trajectory payload into v1.
    checkpoint = next((directory / "trajectories").glob("*.checkpoint.npz"))
    with load_checkpoint(checkpoint) as saved:
        payload = {name: saved[name] for name in saved.files}
    metadata = json.loads(str(payload["metadata"].item()))
    future = {**metadata, "format": "PyRPMD.drpmd.trajectory-checkpoint.v999"}
    atomic_savez(checkpoint, kind=future["format"],
                 **{**payload, "metadata": np.asarray(json.dumps(future))})
    with pytest.raises(CheckpointError, match="Expected checkpoint kind"):
        resume_run(directory)
    atomic_savez(checkpoint, kind=metadata["format"], **payload)

    selected = []
    original_dynamics = drpmd.Dynamics

    def dynamics_factory(*args, **kwargs):
        dynamics = original_dynamics(*args, **kwargs)
        selected.append(dynamics.integrator_name)
        return dynamics

    with monkeypatch.context() as patch:
        patch.setattr(drpmd, "Dynamics", dynamics_factory)
        resumed = resume_run(directory, execution={"loop_backend": "python", "block_steps": 7})
    assert selected == [integrator, integrator]
    for expected, actual in zip(reference["records"], resumed["records"]):
        assert {k: v for k, v in actual.items() if k != "wall_seconds"} == {
            k: v for k, v in expected.items() if k != "wall_seconds"
        }
    for path in (tmp_path / "full" / "trajectories").glob("*.npz"):
        if path.name.endswith(".checkpoint.npz"):
            continue
        with load_checkpoint(path) as expected, load_checkpoint(directory / "trajectories" / path.name) as actual:
            for key in ("final_q", "final_p", "steps", "centroid", "energy_per_bead"):
                np.testing.assert_array_equal(actual[key], expected[key])
    with (load_checkpoint(tmp_path / "full/checkpoints/parent-000000.npz") as expected,
          load_checkpoint(directory / "checkpoints/parent-000000.npz") as actual):
        for key in ("q", "p"):
            np.testing.assert_array_equal(actual[key], expected[key])
        assert json.loads(str(actual["metadata"].item()))["rng"] == json.loads(str(expected["metadata"].item()))["rng"]


@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab", "velocity-verlet"])
def test_direct_drpmd_preserves_matching_bare_sampler_names(tmp_path, integrator):
    from pyrpmd.workflows.drpmd import run_drpmd

    job = collision_job()
    job["beads"] = 4
    job["dynamics"]["integrator"] = integrator
    job["collision"]["max_steps"] = 1
    config, system = JobSpec.from_mapping(job).resolve()
    config["sampling"]["integrator"] = integrator
    summary = run_drpmd(config, system, tmp_path / "bare")
    assert len(summary["records"]) == 2
    config["sampling"]["integrator"] = "normal-mode-bab" if integrator == "cayley-bcb" else "cayley-bcb"
    with pytest.raises(ValueError, match="same selected integrator"):
        run_drpmd(config, system, tmp_path / "mismatch")


@pytest.mark.parametrize("legacy", [False, True])
def test_drpmd_v1_payload_remains_readable_with_or_without_envelope(tmp_path, legacy):
    from pyrpmd.workflows.drpmd import _load_npz

    path = tmp_path / "parent.npz"
    kind = "PyRPMD.drpmd.parent-checkpoint.v1"
    arrays = {"q": np.zeros((4, 2, 3)), "p": np.ones((4, 2, 3)),
              "metadata": np.asarray(json.dumps({"format": kind}))}
    if legacy:
        np.savez_compressed(path, **arrays)
    else:
        atomic_savez(path, kind=kind, **arrays)
    metadata, payload = _load_npz(path, expected_kind=kind)
    assert metadata["format"] == kind
    np.testing.assert_array_equal(payload["q"], arrays["q"])
    np.testing.assert_array_equal(payload["p"], arrays["p"])


@pytest.mark.parametrize("legacy, envelope_kind, payload_kind", [
    (True, None, "PyRPMD.drpmd.parent-checkpoint.v999"),
    (True, None, "PyRPMD.drpmd.trajectory-checkpoint.v1"),
    (False, "PyRPMD.drpmd.parent-checkpoint.v999", "PyRPMD.drpmd.parent-checkpoint.v999"),
    (False, "PyRPMD.drpmd.parent-checkpoint.v1", "PyRPMD.drpmd.parent-checkpoint.v999"),
    (False, "PyRPMD.drpmd.parent-checkpoint.v1", "PyRPMD.drpmd.trajectory-checkpoint.v1"),
    (False, "PyRPMD.drpmd.parent-checkpoint.v999", "PyRPMD.drpmd.parent-checkpoint.v1"),
    (False, "PyRPMD.drpmd.parent-checkpoint.v1", None),
])
def test_drpmd_rejects_future_or_mismatched_payload_schemas(tmp_path, legacy, envelope_kind, payload_kind):
    from pyrpmd.workflows.drpmd import _load_npz

    path = tmp_path / "parent.npz"
    metadata = {} if payload_kind is None else {"format": payload_kind}
    arrays = {"q": np.zeros((4, 2, 3)), "metadata": np.asarray(json.dumps(metadata))}
    if legacy:
        np.savez_compressed(path, **arrays)
    else:
        atomic_savez(path, kind=envelope_kind, **arrays)
    with pytest.raises(CheckpointError, match="format|kind"):
        _load_npz(path, expected_kind="PyRPMD.drpmd.parent-checkpoint.v1")


def test_public_resume_rejects_future_parent_payload_with_valid_integrity(tmp_path):
    directory = tmp_path / "run"
    run(collision_job(), output=directory)
    checkpoint = next((directory / "checkpoints").glob("parent-*.npz"))
    with load_checkpoint(checkpoint) as saved:
        arrays = {name: saved[name] for name in saved.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata["format"] = "PyRPMD.drpmd.parent-checkpoint.v999"
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    atomic_savez(checkpoint, kind=metadata["format"], **arrays)
    with pytest.raises(CheckpointError, match="Expected checkpoint kind"):
        resume_run(directory)


def test_explicit_foreign_v1_manifest_is_not_migrated(tmp_path):
    config, system = JobSpec.from_mapping(collision_job()).resolve()
    _initialize_output(config, system, tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(schema="foreign.run", schema_version=1,
                    physics_fingerprint=fingerprint(config, system))
    path.write_text(json.dumps(manifest))
    original = path.read_bytes()
    with pytest.raises(InputError, match="Unknown run manifest schema"):
        _initialize_output(config, system, tmp_path, resume=True)
    assert path.read_bytes() == original


@pytest.mark.parametrize("legacy", [False, True])
def test_custom_payload_versions_keep_their_own_kind_contract(tmp_path, legacy):
    path = tmp_path / "custom.npz"
    kind = "my.state.v2"
    arrays = {"q": np.array([2.0]), "metadata": np.asarray(json.dumps({"format": kind}))}
    if legacy:
        np.savez_compressed(path, **arrays)
    else:
        atomic_savez(path, kind=kind, **arrays)
    with load_checkpoint(path, expected_kind=kind) as saved:
        np.testing.assert_array_equal(saved["q"], arrays["q"])
    with pytest.raises(CheckpointError, match="Expected checkpoint kind"):
        with load_checkpoint(path, expected_kind="my.state.v1"):
            pass
