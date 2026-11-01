"""Small real-engine phase, provenance and restart gates; no long extension."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("openmm")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.water import native_preparation_extension as extension
from benchmarks.water import native_reference as native
from benchmarks.water.campaign import atomic_npz, sha256


def make_original(directory, platform="Reference"):
    config = native.Config(cells_per_axis=2, cutoff_angstrom=2.5, dt_fs=.1,
                           equilibration_steps=8, equilibration_record_stride=2,
                           segment_steps=16, segments=1, record_stride=2,
                           checkpoint_steps=4, sampling_refresh_fs=.4,
                           seed=3101, platform=platform, trajectory_precision="float64")
    native.run(directory, config)
    return config


@pytest.fixture
def original(tmp_path):
    source = tmp_path / "original"
    base = make_original(source)
    config = extension.prepare_config(str(source), additional_preparation_ps=.0012)
    return source, base, config


def checkpoint(directory, index=-1):
    manifest = json.loads((directory / "manifest.json").read_text())
    with np.load(directory / manifest["chunks"][index]["file"], allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    return manifest, arrays


def refresh_events(directory, manifest):
    events = []
    for record in manifest["chunks"]:
        with np.load(directory / record["file"], allow_pickle=False) as data:
            events.extend(data["refresh_steps"].tolist())
            if "segment" in data:
                assert np.all(data["segment"] == -1)
    return events


def test_branch_matches_continuous_preparation_at_exact_source_refresh_boundary(original, tmp_path):
    source, base, config = original
    before = sha256(source / "manifest.json")
    out = tmp_path / "extension"
    result = extension.run(out, config)
    continuous = tmp_path / "continuous"
    native.run(continuous, replace(base, equilibration_steps=20), max_steps=20)
    _, expected = checkpoint(continuous)
    _, actual = checkpoint(out)
    np.testing.assert_allclose(actual["q"], expected["q"], rtol=0, atol=2e-12)
    np.testing.assert_allclose(actual["p"], expected["p"], rtol=0, atol=2e-11)
    assert str(actual["rng_json"]) == str(expected["rng_json"])
    assert int(actual["active_segment"]) == -1
    assert result["complete"] and result["production_started"] is False
    assert result["refresh_count"] == 3 and result["next_refresh_step"] == 20
    assert refresh_events(out, result) == [8, 12, 16]
    assert sha256(source / "manifest.json") == before
    native.verify_manifest(source, json.loads((source / "manifest.json").read_text()))


@pytest.mark.parametrize("first_steps", [4, 6])
def test_split_restart_preserves_pending_refresh_and_rng(original, tmp_path, first_steps):
    _, _, config = original
    full, split = tmp_path / "full", tmp_path / "split"
    extension.run(full, config)
    partial = extension.run(split, config, max_steps=first_steps)
    assert partial["committed_steps"] == 8 + first_steps
    assert partial["next_refresh_step"] == (12 if first_steps == 4 else 16)
    resumed = extension.run(split, config, resume=True)
    _, expected = checkpoint(full)
    _, actual = checkpoint(split)
    np.testing.assert_allclose(actual["q"], expected["q"], rtol=0, atol=2e-12)
    np.testing.assert_allclose(actual["p"], expected["p"], rtol=0, atol=2e-11)
    assert str(actual["rng_json"]) == str(expected["rng_json"])
    assert refresh_events(split, resumed) == [8, 12, 16]


def test_no_com_projection_or_terminal_nve_draw_and_completed_resume_creates_no_context(original, tmp_path, monkeypatch):
    _, _, config = original
    calls = []
    sample = native.sample_momenta
    def record(*args, **kwargs):
        calls.append(kwargs["remove_com"])
        return sample(*args, **kwargs)
    monkeypatch.setattr(native, "sample_momenta", record)
    out = tmp_path / "extension"
    extension.run(out, config)
    assert calls == [False, False, False]
    def forbidden(*args, **kwargs):
        pytest.fail("a completed preparation branch must not create an engine or draw NVE momenta")
    monkeypatch.setattr(native, "create_engine", forbidden)
    result = extension.run(out, config, resume=True)
    assert result["committed_steps"] == 20 and calls == [False, False, False]


def test_config_preparation_is_read_only_and_source_hash_change_fails_before_context(original, tmp_path, monkeypatch):
    source, _, config = original
    def forbidden(*args, **kwargs):
        pytest.fail("read-only configuration/source guards must run before engine creation")
    monkeypatch.setattr(native, "create_engine", forbidden)
    assert extension.prepare_config(str(source), .0012) == config
    with pytest.raises(ValueError, match="selected original manifest/checkpoint changed"):
        extension.run(tmp_path / "bad", replace(config, source_manifest_sha256="0"*64))
    assert not (tmp_path / "bad").exists()


def test_self_consistent_altered_serialized_model_is_rejected(original, tmp_path):
    import openmm as mm
    from openmm import unit
    source, _, _ = original
    model = mm.XmlSerializer.deserialize((source / "model.xml").read_text())
    force = next(force for force in model.getForces() if isinstance(force, mm.NonbondedForce))
    charge, sigma, epsilon = force.getParticleParameters(0)
    force.setParticleParameters(0, charge + .1*unit.elementary_charge, sigma, epsilon)
    (source / "model.xml").write_text(mm.XmlSerializer.serialize(model))
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"]["model.xml"]["sha256"] = sha256(source / "model.xml")
    path.write_text(json.dumps(manifest, sort_keys=True))
    config = extension.prepare_config(str(source), .0012)
    with pytest.raises(ValueError, match="fresh frozen factory model/integrator"):
        extension.run(tmp_path / "bad", config)
    assert not (tmp_path / "bad").exists()


def test_source_checkpoint_after_nve_draw_is_rejected(original, tmp_path):
    source, base, _ = original
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    selected = next(row for row in manifest["chunks"] if row["step"] == base.equilibration_steps)
    with np.load(source / selected["file"], allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays["active_segment"] = np.asarray(0)
    atomic_npz(source / selected["file"], **arrays)
    selected["sha256"] = sha256(source / selected["file"])
    path.write_text(json.dumps(manifest, sort_keys=True))
    config = extension.prepare_config(str(source), .0012)
    with pytest.raises(ValueError, match="precede every NVE momentum draw"):
        extension.run(tmp_path / "bad", config)


def test_cpu_raw_checkpoint_and_independent_force_cache_bound(tmp_path):
    source = tmp_path / "source-cpu"
    make_original(source, "CPU")
    config = extension.prepare_config(str(source), .0012)
    output = tmp_path / "extension-cpu"
    first = extension.run(output, config, max_steps=4)
    assert first["restart_validations"][-1]["raw_positions_and_half_step_velocities_exact"] is True
    final = extension.run(output, config, resume=True)
    assert all(row["force_RMS_relative_difference"] <= 1e-6 for row in final["restart_validations"])
    assert final["complete"] and final["refresh_count"] == 3

    # Preserve raw q/v but make the stored force and canonical p agree with a
    # deliberately altered force. Algebra alone must not admit this state.
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    selected = next(row for row in manifest["chunks"] if row["step"] == 8)
    with np.load(source / selected["file"], allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays["engine_forces_kj_mol_nm"] += 100.0
    coefficient = .5 * .1 / 1000 * native.AMU_TO_AU / native.VELOCITY_NM_PS
    arrays["p"] += coefficient * 100.0
    atomic_npz(source / selected["file"], **arrays)
    selected["sha256"] = sha256(source / selected["file"])
    path.write_text(json.dumps(manifest, sort_keys=True))
    altered = extension.prepare_config(str(source), .0012)
    with pytest.raises(ValueError, match="force exceeds declared roundoff bound"):
        extension.run(tmp_path / "bad-force", altered)
    assert not (tmp_path / "bad-force").exists()


def test_corrupted_saved_phase_cannot_resume(original, tmp_path):
    _, _, config = original
    out = tmp_path / "extension"
    extension.run(out, config, max_steps=4)
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["next_refresh_step"] = 16  # At step12 the draw at12 is still pending.
    path.write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(ValueError, match="periodic-refresh phase"):
        extension.run(out, config, resume=True)
