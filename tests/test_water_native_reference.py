"""Independent native reference: map, physical checkpoint, and bounded runs."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.water.native_reference import (
    Config, LEGACY_CHECKPOINT_SOURCE_SHA256, create_engine, get_state, run,
    sample_momenta, set_state, validate_checkpoint_state,
)
from benchmarks.water.qtip4pf import initial_grid


def small_config(**kwargs):
    return Config(cells_per_axis=2, cutoff_angstrom=2.5, platform="Reference",
                  dt_fs=.1, sampling_refresh_fs=.4, equilibration_steps=8, equilibration_record_stride=2,
                  segment_steps=16, segments=2, record_stride=1, checkpoint_steps=7,
                  trajectory_precision="float64", **kwargs)


def checkpoint(directory, manifest):
    with np.load(directory / manifest["chunks"][-1]["file"], allow_pickle=False) as data:
        return {name: data[name].copy() for name in ("q", "p", "state_step", "rng_json")}


def test_native_velocity_verlet_matches_pyrpmd_normal_mode_bab_for_one_bead():
    mm = pytest.importorskip("openmm")
    from benchmarks.water.campaign import Config as PyConfig, create_dynamics
    from pyrpmd.dynamics import State
    config = small_config()
    q, box = initial_grid(cells_per_axis=2, density_g_cm3=config.density_g_cm3, seed=5)
    system, integrator, context, masses = create_engine(config, box)
    p = sample_momenta(np.random.default_rng(51), masses, config.temperature_K, remove_com=True)
    set_state(context, q[None], p, masses)
    pyconfig = PyConfig(cells_per_axis=2, beads=1, cutoff_angstrom=2.5, dt_fs=.1,
                        platform="Reference", threads=1, ewald_tolerance=config.ewald_tolerance)
    provider, dynamics = create_dynamics(pyconfig, box)
    try:
        state = State(q[None].copy(), p.copy())
        initial = get_state(context, masses, energy=True)
        np.testing.assert_allclose(initial[2], provider.evaluate(state.q).energy[0], atol=1e-14, rtol=0)
        for _ in range(20):
            integrator.step(1)
            dynamics.step(state)
        native_q, native_p, _ = get_state(context, masses)
        np.testing.assert_allclose(native_q, state.q, atol=2e-10, rtol=0)
        np.testing.assert_allclose(native_p, state.p, atol=2e-8, rtol=0)
    finally:
        provider.close()
        del context, integrator


def test_langevin_middle_zero_friction_and_verlet_have_same_synchronized_state():
    mm = pytest.importorskip("openmm")
    q, box = initial_grid(cells_per_axis=2, density_g_cm3=.997, seed=41)
    outcomes = []
    for integration in ("Verlet", "LangevinMiddle"):
        config = small_config()
        system, integrator, context, masses = create_engine(config, box)
        if integration == "LangevinMiddle":
            del context, integrator
            integrator = mm.LangevinMiddleIntegrator(298, 0, config.dt_fs / 1000)
            context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
        p = sample_momenta(np.random.default_rng(4), masses, 298, remove_com=True)
        set_state(context, q[None], p, masses)
        integrator.step(20)
        outcomes.append(get_state(context, masses))
        del context, integrator
    np.testing.assert_allclose(outcomes[0][0], outcomes[1][0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(outcomes[0][1], outcomes[1][1], atol=1e-9, rtol=0)


def test_native_checkpoint_resume_across_thermostat_and_segment_boundary(tmp_path):
    pytest.importorskip("openmm")
    config = small_config()
    full, split = tmp_path / "full", tmp_path / "split"
    complete = run(full, config)
    interrupted = run(split, config, max_steps=11)
    assert interrupted["committed_steps"] == 11 and not interrupted["complete"]
    resumed = run(split, config, resume=True)
    assert resumed["complete"] and resumed["external_reference"]
    assert not resumed["pyrpmd_production_trajectory"] and not resumed["literature_reproduced"]
    a, b = checkpoint(full, complete), checkpoint(split, resumed)
    for key in ("q", "p"):
        np.testing.assert_allclose(a[key], b[key], atol=1e-12, rtol=0)
    for key in ("state_step", "rng_json"):
        np.testing.assert_array_equal(a[key], b[key])
    segments = {}
    for entry in resumed["chunks"]:
        with np.load(split / entry["file"], allow_pickle=False) as data:
            if "segment" not in data:
                continue
            for segment in (0, 1):
                segments.setdefault(segment, []).extend(data["segment_step"][data["segment"] == segment].tolist())
    assert segments == {0: list(range(17)), 1: list(range(17))}
    assert resumed["environment"]["dependencies"]["openmm"]


def test_environment_config_and_committed_bytes_guard_resume(tmp_path):
    pytest.importorskip("openmm")
    config = small_config()
    manifest = run(tmp_path, config, max_steps=7)
    with pytest.raises(ValueError, match="source/configuration"):
        run(tmp_path, replace(config, dt_fs=.2), resume=True)
    original = (tmp_path / "manifest.json").read_text()
    changed = json.loads(original)
    changed["environment"]["dependencies"]["openmm"] = "different-runtime"
    (tmp_path / "manifest.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="environment"):
        run(tmp_path, config, resume=True)
    (tmp_path / "manifest.json").write_text(original)
    (tmp_path / manifest["chunks"][-1]["file"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        run(tmp_path, config, resume=True)


def test_cpu_liquid_checkpoint_raw_identity_and_exact_force_conversion():
    """Exercise a liquid neighbor list, where canonical p need not be bitwise."""
    pytest.importorskip("openmm")
    config = Config(platform="CPU", threads=1)
    snapshot = Path(__file__).resolve().parents[1] / "docs/validation/literature-audit-2026-11-01/water/classical-pilot/endpoint.npz"
    with np.load(snapshot, allow_pickle=False) as data:
        q, p, box = data["q"].copy(), data["p"].copy(), float(data["box_angstrom"])
    system, integrator, context, masses = create_engine(config, box)
    system2, integrator2, context2, masses2 = create_engine(config, box)
    try:
        set_state(context, q, p, masses)
        integrator.step(10)
        q, p, _, raw = get_state(context, masses, raw=True)
        saved = dict(raw, q=q, p=p)
        context2.loadCheckpoint(context.createCheckpoint())
        check = validate_checkpoint_state(context2, masses2, config, saved)
        assert check["raw_positions_and_half_step_velocities_exact"]
        assert not check["legacy_force_inferred"]
        assert check["canonical_p_conversion_residual_max_au"] < 2e-13
        assert check["force_RMS_relative_difference"] < 1e-6
        # Independently reject a wrong raw velocity, wrong canonical momentum,
        # and an altered force, rather than accepting algebra alone.
        bad = {name: value.copy() for name, value in saved.items()}
        bad["engine_half_step_velocities_nm_ps"][0, 0] += 1e-12
        with pytest.raises(ValueError, match="raw state mismatch"):
            validate_checkpoint_state(context2, masses2, config, bad)
        bad = {name: value.copy() for name, value in saved.items()}
        bad["p"][0, 0, 0] += 1e-5
        with pytest.raises(ValueError, match="canonical momentum"):
            validate_checkpoint_state(context2, masses2, config, bad)
        bad = {name: value.copy() for name, value in saved.items()}
        bad["engine_forces_kj_mol_nm"][0, 0] += 1
        with pytest.raises(ValueError, match="roundoff bound"):
            validate_checkpoint_state(context2, masses2, config, bad)
    finally:
        del context, integrator, context2, integrator2


def test_explicit_checkpoint_contract_migration_preserves_original_evidence(tmp_path):
    pytest.importorskip("openmm")
    from benchmarks.water.campaign import atomic_npz, sha256
    config = small_config()
    manifest = run(tmp_path, config, max_steps=7)
    # Synthetic legacy fixture tests the migration contract; the actual old
    # liquid checkpoints have a separate recorded read-only numerical audit.
    chunk = tmp_path / manifest["chunks"][-1]["file"]
    with np.load(chunk, allow_pickle=False) as data:
        old_arrays = {name: data[name].copy() for name in data.files if not name.startswith("engine_")}
    atomic_npz(chunk, **old_arrays)
    manifest["chunks"][-1]["sha256"] = sha256(chunk)
    manifest["source_sha256"]["benchmarks/water/native_reference.py"] = LEGACY_CHECKPOINT_SOURCE_SHA256
    del manifest["checkpoint_contract_version"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    original_bytes, original_chunk = manifest_path.read_bytes(), chunk.read_bytes()
    with pytest.raises(ValueError, match="source/configuration"):
        run(tmp_path, config, resume=True)
    resumed = run(tmp_path, config, resume=True, max_steps=2, migrate_checkpoint_contract=True)
    assert resumed["committed_steps"] == 9
    assert chunk.read_bytes() == original_chunk
    assert (tmp_path / "manifest.before-checkpoint-contract-v2.json").read_bytes() == original_bytes
    assert resumed["source_migrations"][0]["state_validation"]["legacy_force_inferred"]
    assert resumed["source_migrations"][0]["historical_checkpoint_sha256"] == sha256(chunk)
    finished = run(tmp_path, config, resume=True)
    assert finished["complete"]
    assert finished["restart_validations"][-1]["raw_positions_and_half_step_velocities_exact"]
    with pytest.raises(ValueError, match="specifically audited"):
        run(tmp_path, config, resume=True, migrate_checkpoint_contract=True)


def test_checkpoint_migration_never_accepts_an_unrecognized_source_change(tmp_path):
    pytest.importorskip("openmm")
    config = small_config()
    manifest = run(tmp_path, config, max_steps=7)
    manifest["source_sha256"]["benchmarks/water/native_reference.py"] = LEGACY_CHECKPOINT_SOURCE_SHA256
    manifest["source_sha256"]["benchmarks/water/qtip4pf.py"] = "different-force-field"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source/configuration"):
        run(tmp_path, config, resume=True, migrate_checkpoint_contract=True)


@pytest.mark.parametrize("change", [dict(beads=32), dict(beads=1.0), dict(dt_fs=.3),
                                  dict(seed=0), dict(trajectory_precision="half"),
                                  dict(integration="unknown")])
def test_invalid_native_reference_protocol_is_rejected(change):
    with pytest.raises(ValueError):
        replace(Config(), **change).validate()
