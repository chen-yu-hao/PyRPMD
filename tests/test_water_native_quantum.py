"""Full-PME RPMD: geometric clock, ring momenta, native map and physical restart."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.water.native_quantum import (
    Config, FullRPMDEngine, checkpoint_phase, run, sample_ring_momenta,
)
from pyrpmd.constants import KB_AU


def small_config(**changes):
    base = Config(cells_per_axis=2, beads=4, cutoff_angstrom=2.5, platform="Reference",
                  dt_fs=.1, andersen_mean_interval_fs=.5, equilibration_steps=24,
                  equilibration_record_stride=4, segment_steps=12, segments=2,
                  record_stride=3, checkpoint_steps=7, trajectory_precision="float64")
    return replace(base, **changes)


def test_ring_momentum_projection_removes_only_total_translation():
    masses = np.array([29000., 1800., 1800., 29000., 1800., 1800.])
    raw = sample_ring_momenta(np.random.default_rng(819), masses, 8, 298., remove_com=False)
    projected = sample_ring_momenta(np.random.default_rng(819), masses, 8, 298., remove_com=True)
    np.testing.assert_allclose(projected.sum(axis=(0, 1)), 0, atol=8e-14, rtol=0)
    np.testing.assert_allclose(projected - projected.mean(axis=0), raw - raw.mean(axis=0), atol=1e-14, rtol=0)
    assert np.linalg.norm(projected.sum(axis=1)) > 1  # Internal bead COM modes remain sampled.


def test_ring_momentum_centroid_variance_has_physical_temperature():
    masses, beads, temperature = np.array([1800., 29000.]), 8, 298.
    rng = np.random.default_rng(81)
    draws = np.stack([sample_ring_momenta(rng, masses, beads, temperature, remove_com=False) for _ in range(2000)])
    centroid = draws.mean(axis=1)
    expected = masses[:, None] * KB_AU * temperature
    np.testing.assert_allclose(np.mean(centroid**2, axis=0) / expected, 1, atol=.09, rtol=0)
    bead_expected = masses[None, :, None] * beads * KB_AU * temperature
    np.testing.assert_allclose(np.mean(draws**2, axis=0) / bead_expected, 1, atol=.12, rtol=0)


@pytest.mark.parametrize("change", [
    {"andersen_schedule": "fixed"}, {"andersen_mean_interval_fs": .15},
    {"platform": "CUDA"}, {"platform": "CPU", "platform_properties": {"Threads": "8"}},
])
def test_quantum_configuration_rejects_undeclared_sampling_and_platform_settings(change):
    with pytest.raises(ValueError):
        small_config(**change).validate()


def test_quantum_phase_distinguishes_pending_segment_momenta():
    config = small_config()
    assert checkpoint_phase(config, 0, -1) == "preparation"
    assert checkpoint_phase(config, 24, -1) == "production-awaiting-momentum-refresh"
    assert checkpoint_phase(config, 25, 0) == "production"
    assert checkpoint_phase(config, 36, 0) == "production-awaiting-momentum-refresh"
    assert checkpoint_phase(config, 48, 1) == "complete"
    with pytest.raises(ValueError, match="active NVE"):
        checkpoint_phase(config, 36, 1)


def checkpoint(directory, manifest):
    with np.load(directory / manifest["chunks"][-1]["file"], allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def collect(directory, manifest, key):
    values = []
    for record in manifest["chunks"]:
        with np.load(directory / record["file"], allow_pickle=False) as data:
            if key in data:
                values.append(data[key])
    return np.concatenate(values)


def test_full_rpmd_native_map_and_energy_match_independent_pyrpmd_bab():
    pytest.importorskip("openmm")
    from benchmarks.water.native_optimizations import openmm_frequency_beta
    from benchmarks.water.native_quantum import HARTREE_KJ_MOL, VELOCITY_NM_PS
    from benchmarks.water.qtip4pf import initial_grid
    from benchmarks.water.split_reference import GroupEvaluator
    from pyrpmd.constants import AMU_TO_AU, FS_TO_AU
    from pyrpmd.dynamics import Dynamics, State
    from pyrpmd.potentials.base import PESResult
    config = small_config(beads=8, cutoff_angstrom=3.)
    q, box = initial_grid(cells_per_axis=2, density_g_cm3=.997, seed=14)
    q[3:6, 2] -= .6
    engine = FullRPMDEngine(config, box)
    evaluator = GroupEvaluator(engine.system)
    class Potential:
        def evaluate(self, coordinates):
            energy, force = evaluator.evaluate(coordinates)
            return PESResult(energy, force, np.zeros(len(coordinates), dtype=int))
    independent = Dynamics(Potential(), engine.masses, openmm_frequency_beta(298), 8,
                           config.dt_fs * FS_TO_AU, native=True, threads=1, integrator="normal-mode-bab")
    rng = np.random.default_rng(49)
    q = q[None] + rng.normal(0, .02, (8, 24, 3))
    p = sample_ring_momenta(rng, engine.masses, 8, 298, remove_com=True)
    state = State(q.copy(), p.copy())
    try:
        engine.set_state(q, p)
        for _ in range(3):
            engine.integrator.step(1)
            independent.step(state)
        nq, np_ = engine.physical_state(engine.raw_state())
        np.testing.assert_allclose(nq, state.q, atol=2e-9, rtol=0)
        np.testing.assert_allclose(np_, state.p, atol=5e-8, rtol=0)
        energy = float(engine.integrator.getTotalEnergy() / engine.unit.kilojoule_per_mole) / HARTREE_KJ_MOL
        atomic_energy = independent.energy(State(nq.copy(), np_.copy()))
        potential = float(evaluator.evaluate(nq)[0].sum())
        # A separately computed unit-constant factor applies to kinetic and
        # spring terms (not to potential). It is ~0.9999999989488311 with this
        # runtime's constants, and is not a widened energy-equality tolerance.
        native_to_atomic_factor = VELOCITY_NM_PS**2 / (AMU_TO_AU * HARTREE_KJ_MOL)
        assert energy == pytest.approx(potential + native_to_atomic_factor * (atomic_energy - potential), abs=2e-11)
        assert not engine.integrator.getContractions() and not engine.integrator.getApplyThermostat()
    finally:
        engine.close()
        evaluator.close()


def test_full_rpmd_geometric_clock_and_rng_resume_across_preparation_and_nve_boundaries(tmp_path):
    pytest.importorskip("openmm")
    config = small_config()
    full, split = tmp_path / "full", tmp_path / "split"
    complete = run(full, config)
    partial = run(split, config, max_steps=13)
    assert partial["committed_steps"] == 13
    at_boundary = run(split, config, resume=True, max_steps=11)
    assert str(checkpoint(split, at_boundary)["phase"]) == "production-awaiting-momentum-refresh"
    end_first = run(split, config, resume=True, max_steps=12)
    assert str(checkpoint(split, end_first)["phase"]) == "production-awaiting-momentum-refresh"
    resumed = run(split, config, resume=True)
    first, second = checkpoint(full, complete), checkpoint(split, resumed)
    for key in ("q", "p", "engine_positions_nm", "engine_velocities_nm_ps"):
        np.testing.assert_allclose(second[key], first[key], atol=2e-12, rtol=0)
    for key in ("state_step", "active_segment", "rng_json", "andersen_next_collision_step", "phase"):
        np.testing.assert_array_equal(second[key], first[key])
    np.testing.assert_array_equal(collect(full, complete, "collision_steps"), collect(split, resumed, "collision_steps"))
    np.testing.assert_array_equal(collect(full, complete, "refreshed_NVE_segments"), [0, 1])
    np.testing.assert_array_equal(collect(split, resumed, "refreshed_NVE_segments"), [0, 1])
    events = collect(full, complete, "collision_steps")
    assert events[0] == 0 and np.all(np.diff(events) > 0) and events[-1] < config.equilibration_steps
    # Independent event recurrence, including the number/order of Gaussian draws.
    rng, expected, step = np.random.default_rng(config.seed), [], 0
    while step < config.equilibration_steps:
        expected.append(step)
        rng.normal(size=(config.beads, 3 * config.cells_per_axis**3, 3))
        step += int(rng.geometric(1 / config.mean_collision_steps))
    np.testing.assert_array_equal(events, expected)
    assert resumed["complete"] and not resumed["literature_reproduced"]
    assert all(row["all_copy_positions_and_velocities_exact"] for row in resumed["restart_validations"])


@pytest.mark.parametrize("platform", ["Reference", "CPU"])
def test_full_rpmd_checkpoint_restores_every_copy_and_rejects_changed_force(platform):
    pytest.importorskip("openmm")
    from benchmarks.water.qtip4pf import initial_grid
    config = small_config(platform=platform)
    q, box = initial_grid(cells_per_axis=2, density_g_cm3=.997, seed=72)
    original = FullRPMDEngine(config, box)
    restored = FullRPMDEngine(config, box)
    try:
        rng = np.random.default_rng(85)
        q = q[None] + rng.normal(0, .02, (config.beads, 24, 3))
        original.set_state(q, sample_ring_momenta(rng, original.masses, config.beads, 298, remove_com=False))
        original.integrator.step(3)
        raw = original.raw_state(forces=True)
        q, p = original.physical_state(raw)
        saved = dict(raw, q=q, p=p, openmm_checkpoint=np.frombuffer(original.context.createCheckpoint(), dtype=np.uint8))
        checked = restored.restore(saved)
        assert checked["all_copy_positions_and_velocities_exact"]
        assert checked["physical_q_and_p_exact"]
        changed = {key: value.copy() for key, value in saved.items()}
        changed["engine_forces_kj_mol_nm"][0, 0, 0] += 1
        with pytest.raises(ValueError, match="forces exceed"):
            restored.restore(changed)
        changed = dict(saved)
        changed["engine_positions_nm"] = changed["engine_positions_nm"][:-1]
        with pytest.raises(ValueError, match="all-copy"):
            restored.restore(changed)
    finally:
        original.close()
        restored.close()


def test_full_rpmd_corrupt_completion_or_model_identity_rejects_before_context(tmp_path, monkeypatch):
    pytest.importorskip("openmm")
    from benchmarks.water import native_quantum
    config = small_config()
    manifest = run(tmp_path, config, max_steps=3)
    original = (tmp_path / "manifest.json").read_text()
    monkeypatch.setattr(native_quantum, "FullRPMDEngine", lambda *_: pytest.fail("must reject before constructing an engine"))
    manifest["complete"] = True
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="completion flag"):
        run(tmp_path, config, resume=True)
    manifest = json.loads(original)
    del manifest["artifacts"]["model.xml"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="serialized model"):
        run(tmp_path, config, resume=True)
