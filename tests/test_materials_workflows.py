"""Physical estimator, phase-clock and exact recovery checks for materials."""

import copy
import json

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials.builtin import HarmonicPotential
from pyrpmd.system import System
from pyrpmd.workflows.materials import (
    CHECKPOINT_KIND, OBSERVABLE_NAMES, _sample_replica, _read_stream,
    quantum_observables, resolve_materials, run_materials, reanalyze_diffusion,
)


def system():
    return System(("H", "Ni"), np.array([2.0, 20.0]), np.array([[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]]))


def config(task="diffusion", thermostat="pile-l"):
    return {
        "task": task, "beta": 2.0, "beads": 3, "dt_au": 0.03,
        "seed": 412, "native": False,
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 0.4}},
        "sampling": {"equilibration_steps": 5, "production_steps": 30,
                     "sample_interval": 2, "replicas": 2, "gamma_centroid": 0.6,
                     "thermostat": thermostat, "andersen_interval": 3},
        "dynamics": {"loop_backend": "python", "block_steps": 7},
        "diffusion": {"atom_indices": [0], "axes": [0, 1], "bootstrap": 0},
        "output": {"checkpoint_steps": 4},
    }


def task(cfg, output, replica=0):
    return {"config": cfg, "system": system(), "output": str(output), "replica": replica}


def assert_same_stream(left, right):
    for key in ("positions", "velocities", "observables", "radii_gyration_squared", "sample_steps", "times"):
        np.testing.assert_array_equal(left[key], right[key])
    for key in ("q", "p"):
        np.testing.assert_array_equal(getattr(left["state"], key), getattr(right["state"], key))
    assert left["state"].step == right["state"].step
    assert left["state"].time == right["state"].time
    assert left["rng"] == right["rng"]
    assert left["propagation"] == right["propagation"]


@pytest.mark.parametrize("steps,expected_frames", [
    (40, [1, 11, 21, 31, 41]), (43, [1, 11, 21, 31, 41, 44]),
])
def test_checkpoint_writes_only_new_prefixes_and_completed_resume_is_read_only(
        tmp_path, monkeypatch, steps, expected_frames):
    import pyrpmd.workflows.materials as module

    cfg = config()
    cfg["sampling"].update(equilibration_steps=0, production_steps=steps, sample_interval=1)
    cfg["output"]["checkpoint_steps"] = 10
    original = module.atomic_savez
    counts = []

    def counted(path, **arrays):
        counts.append(len(arrays["sample_steps"]))
        return original(path, **arrays)

    monkeypatch.setattr(module, "atomic_savez", counted)
    first = _sample_replica(task(cfg, tmp_path))
    assert counts == expected_frames
    checkpoint = tmp_path / "checkpoints/materials_0000.npz"
    committed = checkpoint.read_bytes()
    counts.clear()
    resumed = _sample_replica(task(cfg, tmp_path))
    assert counts == []
    assert checkpoint.read_bytes() == committed
    assert_same_stream(first, resumed)


def test_completed_resume_regenerates_exports_without_rewriting_checkpoint(tmp_path, monkeypatch):
    import pyrpmd.workflows.materials as module

    cfg = config()
    cfg["sampling"].update(equilibration_steps=0, production_steps=40, sample_interval=1, replicas=1)
    cfg["output"]["checkpoint_steps"] = 10
    first = run_materials(cfg, system(), tmp_path, return_data=True)
    checkpoint = tmp_path / "checkpoints/materials_0000.npz"
    committed = checkpoint.read_bytes()
    exports = [tmp_path / "samples/materials_0000.npz", tmp_path / "summary.json",
               tmp_path / "results/diffusion.json"]
    for path in exports:
        path.unlink()
    original = module.atomic_savez
    written = []

    def counted(path, **arrays):
        written.append(path)
        return original(path, **arrays)

    monkeypatch.setattr(module, "atomic_savez", counted)
    resumed = run_materials(cfg, system(), tmp_path, return_data=True)
    assert written == [exports[0]]
    assert all(path.exists() for path in exports)
    assert checkpoint.read_bytes() == committed
    assert first["summary"] == resumed["summary"]
    assert_same_stream(first["streams"][0], resumed["streams"][0])


def test_quantum_estimators_match_action_derivative_and_force_virial():
    masses, beta = np.array([2.0, 5.0]), 3.1
    rng = np.random.default_rng(89)
    q = rng.normal(scale=0.3, size=(4, 2, 3))
    p = rng.normal(size=q.shape)
    potential = HarmonicPotential(k=0.7)
    state = State(q, p)
    values, radii = quantum_observables(state, potential, masses, beta)
    beads, atoms, _ = q.shape
    # Independent finite difference of the beta derivative of the primitive
    # path action, including the kinetic normalization term.
    def negative_log_weight(b):
        springs = beads * np.sum(masses[None, :, None] * (q - np.roll(q, 1, axis=0)) ** 2) / (2 * b)
        physical = b * 0.7 * np.sum(q * q) / (2 * beads)
        return 1.5 * beads * atoms * np.log(b) + springs + physical
    energy_derivative = (negative_log_weight(beta + 1e-5) - negative_log_weight(beta - 1e-5)) / 2e-5
    assert values[0] + values[1] == pytest.approx(energy_derivative, abs=2e-10)
    centered = q - q.mean(axis=0)
    expected_virial = 1.5 * atoms / beta + 0.7 * np.sum(centered * centered) / (2 * beads)
    assert values[2] == pytest.approx(expected_virial)
    assert values[3] == pytest.approx(values[0] + values[2])
    np.testing.assert_allclose(radii, np.mean(np.sum(centered ** 2, axis=-1), axis=0))
    # Estimators depend on configurations, not the fictitious momenta.
    changed, _ = quantum_observables(State(q, 7 * p), potential, masses, beta)
    np.testing.assert_array_equal(values[:4], changed[:4])
    assert values[4] != changed[4]
    classical, _ = quantum_observables(State(q[:1], p[:1]), potential, masses, beta)
    assert classical[1] == classical[2] == pytest.approx(1.5 * atoms / beta)


@pytest.mark.parametrize("task_name,thermostat,failure_step", [
    ("pimd", "pile-l", 7), ("diffusion", "pile-l", 4),
    ("diffusion", "andersen", 4), ("diffusion", "pile-l", 13),
])
def test_interrupted_phase_resumes_exactly_with_new_execution_block(tmp_path, monkeypatch, task_name, thermostat, failure_step):
    cfg = config(task_name, thermostat)
    reference = _sample_replica(task(cfg, tmp_path / "reference"))
    if thermostat == "pile-l" and (task_name == "pimd" or failure_step <= 5):
        name = "sample_step"
    else:
        name = "step"
    original = getattr(Dynamics, name)
    def interrupted(self, state, *args, **kwargs):
        result = original(self, state, *args, **kwargs)
        if state.step == failure_step:
            raise RuntimeError("injected interruption after completed step")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, name, interrupted)
        with pytest.raises(RuntimeError, match="injected interruption"):
            _sample_replica(task(cfg, tmp_path / "resumed"))
    saved = _read_stream(cfg, system(), tmp_path / "resumed", 0)
    assert saved["elapsed"] == failure_step - 1
    changed = copy.deepcopy(cfg)
    changed["dynamics"]["block_steps"] = 17
    resumed = _sample_replica(task(changed, tmp_path / "resumed"))
    assert_same_stream(reference, resumed)
    assert resumed["sample_steps"].tolist() == list(range(5, 36, 2))
    assert resumed["times"][0] == 0.0
    assert resumed["state"].step == 35
    # Running a completed stream does not re-sample or consume RNG state.
    assert_same_stream(resumed, _sample_replica(task(changed, tmp_path / "resumed")))


def test_sampling_ends_at_phase_boundary_and_production_matches_nve(tmp_path):
    cfg = config()
    saved = _sample_replica(task(cfg, tmp_path))
    # Build the same initialization/thermalization from its documented stream.
    potential = HarmonicPotential(k=0.4)
    rng = np.random.default_rng(np.random.SeedSequence([412, 1301, 0]))
    dynamics = Dynamics(potential, system().masses, 2.0, 3, 0.03, seed=rng, native=False, loop_backend="python")
    state = State(np.repeat(system().geometry[None], 3, axis=0), dynamics.sample_momenta())
    for _ in range(5):
        dynamics.sample_step(state, gamma_centroid=0.6)
    frozen_rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    for i in range(30):
        dynamics.step(state)
        if (i + 1) % 2 == 0:
            np.testing.assert_array_equal(saved["positions"][(i + 1) // 2], state.q.mean(axis=0))
    np.testing.assert_array_equal(saved["state"].q, state.q)
    assert saved["rng"] == frozen_rng
    assert saved["propagation"]["sampling"]["python_steps"] == 5
    assert saved["propagation"]["production"]["python_steps"] == 30


def test_observation_failure_rolls_back_state_rng_and_sample_together(tmp_path, monkeypatch):
    import pyrpmd.workflows.materials as module
    cfg = config("pimd")
    reference = _sample_replica(task(cfg, tmp_path / "reference"))
    original = module.quantum_observables
    def interrupted(state, *args):
        if state.step == 9:
            raise RuntimeError("injected observable failure")
        return original(state, *args)
    with monkeypatch.context() as patch:
        patch.setattr(module, "quantum_observables", interrupted)
        with pytest.raises(RuntimeError, match="observable failure"):
            _sample_replica(task(cfg, tmp_path / "resumed"))
    saved = _read_stream(cfg, system(), tmp_path / "resumed", 0)
    assert saved["elapsed"] == 8
    assert saved["sample_steps"].tolist() == [5, 7]
    assert_same_stream(reference, _sample_replica(task(cfg, tmp_path / "resumed")))


def test_initial_observation_failure_has_recoverable_state(tmp_path, monkeypatch):
    import pyrpmd.workflows.materials as module
    cfg = config()
    cfg["sampling"]["equilibration_steps"] = 0
    original = module.quantum_observables
    with monkeypatch.context() as patch:
        patch.setattr(module, "quantum_observables", lambda *args: (_ for _ in ()).throw(RuntimeError("initial failure")))
        with pytest.raises(RuntimeError, match="initial failure"):
            _sample_replica(task(cfg, tmp_path / "resumed"))
    saved = _read_stream(cfg, system(), tmp_path / "resumed", 0)
    assert saved["elapsed"] == 0 and saved["positions"].shape == (0, 2, 3)
    reference = _sample_replica(task(cfg, tmp_path / "reference"))
    resumed = _sample_replica(task(cfg, tmp_path / "resumed"))
    assert_same_stream(reference, resumed)


def test_pimd_summary_contains_physical_estimators_separate_from_hamiltonian(tmp_path):
    result = run_materials(config("pimd"), system(), tmp_path)
    assert result["task"] == "pimd"
    assert not result["physical_converged"]
    assert result["phase"] == "equilibrium_sampling"
    assert set(result["estimators"]) == set(OBSERVABLE_NAMES)
    assert result["estimators"]["internal_energy_centroid_virial"]["mean"] != result["estimators"]["ring_polymer_hamiltonian"]["mean"]
    assert result["estimators"]["potential_energy"]["standard_error_between_replicas"] is not None
    json.dumps(result, allow_nan=False)
    with load_checkpoint(result["files"]["trajectories"][0]) as trajectory:
        assert trajectory["positions"].shape == (16, 2, 3)
        assert json.loads(str(trajectory["metadata"]))["positions"] == "unwrapped_ring_polymer_centroids"


def test_diffusion_reanalysis_needs_no_potential_and_can_change_selections(tmp_path, monkeypatch):
    import pyrpmd.workflows.materials as module
    cfg = config()
    result = run_materials(cfg, system(), tmp_path)
    assert result["task"] == "diffusion"
    assert result["phase"] == "unthermostatted_rpmd_production"
    assert not result["physical_converged"]
    assert result["diffusion"]["input"]["atom_indices"] == [0]
    changed = copy.deepcopy(cfg)
    changed["diffusion"].update(axes=[0], reference_atoms=[1], bootstrap=3)
    monkeypatch.setattr(module, "make_task_potential", lambda *args: pytest.fail("offline reanalysis evaluated a potential"))
    replay = reanalyze_diffusion(changed, system(), tmp_path)
    assert replay["diffusion"]["input"]["axes"] == [0]
    json.dumps(replay, allow_nan=False)


def test_checkpoint_rejects_wrong_physics_and_consistent_but_invalid_sample_clock(tmp_path):
    cfg = config()
    _sample_replica(task(cfg, tmp_path))
    changed = copy.deepcopy(cfg)
    changed["dt_au"] *= 0.5
    with pytest.raises(ValueError, match="physics differs"):
        _read_stream(changed, system(), tmp_path, 0)
    path = tmp_path / "checkpoints/materials_0000.npz"
    with load_checkpoint(path) as archive:
        payload = {k: archive[k].copy() for k in archive.files}
    payload["sample_steps"][1] += 1
    atomic_savez(path, kind=CHECKPOINT_KIND, **payload)
    with pytest.raises(ValueError, match="phase clock"):
        _read_stream(cfg, system(), tmp_path, 0)


@pytest.mark.parametrize("field,value,message", [
    ("atom_indices", [], "atom_indices"), ("atom_indices", [0, 0], "atom_indices"),
    ("axes", [True], "axes"), ("reference_atoms", [0], "reference_atoms"),
    ("fit_range", [2.0, 1.0], "fit_range"), ("bootstrap", -1, "bootstrap"),
    ("bootstrap", 1, "bootstrap"), ("bootstrap", None, "bootstrap"),
    ("seed", None, "seed"), ("max_lag", 16, "max_lag"),
    ("block_length", 11, "block_length"), ("fit_range", [0, 0.2], "available lag"),
    ("fit_range", [None, 0.01], "increasing"),
])
def test_analysis_choices_validate_before_sampling(field, value, message):
    cfg = config()
    cfg["diffusion"][field] = value
    with pytest.raises(ValueError, match=message):
        resolve_materials(cfg, system())


@pytest.mark.parametrize("options,message", [
    ({"bootstrap": 1}, "bootstrap"),
    ({"max_lag": 16}, "max_lag"),
    ({"block_length": 11}, "block_length"),
    ({"fit_range": ["0 fs", "1 fs"]}, "available lag"),
])
def test_public_invalid_analysis_fails_before_output_or_provider(tmp_path, monkeypatch, options, message):
    from pyrpmd import run
    import pyrpmd.workflows.materials as module

    raw = {
        "task": "diffusion", "temperature": "300 K", "beads": 1,
        "system": {"symbols": ["H"], "geometry": [[0, 0, 0]], "geometry_unit": "bohr"},
        "potential": {"provider": "builtin", "name": "free"},
        "dynamics": {"timestep": "0.1 fs", "loop_backend": "python"},
        "sampling": {"equilibration_steps": 0, "production_steps": 30,
                     "sample_interval": 2, "replicas": 1},
        "diffusion": options,
    }
    monkeypatch.setattr(module, "make_task_potential", lambda *args: pytest.fail("invalid input constructed a provider"))
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match=message):
        run(raw, output=output)
    assert not output.exists()


def test_thermostat_capability_mismatch_rejected():
    cfg = config()
    cfg["dynamics"]["integrator"] = "velocity-verlet"
    with pytest.raises(ValueError, match="PILE-L"):
        resolve_materials(cfg, system())
    cfg["dynamics"]["integrator"] = "cayley-bcb"
    cfg["dynamics"]["loop_backend"] = "native"
    with pytest.raises(ValueError, match="PILE-L"):
        resolve_materials(cfg, system())


def test_process_replicas_preserve_random_streams(tmp_path):
    cfg = config("pimd")
    reference = run_materials(cfg, system(), tmp_path / "serial", return_data=True)
    cfg["parallel"] = {"backend": "process", "trajectory_workers": 2}
    parallel = run_materials(cfg, system(), tmp_path / "process", return_data=True)
    for left, right in zip(reference["streams"], parallel["streams"]):
        assert_same_stream(left, right)


def test_public_periodic_workflow_keeps_crossings_unwrapped_and_resumes(tmp_path):
    from pyrpmd import run, resume_run, analyze_run, prepare
    raw = {
        "task": "diffusion", "temperature": "300 K", "beads": 1, "seed": 57, "native": False,
        "system": {"symbols": ["H"], "geometry": [[0.1, 0.0, 0.0]], "geometry_unit": "bohr",
                   "cell": [[0.4, 0, 0], [0, 0.4, 0], [0, 0, 0.4]], "cell_unit": "bohr", "pbc": [True, True, True]},
        "potential": {"provider": "builtin", "name": "free"},
        "dynamics": {"timestep": "2 fs", "loop_backend": "python"},
        "sampling": {"equilibration_steps": 4, "production_steps": 30, "sample_interval": 2,
                     "replicas": 2, "thermostat": "andersen", "andersen_interval": 2},
        "diffusion": {"axes": [0, 1, 2], "bootstrap": 0},
    }
    assert prepare(raw, numerical=True)["ready"]
    result = run(raw, output=tmp_path / "run")
    with load_checkpoint(result["files"]["trajectories"][0]) as saved:
        q = saved["positions"]
        velocity = saved["velocities"]
        times = saved["times"]
        assert np.max(np.abs(q - q[0])) > 0.4
        np.testing.assert_allclose(q - q[0], times[:, None, None] * velocity[0], atol=2e-13)
    assert result["diffusion_m2_s"] is None  # Free ballistic motion is not diffusion.
    repeated = resume_run(tmp_path / "run", execution={"block_steps": 19})
    assert repeated["diffusion"] == result["diffusion"]
    reanalyzed = analyze_run(tmp_path / "run", diffusion={"axes": [0]})
    assert reanalyzed["diffusion"]["input"]["axes"] == [0]
