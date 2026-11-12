"""Tensor workflow recovery, phase semantics and independent physical checks."""
import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyrpmd.checkpoint import CheckpointError, atomic_savez, load_checkpoint
from pyrpmd.dynamics import State
from pyrpmd.potentials.builtin import HarmonicPotential
from pyrpmd.system import System
from pyrpmd.torch_backend import TorchDynamics, TorchState
from pyrpmd.workflows import torch_materials as tm
from pyrpmd.workflows.materials import quantum_observables, run_materials, reanalyze_diffusion


def system():
    return System(("H", "He"), np.array([2.0, 5.0]),
                  np.array([[0.1, 0.2, -0.1], [0.3, -0.1, 0.2]]))


def config(task="diffusion", *, neq=5, layout="rfft"):
    return {"task": task, "beta": 2.0, "beads": 4, "dt_au": 0.04, "seed": 41,
            "potential": {"provider": "harmonic", "params": {"k": 0.7}},
            "sampling": {"engine": "torch", "device": "cpu", "normal_mode_layout": layout,
                         "integrator": "bcocb", "thermostat": "pile-l", "gamma_centroid": 0.7,
                         "equilibration_steps": neq, "production_steps": 24,
                         "sample_interval": 2, "replicas": 3},
            "dynamics": {"integrator": "cayley-bcb", "block_steps": 7},
            "diffusion": {"atom_indices": [0], "bootstrap": 0},
            "output": {"checkpoint_steps": 4, "chunk_frames": 3}}


def run(cfg, path):
    return run_materials(cfg, system(), path, return_data=True)


def equal_runs(left, right, left_path, right_path, cfg):
    for a, b in zip(left["streams"], right["streams"]):
        for key in ("positions", "velocities", "observables", "radii_gyration_squared", "sample_steps", "times"):
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(a["state"].q, b["state"].q)
        np.testing.assert_array_equal(a["state"].p, b["state"].p)
        assert a["state"].time == b["state"].time
    a, b = (tm._read_batch(cfg, system(), path) for path in (left_path, right_path))
    for key in ("q", "p", "rng_sampling", "rng_production"):
        np.testing.assert_array_equal(a[key], b[key])
    assert a["production_started"] == b["production_started"]


@pytest.mark.parametrize("task", ["pimd", "diffusion"])
@pytest.mark.parametrize("layout", ["rfft", "rpmdrate"])
@pytest.mark.parametrize("failure_step", [4, 5, 7, 11])
def test_interruption_and_changed_block_chunk_checkpoint_sizes_are_exact(tmp_path, monkeypatch, task, layout, failure_step):
    cfg = config(task, layout=layout)
    ref_path, interrupted_path = tmp_path / "reference", tmp_path / "interrupted"
    reference = run(cfg, ref_path)
    original = TorchDynamics.propagate
    def fail(self, state, steps):
        value = original(self, state, steps)
        if state.step >= failure_step:
            raise RuntimeError("injected interruption after block")
        return value
    with monkeypatch.context() as patch:
        patch.setattr(TorchDynamics, "propagate", fail)
        with pytest.raises(RuntimeError, match="injected interruption"):
            run(cfg, interrupted_path)
    changed = copy.deepcopy(cfg)
    changed["dynamics"]["block_steps"] = 3
    changed["output"].update(checkpoint_steps=7, chunk_frames=5)
    resumed = run(changed, interrupted_path)
    equal_runs(reference, resumed, ref_path, interrupted_path, cfg)
    assert resumed["summary"]["engine"] == "torch"
    assert not resumed["summary"]["converged"]


@pytest.mark.parametrize("neq", [0, 5])
@pytest.mark.parametrize("failure_point", ["observation", "before_commit", "after_commit"])
def test_maxwell_phase_transition_is_committed_exactly_once(tmp_path, monkeypatch, neq, failure_point):
    cfg = config(neq=neq)
    ref_path, interrupted_path = tmp_path / "ref", tmp_path / "interrupted"
    reference = run(cfg, ref_path)
    if failure_point == "observation":
        original = tm._record
        def fail(state, *args):
            if state.step == neq:
                raise OSError("injected phase failure")
            return original(state, *args)
        name = "_record"
    else:
        original = tm.atomic_savez
        def fail(path, **arrays):
            if arrays.get("production_started") and int(arrays["step"]) == neq:
                if failure_point == "after_commit":
                    original(path, **arrays)
                raise OSError("injected phase failure")
            return original(path, **arrays)
        name = "atomic_savez"
    with monkeypatch.context() as patch:
        patch.setattr(tm, name, fail)
        with pytest.raises(RuntimeError, match="injected phase failure"):
            run(cfg, interrupted_path)
    resumed = run(cfg, interrupted_path)
    equal_runs(reference, resumed, ref_path, interrupted_path, cfg)
    # Draw the production Maxwell field with the documented independent seed.
    seed = int(np.random.SeedSequence([41, 1306, 1]).generate_state(1, dtype=np.uint64)[0])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((3, 4, 2, 3), dtype=torch.float64, generator=generator).numpy()
    expected_velocity = (noise * np.sqrt(system().masses[None, None, :, None] * 4 / 2)
                         / system().masses[None, None, :, None]).mean(axis=1)
    for replica, stream in enumerate(resumed["streams"]):
        np.testing.assert_allclose(stream["velocities"][0], expected_velocity[replica], rtol=0, atol=1e-16)
    saved = tm._read_batch(cfg, system(), interrupted_path)
    np.testing.assert_array_equal(saved["rng_production"], generator.get_state().numpy())


def test_multi_replica_chunk_failure_cannot_commit_a_partial_batch(tmp_path, monkeypatch):
    cfg = config(neq=0)
    reference = run(cfg, tmp_path / "reference")
    original = tm.TrajectoryStream.append
    def fail(self, record):
        original(self, record)
        if self.directory.name == "materials_0001" and record["sample_steps"] == 6:
            raise OSError("injected second replica chunk failure")
    with monkeypatch.context() as patch:
        patch.setattr(tm.TrajectoryStream, "append", fail)
        with pytest.raises(RuntimeError, match="second replica"):
            run(cfg, tmp_path / "resumed")
    saved = tm._read_batch(cfg, system(), tmp_path / "resumed")
    assert saved["step"] == 4
    assert [d["frames"] for d in saved["trajectories"]] == [3] * 3
    resumed = run(cfg, tmp_path / "resumed")
    equal_runs(reference, resumed, tmp_path / "reference", tmp_path / "resumed", cfg)


def test_completed_replay_never_constructs_provider_or_rewrites_checkpoint(tmp_path, monkeypatch):
    import pyrpmd.potentials as potentials
    cfg = config()
    reference = run(cfg, tmp_path)
    committed = tm._path(tmp_path).read_bytes()
    for name in ("summary.json", "samples/materials_0000/manifest.json", "results/diffusion.json"):
        (tmp_path / name).unlink()
    def unexpected(*args, **kwargs):
        raise AssertionError("completed replay must be read-only and model-free")
    monkeypatch.setattr(potentials, "make_potential", unexpected)
    monkeypatch.setattr(tm, "atomic_savez", unexpected)
    replay = run(cfg, tmp_path)
    assert tm._path(tmp_path).read_bytes() == committed
    assert replay["summary"] == reference["summary"]
    result = reanalyze_diffusion(cfg, system(), tmp_path)
    assert result["engine"] == "torch"


@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab"])
@pytest.mark.parametrize("layout", ["rfft", "rpmdrate"])
def test_free_centroid_transport_matches_ballistic_solution(tmp_path, integrator, layout):
    cfg = config(layout=layout)
    cfg["potential"] = {"provider": "free"}
    cfg["dynamics"]["integrator"] = integrator
    result = run(cfg, tmp_path)
    for stream in result["streams"]:
        expected = stream["positions"][0] + stream["times"][:, None, None] * stream["velocities"][0]
        np.testing.assert_allclose(stream["positions"], expected, rtol=0, atol=1e-14)
        np.testing.assert_allclose(stream["velocities"], np.broadcast_to(stream["velocities"][0], stream["velocities"].shape),
                                   rtol=0, atol=2e-15)
    assert result["summary"]["production_momenta"] == "independent_real_mass_maxwell"
    assert result["summary"]["nve_energy_diagnostics"]["within_tolerance"]


def test_tensor_quantum_estimators_match_action_derivative_and_numpy():
    from pyrpmd.potentials.torch_workflow import make_workflow_torch_pes
    rng = np.random.default_rng(73)
    q, p = (rng.normal(size=(3, 4, 2, 3)) for _ in range(2))
    cfg, sys = config("pimd"), system()
    potential = HarmonicPotential(k=.7)
    adapter = make_workflow_torch_pes(potential, device="cpu")
    sampler, _ = tm._engines(adapter, cfg, sys, tm.resolve_torch_materials(cfg, sys))
    result = tm.tensor_observables(TorchState.from_arrays(q, p), sampler)
    values = result["observables"].numpy()
    for i in range(3):
        expected, radii = quantum_observables(State(q[i], p[i]), potential, sys.masses, 2.0)
        np.testing.assert_allclose(values[i], expected, atol=2e-14)
        np.testing.assert_allclose(result["radii_gyration_squared"][i].numpy(), radii, atol=1e-15)
        def action(beta):
            spring = 4 * np.sum(sys.masses[None, :, None] * (q[i] - np.roll(q[i], 1, axis=0))**2) / (2 * beta)
            return 12 * np.log(beta) + spring + beta * .7 * np.sum(q[i]**2) / 8
        derivative = (action(2 + 1e-5) - action(2 - 1e-5)) / 2e-5
        assert values[i, 0] + values[i, 1] == pytest.approx(derivative, abs=3e-8)


def rewrite(path, change):
    with load_checkpoint(path) as saved:
        payload = {k: saved[k] for k in saved.files if not k.startswith("_pyrpmd")}
    change(payload)
    atomic_savez(path, kind=tm.KIND, **payload)


@pytest.mark.parametrize("corruption", ["clock", "phase", "q_dtype", "rng_dtype", "frames", "schema", "extra", "sources"])
def test_semantically_invalid_but_integrity_valid_checkpoint_is_rejected(tmp_path, corruption):
    cfg = config()
    run(cfg, tmp_path)
    def change(payload):
        if corruption == "clock":
            payload["time"] = 123.
        elif corruption == "phase":
            payload["production_started"] = False
        elif corruption == "q_dtype":
            payload["q"] = payload["q"].astype(np.float32)
        elif corruption == "rng_dtype":
            payload["rng_sampling"] = payload["rng_sampling"].astype(np.int64)
        elif corruption in ("frames", "schema"):
            descriptors = json.loads(str(payload["trajectories"]))
            if corruption == "frames":
                descriptors[1]["frames"] -= 1
            else:
                descriptors[0]["schema"]["observables"]["shape"] = [4]
            payload["trajectories"] = json.dumps(descriptors)
        elif corruption == "sources":
            payload["sources"] = json.dumps({"missing-source": "bad"})
        else:
            payload["unrecognized"] = 1
    rewrite(tm._path(tmp_path), change)
    with pytest.raises((CheckpointError, ValueError)):
        tm._read_batch(cfg, system(), tmp_path)


def test_runtime_mismatch_rejected_before_advancing_or_rewriting(tmp_path, monkeypatch):
    cfg = config()
    original = TorchDynamics.propagate
    def fail(*args):
        raise RuntimeError("initial interruption")
    with monkeypatch.context() as patch:
        patch.setattr(TorchDynamics, "propagate", fail)
        with pytest.raises(RuntimeError, match="initial interruption"):
            run(cfg, tmp_path)
    def change(payload):
        runtime = json.loads(str(payload["runtime"]))
        runtime["torch"] = "unknown-future-torch"
        payload["runtime"] = json.dumps(runtime)
    rewrite(tm._path(tmp_path), change)
    before = tm._path(tmp_path).read_bytes()
    with pytest.raises(RuntimeError, match="exact resume requires"):
        run(cfg, tmp_path)
    assert tm._path(tmp_path).read_bytes() == before


def test_execution_global_settings_are_restored_after_failure(tmp_path, monkeypatch):
    cfg = config()
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled())
    def fail(*args):
        raise RuntimeError("injected model failure")
    monkeypatch.setattr(TorchDynamics, "propagate", fail)
    with pytest.raises(RuntimeError, match="injected model failure"):
        run(cfg, tmp_path)
    after = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled())
    assert after == before


@pytest.mark.parametrize("key,value", [("device", "cuda:0"), ("normal_mode_layout", "unknown"),
                                     ("threads", True), ("deterministic_algorithms", False),
                                     ("torch", None), ("unexpected", 1)])
def test_completed_checkpoint_runtime_metadata_is_validated_without_torch_execution(tmp_path, key, value):
    cfg = config()
    run(cfg, tmp_path)
    def change(payload):
        runtime = json.loads(str(payload["runtime"]))
        runtime[key] = value
        payload["runtime"] = json.dumps(runtime)
    rewrite(tm._path(tmp_path), change)
    with pytest.raises(CheckpointError, match="runtime|device"):
        tm._read_batch(cfg, system(), tmp_path)


def test_saved_pimd_statistics_match_exact_finite_bead_harmonic_distribution(tmp_path):
    sys = System(("H",), np.array([2.0]), np.zeros((1, 3)))
    cfg = config("pimd")
    cfg.update(beta=4.0, beads=16, dt_au=.5, seed=79)
    cfg["potential"]["params"]["k"] = 2.0
    cfg["sampling"].update(equilibration_steps=600, production_steps=1600, replicas=48,
                           sample_interval=5, gamma_centroid=1.0)
    cfg["output"].update(checkpoint_steps=800, chunk_frames=128)
    cfg.pop("diffusion")
    result = run_materials(cfg, sys, tmp_path, return_data=True)
    # Independently diagonalize the primitive harmonic ring analytically.
    omega = 2 * cfg["beads"] / cfg["beta"] * np.sin(np.pi * np.arange(cfg["beads"]) / cfg["beads"])
    variance = np.sum(1 / (1 + omega**2)) / (cfg["beta"] * sys.masses[0])
    energy = 1.5 * 2.0 * variance
    exact = np.array([energy, energy, 2 * energy])
    means = np.array([stream["observables"][:, [0, 2, 3]].mean(axis=0) for stream in result["streams"]])
    sem = means.std(axis=0, ddof=1) / np.sqrt(len(means))
    assert np.all(np.abs(means.mean(axis=0) - exact) < np.maximum(6 * sem, .025 * exact))
    radius = np.mean([stream["radii_gyration_squared"].mean() for stream in result["streams"]])
    assert radius == pytest.approx(3 * (variance - 1 / (cfg["beta"] * 2.0)), rel=.035)
    assert result["summary"]["converged"] is False
