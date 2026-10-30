"""Adaptive surface FES retains the full slab in both sampling stages."""

import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.dynamics import Dynamics
from pyrpmd.potentials import import_object
from pyrpmd.system import System
from pyrpmd.workflows.fes import run_fes


@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab", "velocity-verlet"])
def test_periodic_fixed_adaptive_pilot_production_and_completed_resume(tmp_path, monkeypatch, integrator):
    system = System(
        ("H", "Cu"), np.array([2.0, 20.0]),
        np.array([[0.3, -0.4, 1.2], [0.3, -0.4, 0.2]]),
        cell=np.diag([4.0, 5.0, 12.0]), pbc=[True, True, False], fixed_atoms=[1],
    )
    factory = Path(__file__).resolve().parents[1] / "examples/surface/toy_surface.py"
    config = {
        "task": "fes", "beta": 2.0, "beads": 2, "dt_au": 0.02,
        "seed": 92, "native": False,
        "dynamics": {"integrator": integrator},
        "potential": {"provider": "python", "factory": f"{factory}:build",
                      "params": {"barrier": 0.2, "transverse_barrier": 0.2,
                                 "height": 1.0, "vertical_k": 1.0}},
        "sampling": {"equilibration_steps": 6, "production_steps": 32,
                     "sample_interval": 2, "andersen_interval": 7, "replicas": 1},
        "cvs": [{"kind": "surface_lateral", "atoms": [0], "reference_atoms": [1],
                 "direction": [1.0, 0.0, 0.0]}],
        "umbrella": {"centers": [-0.25, 0.25], "stiffness": 2.0},
        "adaptive_umbrella": {"pilot_steps": 24, "pilot_equilibration_steps": 8,
                              "max_rounds": 2, "max_windows": 4,
                              "target_overlap": 0.0001, "min_effective_samples": 1.0},
        "analysis": {"bins": 6, "bootstrap": 0, "measure": "cv_density"},
        "output": {"checkpoint_steps": 8},
    }
    model = import_object(f"{factory}:CorrugatedSurface")
    original_bind, original_evaluate = model.bind_system, model.evaluate
    observations = {}

    def bind(provider, bound):
        assert bound.pbc == system.pbc
        assert bound.fixed_atoms == system.fixed_atoms
        np.testing.assert_array_equal(bound.cell, system.cell)
        np.testing.assert_array_equal(bound.geometry, system.geometry)
        original_bind(provider, bound)
        observations[provider] = {"calls": 0, "fixed_force": False}

    def evaluate(provider, q):
        # Check every force evaluation, including trial integration positions.
        np.testing.assert_array_equal(provider.cell, system.cell)
        np.testing.assert_array_equal(q[:, 1], np.broadcast_to(system.geometry[1], q[:, 1].shape))
        result = original_evaluate(provider, q)
        observations[provider]["calls"] += 1
        observations[provider]["fixed_force"] |= bool(np.any(result.forces[:, 1] != 0))
        return result

    original_step = Dynamics.step
    step_count = 0

    def step(dynamics, state, bias=None):
        nonlocal step_count
        assert dynamics.integrator_name == integrator
        assert dynamics.fixed_atoms == system.fixed_atoms
        result = original_step(dynamics, state, bias)
        np.testing.assert_array_equal(state.p[:, 1], 0.0)
        step_count += 1
        return result

    monkeypatch.setattr(model, "bind_system", bind)
    monkeypatch.setattr(model, "evaluate", evaluate)
    monkeypatch.setattr(Dynamics, "step", step)
    result = run_fes(config, system, tmp_path, return_data=True)
    plan_path = tmp_path / "adaptive-plan.json"
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    assert plan["status"] == "frozen"
    assert len(plan["rounds"]) == 1
    assert plan["production_seed"] != plan["rounds"][0]["seed"]
    assert result["summary"]["samples"] == 2 * 16
    assert result["summary"]["adaptive_umbrella"]["pilot_samples_used_in_production"] == 0

    pilot_paths = sorted((tmp_path / "adaptation").rglob("window_*.npz"))
    production_paths = sorted((tmp_path / "checkpoints").glob("window_*.npz"))
    assert len(pilot_paths) == len(production_paths) == 2
    assert len(observations) == 4
    assert all(item["calls"] > 0 and item["fixed_force"] for item in observations.values())
    assert step_count == 2 * (8 + 24) + 2 * (6 + 32)
    for paths, expected_elapsed, expected_samples in ((pilot_paths, 32, 12), (production_paths, 38, 16)):
        for path in paths:
            with load_checkpoint(path, require_integrity=True) as saved:
                assert int(saved["elapsed"]) == expected_elapsed
                assert len(saved["values"]) == expected_samples
                assert np.ptp(saved["values"][:, 0]) > 0
                np.testing.assert_array_equal(saved["q"][:, 1], np.broadcast_to(system.geometry[1], (2, 3)))
                np.testing.assert_array_equal(saved["p"][:, 1], 0.0)
                propagation = json.loads(str(saved["propagation"]))
                assert propagation["python_steps"] == expected_elapsed
                assert propagation["native_steps"] == 0

    previous_calls = {provider: item["calls"] for provider, item in observations.items()}

    def no_propagation(*args, **kwargs):
        raise AssertionError("completed adaptive surface run propagated")

    monkeypatch.setattr(Dynamics, "step", no_propagation)
    monkeypatch.setattr(Dynamics, "propagate_block", no_propagation)
    resumed = run_fes(config, system, tmp_path, return_data=True)
    assert plan_path.read_bytes() == plan_bytes
    assert len(observations) == 6  # Only frozen production windows are reopened.
    assert all(item["calls"] == previous_calls.get(provider, 0)
               for provider, item in observations.items())
    for before, after in zip(result["streams"], resumed["streams"]):
        for key in ("samples", "energies", "last_q", "last_p"):
            np.testing.assert_array_equal(after[key], before[key])
        assert after["propagation"] == before["propagation"]
    np.testing.assert_array_equal(resumed["data"]["free_energy"], result["data"]["free_energy"])
