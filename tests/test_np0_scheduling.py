"""E0 scheduling regressions for the NP0 native-path quick wins."""

import copy

import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics, native_capabilities
from pyrpmd.system import System
from pyrpmd.workflows.materials import _sample_replica
from pyrpmd.workflows.rate import _child_task


pytestmark = pytest.mark.skipif(
    not native_capabilities()["available"], reason="native build required"
)


def _materials_config(*, andersen_interval=7, block_steps=256):
    return {
        "task": "diffusion", "beta": 1.7, "beads": 4, "dt_au": 0.04,
        "seed": 2711, "native": "required",
        "potential": {"provider": "harmonic", "params": {"k": 0.02}},
        "sampling": {"equilibration_steps": 5, "production_steps": 20,
                     "sample_interval": 6, "replicas": 1,
                     "thermostat": "andersen", "andersen_interval": andersen_interval},
        "dynamics": {"loop_backend": "auto", "block_steps": block_steps},
        "diffusion": {"atom_indices": [0], "axes": [0, 1, 2], "bootstrap": 0},
        "output": {"checkpoint_steps": 9, "trajectory_storage": "memory"},
    }


def _materials_task(config, output, *, cell=None, pbc=None):
    return {
        "config": config,
        "system": System(("H",), np.array([2.3]), np.array([[0.4, -0.2, 0.1]]),
                          cell=cell, pbc=pbc),
        "output": str(output), "replica": 0,
    }


def _assert_same_materials_stream(left, right):
    for key in ("positions", "velocities", "observables", "radii_gyration_squared",
                "sample_steps", "times"):
        np.testing.assert_array_equal(left[key], right[key])
    for key in ("q", "p"):
        np.testing.assert_array_equal(getattr(left["state"], key), getattr(right["state"], key))
    assert left["state"].step == right["state"].step
    assert left["state"].time == right["state"].time
    assert left["elapsed"] == right["elapsed"]
    assert left["rng"] == right["rng"]
    assert left["propagation"] == right["propagation"]


@pytest.mark.parametrize("andersen_interval", [1, 7, 10])
def test_materials_andersen_blocks_are_e0_and_bound_native_calls(
        tmp_path, monkeypatch, andersen_interval):
    config = _materials_config(andersen_interval=andersen_interval)
    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, "can_fuse", lambda self, **kwargs: False)
        reference = _sample_replica(
            _materials_task(copy.deepcopy(config), tmp_path / "reference"))

    from pyrpmd import _native
    calls = []
    original = _native.advance

    def recorded_advance(context, binding, q, p, options, mask, records, states):
        calls.append(len(mask))
        return original(context, binding, q, p, options, mask, records, states)

    monkeypatch.setattr(_native, "advance", recorded_advance)
    actual = _sample_replica(_materials_task(config, tmp_path / "actual"))
    _assert_same_materials_stream(reference, actual)
    assert actual["propagation"]["sampling"]["python_steps"] == 0
    assert actual["propagation"]["production"]["python_steps"] == 0
    assert calls and max(calls) <= config["dynamics"]["block_steps"]
    assert len(calls) < reference["elapsed"]


def test_materials_periodic_cell_keeps_step_path(tmp_path, monkeypatch):
    config = _materials_config(andersen_interval=7)
    config["task"] = "pimd"
    config["potential"] = {"provider": "free"}
    config["sampling"]["production_steps"] = 10
    config["sampling"]["equilibration_steps"] = 0
    task = _materials_task(
        config, tmp_path / "periodic", cell=np.eye(3) * 20.0, pbc=(True, True, True)
    )

    def unexpected_block(*args, **kwargs):
        raise AssertionError("periodic materials must keep the step path")

    monkeypatch.setattr(Dynamics, "propagate_block", unexpected_block)
    result = _sample_replica(task)
    assert result["elapsed"] == 10


def _child_task_config(*, native):
    system = System(
        ("H", "H", "H"), np.array([2.0, 2.0, 2.0]),
        np.array([[-1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        ((0,), (1, 2)),
    )
    config = {
        "task": "rate", "beta": 10.0, "beads": 4, "dt_au": 0.01,
        "seed": 7, "native": native,
        "potential": {"provider": "harmonic", "params": {"k": 0.02}},
        "sampling": {}, "recrossing": {"child_batch_pairs": 1},
        "dynamics": {"block_steps": 7},
        "reaction": {"fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
                     "breaking_bonds": [[1, 2]], "rinf": 10.0},
    }
    return system, config, np.repeat(system.geometry[None], 4, axis=0)


def test_recrossing_child_blocks_match_reference_and_boundaries(tmp_path, monkeypatch):
    system, config, q = _child_task_config(native="required")
    task = {
        "config": config, "system": system, "chain": 0, "parent": 0, "pair": 0,
        "q": q, "xi_star": 1.0, "child_steps": 20,
        "output": str(tmp_path / "actual"),
    }
    from pyrpmd import _native
    calls = []
    original = _native.advance

    def recorded_advance(context, binding, qv, pv, options, mask, records, states):
        calls.append(len(mask))
        return original(context, binding, qv, pv, options, mask, records, states)

    monkeypatch.setattr(_native, "advance", recorded_advance)
    actual = _child_task(task)

    reference_task = copy.deepcopy(task)
    reference_task["output"] = str(tmp_path / "reference")
    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, "can_fuse", lambda self, **kwargs: False)
        reference = _child_task(reference_task)

    np.testing.assert_array_equal(actual["numerator"], reference["numerator"])
    assert actual["denominator"] == reference["denominator"]
    assert actual["energy_drift"] == reference["energy_drift"]
    assert actual["endpoint_energy_error_hartree_per_bead"] == reference["endpoint_energy_error_hartree_per_bead"]
    assert calls == [7, 7, 6, 7, 7, 6]
    assert actual["propagation"]["python_steps"] == 0
