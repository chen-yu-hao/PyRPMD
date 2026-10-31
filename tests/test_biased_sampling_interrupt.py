"""Fixed-bias failures must not commit partially mutated state or RNG."""

import copy
import json
import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.dynamics import Dynamics
from pyrpmd.system import System
from pyrpmd.workflows.fes import _sample_window


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("block", [False, True])
def test_fixed_bias_interrupt_preserves_last_durable_checkpoint(tmp_path, monkeypatch, interrupt, block):
    specs = [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}]
    config = {
        "task": "fes", "beta": 2.0, "beads": 3, "dt_au": 0.02, "native": False, "seed": 321,
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 0.2}},
        "sampling": {"equilibration_steps": 0, "production_steps": 20,
                     "sample_interval": 1, "andersen_interval": 3},
        "cvs": specs, "output": {"checkpoint_steps": 5},
    }
    system = System(("H",), np.array([1.0]), np.array([[0.1, -0.05, 0.0]]))

    def task(directory):
        return {"config": config, "system": system, "output": str(directory),
                "specs": specs, "window": 0, "replica": 0,
                "center": [0.1], "stiffness": [[0.3]]}

    # Exercise both fixed-bias control paths without requiring a native binary.
    # propagate_block's Python backend still has the same partial-commit API.
    monkeypatch.setattr(Dynamics, "can_fuse", lambda self, **kwargs: block)
    reference = _sample_window(task(tmp_path / "full"))
    method = "propagate_block" if block else "step"
    original = getattr(Dynamics, method)
    checkpoint = tmp_path / "interrupted/checkpoints/window_00000_replica_0000.npz"
    committed = []

    def fail_after_mutation(self, state, *args, **kwargs):
        result = original(self, state, *args, **kwargs)
        if state.step == 12:
            committed.append(checkpoint.read_bytes())
            raise interrupt("injected asynchronous interruption")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, method, fail_after_mutation)
        with pytest.raises(interrupt, match="asynchronous"):
            _sample_window(task(tmp_path / "interrupted"))
    assert committed == [checkpoint.read_bytes()]
    with load_checkpoint(checkpoint) as saved:
        assert int(saved["elapsed"]) == int(saved["step"]) == 10
    resumed = _sample_window(task(tmp_path / "interrupted"))
    for key in ("last_q", "last_p", "samples", "energies"):
        np.testing.assert_array_equal(resumed[key], reference[key])


@pytest.mark.parametrize("block", [False, True])
def test_failed_andersen_draw_restores_rng_before_checkpointing(tmp_path, monkeypatch, block):
    specs = [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}]
    config = {
        "task": "fes", "beta": 2.0, "beads": 3, "dt_au": 0.02, "native": False, "seed": 321,
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 0.2}},
        "sampling": {"equilibration_steps": 0, "production_steps": 20,
                     "sample_interval": 1, "andersen_interval": 3},
        "cvs": specs, "output": {"checkpoint_steps": 5},
    }
    system = System(("H",), np.array([1.0]), np.array([[0.1, -0.05, 0.0]]))

    def task(directory):
        return {"config": config, "system": system, "output": str(directory),
                "specs": specs, "window": 0, "replica": 0,
                "center": [0.1], "stiffness": [[0.3]]}

    monkeypatch.setattr(Dynamics, "can_fuse", lambda self, **kwargs: block)
    reference = _sample_window(task(tmp_path / "full"))
    original = Dynamics.sample_momenta
    draws, before_failure = 0, None

    def failing_draw(self, *args, **kwargs):
        nonlocal draws, before_failure
        draws += 1
        before = copy.deepcopy(self.rng.bit_generator.state)
        result = original(self, *args, **kwargs)
        # Initial momenta plus the Andersen draws at steps 0, 3 and 6.
        if draws == 4:
            before_failure = before
            assert self.rng.bit_generator.state != before
            raise RuntimeError("injected after consuming thermostat RNG")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, "sample_momenta", failing_draw)
        with pytest.raises(RuntimeError, match="thermostat RNG"):
            _sample_window(task(tmp_path / "interrupted"))
    checkpoint = tmp_path / "interrupted/checkpoints/window_00000_replica_0000.npz"
    with load_checkpoint(checkpoint) as saved:
        assert int(saved["elapsed"]) == int(saved["step"]) == 6
        assert json.loads(str(saved["rng"])) == before_failure
    resumed = _sample_window(task(tmp_path / "interrupted"))
    for key in ("last_q", "last_p", "samples", "energies"):
        np.testing.assert_array_equal(resumed[key], reference[key])
