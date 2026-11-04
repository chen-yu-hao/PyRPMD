"""Control-flow interrupts must not publish partly committed sampler state."""

import json

import numpy as np
import pytest

from pyrpmd.checkpoint import CheckpointError, load_checkpoint
from pyrpmd.constants import KB_AU
from pyrpmd.ffs import ForwardFluxSampler
from pyrpmd.retis import ReplicaExchangeTIS
from pyrpmd.runner import resume_run, run


def test_keyboard_interrupt_after_ffs_random_draw_replays_last_atomic_checkpoint(tmp_path, monkeypatch):
    config = {
        "task": "path_sampling", "temperature": 1 / KB_AU, "beads": 1, "seed": 71,
        "system": {"symbols": ["H"], "masses": [1.],
                   "geometry": [[-1.1, 0., 0.]], "geometry_unit": "bohr"},
        "potential": {"provider": "harmonic", "params": {"k": 1.}},
        "dynamics": {"timestep": "0.04 au", "native": False},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
        "path_sampling": {
            "method": "ffs", "cv": "x", "basin_a": -1., "basin_b": 1.,
            "interfaces": [-1., -.5], "flux_steps": 24, "trials_per_interface": 4,
            "max_trial_steps": 600, "dynamics": "basin-andersen", "collision_rate": 1.,
            "initial_state": {"q": [[-1.1, 0., 0.]], "p": [[-.4, 0., 0.]]},
        },
        "output": {"checkpoint_steps": 5},
    }
    uninterrupted, interrupted = tmp_path / "full", tmp_path / "split"
    run(config, output=uninterrupted)
    ordinary = ForwardFluxSampler._propagate
    calls = 0

    def interrupted_propagate(self, state):
        nonlocal calls
        before = repr(self.rng.bit_generator.state)
        retained = ordinary(self, state)
        calls += 1
        if calls == 7:
            assert repr(self.rng.bit_generator.state) != before
            raise KeyboardInterrupt("interrupt after completed forward random draw")
        return retained

    with monkeypatch.context() as patch:
        patch.setattr(ForwardFluxSampler, "_propagate", interrupted_propagate)
        with pytest.raises(KeyboardInterrupt, match="random draw"):
            run(config, output=interrupted)
    checkpoint = interrupted / "checkpoints" / "path_sampling.npz"
    with load_checkpoint(checkpoint, require_integrity=True) as saved:
        assert len(saved["flux_events"]) == 5
    result = resume_run(interrupted)
    assert result["completed"] is True
    assert result["physical_converged"] is False
    for subdirectory in ("checkpoints", "samples"):
        with load_checkpoint(uninterrupted / subdirectory / "path_sampling.npz", require_integrity=True) as expected:
            with load_checkpoint(interrupted / subdirectory / "path_sampling.npz", require_integrity=True) as actual:
                assert set(actual.files) == set(expected.files)
                for key in actual.files:
                    np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


@pytest.mark.parametrize("burn_in", [5, 1])
def test_keyboard_interrupt_after_retis_sweep_replays_last_atomic_checkpoint(tmp_path, monkeypatch, burn_in):
    # With five burn-in sweeps the third call interrupts warm-up; with one it
    # interrupts production.  In both cases the numerical sweep has returned,
    # but the caller has not yet committed its counters/observation records.
    config = {
        "task": "path_sampling", "temperature": 1 / KB_AU, "beads": 1, "seed": 91,
        "system": {"symbols": ["H"], "masses": [1.],
                   "geometry": [[-.3, 0., 0.]], "geometry_unit": "bohr"},
        "potential": {"provider": "harmonic", "params": {"k": 1.}},
        "dynamics": {"timestep": "0.04 au", "native": False},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
        "path_sampling": {
            "method": "retis", "cv": "x", "basin_a": -1., "basin_b": 1.,
            "interfaces": [-1., -.5], "samples": 7, "burn_in": burn_in,
            "max_path_steps": 600,
            "seeds": [{"q": [[-.3, 0., 0.]], "p": [[1.4, 0., 0.]]}],
            "minus_seed": {"q": [[-1.1, 0., 0.]], "p": [[-.7, 0., 0.]]},
        },
        "output": {"checkpoint_steps": 2},
    }
    uninterrupted, interrupted = tmp_path / "full", tmp_path / "split"
    run(config, output=uninterrupted)
    ordinary = ReplicaExchangeTIS._sweep
    calls = 0

    def interrupted_sweep(self):
        nonlocal calls
        before = repr(self.rng.bit_generator.state)
        result = ordinary(self)
        calls += 1
        if calls == 3:
            assert repr(self.rng.bit_generator.state) != before
            raise KeyboardInterrupt("interrupt after complete numerical RETIS sweep")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(ReplicaExchangeTIS, "_sweep", interrupted_sweep)
        with pytest.raises(KeyboardInterrupt, match="complete numerical RETIS sweep"):
            run(config, output=interrupted)
    with load_checkpoint(interrupted / "checkpoints" / "path_sampling.npz", require_integrity=True) as saved:
        metadata = json.loads(str(saved["metadata"].item()))
        assert metadata["sweeps"] == 2
        assert metadata["burn_in"] == min(2, burn_in)
        assert metadata["samples"] == max(0, 2 - burn_in)
    result = resume_run(interrupted)
    assert result["completed"] is True
    assert result["samples"] == 7
    assert result["physical_converged"] is False
    for subdirectory in ("checkpoints", "samples"):
        with load_checkpoint(uninterrupted / subdirectory / "path_sampling.npz", require_integrity=True) as expected:
            with load_checkpoint(interrupted / subdirectory / "path_sampling.npz", require_integrity=True) as actual:
                assert set(actual.files) == set(expected.files)
                for key in actual.files:
                    np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def test_terminal_ffs_without_a_history_and_missing_checkpoint_do_not_restart(tmp_path):
    config = {
        "task": "path_sampling", "temperature": 1 / KB_AU, "beads": 1,
        "system": {"symbols": ["H"], "masses": [1.],
                   "geometry": [[0., 0., 0.]], "geometry_unit": "bohr"},
        "potential": {"provider": "harmonic", "params": {"k": 1.}},
        "dynamics": {"timestep": "0.04 au", "native": False},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
        "path_sampling": {
            "method": "ffs", "cv": "x", "basin_a": -1., "basin_b": 1.,
            "interfaces": [-1., -.5], "flux_steps": 2, "trials_per_interface": 4,
            "dynamics": "hamiltonian",
            "initial_state": {"q": [[0., 0., 0.]], "p": [[0., 0., 0.]]},
        },
        "output": {"checkpoint_steps": 1},
    }
    output = tmp_path / "no-history"
    result = run(config, output=output)
    assert result["completed"] is True
    assert result["sampling_support_complete"] is False
    assert result["status"] == "no_flux_crossings"
    assert result["operations_completed"] == 2
    assert result["rate_estimate_au"] is None
    assert resume_run(output)["operations_completed"] == 2
    summary = output / "results" / "summary.json"
    saved_summary = summary.read_bytes()
    checkpoint = output / "checkpoints" / "path_sampling.npz"
    checkpoint.unlink()
    with pytest.raises(CheckpointError, match="missing.*recorded progress"):
        resume_run(output)
    assert not checkpoint.exists()
    assert summary.read_bytes() == saved_summary
