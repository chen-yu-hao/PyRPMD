"""Trajectory accounting and energy diagnostics survive persisted restarts."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.workflows import drpmd


def _dynamics():
    dynamics = Dynamics(
        HarmonicPotential(), [1.0], 1.0, 1, 0.2,
        native=False, loop_backend="python",
    )
    # The same Dynamics can have already run other trajectories.
    dynamics.propagation_stats.update(native_steps=11, python_steps=23)
    return dynamics


def _run(directory, *, resume=False, classifier=None, dynamics=None):
    config = {
        "beads": 1,
        "dt_au": 0.2,
        "collision": {"max_steps": 12},
        "outcomes": {"check_interval": 4},
        "output": {"trajectory_stride": 3, "checkpoint_steps": 4},
    }
    return drpmd._trajectory(
        dynamics or _dynamics(),
        State([[[0.5, 0.0, 0.0]]], [[[0.3, 0.0, 0.0]]]),
        classifier, config, Path(directory), 0, 0, {}, "restart-test", resume,
    )


def _interrupt_at(step, *, legacy=False):
    save = drpmd._atomic_npz

    def save_then_interrupt(path, metadata, **arrays):
        is_checkpoint = metadata["format"] == "PyRPMD.drpmd.trajectory-checkpoint.v1"
        if legacy and is_checkpoint:
            metadata = dict(metadata)
            metadata.pop("propagation", None)
        save(path, metadata, **arrays)
        if is_checkpoint and metadata["done"] == step:
            raise KeyboardInterrupt("interrupted after persisted trajectory checkpoint")

    return patch.object(drpmd, "_atomic_npz", side_effect=save_then_interrupt)


@pytest.mark.parametrize("interruption_step", [4, 12])
def test_resumed_trajectory_keeps_all_counters_and_energy_diagnostics(tmp_path, interruption_step):
    expected = _run(tmp_path / "full")
    with _interrupt_at(interruption_step), pytest.raises(KeyboardInterrupt):
        _run(tmp_path / "resumed")
    actual = _run(tmp_path / "resumed", resume=True)

    assert actual["propagation"] == expected["propagation"] == {
        "native_steps": 0, "python_steps": 12,
    }
    assert actual["steps"] == expected["steps"] == 12
    for key in ("initial_energy_per_bead", "final_energy_per_bead", "max_energy_drift_per_bead"):
        assert actual[key] == expected[key]
    assert expected["max_energy_drift_per_bead"] > 0
    with (
        np.load(tmp_path / "full/trajectories/trajectory-00000000.npz") as full,
        np.load(tmp_path / "resumed/trajectories/trajectory-00000000.npz") as resumed,
    ):
        for key in ("final_q", "final_p", "steps", "centroid", "energy_per_bead"):
            np.testing.assert_array_equal(resumed[key], full[key])


def test_legacy_checkpoint_resumes_without_inventing_backend_history(tmp_path):
    expected = _run(tmp_path / "full")
    with _interrupt_at(4, legacy=True), pytest.raises(KeyboardInterrupt):
        _run(tmp_path / "legacy")
    actual = _run(tmp_path / "legacy", resume=True)
    assert actual["propagation"] == {
        "native_steps": 0, "python_steps": 8, "untracked_steps": 4,
    }
    assert sum(actual["propagation"].values()) == actual["steps"] == 12
    for key in ("initial_energy_per_bead", "final_energy_per_bead", "max_energy_drift_per_bead"):
        assert actual[key] == expected[key]


def test_rejected_classifier_step_does_not_inflate_accepted_step_count(tmp_path):
    calls = 0

    def classifier(state):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("classifier failed")
        return None

    actual = _run(tmp_path, classifier=classifier)
    assert actual["status"] == "failed"
    assert actual["steps"] == 7
    assert actual["propagation"] == {"native_steps": 0, "python_steps": 7}
