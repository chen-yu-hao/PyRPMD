"""Long umbrella clocks remain readable without accepting corrupt state clocks."""

import json

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.dynamics import Dynamics
from pyrpmd.system import System
from pyrpmd.trajectory import TrajectoryStream
from pyrpmd.workflows import fes


@pytest.fixture(scope="module")
def accumulated_clock():
    steps, dt = 10_000_000, 0.003
    time = 0.0
    # Reproduce the actual binary64 addition sequence used by Dynamics, rather
    # than estimating its error or assigning time = steps * dt.
    for _ in range(steps):
        time += dt
    assert not np.isclose(time, steps * dt, rtol=1e-10, atol=abs(dt) * 1e-8)
    return steps, dt, time


def long_checkpoint(directory, clock, *, storage="arrays"):
    steps, dt, time = clock
    config = {
        "task": "fes", "beta": 1.7, "beads": 1, "dt_au": dt, "seed": 9131,
        "native": False, "potential": {"provider": "harmonic", "k": 1.0},
        "dynamics": {"loop_backend": "python", "block_steps": 1},
        "sampling": {"equilibration_steps": 0, "production_steps": steps,
                     "sample_interval": steps // 2, "andersen_interval": steps + 3},
        "output": {"checkpoint_steps": steps},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
        "umbrella": {"centers": [[0.0]], "stiffness": 0.4},
        "analysis": {"bins": 3, "bootstrap": 0},
    }
    system = System(("H",), np.array([2.3]), np.zeros((1, 3)))
    _, tasks = fes._make_tasks(config, system, directory)
    task = tasks[0]
    checkpoint = fes._window_checkpoint(task)
    # Supply a finite nuclear state and sparse history in the genuine payload
    # format; this fixture exercises clock validation, not a physical ensemble.
    payload = dict(q=np.zeros((1, 1, 3)), p=np.zeros((1, 1, 3)),
                   step=steps, time=time, elapsed=steps,
                   rng=json.dumps(np.random.default_rng(9131).bit_generator.state),
                   fingerprint=fes._window_fingerprint(task),
                   propagation=json.dumps({"native_steps": 0, "python_steps": steps}))
    history = {"values": np.zeros((2, 1)), "energies": np.zeros(2)}
    if storage == "stream":
        writer = TrajectoryStream(checkpoint.with_suffix(".chunks"),
                                  schema=fes._sample_schema(1))
        writer.extend(history)
        payload["sample_stream"] = json.dumps({
            "format": fes._SAMPLE_STREAM_FORMAT, "trajectory": writer.prepare_commit(),
        })
    else:
        payload.update(history)
    atomic_savez(checkpoint, **payload)
    return task, checkpoint


@pytest.mark.parametrize("storage", ["arrays", "stream"])
def test_long_clock_preflight_completed_resume_and_extension(
        tmp_path, monkeypatch, accumulated_clock, storage):
    task, checkpoint = long_checkpoint(tmp_path, accumulated_clock, storage=storage)
    steps, dt, time = accumulated_clock
    committed = checkpoint.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("read-only validation or completed resume performed new work")

    with monkeypatch.context() as patch:
        patch.setattr(fes, "task_potential", forbidden)
        records = fes.validate_fes_checkpoints(task["config"], task["system"], tmp_path)
    assert records[0]["elapsed"] == steps
    assert records[0]["samples"] == 2
    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, "step", forbidden)
        patch.setattr(Dynamics, "propagate_block", forbidden)
        patch.setattr(fes, "_atomic_npz", forbidden)
        result = fes._sample_window(task)
    assert len(result["samples"]) == 2
    assert checkpoint.read_bytes() == committed

    task["production_steps"] = steps + 1
    fes._sample_window(task)
    with load_checkpoint(checkpoint) as saved:
        assert saved["time"].item() == time + dt
        assert saved["step"].item() == saved["elapsed"].item() == steps + 1


@pytest.mark.parametrize("corruption", [
    "time_step", "state_step", "both_steps", "negative", "nan", "inf",
    "non_scalar", "complex", "unbounded_roundoff",
])
def test_long_clock_corruption_fails_before_provider_or_writes(
        tmp_path, monkeypatch, accumulated_clock, corruption):
    task, checkpoint = long_checkpoint(tmp_path, accumulated_clock)
    steps, dt, _ = accumulated_clock
    with load_checkpoint(checkpoint) as saved:
        payload = dict(saved.items())
    target = steps + 1
    if corruption == "time_step":
        payload["time"] += dt
    elif corruption == "state_step":
        payload["step"] += 1
    elif corruption == "both_steps":
        payload["step"] += 1
        payload["elapsed"] += 1
    elif corruption == "negative":
        payload["time"] *= -1
    elif corruption == "non_scalar":
        payload["time"] = np.atleast_1d(payload["time"])
    elif corruption == "complex":
        payload["time"] = payload["time"].astype(complex)
    elif corruption == "unbounded_roundoff":
        target = 2**53
        payload.update(step=target, elapsed=target, time=target * dt)
    else:
        payload["time"] = float(corruption)
    atomic_savez(checkpoint, **payload)
    committed = checkpoint.read_bytes()
    monkeypatch.setattr(fes, "task_potential", lambda *a, **k: pytest.fail("loaded PES"))
    with pytest.raises(ValueError, match="invalid umbrella checkpoint (clock|time)"):
        fes.validate_fes_checkpoints(task["config"], task["system"], tmp_path,
                                     production_steps=target)
    assert checkpoint.read_bytes() == committed
