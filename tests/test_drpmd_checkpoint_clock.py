"""Long accumulated clocks resume without admitting a changed time or step."""

import json
import math

import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics, State
from pyrpmd.nonadiabatic import DiabaticResult, NRPMDDynamics, NRPMDState
from pyrpmd.potentials import FreePotential
from pyrpmd.workflows import drpmd


class ZeroMatrixPotential:
    def evaluate(self, q):
        return DiabaticResult(np.zeros((len(q), 2, 2)),
                              np.zeros((len(q), 2, 2, *q.shape[1:])))


@pytest.mark.parametrize("steps", [2**53, 10**400])
def test_unbounded_clock_roundoff_fails_closed(steps):
    assert not drpmd._checkpoint_clock_matches(0.0, steps, 0.2)


def long_checkpoint(directory, *, mapping=False):
    steps, dt = 1_000_000, 0.2
    elapsed = 0.0
    # This is the actual binary64 addition sequence used by a stepwise clock,
    # not an estimated perturbation or a multiplication with the step count.
    for _ in range(steps):
        elapsed += dt
    assert not math.isclose(elapsed, steps * dt, rel_tol=1e-11, abs_tol=1e-12)

    state = State(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)))
    config = {"beads": 1, "dt_au": dt, "collision": {"max_steps": 2},
              "output": {"trajectory_stride": steps, "checkpoint_steps": steps}}
    if mapping:
        state = NRPMDState.from_nuclear(state, 2, 0, np.random.default_rng(5))
        dynamics = NRPMDDynamics(ZeroMatrixPotential(), [1.0], 1.0, 1, dt, 2,
                                 native=False)
        config["nonadiabatic"] = {"nstates": 2, "asymptotic_coupling_tolerance": 1e-8}
    else:
        dynamics = Dynamics(FreePotential(), [1.0], 1.0, 1, dt,
                            native=False, loop_backend="python")

    def run(*, resume=True, extend=False):
        return drpmd._trajectory(dynamics, state, None, config, directory,
                                 0, 0, {}, "long-clock", resume,
                                 extend_unresolved=extend)

    run(resume=False)
    checkpoint = directory / "trajectories/trajectory-00000000.checkpoint.npz"
    metadata, arrays = drpmd._load_npz(checkpoint)
    # These zero-force, zero-momentum states are stationary.  Preserve their
    # real checkpoint structure while supplying the million-step clock above.
    metadata.update(step=steps, done=steps, time=elapsed,
                    propagation={"native_steps": 0, "python_steps": steps})
    arrays["trace_steps"][-1] = steps
    drpmd._atomic_npz(checkpoint, metadata, **arrays)
    (directory / "trajectories/trajectory-00000000.json").unlink()
    config["collision"]["max_steps"] = steps + 1
    return run, checkpoint, config, elapsed


@pytest.mark.parametrize("mapping", [False, True], ids=["nuclear", "mapping"])
def test_million_step_accumulated_clock_resumes(mapping, tmp_path):
    run, checkpoint, config, elapsed = long_checkpoint(tmp_path, mapping=mapping)
    result = run()
    assert result["status"] == "unresolved"
    assert result["steps"] == 1_000_001
    metadata, _ = drpmd._load_npz(checkpoint)
    assert metadata["time"] == elapsed + config["dt_au"]
    assert metadata["step"] == metadata["done"] == 1_000_001


def test_completed_long_clock_can_extend_by_one_more_step(tmp_path):
    run, checkpoint, config, elapsed = long_checkpoint(tmp_path)
    config["collision"]["max_steps"] = 1_000_000
    assert run()["steps"] == 1_000_000
    config["collision"]["max_steps"] += 1
    result = run(extend=True)
    assert result["steps"] == 1_000_001
    assert result["propagation"]["python_steps"] == 1_000_001
    metadata, _ = drpmd._load_npz(checkpoint)
    assert metadata["time"] == elapsed + config["dt_au"]


@pytest.mark.parametrize("mapping", [False, True], ids=["nuclear", "mapping"])
@pytest.mark.parametrize("corruption", ["time_step", "state_step", "both_steps", "nan", "inf"])
def test_long_checkpoint_still_rejects_material_clock_corruption(mapping, corruption, tmp_path):
    run, checkpoint, config, _ = long_checkpoint(tmp_path, mapping=mapping)
    metadata, arrays = drpmd._load_npz(checkpoint)
    if corruption == "time_step":
        metadata["time"] += config["dt_au"]
    elif corruption == "state_step":
        metadata["step"] += 1
    elif corruption == "both_steps":
        metadata["step"] += 1
        metadata["done"] += 1
    else:
        metadata["time"] = float(corruption)
    # Deliberately bypass the writer's own finite-JSON guard to exercise the
    # reader against an externally corrupted checkpoint as well.
    np.savez(checkpoint, metadata=np.array(json.dumps(metadata)), **arrays)
    with pytest.raises(ValueError, match="clock"):
        run()
