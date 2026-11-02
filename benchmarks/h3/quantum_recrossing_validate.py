"""Excluded real-PES check of child native restart and observation semantics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.h3.quantum_surface_recrossing import (
    BETA, MASS, FS_TO_AU, BKMP2Potential, Dynamics, State, assign_channel,
    design, load_parents, maxwell_momenta, minimum_pair, sha, source_identity, surface_coordinate,
)
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.io import write_json


def validate(output, library, parent):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    before = source_identity()
    q, _, metadata = load_parents(parent, design(library), calibration=True, count=1)
    seed = metadata["seed"]
    q, channel = assign_channel(q[0], seed, 0)
    p = maxwell_momenta(seed, 0, 0, len(q))
    surface = surface_coordinate()
    started = time.perf_counter()
    rows = []
    for dt in (.1, .05):
        stride = round(1. / dt)
        def dynamics(potential):
            return Dynamics(potential, MASS, BETA, len(q), dt * FS_TO_AU,
                            native="required", loop_backend="native")
        def advance(engine, state, frames):
            observations = []
            for _ in range(frames):
                block = engine.propagate(state, stride, observe=surface,
                                         sample_interval=stride, record_centroids=False)
                if block.backend != "native":
                    raise ValueError("native replay unavailable")
                observations.append([float(block.values[-1, 0]), block.initial_energy,
                                     block.minimum_energy, block.maximum_energy, minimum_pair(state.q)])
            return np.asarray(observations)
        with BKMP2Potential(library) as potential:
            engine = dynamics(potential)
            continuous = State(q, p)
            continuous_values = advance(engine, continuous, 400)
            split = State(q, p)
            prefix = advance(engine, split, 200)
            path = output / f"restart-dt-{dt:g}.npz"
            atomic_savez(path, kind="h3.quantum-child-replay", q=split.q, p=split.p,
                         step=split.step, time_au=split.time, observations=prefix)
        # Recreate both the native dynamics context and provider after storage.
        with load_checkpoint(path, expected_kind="h3.quantum-child-replay", require_integrity=True) as saved:
            resumed = State(saved["q"], saved["p"], step=int(saved["step"]), time=float(saved["time_au"]))
            prefix = saved["observations"]
        with BKMP2Potential(library) as potential:
            engine = dynamics(potential)
            suffix = advance(engine, resumed, 200)
            split_values = np.concatenate((prefix, suffix))
        equal = {"final_q_bitwise_equal": np.array_equal(continuous.q, resumed.q),
                 "final_p_bitwise_equal": np.array_equal(continuous.p, resumed.p),
                 "all_observations_bitwise_equal": np.array_equal(continuous_values, split_values),
                 "step_equal": continuous.step == resumed.step, "time_equal": continuous.time == resumed.time}
        if not all(equal.values()):
            raise ValueError(f"native restart continuation differs: {equal}")
        observations = output / f"observations-dt-{dt:g}.npz"
        atomic_savez(observations, kind="h3.quantum-child-replay-observations",
                     time_fs=np.arange(1, 401), continuous=continuous_values, restored=split_values,
                     final_q=continuous.q, final_p=continuous.p)
        rows.append(dict(timestep_fs=dt, horizon_fs=400, split_after_fs=200, **equal,
                         final_step=continuous.step, maximum_q_difference=float(abs(continuous.q-resumed.q).max()),
                         maximum_p_difference=float(abs(continuous.p-resumed.p).max()),
                         restart_sha256=sha(path), observations_sha256=sha(observations),
                         native_steps=2*round(400/dt), python_steps=0))
    report = dict(schema="pyrpmd.h3.quantum-child-replay.v1", phase="excluded calibration",
                  source=metadata, parent_index=0, channel=channel,
                  settings="Same excluded parent and keyed Maxwell momentum for both step sizes.",
                  rows=rows, source_sha256=before, source_unchanged=before == source_identity(),
                  validation_source_sha256=sha(__file__), elapsed_seconds=time.perf_counter()-started,
                  created_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "native-restart-validation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.output, args.library, args.parent)
    print(json.dumps({"rows": report["rows"], "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
