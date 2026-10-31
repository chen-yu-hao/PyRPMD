#!/usr/bin/env python3
"""Measure independent-trajectory PES batching without claiming GPU speedups.

Run from an installed checkout, for example:
  python benchmarks/batched_dynamics.py --trajectories 16 --beads 32 --steps 128

Optional artificial per-call latency models dispatch/model-launch overhead only;
it is reported explicitly and is never evidence for a particular ML provider.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

from pyrpmd.batched import BatchedDynamics, BatchedState
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.potentials import HarmonicPotential


class CountedHarmonic:
    metadata = {"trajectory_batch_safe": True}

    def __init__(self, latency):
        self.provider = HarmonicPotential(.02)
        self.calls = 0
        self.frames = 0
        self.latency = latency

    def evaluate(self, q):
        self.calls += 1
        self.frames += len(q)
        if self.latency:
            time.sleep(self.latency)
        return self.provider.evaluate(q)


def run(trajectories=16, beads=32, atoms=3, steps=128, repeats=3, native=True,
        call_latency_us=0.):
    for key, value in (("trajectories", trajectories), ("beads", beads), ("atoms", atoms),
                       ("steps", steps), ("repeats", repeats)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not np.isfinite(call_latency_us) or call_latency_us < 0:
        raise ValueError("call latency must be finite and nonnegative")
    rng = np.random.default_rng(48391)
    q, p = rng.normal(size=(2, trajectories, beads, atoms, 3))
    masses = np.linspace(2., 4., atoms)
    scalar_times, batch_times = [], []
    maximum_q_error = maximum_p_error = 0.
    for repeat in range(repeats):
        # Alternate order so warm caches are not systematically assigned to one
        # path. Provider construction is outside both propagation timers.
        final = {}
        for name in (("scalar", "batch") if repeat % 2 == 0 else ("batch", "scalar")):
            potential = CountedHarmonic(call_latency_us*1e-6)
            if name == "scalar":
                dynamics = Dynamics(potential, masses, 10., beads, .005, native=native,
                                    loop_backend="python")
                states = [State(qi, pi) for qi, pi in zip(q, p)]
                start = time.perf_counter()
                for state in states:
                    dynamics.propagate_block(state, steps, record_energy=False)
                scalar_times.append(time.perf_counter()-start)
                final[name] = (np.stack([s.q for s in states]), np.stack([s.p for s in states]))
                scalar_counts = {"pes_calls": potential.calls, "pes_frames": potential.frames}
            else:
                dynamics = BatchedDynamics(potential, masses, 10., beads, .005, native=native)
                state = BatchedState(q, p)
                start = time.perf_counter()
                dynamics.propagate_block(state, steps, record_centroids=False)
                batch_times.append(time.perf_counter()-start)
                final[name] = (state.q, state.p)
                batch_counts = {"pes_calls": potential.calls, "pes_frames": potential.frames}
        maximum_q_error = max(maximum_q_error, float(np.max(np.abs(final["scalar"][0]-final["batch"][0]))))
        maximum_p_error = max(maximum_p_error, float(np.max(np.abs(final["scalar"][1]-final["batch"][1]))))
        np.testing.assert_allclose(final["batch"][0], final["scalar"][0], atol=2e-11, rtol=2e-12)
        np.testing.assert_allclose(final["batch"][1], final["scalar"][1], atol=2e-11, rtol=2e-12)
    return {
        "model": "analytic harmonic reference with explicit optional synthetic call latency",
        "physical_accuracy_claim": False,
        "gpu_benchmark": False,
        "settings": {"trajectories": trajectories, "beads": beads, "atoms": atoms,
                     "steps": steps, "repeats": repeats, "native_requested": native,
                     "synthetic_call_latency_us": call_latency_us, "seed": 48391},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "platform": platform.platform(), "native": native_capabilities()},
        "scalar": {**scalar_counts, "seconds": scalar_times,
                   "median_seconds": float(np.median(scalar_times))},
        "batch": {**batch_counts, "seconds": batch_times,
                  "median_seconds": float(np.median(batch_times))},
        "median_speedup": float(np.median(scalar_times)/np.median(batch_times)),
        "pes_call_reduction": scalar_counts["pes_calls"]/batch_counts["pes_calls"],
        "maximum_q_error_bohr": maximum_q_error,
        "maximum_p_error_au": maximum_p_error,
        "notes": [
            "Both paths use the same ring integrator; scalar uses current Python PES orchestration.",
            "Batch uses block-local force reuse, so frame count falls by almost two in this comparison.",
            "No real ML model, GPU, ab initio PES or transfer-residency speedup is established.",
            "Batch orchestration/observation remains Python; the free-ring map may use native C++.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("trajectories", 16), ("beads", 32), ("atoms", 3), ("steps", 128), ("repeats", 3)):
        parser.add_argument("--"+name, type=int, default=default)
    parser.add_argument("--numpy-ring", action="store_true")
    parser.add_argument("--call-latency-us", type=float, default=0.)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output
    settings = vars(args).copy()
    settings.pop("output")
    settings["native"] = not settings.pop("numpy_ring")
    text = json.dumps(run(**settings), indent=2, allow_nan=False)+"\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
