#!/usr/bin/env python3
"""Deterministic model benchmark: RPC/MTS call counts and approximation error.

No molecular rate, ML throughput or ab initio speedup is inferred from this
artificial harmonic + quartic split. Atomic units are used throughout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from pyrpmd.dynamics import State
from pyrpmd.multitime import MTSDynamics
from pyrpmd.potentials import HarmonicPotential, PESResult
from pyrpmd.potentials.contraction import SplitPotential


class Correction:
    def evaluate(self, q):
        return PESResult(np.sum(0.002 * q**4 + 0.01 * q**2, axis=(1, 2)), -0.008 * q**3 - 0.02 * q)


class Counter:
    def __init__(self, provider):
        self.provider, self.calls, self.frames = provider, 0, 0

    def evaluate(self, q):
        self.calls += 1
        self.frames += len(q)
        return self.provider.evaluate(q)


def run_case(initial, p, pc, dt, inner, steps, native):
    fast, slow = Counter(HarmonicPotential(0.4)), Counter(Correction())
    split = SplitPotential(fast, slow, p, pc)
    dyn = MTSDynamics(split, [2.0], 4.0, p, dt, inner_steps=inner, native=native)
    state = initial.copy()
    energy0 = dyn.energy(state)
    fast.calls = fast.frames = slow.calls = slow.frames = 0
    start = time.perf_counter()
    dyn.propagate(state, steps)
    elapsed = time.perf_counter() - start
    result = {
        "nbeads": p, "contracted_beads": pc, "inner_steps": inner,
        "outer_steps": steps, "outer_dt_au": dt, "time_au": state.time,
        "fast_calls": fast.calls, "fast_frames": fast.frames,
        "slow_calls": slow.calls, "slow_frames": slow.frames,
        "wall_seconds": elapsed, "native_ring_kernel": dyn.native,
    }
    result["relative_hamiltonian_drift"] = float((dyn.energy(state) - energy0) / abs(energy0))
    return state, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nbeads", type=int, default=32)
    parser.add_argument("--contracted-beads", type=int, default=3)
    parser.add_argument("--inner-steps", type=int, default=4)
    parser.add_argument("--outer-steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--no-native", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.outer_steps < 1 or args.inner_steps < 1 or not np.isfinite(args.dt) or args.dt <= 0:
        parser.error("outer-steps, inner-steps and dt must be positive")
    rng = np.random.default_rng(args.seed)
    initial = State(rng.normal(scale=0.3, size=(args.nbeads, 1, 3)), rng.normal(size=(args.nbeads, 1, 3)))
    reference, full = run_case(initial, args.nbeads, args.nbeads, args.dt / args.inner_steps, 1,
                               args.outer_steps * args.inner_steps, not args.no_native)
    exact_split, mts = run_case(initial, args.nbeads, args.nbeads, args.dt, args.inner_steps,
                               args.outer_steps, not args.no_native)
    contracted, rpc_mts = run_case(initial, args.nbeads, args.contracted_beads, args.dt, args.inner_steps,
                                  args.outer_steps, not args.no_native)
    for state, record in ((exact_split, mts), (contracted, rpc_mts)):
        record["q_rms_difference_from_small_step_full"] = float(np.sqrt(np.mean((state.q - reference.q)**2)))
        record["p_rms_difference_from_small_step_full"] = float(np.sqrt(np.mean((state.p - reference.p)**2)))
        record["slow_geometry_count_reduction"] = full["slow_frames"] / record["slow_frames"]
    result = {
        "schema": "pyrpmd-rpc-mts-model-benchmark-v1",
        "model": "harmonic reference plus weak quartic correction; arbitrary atomic-unit parameters",
        "seed": args.seed, "physical_rate_validation": False,
        "small_step_full": full, "mts_full": mts, "rpc_mts": rpc_mts,
        "interpretation": "Counts quantify potential workload. RPC and MTS errors are shown separately; timings are not ML/ab-initio performance predictions.",
    }
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
