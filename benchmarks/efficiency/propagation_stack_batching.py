"""Count shared PES calls with identical scalar propagation schedules.

Run from a built checkout:
  PYTHONPATH=src python benchmarks/efficiency/propagation_stack_batching.py \
      --output benchmarks/efficiency/propagation_stack_batching.json

This measures dispatch/call grouping, not end-to-end simulation speed. Every
case evaluates endpoint energy after every accepted step in both executions.
No timing threshold or physical-convergence claim is made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyrpmd.dynamics import Dynamics, State
from pyrpmd.engine import ForceSplitting
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.potentials.contraction import SplitPotential
from pyrpmd.trajectory_batch import make_trajectory_potential, run_trajectory_batch


class CountedHarmonic:
    metadata = {"trajectory_batch_safe": True}

    def __init__(self, stiffness):
        self.inner = HarmonicPotential(stiffness)
        self.calls = self.frames = self.max_frames = 0

    def evaluate(self, positions):
        self.calls += 1
        self.frames += len(positions)
        self.max_frames = max(self.max_frames, len(positions))
        return self.inner.evaluate(positions)

    def counts(self):
        return {"calls": self.calls, "frames": self.frames, "maximum_frames_per_call": self.max_frames}


def worker(task):
    provider = make_trajectory_potential(task, factory=lambda spec, system: task["scalar_provider"])
    dynamics = Dynamics(
        provider, [2., 3.], 2., task["beads"], .005,
        seed=np.random.SeedSequence([1729, task["index"]]),
        native=False, loop_backend="python", integrator="normal-mode-bab",
        thermostat=task["thermostat"],
        force_splitting=ForceSplitting(3) if task["split"] else None,
    )
    state = State(task["q"], task["p"])
    energy = [dynamics.energy(state)]
    for _ in range(task["steps"]):
        dynamics.step(state)
        energy.append(dynamics.energy(state))
    return state.q, state.p, np.asarray(energy), dynamics.operator_state_dict()


def providers(beads, split):
    if split:
        parts = {"fast": CountedHarmonic(.3), "slow": CountedHarmonic(.1)}
        return SplitPotential(parts["fast"], parts["slow"], beads, max(1, beads // 2)), parts
    parts = {"physical": CountedHarmonic(.4)}
    return parts["physical"], parts


def run_case(mode, split, trajectories, beads, steps):
    thermostat = {"mode": mode}
    if mode == "nhc":
        thermostat.update(tau=.7, chain_length=3, substeps=2)
    elif mode == "svr":
        thermostat["tau"] = .7
    elif mode == "gle":
        thermostat["gle"] = {"A": [[.7, .1], [-.1, .9]], "C": np.eye(2).tolist()}
    scalar, scalar_parts = providers(beads, split)
    batched, batch_parts = providers(beads, split)
    rng = np.random.default_rng(1301)
    tasks = [dict(config={"potential": {}}, system={}, scalar_provider=scalar,
                  index=i, q=rng.normal(scale=.2, size=(beads, 2, 3)),
                  p=rng.normal(size=(beads, 2, 3)), beads=beads, steps=steps,
                  thermostat=thermostat, split=split)
             for i in range(trajectories)]
    reference = [worker(task) for task in tasks]
    actual = run_trajectory_batch(worker, tasks, batched)
    errors = {name: max(float(np.max(np.abs(a[position] - b[position])))
                        for a, b in zip(reference, actual))
              for position, name in enumerate(("position", "momentum", "energy_trace"))}
    auxiliaries_equal = all(a[3] == b[3] for a, b in zip(reference, actual))
    if max(errors.values()) > 1e-12 or not auxiliaries_equal:
        raise AssertionError(f"scalar/batched operator mismatch: {mode}, split={split}: {errors}")
    scalar_counts = {name: part.counts() for name, part in scalar_parts.items()}
    batch_counts = {name: part.counts() for name, part in batch_parts.items()}
    if any(scalar_counts[name]["frames"] != batch_counts[name]["frames"] for name in scalar_parts):
        raise AssertionError("PES frame totals changed with batching")
    return {
        "thermostat": mode, "split_inner_steps": 3 if split else None,
        "contracted_beads": max(1, beads // 2) if split else None,
        "scalar": scalar_counts, "batched": batch_counts,
        "call_reduction_factor": sum(item["calls"] for item in scalar_counts.values())
                                 / sum(item["calls"] for item in batch_counts.values()),
        "maximum_absolute_differences": errors,
        "thermostat_and_rng_states_identical": auxiliaries_equal,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--beads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.trajectories, args.beads, args.steps) < 1:
        parser.error("trajectories, beads and steps must all be positive")
    result = {
        "format": "PyRPMD.propagation-stack-batch-counts.v1",
        "scope": "identical scalar operators, independent harmonic frames, no timing or physical convergence claim",
        "trajectories": args.trajectories, "beads": args.beads, "steps": args.steps,
        "seed": {"initial_conditions": 1301, "operators": 1729},
        "cases": [run_case(mode, split, args.trajectories, args.beads, args.steps)
                  for mode, split in [("none", False), ("andersen", False),
                                      ("pile-l", False), ("nhc", False), ("svr", False),
                                      ("gle", False), ("gle", True)]],
    }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
