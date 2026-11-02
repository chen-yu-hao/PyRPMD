"""Executable, non-production, fixed-full-ring RPMD committor demonstration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyrpmd import CommittorSampler
from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import Dynamics
from pyrpmd.potentials import PESResult


class DoubleWell:
    """U(x,y,z)=(x^2-1)^2+(y^2+z^2)/2 in atomic units."""
    def evaluate(self, q):
        energy = np.sum((q[:, :, 0] ** 2 - 1) ** 2, axis=1)
        energy += 0.5 * np.sum(q[:, :, 1:] ** 2, axis=(1, 2))
        force = -q.copy()
        force[:, :, 0] = -4 * q[:, :, 0] * (q[:, :, 0] ** 2 - 1)
        return PESResult(energy, force)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--beads", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--output", type=Path, default=Path("runs/committor-example"))
    args = parser.parse_args()
    if min(args.shots, args.beads, args.max_steps) < 1:
        parser.error("shots, beads and max-steps must all be positive")
    dynamics = Dynamics(DoubleWell(), [1.0], 4.0, args.beads, 0.01, native=False)
    sampler = CommittorSampler(
        dynamics, Cartesian(0, "x"), basin_a=-0.8, basin_b=0.8,
        max_steps=args.max_steps, seed=args.seed,
    )
    # Fixed collapsed barrier ring: NOT an equilibrium sample of bead modes
    # conditioned on the centroid. Do not interpret as a physical rate.
    result = sampler.sample(np.zeros((args.beads, 1, 3)), n_shots=args.shots)
    summary = result.summary()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez(args.output / "shots.npz", configuration=result.configuration,
             outcomes=result.outcomes, steps=result.steps, final_values=result.final_values)
    (args.output / "rng.json").write_text(
        json.dumps(result.rng_state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
