"""Small optional-PLUMED smoke run on a toy Morse bond; not a molecular FES."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyrpmd.constants import AMU_TO_AU, FS_TO_AU, KB_AU
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.plumed import PlumedBias
from pyrpmd.potentials.builtin import MorsePairPotential


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("runs/plumed-centroid"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    output = args.output.resolve()
    if any(c.isspace() for c in str(output)):
        parser.error("choose an output path without whitespace for PLUMED FILE keywords")
    output.mkdir(parents=True, exist_ok=False)
    temperature, beads = 300.0, 8
    beta, dt = 1 / (KB_AU * temperature), 0.05 * FS_TO_AU
    masses = np.array([1.00782503223, 1.00782503223]) * AMU_TO_AU
    dynamics = Dynamics(MorsePairPotential(), masses, beta, beads, dt, seed=2026)
    q = np.tile([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], (beads, 1, 1))
    state = State(q, dynamics.sample_momenta())
    lines = [
        "UNITS LENGTH=Bohr ENERGY=Ha TIME=atomic",
        "d: DISTANCE ATOMS=1,2 NOPBC",
        f"meta: METAD ARG=d SIGMA=0.1 HEIGHT=0.0001 PACE=20 BIASFACTOR=8 TEMP={temperature} FILE={output / 'HILLS'}",
        f"PRINT ARG=d,meta.bias STRIDE=10 FILE={output / 'COLVAR'}",
    ]
    (output / "plumed.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with PlumedBias(lines, masses, beta, beads, dt, log_file=output / "plumed.log") as bias:
        for _ in range(args.steps):
            bias.step(dynamics, state, thermostat="pile-l", gamma_centroid=0.02 / FS_TO_AU)
            if bias.stop_requested:
                break
        result = bias.evaluate(state.q)
        summary = {
            "model": "toy Morse bond", "converged": False,
            "beads": beads, "temperature_kelvin": temperature,
            "completed_steps": state.step, "time_atomic": state.time,
            "centroid_distance_bohr": float(np.linalg.norm(state.q.mean(axis=0)[1] - state.q.mean(axis=0)[0])),
            "physical_bias_hartree": result.physical_energy,
            "ring_bias_hartree": result.energy,
            "stop_requested": bias.stop_requested,
            "plumed_api_version": bias.api_version,
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
