"""Per-step cost of the native block loop outside the PES.

Uses the built-in native harmonic PES (nearly free) with the published H+CH4
four-channel RPMDrate coordinate, its umbrella bias and flux metric, an energy
check on every step and a CV/centroid observation every 20 steps: the work the
fixed-bias FES/PMF sampler asks of ``Dynamics.propagate_block``. Because the
PES costs almost nothing, the timing is the loop's own data movement,
reaction-coordinate, bias and recording cost. It is a microbenchmark, not a
sampling run; blocks restart from one state.

    python benchmarks/efficiency/native_step_overhead.py --output overhead.json
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from pyrpmd import _native
from pyrpmd.bias import CompositeBias, FluxMetricBias, UmbrellaBias
from pyrpmd.constants import AMU_TO_AU, ANGSTROM_TO_AU, FS_TO_AU, KB_AU
from pyrpmd.cv import RPMDrateCoordinate
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import HarmonicPotential

ROOT = Path(__file__).resolve().parents[2]
HCH4 = ROOT / "benchmarks" / "hch4"


def setup():
    physics = json.loads((HCH4 / "reproduction_protocol.json").read_text())["planned_physics"]
    geometry = np.asarray(json.loads((HCH4 / "geometries.json").read_text())
                          ["official_input"]["coordinates"]) * ANGSTROM_TO_AU
    masses = np.asarray(physics["masses_amu"]) * AMU_TO_AU
    coordinate = RPMDrateCoordinate(
        physics["fragments_zero_based"], masses, geometry,
        physics["forming_bonds_zero_based"], physics["breaking_bonds_zero_based"], 30.0,
        equivalent_paths=physics["equivalent_path_permutations"])
    return geometry, masses, coordinate


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--beads", default="1,16,64,128,256,512")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    geometry, masses, coordinate = setup()
    beta = 1 / (KB_AU * 300.0)
    rows = []
    for beads in (int(value) for value in args.beads.split(",")):
        rng = np.random.default_rng(beads)
        dynamics = Dynamics(HarmonicPotential(k=1e-3), masses, beta, beads, 0.1 * FS_TO_AU,
                            native="required", threads=1, loop_backend="native")
        q = np.repeat(geometry[None], beads, axis=0) + rng.normal(scale=2e-3, size=(beads, 6, 3))
        start = State(q, rng.normal(size=q.shape) * np.sqrt(masses[None, :, None] * beads / beta))
        cases = {"unbiased": None,
                 "umbrella_flux": CompositeBias(UmbrellaBias(coordinate, [0.9], 30.0),
                                                FluxMetricBias(coordinate, masses, beta))}
        observe = np.arange(1, args.steps + 1) % 20 == 0
        for name, bias in cases.items():
            best = np.inf
            for _ in range(args.repeats):
                state = start.copy()
                begin = time.perf_counter()
                result = dynamics.propagate_block(state, args.steps, bias=bias, cv=coordinate,
                                                  observe=observe, check_energy=True, backend="native")
                best = min(best, time.perf_counter() - begin)
                if result.completed != args.steps:
                    raise RuntimeError(result.error)
            rows.append({"beads": beads, "case": name,
                         "microseconds_per_step": best / args.steps * 1e6})
            print(json.dumps(rows[-1]), flush=True)
    if args.output:
        args.output.write_text(json.dumps({
            "schema": "pyrpmd.native-step-overhead.v1",
            "scope": "loop cost outside the PES; built-in harmonic PES, best of repeats",
            "steps_per_block": args.steps, "repeats": args.repeats,
            "platform": platform.platform(), "python": platform.python_version(),
            "native_extension": Path(_native.__file__).name, "results": rows,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
