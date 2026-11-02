"""Short PIGLET-NPT FCC solid mechanics demonstration (not converged data).

Run from a source checkout after installing PyRPMD:
    python examples/path_integral/piglet_npt.py --output /tmp/piglet-solid

The official i-PI P=4,750 K matrices are normalized and temperature-scaled to
the reduced-unit solid beta=5. This is not a new, generically transferable fit.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"benchmarks"/"condensed_phase"))
from models import LennardJonesSolid, build_lj, fcc
from validate import run_trajectory, save_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    if args.steps < 20:
        parser.error("--steps must be at least 20")
    args.output.mkdir(parents=True, exist_ok=True)
    q, cell = fcc(cells=2, density=1.)
    library = build_lj(args.output/"build")
    physical = LennardJonesSolid(library, len(q), cell)
    rows = run_trajectory(physical, q, cell, np.full(len(q), 100.),
                          beta=5., beads=4, dt=.04, seed=20261102,
                          burn=200, steps=args.steps, stride=20,
                          method="piglet", pressure=1., tau=5.)
    save_csv(args.output/"piglet_npt.csv", rows)
    print(json.dumps({"scope":"short mechanics demonstration; establish convergence before physical inference",
                      "samples":len(rows), "mean_volume":np.mean([r["volume"] for r in rows]),
                      "mean_pressure":np.mean([r["pressure_au"] for r in rows]),
                      "target_pressure":1., "heat_capacity":"not inferred from PIGLET fluctuations"}, indent=2))


if __name__ == "__main__":
    main()
