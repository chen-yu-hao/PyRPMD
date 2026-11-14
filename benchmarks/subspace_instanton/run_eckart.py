"""RPMD defect of a symmetric Eckart barrier: ``eps(u, v)``, its steepest-descent part and the remainder.

    python run_eckart.py 6.0 2.1 results/eckart_u6.000_v2.100.json
    python run_eckart.py 2pi/0.8 1.75 results/eckart_u7.854_v1.750.json      # u of T = 0.8 T_c
    python run_eckart.py 2pi/0.55 2.1 results/eckart_u11.424_v2.100_P128.json --nbeads 128

Units hbar = m = omega_b = 1, so ``beta = u`` and the barrier height is ``v``.
``u`` is a number or ``2pi/<T/T_c>``.  The default bead number is 32 up to
``u = 6``, 48 up to ``u = 9`` and 64 beyond.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.subspace_instanton.greens import barrier_rates  # noqa: E402
from benchmarks.subspace_instanton.instanton import find_instanton, steepest_descent_factor  # noqa: E402
from benchmarks.subspace_instanton.models import eckart  # noqa: E402
from benchmarks.subspace_instanton.run_rpmd import solve, tabulated_model  # noqa: E402


def parse_u(text: str) -> float:
    return 2.0 * np.pi / float(text[4:]) if text.startswith("2pi/") else float(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("u", type=parse_u)
    parser.add_argument("v", type=float)
    parser.add_argument("output")
    parser.add_argument("--nbeads", type=int)
    args = parser.parse_args()
    u, v, out_path = args.u, args.v, args.output
    nbeads = args.nbeads or (32 if u <= 6.0 else (48 if u <= 9.0 else 64))
    potential, slope, curvature, _ = eckart(v)
    x = np.arange(-18.0, 18.005, 0.01)
    values = potential(x)
    record = solve(tabulated_model(x, values, "eckart"), u, nbeads, seed=7, walkers=16, sweeps=800)
    record["k_qm"] = barrier_rates(x, values, [u])[u]
    record["eps"] = record["k_qm"] / record["rate"]
    instanton = find_instanton(lambda b: slope(b[:, 0])[:, None], lambda b: curvature(b[:, 0])[:, None, None],
                               [0.0], [1.0], 1.0, u, nbeads=256)
    record.update(u=u, v=v, ln_eps_sd=steepest_descent_factor(u, 1.0, instanton))
    record["remainder"] = float(np.log(record["eps"]) - record["ln_eps_sd"])
    with open(out_path, "w") as handle:
        json.dump(record, handle, indent=1)
    print(f"eckart u={u:.3f} v={v:.3f}: eps = {record['eps']:.4f} +- {record['eps'] * record['relative_error']:.4f} "
          f"A = {record['ln_eps_sd']:.4f} R = {record['remainder']:+.4f} [{record['seconds']:.0f} s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
