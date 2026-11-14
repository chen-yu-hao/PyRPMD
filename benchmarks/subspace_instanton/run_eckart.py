"""RPMD defect of a symmetric Eckart barrier: ``eps(u, v)``, its steepest-descent part and the remainder.

    python run_eckart.py 7.854 1.75 results/eckart_7.854_1.75.json

Units hbar = m = omega_b = 1, so ``beta = u`` and the barrier height is ``v``.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from greens import barrier_rates
from instanton import find_instanton, steepest_descent_factor
from models import eckart
from run_rpmd import solve, tabulated_model


def main() -> int:
    u, v, out_path = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    nbeads = 32 if u <= 6.0 else (48 if u <= 9.0 else 64)
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
