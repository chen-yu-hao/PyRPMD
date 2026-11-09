"""Subspace correction on a 2D Eckart barrier with a coupled vibration.

Exact 2D quantum rate versus  k_RPMD(2D) * C_S  with a one-dimensional
vibrationally adiabatic subspace projected from the 2D potential.  The
vibration softens at the barrier (zero-point energy change) and its
equilibrium shifts (path curvature), so S is a genuine reduction.

    PYTHONPATH=src python examples/subspace_correction/eckart_bath_2d.py --out result.json
"""
import argparse
import json
import time

import numpy as np

from pyrpmd.analysis.subspace_correction import apply_subspace_correction, subspace_correction
from pyrpmd.analysis.subspace_model import model_from_function, project_linear_subspace
from pyrpmd.analysis.subspace_quantum import exact_quantum_rate
from pyrpmd.analysis.subspace_rpmd import ring_polymer_rate

KELVIN = 3.166811563e-6
MASS = 1060.0
V0, A = 0.425 / 27.211386, 0.66
OMEGA0, SOFTEN, SHIFT = 0.010, 0.3, 0.2


def _parts(s):
    sech2 = 1.0 / np.cosh(s / A) ** 2
    return sech2, -2.0 / A * np.tanh(s / A) * sech2


def potential(points):
    s, y = points[..., 0], points[..., 1]
    sech2, _ = _parts(s)
    omega = OMEGA0 * (1.0 - SOFTEN * sech2)
    return V0 * sech2 + 0.5 * MASS * omega**2 * (y - SHIFT * sech2) ** 2


def gradient(points):
    s, y = points[..., 0], points[..., 1]
    sech2, dsech2 = _parts(s)
    omega = OMEGA0 * (1.0 - SOFTEN * sech2)
    domega = -OMEGA0 * SOFTEN * dsech2
    dy = y - SHIFT * sech2
    out = np.empty_like(points)
    out[..., 0] = V0 * dsech2 + MASS * omega * domega * dy**2 - MASS * omega**2 * dy * SHIFT * dsech2
    out[..., 1] = MASS * omega**2 * dy
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[300.0, 200.0])
    parser.add_argument("--beads", type=int, nargs="+", default=[16, 24])
    parser.add_argument("--sweeps", type=int, default=800)
    parser.add_argument("--trajectories", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out")
    args = parser.parse_args()
    s_axis = np.linspace(-8.0, 8.0, 201)
    y_axis = np.linspace(-1.4, 1.6, 31)
    full = model_from_function(potential, [s_axis, y_axis], gradient=gradient, masses=[MASS, MASS],
                               dividing_surface=0.0)
    # The 1D subspace uses mass-weighted s; the complement y is relaxed and
    # its zero-point energy added (vibrationally adiabatic model).
    sub_axis = np.sqrt(MASS) * s_axis
    sub = project_linear_subspace(lambda x: float(potential(x)), gradient, [0.0, 0.0], [MASS, MASS],
                                  [1.0, 0.0], [sub_axis], environment="adiabatic", dividing_surface=0.0)
    rows = []
    for temperature, beads in zip(args.temperatures, args.beads):
        beta = 1.0 / (temperature * KELVIN)
        t0 = time.time()
        exact = exact_quantum_rate(full, beta)
        t1 = time.time()
        rpmd = ring_polymer_rate(full, beta, beads, sweeps=args.sweeps, trajectories=args.trajectories,
                                 seed=args.seed)
        t2 = time.time()
        correction = subspace_correction(sub, beta, beads,
                                         rpmd_options={"sweeps": args.sweeps, "trajectories": args.trajectories,
                                                       "seed": args.seed + 1})
        t3 = time.time()
        # C_S is a ratio of two rates of the same 1D model, so it is
        # dimensionless and multiplies the 2D rate directly.
        applied = apply_subspace_correction(rpmd.rate, correction, standard_error=rpmd.rate_standard_error,
                                            beta=beta)
        row = {
            "temperature_K": temperature, "beads": beads,
            "exact_2d": exact.rate, "rpmd_2d": rpmd.rate, "rpmd_2d_relative_error": rpmd.log_rate_standard_error,
            "C_S_1d": correction.factor, "C_S_log_error": correction.log_standard_error,
            "corrected": applied["corrected_rate"], "corrected_relative_error": applied["corrected_relative_standard_error"],
            "rpmd_over_exact": rpmd.rate / exact.rate, "corrected_over_exact": applied["corrected_rate"] / exact.rate,
            "required_C": exact.rate / rpmd.rate,
            "exact_method": exact.method, "exact_extrapolation_spread": exact.diagnostics.get("relative_extrapolation_spread"),
            "seconds": {"exact_2d": t1 - t0, "rpmd_2d": t2 - t1, "correction_1d": t3 - t2},
        }
        rows.append(row)
        print(f"T={temperature:6.1f} K P={beads:3d}  RPMD/exact={row['rpmd_over_exact']:.3f}  "
              f"C_S={row['C_S_1d']:.3f}+-{row['C_S_log_error']:.3f}  required={row['required_C']:.3f}  "
              f"corrected/exact={row['corrected_over_exact']:.3f}+-{row['corrected_relative_error']:.3f}  "
              f"({t1 - t0:.0f}s/{t2 - t1:.0f}s/{t3 - t2:.0f}s)", flush=True)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"model": {"mass": MASS, "V0": V0, "a": A, "omega0": OMEGA0, "soften": SOFTEN, "shift": SHIFT,
                                 "s_axis": [s_axis[0], s_axis[-1], len(s_axis)],
                                 "y_axis": [y_axis[0], y_axis[-1], len(y_axis)]},
                       "subspace": sub.describe(), "rows": rows}, handle, indent=2)


if __name__ == "__main__":
    main()
