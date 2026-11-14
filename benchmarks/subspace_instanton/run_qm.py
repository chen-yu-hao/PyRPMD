"""Deterministic half: exact quantum rates, reductions and instantons of the valley models.

    python run_qm.py results/qm.json

For every model and temperature (in units of the crossover temperature
``T_c = lambda / 2 pi`` of the full saddle) this writes the exact 2D rate, the
exact rate of every 1D reduction, and the steepest-descent factor
``ln eps_SD = ln[2 pi / (beta eta_1)]`` of the full system and of every
reduction.  No sampling is involved.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline

from greens import barrier_rates, valley_rates
from instanton import find_instanton, steepest_descent_factor
from models import family
from reductions import barrier_frequency, minimum_energy_path, reductions

RATIOS = [1.5, 1.25, 1.0, 0.8, 0.65, 0.55, 0.45]
NAMES = ("mep", "fe", "fe+sct", "bare")


def tabulated_instanton(x, potential, beta, nbeads):
    spline = CubicSpline(x, potential)
    frequency = float(np.sqrt(max(-spline(0.0, 2), 0.0)))
    if frequency == 0.0:
        return None, frequency
    gradient = lambda beads: spline(beads[:, 0], 1)[:, None]
    hessian = lambda beads: spline(beads[:, 0], 2)[:, None, None]
    return find_instanton(gradient, hessian, [0.0], [1.0], frequency, beta, nbeads=nbeads), frequency


def main() -> int:
    out = {"temperature_ratios": RATIOS, "models": {}}
    for name, model in family().items():
        started = time.time()
        saddle, unstable, stable, mode = model.saddle()
        betas = [2.0 * np.pi / (ratio * unstable) for ratio in RATIOS]
        sizes = (40, 56) if name == "M4" else (22, 30)
        coarse, _, _ = valley_rates(model, betas, size=sizes[0])
        fine, energies, probability = valley_rates(model, betas, size=sizes[1])
        refined, _, _ = valley_rates(model, betas, size=sizes[0], ds=0.01, half_length=12.0, points=1200)
        path = minimum_energy_path(model)
        record = {"saddle": saddle.tolist(), "unstable_frequency": unstable, "stable_frequency": stable,
                  "unstable_mode": mode.tolist(), "path_end_s": path["end_s"],
                  "path_max_curvature": float(path["curvature"].max()), "temperatures": {}}
        for ratio, beta in zip(RATIOS, betas):
            classical = np.exp(-beta * model.V0) / np.sqrt(2.0 * np.pi * beta)
            entry = {"beta": beta, "k_qm": fine[beta], "k_qm_over_classical": fine[beta] / classical,
                     "basis_convergence": coarse[beta] / fine[beta] - 1.0,
                     "grid_convergence": refined[beta] / coarse[beta] - 1.0, "reductions": {}}
            instanton = find_instanton(model.gradient, model.hessian, saddle, mode, unstable, beta, nbeads=128)
            if instanton is not None:
                finer = find_instanton(model.gradient, model.hessian, saddle, mode, unstable, beta, nbeads=256)
                entry["instanton"] = {"eta1": finer["eta1"], "eta1_128": instanton["eta1"],
                                      "negative_modes": finer["negative_modes"], "converged": finer["converged"],
                                      "spread": finer["spread"]}
                instanton = finer
            entry["ln_eps_sd"] = steepest_descent_factor(beta, unstable, instanton)
            reduced, mass = reductions(model, beta, path)
            for key in NAMES:
                if key == "bare" and name != "M4":
                    continue
                x, potential = reduced[key]
                rate = barrier_rates(x, potential, [beta])[beta]
                inst, frequency = tabulated_instanton(x, potential, beta, 256)
                item = {"k_qm": rate, "g_qm": float(np.log(fine[beta] / rate)), "top": float(potential.max()),
                        "barrier_frequency": barrier_frequency(x, potential),
                        "ln_eps_sd": steepest_descent_factor(beta, frequency, inst)}
                if inst is not None:
                    item["instanton"] = {"eta1": inst["eta1"], "negative_modes": inst["negative_modes"],
                                         "converged": inst["converged"]}
                item["delta_sd"] = entry["ln_eps_sd"] - item["ln_eps_sd"]
                entry["reductions"][key] = item
            entry["sct_mass_at_saddle"] = float(mass[0])
            record["temperatures"][str(ratio)] = entry
        out["models"][name] = record
        print(name, f"lambda={unstable:.4f} omega={stable:.4f} [{time.time() - started:.0f} s]", flush=True)
        for ratio in RATIOS:
            e = record["temperatures"][str(ratio)]
            line = f"  T/Tc={ratio:<5} kQM/kcl={e['k_qm_over_classical']:9.4g} conv {e['basis_convergence']:+.1e} {e['grid_convergence']:+.1e}" \
                   f"  ln eps_SD={e['ln_eps_sd']:.4f} |"
            for key, item in e["reductions"].items():
                line += f" {key}: G={item['g_qm']:+.3f} dSD={item['delta_sd']:+.3f}"
            print(line, flush=True)
    with open(sys.argv[1], "w") as handle:
        json.dump(out, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
