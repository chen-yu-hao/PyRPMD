"""Elasticity of the RPMD error, eps = d ln C / d ln k_QM, for 1D Eckart barriers.

The transfer theory (docs/subspace-transfer-theory.md) predicts that the
transfer error of a subspace correction is approximately eps * G, where G is
the change of ln k between H_S and H.  eps is measured here by scaling the
barrier height of the 1D Eckart model of benchmarks/subspace_theory/transfer.py
(mass 1060, a = 0.66 bohr, symmetric or 0.2 eV exothermic).

    python benchmarks/subspace_theory/elasticity.py --output elasticity.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from pyrpmd.analysis.subspace_model import model_from_function
from pyrpmd.analysis.subspace_quantum import exact_quantum_rate
from pyrpmd.analysis.subspace_rpmd import ring_polymer_rate

KELVIN = 3.166811563e-6
MASS = 1060.0
V0, A = 0.425 / 27.211386, 0.66
ASYMMETRY = 0.2 / 27.211386


def eckart(scale, drop):
    def potential(points):
        s = np.asarray(points, dtype=float)[..., 0] / np.sqrt(MASS)
        return scale * V0 / np.cosh(s / A) ** 2 - 0.5 * drop * (1.0 + np.tanh(s / A))
    return potential


def point(args):
    scale, barrier, temperature, nbeads, seed = args
    drop = ASYMMETRY if barrier == "asymmetric" else 0.0
    beta = 1.0 / (KELVIN * temperature)
    axis = np.sqrt(MASS) * np.linspace(-16.0, 16.0, 1601)
    s = axis / np.sqrt(MASS)
    potential = eckart(scale, drop)
    top = float(axis[np.argmax(potential(axis[:, None]))])
    model = model_from_function(potential, [axis], dividing_surface=top, reactant_reference=-4.0 * np.sqrt(MASS))
    quantum = exact_quantum_rate(model, beta, method="sylvester", energy_floor=-1.0 / beta, cap_width=0.4 * 16 * np.sqrt(MASS))
    rpmd = ring_polymer_rate(model, beta, nbeads, walkers=64, sweeps=1000, trajectories=2048, seed=seed)
    return {"scale": scale, "barrier": barrier, "temperature": temperature, "nbeads": nbeads,
            "k_qm": quantum.rate, "k_rpmd": rpmd.rate, "log_standard_error": rpmd.log_rate_standard_error,
            "C": quantum.rate / rpmd.rate, "kappa": rpmd.transmission, "grid_max": float(s[-1])}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scales", type=float, nargs="+", default=[0.8, 1.0, 1.25])
    p.add_argument("--temperatures", type=float, nargs="+", default=[600.0, 300.0, 200.0])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    tasks = []
    for barrier in ("symmetric", "asymmetric"):
        for t in args.temperatures:
            nbeads = int(max(16, 8 * np.ceil(2 * 0.01 / (KELVIN * t) / 8)))
            for i, scale in enumerate(args.scales):
                tasks.append((scale, barrier, t, nbeads, 1000 + i))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(point, tasks))
    summary = []
    for barrier in ("symmetric", "asymmetric"):
        for t in args.temperatures:
            group = [r for r in rows if r["barrier"] == barrier and r["temperature"] == t]
            x = np.log([r["k_qm"] for r in group])
            y = np.log([r["C"] for r in group])
            w = 1.0 / np.array([r["log_standard_error"] for r in group]) ** 2
            slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(w))
            # Standard error of the weighted slope.
            xm = np.sum(w * x) / np.sum(w)
            error = float(np.sqrt(1.0 / np.sum(w * (x - xm) ** 2)))
            summary.append({"barrier": barrier, "temperature": t, "elasticity": float(slope),
                            "elasticity_standard_error": error,
                            "C": [r["C"] for r in group], "ln_k_span": float(np.ptp(x))})
            print(f"ELASTICITY {barrier} {t:g} K: eps = {slope:+.3f} +- {error:.3f}  C = "
                  + ", ".join(f"{r['C']:.3f}" for r in group))
    with open(args.output, "w") as handle:
        json.dump({"points": rows, "summary": summary}, handle, indent=1)


if __name__ == "__main__":
    main()
