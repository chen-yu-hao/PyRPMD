"""Stochastic half: RPMD rates of one valley model and of its reductions at one temperature.

    python run_rpmd.py M2 0.8 results/qm.json results/rpmd_M2_0.8.json

Uses ``pyrpmd.analysis.subspace_rpmd.ring_polymer_rate``: the full system with
its analytic potential, each reduction as a tabulated 1D model.  Both are
normalized per unit length at ``s = -14``, where the barrier is below 1e-5.
The exact rates come from ``run_qm.py``.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from models import eckart, family
from reductions import minimum_energy_path, reductions

from pyrpmd.analysis.subspace_correction import harmonic_bead_factor
from pyrpmd.analysis.subspace_model import SubspaceModel
from pyrpmd.analysis.subspace_rpmd import ring_polymer_rate

REFERENCE = -14.0
WINDOWS = np.r_[np.linspace(REFERENCE, -8.0, 4)[:-1], np.linspace(-8.0, 0.0, 45)]
BEADS_FULL = {"M1": {1.25: 32, 0.8: 64, 0.55: 128}, "M2": {1.25: 32, 0.8: 64, 0.55: 96},
              "M3": {1.25: 32, 0.8: 64, 0.55: 128}, "M4": {1.25: 32, 0.8: 64, 0.55: 96}}
BEADS_REDUCED = {1.25: 32, 0.8: 48, 0.55: 64}


def solve(model, beta, nbeads, *, seed, md_time=None, sweeps=1200, trajectories=2048):
    started = time.time()
    result = ring_polymer_rate(model, beta, nbeads, windows=WINDOWS, walkers=32, burn_in=300, sweeps=sweeps,
                               trajectories=trajectories, md_time=md_time, seed=seed)
    return {"rate": result.rate, "relative_error": result.rate_standard_error / result.rate,
            "qtst": result.qtst_rate, "kappa": result.transmission, "nbeads": int(nbeads),
            "seconds": time.time() - started}


def full_model(valley):
    s = np.arange(-17.0, 17.0 + 1e-9, 0.25)
    q = np.arange(-7.0, 7.0 + 1e-9, 0.25)
    grid = np.stack(np.meshgrid(s, q, indexing="ij"), axis=-1)
    return SubspaceModel((s, q), valley.energy(grid), (1.0, 1.0), 0.0, "asymptotic", REFERENCE, {"model": valley.name},
                         function=valley.energy, gradient=valley.gradient)


def tabulated_model(x, potential, name):
    return SubspaceModel((x,), potential, (1.0,), 0.0, "asymptotic", REFERENCE, {"model": name})


def main() -> int:
    name, ratio, qm_path, out_path = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
    only = sys.argv[5:]                                  # optional subset: "full", reduction names, "eckart"
    qm = json.load(open(qm_path))["models"][name]
    entry = qm["temperatures"][str(ratio)]
    beta = entry["beta"]
    valley = family()[name]
    try:
        report = json.load(open(out_path))
    except (OSError, ValueError):
        report = {"model": name, "temperature_ratio": ratio, "beta": beta, "reductions": {}}

    def save():
        with open(out_path, "w") as handle:
            json.dump(report, handle, indent=1)

    if not only or "full" in only:
        nbeads = BEADS_FULL[name][ratio]
        full = solve(full_model(valley), beta, nbeads, seed=11, md_time=50.0 if name == "M4" else None)
        # Finite-bead error of the transverse mode (harmonic, TST level): exact for harmonic modes.
        bead = harmonic_bead_factor([qm["stable_frequency"]], [valley.omega_inf], beta, nbeads)
        full["bead_factor"] = float(bead["factor"] if isinstance(bead, dict) else bead)
        full["rate_bead_corrected"] = full["rate"] / full["bead_factor"]
        full["k_qm"] = entry["k_qm"]
        full["eps"] = entry["k_qm"] / full["rate_bead_corrected"]
        report["full"] = full
        print(f"{name} T/Tc={ratio}: full eps = {full['eps']:.4f} +- {full['eps'] * full['relative_error']:.4f} "
              f"kappa = {full['kappa']:.3f} B_P = {full['bead_factor']:.4f} [{full['seconds']:.0f} s]", flush=True)
        save()
    reduced, _ = reductions(valley, beta, minimum_energy_path(valley))
    for key, item in entry["reductions"].items():
        if only and key not in only:
            continue
        x, potential = reduced[key]
        record = solve(tabulated_model(x, potential, key), beta, BEADS_REDUCED[ratio], seed=5)
        record["k_qm"] = item["k_qm"]
        record["eps"] = item["k_qm"] / record["rate"]
        report["reductions"][key] = record
        print(f"   {key:7s}: eps_S = {record['eps']:.4f} +- {record['eps'] * record['relative_error']:.4f} "
              f"kappa = {record['kappa']:.3f} [{record['seconds']:.0f} s]", flush=True)
        save()
    if not only or "eckart" in only:
        # Symmetric Eckart barrier with the crossover temperature of the full saddle and the
        # vibrationally adiabatic barrier height: the remainder of the path-free estimate.
        from greens import barrier_rates

        frequency = qm["unstable_frequency"]
        height = valley.V0 + 0.5 * (qm["stable_frequency"] - valley.omega_inf)
        v = height / frequency
        potential, _, _, a = eckart(v, frequency)
        x = np.arange(-18.0, 18.005, 0.01)
        values = potential(x)
        record = solve(tabulated_model(x, values, "eckart"), beta, BEADS_REDUCED[ratio], seed=7)
        record["k_qm"] = barrier_rates(x, values, [beta])[beta]
        record["eps"] = record["k_qm"] / record["rate"]
        record["v"] = v
        record["u"] = beta * frequency
        report["eckart"] = record
        print(f"   eckart(u={record['u']:.2f}, v={v:.3f}): eps = {record['eps']:.4f} +- "
              f"{record['eps'] * record['relative_error']:.4f} [{record['seconds']:.0f} s]", flush=True)
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
