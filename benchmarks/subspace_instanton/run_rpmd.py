"""Stochastic half: RPMD rates of one valley model and of its reductions at one temperature.

    python run_rpmd.py M2 0.8 results/qm.json results/rpmd_M2_0.8.json
    python run_rpmd.py M2 0.8 results/qm.json results/rpmd_M2_0.8.json full      # one more run of the full model
    python run_rpmd.py M2 0.8 results/qm.json results/rpmd_M2_0.8.json mep fe    # ... of two reductions

Uses ``pyrpmd.analysis.subspace_rpmd.ring_polymer_rate``: the full system with
its analytic potential, each reduction as a tabulated 1D model with the same
number of beads.  Both are normalized per unit length at ``s = -14``, where the
barrier is below 1e-5.  The exact rates come from ``run_qm.py``.

Every invocation adds one independent run (a new seed) of the selected parts
to the output file; the reported rate is the inverse-variance mean of the
logarithms of all runs of a part.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.subspace_instanton.models import family  # noqa: E402
from benchmarks.subspace_instanton.reductions import minimum_energy_path, reductions  # noqa: E402

from pyrpmd.analysis.subspace_correction import harmonic_bead_factor
from pyrpmd.analysis.subspace_model import SubspaceModel
from pyrpmd.analysis.subspace_rpmd import ring_polymer_rate

REFERENCE = -14.0
WINDOWS = np.r_[np.linspace(REFERENCE, -8.0, 4)[:-1], np.linspace(-8.0, 0.0, 45)]
BEADS = {"M1": {1.25: 32, 0.8: 64, 0.55: 128}, "M2": {1.25: 32, 0.8: 64, 0.55: 96},
         "M3": {1.25: 32, 0.8: 64, 0.55: 128}, "M4": {1.25: 32, 0.8: 64, 0.55: 96}}
# Sampling per run.  The statistical error is that of the thermodynamic integration over
# the barrier top; the 1D reductions are cheap and get twice the walkers and sweeps.
FULL = {"walkers": 32, "sweeps": 600, "seed": 11}
REDUCED = {"walkers": 32, "sweeps": 1600, "seed": 5}
# Reductions that differ from one another for each model: without curvature the SCT
# coordinate is the arc length, and with a constant transverse frequency "fe" is "mep".
REDUCTIONS = {"M1": ("mep", "fe"), "M2": ("mep", "fe", "fe+sct"), "M3": ("mep", "fe", "fe+sct"),
              "M4": ("mep", "fe", "fe+sct", "bare")}


def solve(model, beta, nbeads, *, seed, md_time=None, walkers=32, sweeps=600, trajectories=2048):
    started = time.time()
    result = ring_polymer_rate(model, beta, nbeads, windows=WINDOWS, walkers=walkers, burn_in=200, sweeps=sweeps,
                               trajectories=trajectories, md_time=md_time, seed=seed)
    return {"rate": result.rate, "relative_error": result.rate_standard_error / result.rate,
            "qtst": result.qtst_rate, "qtst_log_error": result.qtst_log_standard_error,
            "kappa": result.transmission,
            "kappa_relative_error": result.transmission_standard_error / result.transmission,
            "nbeads": int(nbeads), "seed": int(seed), "walkers": int(walkers), "sweeps": int(sweeps),
            "seconds": time.time() - started}


def combine(runs):
    """Independent runs of one rate: inverse-variance mean of the logarithms."""
    weights = np.array([run["relative_error"] ** -2 for run in runs])
    mean = lambda values: float(np.sum(weights * np.asarray(values)) / weights.sum())
    return {"rate": float(np.exp(mean(np.log([run["rate"] for run in runs])))),
            "relative_error": float(weights.sum() ** -0.5),
            "qtst": float(np.exp(mean(np.log([run["qtst"] for run in runs])))),
            "kappa": mean([run["kappa"] for run in runs]), "nbeads": runs[0]["nbeads"],
            "seconds": float(sum(run["seconds"] for run in runs)), "runs": runs}


def earlier_runs(record, nbeads, settings):
    """Runs already in the output file that the new one may be averaged with."""
    runs = [] if record is None else record.get("runs", [])
    return [run for run in runs if (run["nbeads"], run["walkers"], run["sweeps"])
            == (nbeads, settings["walkers"], settings["sweeps"])]


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
    only = sys.argv[5:]                                  # optional subset: "full" and reduction names
    qm = json.load(open(qm_path))["models"][name]
    entry = qm["temperatures"][str(ratio)]
    beta = entry["beta"]
    valley = family()[name]
    nbeads = BEADS[name][ratio]
    try:
        report = json.load(open(out_path))
    except (OSError, ValueError):
        report = {"model": name, "temperature_ratio": ratio, "beta": beta, "reductions": {}}

    def save():
        with open(out_path, "w") as handle:
            json.dump(report, handle, indent=1)

    def add_run(record, model, settings, **options):
        runs = earlier_runs(record, nbeads, settings)
        runs.append(solve(model, beta, nbeads, seed=settings["seed"] + 1000 * len(runs), walkers=settings["walkers"],
                          sweeps=settings["sweeps"], **options))
        return combine(runs)

    if not only or "full" in only:
        full = add_run(report.get("full"), full_model(valley), FULL, md_time=50.0 if name == "M4" else None)
        # Finite-bead error of the transverse mode (harmonic, TST level): exact for harmonic modes.
        full["bead_factor"] = harmonic_bead_factor([qm["stable_frequency"]], [valley.omega_inf], beta, nbeads)["factor"]
        full["rate_bead_corrected"] = full["rate"] / full["bead_factor"]
        full["k_qm"] = entry["k_qm"]
        full["eps"] = entry["k_qm"] / full["rate_bead_corrected"]
        report["full"] = full
        print(f"{name} T/Tc={ratio}: full eps = {full['eps']:.4f} +- {full['eps'] * full['relative_error']:.4f} "
              f"kappa = {full['kappa']:.3f} B_P = {full['bead_factor']:.4f} runs = {len(full['runs'])} "
              f"[{full['seconds']:.0f} s]", flush=True)
        save()
    reduced, _ = reductions(valley, beta, minimum_energy_path(valley))
    for key in REDUCTIONS[name]:
        if only and key not in only:
            continue
        x, potential = reduced[key]
        record = add_run(report["reductions"].get(key), tabulated_model(x, potential, key), REDUCED)
        record["k_qm"] = entry["reductions"][key]["k_qm"]
        record["eps"] = record["k_qm"] / record["rate"]
        report["reductions"][key] = record
        print(f"   {key:7s}: eps_S = {record['eps']:.4f} +- {record['eps'] * record['relative_error']:.4f} "
              f"kappa = {record['kappa']:.3f} runs = {len(record['runs'])} [{record['seconds']:.0f} s]", flush=True)
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
