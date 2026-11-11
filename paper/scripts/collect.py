"""Collect compact, plot-ready records from completed PyRPMD runs.

Usage: python paper/scripts/collect.py RUNS_DIRECTORY SCRATCH_DATA_DIRECTORY

RUNS_DIRECTORY holds the run directories produced by paper/scripts/run_all.sh;
SCRATCH_DATA_DIRECTORY holds integrators.json, throughput.json and
quantum_statistics.json written by the measurement scripts.  Everything the
manuscript quotes is written to paper/data/.
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples" / "tutorials"))
from reference_checks import BETA, MASS, compare, harmonic_reference  # noqa: E402

runs, scratch = Path(sys.argv[1]), Path(sys.argv[2])
out = REPO / "paper" / "data"
out.mkdir(exist_ok=True)


def dump(name, obj):
    (out / name).write_text(json.dumps(obj, indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else float(v)) + "\n")


# 1. Analytic tutorials: the repository's own comparison records.
tutorials = {name: compare(name, runs / name) for name in
             ("harmonic_pimd", "periodic_site_rate", "double_well_metad",
              "free_diffusion", "harmonic_spectroscopy")}
dump("tutorials.json", tutorials)

# 2. Bead-number scan of the harmonic oscillator.
k = 0.02
omega = np.sqrt(k / MASS)
rows = []
for beads in (1, 2, 4, 8, 16, 32, 64):
    est = json.loads((runs / f"bead_p{beads}/results/summary.json").read_text())["estimators"]
    ref = harmonic_reference(beads=beads)
    rows.append({"beads": beads,
                 "kinetic_cv": est["kinetic_centroid_virial"]["mean"],
                 "kinetic_cv_sem": est["kinetic_centroid_virial"]["standard_error_between_replicas"],
                 "potential": est["potential_energy"]["mean"],
                 "potential_sem": est["potential_energy"]["standard_error_between_replicas"],
                 "finite_p_reference": ref["kinetic_centroid_virial"]})
dump("beadscan.json", {"rows": rows, "beta_hbar_omega": BETA * omega,
                       "exact_quantum_kinetic": 0.75 * omega / np.tanh(BETA * omega / 2),
                       "classical_kinetic": 1.5 / BETA})

# 3. Well-tempered metadynamics free energy against the bin-averaged Boltzmann result.
from scipy.integrate import quad  # noqa: E402
with np.load(runs / "double_well_metad/fes.npz", allow_pickle=False) as saved:
    edges = saved["edges_0"]
    grid = 0.5 * (edges[:-1] + edges[1:])
    curves = saved["free_energy_per_replica"]
    mask = saved["mask_per_replica"]
exact = np.array([-np.log(quad(lambda x: np.exp(-BETA * 0.004 * (x * x - 1) ** 2), a, b)[0] / (b - a)) / BETA
                  for a, b in zip(edges[:-1], edges[1:])])
aligned = []
for curve, hidden in zip(curves, mask):
    curve = np.where(hidden, np.nan, curve)
    wells = 0.5 * (np.interp(-1.0, grid, curve) + np.interp(1.0, grid, curve))
    aligned.append(curve - wells)
aligned = np.array(aligned)
exact -= 0.5 * (np.interp(-1.0, grid, exact) + np.interp(1.0, grid, exact))
dump("metad_fes.json", {"x_bohr": grid, "exact_kbt": exact * BETA,
                        "mean_kbt": np.nanmean(aligned, axis=0) * BETA,
                        "sem_kbt": np.nanstd(aligned, axis=0, ddof=1) / 2 * BETA,
                        "replicas": len(aligned)})

# 4. TRPMD spectrum near the harmonic line.
with np.load(runs / "harmonic_spectroscopy/results/ir-spectrum.npz", allow_pickle=False) as saved:
    wn = saved["wavenumber_cm1"]
    intensity = (saved["per_replica"] * saved["angular_frequency_au"][None] ** 2).mean(axis=0)
window = (wn > 900) & (wn < 1400)
dump("spectrum.json", {"wavenumber_cm1": wn[window], "intensity": intensity[window] / intensity[window].max()})

# 5. Reactive exchange model: dRPMD against Bennett-Chandler.
drpmd = []
for suffix in ("", "_s1", "_s2", "_s3", "_s4"):
    s = json.loads((runs / f"exchange_drpmd{suffix}/results/summary.json").read_text())
    assert s["all_trajectories_resolved"]
    drpmd.append({"reactive": s["counts"]["product"], "trajectories": sum(s["counts"].values()),
                  "rate_cm3_s": s["total_reaction"]["rate_cm3_molecule_s"]})
bc = {}
for label, names in (("ui", ["exchange_rate_ui"] + [f"exchange_rate_ui_s{i}" for i in (1, 2, 3)]),
                     ("mbar", ["exchange_rate_mbar"] + [f"exchange_rate_mbar_s{i}" for i in (1, 2, 3)])):
    rows = []
    for name in names:
        if not (runs / name / "results/summary.json").exists():
            continue
        s = json.loads((runs / name / "results/summary.json").read_text())
        p = json.loads((runs / name / "rate_pmf.json").read_text())
        rows.append({"rate_cm3_s": s["rate_cm3_s"], "rate_standard_error_cm3_s": s["rate_standard_error_cm3_s"],
                     "kappa": s["kappa"], "kappa_standard_error": s["kappa_standard_error"],
                     "plateau_detected": s["plateau_detected"], "k_qtst_cm3_s": p["k_qtst_cm3_s"],
                     "barrier_hartree": p["free_energy_barrier_hartree"],
                     "barrier_standard_error_hartree": p["barrier_standard_error_hartree"]})
    bc[label] = rows


def mean_sem(values):
    return statistics.mean(values), (statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else None)


summary = {"drpmd": dict(zip(("rate_cm3_s", "sem_cm3_s"), mean_sem([r["rate_cm3_s"] for r in drpmd])),
                         reactive=sum(r["reactive"] for r in drpmd),
                         trajectories=sum(r["trajectories"] for r in drpmd)),
           "bc_ui": dict(zip(("rate_cm3_s", "sem_cm3_s"), mean_sem([r["rate_cm3_s"] for r in bc["ui"]])),
                         kappa=statistics.mean(r["kappa"] for r in bc["ui"]),
                         k_qtst_cm3_s=statistics.mean(r["k_qtst_cm3_s"] for r in bc["ui"])),
           "bc_mbar": dict(zip(("rate_cm3_s", "sem_cm3_s"), mean_sem([r["rate_cm3_s"] for r in bc["mbar"]])),
                           kappa=statistics.mean(r["kappa"] for r in bc["mbar"]),
                           k_qtst_cm3_s=statistics.mean(r["k_qtst_cm3_s"] for r in bc["mbar"]))}
for key in ("bc_ui", "bc_mbar"):
    summary[key]["seeds"] = len(bc[key[3:]])
    summary[key]["k_qtst_sem_cm3_s"] = mean_sem([r["k_qtst_cm3_s"] for r in bc[key[3:]]])[1]
    summary[key]["kappa_sem"] = mean_sem([r["kappa"] for r in bc[key[3:]]])[1]
if summary["bc_mbar"]["sem_cm3_s"] is None:  # single seed: joint bootstrap error
    summary["bc_mbar"]["sem_cm3_s"] = bc["mbar"][0]["rate_standard_error_cm3_s"]
pmf = np.loadtxt(runs / "exchange_rate_ui/rate_pmf.csv", delimiter=",", skiprows=1)
rec = np.loadtxt(runs / "exchange_rate_ui/recrossing.csv", delimiter=",", skiprows=1)
dump("exchange.json", {"drpmd_seeds": drpmd, "bennett_chandler": bc, "summary": summary,
                       "kbt_hartree": 1 / (BETA * 300 / 1500),
                       "pmf": {"xi": pmf[::5, 0], "free_energy_hartree": pmf[::5, 1]},
                       "recrossing": {"time_fs": rec[:, 0] * 0.024188843265857, "kappa": rec[:, 1],
                                      "standard_error": rec[:, 2]}})

# 6. Records written directly by the measurement scripts.
for name in ("integrators.json", "throughput.json", "parallel.json"):
    if (scratch / name).exists():
        dump(name, json.loads((scratch / name).read_text()))
qs = json.loads((scratch / "quantum_statistics.json").read_text())
a = qs["phases"]["A"]
dump("trace_convergence.json", {"beta_hbar_omega": qs["beta_times_omega"],
                                "rows": qs["trace_convergence_per_Cartesian_mode"],
                                "ti_energy_hartree": a["heat_and_energy"]["internal_energy_hartree"],
                                "ti_energy_sem_hartree": a["heat_and_energy"]["internal_energy_standard_error_hartree"],
                                "ti_beads": a["heat_and_energy"]["beads"]})
print("wrote", sorted(p.name for p in out.iterdir()))
