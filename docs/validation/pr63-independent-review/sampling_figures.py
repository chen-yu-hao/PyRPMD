"""Rebuild sampling figure and tables from the independent raw results."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

OUT = Path(__file__).resolve().parent
reports = [json.loads((OUT / f"sampling_results_dt{dt}.json").read_text()) for dt in (0.04, 0.02)]
first = reports[0]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 140})
fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
names = ["Dihedral", "CoordinationNumber", "RMSD", "PathCV"]
errors = [max(row["derivative_max_abs"] for row in first["cv"] if row["cv"] == name) for name in names]
ax = axes[0, 0]
ax.bar(np.arange(4), errors, color="#3776ab")
ax.set_xticks(np.arange(4), ["Dihedral", "Coordination", "RMSD", "Path (s,z)"])
ax.set_yscale("log")
ax.set_ylabel("Max absolute derivative error")
ax.set_title("A. Full-ring CVs vs fourth-order finite differences")
ax.grid(axis="y", alpha=0.2)

ax = axes[0, 1]
rows = first["opes"]
x = np.arange(len(rows))
ax.bar(x - 0.18, [r["max_energy_error"] for r in rows], 0.36, label="Bias energy", color="#3776ab")
ax.bar(x + 0.18, [r["max_gradient_error"] for r in rows], 0.36, label="Bias gradient", color="#dc7f37")
ax.set_xticks(x, ["METAD\nfixed", "METAD\nadaptive", "Explore\nfixed", "Explore\nadaptive"])
ax.set_yscale("log")
ax.set_ylabel("Max absolute error (atomic units)")
ax.set_title("B. OPES vs independent Gaussian-image sums")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)

retis_rows = []
ax = axes[1, 0]
exact = np.array(first["retis"]["continuum_reference_probabilities"])
ax.scatter(np.arange(3), exact, marker="_", s=350, c="black", linewidths=2, label="Continuum flux reference")
for offset, report, color in zip((-0.09, 0.09), reports, ("#3776ab", "#dc7f37")):
    run = report["retis"]
    probabilities = np.array([seed["probabilities"] for seed in run["seeds"]])
    means = probabilities.mean(axis=0)
    intervals = t.ppf(0.975, len(probabilities) - 1) * probabilities.std(axis=0, ddof=1) / np.sqrt(len(probabilities))
    dt = run["seeds"][0]["dt"]
    ax.errorbar(np.arange(3) + offset, means, intervals, fmt="o", color=color, capsize=4, label=f"dt={dt}")
    for i, (value, ci, truth) in enumerate(zip(means, intervals, exact)):
        retis_rows.append(dict(dt=dt, quantity=f"interface_{i}", estimate=value, seed_t_95_halfwidth=ci, continuum_reference=truth))
ax.set_xticks(np.arange(3), ["-0.8 → -0.5", "-0.5 → -0.2", "-0.2 → B"])
ax.set_ylabel("Conditional crossing probability")
ax.set_title("C. Actual quartic-barrier RETIS chains")
ax.set_ylim(0.4, 1.0)
ax.grid(axis="y", alpha=0.2)
ax.legend(frameon=False, fontsize=9)

ax = axes[1, 1]
truth = first["retis"]["continuum_reference_product"]
ax.axhline(truth, color="black", lw=1.5, ls="--", label=f"Continuum reference {truth:.6f}")
for i, (report, color) in enumerate(zip(reports, ("#3776ab", "#dc7f37"))):
    run = report["retis"]
    values = np.array([seed["product"] for seed in run["seeds"]])
    ax.scatter(i + np.linspace(-0.12, 0.12, len(values)), values, color=color, alpha=0.6, s=25)
    ax.errorbar(i, run["mean_product"], run["seed_t_95_halfwidth"], fmt="D", color=color, capsize=5)
    retis_rows.append(dict(dt=run["seeds"][0]["dt"], quantity="product", estimate=run["mean_product"],
                           seed_t_95_halfwidth=run["seed_t_95_halfwidth"], continuum_reference=truth))
ax.set_xticks([0, 1], ["dt=0.04", "dt=0.02"])
ax.set_ylabel("Product of conditional crossing probabilities")
ax.set_title("D. Four seeds; intervals reflect seed variability")
ax.grid(axis="y", alpha=0.2)
ax.legend(frameon=False, fontsize=9)
fig.suptitle("PR #63 independent enhanced-sampling review\nEquation checks and model-system evidence; no molecular-rate convergence claim", fontsize=14)
fig.savefig(OUT / "sampling_validation.png")
fig.savefig(OUT / "sampling_validation.pdf")
plt.close(fig)

with (OUT / "sampling_retis_summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(retis_rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(retis_rows)
coarse = np.array([s["product"] for s in reports[0]["retis"]["seeds"]])
fine = np.array([s["product"] for s in reports[1]["retis"]["seeds"]])
delta = fine - coarse
summary = dict(paired_fine_minus_coarse_mean=float(delta.mean()),
               paired_seed_t_95_halfwidth=float(t.ppf(.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))),
               seed_count=len(delta), note="Pointwise intervals from four seeds; no joint/physical convergence guarantee")
(OUT / "sampling_timestep_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
