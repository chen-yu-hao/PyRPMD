"""Plot a complete quantum H3 cohort and retain the earlier failed estimates.

The gray paper-precision bands are descriptive, not confidence intervals.
This renderer never resamples, drops a seed, or changes an acceptance gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(analysis, historical, output):
    analysis, historical, output = map(Path, (analysis, historical, output))
    report_path = analysis / "quantum-rate-comparison.json"
    curve_path = analysis / "kappa-curve.csv"
    report = json.loads(report_path.read_text())
    if (not report["criteria"]["complete_registered_cohort"]
            or report["completed_seeds"] != report["expected_seeds"]):
        raise ValueError("A complete registered cohort is required; partial diagnostics cannot produce this final figure")
    prior_paths = [historical / name / "summary.json" for name in ("long-a", "long-b")]
    previous = [json.loads(path.read_text()) for path in prior_paths]
    if any(row.get("status") != "pilot_requires_convergence" for row in previous):
        raise ValueError("The historical comparison must retain its original unconverged classification")
    curve = np.genfromtxt(curve_path, delimiter=",", names=True)
    required = ("time_fs", "kappa_seed_mean", "pointwise_CI95_low", "pointwise_CI95_high")
    if tuple(curve.dtype.names) != required or not all(np.isfinite(curve[name]).all() for name in required):
        raise ValueError("Unexpected/nonfinite transmission curve")
    if not np.array_equal(curve["time_fs"], np.arange(1, report["horizon_fs"] + 1)):
        raise ValueError("Transmission curve time grid differs from the analyzed horizon")
    statistics, paper = report["statistics"], report["paper"]
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "legend.fontsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    blue, orange, gray = "#1769aa", "#b84b31", "#666666"
    figure = plt.figure(figsize=(12.8, 8.3))
    grid = figure.add_gridspec(2, 3, height_ratios=(1, 1.15),
                              left=.20, right=.97, bottom=.22, top=.88,
                              wspace=.35, hspace=.48)
    quantities = [
        ("A  Transition-state rate", "qtst_cm3_molecule_s",
         statistics["qtst_geometric_mean_cm3_molecule_s"], statistics["qtst_CI95_cm3_molecule_s"],
         [row["pmf"]["k_qtst_cm3_s"] for row in previous]),
        ("B  Transmission coefficient", "kappa",
         statistics["kappa_arithmetic_mean"], statistics["kappa_CI95"],
         [row["kappa"] for row in previous]),
        ("C  Complete RPMD rate", "rate_cm3_molecule_s",
         statistics["rate_geometric_mean_cm3_molecule_s"], statistics["rate_CI95_cm3_molecule_s"],
         [row["rate_estimate_cm3_s"] for row in previous]),
    ]
    for index, (title, key, central, interval, older) in enumerate(quantities):
        ax = figure.add_subplot(grid[0, index])
        target = paper[key]
        relative, bounds = central / target, np.asarray(interval) / target
        ax.axvspan(.95, 1.05, color="#dddddd", zorder=0)
        ax.axvline(1., color="#333333", linestyle=":", lw=1)
        ax.scatter(np.asarray(older) / target, [2, 1], marker="x", s=52, color=orange, zorder=3)
        ax.errorbar(relative, 0, xerr=[[relative - bounds[0]], [bounds[1] - relative]],
                    fmt="o", ms=6, capsize=4, lw=1.7, color=blue, zorder=4)
        ax.set(ylim=(-.55, 2.55), title=title, xlabel="Estimate / published value")
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(["Earlier umbrella A\n(unconverged)",
                            "Earlier umbrella B\n(unconverged)",
                            f"Independent source\n({len(report['completed_seeds'])} chains)"] if index == 0 else [])
        if key == "kappa":
            lo = min(.87, float(bounds[0]) - .02)
            hi = max(1.06, float(bounds[1]) + .02)
            ax.set_xlim(lo, hi)
        else:
            ax.set_xscale("log")
            ax.set_xlim(min(.40, float(bounds[0]) * .9), max(3.1, float(bounds[1]) * 1.1))
            ax.xaxis.set_major_locator(FixedLocator([.5, 1, 2, 3]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.minorticks_off()
        ax.text(.5, -.29, f"New point: {100 * (relative - 1):+.2f}%", transform=ax.transAxes,
                ha="center", color=blue, fontsize=10)

    ax = figure.add_subplot(grid[1, :])
    ax.fill_between(curve["time_fs"], curve["pointwise_CI95_low"], curve["pointwise_CI95_high"],
                    color=blue, alpha=.17, label="Between-chain pointwise 95% interval")
    ax.plot(curve["time_fs"], curve["kappa_seed_mean"], color=blue, lw=1.9,
            label="Native dynamics; dt = 0.05 fs")
    ax.axhline(paper["kappa"], color="#333333", lw=1.2, linestyle=":",
               label=f"Paper plateau = {paper['kappa']:g}")
    lower, upper = map(int, report["primary_plateau_fs"].split("-"))
    ax.axvspan(lower, upper, color="#999999", alpha=.12, label="Registered final plateau window")
    ax.set(xlim=(0, report["horizon_fs"]), xlabel="Physical child time / fs",
           ylabel=r"$\kappa(t)$", title="D  Fresh children from every registered independent parent chain")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    figure.suptitle("H + H₂ at 300 K on BKMP2: independently normalized P128 RPMD", fontsize=15)
    passed = report["fixed_P_independent_rate_gates_passed"]
    rate, interval = statistics["rate_geometric_mean_cm3_molecule_s"], statistics["rate_CI95_cm3_molecule_s"]
    footer = (
        f"Full rate: {rate:.6g} [{interval[0]:.6g}, {interval[1]:.6g}] cm³ molecule⁻¹ s⁻¹ (95%). "
        f"Registered fixed-P rate gates: {'PASS' if passed else 'FAIL'}.\n"
        "New intervals use complete independent chains; the full-rate interval retains partition–transmission covariance.\n"
        "Gray bands in A–C denote the paper's approximately ±5% precision, not a reconstructed confidence interval.\n"
        "Earlier points failed sampling convergence; their unreliable bootstrap intervals are not used here. "
        "Separate bead/domain controls remain separate evidence."
    )
    figure.text(.08, .10, footer, va="center", fontsize=9, color=gray, linespacing=1.6)
    outputs = []
    for extension in ("png", "pdf"):
        path = output / f"quantum-convergence.{extension}"
        figure.savefig(path, dpi=180)
        outputs.append(path)
    plt.close(figure)
    metadata = {
        "schema": "pyrpmd.h3.quantum-convergence-figure.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plot_source_sha256": sha(__file__),
        "inputs": [{"path": str(path), "sha256": sha(path)}
                   for path in [report_path, curve_path, *prior_paths]],
        "figures": {path.name: sha(path) for path in outputs},
        "complete_cohort": report["completed_seeds"],
        "fixed_P_gates_passed": passed,
        "plot_changes_no_scientific_estimator_or_gate": True,
    }
    (output / "figure-provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.analysis, args.historical, args.output), indent=2))
