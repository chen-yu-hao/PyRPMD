"""Plot the retained classical convergence evidence without resampling.

The paper's approximate five-percent precision is a tolerance band, not a
newly invented confidence interval. All campaign stages found in the input
folder are displayed, including those that fail the registered gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pyrpmd.constants import KB_AU


def plot(evidence: Path, output: Path):
    evidence, output = Path(evidence), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs = []

    def read_json(path):
        inputs.append(path)
        return json.loads(path.read_text())

    reference_path = evidence / "reference-384/conditional-reference.npz"
    inputs.append(reference_path)
    with np.load(reference_path, allow_pickle=False) as data:
        xi, density = data["xi"].copy(), data["density"].copy()
    reference = read_json(evidence / "reference-384/reference.json")
    comparison = read_json(evidence / "transmission/transmission-comparison.json")
    old_path = evidence / "window-diagnosis-pmf.csv"
    inputs.append(old_path)
    old = np.genfromtxt(old_path, delimiter=",", names=True)
    stages = sorted(evidence.glob("stage-*/campaign-comparison.json"),
                    key=lambda path: int(path.parent.name.split("-")[-1]))
    records = [read_json(path) for path in stages]
    beta = 1.0 / (300.0 * KB_AU)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 11, "legend.fontsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2))
    fig.subplots_adjust(left=.075, right=.98, bottom=.20, top=.90, wspace=.28, hspace=.42)
    blue, orange, gray, green = "#2166ac", "#d6604d", "#777777", "#1b7837"

    ax = axes[0, 0]
    ax.plot(old["xi"], beta*old["exact_equilibrium"], color="black", lw=2,
            label="Independent equilibrium quadrature")
    ax.plot(old["xi"], beta*old["all"], color=orange, lw=1.8,
            label="Previous short P=1 MD estimate")
    ax.plot(old["xi"], beta*old["first_half"], color=orange, lw=.8, alpha=.5)
    ax.plot(old["xi"], beta*old["second_half"], color=orange, lw=.8, alpha=.5)
    ax.set(xlim=(0, 1.02), xlabel=r"Reaction coordinate $\xi$",
           ylabel=r"Free energy / $k_BT$", title="A  Accumulated short-window sampling error")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[0, 1]
    for stiffness, color in ((2., orange), (30., blue)):
        logp = np.log(density)-.5*beta*stiffness*(xi-1.)**2
        probability = np.exp(logp-logp.max())
        probability /= trapezoid(probability, xi)
        ax.plot(xi, probability, color=color, lw=1.8,
                label=f"Umbrella stiffness {stiffness:g} hartree")
    ax.axvspan(.998, 1.002, color=green, alpha=.15, label=r"Barrier interval $1\pm0.002$")
    ax.set(xlim=(.87, 1.13), ylim=(1e-5, 100), yscale="log",
           xlabel=r"Reaction coordinate $\xi$", ylabel="Exact biased probability density",
           title="B  A soft barrier window is bimodal")
    ax.legend(loc="lower left", frameon=False)

    ax = axes[1, 0]
    target = reference["rate_at_xi_one_cm3_molecule_s"] * 1e18
    ax.axhline(target, color="black", lw=1.5, label="Independent TST quadrature")
    for index, report in enumerate(records):
        stage = report["production_sweeps_per_ladder"]
        pooled = report["pooled_wham"]["qtst_cm3_molecule_s"]*1e18
        interval = np.array(report["independent_seed_student_t_95_interval_cm3_molecule_s"])*1e18
        ax.errorbar(index, pooled, yerr=[[pooled-interval[0]], [interval[1]-pooled]],
                    fmt="o", color=blue, capsize=5, lw=1.5,
                    label="Pooled WHAM; four-seed 95% interval" if index == 0 else None)
        per_seed = [row["qtst_cm3_molecule_s"]*1e18 for row in report["per_seed"]]
        ax.scatter(index+np.linspace(-.12, .12, len(per_seed)), per_seed,
                   s=18, color=gray, alpha=.75, zorder=3,
                   label="Individual seeds" if index == 0 else None)
        ax.text(index, .97, "pass" if report["equilibrium_sampling_criteria_passed"] else "gates fail",
                transform=ax.get_xaxis_transform(), ha="center", va="top", color=gray, fontsize=8)
    ax.set_xticks(np.arange(len(records)),
                  [f'{r["production_sweeps_per_ladder"]:,}' for r in records])
    ax.set(xlim=(-.45, len(records)-.55), xlabel="Production sweeps per independent ladder",
           ylabel=r"Classical TST rate / $10^{-18}$ cm$^3$ molecule$^{-1}$ s$^{-1}$",
           title="C  Every fixed-budget stage is retained")
    ax.legend(loc="lower left", frameon=False)

    ax = axes[1, 1]
    for timestep, style, color in ((.1, "--", orange), (.05, "-", blue)):
        traces = []
        for seed in comparison["independent_seeds"]:
            path = evidence / f"transmission/seed-{seed}/dt-{timestep:g}/kappa.csv"
            inputs.append(path)
            values = np.genfromtxt(path, delimiter=",", names=True)
            traces.append(values["kappa"])
        traces = np.array(traces)
        mean = traces.mean(axis=0)
        halfwidth = student_t.ppf(.975, len(traces)-1)*traces.std(axis=0, ddof=1)/np.sqrt(len(traces))
        time_fs = values["time_fs"]
        ax.plot(time_fs, mean, style, color=color, lw=1.8,
                label=f"Native dynamics, dt={timestep:g} fs")
        if timestep == .05:
            ax.fill_between(time_fs, mean-halfwidth, mean+halfwidth, color=blue, alpha=.15,
                            label="Four-seed pointwise 95% interval")
    ax.axhline(comparison["paper_classical_kappa"], color="black", lw=1,
               linestyle=":", label="Paper classical transmission coefficient")
    ax.set(xlim=(0, 400), ylim=(.9905, 1.001), xlabel="Physical child time / fs",
           ylabel=r"Transmission coefficient $\kappa(t)$",
           title="D  Independent surface source + native dynamics")
    ax.legend(loc="lower right", frameon=False)

    rate = comparison["independent_classical_rate_cm3_molecule_s"]*1e18
    interval = np.array(comparison["independent_classical_rate_95_interval"])*1e18
    fig.suptitle("H + H₂ at 300 K on BKMP2: classical convergence and independent reference", fontsize=15)
    fig.text(.075, .065,
             f"Independent classical rate: {rate:.6f} [{interval[0]:.6f}, {interval[1]:.6f}] × 10⁻¹⁸ cm³ molecule⁻¹ s⁻¹ (95%).\n"
             "Paper: 1.55 × 10⁻¹⁸ with approximately 5% reported precision. Agreement is within paper precision;\n"
             "the paper point is outside our smaller interval. This reference does not validate the umbrella or quantum result.",
             fontsize=10, va="center")
    for extension in ("png", "pdf"):
        fig.savefig(output/f"classical-convergence.{extension}", dpi=180)
    plt.close(fig)
    metadata = {"schema": "pyrpmd.h3.classical-evidence-figure.v1",
                "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "inputs": [{"path": str(path.relative_to(evidence)),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in inputs],
                "figures": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in output.glob("classical-convergence.*")}}
    (output/"figure-provenance.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.evidence, args.output)
