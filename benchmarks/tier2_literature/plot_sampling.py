#!/usr/bin/env python3
"""Render the saved sampling campaign without rerunning any dynamics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"metad_fixed": "#1864ab", "explore_fixed": "#e67700",
          "metad_adaptive": "#6741d9", "explore_adaptive": "#2b8a3e"}
NAMES = {"metad_fixed": "OPES fixed", "explore_fixed": "Explore fixed",
         "metad_adaptive": "OPES Welford", "explore_adaptive": "Explore Welford"}


def finish(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(output):
    output = Path(output)
    figures = output/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                        "axes.spines.right": False, "figure.facecolor": "white"})
    results = [(path.parent, json.loads(path.read_text()))
               for path in sorted((output/"opes_v2").glob("*/metrics.json"))]
    baseline = [(path, result) for path, result in results
                if result["dt"] == .005 and result["initial_sigma"] == .1]
    if baseline:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for column, cv in enumerate("xy"):
            ax, transitions = axes[:, column]
            data = [(path, item) for path, item in baseline if item["cv"] == cv]
            reference_drawn, shown = False, set()
            for path, item in data:
                raw = np.load(path/"trajectory.npz")
                key = item["mode"] + ("_adaptive" if item["adaptive_sigma"] else "_fixed")
                exact = raw["exact_fes"]-raw["exact_fes"].min()
                if not reference_drawn:
                    ax.plot(raw["centers"], exact, color="black", lw=2, label="Exact quadrature")
                    reference_drawn = True
                estimate = raw["direct_fes"]
                mask = exact <= 10
                estimate = estimate - np.mean(estimate[mask]-exact[mask])
                ax.plot(raw["centers"], estimate, color=COLORS[key], alpha=.65, lw=1,
                        label=NAMES[key] if key not in shown else None)
                shown.add(key)
            ax.set(xlabel=f"CV {cv}", ylabel="Direct KDE FES / kBT", ylim=(-8, 28),
                   title=f"CV {cv}: every independent seed, 160 updates")
            ax.axhspan(10, 28, color="grey", alpha=.05)
            ax.legend(fontsize=8, frameon=False)
            for offset, key in enumerate(COLORS):
                selected = [item for _, item in data if item["mode"] + ("_adaptive" if item["adaptive_sigma"] else "_fixed") == key]
                values = [item["transitions_adapting"] for item in selected]
                x = offset + np.linspace(-.15, .15, len(values))
                transitions.scatter(x, values, color=COLORS[key], s=35, zorder=3)
                if values:
                    transitions.hlines(np.mean(values), offset-.22, offset+.22, color=COLORS[key], lw=2)
            transitions.set(xticks=range(4), xticklabels=[NAMES[key].replace(" ", "\n") for key in COLORS],
                            ylabel="Basin-core crossings", title="Adaptation phase: points = seeds; line = mean")
            transitions.grid(axis="y", alpha=.2)
        fig.suptitle("Published Müller model: short-trajectory stress test\nDifferent bandwidth rule and 2,500× fewer adaptation steps than the paper", fontsize=14)
        finish(fig, figures/"opes_fes_and_crossings")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
        for ax, cv in zip(axes, "xy"):
            data = [item for _, item in baseline if item["cv"] == cv]
            ax.axhline(5.712591741222511, color="black", ls="--", lw=1.5, label="Exact ΔFₓ = 5.71259")
            labels = []
            for offset, key in enumerate(COLORS):
                selected = [item for item in data if item["mode"] + ("_adaptive" if item["adaptive_sigma"] else "_fixed") == key]
                valid = [item["frozen_delta_F_x_kBT"] for item in selected if item["frozen_delta_F_x_kBT"] is not None]
                ax.scatter(offset+np.linspace(-.16, .16, len(valid)), valid, s=40, color=COLORS[key])
                labels.append(NAMES[key].replace(" ", "\n")+f"\n{len(valid)}/{len(selected)} finite")
            ax.set(xticks=range(4), xticklabels=labels, ylabel="Frozen, reweighted ΔFₓ / kBT",
                   title=f"Bias CV {cv}: all finite seed estimates")
            ax.legend(frameon=False, fontsize=9)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle("Missing basins produce undefined estimates; they are not replaced by zero", fontsize=13)
        finish(fig, figures/"opes_basin_free_energies")

    comparator = output/"plumed_comparator"/"direct_energy_force.csv"
    if comparator.exists():
        with comparator.open() as file:
            rows = list(csv.DictReader(file))
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
        for ax, quantity in zip(axes, ("energy", "force")):
            for mode, color in (("metad", COLORS["metad_fixed"]), ("explore", COLORS["explore_fixed"])):
                values = [r for r in rows if r["mode"] == mode]
                updates = sorted(set(int(r["deposition"]) for r in values))
                for algorithm, style in (("pyrpmd", "-"), ("legacy_v1", "--")):
                    errors = [max(abs(float(r[f"plumed_{quantity}"])-float(r[f"{algorithm}_{quantity}"]))
                                  for r in values if int(r["deposition"]) == update) for update in updates]
                    ax.semilogy(updates, np.maximum(errors, 1e-16), color=color, ls=style,
                                label=f"{mode}: {'new v2' if algorithm == 'pyrpmd' else 'legacy v1'}")
            ax.set(xlabel="Completed depositions", ylabel=f"Maximum absolute {quantity} difference",
                   title=f"Direct {quantity}: 21 probes / deposition")
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle("Live PLUMED 2.9.4 comparison: fixed widths, compression disabled, no interpolation", fontsize=12)
        finish(fig, figures/"plumed_initialization_regression")

    path = output/"replica_exchange"/"summary.json"
    if path.exists():
        rows = json.loads(path.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, scheme in zip(axes, ("temperature", "hamiltonian")):
            for index, observable in enumerate(("centroid_variance", "bead_variance", "momentum_variance")):
                data = [r for r in rows if r["scheme"] == scheme and r["observable"] == observable]
                ax.errorbar(np.arange(3)+(index-1)*.12, [r["mean"]/r["reference"] for r in data],
                            yerr=[r["95CI_halfwidth"]/r["reference"] for r in data], fmt="o", capsize=3,
                            label=observable.replace("_", " "))
            ax.axhline(1., color="black", ls="--", lw=1)
            ax.set(xticks=[0, 1, 2], xlabel="Thermodynamic slot", ylabel="Measured / exact finite-P variance",
                   title=f"{scheme.title()} exchange: four seeds")
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle("Thermal ensemble engineering check; bars = 95% t intervals across seeds", fontsize=13)
        finish(fig, figures/"replica_exchange_ensembles")

    path = output/"ml_lifecycle"/"training_and_heldout.npz"
    if path.exists():
        data = np.load(path)
        if {"force_error", "force_disagreement"} <= set(data.files):
            fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
            for label, ids, color in (("Held-out interpolation", slice(0, 400), "#1864ab"),
                                      ("Held-out extrapolation", slice(400, None), "#e67700")):
                ax.loglog(data["force_disagreement"][ids], data["force_error"][ids], ".", ms=4, color=color, alpha=.6, label=label)
            ax.set(xlabel="Committee force disagreement / atomic units", ylabel="Actual maximum force-vector error / atomic units",
                   title="Trained TorchScript committee: disagreement is not an error bound")
            ax.legend(frameon=False)
            finish(fig, figures/"ml_heldout_error")

    shared = sorted((output/"shared_metadynamics").glob("seed_*.npz"))
    if shared:
        datasets = [np.load(path) for path in shared]
        grid = datasets[0]["grid"]
        normalize = lambda values: values/values.sum()
        hist = np.mean([normalize(data["histogram"]) for data in datasets], axis=0)
        weighted = np.mean([normalize(data["weighted_histogram"]) for data in datasets], axis=0)
        frozen = np.mean([normalize(np.exp(-grid*grid/2-data["bias"]+data["bias"].min())) for data in datasets], axis=0)
        ideal, unbiased = normalize(np.exp(-grid*grid/10)), normalize(np.exp(-grid*grid/2))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        axes[0].plot(grid, hist, label="Sampled frozen stage", color="#1864ab")
        axes[0].plot(grid, frozen, "--", label="Exact frozen ensemble", color="black")
        axes[0].plot(grid, ideal, ":", label="Ideal well-tempered target", color="#e67700", lw=2)
        axes[1].plot(grid, weighted, label="Reweighted samples", color="#1864ab")
        axes[1].plot(grid, unbiased, "--", label="Exact unbiased harmonic", color="black")
        for ax in axes:
            ax.set(xlabel="x", ylabel="Probability per bin", xlim=(-6, 6))
            ax.legend(frameon=False, fontsize=9)
        axes[0].set_title("Stationary sampling vs adaptive-bias convergence")
        axes[1].set_title("Reweighting restores the unbiased distribution")
        fig.suptitle("Shared MetaD: four walkers, four independent bias histories", fontsize=13)
        finish(fig, figures/"shared_metadynamics_frozen")

    kinetics = [item for path in sorted((output/"kinetics").glob("summary_dt*.json"))
                for item in json.loads(path.read_text())]
    if kinetics:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
        x = np.linspace(-1.65, .8, 501)
        u = 6*(x*x-1)**2
        v = 2*np.maximum(1-((x+1)/.65)**2, 0)**3
        axes[0].plot(x, u, color="black", label="U(x)")
        axes[0].plot(x, u+v, color="#1864ab", label="U(x) + static bias")
        axes[0].plot(x, v, color="#e67700", ls="--", label="Static bias")
        axes[0].axvspan(-.35, .8, color="#2b8a3e", alpha=.08, label="Audited transition region")
        axes[0].set(xlabel="x", ylabel="Energy / kBT", ylim=(-.2, 9), title="Zero bias and gradient near the transition")
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].axhline(kinetics[0]["exact_unbiased_MFPT"], color="black", ls="--", label="Exact reflecting/absorbing MFPT")
        for i, item in enumerate(kinetics):
            color = "#1864ab" if item["static_flooding_bias"] else "#e67700"
            axes[1].errorbar(i, item["rescaled_MFPT"]["mean"], yerr=item["rescaled_MFPT"]["95CI_halfwidth"], fmt="o", color=color, capsize=4)
        axes[1].set(xticks=range(len(kinetics)),
            xticklabels=[("Rescaled" if item["static_flooding_bias"] else "Unbiased")+f"\ndt={item['dt']:g}" for item in kinetics],
            ylabel="Mean first-passage time / model time", title="128 independent paths per point; 95% intervals")
        axes[1].legend(frameon=False, fontsize=8)
        fig.suptitle("Biased-clock analysis on actual overdamped paths; static-bias engineering reference", fontsize=12)
        finish(fig, figures/"biased_kinetics_reference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/validation/tier2-literature-2026-11-01/sampling"))
    main(parser.parse_args().output)
