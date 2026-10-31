"""Collect every completed settings-scan case, with no parameter filtering.

The JSON retains all per-window diagnostics; the CSV is one row per completed
case. The figure deliberately separates fixed-step thermostat comparisons
from equal-physical-time timestep comparisons.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyrpmd.analysis.fes import statistical_inefficiency


def collect(root):
    cases = []
    for path in sorted(root.glob("*/summary.json")):
        case = json.loads(path.read_text())
        case["case"] = path.parent.name
        resolved = json.loads((path.parent / "input-resolved.json").read_text())
        case["sample_interval"] = resolved["config"]["sampling"]["sample_interval"]
        case["integration_steps"] = case["window_count"] * (case["production_steps"] + case["equilibration_steps"])
        for window in case["windows"]:
            if "mean_and_variance_ess" not in window:
                checkpoint = path.parent/"checkpoints"/f"window_{window['window']:05d}_replica_0000.npz"
                with np.load(checkpoint) as saved:
                    q = saved["values"]
                window["mean_and_variance_ess"] = len(q)/statistical_inefficiency(np.c_[q, (q-q.mean(0))**2])
        case["mean_and_variance_ess_min"] = min(w["mean_and_variance_ess"] for w in case["windows"])
        case["mean_and_variance_ess_median"] = float(np.median([w["mean_and_variance_ess"] for w in case["windows"]]))
        case["mean_and_variance_ess_below_30"] = sum(w["mean_and_variance_ess"] < 30 for w in case["windows"])
        cases.append(case)
    return cases


def plot(cases, root):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    full = [c for c in cases if c["model"] == "ackley" and c["window_count"] == 49 and c["dt_fs"] == .1]
    for x, a in enumerate([100, 500, 1000]):
        group = [c for c in full if c["andersen_interval"] == a]
        for i, c in enumerate(group):
            axes[0,0].scatter(x+(i-1)*.08, c["mean_and_variance_ess_min"], color="#126d91", s=40)
            axes[0,1].scatter(x+(i-1)*.08, c["normalized_mean_error_rms"], color="#126d91", label="Mean" if x==0 and i==0 else None)
            axes[0,1].scatter(x+(i-1)*.08, c["normalized_variance_error_rms"], color="#c46328", marker="s", label="Variance" if x==0 and i==0 else None)
    axes[0,0].axhline(30, color="#9d3131", ls="--", lw=1, label="ESS target = 30")
    axes[0,0].set(title="Ackley: minimum ESS over 49 windows", ylabel="ESS for coordinates and squared deviations")
    axes[0,0].legend()
    axes[0,1].set(title="Ackley: window moments vs independent quadrature", ylabel="Dimensionless RMS error")
    axes[0,1].legend()
    for ax in axes[0]:
        ax.set_xticks(range(3), ["10 fs", "50 fs", "100 fs"])
        ax.set_xlabel("Collision period (dt = 0.1 fs, 80k production steps)")
    for x, (dt, a) in enumerate([(.1, 500), (.5, 100)]):
        group = [c for c in cases if c["model"] == "ackley" and c["window_count"] == 6 and c["dt_fs"] == dt and c["andersen_interval"] == a and c["production_time_fs"] == 8000]
        for i, c in enumerate(group):
            axes[1,0].scatter(x+(i-1)*.08, c["normalized_mean_error_rms"], color="#126d91", label="Mean" if x==0 and i==0 else None)
            axes[1,0].scatter(x+(i-1)*.08, c["normalized_variance_error_rms"], color="#c46328", marker="s", label="Variance" if x==0 and i==0 else None)
    axes[1,0].set_xticks([0,1], ["dt 0.1 fs / 80k steps", "dt 0.5 fs / 16k steps"])
    axes[1,0].set(title="Equal physical time: 8 ps, collision period 50 fs", ylabel="Dimensionless RMS error (6 windows)")
    axes[1,0].legend()
    for x, a in enumerate([25, 50, 100, 200]):
        group = [c for c in cases if c["model"] == "double_well" and c["dt_fs"] == .1 and c["andersen_interval"] == a]
        for i, c in enumerate(group):
            axes[1,1].scatter(x+(i-1)*.08, c["normalized_variance_error_rms"], color="#c46328", marker="s")
    axes[1,1].set_xticks(range(4), ["2.5 fs", "5 fs", "10 fs", "20 fs"])
    axes[1,1].set(title="Double well: variance mixing (5 test windows)", xlabel="Collision period (dt = 0.1 fs, 80k steps)", ylabel="Dimensionless variance RMS error")
    fig.suptitle("Thermostat tuning: independent seeds 5101–5103\nCoordinate ESS alone can miss slow variance relaxation", fontsize=14)
    fig.savefig(root/"settings_scan.png", dpi=180)
    fig.savefig(root/"settings_scan.pdf")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args=p.parse_args()
    cases=collect(args.root)
    result={"case_count":len(cases), "integration_steps":sum(c["integration_steps"] for c in cases),
            "notes":["All completed parameter attempts are retained.",
                     "These are individual biased-window diagnostics, not a reconstructed FES.",
                     "Wall times include quadrature and sometimes reused checkpoints; do not compare them as throughput.",
                     "Changing timestep at fixed step count changes physical time. Only marked equal-time comparisons isolate timestep effects.",
                     "ESS remains a finite-trajectory diagnostic, not proof of equilibration.",
                     "Mean errors are divided by exact window standard deviations; variance errors are relative variance errors."],
            "cases": cases}
    (args.root/"settings_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    rows=[{k:v for k,v in c.items() if k != "windows"} for c in cases]
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (args.root/"settings_summary.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=keys);writer.writeheader();writer.writerows(rows)
    plot(cases,args.root)
    print(json.dumps({k:v for k,v in result.items() if k != "cases"}))
