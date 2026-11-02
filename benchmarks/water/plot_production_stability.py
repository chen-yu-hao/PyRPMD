#!/usr/bin/env python3
"""Descriptive, preregistered production halves for the complete original cohort.

All 25 segments remain in the primary means and display. Only the balanced
12-versus-12 contrast omits segment 12. Eight paired chain differences are the
uncertainty units; this supplement introduces no hard gate or paper comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.water.analyze_native import chain_uncertainty
from benchmarks.water.campaign import atomic_json, sha256
from benchmarks.water.plot_native_campaign import load_complete_cohort

SUPPLEMENT_SHA256 = "ac930daf5280da0bb1fb79fc17d51a4d2d0a83fa5db5bd8c139758ab885d1065"
SEEDS = list(range(1101, 1109))
ENDPOINTS = (("VACF", "D_VACF_2ps", "mean_D_VACF_2ps"),
             ("MSD", "D_MSD_1_2ps", "mean_D_MSD_1_2ps"))


def paired_halves(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (8, 25) or not np.isfinite(values).all():
        raise ValueError("every one of the eight by 25 segment estimates must be finite")
    early, late = values[:, :12].mean(axis=1), values[:, 13:].mean(axis=1)
    return {"early_chain_means": early.tolist(), "late_chain_means": late.tolist(),
            "all25_chain_means": values.mean(axis=1).tolist(),
            "paired_chain_differences": (late - early).tolist(),
            "paired_difference": chain_uncertainty(late - early)}


def load_diagnostics(directory, protocol_path, supplement_path, exit_proof_path):
    supplement_path, exit_proof_path = Path(supplement_path), Path(exit_proof_path)
    supplement = json.loads(supplement_path.read_text())
    if (sha256(supplement_path) != SUPPLEMENT_SHA256
            or supplement["fixed_seeds"] != SEEDS
            or supplement["early_segments"] != list(range(12))
            or supplement["late_segments"] != list(range(13, 25))
            or supplement["original_protocol_sha256"] != sha256(protocol_path)):
        raise ValueError("the sealed production-stability supplement or original protocol differs")
    for name, expected in supplement["frozen_analysis_sources_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError("the frozen analysis dependency changed")
    data = load_complete_cohort(directory, protocol_path)
    proof = json.loads(exit_proof_path.read_text())
    rows = proof.get("completed_chains", [])
    if (proof.get("schema") != "pyrpmd.water-original-coordinator-normal-exit.v1"
            or type(proof.get("process_exit_code")) is not int or proof["process_exit_code"] != 0
            or proof.get("pool_drained") is not True
            or proof.get("all_original_leases_acquired_and_released") is not True
            or proof.get("coordinator_finished_event", {}).get("event") != "coordinator_finished"
            or proof.get("coordinator_finished_event", {}).get("failed") is not False
            or len(rows) != 8 or sorted(row["seed"] for row in rows) != SEEDS):
        raise ValueError("the complete original cohort needs an observed normal exit and released leases")
    by_seed = {row["seed"]: row for row in rows}
    for row in data["chains"]:
        observed = by_seed[row["seed"]]
        if (observed.get("complete") is not True
                or observed.get("manifest_sha256") != row["manifest_sha256"]
                or observed.get("committed_steps") != row["committed_steps"]):
            raise ValueError("the observed exit and analyzed original manifest identities differ")
    endpoints = {}
    for label, key, mean_key in ENDPOINTS:
        values = np.asarray([[part[key] for part in row["segments"]] for row in data["chains"]], dtype=float)
        summary = paired_halves(values)
        if not np.allclose(summary["all25_chain_means"], [row[mean_key] for row in data["chains"]],
                           atol=1e-13, rtol=1e-11):
            raise ValueError("all 25 segment means must reproduce the original primary chain means")
        endpoints[label] = {"values": values, "summary": summary}
    data.update(endpoints=endpoints, supplement=supplement, exit_proof=proof)
    data["input_sha256"].update(supplement=sha256(supplement_path), exit_proof=sha256(exit_proof_path))
    return data


def plot(directory, protocol_path, supplement_path, exit_proof_path, output):
    inputs = {"report.json": Path(directory) / "report.json", "curves.npz": Path(directory) / "curves.npz",
              "protocol.json": Path(protocol_path), "supplement": Path(supplement_path),
              "exit_proof": Path(exit_proof_path)}
    sources = {name: sha256(ROOT / name) for name in (
        "benchmarks/water/plot_production_stability.py", "benchmarks/water/plot_native_campaign.py",
        "benchmarks/water/analyze_native.py", "benchmarks/water/campaign.py")}
    data = load_diagnostics(directory, protocol_path, supplement_path, exit_proof_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_rows, segment_rows = [], []
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    colors = plt.get_cmap("tab10").colors
    for axis, (label, _, _) in zip(axes, ENDPOINTS):
        values, summary = data["endpoints"][label]["values"], data["endpoints"][label]["summary"]
        for index, seed in enumerate(SEEDS):
            axis.plot(range(25), values[index], color=colors[index], lw=.9, alpha=.8, label=str(seed))
            summary_rows.append({"endpoint": label, "seed": seed,
                                 "early_segments0_11": summary["early_chain_means"][index],
                                 "late_segments13_24": summary["late_chain_means"][index],
                                 "late_minus_early": summary["paired_chain_differences"][index],
                                 "primary_all25_segments": summary["all25_chain_means"][index]})
            for number, value in enumerate(values[index]):
                segment_rows.append({"endpoint": label, "seed": seed, "segment": number,
                                     "D_angstrom2_ps": float(value), "retained_in_primary": True,
                                     "balanced_half": "early" if number < 12 else "late" if number > 12 else "middle"})
        axis.plot(range(25), values.mean(axis=0), color="black", lw=2, label="Eight-chain mean")
        axis.axvspan(11.5, 12.5, color="0.85", alpha=.5)
        axis.set(xlabel="NVE segment index (each segment is 4 ps)", ylabel="D (Å²/ps)",
                 title="VACF integral to 2 ps" if label == "VACF" else "MSD slope over 1–2 ps",
                 xlim=(-.5, 24.5), xticks=[0, 4, 8, 12, 16, 20, 24])
        axis.grid(alpha=.2)
        ci = summary["paired_difference"]
        axis.text(.02, .98, f"Late−early: {ci['mean']:+.5f}\n95% paired-chain CI [{ci['CI95'][0]:+.5f}, {ci['CI95'][1]:+.5f}]",
                  transform=axis.transAxes, va="top", fontsize=9,
                  bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
    synthetic = data["report"].get("synthetic_data") is True
    fig.suptitle("SYNTHETIC DATA — DISPLAY CHECK ONLY" if synthetic else
                 "Original fixed water cohort: all 200 NVE segment means", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=9, loc="upper center", bbox_to_anchor=(.5, .925), fontsize=8)
    fig.text(.5, .065, "OpenMM external reference · Eight paired chain differences, nominal t interval (df 7).\n"
             "Segments 0–11 vs 13–24; middle 12 remains in every primary mean and display. No new hard gate.\n"
             "Original preparation failures are retained; this descriptive diagnostic does not establish paper agreement.",
             ha="center", va="center", fontsize=9)
    fig.subplots_adjust(top=.82, bottom=.22, left=.075, right=.98, wspace=.25)
    for extension in ("png", "svg"):
        fig.savefig(output / f"production-stability.{extension}", dpi=180)
    plt.close(fig)
    for name, rows in (("production-segment-means.csv", segment_rows), ("production-half-chain-means.csv", summary_rows)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if any(sha256(path) != data["input_sha256"][key] for key, path in inputs.items()):
        raise RuntimeError("an input changed during the offline diagnostic")
    if any(sha256(ROOT / path) != expected for path, expected in sources.items()):
        raise RuntimeError("a diagnostic source changed during plotting")
    report = {"schema": "pyrpmd.water-production-stability-diagnostic.v1", "synthetic_data": synthetic,
              "external_OpenMM_reference": True, "new_hard_gate": False, "literature_reproduced": False,
              "original_eligibility_checks": data["report"]["eligibility_checks"],
              "input_sha256": data["input_sha256"], "source_sha256": sources,
              "summaries": {label: value["summary"] for label, value in data["endpoints"].items()},
              "interpretation": data["supplement"]["statistics"]["interpretation"],
              "artifacts": {name: sha256(output / name) for name in (
                  "production-stability.png", "production-stability.svg", "production-segment-means.csv",
                  "production-half-chain-means.csv")}}
    atomic_json(output / "production-stability.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--exit-proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.directory, args.protocol, args.supplement, args.exit_proof, args.output)


if __name__ == "__main__":
    main()
