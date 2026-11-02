#!/usr/bin/env python3
"""Plot a complete fixed eight-chain external water reference cohort.

The helper consumes the streaming analyzer's compact report/curves. It refuses
incomplete or duplicate cohorts, checks curve checksums and independent-chain
uncertainty, and retains failed preparation gates in the figure and tables.
Paper comparison remains disabled when the analyzer withheld eligibility.
No raw trajectory, random state, or running source is changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.water.campaign import atomic_json, sha256
from pyrpmd.constants import KCAL_MOL_TO_AU


def independent_summary(values):
    """Eight chain means, never the 200 adjacent segments or time origins."""
    values = np.asarray(values, dtype=float)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError("exactly eight finite independent-chain means are required")
    sem = float(values.std(ddof=1) / np.sqrt(8))
    return {"mean": float(values.mean()), "standard_error": sem,
            "CI95_halfwidth": float(student_t.ppf(.975, 7) * sem)}


def load_complete_cohort(directory, protocol_path):
    directory, protocol_path = Path(directory), Path(protocol_path)
    report_path, curves_path = directory / "report.json", directory / "curves.npz"
    report_bytes, protocol_bytes = report_path.read_bytes(), protocol_path.read_bytes()
    report, protocol = json.loads(report_bytes), json.loads(protocol_bytes)
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    if (report.get("schema") != "pyrpmd.water-native-chain-analysis.v1"
            or report.get("external_reference") is not True
            or report.get("pyrpmd_production_trajectory") is not False):
        raise ValueError("expected the explicitly labeled native OpenMM external-reference report")
    if report.get("all_predeclared_chains_complete") is not True:
        raise ValueError("complete all eight registered chains before aggregate or paper-comparison plotting")
    expected_seeds = protocol["fixed_stage"]["seeds"]
    chains = sorted(report["chains"], key=lambda row: row["seed"])
    if (len(expected_seeds) != 8 or len(set(expected_seeds)) != 8
            or [row["seed"] for row in chains] != sorted(expected_seeds)
            or not all(row["complete"] is True for row in chains)):
        raise ValueError("the report must contain every fixed seed exactly once")
    if report["protocol_sha256"] != protocol_sha:
        raise ValueError("report and protocol checksums differ")
    curves_sha = sha256(curves_path)
    if report["artifacts"]["curves.npz"]["sha256"] != curves_sha:
        raise ValueError("compact curve checksum mismatch")
    segments = protocol["fixed_stage"]["NVE_segments_per_chain"]
    for row in chains:
        config = row["config"]
        total = config["equilibration_steps"] + config["segment_steps"] * config["segments"]
        if (row["committed_steps"] != total or row.get("incomplete_segment") is not None
                or config["segments"] != segments or len(row["segments"]) != segments
                or [part["segment"] for part in row["segments"]] != list(range(segments))
                or not all(part["included_in_mean"] is True for part in row["segments"])):
            raise ValueError("all registered production segments must be complete and retained")
        if config["beads"] != 1 or config["isotope"] != "H2O":
            raise ValueError("this figure describes the registered classical H2O cohort")
        if (not np.isclose(config["equilibration_steps"] * config["dt_fs"] / 1000, 100, atol=1e-12, rtol=0)
                or not np.isclose(config["segment_steps"] * config["dt_fs"] / 1000, 4, atol=1e-12, rtol=0)):
            raise ValueError("this original-cohort figure requires 100 ps preparation and 4 ps NVE segments")
        difference = row["sampling"]["potential_per_molecule_hartree"]["half_mean_difference"]
        if difference is None or not np.isfinite(difference):
            raise ValueError("complete preparation energy diagnostics are required")
    checks = report["eligibility_checks"]
    if checks.get("fixed_stage_complete") is not True:
        raise ValueError("the analyzer has not marked the fixed stage complete")
    paper_allowed = all(value is True for value in checks.values())
    if (report.get("literature_comparison") is not None) != paper_allowed:
        raise ValueError("paper-comparison eligibility disagrees with the fixed-stage analysis")
    stats = {}
    for label, key in (("VACF", "mean_D_VACF_2ps"), ("MSD", "mean_D_MSD_1_2ps")):
        values = np.asarray([row[key] for row in chains])
        summary = independent_summary(values)
        recorded = report[f"diffusion_{label}_angstrom2_ps"]
        if recorded["chains"] != 8 or recorded["degrees_of_freedom"] != 7:
            raise ValueError("reported uncertainty must use eight chain means and seven degrees of freedom")
        if any(not np.isclose(summary[name], recorded[name], rtol=1e-11, atol=1e-13) for name in summary):
            raise ValueError("reported uncertainty is inconsistent with the independent-chain means")
        stats[label] = {"values": values, **summary}
    curves = {}
    with np.load(curves_path, allow_pickle=False) as data:
        for row in chains:
            prefix = f"seed_{row['seed']}_"
            values = {name: np.asarray(data[prefix + name], dtype=float) for name in (
                "lag_time_ps", "mean_msd_angstrom2", "mean_D_VACF_angstrom2_ps")}
            lag = values["lag_time_ps"]
            if (lag.ndim != 1 or len(lag) < 3 or lag[0] != 0 or not np.all(np.diff(lag) > 0)
                    or not np.allclose(np.diff(lag), row["config"]["record_stride"] * row["config"]["dt_fs"] / 1000,
                                       rtol=1e-11, atol=1e-13)
                    or any(value.shape != lag.shape or not np.isfinite(value).all() for value in values.values())
                    or not np.isclose(lag[-1], 2.0, atol=1e-12, rtol=0)):
                raise ValueError("invalid or nonuniform compact transport curve")
            if not np.isclose(values["mean_D_VACF_angstrom2_ps"][-1], row["mean_D_VACF_2ps"], rtol=1e-11, atol=1e-13):
                raise ValueError("VACF curve endpoint and reported chain mean disagree")
            if curves and not np.array_equal(lag, next(iter(curves.values()))["lag_time_ps"]):
                raise ValueError("all chain curves must use the same lag grid")
            curves[row["seed"]] = values
    if sha256(curves_path) != curves_sha:
        raise RuntimeError("compact transport curves changed while being read")
    return {"report": report, "protocol": protocol, "chains": chains, "curves": curves,
            "stats": stats, "paper_allowed": paper_allowed,
            "input_sha256": {"report.json": report_sha, "curves.npz": curves_sha,
                             "protocol.json": protocol_sha}}


def pointwise_interval(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] != 8 or not np.isfinite(values).all():
        raise ValueError("pointwise bands require eight finite chain-mean curves")
    return values.mean(axis=0), student_t.ppf(.975, 7) * values.std(axis=0, ddof=1) / np.sqrt(8)


def plot(directory, protocol_path, output):
    data = load_complete_cohort(directory, protocol_path)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report, protocol, chains, curves = (data[name] for name in ("report", "protocol", "chains", "curves"))
    synthetic = report.get("synthetic_data") is True
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = [row["seed"] for row in chains]
    lag = curves[seeds[0]]["lag_time_ps"]
    gk_curves = np.stack([curves[seed]["mean_D_VACF_angstrom2_ps"] for seed in seeds])
    msd_curves = np.stack([curves[seed]["mean_msd_angstrom2"] for seed in seeds])
    gk, gk_half = pointwise_interval(gk_curves)
    msd, msd_half = pointwise_interval(msd_curves)
    u_difference = np.asarray([row["sampling"]["potential_per_molecule_hartree"]["half_mean_difference"]
                               / KCAL_MOL_TO_AU for row in chains])
    u_threshold = protocol["predeclared_eligibility_checks"]["max_split_mean_potential_difference_kcal_mol_water"]
    failed = [key for key, value in report["eligibility_checks"].items() if value is not True]
    names = {"sampling_potential_per_molecule_hartree": "preparation energy",
             "sampling_temperature_momenta_K": "preparation temperature", "sampling_mean_OH_angstrom": "preparation OH",
             "sampling_mean_HOH_degrees": "preparation HOH", "energy_envelope": "NVE energy envelope",
             "energy_linear_drift": "NVE drift", "precision": "interval width", "paired_equivalence": "paired equivalence",
             "MSD_VACF_compatibility": "MSD–VACF compatibility", "VACF_cutoff_compatibility": "VACF cutoff compatibility",
             "MSD_window_compatibility": "MSD-window compatibility"}
    failed_labels = [names.get(key, key) for key in failed]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "pyrpmd-water-fixed-eight-chain-cohort"}):
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.5))
        fig.subplots_adjust(left=.085, right=.98, top=.82, bottom=.17, hspace=.42, wspace=.29)
        fig.suptitle("q-TIP4P/F classical diffusion — fixed eight-chain cohort", x=.085, y=.97, ha="left", fontsize=19, weight="bold")
        config = chains[0]["config"]
        fig.text(.085, .925, f"{config['cells_per_axis']**3} H2O · {config['temperature_K']:g} K · P = 1 · full PME · "
                 f"{config['dt_fs']:g} fs · external OpenMM reference", color="#4e5e70", fontsize=11)
        status = ("SYNTHETIC DATA — LAYOUT/ANALYSIS CHECK ONLY" if synthetic else
                  "Fixed cohort complete; predeclared eligibility checks failed" if failed else
                  "Fixed numerical checks passed; physical convergence and reproduction remain uncertified")
        fig.text(.085, .88, status, color="#a8342c" if failed or synthetic else "#354f66", fontsize=11, weight="bold")
        ax = axes[0, 0]
        x = np.arange(8)
        for label, offset, color, marker in (("VACF", -.10, "#1769a3", "o"), ("MSD", .10, "#c37824", "s")):
            stats = data["stats"][label]
            ax.scatter(x + offset, stats["values"], color=color, marker=marker, s=27, label=f"{label}: chain means")
            ax.errorbar([9 + offset], [stats["mean"]], yerr=[stats["CI95_halfwidth"]], color=color,
                        marker=marker, capsize=4, markersize=6)
        if data["paper_allowed"]:
            reference = protocol["literature_comparison"]
            ax.axhspan(reference["reference"]-reference["standard_error"], reference["reference"]+reference["standard_error"],
                       color="#7e6f93", alpha=.15, label="Paper value ± reported 1 SE")
            ax.axhline(reference["reference"], color="#7e6f93", ls=":", lw=1)
        ax.set_xticks([*x, 9], [*[str(seed) for seed in seeds], "All 8"], rotation=45, ha="right")
        ax.set_ylabel("Diffusion estimate (Å²/ps)")
        ax.set_title("A  Equal weight per seed; aggregate bars = 95% CI", loc="left", fontsize=11)
        ax.legend(fontsize=8, loc="best", frameon=False)
        ax = axes[0, 1]
        ax.plot(lag, gk_curves.T, color="#b6c6d3", lw=.7, alpha=.65)
        ax.fill_between(lag, gk-gk_half, gk+gk_half, color="#1769a3", alpha=.16)
        ax.plot(lag, gk, color="#1769a3", lw=2)
        ax.axvline(2, color="#555f6a", lw=.8, ls="--")
        ax.set(xlabel="VACF integration cutoff (ps)", ylabel="Running integral / 3 (Å²/ps)", xlim=(0, 2.05))
        ax.set_title("B  Fixed 2 ps primary cutoff; pointwise 95% band", loc="left", fontsize=11)
        ax = axes[1, 0]
        ax.plot(lag, msd_curves.T, color="#d8c5ad", lw=.7, alpha=.65)
        ax.fill_between(lag, msd-msd_half, msd+msd_half, color="#c37824", alpha=.16)
        ax.plot(lag, msd, color="#c37824", lw=2)
        ax.axvspan(1, 2, color="#526577", alpha=.10, label="Registered 1–2 ps fit window")
        ax.set(xlabel="Lag time (ps)", ylabel="Molecular COM MSD (Å²)", xlim=(0, 2))
        ax.set_title("C  MSD of molecular centers of mass", loc="left", fontsize=11)
        ax.legend(fontsize=8, loc="upper left", frameon=False)
        ax = axes[1, 1]
        ax.axvspan(-u_threshold, u_threshold, color="#437b68", alpha=.13, label="Predeclared ± threshold")
        ax.axvline(0, color="#66717b", lw=.7)
        colors = ["#a8342c" if abs(value) > u_threshold else "#356b8c" for value in u_difference]
        ax.barh(np.arange(8), u_difference, color=colors, height=.65)
        ax.set_yticks(np.arange(8), [str(seed) for seed in seeds]); ax.invert_yaxis()
        extent = max(u_threshold*1.4, float(np.max(abs(u_difference)))*1.18)
        ax.set(xlabel="U(75–100 ps) − U(50–75 ps) (kcal/mol/water)", xlim=(-extent, extent))
        ax.set_title("D  Original preparation gate; failed seeds retained", loc="left", fontsize=11)
        ax.legend(fontsize=8, loc="best", frameon=False)
        for ax in axes.flat:
            ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
        footer = ("Uncertainty unit: 8 independently seeded chain means, df = 7; "
                  f"{protocol['fixed_stage']['NVE_segments_per_chain']} adjacent NVE segments per seed are not independent replicas.\n"
                  "Bands in B/C are pointwise, not simultaneous. All registered segments are included. "
                  "The original protocol and preparation failures are unchanged.")
        fig.text(.085, .09, footer, fontsize=9, color="#4e5e70", va="top", linespacing=1.5)
        if failed:
            fig.text(.085, .035, "Unmet checks: " + "; ".join(failed_labels) + ". Paper comparison withheld.",
                     fontsize=8.5, color="#a8342c", va="bottom", wrap=True)
        else:
            fig.text(.085, .035, "Paper: Habershon et al., JCP 131, 024501 (2009), DOI 10.1063/1.3167790. "
                     "Its shaded uncertainty is 1 SE; our aggregate error bars are 95% Student-t intervals.", fontsize=8.2, color="#4e5e70")
        png, svg = output / "fixed-cohort-transport.png", output / "fixed-cohort-transport.svg"
        fig.savefig(png, dpi=170, facecolor="white")
        fig.savefig(svg, facecolor="white", metadata={"Date": None})
        plt.close(fig)
    table = output / "chain-summary.csv"
    with table.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "complete_NVE_segments", "D_VACF_2ps_angstrom2_ps", "D_MSD_1_2ps_angstrom2_ps",
                         "preparation_delta_U_kcal_mol_water", "preparation_U_gate_passed", "synthetic_data"])
        for index, row in enumerate(chains):
            writer.writerow([row["seed"], len(row["segments"]), row["mean_D_VACF_2ps"], row["mean_D_MSD_1_2ps"],
                             u_difference[index], abs(u_difference[index]) <= u_threshold, synthetic])
    provenance = {"schema": "pyrpmd.water-complete-cohort-figure.v1", "synthetic_data": synthetic,
                  "external_OpenMM_reference": True, "pyrpmd_production_trajectory": False,
                  "all_eight_fixed_chains_complete": True, "independence_unit": "seed-chain mean", "degrees_of_freedom": 7,
                  "paper_comparison_enabled": data["paper_allowed"], "physical_convergence_certified_by_plot": False,
                  "failed_predeclared_checks": failed, "input_sha256": data["input_sha256"],
                  "plot_source_sha256": sha256(Path(__file__)), "matplotlib_version": matplotlib.__version__,
                  "analysis_source_sha256": report["source_sha256"],
                  "artifacts": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in (png, svg, table)}}
    atomic_json(output / "figure-provenance.json", provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.analysis, args.protocol, args.output), indent=2))


if __name__ == "__main__":
    main()
