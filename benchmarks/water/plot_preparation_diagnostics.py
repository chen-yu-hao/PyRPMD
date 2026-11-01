#!/usr/bin/env python3
"""Show every original preparation scalar and RDF after all eight chains finish.

This is an offline supplement to plot_native_campaign, with its complete-cohort
and independent-chain identity checks. Original scalar thresholds are unchanged.
RDF differences and pointwise intervals are descriptive, not an equilibrium or
simultaneous equivalence test. No sampler, trajectory or random state is edited.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.water.campaign import atomic_json, sha256
from benchmarks.water.plot_native_campaign import load_complete_cohort, pointwise_interval
from pyrpmd.constants import KCAL_MOL_TO_AU

SCALARS = (
    ("potential_per_molecule_hartree", "Potential energy", "kcal/mol/water",
     "max_split_mean_potential_difference_kcal_mol_water", 1 / KCAL_MOL_TO_AU),
    ("temperature_momenta_K", "Momentum temperature", "K",
     "max_split_mean_temperature_difference_K", 1.0),
    ("mean_OH_angstrom", "Mean O–H length", "Å",
     "max_split_mean_OH_difference_angstrom", 1.0),
    ("mean_HOH_degrees", "Mean H–O–H angle", "degrees",
     "max_split_mean_angle_difference_degrees", 1.0),
)


def load_diagnostics(directory, protocol_path):
    data = load_complete_cohort(directory, protocol_path)
    rows, limits = [], data["protocol"]["predeclared_eligibility_checks"]
    for chain in data["chains"]:
        config = chain["config"]
        expected = round(25000 / (config["equilibration_record_stride"] * config["dt_fs"]))
        for key, label, unit, threshold_key, scale in SCALARS:
            record = chain["sampling"][key]
            means = np.asarray(record["last_50ps_half_means"], dtype=float)
            difference = record["half_mean_difference"]
            if (means.shape != (2,) or not np.isfinite(means).all()
                    or difference is None or not np.isfinite(difference)
                    or not np.isclose(means[1] - means[0], difference, rtol=1e-10, atol=1e-12)
                    or record["half_window_observation_counts"] != [expected, expected]
                    or record["expected_observations_per_half_window"] != expected):
                raise ValueError("each fixed scalar window must contain all registered finite observations")
            threshold = float(limits[threshold_key])
            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError("the original scalar threshold must be finite and positive")
            rows.append({"seed": chain["seed"], "observable": key, "label": label, "unit": unit,
                         "mean_50_75_ps": float(means[0] * scale),
                         "mean_75_100_ps": float(means[1] * scale),
                         "late_minus_early": float(difference * scale),
                         "original_absolute_threshold": threshold,
                         "original_scalar_gate_passed": bool(abs(difference * scale) <= threshold)})
    curves_path = Path(directory) / "curves.npz"
    early, late, radius = [], [], None
    with np.load(curves_path, allow_pickle=False) as curves:
        for chain in data["chains"]:
            prefix = f"seed_{chain['seed']}_"
            r, first, second = (np.asarray(curves[prefix + name], dtype=float) for name in
                                ("rdf_r_angstrom", "rdf_early_late", "rdf_late"))
            if (r.ndim != 1 or len(r) < 3 or not np.isfinite(r).all()
                    or r[0] <= 0 or not np.all(np.diff(r) > 0)
                    or any(g.shape != r.shape or not np.isfinite(g).all() or np.any(g < 0)
                           for g in (first, second))
                    or (radius is not None and not np.array_equal(radius, r))):
                raise ValueError("all eight finite, nonnegative RDFs must use one identical radial grid")
            radius = r
            early.append(first)
            late.append(second)
    if sha256(curves_path) != data["input_sha256"]["curves.npz"]:
        raise RuntimeError("compact curves changed while reading preparation diagnostics")
    data.update(scalar_rows=rows, rdf_radius=radius, rdf_early=np.asarray(early), rdf_late=np.asarray(late))
    return data


def plot(directory, protocol_path, output):
    data = load_diagnostics(directory, protocol_path)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = [row["seed"] for row in data["chains"]]
    synthetic = data["report"].get("synthetic_data") is True
    banner = "SYNTHETIC DATA — DISPLAY CHECK ONLY" if synthetic else "Original fixed cohort; all eight seeds and original preparation failures retained"
    scalar_csv = output / "preparation-scalars.csv"
    with scalar_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data["scalar_rows"][0]))
        writer.writeheader()
        writer.writerows(data["scalar_rows"])
    r, early, late = (data[key] for key in ("rdf_radius", "rdf_early", "rdf_late"))
    statistics = {name: pointwise_interval(values) for name, values in
                  (("early", early), ("late", late), ("paired_difference", late - early))}
    rdf_csv = output / "preparation-rdf.csv"
    with rdf_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        names = ["radius_angstrom"]
        names += [f"seed_{seed}_{period}" for period in ("g_50_75ps", "g_75_100ps", "late_minus_early") for seed in seeds]
        names += [f"{period}_{stat}" for period in statistics for stat in ("mean", "pointwise_CI95_halfwidth")]
        writer.writerow(names)
        for i in range(len(r)):
            writer.writerow([r[i], *early[:, i], *late[:, i], *(late - early)[:, i],
                             *(array[i] for pair in statistics.values() for array in pair)])
    artifacts = [scalar_csv, rdf_csv]
    style = {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
             "svg.hashsalt": "pyrpmd-water-preparation-complete-eight"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
        fig.subplots_adjust(left=.085, right=.98, top=.81, bottom=.12, hspace=.48, wspace=.31)
        fig.suptitle("q-TIP4P/F — original preparation diagnostics", x=.085, y=.97,
                     ha="left", fontsize=19, weight="bold")
        config = data["chains"][0]["config"]
        fig.text(.085, .925, f"{config['cells_per_axis']**3} H2O · {config['temperature_K']:g} K · "
                 "classical P = 1 · 100 ps preparation · external OpenMM reference", color="#4e5e70")
        fig.text(.085, .875, banner, color="#a8342c", weight="bold")
        for panel, (key, label, unit, _, _), ax in zip("ABCD", SCALARS, axes.flat):
            rows = [row for row in data["scalar_rows"] if row["observable"] == key]
            values = np.asarray([row["late_minus_early"] for row in rows])
            threshold = rows[0]["original_absolute_threshold"]
            ax.axvspan(-threshold, threshold, color="#437b68", alpha=.14)
            ax.axvline(0, color="#66717b", lw=.7)
            ax.barh(np.arange(8), values, height=.64,
                    color=["#356b8c" if row["original_scalar_gate_passed"] else "#a8342c" for row in rows])
            extent = max(threshold * 1.35, float(np.max(abs(values))) * 1.16)
            ax.set_yticks(np.arange(8), [str(seed) for seed in seeds]); ax.invert_yaxis()
            ax.set(xlim=(-extent, extent), xlabel=f"75–100 ps minus 50–75 ps ({unit})")
            ax.set_title(f"{panel}  {label}: original threshold ±{threshold:g}", loc="left", fontsize=11)
            ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)
        fig.text(.085, .055, "Each bar is one fixed seed. Shading is the unchanged original absolute threshold.\n"
                 "Passing this scalar screen does not prove equilibrium; no seed or time window is selected or replaced.",
                 fontsize=9, color="#4e5e70", linespacing=1.5, va="top")
        for suffix in ("png", "svg"):
            path = output / f"preparation-scalars.{suffix}"
            fig.savefig(path, dpi=170, facecolor="white", **({"metadata": {"Date": None}} if suffix == "svg" else {}))
            artifacts.append(path)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.5))
        fig.subplots_adjust(left=.075, right=.98, top=.72, bottom=.21, wspace=.28)
        fig.suptitle("q-TIP4P/F — all-eight preparation RDF diagnostics", x=.075, y=.96,
                     ha="left", fontsize=18, weight="bold")
        fig.text(.075, .90, "Oxygen–oxygen RDF · P = 1 · original fixed windows · external OpenMM reference", color="#4e5e70")
        fig.text(.075, .84, banner, color="#a8342c", weight="bold", fontsize=10)
        for values, name, label, color in ((early, "early", "50–75 ps", "#1769a3"),
                                           (late, "late", "75–100 ps", "#c37824")):
            mean, half = statistics[name]
            axes[0].plot(r, values.T, color=color, lw=.5, alpha=.11)
            axes[0].fill_between(r, mean-half, mean+half, color=color, alpha=.13)
            axes[0].plot(r, mean, color=color, lw=1.7, label=label)
        axes[0].set(ylabel="gOO(r)", title="A  Equal weight per seed; all curves retained")
        axes[0].legend(frameon=False, fontsize=9)
        mean, half = statistics["paired_difference"]
        axes[1].plot(r, (late - early).T, color="#8195a8", lw=.65, alpha=.45)
        axes[1].fill_between(r, mean-half, mean+half, color="#3b586f", alpha=.17)
        axes[1].plot(r, mean, color="#2e485e", lw=1.6)
        axes[1].axhline(0, color="#555f6a", lw=.8, ls="--")
        axes[1].set(ylabel="gOO(75–100 ps) − gOO(50–75 ps)", title="B  Paired differences across the same eight seeds")
        for ax in axes:
            ax.set(xlabel="O–O separation (Å)", xlim=(0, r[-1]))
            ax.grid(alpha=.15); ax.set_axisbelow(True)
        fig.text(.075, .105, "Bands: nominal pointwise 95% Student-t intervals over 8 chain means (df = 7), not simultaneous bounds.\n"
                 "These curves are descriptive: no RDF threshold, equivalence test, equilibrium certification or diffusion claim is added.\n"
                 "All radial bins and per-seed curves are retained in preparation-rdf.csv; overlapping bands do not establish equivalence.",
                 fontsize=9, color="#4e5e70", linespacing=1.5, va="top")
        for suffix in ("png", "svg"):
            path = output / f"preparation-rdf.{suffix}"
            fig.savefig(path, dpi=170, facecolor="white", **({"metadata": {"Date": None}} if suffix == "svg" else {}))
            artifacts.append(path)
        plt.close(fig)
    provenance = {"schema": "pyrpmd.water-original-preparation-figures.v1", "synthetic_data": synthetic,
                  "external_OpenMM_reference": True, "pyrpmd_production_trajectory": False,
                  "all_eight_fixed_chains_complete": True, "fixed_seeds": seeds,
                  "scalar_thresholds_changed": False, "RDF_hard_gate_adopted": False,
                  "diffusion_evidence": False, "physical_convergence_certified": False,
                  "uncertainty_unit": "seed-chain mean", "degrees_of_freedom": 7,
                  "RDF_intervals": "nominal pointwise 95% Student-t intervals; not simultaneous or an equivalence test",
                  "input_sha256": data["input_sha256"], "plot_source_sha256": sha256(Path(__file__)),
                  "complete_cohort_helper_sha256": sha256(ROOT / "benchmarks/water/plot_native_campaign.py"),
                  "matplotlib_version": matplotlib.__version__,
                  "artifacts": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in artifacts}}
    atomic_json(output / "preparation-figure-provenance.json", provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.directory, args.protocol, args.output)


if __name__ == "__main__":
    main()
