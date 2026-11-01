"""Compare saved multichain NRPMD statistics with the independent grid reference.

Run after ``adaptive_nrpmd_study.py`` and ``adaptive_nrpmd_reference.py``.
No dynamics or sampling is rerun. JSON/CSV comparisons use the largest-box
reference for each *matching finite bead count*. Optional figures require the
``analysis`` extra (matplotlib). An unavailable production estimate stays
unavailable even if exploratory bootstrap draws happened to remain finite.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(study_directory, reference_file):
    directory, reference_file = Path(study_directory), Path(reference_file)
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    reference = json.loads(reference_file.read_text())
    targets = {(r["potential_V"], r["nbeads"]): r for r in reference["preferred_records"]}
    grids = {(r["potential_V"], r["nbeads"]): r for r in reference["convergence"]["cases"]}
    rows, cases, report_hashes = [], [], {}
    for entry in summary["cases"]:
        report_path = directory / entry["report"]
        report = json.loads(report_path.read_text())
        report_hashes[entry["report"]] = hashlib.sha256(report_path.read_bytes()).hexdigest()
        key = (entry["potential_V"], entry["beads"])
        if (key != (report["potential_V"], report["beads"])
                or entry["point_id"] != report["point_id"]):
            raise ValueError(f"summary/report point identity differs for {key}")
        if any(entry[name] != report[name] for name in ("observables", "bootstrap", "production_reweighting")):
            raise ValueError(f"summary/report sampling results differ for {key}")
        if key not in targets:
            raise ValueError(f"no matching finite-bead reference for {key}")
        target, grid = targets[key], grids[key]
        if target["model_scope"] != report["model_scope"]:
            raise ValueError(f"reference/study model differs for {key}")
        if any(report["source_identity"]["files"].get(name) != digest
               for name, digest in target["model_source_identity"]["files"].items()):
            raise ValueError(f"reference/study model source hashes differ for {key}")
        if (report["config"]["beta"] != target["beta_au"]
                or report["config"]["mass"] != target["mass_au"]
                or report["reservoir_chemical_potential_au"] != target["chemical_potential_au"]):
            raise ValueError(f"thermodynamic parameters differ for {key}")
        estimates, bootstrap = entry["observables"], entry["bootstrap"]
        available = estimates is not None and entry["production_reweighting"]["available"]
        intervals = available and bootstrap["confidence_intervals_available"]
        for index, name in enumerate(report["observable_names"]):
            ref = target["observables"][name]
            estimate = estimates[name] if available else None
            error = bootstrap["standard_error"][index] if intervals else None
            lower = bootstrap["percentile_interval"][0][index] if intervals else None
            upper = bootstrap["percentile_interval"][1][index] if intervals else None
            rows.append({
                "point_id": entry["point_id"], "potential_V": key[0], "beads": key[1],
                "observable": name, "estimate": estimate, "standard_error": error,
                "ci95_lower": lower, "ci95_upper": upper,
                "finite_bead_reference": ref,
                "difference": None if estimate is None else estimate-ref,
                "reference_in_pointwise_ci95": None if lower is None else lower <= ref <= upper,
                "reference_spacing_change": grid["fine_minus_coarse"][name],
                "reference_last_domain_change": grid["farther_minus_wider"][name],
                "estimate_available": available, "ci_available": intervals,
            })
        cases.append({
            "point_id": entry["point_id"], "potential_V": key[0], "beads": key[1],
            "available": available, "confidence_intervals_available": intervals,
            "unavailable_reasons": entry["production_reweighting"]["unavailable_reasons"],
            "average_sign": entry["production_reweighting"]["average_sign"],
            "positive_effective_sample_diagnostic": entry["production_reweighting"]["correlation_adjusted_ess_diagnostic"],
            "signed_effective_sample_diagnostic": entry["production_reweighting"]["signed_ess_diagnostic"],
            "path_stopping_reason": entry["path_stopping_reason"],
            "initial_centers": report["path"]["initial_centers"],
            "final_centers": report["path"]["centers"],
            "charge": entry["charge"], "dynamics": report["dynamics"],
        })
    result = {
        "schema_version": 1, "cases": cases, "comparisons": rows,
        "input_sha256": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (summary_path, reference_file)},
        "report_sha256": report_hashes,
        "reference_scope": "same_finite_bead_equilibrium_measure; mapping_integrated_analytically",
        "uncertainty_scope": "pointwise_source_chain_block_bootstrap_refitting_MBAR_conditional_on_frozen_path",
        "reference_error_scope": "observed_grid_and_domain_sensitivity_not_rigorous_error_bounds",
        "bead_convergence_at_U0": {
            "eight_minus_four": reference["convergence"]["eight_minus_four"],
            "sixteen_minus_eight": reference["convergence"]["sixteen_minus_eight"],
            "scope": "independent_reference_on_common_coarse_grid_and_domain",
        },
        "physical_converged": False, "rate_constant": None,
    }
    (directory / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (directory / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return result


def plot_comparison(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none",
                         "svg.hashsalt": "pyrpmd-adaptive-nrpmd-study"})
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.4), constrained_layout=True)
    cases, rows = result["cases"], result["comparisons"]
    beads = sorted({c["beads"] for c in cases})
    voltages = sorted({c["potential_V"] for c in cases})
    colors = {p: color for p, color in zip(beads, ("#2166ac", "#1b9e77", "#c44e52", "#8172b2"))}
    at_zero = [c for c in cases if c["potential_V"] == 0.]
    for c in at_zero:
        centers = np.asarray(c["final_centers"])
        label = f"P={c['beads']}"
        if c["path_stopping_reason"] == "unresolved_pilot_sign":
            label += " (pilot unresolved)"
        axes[0, 0].plot(*centers.T, "o-", color=colors[c["beads"]], label=label)
    if at_zero:
        axes[0, 0].plot(*np.asarray(at_zero[0]["initial_centers"]).T, "k--", lw=1, label="initial")
    axes[0, 0].set(title="Frozen production windows at U=0 V", xlabel="x center / bohr", ylabel="y center / bohr")
    for axis, name, title, label in ((axes[0, 1], "x", "Mean nuclear coordinate", "mean x / bohr"),
                                    (axes[0, 2], "electron_number_bead", "Equilibrium electron number", "mean N")):
        for p in beads:
            selected = sorted((r for r in rows if r["beads"] == p and r["observable"] == name), key=lambda r: r["potential_V"])
            axis.plot([r["potential_V"] for r in selected], [r["finite_bead_reference"] for r in selected],
                      "-", color=colors[p], alpha=.65, label=f"P={p} grid")
            for r in selected:
                if r["estimate_available"]:
                    axis.scatter(r["potential_V"], r["estimate"], color=colors[p], s=32, zorder=3)
                    if r["ci_available"]:
                        axis.vlines(r["potential_V"], r["ci95_lower"], r["ci95_upper"], color=colors[p], lw=1.5)
        axis.set(title=title, xlabel="model potential / V", ylabel=label)
    for name, marker, label in (("x2_bead", "o", "bead x squared"), ("x2_centroid", "s", "centroid x squared")):
        selected = sorted((r for r in rows if r["potential_V"] == 0. and r["observable"] == name), key=lambda r: r["beads"])
        axes[1, 0].plot([r["beads"] for r in selected], [r["finite_bead_reference"] for r in selected], marker+"--", label=label+" grid")
        for r in selected:
            if r["estimate_available"]:
                axes[1, 0].scatter(r["beads"], r["estimate"], color="black", marker=marker, s=25)
                if r["ci_available"]:
                    axes[1, 0].vlines(r["beads"], r["ci95_lower"], r["ci95_upper"], color="black")
    axes[1, 0].set(title="Finite-bead statistics at U=0 V", xlabel="beads P", ylabel="second moment / bohr squared")
    axes[1, 0].set_xticks(beads)
    for voltage in voltages:
        selected = sorted((c for c in cases if c["potential_V"] == voltage), key=lambda c: c["beads"])
        axes[1, 1].plot([c["beads"] for c in selected], [c["average_sign"] for c in selected], "o-", label=f"U={voltage:+g} V")
        for c in selected:
            if not c["available"]:
                axes[1, 1].scatter(c["beads"], c["average_sign"], marker="x", s=90, color="black", zorder=5)
    axes[1, 1].set(title="SEO sign (x: production gate failed)", xlabel="beads P", ylabel="weighted average sign", ylim=(0, 1.05))
    axes[1, 1].set_xticks(beads)
    for c in cases:
        dynamic = c["dynamics"]
        if dynamic["available"]:
            axes[1, 2].loglog(dynamic["dts"], dynamic["maximum_absolute_hamiltonian_drift"], "o-",
                              color=colors[c["beads"]], alpha=.65)
    axes[1, 2].set(title="Same-parent timestep check", xlabel="timestep / atomic time", ylabel="maximum Hamiltonian drift / hartree")
    for axis in axes.ravel():
        axis.grid(alpha=.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8, frameon=False)
    figure.suptitle("Adaptive constant-potential thermal NRPMD: analytic model\nPoints: sampled estimates; bars: pointwise 95% block-bootstrap intervals", fontsize=14)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    svg_path = output / "study-comparison.svg"
    figure.savefig(svg_path, metadata={"Date": None})
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
                        encoding="utf-8")
    figure.savefig(output / "study-comparison.png", dpi=160)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    result = compare(args.study, args.reference)
    if args.plot:
        plot_comparison(result, args.study / "figures")


if __name__ == "__main__":
    main()
