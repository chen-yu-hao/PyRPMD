#!/usr/bin/env python3
"""Generate review tables from immutable saved sampling evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def load(path):
    return json.loads(path.read_text())


def number(value, digits=3):
    return "undefined" if value is None else f"{value:.{digits}g}"


def interval(values):
    values = np.asarray(values, float)
    if len(values) < 2:
        return number(values[0]) if len(values) else "undefined"
    return f"{values.mean():.4g} ± {t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)):.3g}"


def table(lines, header, rows):
    def cell(value):
        return str(value).replace("|", "\\|")
    lines.append("| " + " | ".join(map(cell, header)) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    lines.append("")


def main(output):
    output = Path(output)
    all_results = [load(path) for path in sorted((output/"opes_v2").glob("*/metrics.json"))]
    baseline = [r for r in all_results if r["dt"] == .005 and r["initial_sigma"] == .1]
    lines = ["# Sampling validation: saved numerical results", "",
             "Generated from saved raw trajectories and per-run metrics by `report_sampling.py`. "
             "See [README](README.md) for method/reference scope and the [source map](source_map.json) for primary sources. "
             "Intervals below are 95% Student-t intervals across the stated independent samples; they do not establish physical convergence.", ""]
    ref = load(output/"reference.json")
    lines += ["## Published numerical reference", ""]
    table(lines, ["Basin definition", "Published printed value / kBT", "Computed quadrature / kBT"], [
        ["x<0 versus x>0", ref["published_delta_f_x_printed"], f"{ref['delta_f_x']:.12f}"],
        ["y>x+1.3 versus y<x+1.3", ref["published_delta_f_oblique_printed"], f"{ref['delta_f_oblique']:.12f}"]])
    lines += ["The printed parentheses are retained literally; no statistical uncertainty interpretation is imposed. "
              "These reference values are integrations of the authors' potential, not measurements from the biased trajectories.", ""]
    lines += ["## Live PLUMED fixed-width comparison", ""]
    comparator = load(output/"plumed_comparator/summary.json")
    table(lines, ["Mode", "Probes", "Maximum |ΔV|", "Maximum |Δforce|", "Legacy maximum |ΔV|"], [
        [r["mode"], r["depositions"]*r["probe_points_per_deposition"], number(r["max_energy_difference"]),
         number(r["max_force_difference"]), number(r["legacy_v1_max_energy_difference"])] for r in comparator])
    lines += ["Both new initial biases equal PLUMED's −20. Direct E/F calls use identical centers, literal physical sigma 0.1, "
              "no compression, and no interpolation. This is implementation equivalence for the stated restricted settings.", "",
              "![PLUMED initialization comparison](figures/plumed_initialization_regression.png)", ""]

    if baseline:
        lines += ["## Müller trajectory campaign", "",
                  f"**{len(baseline)}/32 primary trajectories completed.** Every trajectory contains80000 adaptation steps "
                  "and40000 frozen steps at dt0.005, with160 updates. The primary campaign uses four independent seeds per CV/variant. "
                  "Each frozen estimate uses3200 retained frames before conditioning on the published rectangle.", ""]
        rows, basin_rows = [], []
        seeds = (57859, 28661, 38795, 53730)
        for cv in "xy":
            for mode, adaptive in (("metad", False), ("explore", False), ("metad", True), ("explore", True)):
                selected = sorted([r for r in baseline if (r["cv"], r["mode"], r["adaptive_sigma"]) == (cv, mode, adaptive)],
                                  key=lambda r: seeds.index(r["seed"]))
                if not selected:
                    continue
                label = mode+(" Welford" if adaptive else " fixed")
                rows.append([cv, label, len(selected),
                    ", ".join(str(r["transitions_adapting"]) for r in selected),
                    interval([r["direct_FES_RMSE_kBT"] for r in selected]),
                    interval([r["JS_unbiased_reweighted"] for r in selected]),
                    interval([r["JS_sample_to_exact_frozen_ensemble"] for r in selected]),
                    f"{min(r['reweighting_Kish_ESS'] for r in selected):.0f}–{max(r['reweighting_Kish_ESS'] for r in selected):.0f}"])
                finite = [r["frozen_delta_F_x_kBT"] for r in selected if r["frozen_delta_F_x_kBT"] is not None]
                basin_rows.append([cv, label,
                    "; ".join(number(r["frozen_delta_F_x_kBT"], 5) for r in selected),
                    "; ".join(number(r["frozen_delta_F_oblique_kBT"], 5) for r in selected),
                    f"{len(finite)}/{len(selected)}",
                    interval(finite) if len(finite) == len(selected) else "not estimable across all seeds"])
        table(lines, ["CV", "Variant", "Seeds", "Adaptation crossings, per seed", "Direct FES RMSE / kBT", "Reweighted JS", "JS vs exact frozen ensemble", "Weight-only ESS range"], rows)
        lines += ["RMSE is evaluated where the exact F−Fmin≤10 kBT, with an optimized additive constant. "
                  "JS uses natural logarithms and is bounded by log2≈0.693. Small global JS can conceal a missed rare basin; "
                  "ESS measures weights only and ignores autocorrelation.", ""]
        table(lines, ["CV", "Variant", "ΔFₓ per seed / kBT", "Oblique ΔF per seed / kBT", "Finite ΔFₓ", "All-seed ΔFₓ interval"], basin_rows)
        lines += ["Seed order in both tables is57859,28661,38795,53730. Undefined means one required basin had zero retained "
                  "weighted population. Finite values from only a subset are not substituted for the missing runs. "
                  "The large variation, undefined basin estimates and finite-bias mixing errors rule out a convergence claim at this budget.", "",
                  "![OPES FES and crossings](figures/opes_fes_and_crossings.png)", "",
                  "![OPES basin free energies](figures/opes_basin_free_energies.png)", ""]
        lines += [f"Across the primary trajectories the largest audited spline energy error is {max(r['spline_max_energy_error_kBT'] for r in baseline):.3g} kBT "
                  f"and force error is {max(r['spline_max_force_error'] for r in baseline):.3g} model units. "
                  "These are sampled grid-midpoint audits, not a uniform mathematical bound.", ""]

    path = output/"sensitivity_harmonized.json"
    if path.exists():
        lines += ["## Timestep and bandwidth sensitivity", "",
                  "Two seeds were declared before running this check. Adaptation time400, frozen time200, update interval2.5, "
                  "saved-frame interval0.05 and discarded frozen time40 are held fixed. Half-timestep trajectories therefore use "
                  "twice as many MD steps. The separate postanalysis harmonizes the physical relaxation window.", ""]
        table(lines, ["Seed", "dt", "σ", "Crossings", "Direct FES RMSE / kBT", "Harmonized ΔFₓ", "Reweighted JS"], [
            [r["seed"], r["dt"], r["sigma"], r["transitions_adapting"], number(r["direct_FES_RMSE_kBT"],4),
             number(r["frozen_delta_F_x_kBT"],5), number(r["JS_unbiased_reweighted"],4)] for r in load(path)])
        lines += ["This test exposes finite-trajectory/width sensitivity; two stochastic runs do not demonstrate timestep convergence. "
                  "No favorable parameter choice is selected as the primary result after viewing these outcomes.", ""]

    path = output/"replica_exchange/summary.json"
    if path.exists():
        rows = load(path)
        lines += ["## Replica-exchange thermal ensembles", "",
                  "Four seeds, three thermodynamic slots, P=4, dt0.02, 24000 MD steps per slot, with20% discarded. "
                  "Reference values are exact Gaussian moments of the finite-P harmonic Hamiltonian, not continuum quantum limits.", ""]
        table(lines, ["Exchange", "Slot", "Observable", "Exact", "Four-seed mean ±95% interval", "Relative error", "Reference in interval"], [
            [r["scheme"], r["slot"], r["observable"], number(r["reference"],5),
             f"{r['mean']:.5g} ± {r['95CI_halfwidth']:.3g}", f"{100*(r['mean']/r['reference']-1):+.2f}%",
             "yes" if abs(r["mean"]-r["reference"]) <= r["95CI_halfwidth"] else "no"] for r in rows])
        covered = sum(abs(r["mean"]-r["reference"]) <= r["95CI_halfwidth"] for r in rows)
        lines += [f"{covered}/{len(rows)} references lie within their individual95% intervals. The remaining discrepancy is reported "
                  "rather than treated as a passing convergence test. Intervals are not corrected for18 simultaneous comparisons, "
                  "and finite timestep/error can remain.", "", "![Exchange ensembles](figures/replica_exchange_ensembles.png)", ""]

    path = output/"shared_metadynamics/summary.json"
    if path.exists():
        summary = load(path)
        records = [load(p) for p in sorted(path.parent.glob("seed_*.json"))]
        lines += ["## Shared well-tempered MetaD", "",
                  "Four independent seeds, four simultaneous walkers, 20000 adaptive +30000 frozen steps per walker; "
                  "800 total hills per history. The stationary comparison uses only the frozen stage.", ""]
        table(lines, ["Quantity", "Four-seed mean ±95% interval"], [
            [key, f"{item['mean']:.6g} ± {item['95CI_halfwidth']:.3g}"] for key,item in summary.items()])
        lines += [f"The exact unbiased variance is1. Simultaneous pre-update hill-height error is at most "
                  f"{max(r['maximum_simultaneous_height_error'] for r in records):.3g}; permuting walkers changes the final bias by at most "
                  f"{max(r['final_permuted_walker_bias_difference'] for r in records):.3g}. "
                  "The frozen distribution agrees much more closely with its own exact equilibrium than with the ideal WT law: "
                  "this separates correct stationary sampling from incomplete bias convergence. No detailed-balance claim is made while hills are added.", "",
                  "![Shared MetaD distributions](figures/shared_metadynamics_frozen.png)", ""]

    summaries = [r for p in sorted((output/"kinetics").glob("summary_dt*.json")) for r in load(p)]
    if summaries:
        lines += ["## Audited biased first-passage clock", "",
                  "Independent overdamped trajectories use a static smooth flooding-shaped bias with zero bias/gradient in the entire "
                  "transition region. The exact reflecting/absorbing mean first-passage time is computed by quadrature. "
                  "These are128 independent events per row; none were silently dropped or censored.", ""]
        table(lines, ["dt", "Bias", "Exact MFPT", "Rescaled MFPT ±95% interval", "Raw mean time", "Mean boost", "Refitted KS bootstrap p"], [
            [r["dt"], "static" if r["static_flooding_bias"] else "none", number(r["exact_unbiased_MFPT"],6),
             f"{r['rescaled_MFPT']['mean']:.5g} ± {r['rescaled_MFPT']['95CI_halfwidth']:.3g}",
             number(r["raw_mean_elapsed"],5), number(r["mean_boost"],4), number(r["rate_analysis"]["ks_refitted_bootstrap_pvalue"],3)]
            for r in summaries])
        lines += ["All units here are numerical model units. Every simulated integration interval is included in the boost integral. "
                  "The transition-region maximum bias and gradient are both zero, and there are no deposition events. "
                  "Finite-sample uncertainty and Euler–Maruyama timestep error remain. This validates an analysis workflow on an audited model, "
                  "not native infrequent-MetaD or OPES-flooding deposition kinetics.", "",
                  "![Biased-clock reference](figures/biased_kinetics_reference.png)", ""]

    path = output/"ml_lifecycle/summary.json"
    if path.exists():
        ml = load(path)
        lines += ["## Actual trained TorchScript lifecycle", "",
                  "Four bootstrap supervised polynomial models are fitted to512 noisy energy labels of a known smooth function and exported "
                  "as TorchScript. Each held-out region contains400 geometries. The acquisition threshold is the95th percentile "
                  "of the interpolation-set disagreement; this is not a calibrated error threshold.", ""]
        table(lines, ["Region", "Energy RMSE", "Force RMSE", "Mean force disagreement", "Above acquisition threshold"], [
            [name, number(ml[name]["energy_RMSE"],5), number(ml[name]["force_RMSE"],5),
             number(ml[name]["mean_force_disagreement"],5), f"{100*ml[name]['acquired_fraction']:.1f}%"]
            for name in ("interpolation", "extrapolation")])
        lines += [f"The monitored 400-step PIMD run performs {ml['inspection_records']} inspections and collects "
                  f"{ml['selected_bead_geometries']} unlabeled geometries, reaching its configured cap. "
                  f"The maximum difference from the unmonitored positions, velocities and observables is {ml['monitored_trajectory_max_difference']:.1f}. "
                  f"Full Δ-learning versus the full learned model has maximum force difference {ml['full_delta_vs_full_model_force_max_error']:.3g}; "
                  f"the contracted Hamiltonian force agrees with a finite-difference derivative to {ml['contracted_force_chain_rule_max_error']:.3g}. "
                  f"Contracting 4 beads to 1 changes this sampled total energy by {ml['contraction_total_energy_change']:.6g}, "
                  "which demonstrates the approximation rather than claiming exact target energies. All energy/force units are atomic units.", "",
                  "The extrapolation force error substantially exceeds the committee spread despite their high correlation. "
                  "No quantum-chemistry labeling or model retraining is automated by the collector.", "",
                  "![ML held-out errors](figures/ml_heldout_error.png)", ""]
    lines += ["## Provenance and reproducibility", "",
              "[campaign_code_provenance.json](campaign_code_provenance.json) records driver archives, stable function hashes, "
              "launch commands and postlaunch restore-validation changes. The originals before the initialization fix are retained separately. "
              "The final test/CI evidence belongs to the overall validation report; trajectory data are not stamped as if they were generated "
              "by later reporting or compatibility-only edits.", ""]
    # Keep generated prose readable while preserving numerical values.
    text = "\n".join(lines)
    for left,right in (("contains80000","contains 80,000"),("and40000","and 40,000"),("with160","with 160"),
                       ("uses3200","uses 3,200"),("is57859,28661,38795,53730","is 57859, 28661, 38795, 53730"),
                       ("dt0.","dt 0."),("time400","time 400"),("time200","time 200"),("interval2.5","interval 2.5"),
                       ("interval0.05","interval 0.05"),("time40","time 40"),("with20%","with 20%"),
                       ("individual95%","individual 95%"),("for18","for 18"),("+30000","+ 30,000"),
                       ("is1.","is 1."),("are128","are 128"),("to512","to 512"),("contains400","contains 400"),
                       ("the95th","the 95th"),("monitored400-step","monitored 400-step"),("Contracting4","Contracting 4"),
                       ("beads to1","beads to 1")):
        text = text.replace(left,right)
    (output/"RESULTS.md").write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/validation/tier2-literature-2026-11-01/sampling"))
    main(parser.parse_args().output)
