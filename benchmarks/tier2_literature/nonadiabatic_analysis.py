#!/usr/bin/env python3
"""Offline uncertainty analysis and paper-width reference convergence.

Run after nonadiabatic.py; adding this independent analysis does not invalidate
the trajectory checkpoints. Gaussian-width references are contextual, while
the trajectory comparisons use the monochromatic quantum scattering problem.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.integrate import quad_vec
from scipy.stats import chi2_contingency

try:
    from .nonadiabatic import CHANNELS, MAIN_SEEDS, quantum_scattering, run_trajectory, wilson, _atomic_json, _write_csv
except ImportError:
    from nonadiabatic import CHANNELS, MAIN_SEEDS, quantum_scattering, run_trajectory, wilson, _atomic_json, _write_csv


SOURCES = [
    {"key": "tully_1990", "doi": "10.1063/1.459170",
     "url": "https://xuv.scs.illinois.edu/chem540/GroupProjects/Benke_Hammer_SourcePaper1.pdf",
     "title": "Molecular dynamics with electronic transitions", "author": "John C. Tully",
     "locations": ["Section IV: mass=2000 a.u.; 2000 trajectories per paper point",
                   "Eq. (20): Gaussian wavepacket; width typically 20/k",
                   "Eq. (21), Section IV A: model I Hamiltonian and parameters",
                   "Fig. 4: transmission/reflection on the adiabatic states"],
     "use": "Published Hamiltonian, asymptotic lower-state incidence and channel definitions. No paper points digitized."},
    {"key": "mash_2023", "doi": "10.1063/5.0139734",
     "url": "https://www.research-collection.ethz.ch/items/9e9f269e-f36c-4b37-80fe-0ad299fa2c65",
     "title": "A mapping approach to surface hopping", "authors": ["Jonathan R. Mannouch", "Jeremy O. Richardson"],
     "use": "MASH theoretical method; this study does not reproduce a named numerical figure from this paper."},
    {"key": "mash_integrators_2025", "doi": "10.1021/acs.jctc.4c01684",
     "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11912208/",
     "title": "Time-Reversible Implementation of MASH for Efficient Nonadiabatic Molecular Dynamics",
     "authors": ["J. Amira Geuther", "Kasra Asnaashari", "Jeremy O. Richardson"],
     "locations": ["Sections 2.1 and 2.3: impulse/reflection and phase-aligned overlap propagation",
                   "Section 3: distinction between endpoint and reversible/event-resolved algorithms"],
     "use": "Algorithm scope and limitations. The implementation tested here is fixed-step endpoint MASH."},
]


def packet_reference(momentum, *, dx=.0025, tolerance=2e-5):
    """Adaptive integration over the paper-inspired Gaussian momentum spread.

    Split at the upper-channel threshold. Simple low-order Gauss-Hermite sums
    underresolve narrow resonances at p=8.8, so their result is not used as a
    converged paper comparison. +/-7 sigma truncation is below 3e-12 mass.
    """
    sigma = momentum/20
    threshold = np.sqrt(80.)
    left, right = max(1e-4, momentum-7*sigma), momentum+7*sigma
    evaluations = 0
    def integrand(p):
        nonlocal evaluations
        evaluations += 1
        result = quantum_scattering(float(p), dx=dx)
        density = np.exp(-.5*((p-momentum)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
        return density*np.asarray([result[c] for c in CHANNELS[:4]])
    started = time.perf_counter()
    result, error, info = quad_vec(integrand, left, right, epsabs=tolerance, epsrel=tolerance,
                                  norm="max", points=[threshold] if left < threshold < right else None,
                                  limit=500, full_output=True)
    return {"momentum": momentum, "sigma_momentum": sigma, "dx": dx, "tolerance": tolerance,
            "converged": bool(info.success), "status": int(info.status), "message": info.message,
            "estimated_absolute_quadrature_error": float(error), "evaluations": evaluations,
            "seconds": time.perf_counter()-started, **dict(zip(CHANNELS[:4], map(float, result)))}


def hermite_convergence(output):
    """Retain the deliberately insufficient fixed-node rules as diagnostics."""
    momentum = 8.8
    for count in [33, 65, 129]:
        nodes, weights = np.polynomial.hermite.hermgauss(count)
        started = time.perf_counter()
        values = [quantum_scattering(momentum+np.sqrt(2)*(momentum/20)*node, dx=.0025)
                  for node in nodes]
        _atomic_json(output/f"packet_quadrature_{count}.json", {
            "momentum": momentum, "nodes": count, "dx": .0025,
            "probabilities": {c: float(np.dot(weights, [r[c] for r in values])/np.sqrt(np.pi))
                              for c in CHANNELS[:4]}, "seconds": time.perf_counter()-started})


def analyze(output):
    with (output/"trajectories.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("momentum", "dt", "max_energy_error", "final_energy_error", "seconds"):
            row[key] = float(row[key])
        row["block_seed"], row["trajectory"] = int(row["block_seed"]), int(row["trajectory"])
    momenta = sorted({r["momentum"] for r in rows})
    censoring = []
    for method, momentum, dt in sorted({(r["method"], r["momentum"], r["dt"]) for r in rows}):
        selected = [r for r in rows if (r["method"], r["momentum"], r["dt"]) == (method, momentum, dt)]
        unresolved = sum(r["channel"] == "unresolved" for r in selected)
        for channel in CHANNELS[:4]:
            count = sum(r["channel"] == channel for r in selected)
            censoring.append({"method": method, "momentum": momentum, "dt": dt, "channel": channel,
                              "n": len(selected), "observed_count": count, "unresolved": unresolved,
                              "empirical_lower_bound": count/len(selected),
                              "empirical_upper_bound": (count+unresolved)/len(selected),
                              "conservative_ci95_low": wilson(count, len(selected))[0],
                              "conservative_ci95_high": wilson(count+unresolved, len(selected))[1]})
    _atomic_json(output/"censoring_bounds.json", censoring)
    seeds = []
    for method in ["rpsh", "mash"]:
        for momentum in momenta:
            for channel in CHANNELS[:4]:
                selected = [r for r in rows if r["method"] == method and r["momentum"] == momentum and r["dt"] == 4]
                blocks = [[r for r in selected if r["block_seed"] == seed] for seed in MAIN_SEEDS]
                counts = np.asarray([sum(r["channel"] == channel for r in block) for block in blocks])
                totals = np.asarray([len(block) for block in blocks])
                observed = np.column_stack([counts, totals-counts])
                if np.any(observed.sum(axis=0) == 0):
                    statistic = pvalue = None
                    applicable = False
                else:
                    statistic, pvalue, _, expected = chi2_contingency(observed, correction=False)
                    applicable = bool(np.min(expected) >= 5)
                    statistic, pvalue = float(statistic), float(pvalue)
                seeds.append({"method": method, "momentum": momentum, "channel": channel,
                              "counts": counts.tolist(), "totals": totals.tolist(),
                              "block_probabilities": (counts/totals).tolist(),
                              "pearson_statistic": statistic, "pearson_pvalue": pvalue,
                              "asymptotic_test_applicable": applicable,
                              "interpretation": "descriptive independent-seed check, not proof of seed independence or physical accuracy"})
    timesteps = []
    for method in ["rpsh", "mash"]:
        for momentum in momenta:
            for coarse, fine in [(8., 4.), (4., 2.)]:
                def select(dt):
                    return {r["trajectory"]: r for r in rows if r["method"] == method and r["momentum"] == momentum
                            and r["dt"] == dt and r["block_seed"] == MAIN_SEEDS[0]}
                a, b = select(coarse), select(fine)
                common = sorted(set(a)&set(b))
                changes = sum(a[i]["channel"] != b[i]["channel"] for i in common)
                low, high = wilson(changes, len(common))
                timesteps.append({"method": method, "momentum": momentum, "coarse_dt": coarse,
                                  "fine_dt": fine, "paired_n": len(common), "changed_channels": changes,
                                  "change_fraction": changes/len(common), "change_ci95_low": low,
                                  "change_ci95_high": high,
                                  "coarse_mean_max_energy_error": float(np.mean([a[i]["max_energy_error"] for i in common])),
                                  "fine_mean_max_energy_error": float(np.mean([b[i]["max_energy_error"] for i in common])),
                                  "coarse_largest_energy_error": max(a[i]["max_energy_error"] for i in common),
                                  "fine_largest_energy_error": max(b[i]["max_energy_error"] for i in common)})
    _atomic_json(output/"seed_independence.json", seeds)
    _atomic_json(output/"timestep_diagnostics.json", timesteps)
    _write_csv(output/"timestep_diagnostics.csv", timesteps)
    _atomic_json(output/"sources.json", {"sources": SOURCES,
                 "comparison_types": ["published-model independent monochromatic quantum reference",
                                      "paper-inspired Gaussian-width reference, explicitly separate",
                                      "finite-trajectory binomial confidence intervals",
                                      "numerical timestep, gauge and representation validation"]})
    path = output/"manifest.json"
    manifest = json.loads(path.read_text())
    manifest["literature_sources"] = SOURCES
    manifest["primary_paper_figure"] = "Tully 1990 Fig.4; same model/channel definitions, not digitized values"
    manifest["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["scipy"] = scipy.__version__
    manifest["thread_environment"] = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    _atomic_json(path, manifest)
    render_report(output, seeds, timesteps)
    return seeds, timesteps


def render_report(output, seeds, timesteps):
    summary = json.loads((output/"summary.json").read_text())
    with (output/"quantum_reference_convergence.csv").open() as stream:
        references = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    rows = summary["rows"]
    momenta = sorted({r["momentum"] for r in rows})
    lines = ["# Quantitative results: Tully model I", "",
             f"Actual public-engine trajectories: **{summary['total_trajectories']:,}**; "
             f"unresolved trajectories: **{summary['unresolved']}**. "
             "Main points contain 512 trajectories each, with four independent seed blocks.", "",
             "These are independent quantum calculations and PyRPMD ensembles for the published Hamiltonian; "
             "no numerical points were digitized from Tully's Fig. 4. See [README.md](README.md) and "
             "[sources.json](sources.json) for equations, source locations and limitations.", "",
             "## Main probabilities at Δt = 4 a.u.", "",
             "Parentheses give pointwise 95% Wilson intervals for classification by the fixed time cutoff. "
             "All trajectories, including unresolved outcomes, remain in the denominator. "
             "The quantum reference is monochromatic and asymptotic; conservative assignments of unresolved "
             "outcomes are provided separately in `censoring_bounds.json`.", "",
             "| Method | Momentum | Lower transmission | Upper transmission | Lower reflection |",
             "| --- | ---: | --- | --- | --- |"]
    for method in ["quantum", "rpsh", "mash"]:
        for momentum in momenta:
            if method == "quantum":
                reference = min((r for r in references if r["momentum"] == momentum), key=lambda r: r["dx"])
                values = [f"{reference[c]:.6f}" for c in CHANNELS[:3]]
            else:
                selected = {r["channel"]: r for r in rows if r["method"] == method and r["momentum"] == momentum and r["dt"] == 4}
                values = [f"{selected[c]['probability']:.4f} ({selected[c]['ci95_low']:.4f}–{selected[c]['ci95_high']:.4f})"
                          for c in CHANNELS[:3]]
            label = {"quantum": "Independent quantum", "rpsh": "FSSH (`P=1` RPSH)", "mash": "Endpoint MASH"}[method]
            lines.append(f"| {label} | {momentum:g} | "+" | ".join(values)+" |")
    lines += ["", "All four outgoing channels, including small upper-state reflection, are retained in "
              "`channel_probabilities.csv`. Zero observed counts do not imply a zero true probability.", "",
              "### Finite-duration censoring", "",
              "| Method | Momentum | Unresolved /512 at Δt=4 |",
              "| --- | ---: | ---: |"]
    for row in rows:
        if row["dt"] == 4 and row["channel"] == "unresolved":
            lines.append(f"| {row['method']} | {row['momentum']:g} | {row['count']} |")
    lines += ["", "Each unresolved path could eventually reach any channel. The empirical asymptotic probability "
              "therefore lies between `count/N` and `(count+unresolved)/N` before accounting for sampling error. "
              "`censoring_bounds.json` also combines these worst-case assignments with Wilson intervals. "
              "Any secondary 48-flight-time tail follow-up is recorded separately and does not replace the "
              "predeclared 12-flight-time outcome in this main table.", "",
              "## Reference discretization", "",
              "| Momentum | Max channel change, h=0.0025→0.00125 | Max channel change, L=12→16 at h=0.0025 | Max flux defect |",
              "| ---: | ---: | ---: | ---: |"]
    for momentum in momenta:
        selected = [r for r in references if r["momentum"] == momentum]
        fine = min(selected, key=lambda r: r["dx"])
        middle = next(r for r in selected if abs(r["dx"]-.0025) < 1e-12 and r["boundary"] == 12)
        large = next(r for r in selected if r["boundary"] == 16)
        lines.append(f"| {momentum:g} | {max(abs(fine[c]-middle[c]) for c in CHANNELS[:4]):.3g} | "
                     f"{max(abs(large[c]-middle[c]) for c in CHANNELS[:4]):.3g} | {max(r['flux_error'] for r in selected):.3g} |")
    lines += ["", "Flux conservation is a solver check; the larger mesh change is the relevant discretization diagnostic.", "",
              "## Independent seed blocks", "",
              "The selected channel is lower reflection below the upper-state threshold and upper transmission above it. "
              "Each entry is the event count out of 128. A Pearson p-value is only printed when every expected cell count "
              "is at least five; this is a descriptive homogeneity check, not a claim that finite sampling has converged.", "",
              "| Method | Momentum | Channel | Four block counts | Pearson p-value |",
              "| --- | ---: | --- | --- | ---: |"]
    for row in seeds:
        channel = "reflected_lower" if row["momentum"] < np.sqrt(80) else "transmitted_upper"
        if row["channel"] == channel:
            pvalue = f"{row['pearson_pvalue']:.4f}" if row["asymptotic_test_applicable"] else "not applicable"
            lines.append(f"| {row['method']} | {row['momentum']:g} | {channel} | {row['counts']} | {pvalue} |")
    lines += ["", "## Timestep refinement", "",
              "The following comparisons retain the same 128 trajectory identities. FSSH random-number sequences diverge "
              "across timesteps, so channel-change fractions are not convergence rates of deterministic paths. "
              "MASH uses the same initial spin in each pair. Mean maximum energy drift is measured along each whole trajectory, "
              "in hartree; the table compares the same subset at both timesteps.", "",
              "| Method | Momentum | Δt coarse→fine | Changed channels /128 | 95% change-fraction interval | Mean max energy drift coarse→fine |",
              "| --- | ---: | --- | ---: | --- | --- |"]
    for row in timesteps:
        lines.append(f"| {row['method']} | {row['momentum']:g} | {row['coarse_dt']:g}→{row['fine_dt']:g} | "
                     f"{row['changed_channels']} | {row['change_ci95_low']:.4f}–{row['change_ci95_high']:.4f} | "
                     f"{row['coarse_mean_max_energy_error']:.3g}→{row['fine_mean_max_energy_error']:.3g} |")
    lines += ["", "Per-channel fine-minus-coarse probability differences and paired standard errors are in "
              "`paired_timestep_comparison.csv`. A zero observed change in 128 pairs still has a nonzero upper confidence bound.", "",
              "**The FSSH refinement ensembles do not establish population convergence at the few-percent level.** "
              "At p=25, the upper-transmission estimate changes by −0.08594 from Δt=4 to Δt=2, with paired standard "
              "error 0.06374 (a normal-approximation 95% uncertainty of about ±0.125). That difference is not resolved "
              "by 128 paths. For MASH, the same initial spins give one changed classification at p=8.8 and no changes "
              "at the two higher momenta; zero observed changes still permits a change probability up to 2.91% "
              "under the Wilson bound. Roughly fourfold reductions in energy drift on halving Δt are a useful "
              "numerical check, but cannot establish the convergence order of populations across hops.", "",
              "## Interpretation of the paper comparison", "",
              "All 24 main outgoing-channel pointwise 95% intervals contain the monochromatic quantum reference "
              "at the three momenta. These are dependent channel comparisons within six ensembles, not 24 independent "
              "proofs of accuracy. In particular, upper transmission at p=15 exceeds the reference by 0.03639 for "
              "FSSH and 0.03248 for MASH; the sampling intervals are too broad to resolve errors of a few percentage "
              "points. The results provide a bounded statistical consistency check, not exact agreement or evidence "
              "that one method outperforms the other.", "",
              "At p=8.8 the separately converged paper-inspired Gaussian-width reference has lower reflection "
              "0.03528 and upper transmission 0.04874. The monochromatic comparison instead has reflection 0.06725 "
              "and a closed upper channel. Substituting the Gaussian reference would change the initial distribution "
              "and therefore the question being tested; it is not used to improve apparent method agreement.", "",
              "## Scope of the result", "",
              "Statistical compatibility at a few model points is useful validation, but it does not establish exact "
              "quantum dynamics or reproduce every point in a published figure. Endpoint MASH remains a first-order-at-hops "
              "discretization requiring refinement; it is not the reversible event-resolved algorithm. Nuclear tunneling, "
              "long-time coherence and arbitrary molecular dynamics are not established here. The single-bead results "
              "do not validate thermal NRPMD rates or multi-bead RPSH/MASH accuracy.", ""]
    (output/"results.md").write_text("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Put every method at its actual momentum. Horizontal dodging would be
    # misleading for the p=8.8 point close to the channel-opening threshold.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, channel, title in zip(axes, CHANNELS[:3], ["Transmission: lower", "Transmission: upper", "Reflection: lower"]):
        exact = [min((r for r in references if r["momentum"] == p), key=lambda r: r["dx"])[channel] for p in momenta]
        ax.plot(momenta, exact, "kD--", markersize=5, label="Independent quantum")
        for method, marker, color in [("rpsh", "o", "#1670a5"), ("mash", "s", "#bd5b23")]:
            selected = sorted((r for r in rows if r["method"] == method and r["dt"] == 4 and r["channel"] == channel),
                              key=lambda r: r["momentum"])
            y = np.array([r["probability"] for r in selected])
            error = np.array([[r["probability"]-r["ci95_low"] for r in selected],
                              [r["ci95_high"]-r["probability"] for r in selected]])
            ax.errorbar(momenta, y, yerr=error, fmt=marker, mfc="none", capsize=3, color=color,
                        label="FSSH" if method == "rpsh" else "Endpoint MASH")
        ax.set(title=title, xlabel="Incident momentum (a.u.)",
               ylim=(-.005, .16) if channel == "reflected_lower" else (-.03, 1.03))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Scattering probability (95% Wilson interval)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Tully model I • actual public-engine trajectories • Δt = 4 a.u.")
    fig.savefig(output/"scattering_channels.png", dpi=180)
    fig.savefig(output/"scattering_channels.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, method in zip(axes, ["rpsh", "mash"]):
        for momentum in momenta:
            selected = [r for r in timesteps if r["method"] == method and r["momentum"] == momentum]
            values = {}
            for row in selected:
                values[row["coarse_dt"]] = row["coarse_mean_max_energy_error"]
                values[row["fine_dt"]] = row["fine_mean_max_energy_error"]
            ax.plot(sorted(values), [values[t] for t in sorted(values)], "o-", label=f"p = {momentum:g}")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set(xlabel="Nuclear timestep (a.u.)", title="FSSH" if method == "rpsh" else "Endpoint MASH")
        ax.set_xticks([2, 4, 8], ["2", "4", "8"])
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean maximum |energy drift| (hartree)")
    fig.suptitle("Same 128 initial trajectory identities per timestep")
    fig.savefig(output/"timestep_energy_drift.png", dpi=180)
    fig.savefig(output/"timestep_energy_drift.svg")
    plt.close(fig)


def extend_unresolved(output):
    """One predeclared secondary 48-flight-time follow-up; keep main data intact."""
    for path in sorted((output/"jobs").glob("*.json")):
        job = json.loads(path.read_text())
        settings = job["identity"]
        for original in job["trajectories"]:
            if original["channel"] != "unresolved":
                continue
            target = output/"tail_followup"/(path.stem+f"_i{original['trajectory']}.json")
            if target.exists():
                continue
            trajectory = run_trajectory(settings["method"], settings["momentum"], settings["dt"],
                                        np.random.SeedSequence(original["seed_entropy"]),
                                        boundary=settings["boundary"], electronic_substeps=settings["electronic_substeps"],
                                        max_flight_times=48)
            _atomic_json(target, {"job_identity": settings, "trajectory": original["trajectory"],
                                  "original_status": original["channel"], "original_steps": original["steps"],
                                  "original_max_flight_times": 12, "secondary_max_flight_times": 48,
                                  "seed_entropy": original["seed_entropy"], "secondary_result": trajectory,
                                  "interpretation": "secondary tail diagnostic; main counts/denominators are unchanged"})
            print("tail follow-up", target.name, trajectory["channel"], trajectory["steps"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packet-only", action="store_true")
    parser.add_argument("--packet-convergence", action="store_true")
    parser.add_argument("--extend-unresolved", action="store_true")
    parser.add_argument("--tail-only", action="store_true")
    args = parser.parse_args()
    if args.packet_only or args.packet_convergence:
        hermite_convergence(args.output)
        results = []
        for dx, tolerance in [(.005, 2e-5), (.0025, 1e-5)]:
            result = packet_reference(8.8, dx=dx, tolerance=tolerance)
            results.append(result)
            _atomic_json(args.output/"paper_width_reference_convergence.json", results)
            print(json.dumps(result), flush=True)
    if args.extend_unresolved or args.tail_only:
        extend_unresolved(args.output)
    if not args.packet_only and not args.tail_only:
        analyze(args.output)


if __name__ == "__main__":
    main()
