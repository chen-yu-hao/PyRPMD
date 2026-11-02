#!/usr/bin/env python3
"""Post hoc, descriptive stationary VACF–MSD identity for the complete cohort.

This offline diagnostic preserves every original chain/segment and failed gate.
It is not a new eligibility screen, equilibrium test, or paper comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import cumulative_trapezoid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.water.analyze_native import chain_uncertainty
from benchmarks.water.campaign import atomic_json, sha256
from benchmarks.water.plot_native_campaign import pointwise_interval
from benchmarks.water.plot_production_stability import load_diagnostics

WINDOWS = ((.5, 1.), (1., 1.5), (1., 2.), (1.5, 2.))
SOURCE_NAMES = (
    "benchmarks/water/vacf_msd_identity.py",
    "benchmarks/water/plot_production_stability.py",
    "benchmarks/water/plot_native_campaign.py",
    "benchmarks/water/analyze_native.py",
    "benchmarks/water/campaign.py",
)


def stationary_curves(lag, vacf):
    """Integrate a piecewise linear three-dimensional velocity-dot VACF.

    Units: lag ps, VACF A^2/ps^2, MSD A^2, D A^2/ps. The first integral
    is exact for the linear interpolant. Its second integral is evaluated
    analytically on each interval, so constant/linear VACFs are exact even on
    a nonuniform grid. The double-trapezoid result is retained as a numerical
    quadrature diagnostic, never a model or eligibility selection.
    """
    lag, vacf = np.asarray(lag, float), np.asarray(vacf, float)
    if (lag.ndim != 1 or len(lag) < 2 or vacf.shape != lag.shape
            or not np.isfinite(lag).all() or not np.isfinite(vacf).all()
            or lag[0] != 0 or not np.all(np.diff(lag) > 0)):
        raise ValueError("finite matching curves on a strictly increasing lag grid starting at zero are required")
    h = np.diff(lag)
    integral = cumulative_trapezoid(vacf, lag, initial=0.)
    increments = h * integral[:-1] + h*h*(2*vacf[:-1]+vacf[1:])/6
    msd = 2*np.r_[0., np.cumsum(increments)]
    return {"D_VACF": integral/3, "MSD_from_VACF": msd,
            "MSD_double_trapezoid": 2*cumulative_trapezoid(integral, lag, initial=0.)}


def slope_D(lag, msd, window):
    """The unchanged unweighted linear MSD slope divided by six."""
    lag, msd = np.asarray(lag, float), np.asarray(msd, float)
    low, high = window
    if (lag.ndim != 1 or msd.shape != lag.shape or not np.isfinite(msd).all()
            or not np.isfinite(lag).all() or low >= high):
        raise ValueError("a finite MSD curve and ordered fit window are required")
    mask = (lag >= low-1e-12) & (lag <= high+1e-12)
    if mask.sum() < 2 or lag[mask][0] > low+1e-12 or lag[mask][-1] < high-1e-12:
        raise ValueError("the complete original fit window must be represented")
    return float(np.polyfit(lag[mask], msd[mask], 1)[0]/6)


def paired_chain_summary(observed, reconstructed):
    observed, reconstructed = np.asarray(observed, float), np.asarray(reconstructed, float)
    if observed.shape != (8,) or reconstructed.shape != (8,):
        raise ValueError("exactly eight paired chain means are required")
    return {"observed": chain_uncertainty(observed), "reconstructed": chain_uncertainty(reconstructed),
            "observed_minus_reconstructed": chain_uncertainty(observed-reconstructed)}


def diagnose(directory, protocol, supplement, exit_proof, output, *, render=True):
    directory, output = Path(directory), Path(output)
    sources = {name: sha256(ROOT/name) for name in SOURCE_NAMES}
    data = load_diagnostics(directory, protocol, supplement, exit_proof)
    if [row["seed"] for row in data["chains"]] != list(range(1101, 1109)):
        raise ValueError("this diagnostic requires the complete original eight-seed cohort")
    output.mkdir(parents=True, exist_ok=False)
    segments, chains, curve_rows = [], [], []
    observed_curves, reconstructed_curves, running_curves = [], [], []
    max_quadrature_difference = 0.
    with np.load(directory/"curves.npz", allow_pickle=False) as archive:
        for chain in data["chains"]:
            seed = chain["seed"]
            prefix = f"seed_{seed}_"
            lag = data["curves"][seed]["lag_time_ps"]
            observed, predicted, vacfs = [], [], []
            for index, original in enumerate(chain["segments"]):
                msd = np.asarray(archive[f"{prefix}segment_{index}_msd_angstrom2"], float)
                vacf = np.asarray(archive[f"{prefix}segment_{index}_vacf_angstrom2_ps2"], float)
                if msd.shape != lag.shape or not np.isfinite(msd).all():
                    raise ValueError("every saved segment MSD must be finite and use the complete lag grid")
                derived = stationary_curves(lag, vacf)
                prediction = derived["MSD_from_VACF"]
                observed.append(msd); predicted.append(prediction); vacfs.append(vacf)
                max_quadrature_difference = max(max_quadrature_difference,
                    float(np.max(abs(prediction-derived["MSD_double_trapezoid"]))))
                row = {"seed": seed, "segment": index, "retained_in_primary": True,
                       "observed_D_MSD_0.5_1ps": slope_D(lag, msd, (.5, 1.)),
                       "observed_D_MSD_1_2ps": slope_D(lag, msd, (1., 2.)),
                       "reconstructed_D_MSD_0.5_1ps": slope_D(lag, prediction, (.5, 1.)),
                       "reconstructed_D_MSD_1_2ps": slope_D(lag, prediction, (1., 2.)),
                       "observed_minus_reconstructed_MSD_2ps": float(msd[-1]-prediction[-1])}
                if not np.isclose(row["observed_D_MSD_1_2ps"], original["D_MSD_1_2ps"], rtol=1e-10, atol=1e-12):
                    raise ValueError("the saved segment curve does not reproduce its original slope")
                segments.append(row)
            mean_msd, mean_prediction, mean_vacf = [np.mean(values, axis=0) for values in (observed, predicted, vacfs)]
            derived = stationary_curves(lag, mean_vacf)
            if (not np.allclose(mean_msd, data["curves"][seed]["mean_msd_angstrom2"], rtol=1e-12, atol=1e-12)
                    or not np.allclose(mean_vacf, archive[prefix+"mean_vacf_angstrom2_ps2"], rtol=1e-12, atol=1e-12)
                    or not np.allclose(derived["D_VACF"], data["curves"][seed]["mean_D_VACF_angstrom2_ps"], rtol=1e-12, atol=1e-12)
                    or not np.allclose(mean_prediction, derived["MSD_from_VACF"], rtol=1e-12, atol=1e-12)):
                raise ValueError("all 25 segment curves must reproduce their saved means and linear-integral identities")
            observed_curves.append(mean_msd); reconstructed_curves.append(mean_prediction)
            running_curves.append(derived["D_VACF"])
            row = {"seed": seed, "segments_retained": 25, "windows": {}}
            for window in WINDOWS:
                key = f"{window[0]:g}-{window[1]:g}"
                row["windows"][key] = {"observed": slope_D(lag, mean_msd, window),
                                        "reconstructed": slope_D(lag, mean_prediction, window)}
            row["observed_short_minus_late"] = row["windows"]["0.5-1"]["observed"]-row["windows"]["1-2"]["observed"]
            row["reconstructed_short_minus_late"] = row["windows"]["0.5-1"]["reconstructed"]-row["windows"]["1-2"]["reconstructed"]
            chains.append(row)
            for j, time in enumerate(lag):
                curve_rows.append({"seed": seed, "lag_ps": float(time), "observed_MSD_angstrom2": float(mean_msd[j]),
                                   "reconstructed_MSD_angstrom2": float(mean_prediction[j]),
                                   "observed_minus_reconstructed_MSD_angstrom2": float(mean_msd[j]-mean_prediction[j]),
                                   "D_VACF_angstrom2_ps": float(derived["D_VACF"][j])})
    summaries = {key: paired_chain_summary([row["windows"][key]["observed"] for row in chains],
                                          [row["windows"][key]["reconstructed"] for row in chains])
                 for key in chains[0]["windows"]}
    contrast = paired_chain_summary([row["observed_short_minus_late"] for row in chains],
                                    [row["reconstructed_short_minus_late"] for row in chains])
    original_difference = data["report"]["MSD_window_comparison"]["0.5-1"]["paired_difference"]
    if any(not np.isclose(contrast["observed"][key], original_difference[key], rtol=1e-11, atol=1e-13)
           for key in ("mean", "standard_error", "CI95_halfwidth")):
        raise ValueError("the diagnostic must retain the original failed short-window contrast exactly within rounding")
    for name, rows in (("segment-identity.csv", segments), ("chain-curves.csv", curve_rows)):
        with (output/name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if render:
        render_figure(lag, np.asarray(observed_curves), np.asarray(reconstructed_curves),
                      np.asarray(running_curves), summaries, output)
    paths = {"report.json": directory/"report.json", "curves.npz": directory/"curves.npz", "protocol.json": Path(protocol),
             "supplement": Path(supplement), "exit_proof": Path(exit_proof)}
    if any(sha256(path) != data["input_sha256"][key] for key, path in paths.items()):
        raise RuntimeError("a frozen complete-cohort input changed during the diagnostic")
    if any(sha256(ROOT/name) != value for name, value in sources.items()):
        raise RuntimeError("a diagnostic dependency changed")
    report = {"schema": "pyrpmd.water-stationary-vacf-msd-identity.v1", "post_hoc_descriptive_diagnostic": True,
              "external_OpenMM_reference": True, "new_hard_gate": False, "physical_steps": 0, "PES_evaluations": 0,
              "literature_reproduced": False, "original_literature_comparison": data["report"]["literature_comparison"],
              "original_eligibility_checks": data["report"]["eligibility_checks"],
              "input_sha256": data["input_sha256"], "source_sha256": sources,
              "identity": "MSD(t) = 2 integral_0^t (t-tau) <v(0) dot v(tau)> d tau; D_GK(t) = integral_0^t Cvv(tau) d tau / 3",
              "quadrature": "exact integration of the saved piecewise linear VACF; a double-trapezoid alternative is retained only as a numerical diagnostic",
              "maximum_segment_MSD_difference_from_double_trapezoid_angstrom2": max_quadrature_difference,
              "uncertainty_unit": "Eight equally weighted seed-chain means, nominal Student-t df7; adjacent segments and time origins are not independent replicas.",
              "interval_scope": "Descriptive marginal/pointwise intervals; no simultaneous coverage or equilibrium/equivalence claim.",
              "all_chains": chains, "window_summaries": summaries, "short_minus_late_contrast": contrast,
              "limitations": ["The ensemble identity assumes a stationary velocity correlation; this cohort retains failed potential-energy preparation checks.",
                              "Finite trajectories use different time-origin subsets for each lag; their empirical MSD and lag-wise VACF need not obey the ensemble identity exactly.",
                              "Interpolating the saved 5fs VACF cannot recover unrecorded high-frequency details; the separate dense cadence control remains independent evidence.",
                              "A transient running diffusion integral and a nonzero long-time MSD intercept can produce lag-window dependence and log-MSD slopes below one.",
                              "Longer preparation cannot be assumed to eliminate preasymptotic MSD curvature; any longer-lag production study requires a separate protocol and new data."],
              "artifacts": {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}}
    atomic_json(output/"report.json", report)
    return report


def render_figure(lag, observed, reconstructed, running, summaries, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "pyrpmd-water-vacf-msd-identity"}):
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.5))
        axis = axes[0, 0]
        axis.plot(lag, observed.mean(axis=0), label="Saved MSD", color="#1964a0")
        axis.plot(lag, reconstructed.mean(axis=0), "--", label="From stationary VACF identity", color="#c65d16")
        axis.set(xlabel="Lag (ps)", ylabel="MSD (Å²)", title="A. Eight-chain mean displacement"); axis.legend(fontsize=9)
        axis = axes[0, 1]
        mean, half = pointwise_interval(observed-reconstructed)
        axis.plot(lag, mean, color="#6944a5"); axis.fill_between(lag, mean-half, mean+half, color="#6944a5", alpha=.15)
        axis.axhline(0, color=".5", lw=.8)
        axis.set(xlabel="Lag (ps)", ylabel="Saved − reconstructed MSD (Å²)", title="B. Finite-record identity residual")
        axis = axes[1, 0]
        mean, half = pointwise_interval(running)
        axis.plot(lag, mean, color="#1964a0"); axis.fill_between(lag, mean-half, mean+half, color="#1964a0", alpha=.15)
        axis.set(xlabel="Lag cutoff (ps)", ylabel="Running VACF integral / 3 (Å²/ps)", title="C. Transient running diffusion integral")
        axis = axes[1, 1]
        labels = list(summaries)
        for delta, key, label, color in ((-.1, "observed", "Saved MSD", "#1964a0"),
                                        (.1, "reconstructed", "From VACF identity", "#c65d16")):
            means = [summaries[name][key]["mean"] for name in labels]
            half = [summaries[name][key]["CI95_halfwidth"] for name in labels]
            axis.errorbar(np.arange(len(labels))+delta, means, yerr=half, fmt="o", color=color, capsize=3, label=label)
        axis.axvspan(-.45, .45, color="#b93435", alpha=.07)
        axis.set(xticks=range(len(labels)), xticklabels=labels, xlabel="MSD fit interval (ps)", ylabel="Slope / 6 (Å²/ps)",
                 title="D. Original fit windows, all chains retained"); axis.legend(fontsize=9)
        for axis in axes.flat:
            axis.grid(alpha=.18)
        fig.suptitle("Original water cohort: stationary VACF–MSD identity diagnostic", fontsize=14)
        fig.text(.5, .045, "OpenMM external reference · Post hoc description, no new gate or paper comparison.\n"
                 "Eight chain means; nominal df7 intervals (pointwise bands). Original preparation and short-window failures remain.\n"
                 "Stationarity and infinite-record assumptions are not established by agreement between these curves.",
                 ha="center", va="center", fontsize=9)
        fig.subplots_adjust(top=.92, bottom=.16, left=.085, right=.98, hspace=.36, wspace=.28)
        for extension in ("png", "svg"):
            fig.savefig(output/f"vacf-msd-identity.{extension}", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--exit-proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.directory, args.protocol, args.supplement, args.exit_proof, args.output)
    print(json.dumps(result["short_minus_late_contrast"], indent=2))


if __name__ == "__main__":
    main()
