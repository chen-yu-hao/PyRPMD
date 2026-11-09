"""Reanalyse two *completed* equal-budget BKMP2 rate jobs; never invent missing data.

Run both jobs with the paired inputs in equal_cost_inputs/. Both MBAR and UI
use identical, real, rate_flux production samples within each method. Each
method selects its own MBAR PMF maximum; its UI estimate uses the SAME xi*.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HARTREE_TO_KCAL = 627.5094740631
FIXED_PMF_STEPS = 96 * 2 * (3000 + 10000)
PARENT_STEPS = 2 * (2000 + 32 * 100)
CHILD_STEPS = 2 * 32 * 2 * 2 * 800
PLANNED_RECROSSING_STEPS = PARENT_STEPS + CHILD_STEPS


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_case(directory, method):
    directory = Path(directory).resolve()
    summary_path = directory / "summary.json"
    pmf_path = directory / "rate_pmf.npz"
    if not summary_path.is_file() or not pmf_path.is_file():
        raise FileNotFoundError(
            f"{directory}: completed summary.json and rate_pmf.npz required; no extrapolation"
        )
    summary = _json(summary_path)
    pmf = summary.get("pmf", {})
    ui = pmf.get("ui_comparison", {})
    if not pmf.get("ready") or not ui.get("ready"):
        raise ValueError(
            f"{directory}: MBAR or paired UI PMF unsupported; "
            f"PMF status={pmf.get('status')}; UI reason={ui.get('reason')}"
        )
    if pmf.get("support", {}).get("selection") != "maximum_supported_bin_center":
        raise ValueError(f"{directory}: xi* was not chosen automatically from MBAR PMF")
    if pmf.get("sampling_measure") != "rate_flux":
        raise ValueError(f"{directory}: not rate_flux; cannot compare with RPMDrate")
    xi_star = float(pmf["xi_star"])
    if abs(xi_star - float(ui["xi_star"])) > 1e-12:
        raise ValueError("UI was not evaluated at the primary MBAR dividing surface")
    with np.load(pmf_path, allow_pickle=False) as z:
        xi = np.asarray(z["xi"], float).reshape(-1)
        energy = np.asarray(z["free_energy"], float).reshape(-1)
        mask = np.asarray(z["mask"], bool).reshape(-1)
    if xi.shape != energy.shape or xi.shape != mask.shape:
        raise ValueError("PMF array shapes disagree")
    ui_x = np.asarray(ui["xi"], float).reshape(-1)
    ui_y = np.asarray(ui["free_energy"], float).reshape(-1)
    ess = np.asarray(pmf.get("effective_samples_per_stream", []), float).reshape(-1)
    if not len(ess) or not np.all(np.isfinite(ess)):
        raise ValueError("missing valid per-stream ESS")
    if method == "fixed96":
        budget = {
            "target_integration_steps": FIXED_PMF_STEPS,
            "used_integration_steps": FIXED_PMF_STEPS,
            "pilot_integration_steps": 0,
            "unspent_integration_steps": 0,
        }
    elif method == "adaptive":
        budget = pmf.get("adaptive_umbrella", {}).get("integration_budget")
        if budget is None:
            raise ValueError("missing accountable adaptive pilot + frozen production budget")
    else:
        raise ValueError(f"unknown method {method}")
    if int(budget["target_integration_steps"]) != FIXED_PMF_STEPS:
        raise ValueError("PMF budgets have different targets")
    mbar = float(pmf["free_energy_barrier_hartree"]) * HARTREE_TO_KCAL
    ui_barrier = float(ui["free_energy_barrier_hartree"]) * HARTREE_TO_KCAL
    rate_estimate = summary.get("rate_estimate_cm3_s")
    rate_gated = summary.get("rate_cm3_s")
    return {
        "method": method,
        "directory": str(directory),
        "inputs_sha256": {"summary": _hash(summary_path), "pmf": _hash(pmf_path)},
        "xi_star": xi_star,
        "MBAR_barrier_kcal_mol": mbar,
        "UI_barrier_kcal_mol": ui_barrier,
        "MBAR_minus_UI_kcal_mol": mbar - ui_barrier,
        "MBAR_barrier_SE_kcal_mol": (
            float(pmf["barrier_standard_error_hartree"]) * HARTREE_TO_KCAL
            if pmf.get("barrier_standard_error_hartree") is not None else None
        ),
        "UI_barrier_SE_kcal_mol": (
            float(ui.get("ui_barrier_statistics", {}).get("standard_error_hartree")) * HARTREE_TO_KCAL
            if ui.get("ui_barrier_statistics", {}).get("standard_error_hartree") is not None else None
        ),
        "paired_difference_SE_kcal_mol": (
            float(ui.get("paired_difference_statistics", {}).get("standard_error_hartree")) * HARTREE_TO_KCAL
            if ui.get("paired_difference_statistics", {}).get("standard_error_hartree") is not None else None
        ),
        "overlap_connected": bool(pmf["overlap_connected"]),
        "MBAR_solver_converged": bool(pmf.get("solver", {}).get("converged")),
        "minimum_stream_ESS": float(ess.min()),
        "median_stream_ESS": float(np.median(ess)),
        "effective_samples_per_stream": ess.tolist(),
        "streams": int(len(ess)),
        "kappa": summary.get("kappa"),
        "kappa_standard_error": summary.get("kappa_standard_error"),
        "recrossing_plateau_detected": bool(summary.get("plateau_detected", False)),
        "child_trajectories": summary.get("child_trajectories"),
        "rate_estimate_cm3_molecule_s": rate_estimate,
        "accepted_rate_cm3_molecule_s": rate_gated,
        "rate_standard_error_cm3_molecule_s": summary.get("rate_standard_error_cm3_s"),
        "run_status": summary.get("status"),
        "reported_converged": bool(summary.get("converged", False)),
        "PMF_budget": budget,
        "xi": xi.tolist(), "MBAR_pmf_hartree": np.where(mask, np.nan, energy).tolist(),
        "UI_xi": ui_x.tolist(), "UI_pmf_hartree": ui_y.tolist(),
    }


def compare(fixed, adaptive):
    a, b = read_case(fixed, "fixed96"), read_case(adaptive, "adaptive")
    u1 = int(a["PMF_budget"]["used_integration_steps"])
    u2 = int(b["PMF_budget"]["used_integration_steps"])
    unspent = FIXED_PMF_STEPS - u2
    if unspent < 0 or unspent / FIXED_PMF_STEPS > 0.001:
        raise ValueError("adaptive MD-step budget exceeds target or differs by more than 0.1%")
    return {
        "schema": "pyrpmd.h3.equal-cost-mbar-ui-rate.v1",
        "same_target_pmf_integration_steps": FIXED_PMF_STEPS,
        "actual_fixed_pmf_steps": u1,
        "actual_adaptive_pmf_steps": u2,
        "adaptive_unspent_fraction": unspent / FIXED_PMF_STEPS,
        "planned_identical_recrossing_steps_per_method": PLANNED_RECROSSING_STEPS,
        "common_conditions": {
            "temperature_K": 1500, "beads": 16, "timestep_fs": 0.1,
            "rinf_bohr": 10, "beta_k": 400,
            "coordinate": "RPMDrate centroid xi; two equivalent H-exchange paths",
            "PMF_measure": "rate_flux", "PMF_reference_xi": 0,
            "xi_star": "independent MBAR auto maximum per sampling method",
            "UI_surface": "same MBAR-selected xi* within each method",
        },
        "fixed96": a, "adaptive": b,
        "comparison": {
            "delta_xi_star_adaptive_minus_fixed": b["xi_star"] - a["xi_star"],
            "delta_MBAR_barrier_kcal_mol": b["MBAR_barrier_kcal_mol"] - a["MBAR_barrier_kcal_mol"],
            "delta_UI_barrier_kcal_mol": b["UI_barrier_kcal_mol"] - a["UI_barrier_kcal_mol"],
            "delta_rate_estimate_cm3_molecule_s": (
                b["rate_estimate_cm3_molecule_s"] - a["rate_estimate_cm3_molecule_s"]
                if b["rate_estimate_cm3_molecule_s"] is not None
                and a["rate_estimate_cm3_molecule_s"] is not None else None
            ),
        },
        "converged": False,
        "limitations": [
            "Matched integration-step counts are not equal wall time or PES force-evaluation counts.",
            "Adaptive pilots are charged but excluded from frozen-production MBAR/UI samples.",
            "Same target recrossing parent/child steps; actual retries may differ.",
            "UI is evaluated at each method's MBAR-selected surface and is not an independent recrossing estimate.",
            "Reported rates are finite-time estimates unless PMF support, plateau and energy drift gates pass.",
            "PMF peak-selection uncertainty is not included in fixed-surface bootstrap.",
            "One seed and 16 beads do not establish physical, time-step or sampling convergence.",
        ],
    }


def render(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    for method, label in (("fixed96", "Fixed 96"), ("adaptive", "Adaptive")):
        case = result[method]
        ax.plot(case["xi"], np.array(case["MBAR_pmf_hartree"]) * HARTREE_TO_KCAL,
                label=f"{label} MBAR")
        ax.plot(case["UI_xi"], np.array(case["UI_pmf_hartree"]) * HARTREE_TO_KCAL,
                linestyle="--", label=f"{label} Gaussian UI")
        ax.axvline(case["xi_star"], alpha=0.35, linestyle=":")
    ax.set(xlabel="RPMDrate ξ (centroid)", ylabel="W(ξ) − W(0) (kcal/mol)",
           title="BKMP2 matched integration-step PMF comparison")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "pmf_mbar_ui.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    keys = ["fixed96", "adaptive"]
    values = [result[m]["MBAR_minus_UI_kcal_mol"] for m in keys]
    errs = [result[m]["paired_difference_SE_kcal_mol"] for m in keys]
    ax.bar(keys, values)
    if all(v is not None for v in errs):
        ax.errorbar(keys, values, yerr=errs, fmt="none", capsize=5)
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set(ylabel="MBAR − UI barrier (kcal/mol)",
           title="Paired estimator differences (1 SE if available)")
    fig.tight_layout()
    fig.savefig(output / "mbar_minus_ui.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", type=Path, required=True, help="completed fixed96 run directory")
    parser.add_argument("--adaptive", type=Path, required=True, help="completed adaptive run directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.fixed, args.adaptive)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    columns = ["method", "xi_star", "MBAR_barrier_kcal_mol", "UI_barrier_kcal_mol",
               "MBAR_minus_UI_kcal_mol", "overlap_connected", "minimum_stream_ESS",
               "median_stream_ESS", "kappa", "rate_estimate_cm3_molecule_s",
               "accepted_rate_cm3_molecule_s", "recrossing_plateau_detected"]
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: result[m].get(k) for k in columns} for m in ("fixed96", "adaptive"))
    render(result, args.output)
    print(json.dumps({
        "fixed_MBAr_minus_UI_kcal_mol": result["fixed96"]["MBAR_minus_UI_kcal_mol"],
        "adaptive_MBAr_minus_UI_kcal_mol": result["adaptive"]["MBAR_minus_UI_kcal_mol"],
        "delta_xi_star": result["comparison"]["delta_xi_star_adaptive_minus_fixed"],
        "delta_rate": result["comparison"]["delta_rate_estimate_cm3_molecule_s"],
        "PMF_budget_gap_fraction": result["adaptive_unspent_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
