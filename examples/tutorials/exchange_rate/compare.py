"""Read-only reactive-event and independent-parent rate comparison."""
from pathlib import Path
import argparse
import csv
import json
import math

import numpy as np

HISTORICAL_BC = {
    "rate_cm3_s": 1.142e-11, "standard_error_cm3_s": 0.051e-11,
    "source": "https://github.com/chen-yu-hao/PyRPMD/issues/170",
    "label": "reported stiff-window UI D+H2 teaching-model result; not rerun here",
}


def compare(run, *, bc_run=None, smoke=False):
    run = Path(run)
    summary = json.loads((run / "results/summary.json").read_text())
    counts = summary["counts"]
    reasons = []
    if counts.get("product", 0) < 1:
        reasons.append("no reactive DH+H event observed")
    if counts.get("failed", 0) or counts.get("unresolved", 0):
        reasons.append("failed or unresolved collisions prevent rate comparison")
    if summary.get("sampling") != "thermal_flux":
        reasons.append("input is not the declared thermal-flux ensemble")
    if summary.get("temperature_K") != 1500 or summary.get("beads") != 4:
        reasons.append("comparison requires the declared 1500 K, P=4 ensemble")
    result = {"scope": "mechanics_only" if smoke else "teaching_model_rate_comparison",
              "counts": counts, "parents": summary["parents"], "reasons": reasons,
              "scientific_convergence": False, "rate_comparison_performed": False}
    if smoke:
        result["passed"] = not reasons
        return result
    estimate = summary["total_reaction"]["rate_cm3_molecule_s"]
    if estimate is None:
        reasons.append("the dRPMD workflow withheld its rate")
    if counts.get("product", 0) < 20 or summary["parents"] < 8:
        reasons.append("need at least 20 observed reactions and 8 independent parent chains")
    if reasons:
        result["passed"] = False
        return result
    table = run / summary["trajectory_table"]
    groups = {}
    with table.open(newline="") as stream:
        for record in csv.DictReader(stream):
            group = groups.setdefault(int(record["parent_id"]), [0, 0])
            group[0] += record["status"] == "product"
            group[1] += 1
    if sum(group[1] for group in groups.values()) != summary["trajectories"]:
        raise ValueError("trajectory table count differs from the summary")
    totals = np.asarray([groups[key] for key in sorted(groups)], dtype=float)
    rng = np.random.default_rng(170165)
    resampled = totals[rng.integers(0, len(totals), size=(2000, len(totals)))].sum(axis=1)
    prefactor = estimate / summary["total_reaction"]["event_fraction"]
    error = float(np.std(prefactor * resampled[:, 0] / resampled[:, 1], ddof=1))
    reference = HISTORICAL_BC.copy()
    if bc_run is not None:
        bc = json.loads((Path(bc_run) / "results/summary.json").read_text())
        value, sem = bc.get("rate_cm3_s"), bc.get("rate_standard_error_cm3_s")
        if value is None or sem is None or not bc.get("rate_uncertainty", {}).get("available", False):
            reasons.append("BC rate or full PMF+recrossing standard error is withheld")
        else:
            reference = {"rate_cm3_s": value, "standard_error_cm3_s": sem,
                         "source": str(bc_run), "label": "supplied Bennett-Chandler run"}
    if not math.isfinite(error) or error <= 0:
        reasons.append("independent-parent bootstrap has no usable variation")
    result.update(rate_cm3_s=estimate, parent_cluster_standard_error_cm3_s=error, reference=reference)
    if reasons:
        result["passed"] = False
        return result
    combined = math.hypot(error, reference["standard_error_cm3_s"])
    deviation = abs(estimate - reference["rate_cm3_s"]) / combined
    result.update(rate_comparison_performed=True, deviation_in_combined_standard_errors=deviation,
                  passed=deviation <= 3)
    if not result["passed"]:
        reasons.append("dRPMD and Bennett-Chandler differ by more than three combined standard errors")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--bc-run", type=Path, help="Use a completed BC run instead of the explicitly historical issue value")
    parser.add_argument("--smoke", action="store_true", help="Require reactive complete trajectories without a quantitative rate claim")
    args = parser.parse_args()
    report = compare(args.run, bc_run=args.bc_run, smoke=args.smoke)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["passed"] else 1)
