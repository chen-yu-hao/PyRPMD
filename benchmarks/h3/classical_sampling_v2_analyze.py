"""Compare excluded kernel-calibration runs by variance times CPU cost.

No published value or distance to an oracle enters kernel selection. Exact
window moments are diagnostic checks only. Calibration blocks remain
correlated within full ladders and are not promoted to primary replicas.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.h3.classical_sampling import umbrella_moments
from benchmarks.h3.classical_sampling_campaign import file_hash
from benchmarks.h3.classical_sampling_v2 import restored_state
from benchmarks.h3.mixing_analysis import analyze_run, histogram_samples, rate_candidate, solve_wham
from pyrpmd.constants import KB_AU
from pyrpmd.io import write_json


def analyze(data, reference, output):
    data, reference, output = Path(data), Path(reference), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((data/"protocol.json").read_text())
    schedule = json.loads((data/"schedule.json").read_text())
    if file_hash(data/"schedule.json") != protocol["schedule_sha256"]:
        raise ValueError("calibration schedule checksum mismatch")
    if file_hash(reference) != protocol["reference_sha256"]:
        raise ValueError("calibration oracle checksum mismatch")
    beta = 1/(KB_AU*300.)
    centers, stiffness = np.array(schedule["centers"]), np.array(schedule["stiffness"])
    with np.load(reference, allow_pickle=False) as oracle:
        exact_mean, exact_variance = umbrella_moments(oracle["xi"], oracle["density"], centers,
                                                     beta=beta, stiffness=stiffness)
    n = protocol["production_sweeps"]
    block_size = protocol["analysis"]["block_sweeps"]
    if n % block_size:
        raise ValueError("registered block size must divide the calibration budget")
    width = protocol["analysis"]["histogram_bin_width"]
    halfwidth = protocol["analysis"]["local_log_density_fit_halfwidth"]
    candidates = []
    for candidate in protocol["candidates"]:
        xi_moves = candidate["xi_moves"]
        records = []
        for seed in candidate["seeds"]:
            run = data/f"xi-{xi_moves}"/f"seed-{seed}"
            settings = json.loads((run/"settings.json").read_text())
            execution = json.loads((run/"run-report.json").read_text())
            if (not execution["completed"] or not execution["source_unchanged"]
                    or execution["production_sweeps"] != n
                    or settings["source_sha256"]["benchmarks/h3/classical_sampling_v2.py"] != protocol["sampler_source_sha256"]):
                raise ValueError("calibration run is incomplete or has an unregistered source")
            state = restored_state(run/"checkpoint.npz", settings)
            if file_hash(run/"checkpoint.npz") != execution["checkpoint_sha256"]:
                raise ValueError("completed calibration checkpoint hash mismatch")
            samples = state["samples"]
            report, fit = analyze_run(run)
            target = output/f"xi-{xi_moves}"/f"seed-{seed}"
            target.mkdir(parents=True, exist_ok=True)
            write_json(target/"wham-report.json", report)
            np.savez_compressed(target/"wham-fit.npz", **fit)
            blocks = []
            for start in range(0, n, block_size):
                histogram, edges = histogram_samples(samples[start:start+block_size], width=width)
                block_fit = solve_wham(histogram, edges, centers, stiffness, beta)
                estimate = rate_candidate(block_fit, beta, halfwidth=halfwidth)
                blocks.append({"start": start, "stop": start+block_size,
                               "overlap_connected": block_fit["overlap_connected"], **estimate})
            diagnostic_se = np.sqrt(samples.var(axis=0)*np.asarray(report["cv_inefficiencies"])/n)
            z = (samples.mean(axis=0)-exact_mean)/diagnostic_se
            primary = next(row for row in report["estimates"] if row["bin_width"] == width)
            record = {"seed": seed, "xi_moves": xi_moves, "log_rate_ratio": primary["log_density_ratio"],
                      "qtst_cm3_molecule_s": primary["qtst_cm3_molecule_s"],
                      "blocks": blocks, "cpu_elapsed_seconds": execution["cpu_elapsed_seconds"],
                      "elapsed_seconds": execution["elapsed_seconds"],
                      "xi_acceptance_active_windows": execution["xi_acceptance_active_windows"],
                      "oracle_mean_diagnostic_z": z, "variance_ratio_to_oracle": samples.var(axis=0)/exact_variance,
                      "cv_inefficiencies": report["cv_inefficiencies"],
                      "overlap_connected": all(row["overlap_connected"] for row in report["estimates"]),
                      "checkpoint_sha256": execution["checkpoint_sha256"],
                      "settings_sha256": file_hash(run/"settings.json"),
                      "primary_campaign_observations": False}
            write_json(target/"calibration-record.json", record)
            records.append(record)
            print(json.dumps({"calibration_completed": seed, "xi_moves": xi_moves,
                              "qtst": record["qtst_cm3_molecule_s"]}), flush=True)
        logs = np.array([row["log_rate_ratio"] for row in records])
        block_logs = np.array([b["log_density_ratio"] for row in records for b in row["blocks"]])
        full_variance_scale = float(logs.var(ddof=1)*n)
        block_variance_scale = float(block_logs.var(ddof=1)*block_size)
        variance_scale = max(full_variance_scale, block_variance_scale)
        cpu_per_sweep = float(np.mean([r["cpu_elapsed_seconds"] for r in records])/(n+protocol["warmup_sweeps"]))
        forecasts = {}
        for replicas in (4, 8, 16):
            required = variance_scale*student_t.ppf(.975, replicas-1)**2/(replicas*np.log1p(.01)**2)
            rounded = 2**int(np.ceil(np.log2(max(1, required))))
            forecasts[str(replicas)] = {"production_sweeps_per_seed_rough_estimate": float(required),
                                        "rounded_up_power_of_two": rounded,
                                        "total_process_cpu_hours": rounded*replicas*cpu_per_sweep/3600}
        candidates.append({"xi_moves": xi_moves, "independent_seeds": candidate["seeds"],
                           "whole_seed_log_variance_times_sweeps": full_variance_scale,
                           "block_log_variance_times_sweeps": block_variance_scale,
                           "conservative_log_variance_times_sweeps": variance_scale,
                           "process_cpu_seconds_per_sweep": cpu_per_sweep,
                           "variance_cpu_cost_product": variance_scale*cpu_per_sweep,
                           "rough_one_percent_precision_forecast": forecasts,
                           "maximum_abs_oracle_mean_diagnostic_z": float(max(np.max(np.abs(r["oracle_mean_diagnostic_z"])) for r in records)),
                           "all_overlap_connected": all(r["overlap_connected"] for r in records),
                           "per_seed": records})
    control = next(row for row in candidates if row["xi_moves"] == 0)
    best = min((row for row in candidates if row["xi_moves"] > 0), key=lambda row: row["variance_cpu_cost_product"])
    gain = control["variance_cpu_cost_product"]/best["variance_cpu_cost_product"]
    report = {"schema": "pyrpmd.h3.classical-excluded-calibration-comparison.v2",
              "candidates": candidates, "best_measured_xi_moves": best["xi_moves"],
              "measured_efficiency_gain_vs_control": gain,
              "registered_minimum_efficiency_gain_met": gain >= protocol["analysis"]["minimum_measured_efficiency_gain_to_start_primary"],
              "selection_uses_a_paper_value_or_reference_error": False,
              "calibration_excluded_from_all_primary_estimates": True,
              "source_sha256": file_hash(Path(__file__)),
              "analysis_source_sha256": file_hash(ROOT/"benchmarks/h3/mixing_analysis.py"),
              "protocol_sha256": file_hash(data/"protocol.json"), "reference_sha256": file_hash(reference),
              "scope": "Efficiency calibration only. Rough variance forecasts from four seeds have substantial uncertainty; no main result is validated by this report."}
    write_json(output/"calibration-comparison.json", report)
    print(json.dumps({key: report[key] for key in ("best_measured_xi_moves", "measured_efficiency_gain_vs_control",
                                                  "registered_minimum_efficiency_gain_met")}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.data, args.reference, args.output)
