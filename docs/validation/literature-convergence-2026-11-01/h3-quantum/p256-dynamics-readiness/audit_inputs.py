"""Read-only inventory of the four frozen P256 equilibrium control states.

No potential is constructed or evaluated, no random numbers are drawn, and
no sampler/dynamics function is called. Only this audit's new output is written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import scipy
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
from pyrpmd.checkpoint import load_checkpoint

EVIDENCE = ROOT / "docs/validation/literature-convergence-2026-11-01"
SEEDS = (825601, 825603, 825607, 825609)
CHANGED_ARRAYS = ("fingerprint", "endpoint_q", "endpoint_completed_sweeps")


def identity(path):
    path = Path(path)
    raw = path.read_bytes()
    try:
        name = str(path.relative_to(ROOT))
    except ValueError:
        name = str(path)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def array_identity(array):
    array = np.asarray(array)
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "bytes": int(array.nbytes),
            "content_sha256": hashlib.sha256(header + b"\n" + array.tobytes(order="C")).hexdigest()}


def final_geometry(radius, ratio, eta_R, eta_r):
    cosine = 1 - ratio * ratio / 2
    centroid = np.array([[radius, 0., 0.], [0., 0., 0.],
                         [radius * cosine, radius * np.sqrt(max(0., 1 - cosine * cosine)), 0.]])
    R = centroid[0] - (centroid[1] + centroid[2]) / 2 + eta_R
    relative = centroid[2] - centroid[1] + eta_r
    q = np.stack((2 * R / 3, -R / 3 - relative / 2, -R / 3 + relative / 2), axis=-2)
    center = q.mean(axis=0)
    distances = [np.linalg.norm(q[:, a] - q[:, b], axis=1)
                 for a, b in ((0, 1), (0, 2), (1, 2))]
    d01, d02, d12 = (float(np.linalg.norm(center[a] - center[b]))
                     for a, b in ((0, 1), (0, 2), (1, 2)))
    return {"geometry_reconstructible": True, "shape": list(q.shape),
            "minimum_bead_pair_bohr": float(np.min(distances)),
            "centroid_pair_distances_bohr": {"01": d01, "02": d02, "12": d12},
            "absolute_TS_surface_error_bohr": abs(d01 - d12),
            "maximum_bead_COM_coordinate_bohr": float(np.max(abs(q.mean(axis=1)))),
            "reconstructible_historical_parent_count": 1,
            "interpretation": "Only the final lambda=1 state; not a saved parent sequence."}


def audit():
    started = time.perf_counter()
    cpu_started = resource.getrusage(resource.RUSAGE_SELF)
    controls = EVIDENCE / "surface-partition/controls"
    protocol_path = controls / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    case = next(row for row in protocol["cases"] if row["name"] == "p256-cutoff05")
    assert tuple(case["seeds"]) == SEEDS and case["beads"] == 256 and case["minimum"] == .5
    current_runtime = {"python": platform.python_version(), "numpy": np.__version__,
                       "scipy": scipy.__version__, "platform": platform.platform()}
    watched = [protocol_path, controls / "p256-case-assessment.json",
               ROOT / "benchmarks/h3/surface_partition_controls.py",
               EVIDENCE / "surface-partition/primary-protocol.json",
               EVIDENCE / "h3-quantum/diatom-transfer.json"]
    initial_identities = {str(path): identity(path) for path in watched}
    source_checks = []
    for name, digest in protocol["source_identity"]["source_sha256"].items():
        path = ROOT / name
        row = identity(path)
        initial_identities[str(path)] = dict(row)
        row["registered_sha256"] = digest
        row["matches_registered"] = row["sha256"] == digest
        assert row["matches_registered"], name
        source_checks.append(row)
        watched.append(path)
    assert identity(watched[2])["sha256"] == protocol["controls_source_sha256"]
    runtime_matches = current_runtime == protocol["source_identity"]["runtime"]
    rows = []
    for seed in SEEDS:
        directory = controls / case["name"] / f"seed-{seed}"
        paths = {name: directory / name for name in (
            "checkpoint.npz", "settings.json", "report.json", "reference-domain.json",
            "reference.json", "progress.json")}
        watched.extend(paths.values())
        ids = {name: identity(path) for name, path in paths.items()}
        initial_identities.update({str(paths[name]): row for name, row in ids.items()})
        settings = json.loads(paths["settings.json"].read_text())
        report = json.loads(paths["report.json"].read_text())
        expected = dict(protocol["primary_TS_settings_template"], seed=seed, beads=256,
                        minimum_bead_pair_bohr=.5, snapshot_interval=0)
        assert settings == expected
        assert report["complete"] and report["source_unchanged"]
        assert report["seed"] == seed and report["phase"] == "production"
        assert report["endpoint_snapshot_count"] == 0
        assert ids["checkpoint.npz"]["sha256"] == report["checkpoint_sha256"]
        expected_fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        with load_checkpoint(paths["checkpoint.npz"], expected_kind="h3.surface-partition",
                             require_integrity=True) as saved:
            arrays = {name: array_identity(saved[name]) for name in saved.files}
            assert str(saved["fingerprint"].item()) == expected_fingerprint
            assert str(saved["domain_sha256"].item()) == ids["reference-domain.json"]["sha256"]
            assert int(saved["completed"]) == 22000
            assert saved["endpoint_q"].shape == (0, 256, 3, 3)
            assert saved["endpoint_completed_sweeps"].shape == (0,)
            assert saved["delta_energy"].shape == (20000, 9)
            assert saved["diagnostics"].shape == (2000, 9, 8)
            assert np.isfinite(saved["delta_energy"]).all()
            assert np.isfinite(saved["diagnostics"]).all()
            final = final_geometry(float(saved["radius"][-1]), float(saved["ratio"][-1]),
                                   saved["eta_R"][-1], saved["eta_r"][-1])
            rng = json.loads(str(saved["rng"].item()))
        assert final["minimum_bead_pair_bohr"] >= .5
        assert final["absolute_TS_surface_error_bohr"] < 1e-10
        assert final["maximum_bead_COM_coordinate_bohr"] < 1e-10
        rows.append({"seed": seed, "files": ids, "complete": True,
                     "settings_exactly_match_registered_control": True,
                     "integrity_envelope_verified": True,
                     "settings_fingerprint": expected_fingerprint,
                     "all_array_identities": arrays,
                     "bitwise_replay_required_arrays": [name for name in arrays if name not in CHANGED_ARRAYS],
                     "snapshot_count": 0, "snapshot_interval": 0,
                     "gradients_saved": False, "Maxwell_momenta_saved": False,
                     "RNG_bit_generator": rng["bit_generator"],
                     "last_lambda1_geometry": final,
                     "original_sampling_wall_seconds": report["elapsed_this_call_seconds"],
                     "original_process_CPU_seconds": None,
                     "BAR_log_TS_quantum_classical_ratio": report["analysis"]["log_quantum_classical_ratio"]})
    base = EVIDENCE / "h3-quantum/recrossing-stage-200-pairs-2"
    comparison_path = base / "quantum-rate-comparison.json"
    initial_identities[str(comparison_path)] = identity(comparison_path)
    comparison = json.loads(comparison_path.read_text())
    assert comparison["pair_batches"] == [0]
    assert len(comparison["completed_seeds"]) == 8
    watched.append(comparison_path)
    timings = []
    for path in sorted((base / "per-seed").glob("seed-*/dt-*/report.json")):
        initial_identities[str(path)] = identity(path)
        report = json.loads(path.read_text())
        assert report["complete"] and report["pairs_per_parent"] == 2
        assert report["parents_completed"] == 200
        watched.append(path)
        timings.append({**identity(path), "seed": report["seed"],
                        "dt_fs": report["timestep_fs"],
                        "native_steps": report["native_steps_this_stage"],
                        "original_worker_wall_seconds": report["elapsed_this_call_seconds"]})
    assert len(timings) == 16
    total_time = sum(row["original_worker_wall_seconds"] for row in timings)
    total_steps = sum(row["native_steps"] for row in timings)
    last_half = [row for row in timings if row["seed"] in (721313, 721317, 721321, 721323)]
    later_ratio = (sum(row["original_worker_wall_seconds"] for row in last_half) /
                   sum(row["native_steps"] for row in last_half))
    statistics = comparison["statistics"]
    sigma_kappa = statistics["kappa_statistics"]["standard_error"] * np.sqrt(8)
    sigma_log_rate = statistics["log_rate_statistics"]["standard_error"] * np.sqrt(8)
    df = (1 / 4 + 1 / 8)**2 / ((1 / 4)**2 / 3 + (1 / 8)**2 / 7)
    half_kappa = float(student_t.ppf(.975, 3) * sigma_kappa / 2)
    precision = {
        "basis": "Illustration only: assume P256 per-chain variance equals completed two-pair P128 variance. No active extra-pair data read.",
        "P128_kappa_per_chain_sd": float(sigma_kappa),
        "P128_log_rate_per_chain_sd": float(sigma_log_rate),
        "predicted_P256_four_chain_kappa_95_halfwidth_absolute": half_kappa,
        "predicted_P256_four_chain_kappa_95_halfwidth_relative": half_kappa / statistics["kappa_arithmetic_mean"],
        "equal_variance_four_vs_eight_Welch_df": float(df),
        "predicted_four_vs_eight_kappa_difference_95_halfwidth_absolute":
            float(student_t.ppf(.975, df) * sigma_kappa * np.sqrt(1 / 4 + 1 / 8)),
        "predicted_four_vs_eight_log_rate_difference_95_halfwidth":
            float(student_t.ppf(.975, df) * sigma_log_rate * np.sqrt(1 / 4 + 1 / 8)),
        "planned_full_rate_equivalence_margin_log": float(np.log(1.05)),
        "not_a_power_certificate": True,
        "four_pairs_per_parent_may_reduce_momentum_noise_but_not_parent_chain_variance": True}
    # Explicitly verify the inspected frozen inputs again before writing a receipt.
    unchanged = all(identity(path) == row for path, row in initial_identities.items())
    assert unchanged
    cpu = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "schema": "pyrpmd.h3.p256-parent-readiness-readonly-audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_source": identity(__file__), "input_files": list(initial_identities.values()),
        "current_runtime": current_runtime, "runtime_matches_original": runtime_matches,
        "registered_source_checks": source_checks, "frozen_inputs_unchanged_after_read": unchanged,
        "cohorts": rows,
        "currently_saved_parent_configurations": 0,
        "available_final_lambda1_states": 4,
        "replay_target_parents": 800,
        "replay_completed_sweeps": list(range(2100, 22001, 100)),
        "original_P256_timed_sampling_wall_seconds_total": sum(row["original_sampling_wall_seconds"] for row in rows),
        "original_P256_process_CPU_seconds_total": None,
        "timing_limitation": "The original producer used perf_counter after reference/domain initialization and before final BAR analysis. CPU/rusage and whole-call time were not recorded. Its one-CPU affinity bounds CPU only for the timed region; setup/analysis and contention require new measurements.",
        "completed_P128_base_timing_reports": timings,
        "completed_P128_base_total_native_steps": total_steps,
        "completed_P128_base_total_worker_wall_seconds": total_time,
        "P256_child_cost_extrapolation": [
            {"parents_per_seed": 200, "independent_chains": 4, "pairs_per_parent": pairs,
             "dt_fs": [.1, .05], "horizon_fs": 200,
             "native_steps": int(4 * 200 * pairs * 2 * (200 / .1 + 200 / .05)),
             "worker_wall_seconds_linear_bead_from_all_base":
                float(4 * 200 * pairs * 2 * (200 / .1 + 200 / .05) * 2 * total_time / total_steps),
             "worker_wall_seconds_linear_bead_from_later_four_base_seeds":
                float(4 * 200 * pairs * 2 * (200 / .1 + 200 / .05) * 2 * later_ratio),
             "measurement": "not measured at P256; linear bead-cost extrapolation from full immutable P128 base reports"}
            for pairs in (2, 4)],
        "four_vs_eight_precision_illustration": precision,
        "no_new_science_executed": {"PES_evaluations": 0, "Monte_Carlo_sweeps": 0,
                                    "dynamics_steps": 0, "RNG_draws": 0},
        "audit_execution": {"wall_seconds": time.perf_counter() - started,
                            "process_user_CPU_seconds": cpu.ru_utime - cpu_started.ru_utime,
                            "process_system_CPU_seconds": cpu.ru_stime - cpu_started.ru_stime,
                            "peak_process_RSS_KiB": cpu.ru_maxrss},
        "dynamic_bead_convergence_demonstrated": False,
        "ready_for_existing_primary_child_driver": False,
        "readiness_reason": "No saved P256 parent sequence, and the existing child driver validates the separate P128 primary parent protocol. A new reviewed P256 replay/parent protocol and child binding are needed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(identity(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
