"""Chain-level analysis of independently normalized quantum H3 RPMD rates.

The same chain supplies its BAR surface partition and its child parents.
Their covariance is retained by combining log(Z_TS) and log(kappa) before
forming the eight-chain interval. Individual children are never independent
uncertainty units. Partial cohorts cannot satisfy the registered gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.h3.quantum_surface_recrossing import (
    CHUNK_PARENTS, EVIDENCE, PAIRS, PARENT_PROTOCOL, fingerprint, load_parents,
    read_chunk, run_directory, sha,
)
from pyrpmd.io import write_json

PAPER = {"doi": "10.1016/j.cpc.2012.10.017", "table": 2, "temperature_K": 300.,
         "beads": 128, "qtst_cm3_molecule_s": 2.51e-16, "kappa": .776,
         "rate_cm3_molecule_s": 1.95e-16, "reported_approximate_relative_precision": .05,
         "precision_note": "The paper's approximately 5% sampling precision is not a separately reconstructed modern 95% CI."}


def interval(values):
    values = np.asarray(values, float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite independent chain values required")
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    half = float(student_t.ppf(.975, len(values)-1) * se)
    return {"mean": mean, "standard_error": se, "CI95": [mean-half, mean+half],
            "halfwidth95": half, "degrees_of_freedom": len(values)-1}


def paired_rate_statistics(log_ts_correction, kappa, log_reactant_correction, classical_tst):
    log_ts = np.asarray(log_ts_correction, float)
    kappa = np.asarray(kappa, float)
    if log_ts.shape != kappa.shape or np.any(kappa <= 0) or classical_tst <= 0:
        raise ValueError("positive same-chain transmission and classical TST are required")
    log_kappa = np.log(kappa)
    log_qtst = np.log(classical_tst) + log_ts - log_reactant_correction
    log_rate = log_qtst + log_kappa
    q, k, rate = interval(log_qtst), interval(kappa), interval(log_rate)
    covariance = np.cov(np.stack((log_ts, log_kappa)), ddof=1) / len(log_ts)
    variance_identity = float(covariance.sum())
    if not np.isclose(variance_identity, rate["standard_error"]**2, rtol=1e-10, atol=1e-18):
        raise ValueError("same-chain covariance identity failed")
    return {
        "qtst_geometric_mean_cm3_molecule_s": float(np.exp(q["mean"])),
        "qtst_CI95_cm3_molecule_s": np.exp(q["CI95"]).tolist(), "log_qtst_statistics": q,
        "kappa_arithmetic_mean": k["mean"], "kappa_CI95": k["CI95"],
        "kappa_statistics": k, "kappa_geometric_mean": float(np.exp(log_kappa.mean())),
        "rate_geometric_mean_cm3_molecule_s": float(np.exp(rate["mean"])),
        "rate_CI95_cm3_molecule_s": np.exp(rate["CI95"]).tolist(), "log_rate_statistics": rate,
        "same_chain_log_partition_log_kappa_covariance_of_mean": covariance.tolist(),
        "paired_log_rate_variance": variance_identity,
        "incorrect_independence_log_rate_variance_for_comparison_only": float(np.trace(covariance)),
        "per_seed_log_qtst": log_qtst.tolist(), "per_seed_log_rate": log_rate.tolist(),
        "central_estimator": "geometric mean of the independent-chain rate estimates; kappa arithmetic mean is also reported",
        "uncertainty_unit": "complete independent parent chain, including its correlated partition and all children",
    }


def equivalent_paired_difference(first, second, margin):
    result = interval(np.asarray(first)-np.asarray(second))
    result["absolute_margin"] = float(margin)
    result["CI95_wholly_within_margin"] = bool(result["CI95"][0] >= -margin and result["CI95"][1] <= margin)
    return result


def validate_reactant_reference(reference, protocol, parent_protocol):
    """A deterministic number is usable only for this same Hamiltonian."""
    rows = [row for row in reference["bead_refinement"] if row["beads"] == protocol["beads"]]
    if len(rows) != 1:
        raise ValueError("exactly one matching finite-P reactant row required")
    row = rows[0]
    beta_from_kernel = row["free_kernel_width_bohr"]**2 * reference["reduced_mass_au"] * row["beads"]
    checks = (reference["numerical_converged"], reference["primary_beads"] == protocol["beads"],
              reference["temperature_K"] == protocol["temperature_K"],
              reference["reduced_mass_au"] == protocol["mass_au"]/2,
              reference["provenance"]["pes"] == protocol["model"],
              row["domain_bohr"] == [protocol["static_domain_minimum_pair_bohr"], parent_protocol["maximum_centroid_bond_bohr"]],
              np.isclose(beta_from_kernel, protocol["beta"], rtol=1e-14, atol=0),
              row["log_quantum_classical_ratio"] == reference["primary_log_quantum_classical_ratio"])
    if not all(checks):
        raise ValueError("reactant reference temperature/mass/P/domain/model/normalization does not match")


def completed_seed(data, seed, horizon, batches):
    for dt in (.1, .05):
        for batch in batches:
            path = run_directory(data, seed, dt, horizon, batch) / "report.json"
            if not path.exists() or not json.loads(path.read_text())["complete"]:
                return False
    return True


def load_seed(data, protocol, seed, timestep, horizon, batches, parent_identity):
    total_numerator = np.zeros(horizon)
    total_denominator = 0.
    half_numerator = np.zeros((2, horizon))
    half_denominator = np.zeros(2)
    receipts, momentum_hashes = [], []
    minimum, maximum_energy, crossings, steps = np.inf, 0., 0, 0
    for batch in batches:
        run = run_directory(data, seed, timestep, horizon, batch)
        settings = json.loads((run / "settings.json").read_text())
        report = json.loads((run / "report.json").read_text())
        expected = {"phase": "primary", "seed": seed, "timestep_fs": timestep, "horizon_fs": horizon,
                    "pair_batch": batch, "beads": protocol["beads"],
                    "parents": protocol["parents_per_seed"], "parent_identity": parent_identity,
                    "source_sha256": protocol["source_sha256"], "protocol_sha256": fingerprint(protocol),
                    "momentum_pair_indices": list(range(PAIRS*batch, PAIRS*(batch+1)))}
        if any(settings.get(key) != value for key, value in expected.items()):
            raise ValueError("child settings/parent identity differ from the registered cohort")
        if not report["complete"] or not report["source_unchanged"]:
            raise ValueError("complete unchanged child runs required")
        if report["settings_sha256"] != sha(run / "settings.json"):
            raise ValueError("child report/settings checksum mismatch")
        inventory = {item["file"]: item["sha256"] for item in report["chunks"]}
        names = {f"chunk-{left:04d}.npz" for left in range(0, protocol["parents_per_seed"], CHUNK_PARENTS)}
        if set(inventory) != names or len(inventory) != len(report["chunks"]):
            raise ValueError("registered child chunk inventory is incomplete or duplicated")
        seed_native_steps = 0
        for left in range(0, protocol["parents_per_seed"], CHUNK_PARENTS):
            path = run / f"chunk-{left:04d}.npz"
            if sha(path) != inventory[path.name]:
                raise ValueError("child chunk changed after its completed report")
            indices = np.arange(left, min(left+CHUNK_PARENTS, protocol["parents_per_seed"]))
            saved = read_chunk(path, fingerprint(settings), indices)
            if not np.array_equal(saved["time_fs"], np.arange(1, horizon+1)):
                raise ValueError("child observation time grid differs")
            if not np.array_equal(saved["momentum_pair_index"], expected["momentum_pair_indices"]):
                raise ValueError("child momentum pair identity differs")
            numerator = saved["numerator"]
            denominator = saved["denominator"]
            shape = (len(indices), PAIRS)
            if numerator.shape != shape+(horizon,) or denominator.shape != shape:
                raise ValueError("child flux array shape differs")
            # Replay the stored signed-flux estimator. This is independent of
            # the trajectory file's reported kappa and reported plateau fits.
            sign = ((saved["surface_values"][:, :, 0] > 0).astype(float)
                    - (saved["surface_values"][:, :, 1] > 0))
            replay = saved["velocity"][..., None] / saved["flux_metric"][:, None, None] * sign
            replay_denominator = abs(saved["velocity"]) / saved["flux_metric"][:, None]
            if not np.array_equal(numerator, replay) or not np.array_equal(denominator, replay_denominator):
                raise ValueError("stored paired flux estimator does not replay exactly")
            total_numerator += numerator.sum(axis=(0, 1))
            total_denominator += float(denominator.sum())
            for half in (0, 1):
                selected = (indices < protocol["parents_per_seed"]//2) if half == 0 else (indices >= protocol["parents_per_seed"]//2)
                half_numerator[half] += numerator[selected].sum(axis=(0, 1))
                half_denominator[half] += float(denominator[selected].sum())
            minimum = min(minimum, float(saved["minimum_pair_at_1fs"].min()))
            crossings += int(np.sum(saved["minimum_pair_at_1fs"].min(axis=-1) < protocol["static_domain_minimum_pair_bohr"]))
            extrema = saved["energy_extrema"]
            error = np.maximum(abs(extrema[..., 1]-extrema[..., 0]), abs(extrema[..., 2]-extrema[..., 0]))
            maximum_energy = max(maximum_energy, float(error.max()*protocol["beta"]/protocol["beads"]))
            seed_native_steps += int(saved["native_steps_this_stage"])
            # Initial p and channel are compared across timestep pairs below.
            import hashlib
            momentum_hashes.append({"batch": batch, "chunk": path.name,
                "initial_p": hashlib.sha256(saved["initial_p"].tobytes()).hexdigest(),
                "channel": hashlib.sha256(saved["channel"].tobytes()).hexdigest()})
        if seed_native_steps != report["native_steps_this_stage"] or report["python_steps"] != 0:
            raise ValueError("native child accounting differs")
        steps += report["unique_native_steps_all_stages"]
        receipts.append({"batch": batch, "report_sha256": sha(run / "report.json"),
                         "settings_sha256": sha(run / "settings.json"), "chunks": report["chunks"]})
    curve = total_numerator / total_denominator
    half_curves = half_numerator / half_denominator[:, None]
    times = np.arange(1, horizon+1)
    plateaus = {f"{lo}-{hi}": float(curve[(times >= lo) & (times <= hi)].mean())
                for lo, hi in protocol["plateau_windows_fs"] if hi <= horizon}
    half_plateaus = {f"{lo}-{hi}": half_curves[:, (times >= lo) & (times <= hi)].mean(axis=1).tolist()
                     for lo, hi in protocol["plateau_windows_fs"] if hi <= horizon}
    return {"seed": seed, "timestep_fs": timestep, "horizon_fs": horizon,
            "plateaus": plateaus, "parent_time_half_plateaus": half_plateaus,
            "mean_denominator": total_denominator/(protocol["parents_per_seed"]*PAIRS*len(batches)),
            "minimum_pair_at_1fs_bohr": minimum, "children_with_sampled_domain_crossing": crossings,
            "maximum_energy_error_kbt_per_bead": maximum_energy, "unique_native_steps": steps,
            "momentum_pairing_hashes": momentum_hashes, "curve": curve, "receipts": receipts}


def analyze(data, output, *, horizon=200, batches=(0,), available_only=False):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((data / "protocol.json").read_text())
    protocol.pop("created_before_primary_children_utc")
    if sha(PARENT_PROTOCOL) != protocol["parent_protocol_sha256"]:
        raise ValueError("surface parent protocol changed")
    if tuple(batches) not in ((0,), (0, 1)) or horizon not in (200, 400):
        raise ValueError("only registered time and pair-batch refinements are allowed")
    seeds = [seed for seed in protocol["seeds"] if completed_seed(data, seed, horizon, batches)]
    if not available_only and seeds != protocol["seeds"]:
        raise ValueError("the complete eight-chain cohort is not ready")
    if len(seeds) < 2:
        raise ValueError("at least two complete independent chains are needed for a diagnostic interval")
    complete = seeds == protocol["seeds"]
    baseline = json.loads((EVIDENCE / "h3-classical/transmission/transmission-comparison.json").read_text())
    transfer = json.loads((EVIDENCE / "h3-quantum/diatom-transfer.json").read_text())
    parent_protocol = json.loads(PARENT_PROTOCOL.read_text())
    validate_reactant_reference(transfer, protocol, parent_protocol)
    classical_tst = baseline["independent_classical_tst_cm3_molecule_s"]
    log_ts, parent_reports, by_dt = [], [], {"0.1": [], "0.05": []}
    for seed in seeds:
        directory = PARENT_PROTOCOL.parent / "primary/ts" / f"seed-{seed}"
        _, _, parent_identity = load_parents(directory, protocol)
        parent = json.loads((directory / "report.json").read_text())
        parent_reports.append(parent)
        log_ts.append(parent_identity["BAR_log_TS_quantum_classical_ratio"])
        for dt in (.1, .05):
            by_dt[str(dt)].append(load_seed(data, protocol, seed, dt, horizon, batches, parent_identity))
        if by_dt["0.1"][-1]["momentum_pairing_hashes"] != by_dt["0.05"][-1]["momentum_pairing_hashes"]:
            raise ValueError("timestep runs did not use identical keyed momenta/channels")
    last = "180-200" if horizon == 200 else "380-400"
    prior = "80-100" if horizon == 200 else "180-200"
    primary_rows = by_dt[str(protocol["primary_timestep_fs"])]
    kappas = [row["plateaus"][last] for row in primary_rows]
    statistics = paired_rate_statistics(log_ts, kappas, transfer["primary_log_quantum_classical_ratio"], classical_tst)
    plateau = equivalent_paired_difference(kappas, [row["plateaus"][prior] for row in primary_rows],
                                          protocol["plateau_equivalence_absolute_margin"])
    timestep = equivalent_paired_difference([row["plateaus"][last] for row in by_dt["0.1"]], kappas,
                                           protocol["timestep_equivalence_absolute_margin"])
    parent_time = equivalent_paired_difference([row["parent_time_half_plateaus"][last][1] for row in primary_rows],
        [row["parent_time_half_plateaus"][last][0] for row in primary_rows], .03)
    # This last 0.03 interval is a labeled additional stationarity diagnostic,
    # not an unregistered substitute for the fixed plateau/precision gates.
    parent_time["scope"] = "Supplementary independent-chain paired first/second parent-half diagnostic."
    normalization = interval([row["mean_denominator"] for row in primary_rows])
    rate = statistics["rate_geometric_mean_cm3_molecule_s"]
    qrate = statistics["qtst_geometric_mean_cm3_molecule_s"]
    rate_difference = rate/PAPER["rate_cm3_molecule_s"]-1.
    rate_ci = statistics["rate_CI95_cm3_molecule_s"]
    flags = parent_protocol["seed_sampling_flags"]
    source_flags = []
    for parent in parent_reports:
        analysis = parent["analysis"]
        source_flags.append({"seed": parent["seed"],
            "BAR_overlap": analysis["minimum_BAR_overlap"] >= flags["minimum_BAR_overlap"],
            "energy_ESS": min(x for x in analysis["diagnostic_energy_effective_samples"] if x is not None) >= flags["minimum_energy_ESS"],
            "half_log_difference": abs(analysis["second_half_log_ratio"]-analysis["first_half_log_ratio"]) <= flags["maximum_absolute_half_log_difference"]})
    criteria = {"complete_registered_cohort": complete,
                "parent_sampling_flags": all(all(v for k, v in row.items() if k != "seed") for row in source_flags),
                "time_plateau_equivalence": plateau["CI95_wholly_within_margin"],
                "timestep_equivalence": timestep["CI95_wholly_within_margin"],
                "kappa_statistical_precision": statistics["kappa_statistics"]["halfwidth95"]/statistics["kappa_arithmetic_mean"] <= protocol["precision_gate_relative_kappa_95_halfwidth"],
                "no_sampled_static_domain_crossings": all(row["children_with_sampled_domain_crossing"] == 0 for rows in by_dt.values() for row in rows),
                "primary_maximum_energy_error_at_most_01_kbt_per_bead": max(row["maximum_energy_error_kbt_per_bead"] for row in primary_rows) <= .1,
                "full_rate_95_log_halfwidth_at_most_5_percent": statistics["log_rate_statistics"]["halfwidth95"] <= np.log(1.05),
                "paper_full_rate_point_within_reported_5_percent": abs(rate_difference) <= .05,
                "paper_qtst_point_within_reported_5_percent": abs(qrate/PAPER["qtst_cm3_molecule_s"]-1.) <= .05,
                "paper_kappa_point_within_reported_5_percent": abs(statistics["kappa_arithmetic_mean"]/PAPER["kappa"]-1.) <= .05}
    curves = np.asarray([row["curve"] for row in primary_rows])
    curve_mean = curves.mean(axis=0)
    curve_half = student_t.ppf(.975, len(seeds)-1)*curves.std(axis=0, ddof=1)/np.sqrt(len(seeds))
    for rows in by_dt.values():
        for row in rows:
            row.pop("curve")
    report = {"schema": "pyrpmd.h3.independent-quantum-rate-comparison.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "expected_seeds": protocol["seeds"], "completed_seeds": seeds, "horizon_fs": horizon,
              "pair_batches": list(batches), "primary_plateau_fs": last, "statistics": statistics,
              "time_plateau_comparison": plateau, "paired_timestep_comparison": timestep,
              "parent_sampling_time_halves": parent_time, "mean_denominator_diagnostic": normalization,
              "parent_sampling_flags": source_flags, "criteria": criteria,
              "fixed_P_independent_rate_gates_passed": bool(all(criteria.values())),
              "paper": PAPER, "relative_full_rate_difference_from_paper": rate_difference,
              "relative_qtst_difference_from_paper": qrate/PAPER["qtst_cm3_molecule_s"]-1.,
              "paper_full_rate_point_inside_new_CI95": bool(rate_ci[0] <= PAPER["rate_cm3_molecule_s"] <= rate_ci[1]),
              "new_CI95_wholly_inside_paper_5_percent_band": bool(rate_ci[0] >= .95*PAPER["rate_cm3_molecule_s"] and rate_ci[1] <= 1.05*PAPER["rate_cm3_molecule_s"]),
              "classical_tst_reference_cm3_molecule_s": classical_tst,
              "finite_P_reactant_log_quantum_classical_ratio": transfer["primary_log_quantum_classical_ratio"],
              "per_timestep_per_seed": by_dt, "protocol_sha256": sha(data/"protocol.json"),
              "parent_protocol_sha256": sha(PARENT_PROTOCOL), "analysis_source_sha256": sha(__file__),
              "reactant_reference_sha256": sha(EVIDENCE/"h3-quantum/diatom-transfer.json"),
              "classical_reference_sha256": sha(EVIDENCE/"h3-classical/transmission/transmission-comparison.json"),
              "production_umbrella_validated_by_this_result": False,
              "energy_gate_scope": "Maximum absolute full-H error times beta/P <=0.1 at the primary 0.05-fs step; a numerical stability screen, not a bound on rate bias. The paired timestep-equivalence test remains separate.",
              "bead_and_domain_refinements": "Separate fixed-protocol equilibrium controls; their evidence is not replaced by these child checks.",
              "limitations": ["The minimum-pair record covers 1-fs boundaries, not every physical step.",
                              "Intervals describe this sampling experiment; they exclude PES/model uncertainty.",
                              "A completed fixed-P reference rate does not validate unrelated reactions or the production MD umbrella sampler."]}
    write_json(output/"quantum-rate-comparison.json", report)
    np.savetxt(output/"kappa-curve.csv", np.column_stack((np.arange(1, horizon+1), curve_mean, curve_mean-curve_half, curve_mean+curve_half)),
               delimiter=",", header="time_fs,kappa_seed_mean,pointwise_CI95_low,pointwise_CI95_high", comments="")
    print(json.dumps({"seeds": seeds, "rate": rate, "rate_CI95": rate_ci,
                      "criteria": {key: bool(value) for key, value in criteria.items()}}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=(200, 400), default=200)
    parser.add_argument("--include-extra-pairs", action="store_true")
    parser.add_argument("--available-only", action="store_true")
    args = parser.parse_args()
    analyze(args.data, args.output, horizon=args.horizon,
            batches=(0, 1) if args.include_extra_pairs else (0,), available_only=args.available_only)
