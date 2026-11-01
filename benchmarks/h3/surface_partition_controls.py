"""Fixed independent bead/domain controls for the equilibrium H3 QTST factor.

The completed P128 parent cohorts are immutable. These experiments use new
seeds, directories and protocol, and do not generate or extend child parents.
They test the equilibrium factor only, not bead convergence of recrossing.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.h3.classical_tst import MASS_AU
from benchmarks.h3.pes import BKMP2Potential
from benchmarks.h3.surface_partition import _provenance, reference_quadrature, run
from benchmarks.h3.surface_partition_campaign import DEFAULT_OUTPUT, TS_SEEDS, seed_interval, sha
from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.constants import KB_AU
from pyrpmd.io import write_json

CASES = (
    {"name": "p64-cutoff05", "beads": 64, "minimum": .5, "seeds": [826401, 826403, 826407, 826409]},
    {"name": "p256-cutoff05", "beads": 256, "minimum": .5, "seeds": [825601, 825603, 825607, 825609]},
    {"name": "p128-cutoff04", "beads": 128, "minimum": .4, "seeds": [820401, 820403, 820407, 820409]},
    {"name": "p128-cutoff06", "beads": 128, "minimum": .6, "seeds": [820601, 820603, 820607, 820609]},
)


def matching_reference(reference, beads, minimum, maximum=8.):
    rows = reference["bead_refinement"]+reference["wall_stress_checks"]
    selected = [row for row in rows if row["beads"] == beads
                and row["domain_bohr"] == [minimum, maximum]]
    if len(selected) != 1:
        raise ValueError("exactly one matching finite-P/domain reactant reference required")
    return selected[0]


def qtst_offset(log_ts_classical, log_reactant_classical, log_reactant_quantum_classical):
    """All control comparisons have the same temperature/mass/channel prefactor."""
    return log_ts_classical-log_reactant_classical-log_reactant_quantum_classical


def independent_difference(control, baseline, margin=np.log(1.05)):
    """Welch interval for independent control and baseline mean log estimates."""
    first, second = np.asarray(control, float), np.asarray(baseline, float)
    if (first.ndim != 1 or second.ndim != 1 or min(len(first), len(second)) < 2
            or not np.isfinite(first).all() or not np.isfinite(second).all() or margin <= 0):
        raise ValueError("two finite independent groups and positive margin required")
    v1, v0 = first.var(ddof=1)/len(first), second.var(ddof=1)/len(second)
    se = float(np.sqrt(v1+v0))
    degrees = ((v1+v0)**2/(v1*v1/(len(first)-1)+v0*v0/(len(second)-1))) if se else None
    half = float(student_t.ppf(.975, degrees)*se) if se else 0.
    difference = float(first.mean()-second.mean())
    bounds = [difference-half, difference+half]
    return {"mean_log_difference": difference, "standard_error": se,
            "Welch_degrees_of_freedom": float(degrees) if degrees is not None else None,
            "CI95_log_difference": bounds, "relative_point_difference": float(np.expm1(difference)),
            "CI95_relative_difference": np.expm1(bounds).tolist(),
            "equivalence_log_margin": float(margin),
            "CI95_wholly_within_margin": bool(bounds[0] >= -margin and bounds[1] <= margin),
            "uncertainty_unit": "independent seed chain in each group; shared baseline is not a new independent sample per control"}


def _primary_inputs(base):
    base = Path(base)
    primary = json.loads((base/"primary-protocol.json").read_text())
    summary = json.loads((base/"primary-summary.json").read_text())
    if not summary["complete"] or not summary["surfaces"]["ts"]["complete"]:
        raise ValueError("all registered primary seeds must finish before controls are registered")
    return primary, summary


def expected_settings(protocol, case, seed):
    return dict(protocol["primary_TS_settings_template"], seed=seed, beads=case["beads"],
                minimum_bead_pair_bohr=case["minimum"], snapshot_interval=0)


def register(output, library):
    output = Path(output)
    base = output.parent
    primary, summary = _primary_inputs(base)
    source = _provenance(BKMP2Potential(library))
    if source != primary["source_identity"]:
        raise ValueError("controls require the unchanged frozen primary sampler and PES")
    template_path = base/"primary/ts"/f"seed-{TS_SEEDS[0]}"/"settings.json"
    template = json.loads(template_path.read_text())
    reference_path = base.parent/"h3-quantum/diatom-transfer.json"
    reference = json.loads(reference_path.read_text())
    if (not reference["numerical_converged"] or reference["temperature_K"] != 300.
            or reference["reduced_mass_au"] != MASS_AU/2
            or reference["provenance"]["pes"] != source["model"]):
        raise ValueError("finite-P reactant oracle physical identity differs")
    path = output/"protocol.json"
    if path.exists():
        protocol = json.loads(path.read_text())
        checks = (protocol["source_identity"] == source, protocol["cases"] == list(CASES),
                  protocol["controls_source_sha256"] == sha(__file__),
                  protocol["primary_protocol_sha256"] == sha(base/"primary-protocol.json"),
                  protocol["primary_summary_sha256"] == sha(base/"primary-summary.json"),
                  protocol["primary_TS_settings_sha256"] == sha(template_path),
                  protocol["reactant_reference_sha256"] == sha(reference_path))
        if not all(checks):
            raise ValueError("registered controls/source/reference identity differs")
        return protocol
    provider = BKMP2Potential(library)
    beta = 1/(KB_AU*300.)
    classical = {}
    for minimum in (.4, .5, .6):
        rows = {}
        for surface in ("ts", "reactant"):
            coarse = reference_quadrature(provider, surface, beta, minimum=minimum,
                                           radial_order=192, angular_order=96)
            fine = reference_quadrature(provider, surface, beta, minimum=minimum,
                                         radial_order=256, angular_order=128)
            rows[surface] = {"log_classical_shape_integral": coarse.log_classical,
                             "log_reference_shape_integral": coarse.log_reference,
                             "log_plateau_probability": coarse.log_plateau-coarse.log_reference,
                             "refinement_log_difference": coarse.log_classical-fine.log_classical}
        classical[str(minimum)] = rows
    times = [json.loads((base/"primary/ts"/f"seed-{seed}"/"report.json").read_text())["elapsed_this_call_seconds"]
             for seed in TS_SEEDS]
    work_units = sum(len(case["seeds"])*case["beads"]/128 for case in CASES)
    protocol = {"schema": "pyrpmd.h3.equilibrium-bead-domain-controls.v1",
                "created_before_control_sampling_utc": datetime.now(timezone.utc).isoformat(),
                "phase": "registered independent equilibrium controls", "cases": list(CASES),
                "temperature_K": 300., "mass_au": MASS_AU, "warmup": 2000, "production": 20000,
                "lambda_grid": primary["lambda_grid"], "radius_step": .08, "ratio_step": .03,
                "alpha_R": .7, "alpha_r": .7, "swap_interval": 1,
                "sample_interval": 1, "diagnostic_interval": 10, "snapshot_interval": 0,
                "domain_samples": 131072, "maximum_centroid_bond_bohr": 8.,
                "radial_order": 192, "angular_order": 96, "checkpoint_interval": 1000,
                "seed_sampling_flags": primary["seed_sampling_flags"],
                "primary_TS_settings_template": template,
                "primary_TS_settings_sha256": sha(template_path),
                "classical_normalization": classical,
                "exact_reactant_rows": {case["name"]: matching_reference(reference, case["beads"], case["minimum"])
                                         for case in CASES},
                "base_exact_reactant_log_ratio": reference["primary_log_quantum_classical_ratio"],
                "equivalence_log_margin": float(np.log(1.05)),
                "equivalence_rule": "full independent-group 95% Welch interval inside +/-log(1.05)",
                "insufficient_power_rule": "Report diagnostic/inconclusive; no adaptation to literature point agreement.",
                "uncertainty_unit": "independent seed chain; exact reactant quadrature error checked separately",
                "scope": "Equilibrium QTST factor only; no recrossing or full-rate bead convergence claim.",
                "immutable_parent_rule": "Do not modify or extend any primary20k parent cohort or its provenance files.",
                "termination": "all16 new seeds reach their fixed counts, irrespective of literature proximity",
                "maximum_concurrent_CPUs": 2,
                "estimated_wall_seconds_from_completed_primary": float(np.mean(times)*work_units/2),
                "wall_estimate_assumptions": "Linear bead-count cost, two CPUs, same shared-host contention; excludes any queue delay.",
                "source_identity": source, "controls_source_sha256": sha(__file__),
                "primary_protocol_sha256": sha(base/"primary-protocol.json"),
                "primary_summary_sha256": sha(base/"primary-summary.json"),
                "reactant_reference_sha256": sha(reference_path),
                "primary_log_TS_corrections": summary["surfaces"]["ts"]["log_quantum_classical_ratio"]["individual_seed_values"]}
    output.mkdir(parents=True, exist_ok=True)
    write_json(path, protocol)
    return protocol


def execute_seed(output, library, case_name, seed, cpu):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
    protocol = register(output, library)
    case = next((case for case in protocol["cases"] if case["name"] == case_name), None)
    if case is None or seed not in case["seeds"]:
        raise ValueError("unregistered equilibrium control")
    path = Path(output)/case_name/f"seed-{seed}"
    if (path/"report.json").exists():
        old = json.loads((path/"report.json").read_text())
        if old["complete"]:
            if json.loads((path/"settings.json").read_text()) != expected_settings(protocol, case, seed):
                raise ValueError("completed control settings differ from the registered experiment")
            if sha(path/"checkpoint.npz") != old["checkpoint_sha256"]:
                raise ValueError("completed control checkpoint changed")
            with load_checkpoint(path/"checkpoint.npz", require_integrity=True):
                pass
            return
    run(path, library, surface="ts", beads=case["beads"], minimum=case["minimum"], seed=seed,
        warmup=protocol["warmup"], production=protocol["production"],
        lambdas=np.asarray(protocol["lambda_grid"]), radius_step=.08, ratio_step=.03,
        alpha_R=.7, alpha_r=.7, swap_interval=1, domain_samples=131072,
        snapshot_interval=0, checkpoint_interval=1000, phase="production",
        resume=(path/"settings.json").exists())


def analyze(output, library):
    protocol = register(output, library)
    classical = protocol["classical_normalization"]
    base_offset = qtst_offset(classical["0.5"]["ts"]["log_classical_shape_integral"],
        classical["0.5"]["reactant"]["log_classical_shape_integral"], protocol["base_exact_reactant_log_ratio"])
    baseline = np.asarray(protocol["primary_log_TS_corrections"])+base_offset
    rows = []
    for case in protocol["cases"]:
        seeds = []
        for seed in case["seeds"]:
            path = Path(output)/case["name"]/f"seed-{seed}"
            if not (path/"report.json").exists():
                continue
            report = json.loads((path/"report.json").read_text())
            if not report["complete"] or not report["source_unchanged"]:
                continue
            if json.loads((path/"settings.json").read_text()) != expected_settings(protocol, case, seed):
                raise ValueError("control settings differ from the registered experiment")
            if sha(path/"checkpoint.npz") != report["checkpoint_sha256"]:
                raise ValueError("control checkpoint/report changed")
            analysis = report["analysis"]
            seeds.append({"seed": seed, "log_TS_quantum_classical_ratio": analysis["log_quantum_classical_ratio"],
                          "minimum_BAR_overlap": analysis["minimum_BAR_overlap"],
                          "minimum_energy_ESS": min(x for x in analysis["diagnostic_energy_effective_samples"] if x is not None),
                          "half_log_difference": analysis["second_half_log_ratio"]-analysis["first_half_log_ratio"],
                          "checkpoint_sha256": report["checkpoint_sha256"], "report_sha256": sha(path/"report.json")})
        row = {**case, "completed_seed_results": seeds, "complete": len(seeds) == len(case["seeds"])}
        if len(seeds) >= 2:
            law = classical[str(case["minimum"])]
            offset = qtst_offset(law["ts"]["log_classical_shape_integral"],
                law["reactant"]["log_classical_shape_integral"],
                protocol["exact_reactant_rows"][case["name"]]["log_quantum_classical_ratio"])
            corrected = np.asarray([seed["log_TS_quantum_classical_ratio"] for seed in seeds])+offset
            row["independent_seed_statistics"] = seed_interval(corrected)
            row["relative_QTST_comparison"] = independent_difference(corrected, baseline,
                                                                       protocol["equivalence_log_margin"])
            row["paired_seed_half_log_difference"] = seed_interval([s["half_log_difference"] for s in seeds])
            row["classical_and_exact_reactant_log_offset"] = offset
        flags = protocol["seed_sampling_flags"]
        row["seed_sampling_flags_passed"] = all(
            s["minimum_BAR_overlap"] >= flags["minimum_BAR_overlap"]
            and s["minimum_energy_ESS"] >= flags["minimum_energy_ESS"]
            and abs(s["half_log_difference"]) <= flags["maximum_absolute_half_log_difference"] for s in seeds)
        rows.append(row)
    complete = all(row["complete"] for row in rows)
    quadrature = max(abs(value["refinement_log_difference"]) for entry in classical.values() for value in entry.values())
    passed = complete and quadrature < 1e-7 and all(
        row["relative_QTST_comparison"]["CI95_wholly_within_margin"] and row["seed_sampling_flags_passed"] for row in rows)
    result = {"schema": protocol["schema"], "complete": complete, "controls": rows,
              "baseline_seeds": list(TS_SEEDS), "baseline_log_QTST_without_common_prefactor": baseline,
              "maximum_classical_quadrature_log_change": quadrature,
              "equilibrium_control_equivalence_gates_passed": bool(passed),
              "protocol_sha256": sha(Path(output)/"protocol.json"),
              "physical_converged": False, "literature_reproduced": False,
              "limitations": ["Only equilibrium QTST factors are compared; no recrossing children are generated for these controls.",
                              "Failure of a 5% equivalence interval may mean a resolved change or insufficient statistical power.",
                              "All comparisons share the same frozen eight-seed baseline; they are not independent of one another."]}
    write_json(Path(output)/"comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT/"controls")
    parser.add_argument("--library", type=Path, default=ROOT/"benchmarks/h3/build/libbkmp2.so")
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--execute-seed", type=int)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    protocol = register(args.output, args.library)
    if args.register_only:
        print(json.dumps({"protocol": str(args.output/"protocol.json"), "sha256": sha(args.output/"protocol.json"),
                          "estimated_wall_seconds": protocol["estimated_wall_seconds_from_completed_primary"]}), flush=True)
        return
    if args.execute_seed is not None:
        execute_seed(args.output, args.library, args.case, args.execute_seed, args.cpu)
        return
    if not args.analyze_only:
        cores = sorted(os.sched_getaffinity(0))[-2:] if hasattr(os, "sched_getaffinity") else [0]
        tasks = [(case["name"], seed) for case in protocol["cases"] for seed in case["seeds"]]
        def worker(index):
            for case_name, seed in tasks[index::len(cores)]:
                path = args.output/case_name
                path.mkdir(parents=True, exist_ok=True)
                with (path/f"seed-{seed}.log").open("a") as stream:
                    subprocess.run([sys.executable, __file__, "--output", str(args.output), "--library", str(args.library),
                        "--case", case_name, "--execute-seed", str(seed), "--cpu", str(cores[index])],
                        stdout=stream, stderr=subprocess.STDOUT, check=True)
                print(json.dumps({"completed_case": case_name, "seed": seed}), flush=True)
        with ThreadPoolExecutor(max_workers=len(cores)) as pool:
            list(pool.map(worker, range(len(cores))))
    result = analyze(args.output, args.library)
    print(json.dumps({"complete": result["complete"], "equilibrium_gates": result["equilibrium_control_equivalence_gates_passed"]}), flush=True)


if __name__ == "__main__":
    main()
