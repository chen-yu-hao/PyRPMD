"""Registered independent-seed finite-P dividing-surface partition campaign."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
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
from benchmarks.h3.pes import BKMP2Potential
from benchmarks.h3.surface_partition import _provenance, run
from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.io import write_json

TS_SEEDS = (721301, 721303, 721307, 721309, 721313, 721317, 721321, 721323)
REACTANT_SEEDS = (721401, 721403, 721407, 721409)
DEFAULT_OUTPUT = ROOT/"docs/validation/literature-convergence-2026-11-01/surface-partition"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design(library):
    return {
        "schema": "pyrpmd.h3.surface-partition-primary.v1", "phase": "registered primary",
        "ts_seeds": list(TS_SEEDS), "reactant_normalization_control_seeds": list(REACTANT_SEEDS),
        "beads": 128, "temperature_K": 300., "warmup_sweeps": 2000,
        "production_sweeps": 20000, "lambda_grid": (np.linspace(0., 1., 9)**2).tolist(),
        "minimum_bead_pair_bohr": .5, "maximum_centroid_bond_bohr": 8.,
        "radius_step": .08, "ratio_step": .03, "alpha_R": .7,
        "alpha_r_by_surface": {"ts": .7, "reactant": .45},
        "swap_interval": 1, "sample_interval": 1, "diagnostic_interval": 10,
        "domain_samples_per_independent_seed": 131072,
        "reference_radial_order": 192, "reference_angular_order": 96,
        "checkpoint_interval": 1000, "TS_snapshot_interval": 100,
        "TS_snapshots_per_seed": 200, "TS_snapshot_completed_sweeps": list(range(2100, 22001, 100)),
        "TS_snapshot_source": "static equilibrium Monte Carlo at lambda=1; total COM modes analytically marginalized",
        "TS_snapshot_keys": {"endpoint_q": "float64 (200,128,3,3), bohr; active centroid r01=r12<=r02",
                             "endpoint_completed_sweeps": "int64 (200,), completed Monte Carlo sweeps including warmup"},
        "TS_snapshot_selection": "retain every registered parent; no extra sampler RNG; child channel/momenta use separate RNG",
        "uncertainty": "equal-weight mean seed log ratios; two-sided 95% Student-t CI with n_seed-1 degrees of freedom",
        "temporal_diagnostics": "paired seed differences of first/second half log ratios, endpoint radius and mean bead energy",
        "seed_sampling_flags": {"minimum_BAR_overlap": .2, "minimum_energy_ESS": 200,
                                "maximum_absolute_half_log_difference": .2},
        "estimator": "adjacent-lambda BAR; trapezoid and Simpson TI only diagnostic cross-checks",
        "reactant_primary_denominator": "independent finite-P radial-transfer result; stochastic reactant is a validation control",
        "calibration_excluded": ["calibration-ts", "calibration-reactant", "efficiency-ts", "efficiency-reactant",
                                  "ts-importance.npz", "reactant-importance.npz"],
        "termination": "fixed counts for every seed; no stopping or extension based on a literature-rate point",
        "maximum_concurrent_CPUs": 2,
        "source_identity": _provenance(BKMP2Potential(library)),
        "campaign_source_sha256": sha(__file__),
        "test_source_sha256": sha(ROOT/"tests/test_h3_surface_partition.py"),
    }


def register(output, library):
    output = Path(output)
    expected = design(library)
    path = output/"primary-protocol.json"
    if path.exists():
        actual = json.loads(path.read_text())
        actual.pop("created_before_primary_sampling_utc")
        if actual != expected:
            raise ValueError("registered primary source/protocol differs; preserve the original campaign")
    else:
        write_json(path, dict(expected, created_before_primary_sampling_utc=datetime.now(timezone.utc).isoformat()))
    return expected


def seed_path(output, surface, seed):
    return Path(output)/"primary"/surface/f"seed-{seed}"


def execute_seed(output, library, surface, seed, cpu):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
    protocol = register(output, library)
    if seed not in (TS_SEEDS if surface == "ts" else REACTANT_SEEDS):
        raise ValueError("unregistered seed")
    path = seed_path(output, surface, seed)
    report_path = path/"report.json"
    if report_path.exists():
        old = json.loads(report_path.read_text())
        if old["complete"]:
            if old["checkpoint_sha256"] != sha(path/"checkpoint.npz"):
                raise ValueError("completed primary checkpoint differs from its report")
            with load_checkpoint(path/"checkpoint.npz", require_integrity=True):
                pass
            return
    report = run(path, library, surface=surface, beads=protocol["beads"], seed=seed,
                 warmup=protocol["warmup_sweeps"], production=protocol["production_sweeps"],
                 lambdas=np.asarray(protocol["lambda_grid"]), radius_step=protocol["radius_step"],
                 ratio_step=protocol["ratio_step"], alpha_R=protocol["alpha_R"],
                 alpha_r=protocol["alpha_r_by_surface"][surface], swap_interval=protocol["swap_interval"],
                 domain_samples=protocol["domain_samples_per_independent_seed"],
                 snapshot_interval=protocol["TS_snapshot_interval"] if surface == "ts" else 0,
                 checkpoint_interval=protocol["checkpoint_interval"], phase="production",
                 resume=(path/"settings.json").exists())
    print(json.dumps({"surface": surface, "seed": seed, "complete": report["complete"],
                      "endpoint_count": report["endpoint_snapshot_count"],
                      "checkpoint": str(path/"checkpoint.npz")}), flush=True)


def seed_interval(values):
    values = np.asarray(values, float)
    mean = float(values.mean())
    se = float(values.std(ddof=1)/np.sqrt(len(values))) if len(values) > 1 else None
    half = float(student_t.ppf(.975, len(values)-1)*se) if se is not None else None
    return {"seed_count": len(values), "mean": mean, "standard_error": se,
            "CI95": [mean-half, mean+half] if half is not None else None,
            "uncertainty_unit": "independent seed chain", "individual_seed_values": values.tolist()}


def analyze(output, library):
    protocol = register(output, library)
    results = {}
    for surface, seeds in (("ts", TS_SEEDS), ("reactant", REACTANT_SEEDS)):
        rows = []
        for seed in seeds:
            path = seed_path(output, surface, seed)
            if not (path/"report.json").exists():
                continue
            report = json.loads((path/"report.json").read_text())
            if not report["complete"]:
                continue
            if sha(path/"checkpoint.npz") != report["checkpoint_sha256"]:
                raise ValueError("checkpoint/report checksum differs")
            with load_checkpoint(path/"checkpoint.npz", require_integrity=True) as saved:
                diag = saved["diagnostics"][:, -1]
                half = len(diag)//2
                radius_difference = float(diag[half:, 0].mean()-diag[:half, 0].mean())
                energy_difference = float(diag[half:, 3].mean()-diag[:half, 3].mean())
            a = report["analysis"]
            rows.append({"seed": seed, "log_ratio": a["log_quantum_classical_ratio"],
                         "half_log_difference": a["second_half_log_ratio"]-a["first_half_log_ratio"],
                         "endpoint_radius_half_difference_bohr": radius_difference,
                         "endpoint_energy_half_difference_hartree": energy_difference,
                         "minimum_BAR_overlap": a["minimum_BAR_overlap"],
                         "minimum_energy_ESS": min(x for x in a["diagnostic_energy_effective_samples"] if x is not None),
                         "endpoint_snapshot_count": report["endpoint_snapshot_count"],
                         "checkpoint_sha256": report["checkpoint_sha256"]})
        if rows:
            results[surface] = {"complete": len(rows) == len(seeds), "seeds": rows,
                "log_quantum_classical_ratio": seed_interval([x["log_ratio"] for x in rows]),
                "paired_seed_half_log_difference": seed_interval([x["half_log_difference"] for x in rows]),
                "paired_seed_radius_half_difference_bohr": seed_interval([x["endpoint_radius_half_difference_bohr"] for x in rows]),
                "paired_seed_energy_half_difference_hartree": seed_interval([x["endpoint_energy_half_difference_hartree"] for x in rows])}
    transfer = ROOT/"docs/validation/literature-convergence-2026-11-01/h3-quantum/diatom-transfer.json"
    exact = json.loads(transfer.read_text())
    if "reactant" in results:
        control = results["reactant"]["log_quantum_classical_ratio"]
        target = exact["primary_log_quantum_classical_ratio"]
        results["reactant"]["finite_P_reference_log_ratio"] = target
        results["reactant"]["difference_from_finite_P_reference"] = control["mean"]-target
        results["reactant"]["CI95_contains_finite_P_reference"] = (
            control["CI95"][0] <= target <= control["CI95"][1] if control["CI95"] is not None else None)
    summary = {"schema": protocol["schema"], "complete": all(results.get(s, {}).get("complete", False) for s in ("ts", "reactant")),
               "surfaces": results, "protocol_sha256": sha(Path(output)/"primary-protocol.json"),
               "finite_P_reactant_reference_sha256": sha(transfer),
               "physical_converged": False, "literature_reproduced": False,
               "limitations": ["Fixed-surface equilibrium partition only; recrossing and bead/domain refinement remain separate.",
                               "Temporal difference CIs are mixing diagnostics, not tests of exact equivalence.",
                               "A partially completed campaign must not be treated as the registered final estimate."]}
    write_json(Path(output)/"primary-summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--library", type=Path, default=ROOT/"benchmarks/h3/build/libbkmp2.so")
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--execute-seed", type=int)
    parser.add_argument("--surface", choices=("ts", "reactant"))
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    register(args.output, args.library)
    if args.register_only:
        return
    if args.execute_seed is not None:
        execute_seed(args.output, args.library, args.surface, args.execute_seed, args.cpu)
        return
    if not args.analyze_only:
        available = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [0]
        cores = available[-2:]
        def worker(index):
            tasks = []
            for position in range(index, len(TS_SEEDS), len(cores)):
                tasks.append(("ts", TS_SEEDS[position]))
                if position < len(REACTANT_SEEDS):
                    tasks.append(("reactant", REACTANT_SEEDS[position]))
            for surface, seed in tasks:
                path = seed_path(args.output, surface, seed)
                path.mkdir(parents=True, exist_ok=True)
                # Log is outside the run directory because run() requires a new
                # directory to contain no unrelated preexisting files.
                log = path.parent/f"seed-{seed}.log"
                with log.open("a") as stream:
                    subprocess.run([sys.executable, __file__, "--output", str(args.output), "--library", str(args.library),
                        "--execute-seed", str(seed), "--surface", surface, "--cpu", str(cores[index])],
                        stdout=stream, stderr=subprocess.STDOUT, check=True)
                print(json.dumps({"completed_surface": surface, "seed": seed, "checkpoint": str(path/"checkpoint.npz")}), flush=True)
        with ThreadPoolExecutor(max_workers=len(cores)) as pool:
            list(pool.map(worker, range(len(cores))))
    summary = analyze(args.output, args.library)
    print(json.dumps({"complete": summary["complete"], "summary": str(args.output/"primary-summary.json")}), flush=True)


if __name__ == "__main__":
    main()
