#!/usr/bin/env python3
"""Matched real BKMP2 child tasks: setup reuse and cross-trajectory batching.

This is a propagation/performance regression, not a thermal reaction rate.
Every variant starts from the same constrained coordinates and per-pair RNG.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np

from pyrpmd.constants import AMU_TO_AU, FS_TO_AU, KB_AU
from pyrpmd.dynamics import native_capabilities
from pyrpmd.parallel import run_tasks
from pyrpmd.potentials.python import import_object
from pyrpmd.system import System
from pyrpmd.workflows.batched_recrossing import _child_batch_task
from pyrpmd.workflows.rate import _child_task, project_positions, rate_coordinate


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "serial_reload": {"mode": "serial", "reuse": False, "batch_pairs": 1},
    "serial_reuse": {"mode": "serial", "reuse": True, "batch_pairs": 1},
    "bead2_reload": {"mode": "bead", "reuse": False, "batch_pairs": 1},
    "bead2_reuse": {"mode": "bead", "reuse": True, "batch_pairs": 1},
    "batch4": {"mode": "serial", "reuse": True, "batch_pairs": 4},
    "batch8": {"mode": "serial", "reuse": True, "batch_pairs": 8},
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration(library, beads=16, steps=128, pairs=8):
    geometry = np.array([[-1.757, 0., 0.], [0., 0., 0.], [1.757, 0., 0.]])
    system = System(("H", "H", "H"), np.full(3, 1.00782503223*AMU_TO_AU),
                    geometry, ((0,), (1, 2)), multiplicity=2)
    config = {
        "task": "rate", "temperature_K": 1000., "beta": 1/(1000*KB_AU),
        "beads": beads, "dt_au": .1*FS_TO_AU, "seed": 41873, "native": "required",
        "potential": {"provider": "python", "factory": str(ROOT/"benchmarks/h3/pes.py")+":build",
                      "params": {"library": str(library)}},
        "sampling": {}, "dynamics": {"integrator": "cayley-bcb", "loop_backend": "auto", "block_steps": 128},
        "reaction": {"fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
                     "breaking_bonds": [[1, 2]], "rinf": 10.},
        "recrossing": {"constraint_tolerance": 1e-10},
    }
    surface = rate_coordinate(config, system).dividing_surface(1.)
    rng = np.random.default_rng(80413)
    tasks = []
    for pair in range(pairs):
        fluctuations = rng.normal(scale=.035, size=(beads, 3, 3))
        fluctuations -= fluctuations.mean(axis=0)
        q = project_positions(np.repeat(geometry[None], beads, axis=0)+fluctuations,
                              surface, system.masses)
        tasks.append({"config": config, "system": system, "chain": pair % 2,
                      "parent": pair//2, "pair": pair, "q": q, "xi_star": 1.,
                      "child_steps": steps})
    return config, system, tasks


def run(library, output, *, beads=16, steps=128, pairs=8, repeats=3, cases=None):
    library = Path(library).resolve()
    config, system, templates = configuration(library, beads, steps, pairs)
    potential = import_object(config["potential"]["factory"])(library=library)
    provenance = {"metadata": potential.metadata, "build": potential.provenance}
    potential.close()
    source_paths = ["src/pyrpmd/batched.py", "src/pyrpmd/dynamics.py", "src/pyrpmd/parallel/execution.py",
                    "src/pyrpmd/workflows/rate.py", "src/pyrpmd/workflows/batched_recrossing.py",
                    "src/pyrpmd/propagation.py", "benchmarks/h3/pes.py", "benchmarks/tier1_recrossing.py"]
    selected = list(CASES) if cases is None else list(cases)
    if not selected or any(case not in CASES for case in selected):
        raise ValueError("unknown/empty benchmark case selection")
    results = {case: {"settings": CASES[case], "seconds": [], "comparison": []} for case in selected}
    reference = None
    with tempfile.TemporaryDirectory(prefix="pyrpmd-tier1-recrossing-") as temporary:
        for repeat in range(repeats):
            # Rotate case order between repeats to distribute warm-machine effects.
            order = selected[repeat % len(selected):]+selected[:repeat % len(selected)]
            for case in order:
                settings = CASES[case]
                current = copy.deepcopy(config)
                parallel = {"backend": "serial", "mode": settings["mode"],
                            "trajectory_workers": 1, "threads_per_rank": 1,
                            "bead_workers": 2 if settings["mode"] == "bead" else 1,
                            "reuse_potentials": settings["reuse"]}
                current["parallel"] = parallel
                current["recrossing"]["child_batch_pairs"] = settings["batch_pairs"]
                directory = Path(temporary)/f"{case}-{repeat}"
                tasks = [dict(task, config=current, output=str(directory)) for task in templates]
                # Timed scope includes potential/model loading, worker-pool
                # startup/shutdown, propagation and normal checkpoint writes.
                started = time.perf_counter()
                if settings["batch_pairs"] > 1:
                    grouped = [tasks[i:i+settings["batch_pairs"]] for i in range(0, len(tasks), settings["batch_pairs"])]
                    nested = run_tasks(_child_batch_task, grouped, parallel)
                    children = [child for group in nested for child in group]
                else:
                    children = run_tasks(_child_task, tasks, parallel)
                elapsed = time.perf_counter()-started
                if reference is None:
                    reference = children
                denominator_error = max(abs(a["denominator"]-b["denominator"]) for a, b in zip(children, reference))
                numerator_error = max(float(np.max(np.abs(a["numerator"]-b["numerator"]))) for a, b in zip(children, reference))
                endpoint_error_difference = max(abs(a["endpoint_energy_error_hartree_per_bead"]
                                                     -b["endpoint_energy_error_hartree_per_bead"])
                                                for a, b in zip(children, reference))
                if denominator_error != 0 or numerator_error != 0 or endpoint_error_difference > 5e-12:
                    raise AssertionError(f"child results changed for {case}: "
                                         f"denominator={denominator_error}, numerator={numerator_error}, "
                                         f"endpoint={endpoint_error_difference}")
                results[case]["seconds"].append(elapsed)
                results[case]["comparison"].append({
                    "maximum_absolute_denominator_difference": denominator_error,
                    "maximum_absolute_numerator_difference": numerator_error,
                    "maximum_endpoint_energy_error_difference_hartree_per_bead": endpoint_error_difference,
                    "maximum_endpoint_energy_error_hartree_per_bead": max(
                        child["endpoint_energy_error_hartree_per_bead"] for child in children),
                    "native_loop_steps": sum(child["propagation"].get("native_steps", 0) for child in children),
                    "python_loop_steps": sum(child["propagation"].get("python_steps", 0) for child in children),
                })
                print(f"{case} repeat {repeat+1}/{repeats}: {elapsed:.6f} s; "
                      f"numerator difference {numerator_error:.2g}", flush=True)
    for row in results.values():
        row["median_seconds"] = float(np.median(row["seconds"]))
    if "serial_reuse" in results:
        baseline = results["serial_reuse"]["median_seconds"]
        for row in results.values():
            row["speedup_vs_serial_reuse"] = baseline/row["median_seconds"]
    reuse_speedups = {}
    for mode in ("serial", "bead2"):
        if mode+"_reload" in results and mode+"_reuse" in results:
            reuse_speedups[mode] = results[mode+"_reload"]["median_seconds"]/results[mode+"_reuse"]["median_seconds"]
    report = {
        "schema": "pyrpmd.tier1.recrossing-performance.v1", "passed": True,
        "scope": "Matched short BKMP2 child propagation and setup cost, not a thermal-rate prediction",
        "physical_rate_converged": False, "gpu_benchmark": False,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "platform": platform.platform(), "logical_cpus": os.cpu_count(),
                    "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                    "native": native_capabilities()},
        "workload": {"temperature_K": 1000., "beads": beads, "atoms": 3, "dt_fs": .1,
                     "child_steps": steps, "child_pairs": pairs, "trajectories": 2*pairs,
                     "repeats": repeats, "dividing_surface_xi": 1., "rinf_bohr": 10.,
                     "momentum_seed": config["seed"], "coordinate_seed": 80413,
                     "coordinate_maximum_surface_residual": max(abs(rate_coordinate(config, system).dividing_surface(1.).value(t["q"])[0]) for t in templates),
                     "initial_coordinate_sha256": hashlib.sha256(np.stack([t["q"] for t in templates]).tobytes()).hexdigest()},
        "potential": provenance,
        "source_sha256": {path: sha256(ROOT/path) for path in source_paths},
        "native_extension_sha256": sha256(Path(__import__("pyrpmd._native", fromlist=["__file__"]).__file__)),
        "variants": results, "setup_reuse_speedup": reuse_speedups,
        "reference_denominators": [r["denominator"] for r in reference],
        "reference_numerator_sha256": hashlib.sha256(np.stack([r["numerator"] for r in reference]).tobytes()).hexdigest(),
        "limitations": [
            "Starts are constrained near the TS, not equilibrium parent samples; no thermal rate can be inferred.",
            "BKMP2 is inexpensive; fully fused scalar C++ may beat Python-orchestrated batches.",
            "Reload=False/True contrast removes per-task factory/pool setup, not a large ML-model load benchmark.",
            "One task worker and, for bead cases, two bead worker processes; no GPU or external cluster.",
            "Timing includes normal checkpoint writes and is hardware/load dependent.",
        ],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(output), "setup_reuse_speedup": reuse_speedups,
                      "median_seconds": {case: row["median_seconds"] for case, row in results.items()}}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beads", type=int, default=16)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cases", nargs="+", choices=list(CASES))
    args = parser.parse_args()
    for name in ("beads", "steps", "pairs", "repeats"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    run(**vars(args))


if __name__ == "__main__":
    main()
