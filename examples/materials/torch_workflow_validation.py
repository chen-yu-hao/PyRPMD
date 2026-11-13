"""Reproduce tensor-workflow harmonic statistics and CPU batch timings.

Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1. Optional Torch must be
installed. Timings include trajectory/checkpoint I/O and result analysis;
they are not GPU measurements or molecular physical-convergence evidence.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import tempfile
import time

import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from pyrpmd.system import System
from pyrpmd.workflows.materials import run_materials


def specification(*, beads=16, replicas=32, dt=.5, neq=2000, nprod=8000, interval=10):
    return {"task": "pimd", "beta": 4.0, "beads": beads, "dt_au": dt, "seed": 79,
            "potential": {"provider": "harmonic", "params": {"k": 2.0}},
            "sampling": {"engine": "torch", "device": "cpu", "integrator": "bcocb",
                         "thermostat": "pile-l", "gamma_centroid": 1.0,
                         "equilibration_steps": neq, "production_steps": nprod,
                         "sample_interval": interval, "replicas": replicas},
            "dynamics": {"integrator": "cayley-bcb", "block_steps": 256},
            "parallel": {"backend": "serial", "threads_per_rank": 1},
            "output": {"checkpoint_steps": 2000, "chunk_frames": 128}}


def harmonic_case(output, beads):
    config = specification(beads=beads)
    system = System(("H",), np.array([2.0]), np.zeros((1, 3)))
    result = run_materials(config, system, output, return_data=True)
    beta, mass, stiffness = config["beta"], system.masses[0], 2.0
    ring = 2 * beads / beta * np.sin(np.pi * np.arange(beads) / beads)
    omega = np.sqrt(stiffness / mass)
    variance = np.sum(1 / (omega**2 + ring**2)) / (beta * mass)
    target = np.array([1.5 * stiffness * variance] * 2 + [3 * stiffness * variance])
    means = np.array([stream["observables"][:, [0, 2, 3]].mean(axis=0) for stream in result["streams"]])
    mean, sem = means.mean(axis=0), means.std(axis=0, ddof=1) / np.sqrt(len(means))
    radius = float(np.mean([stream["radii_gyration_squared"].mean() for stream in result["streams"]]))
    return {"beads": beads, "replicas": len(means), "equilibration_steps": 2000,
            "production_steps": 8000, "dt_au": config["dt_au"], "beta_au": beta,
            "names": ["potential_energy", "kinetic_centroid_virial", "internal_energy_centroid_virial"],
            "mean_hartree": mean.tolist(), "sem_between_replicas_hartree": sem.tolist(),
            "exact_finite_beads_hartree": target.tolist(), "relative_error": (mean / target - 1).tolist(),
            "quantum_continuum_internal_energy_hartree": float(1.5 * omega / np.tanh(beta * omega / 2)),
            "radius_gyration_squared_bohr2": radius,
            "exact_radius_gyration_squared_bohr2": float(3 * (variance - 1 / (beta * stiffness))),
            "within_six_standard_errors": bool(np.all(np.abs(mean - target) <= 6 * sem))}


def timing_case(directory, replicas, repeats):
    config = specification(beads=32, replicas=replicas, neq=64, nprod=256, interval=16)
    system = System(("H", "H", "H"), np.full(3, 2.0), np.zeros((3, 3)))
    samples = {engine: [] for engine in ("legacy", "torch")}
    for repeat in range(repeats):
        order = ("legacy", "torch") if repeat % 2 == 0 else ("torch", "legacy")
        for engine in order:
            current = copy.deepcopy(config)
            current["sampling"]["engine"] = engine
            if engine == "legacy":
                current["sampling"].pop("device")
            start_wall, start_cpu = time.perf_counter(), time.process_time()
            run_materials(current, system, directory / f"{replicas}-{repeat}-{engine}")
            samples[engine].append({"wall_seconds": time.perf_counter() - start_wall,
                                    "process_cpu_seconds": time.process_time() - start_cpu})
    medians = {engine: {key: statistics.median(sample[key] for sample in values)
                        for key in ("wall_seconds", "process_cpu_seconds")}
               for engine, values in samples.items()}
    return {"replicas": replicas, "beads": 32, "atoms": 3, "steps_per_replica": 320,
            "repeats": repeats, "samples": samples, "median": medians,
            "wall_speedup": medians["legacy"]["wall_seconds"] / medians["torch"]["wall_seconds"],
            "cpu_speedup": medians["legacy"]["process_cpu_seconds"] / medians["torch"]["process_cpu_seconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, default=Path("."))
    parser.add_argument("--hardware", default=platform.processor() or "not reported")
    parser.add_argument("--beads", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or min(args.beads + args.batches) < 1:
        parser.error("sizes and repeat count must be positive")
    args.scratch.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1), tempfile.TemporaryDirectory(dir=args.scratch) as temporary:
        directory = Path(temporary)
        # Exclude import, allocator and FFT initialization from timings.
        warmup = specification(beads=4, replicas=2, neq=2, nprod=4, interval=1)
        run_materials(warmup, System(("H",), np.array([2.0]), np.zeros((1, 3))), directory / "warmup")
        report = {"schema": "pyrpmd.torch-materials-validation.v1",
                  "created_utc": datetime.now(timezone.utc).isoformat(), "hardware": args.hardware,
                  "platform": platform.platform(), "python": platform.python_version(),
                  "numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__),
                  "torch_num_threads": torch.get_num_threads(), "threadpools": threadpool_info(),
                  "scope": "fixed-cell float64 CPU harmonic workflow; no GPU or molecular convergence claim",
                  "timing_scope": "end-to-end PIMD including construction, immutable trajectory/checkpoint I/O and analysis; warmed imports",
                  "harmonic": [harmonic_case(directory / f"harmonic-{beads}", beads) for beads in args.beads],
                  "timings": [timing_case(directory, batch, args.repeats) for batch in args.batches]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"harmonic_max_relative_errors": [max(map(abs, case["relative_error"])) for case in report["harmonic"]],
                          "workflow_wall_speedups": [case["wall_speedup"] for case in report["timings"]],
                          "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
