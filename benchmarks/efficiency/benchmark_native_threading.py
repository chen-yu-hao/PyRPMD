"""Native ring-polymer propagation scaling and determinism benchmark.

Run once per requested OpenMP thread count, for example::

  OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 \\
    python benchmarks/efficiency/benchmark_native_threading.py \\
      --threads 4 --output native-threading-4.json

The timed region is ``Dynamics.propagate`` with an analytic harmonic PES and a
single final observation. Construction and warm-up are excluded. This measures
software execution only; it is not evidence of physical sampling convergence.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
from pathlib import Path
import statistics
import time

import numpy as np

from pyrpmd import Dynamics, State
from pyrpmd.dynamics import native_capabilities
from pyrpmd.potentials import HarmonicPotential


def hardware(threads: int) -> dict:
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cpu = next(
            (line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
             if line.startswith("model name")),
            cpu,
        )
    quota = Path("/sys/fs/cgroup/cpu.max")
    return {
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroup_cpu_max": quota.read_text().strip() if quota.is_file() else None,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "native": native_capabilities(),
        "threads": threads,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OMP_DYNAMIC": os.environ.get("OMP_DYNAMIC"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def initialize(natoms: int, beads: int, threads: int, integrator: str) -> tuple[Dynamics, State]:
    rng = np.random.default_rng(1907)
    q = rng.normal(0.0, 0.2, (beads, natoms, 3))
    p = rng.normal(0.0, 0.1, q.shape)
    dynamics = Dynamics(
        HarmonicPotential(k=0.002),
        np.full(natoms, 1836.0),
        beta=1000.0,
        nbeads=beads,
        dt=0.1,
        threads=threads,
        native="required",
        loop_backend="auto",
        integrator=integrator,
        seed=7,
    )
    state = State(q, p)
    dynamics.energy(state)
    dynamics.propagate(state, 10, sample_interval=10, record_centroids=False)
    return dynamics, state


def timed(dynamics: Dynamics, state: State, steps: int) -> tuple[float, float, str]:
    enabled = gc.isenabled()
    gc.disable()
    try:
        cpu_start = time.process_time_ns()
        wall_start = time.perf_counter_ns()
        result = dynamics.propagate(
            state, steps, sample_interval=steps, record_centroids=False
        )
        wall = (time.perf_counter_ns() - wall_start) * 1e-9
        cpu = (time.process_time_ns() - cpu_start) * 1e-9
    finally:
        if enabled:
            gc.enable()
    digest = hashlib.sha256(state.q.tobytes() + state.p.tobytes()).hexdigest()
    return wall, cpu, result.backend + ":" + digest


def benchmark(
    natoms: int,
    beads: int,
    threads: int,
    integrator: str,
    repeats: int,
    target_seconds: float,
    fixed_steps: int | None,
) -> dict:
    dynamics, state = initialize(natoms, beads, threads, integrator)
    pilot, _, _ = timed(dynamics, state, 10)
    steps = fixed_steps or max(5, min(20000, int(target_seconds / (pilot / 10))))
    wall_samples: list[float] = []
    cpu_samples: list[float] = []
    hashes: list[str] = []
    selected = None
    for _ in range(repeats):
        dynamics, state = initialize(natoms, beads, threads, integrator)
        wall, cpu, identity = timed(dynamics, state, steps)
        selected, digest = identity.split(":", 1)
        wall_samples.append(wall * 1e6 / steps)
        cpu_samples.append(cpu * 1e6 / steps)
        hashes.append(digest)
    return {
        "natoms": natoms,
        "beads": beads,
        "threads": threads,
        "integrator": integrator,
        "selected_backend": selected,
        "steps_per_repeat": steps,
        "repeats": repeats,
        "wall_microseconds_per_step": wall_samples,
        "median_wall_microseconds_per_step": statistics.median(wall_samples),
        "process_cpu_microseconds_per_step": cpu_samples,
        "median_process_cpu_microseconds_per_step": statistics.median(cpu_samples),
        "state_sha256": hashes,
        "repeat_states_bitwise_identical": len(set(hashes)) == 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--natoms", type=int, nargs="+", default=[64, 1000, 8000])
    parser.add_argument("--beads", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument(
        "--integrators", nargs="+", choices=("cayley-bcb", "normal-mode-bab"),
        default=["cayley-bcb", "normal-mode-bab"],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--target-seconds", type=float, default=0.2)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("threads and repeats must be positive")
    if any(value < 1 for value in (*args.natoms, *args.beads)):
        parser.error("atom and bead counts must be positive")
    if not np.isfinite(args.target_seconds) or args.target_seconds <= 0:
        parser.error("target-seconds must be positive and finite")
    if args.steps is not None and args.steps < 1:
        parser.error("steps must be positive")
    if os.environ.get("OMP_NUM_THREADS") != str(args.threads):
        parser.error("set OMP_NUM_THREADS equal to --threads before importing PyRPMD")
    if os.environ.get("OMP_DYNAMIC", "FALSE").upper() not in ("FALSE", "0"):
        parser.error("set OMP_DYNAMIC=FALSE for reproducible thread counts")
    results = {
        "hardware": hardware(args.threads),
        "method": {
            "potential": "HarmonicPotential(k=0.002)",
            "beta": 1000.0,
            "dt": 0.1,
            "mass": 1836.0,
            "warmup_steps": 10,
            "construction_included": False,
            "observation": "one final sample; energy checked every step",
            "timing": "perf_counter_ns and process_time_ns",
            "physical_converged": False,
        },
        "results": [],
    }
    for integrator in args.integrators:
        for natoms in args.natoms:
            for beads in args.beads:
                row = benchmark(
                    natoms, beads, args.threads, integrator, args.repeats,
                    args.target_seconds, args.steps,
                )
                results["results"].append(row)
                print(json.dumps(row), flush=True)
                args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
