"""Harmonic built-in fused-loop timing; run from an installed PyRPMD environment.

Examples:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/benchmark_builtin_native.py \
      --label before --backends auto --output before.json
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/benchmark_builtin_native.py \
      --label after --backends auto python --output after.json

Times include Dynamics.propagate execution, per-step energy accounting, and
one final observation. They exclude PES/integrator construction and warm-up.
Wall time and process CPU time are recorded separately for each repeat.
No equilibrium, physical convergence, or scaling extrapolation is implied.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
from pathlib import Path
import statistics
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

from pyrpmd import Dynamics, State
from pyrpmd.dynamics import native_capabilities
from pyrpmd.potentials import HarmonicPotential


def hardware():
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cpu = next((line.split(":", 1)[1].strip()
                    for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    quota = Path("/sys/fs/cgroup/cpu.max")
    return {
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroup_cpu_max": quota.read_text().strip() if quota.is_file() else None,
        "python": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "numpy": np.__version__,
        "native": native_capabilities(),
        "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
        "OPENBLAS_NUM_THREADS": os.environ["OPENBLAS_NUM_THREADS"],
        "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
        "integrator_threads": 1,
    }


def initialize(natoms, backend):
    rng = np.random.default_rng(1907)
    q = rng.normal(0, 0.2, (32, natoms, 3))
    p = rng.normal(0, 0.1, q.shape)
    potential = HarmonicPotential(k=0.002)
    dynamics = Dynamics(potential, np.full(natoms, 1836.0), beta=1000.0,
                        nbeads=32, dt=0.1, threads=1, native="required",
                        loop_backend=backend, integrator="cayley-bcb", seed=7)
    state = State(q, p)
    dynamics.energy(state)
    dynamics.propagate(state, 10, sample_interval=10, record_centroids=False)
    return potential, dynamics, state


def timed(dynamics, state, steps):
    enabled = gc.isenabled()
    gc.disable()
    try:
        cpu_start = time.process_time_ns()
        start = time.perf_counter_ns()
        result = dynamics.propagate(state, steps, sample_interval=steps,
                                    record_centroids=False)
        elapsed = (time.perf_counter_ns() - start) * 1e-9
        cpu_elapsed = (time.process_time_ns() - cpu_start) * 1e-9
    finally:
        if enabled:
            gc.enable()
    return elapsed, cpu_elapsed, result


def benchmark(natoms, backend, repeats, target, fixed_steps):
    potential, dynamics, state = initialize(natoms, backend)
    pilot, _, _ = timed(dynamics, state, 20)
    steps = fixed_steps or max(20, min(30000, int(target / (pilot / 20))))
    potential.close()
    timings, cpu_timings, selected, stats = [], [], None, None
    for _ in range(repeats):
        potential, dynamics, state = initialize(natoms, backend)
        elapsed, cpu_elapsed, result = timed(dynamics, state, steps)
        assert np.isfinite(state.q).all() and np.isfinite(state.p).all()
        timings.append(elapsed * 1e6 / steps)
        cpu_timings.append(cpu_elapsed * 1e6 / steps)
        selected, stats = result.backend, dynamics.loop_stats
        potential.close()
    return {
        "natoms": natoms, "beads": 32, "requested_backend": backend,
        "selected_backend": selected, "steps_per_repeat": steps,
        "repeats": repeats, "microseconds_per_step": timings,
        "median_microseconds_per_step": statistics.median(timings),
        "minimum_microseconds_per_step": min(timings),
        "maximum_microseconds_per_step": max(timings),
        "process_cpu_microseconds_per_step": cpu_timings,
        "median_process_cpu_microseconds_per_step": statistics.median(cpu_timings),
        "minimum_process_cpu_microseconds_per_step": min(cpu_timings),
        "maximum_process_cpu_microseconds_per_step": max(cpu_timings),
        "loop_stats_last_repeat_including_warmup": stats,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--backends", nargs="+", choices=("auto", "native", "python"), default=["auto"])
    parser.add_argument("--natoms", type=int, nargs="+", default=[3, 300, 3000])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--target-seconds", type=float, default=0.25)
    parser.add_argument("--steps", type=int, help="Use identical fixed step counts instead of timing calibration")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or any(natoms < 1 for natoms in args.natoms):
        parser.error("repeats and atom counts must be positive")
    if not np.isfinite(args.target_seconds) or args.target_seconds <= 0:
        parser.error("target-seconds must be positive and finite")
    if args.steps is not None and args.steps < 1:
        parser.error("steps must be positive")
    results = {"label": args.label, "hardware": hardware(), "method": {
        "potential": "HarmonicPotential(k=0.002)", "integrator": "cayley-bcb",
        "beta": 1000.0, "dt": 0.1, "mass": 1836.0, "thermostat": "none",
        "warmup_steps": 10, "observation": "one final sample, no centroids; energy extrema every step",
        "construction_included": False, "timing_clock": "time.perf_counter_ns",
        "process_cpu_timing_clock": "time.process_time_ns",
        "random_initial_state_seed": 1907,
    }, "results": []}
    print(json.dumps({"label": args.label, "hardware": results["hardware"]}), flush=True)
    for backend in args.backends:
        for natoms in args.natoms:
            row = benchmark(natoms, backend, args.repeats, args.target_seconds, args.steps)
            results["results"].append(row)
            print(json.dumps(row), flush=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
