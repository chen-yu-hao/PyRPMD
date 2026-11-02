"""Reproducible single-process exchange timings; no convergence claims.

Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 before Python starts.
The three variants share the same positions and are interleaved to reduce
timing drift. Both elapsed and process CPU times are reported in milliseconds.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np

from pyrpmd.pimd_methods import bosonic_exchange


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=positive_integer, nargs="+", default=[8, 32, 128])
    parser.add_argument("--beads", type=positive_integer, default=32)
    parser.add_argument("--rounds", type=positive_integer, default=7)
    parser.add_argument("--calls", type=positive_integer, default=5)
    args = parser.parse_args()
    records = []
    for particles in args.particles:
        q = np.random.default_rng(2080 + particles).normal(scale=.3, size=(args.beads, particles, 3))
        variants = {
            "bosonic": {},
            "fermionic_diagnostics": {"statistics": "fermionic"},
            "minimum_image_bosonic": {"cell": np.eye(3) * 8, "periodic_mode": "minimum_image"},
        }
        timings = {name: [] for name in variants}
        for options in variants.values():
            bosonic_exchange(q, 1., 4., **options)
        names = list(variants)
        for repeat in range(args.rounds):
            offset = repeat % len(names)
            for name in names[offset:] + names[:offset]:
                wall, cpu = time.perf_counter_ns(), time.process_time_ns()
                for _ in range(args.calls):
                    bosonic_exchange(q, 1., 4., **variants[name])
                timings[name].append(((time.perf_counter_ns() - wall) / args.calls / 1e6,
                                      (time.process_time_ns() - cpu) / args.calls / 1e6))
        records.append({"particles": particles, "beads": args.beads,
                        "median_ms_per_call": {
                            name: dict(zip(("wall", "process_cpu"), np.median(values, axis=0).tolist()))
                            for name, values in timings.items()}})
    print(json.dumps({"platform": platform.platform(), "python": platform.python_version(),
                      "numpy": np.__version__, "visible_cpus": os.cpu_count(),
                      "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                      "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                      "rounds": args.rounds, "calls_per_round": args.calls, "measurements": records},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
