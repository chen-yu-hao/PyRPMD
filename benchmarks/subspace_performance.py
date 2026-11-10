"""Reproducible subspace performance pilots, not physically converged rates.

Run from a checkout with PYTHONPATH=src. Use --legacy with the baseline checkout
on PYTHONPATH to measure the original implementation with identical science.
Every repeat builds a fresh static-grid model; report timings, results and host
metadata together. Compilation and interpreter startup are excluded.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from pyrpmd.analysis.subspace_model import SubspaceModel
from pyrpmd.analysis.subspace_correction import subspace_correction
from pyrpmd.analysis.subspace_rpmd import ring_polymer_rate


def model(dimension):
    x = np.linspace(-8., 8., 321 if dimension == 1 else 81)
    axes = (x,) if dimension == 1 else (x, np.linspace(-1., 1., 9))
    grid = np.meshgrid(*axes, indexing="ij")
    potential = (0.425 / 27.211386) / np.cosh(grid[0] / .66)**2
    if dimension == 2:
        potential += .5 * 1060. * .01**2 * grid[1]**2
    return SubspaceModel(axes, potential, np.full(dimension, 1060.), 0., "asymptotic", -6., {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dimension", type=int, choices=(1, 2), default=1)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--rpmd-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error("repeats and threads must be positive")
    options = dict(windows=16, walkers=8, burn_in=20, sweeps=60, trajectories=128,
                   md_dt=2., md_time=1500., bootstrap=80, seed=19)
    beta, beads = 1. / (3.166811563e-6 * 300.), 16
    cases = [("original", None, None)] if args.legacy else [
        ("numpy_lapack", "numpy", "lapack"), ("native_1_blocked", "native", "blocked"),
        (f"native_{args.threads}_blocked", "native", "blocked")]
    records = []
    with threadpool_limits(limits=1, user_api="blas"):
        blas = threadpool_info()
        for label, backend, solver in cases:
            times, results = [], []
            for repeat in range(args.repeats):
                m = model(args.dimension)
                opts = dict(options)
                if backend:
                    opts.update(backend=backend, threads=args.threads if label.startswith(f"native_{args.threads}_") else 1)
                quantum = {} if args.legacy else {"sylvester_solver": solver}
                started = perf_counter()
                if args.rpmd_only:
                    result = ring_polymer_rate(m, beta, beads, **opts)
                    result_values = {"rpmd_rate": result.rate, "stderr": result.rate_standard_error,
                                     "kappa": result.transmission}
                else:
                    result = subspace_correction(m, beta, beads, quantum_options=quantum, rpmd_options=opts)
                    result_values = {"factor": result.factor, "quantum_rate": result.quantum_rate,
                                     "rpmd_rate": result.rpmd_rate, "log_standard_error": result.log_standard_error}
                times.append(perf_counter() - started); results.append(result_values)
            records.append({"case": label, "seconds": times, "median_seconds": float(np.median(times)), "results": results})
            print(json.dumps(records[-1]), flush=True)
    import pyrpmd.analysis.subspace_rpmd as implementation
    report = {"scope": "fixed-workload numerical performance pilot; not physical convergence",
              "physical_converged": False, "dimension": args.dimension, "beads": beads,
              "beta": beta, "rpmd_options": options, "rpmd_only": args.rpmd_only,
              "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
              "platform": platform.platform(), "cpu_count": os.cpu_count(),
              "affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "blas": blas, "implementation_sha256": sha256(Path(implementation.__file__).read_bytes()).hexdigest(),
              "records": records}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
