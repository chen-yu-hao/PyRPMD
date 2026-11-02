"""Reproducible IESH determinant-ratio microbenchmark, not full trajectory speed.

Compare a single matrix solve with an independent determinant evaluation for
every Pauli-allowed one-electron excitation.  Optimized timings include input,
orthonormality and conditioning checks.  The explicit implementation caches the
source determinant and reuses each candidate matrix across target orbitals.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
        examples/nonadiabatic/iesh_performance.py --nstates 64 --nelectrons 32
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from pyrpmd.iesh import slater_determinant_ratios


def explicit_determinant_ratios(coefficients, occupied):
    """Independent reference using the defining Slater determinant formula."""
    overlap = coefficients[occupied].copy()
    source = np.linalg.det(overlap)
    if source == 0.:
        raise ValueError("explicit source determinant underflowed or is singular")
    result = np.zeros((len(coefficients), len(occupied)), dtype=complex)
    result[occupied] = np.eye(len(occupied))
    unoccupied = np.setdiff1d(np.arange(len(coefficients)), occupied)
    for electron in range(len(occupied)):
        candidate = overlap.copy()
        for target in unoccupied:
            candidate[electron] = coefficients[target]
            result[target, electron] = np.linalg.det(candidate)/source
    return result


def hardware_info():
    model = platform.processor() or "unavailable"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.partition(":")[2].strip()
                break
    except OSError:
        pass
    return {"cpu_model": model, "platform": platform.platform(),
            "logical_cpus_visible": os.cpu_count(),
            "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "python": platform.python_version(), "numpy": np.__version__,
            "thread_environment": {key: os.environ.get(key)
                                   for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def benchmark(*, nstates=64, nelectrons=32, repeats=15, warmups=3, seed=701):
    for name, value, minimum in (("nstates", nstates, 2), ("nelectrons", nelectrons, 1),
                                 ("repeats", repeats, 1), ("warmups", warmups, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if nelectrons >= nstates:
        raise ValueError("benchmark requires at least one empty orbital")
    rng = np.random.default_rng(seed)
    coefficients = np.linalg.qr(rng.normal(size=(nstates, nelectrons))
                                + 1j*rng.normal(size=(nstates, nelectrons)))[0]
    occupied = np.arange(nelectrons)
    optimized = slater_determinant_ratios(coefficients, occupied)
    reference = explicit_determinant_ratios(coefficients, occupied)
    error = float(np.max(np.abs(optimized-reference)))
    np.testing.assert_allclose(optimized, reference, rtol=5e-12, atol=5e-12)
    timings = {}
    for label, function in (("single_solve", slater_determinant_ratios),
                             ("separate_determinants", explicit_determinant_ratios)):
        for _ in range(warmups):
            function(coefficients, occupied)
        elapsed = []
        for _ in range(repeats):
            started = time.perf_counter()
            function(coefficients, occupied)
            elapsed.append(time.perf_counter()-started)
        timings[label] = {"median_seconds": float(np.median(elapsed)),
                          "min_seconds": float(np.min(elapsed)),
                          "max_seconds": float(np.max(elapsed))}
    return {"benchmark": "all_single_excitation_slater_determinant_ratios",
            "scope": "one electronic kernel; excludes force/eigenvalue evaluation and full trajectories",
            "nstates": nstates, "nelectrons": nelectrons,
            "allowed_single_excitations": nelectrons*(nstates-nelectrons),
            "seed": seed, "warmups": warmups, "repeats": repeats,
            "max_absolute_ratio_error": error, "timings": timings,
            "median_speedup": timings["separate_determinants"]["median_seconds"]
                              / timings["single_solve"]["median_seconds"],
            "runtime": hardware_info()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nstates", type=int, default=64)
    parser.add_argument("--nelectrons", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=701)
    args = parser.parse_args()
    print(json.dumps(benchmark(**vars(args)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
