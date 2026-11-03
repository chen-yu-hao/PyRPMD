"""Compare the old splitting phase projection with the Householder update.

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/instanton_phase_projection_benchmark.py \
    --sizes 256 512 1024 2048 --repeats 5 --output phase-projection.json

Both timings include the same dense symmetric eigensolve. The input is the
mass-weighted fluctuation operator of an analytic quartic kink; constructing
it and all warmups are excluded. This is a numerical-kernel benchmark, not a
measurement of a full molecular splitting calculation or physical convergence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import time
import tracemalloc

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from pyrpmd.tunnelling import _phase_projected_spectrum


def _reference_spectrum(operator, tangent):
    complement = np.linalg.qr(tangent[:, None], mode="complete")[0][:, 1:]
    return np.linalg.eigvalsh(complement.T @ operator @ complement)


def _kink_operator(size):
    mass, height = 1822., 0.01
    omega = math.sqrt(8 * height / mass)
    duration = 28 / omega
    dtau = duration / (size + 1)
    q = np.tanh(omega * np.linspace(-duration / 2, duration / 2, size + 2) / 2)
    q[0], q[-1] = -1., 1.
    diagonal = 2 / dtau**2 + 4 * height / mass * (3 * q[1:-1]**2 - 1)
    operator = np.diag(diagonal)
    adjacent = np.arange(size - 1)
    operator[adjacent, adjacent + 1] = operator[adjacent + 1, adjacent] = -1 / dtau**2
    tangent = (q[2:] - q[:-2]) * math.sqrt(mass) / (2 * dtau)
    tangent /= np.linalg.norm(tangent)
    return operator, tangent


def _cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unavailable"


def _peak_traced_allocation(function, operator, tangent):
    # NumPy's tracked allocations are visible here, but this is not process
    # RSS: native BLAS/LAPACK workspaces and allocator caches may be untracked.
    tracemalloc.start()
    try:
        function(operator, tangent)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def benchmark(sizes, repeats):
    report = {"schema": "pyrpmd.splitting-phase-projection-benchmark.v1",
              "hardware": {"cpu": _cpu_model(), "platform": platform.platform()},
              "software": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__},
              "threads": 1, "repeats": repeats,
              "scope": "phase removal plus complete dense symmetric eigensolve",
              "input": "analytic quartic kink; mass=1822, height=0.01, duration=28/omega; atomic units",
              "physical_converged": False, "results": []}
    methods = {"qr_reference": _reference_spectrum, "householder": _phase_projected_spectrum}
    with threadpool_limits(limits=1):
        report["threadpools"] = threadpool_info()
        for size in sizes:
            operator, tangent = _kink_operator(size)
            spectra = {name: function(operator, tangent) for name, function in methods.items()}
            np.testing.assert_allclose(spectra["householder"], spectra["qr_reference"],
                                       rtol=1e-10, atol=1e-12)
            elapsed = {name: {"wall_s": [], "process_cpu_s": []} for name in methods}
            for repeat in range(repeats):
                order = tuple(methods) if repeat % 2 == 0 else tuple(reversed(methods))
                for name in order:
                    wall, cpu = time.perf_counter(), time.process_time()
                    methods[name](operator, tangent)
                    elapsed[name]["process_cpu_s"].append(time.process_time() - cpu)
                    elapsed[name]["wall_s"].append(time.perf_counter() - wall)
            row = {"interior_coordinates": size,
                   "max_absolute_spectral_difference": float(np.max(abs(
                       spectra["householder"] - spectra["qr_reference"]))),
                   "timings": elapsed}
            for name, function in methods.items():
                row[name] = {"median_wall_s": statistics.median(elapsed[name]["wall_s"]),
                             "median_process_cpu_s": statistics.median(elapsed[name]["process_cpu_s"]),
                             "peak_traced_allocation_bytes": _peak_traced_allocation(function, operator, tangent)}
            row["wall_speedup"] = row["qr_reference"]["median_wall_s"] / row["householder"]["median_wall_s"]
            row["process_cpu_speedup"] = (row["qr_reference"]["median_process_cpu_s"]
                                           / row["householder"]["median_process_cpu_s"])
            report["results"].append(row)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 2 or size > 4096 for size in args.sizes):
        parser.error("require repeats >= 1 and every size in [2,4096]")
    text = json.dumps(benchmark(args.sizes, args.repeats), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
