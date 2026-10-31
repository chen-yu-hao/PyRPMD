"""Reproducible CPU lookup/replay timing and interpolation-error experiment.

Run from a checkout:
  PYTHONPATH=src python benchmarks/materials/grid_bias.py --output result.json

The synthetic standard-MetaD histories share the same deterministic prefix.
They are mathematical Gaussian sums, not sampled molecular trajectories;
their unused bias_before audit field is zero. Setup never calls deposit() on
the direct backend, which would hide an O(number of hills squared) experiment
inside the setup. Grid construction measures replay/deposition at every node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import statistics
import time

import numpy as np

from pyrpmd.bias import GridMetadynamicsBias, MetadynamicsBias
from pyrpmd.cv import CVStack, Cartesian


def run(*, counts=(100, 1000, 10000), queries=100, repeats=5):
    root = Path(__file__).resolve().parents[2]
    result = {
        "experiment": "synthetic standard-MetaD CPU scalar energy-and-gradient lookup",
        "numpy": np.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "queries_per_repeat": queries,
        "repeats": repeats, "seed": 17031,
        "timing_statistic": "median of repeats; perf_counter wall time and process_time CPU time",
        "bias_source_sha256": hashlib.sha256((root / "src/pyrpmd/bias.py").read_bytes()).hexdigest(),
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "CPU bias only; excludes PES/integrator, IO, checkpoint and molecular convergence",
        "results": [],
    }
    for dimension in (1, 2):
        rng = np.random.default_rng(17031 + dimension)
        cv = CVStack([Cartesian(0, axis) for axis in range(dimension)])
        history = [{"center": center.tolist(), "height": 0.01,
                    "step": index + 1, "time": (index + 1) * 0.1}
                   for index, center in enumerate(rng.uniform(-1.9, 1.9, (max(counts), dimension)))]
        points = rng.uniform(-1.9, 1.9, (queries, dimension))
        error_points = rng.uniform(-2, 2, (256, dimension))
        for count in counts:
            biases = {}
            record = {"dimension": dimension, "hills": count,
                      "sigma": [0.4] * dimension, "grid_bins": [80] * dimension,
                      "grid_ranges": [[-2, 2]] * dimension}
            for label, factory in (("direct", MetadynamicsBias), ("grid", GridMetadynamicsBias)):
                extra = {} if label == "direct" else {"grid_ranges": record["grid_ranges"], "grid_bins": 80}
                start = time.perf_counter()
                cpu_start = time.process_time()
                bias = factory(cv, record["sigma"], 0.01, 2.0, mode="standard", hills=history[:count], **extra)
                record[label + "_build_cpu_seconds"] = time.process_time() - cpu_start
                record[label + "_build_seconds"] = time.perf_counter() - start
                biases[label] = bias
                for point in points[:10]:
                    bias.physical_energy_gradient(point)
                timings, cpu_timings = [], []
                for _ in range(repeats):
                    start = time.perf_counter()
                    cpu_start = time.process_time()
                    for point in points:
                        bias.physical_energy_gradient(point)
                    cpu_timings.append((time.process_time() - cpu_start) / len(points))
                    timings.append((time.perf_counter() - start) / len(points))
                record[label + "_lookup_seconds"] = statistics.median(timings)
                record[label + "_lookup_repeats_seconds"] = timings
                record[label + "_lookup_cpu_seconds"] = statistics.median(cpu_timings)
                record[label + "_lookup_repeats_cpu_seconds"] = cpu_timings
            error = np.zeros(2)
            scale = np.zeros(2)
            for start in range(0, len(error_points), 32):
                points_chunk = error_points[start:start + 32]
                exact = biases["direct"].physical_energy_gradient(points_chunk)
                grid = biases["grid"].physical_energy_gradient(points_chunk)
                for index, (a, b) in enumerate(zip(exact, grid)):
                    error[index] = max(error[index], float(np.max(np.abs(a - b))))
                    scale[index] = max(scale[index], float(np.max(np.abs(a))))
            record["energy_max_abs_error_hartree"] = float(error[0])
            record["gradient_max_abs_error_hartree_per_bohr"] = float(error[1])
            record["energy_max_abs_error_over_max_abs_reference"] = float(error[0] / scale[0])
            record["gradient_max_abs_error_over_max_abs_reference"] = float(error[1] / scale[1])
            record["lookup_speedup"] = record["direct_lookup_seconds"] / record["grid_lookup_seconds"]
            record["lookup_cpu_speedup"] = record["direct_lookup_cpu_seconds"] / record["grid_lookup_cpu_seconds"]
            record["grid_replay_seconds_per_hill"] = record["grid_build_seconds"] / count
            result["results"].append(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.queries < 1 or args.repeats < 1:
        parser.error("queries and repeats must be positive")
    result = run(queries=args.queries, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for row in result["results"]:
        print(f"D={row['dimension']} H={row['hills']:5d}: "
              f"direct={row['direct_lookup_seconds'] * 1e6:.1f} us "
              f"grid={row['grid_lookup_seconds'] * 1e6:.1f} us "
              f"speedup={row['lookup_speedup']:.2f} "
              f"cpu_speedup={row['lookup_cpu_speedup']:.2f} "
              f"max_dV={row['energy_max_abs_error_hartree']:.3g}")


if __name__ == "__main__":
    main()
