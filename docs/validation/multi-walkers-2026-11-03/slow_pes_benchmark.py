"""Reproducible barrier-overhead microbenchmark for T38.

This is a scheduler benchmark, not a physical convergence claim.  Each sleep
stands in for one variable-latency PES evaluation.  Synchronous workers wait at
every deposition barrier; asynchronous workers only append their own records.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import numpy as np

from pyrpmd.walker_exchange import AsyncWalkerStore


def delays(walkers: int, steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Variable wall time resembles a slow external evaluator with occasional
    # long SCF/neighbor-list iterations.  The array is generated once so both
    # modes execute exactly the same artificial work.
    return rng.lognormal(mean=np.log(0.002), sigma=0.85, size=(walkers, steps))


def sync_worker(index, values, pace, barrier):
    for step, delay in enumerate(values, 1):
        time.sleep(float(delay))
        if step % pace == 0:
            barrier.wait()


def async_worker(index, values, pace, directory, walkers):
    with AsyncWalkerStore(
            directory, method="metadynamics", walker=index, walkers=walkers,
            identity={"benchmark": "variable-latency-sleep-v1", "walkers": walkers}) as store:
        for step, delay in enumerate(values, 1):
            time.sleep(float(delay))
            if step % pace == 0:
                store.append(
                    {"center": [float(index)], "height": 0.01,
                     "bias_before": 0.0}, step=step, time=float(step))
                store.poll()


def measure(mode, values, pace):
    walkers = len(values)
    context = mp.get_context("spawn")
    temporary = tempfile.TemporaryDirectory() if mode == "async-files" else None
    barrier = context.Barrier(walkers) if mode == "mpi-sync" else None
    target = sync_worker if mode == "mpi-sync" else async_worker
    processes = []
    start = time.perf_counter()
    for index in range(walkers):
        args = ((index, values[index], pace, barrier) if barrier is not None else
                (index, values[index], pace, temporary.name, walkers))
        process = context.Process(target=target, args=args)
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode:
            raise RuntimeError(f"worker exited with status {process.exitcode}")
    elapsed = time.perf_counter() - start
    if temporary is not None:
        temporary.cleanup()
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--walkers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--pace", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3811)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    work = delays(args.walkers, args.steps, args.seed)
    times = {mode: [] for mode in ("mpi-sync", "async-files")}
    for _ in range(args.repeats):
        for mode in times:
            times[mode].append(measure(mode, work, args.pace))
    medians = {mode: statistics.median(values) for mode, values in times.items()}
    evaluations = args.walkers * args.steps
    result = {
        "schema_version": 1,
        "benchmark": "variable-latency artificial PES scheduler microbenchmark",
        "walkers": args.walkers, "steps_per_walker": args.steps,
        "deposition_pace": args.pace, "repeats": args.repeats,
        "sleep_seconds": {"sum": float(work.sum()),
                          "minimum": float(work.min()),
                          "median": float(np.median(work)),
                          "maximum": float(work.max())},
        "seconds": times, "median_seconds": medians,
        "evaluations_per_second": {
            mode: evaluations / elapsed for mode, elapsed in medians.items()},
        "async_speedup": medians["mpi-sync"] / medians["async-files"],
        "hardware": {"platform": platform.platform(),
                     "processor": platform.processor(),
                     "logical_cpus": os.cpu_count(),
                     "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                     "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
        "note": "Sleep emulates variable PES latency; this is not a molecular-dynamics or physical-convergence benchmark.",
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
