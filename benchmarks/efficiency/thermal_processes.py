#!/usr/bin/env python3
"""Measure concurrent Dynamics workers with explicit threads and no CPU binding.

Run the same script with PYTHONPATH pointing to each independently built
checkout. Leave OMP_NUM_THREADS unset to reproduce the PR #115 review case.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import statistics
import time

from thermal_throughput import environment, make_case


def initialize(barrier):
    global start_barrier
    start_barrier = barrier


def measure(options):
    tasks = Path("/proc/self/task")
    before = len(list(tasks.iterdir())) if tasks.is_dir() else None
    run, owner = make_case(options["atoms"], options["beads"], "pile-l",
                           options["rng"], "dynamics", options["threads"])
    run(2)
    times = []
    for _ in range(options["repeats"]):
        start_barrier.wait(timeout=120)
        start = time.perf_counter()
        run(options["block_steps"])
        times.append(time.perf_counter()-start)
    return {"seconds": times, "steps_per_second": options["block_steps"]/statistics.median(times),
            "process_threads_before": before,
            "process_threads_after": len(list(tasks.iterdir())) if tasks.is_dir() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--atoms", type=int, default=512)
    parser.add_argument("--beads", type=int, default=32)
    parser.add_argument("--rng", choices=("mt19937_64", "philox-v1"), default="mt19937_64")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--block-steps", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.workers, args.threads, args.atoms, args.beads, args.repeats, args.block_steps) < 1:
        parser.error("workers/threads/atoms/beads/repeats/block-steps must be positive")
    context = multiprocessing.get_context("spawn")
    options = {**vars(args), "output": str(args.output)}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                             initializer=initialize, initargs=(context.Barrier(args.workers),)) as pool:
        rows = list(pool.map(measure, [options]*args.workers))
    result = {"schema": "thermal-processes-v1", "complete": True, 
              "environment": environment(), "options": options, "workers": rows,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
