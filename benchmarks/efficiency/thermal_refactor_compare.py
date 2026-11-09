#!/usr/bin/env python3
"""Paired whole-API timings of two independently built source checkouts.

No compilation, CPU affinity changes or background jobs are performed here.
Run this after building both checkouts with identical flags. Runs alternate
baseline/candidate order and preserve all timings rather than selecting minima.

python benchmarks/efficiency/thermal_refactor_compare.py \
    --baseline /path/to/pr115 --candidate /path/to/refactor --output compare.json
"""
from __future__ import annotations
import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import subprocess
import sys
import time


def cases():
    result = []
    for atoms, beads in ((3, 1), (3, 32), (64, 32), (512, 32), (4096, 32)):
        for mode in ("none", "pile-l"):
            result.append(("dynamics", atoms, beads, mode, "mt19937_64", 1, True))
    for api in ("dynamics", "native"):
        for threads in (1, 4):
            result.append((api, 4096, 32, "pile-l", "philox-v1", threads, True))
    result.append(("dynamics", 4096, 32, "pile-l", "mt19937_64", 1, False))
    return result


def make_case(case):
    import numpy as np
    from pyrpmd.dynamics import Dynamics, State
    from pyrpmd.native_ensemble import NativeEnsemble
    from pyrpmd.potentials import HarmonicPotential
    api, atoms, beads, mode, rng, threads, centroids = case
    q = np.full((beads, atoms, 3), .1)
    masses = np.full(atoms, 2.)
    bath = "none" if mode == "none" else {"mode": mode, "rng": rng, "gamma_centroid": 1.}
    if api == "native":
        owner = NativeEnsemble(q, masses, 2., .01, thermostat=bath, threads=threads,
                               seed=1749, potential={"kind": "harmonic", "k": .5})
        return lambda n: owner.run(n, record_centroids=centroids)
    owner = Dynamics(HarmonicPotential(k=.5), masses, 2., beads, .01, thermostat=bath,
                     threads=threads, seed=1749, potential_identity="thermal-throughput-harmonic-v1")
    state = State(q, np.zeros_like(q))
    return lambda n: owner.propagate(state, n, record_centroids=centroids)


def worker(args):
    import numpy as np
    import pyrpmd
    from pyrpmd import _thermal
    source = Path(pyrpmd.__file__).resolve().parents[2]
    if source != args.worker.resolve():
        raise RuntimeError(f"wrong imported source: {source}; expected {args.worker.resolve()}")
    if not Path(_thermal.__file__).resolve().is_relative_to(source):
        raise RuntimeError("native extension is not from the selected checkout")
    measurements = []
    matrix = cases()
    matrix = matrix[args.rotation:] + matrix[:args.rotation]
    for case in matrix:
        run = make_case(case)
        run(8)  # construction, preparation and warm-up are outside timings
        # Keep the small-system runs long enough to measure call overhead.
        steps = max(args.block_steps, 256) if case[1] == 3 else args.block_steps
        # Fixed (not timing-adaptive) repetitions, identical in both trees.
        # Sub-millisecond small-system timings otherwise mostly measure noise.
        calls = (256 if case[2] == 1 else 32) if case[1] == 3 else (16 if case[1] == 64 else 4 if case[1] == 512 else 1)
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter_ns()
            for _ in range(calls):
                run(steps)
            samples.append((time.perf_counter_ns() - start) * 1e-9)
        row = {"case": list(case), "steps": steps, "calls_per_repeat": calls,
               "steps_per_repeat": steps*calls, "seconds": samples,
               "steps_per_second": steps*calls / statistics.median(samples)}
        if case == ("dynamics", 4096, 32, "pile-l", "mt19937_64", 1, True):
            profile = cProfile.Profile()
            for _ in range(5):
                profile.runcall(run, 2)
            stats = pstats.Stats(profile)
            native = sum(v[2] for k, v in stats.stats.items()
                         if k[0] == "~" and "pyrpmd._thermal.run" in k[2])
            row["two_step_profile"] = {"total_seconds": stats.total_tt,
                                       "native_seconds": native,
                                       "outside_over_native": (stats.total_tt-native)/native if native else None}
        measurements.append(row)
    return {"source": str(source), "python": sys.version, "numpy": np.__version__,
            "thermal_binary": str(_thermal.__file__),
            "thermal_sha256": hashlib.sha256(Path(_thermal.__file__).read_bytes()).hexdigest(),
            "measurements": measurements}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--block-steps", type=int, default=16)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--rotation", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.rounds, args.repeats, args.block_steps) < 1:
        parser.error("rounds, repeats and block-steps must be positive")
    if args.worker:
        print(json.dumps(worker(args)))
        return
    if any(v is None for v in (args.baseline, args.candidate, args.output)):
        parser.error("baseline, candidate and output are required")
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "OMP_DYNAMIC": "FALSE"}
    report = {"schema": "pyrpmd.thermal-refactor-comparison.v1", "complete": False,
              "physical_converged": False, "protocol": {"rounds": args.rounds, "repeats": args.repeats,
              "block_steps": args.block_steps, "warmup_steps": 8, "ordering": "alternating paired subprocesses", "small_case_calls": "fixed 256/32 for N=3 P=1/32; 16 for N=64; 4 for N=512"},
              "platform": platform.platform(),
              "processor": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                                 if line.startswith("model name")), platform.processor()) if Path("/proc/cpuinfo").exists() else platform.processor(),
              "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip() if Path("/sys/fs/cgroup/cpu.max").exists() else None,
              "thread_environment": {k: env.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_THREAD_LIMIT", "OMP_PLACES", "OMP_PROC_BIND")},
              "runs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for rnd in range(args.rounds):
        for label in (("baseline", "candidate") if rnd % 2 == 0 else ("candidate", "baseline")):
            source = getattr(args, label).resolve()
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(source),
                       "--block-steps", str(args.block_steps), "--repeats", str(args.repeats),
                       "--rotation", str(rnd % len(cases()))]
            done = subprocess.run(command, check=True, capture_output=True, text=True,
                                  env={**env, "PYTHONPATH": str(source / "src")})
            row = {"round": rnd, "label": label, **json.loads(done.stdout)}
            report["runs"].append(row)
            args.output.write_text(json.dumps(report, indent=2)+"\n")
            print(f"round {rnd+1}: {label} complete", flush=True)
    summary = []
    for case in cases():
        rows = {label: [next(v for v in run["measurements"] if v["case"] == list(case))
                       for run in report["runs"] if run["label"] == label]
                for label in ("baseline", "candidate")}
        rates = {label: [v["steps_per_second"] for v in rows[label]] for label in rows}
        ratios = [c/b for b, c in zip(rates["baseline"], rates["candidate"])]
        item = {"case": list(case), "baseline_sps": statistics.median(rates["baseline"]),
                "candidate_sps": statistics.median(rates["candidate"]),
                "paired_speedups": ratios, "median_paired_speedup": statistics.median(ratios)}
        if "two_step_profile" in rows["baseline"][0]:
            item["two_step_outside_over_native"] = {label: [v["two_step_profile"]["outside_over_native"] for v in rows[label]] for label in rows}
        summary.append(item)
    report.update(complete=True, summary=summary)
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
