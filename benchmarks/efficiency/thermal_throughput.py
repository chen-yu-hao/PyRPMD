#!/usr/bin/env python3
"""Matched thermal throughput and short-call profiles (no startup timing).

Run this same file against each installed checkout, with OMP_NUM_THREADS set
before Python starts. --baseline selects the review's small/P32 cases; the
full default matrix covers N=64,1000,4096 and P=8,32,128. JSON retains every
wall time, the imported source, compiler, affinity and observation contract.
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from pyrpmd.dynamics import Dynamics, State
from pyrpmd.native_ensemble import NativeEnsemble
from pyrpmd.potentials import HarmonicPotential
import pyrpmd


def thermostat(mode, beads, rng):
    spec = {"mode": mode}
    if rng != "mt19937_64":
        spec["rng"] = rng
    if mode in ("gle", "piglet"):
        drift = [[1., .2], [-.2, 1.]]
        spec["gle"] = {"A": drift}
        if mode == "piglet":
            spec["gle"].update(A=[drift]*beads, C=np.eye(2).tolist(),
                               beta=2., beads=beads,
                               provenance="Synthetic canonical PIGLET-shaped benchmark, not calibrated quantum data")
    return spec


def make_case(atoms, beads, mode, rng, api):
    q = np.full((beads, atoms, 3), .1)
    masses = np.full(atoms, 2.)
    bath = "none" if mode == "none" else thermostat("pile-l" if mode == "bzp" else mode, beads, rng)
    if api == "native":
        options = ({"potential": {"kind": "free"}, "cell": np.eye(3)*10,
                    "barostat": {"mode": "bzp", "pressure": (atoms+1)/2000., "tau": 10.}}
                   if mode == "bzp" else {"potential": {"kind": "harmonic", "k": .5}})
        engine = NativeEnsemble(q, masses, 2., .01, thermostat=bath, seed=1749, **options)
        return lambda steps: engine.run(steps, record_centroids=True), engine
    if mode == "bzp":
        return None, None
    dynamics = Dynamics(HarmonicPotential(k=.5), masses, 2., beads, .01,
                        thermostat=bath, potential_identity="thermal-throughput-harmonic-v1", seed=1749,
                        threads=int(os.environ.get("OMP_NUM_THREADS", "1")))
    state = State(q, np.zeros_like(q))
    return lambda steps: dynamics.propagate(state, steps), dynamics


def environment():
    source = Path(pyrpmd.__file__).resolve().parents[2]
    def command(args):
        return subprocess.check_output(args, text=True, cwd=source).strip()
    return {"hostname": platform.node(), "platform": platform.platform(),
            "python": sys.version, "numpy": np.__version__, "source": str(source),
            "revision": command(["git", "rev-parse", "HEAD"]),
            "dirty": bool(command(["git", "status", "--porcelain"])),
            "source_sha256": {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in (
                "native/thermal.hpp", "native/thermal.cpp", "native/philox.h",
                "src/pyrpmd/native_ensemble.py",
                "src/pyrpmd/propagation.py", "src/pyrpmd/block_propagation.py",
                "src/pyrpmd/dynamics.py", "src/pyrpmd/engine.py",
                "src/pyrpmd/potentials/_builtin_native.py") if (source/name).is_file()},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "compiler": command([os.environ.get("CXX", "c++"), "--version"]),
            "build_flags": {key: os.environ.get(key) for key in ("CXXFLAGS", "CFLAGS", "PYRPMD_NO_OPENMP")},
            "cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                         if line.startswith("model name")), platform.processor()) if Path("/proc/cpuinfo").exists() else platform.processor(),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OPENBLAS_NUM_THREADS")}}


def measure(atoms, beads, mode, args):
    run, owner = make_case(atoms, beads, mode, args.rng, args.api)
    row = {"atoms": atoms, "beads": beads, "thermostat": mode, "rng": None if mode == "none" else args.rng, "api": args.api}
    if run is None:
        return {**row, "status": "unsupported", "reason": "BZP native built-in model uses the NativeEnsemble API"}
    run(2)
    start = time.perf_counter(); run(args.block_steps); pilot = time.perf_counter()-start
    calls = max(1, int(np.ceil(args.seconds/max(pilot, 1e-9))))
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        for _ in range(calls):
            run(args.block_steps)
        times.append(time.perf_counter()-start)
    row.update(status="measured", steps_per_repeat=calls*args.block_steps, seconds=times,
               steps_per_second=calls*args.block_steps/statistics.median(times))
    if args.profile and args.api == "dynamics" and mode != "none":
        profile = cProfile.Profile()
        for _ in range(args.repeats):
            profile.runcall(run, 2)
        stats = pstats.Stats(profile)
        native = sum(data[2] for key, data in stats.stats.items()
                     if key[0] == "~" and "pyrpmd._thermal.run" in key[2])
        row["short_call_profile"] = {
            "steps_per_call": 2, "calls": args.repeats, "total_seconds": stats.total_tt, "native_seconds": native,
            "outside_native_seconds": stats.total_tt-native,
            "outside_over_native": (stats.total_tt-native)/native if native else None,
            "functions": [{"name": ":".join(map(str, key)), "calls": value[1],
                           "self_seconds": value[2], "cumulative_seconds": value[3]}
                          for key, value in sorted(stats.stats.items(), key=lambda item: -item[1][3])[:30]],
        }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--atoms", type=int, nargs="+")
    parser.add_argument("--beads", type=int, nargs="+")
    parser.add_argument("--thermostats", nargs="+", choices=("none", "pile-l", "pile-g", "gle", "piglet", "nhc", "bzp"))
    parser.add_argument("--rng", choices=("mt19937_64", "philox-v1"), default="mt19937_64")
    parser.add_argument("--api", choices=("dynamics", "native"), default="dynamics")
    parser.add_argument("--seconds", type=float, default=.3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--block-steps", type=int, default=64)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0 or args.repeats < 1 or not 1 <= args.block_steps <= 256:
        parser.error("seconds/repeats must be positive; block-steps must be 1..256")
    if any(value < 1 for value in (args.atoms or [])+(args.beads or [])):
        parser.error("atoms and beads must be positive")
    if args.baseline and (args.atoms or args.beads):
        parser.error("--baseline fixes the atom/bead cases; omit --atoms and --beads")
    cases = ([(3, 1), (3, 32), (64, 32), (512, 32), (4096, 32)] if args.baseline else
             [(n, p) for n in args.atoms or [64, 1000, 4096] for p in args.beads or [8, 32, 128]])
    modes = args.thermostats or (["none", "pile-l"] if args.baseline else
                                ["none", "pile-l", "pile-g", "gle", "piglet", "nhc", "bzp"])
    result = {"schema": "thermal-throughput-v1", "complete": False, "physical_converged": False, "environment": environment(),
              "options": {**vars(args), "output": str(args.output)}, "measurements": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for atoms, beads in cases:
        for mode in modes:
            row = measure(atoms, beads, mode, args)
            result["measurements"].append(row)
            args.output.write_text(json.dumps(result, indent=2)+"\n")
            print(json.dumps({key: value for key, value in row.items() if key != "short_call_profile"}), flush=True)
    result["complete"] = True
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
