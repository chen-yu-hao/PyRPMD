"""Throughput of the three execution paths for deterministic RPMD propagation.

Harmonic built-in potential, cayley-bcb, dt = 0.1 au.  Paths:
  fused   : loop_backend="auto", native=True  (whole block in C++)
  substep : loop_backend="python", native=True (Python loop, compiled substeps)
  numpy   : native=False                       (NumPy reference implementation)
Reported: median wall time per step over repeats (warm), one integrator thread.
"""
import json
import os
import statistics
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from pyrpmd import Dynamics, State
from pyrpmd.potentials import HarmonicPotential

PATHS = {"fused": dict(native="required", loop_backend="auto"),
         "substep": dict(native="required", loop_backend="python"),
         "numpy": dict(native=False, loop_backend="python")}


def run(path, beads, natoms, steps, repeats=5, threads=1):
    rng = np.random.default_rng(1907)
    q = rng.normal(0, 0.2, (beads, natoms, 3))
    p = rng.normal(0, 0.1, q.shape)
    with HarmonicPotential(k=0.002) as pes:
        dyn = Dynamics(pes, np.full(natoms, 1836.0), beta=1000.0, nbeads=beads, dt=0.1,
                       threads=threads, integrator="cayley-bcb", seed=7, **PATHS[path])
        state = State(q.copy(), p.copy())
        dyn.propagate(state, min(steps, 16), sample_interval=min(steps, 16), record_centroids=False)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            dyn.propagate(state, steps, sample_interval=steps, record_centroids=False)
            times.append((time.perf_counter_ns() - t0) * 1e-9 / steps)
        stats = dict(dyn.loop_stats)
        final = (state.q.copy(), state.p.copy())
    return {"path": path, "beads": beads, "atoms": natoms, "steps": steps, "threads": threads,
            "seconds_per_step_median": statistics.median(times),
            "seconds_per_step_min": min(times), "seconds_per_step_max": max(times),
            "loop_stats": stats}, final


rows = []
# (beads, atoms): gas-phase triatomic sizes and a condensed-phase size
for beads, natoms in ((16, 3), (32, 3), (64, 3), (128, 3), (256, 3), (32, 64), (32, 512)):
    finals = {}
    for path in PATHS:
        steps = {"fused": 20000, "substep": 2000, "numpy": 1000}[path]
        if natoms >= 64:
            steps //= 20 if natoms >= 512 else 5
        row, finals[path] = run(path, beads, natoms, steps)
        rows.append(row)
        print(row, file=sys.stderr)
# Equal-step agreement of the three paths from the same initial state.
checks = []
for beads, natoms in ((32, 3), (32, 64)):
    out = {}
    for path in PATHS:
        rng = np.random.default_rng(1907)
        q = rng.normal(0, 0.2, (beads, natoms, 3))
        p = rng.normal(0, 0.1, q.shape)
        with HarmonicPotential(k=0.002) as pes:
            dyn = Dynamics(pes, np.full(natoms, 1836.0), beta=1000.0, nbeads=beads, dt=0.1,
                           threads=1, integrator="cayley-bcb", seed=7, **PATHS[path])
            state = State(q, p)
            dyn.propagate(state, 1000, sample_interval=1000, record_centroids=False)
            out[path] = (state.q.copy(), state.p.copy())
    checks.append({"beads": beads, "atoms": natoms, "steps": 1000,
                   "max_abs_q_fused_minus_numpy": float(np.abs(out["fused"][0] - out["numpy"][0]).max()),
                   "max_abs_p_fused_minus_numpy": float(np.abs(out["fused"][1] - out["numpy"][1]).max()),
                   "max_abs_q_substep_minus_numpy": float(np.abs(out["substep"][0] - out["numpy"][0]).max())})
    print(checks[-1], file=sys.stderr)
cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")), "")
json.dump({"cpu": cpu, "rows": rows, "agreement": checks}, sys.stdout, indent=1)
