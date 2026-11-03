"""Reproducible CPU cost/energy smoke measurement for BA-RP-MASH.

Run from an installed checkout with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1:
    python examples/nonadiabatic/extensions/rp_mash_bead_benchmark.py

This times the adaptive reference implementation on Tully I, not a converged
physical ensemble or a GPU/native benchmark. JSON is printed to stdout.
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import scipy

from pyrpmd.dynamics import State
from pyrpmd.potentials.nonadiabatic_models import TullyModelOne
from pyrpmd.rp_mash_bead import BARPMASHDynamics


class CountedPotential:
    def __init__(self):
        self.provider = TullyModelOne()
        self.calls = 0
        self.frames = 0

    def evaluate(self, q):
        self.calls += 1
        self.frames += len(q)
        return self.provider.evaluate(q)


def positive_integer(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def measure(beads, steps, repeats):
    model = CountedPotential()
    driver = BARPMASHDynamics(model, [2000.], 1000., beads, .1, seed=41,
                             rtol=1e-9, atol=1e-11, max_step=.025)
    q = np.zeros((beads, 1, 3))
    q[:, 0, 0] = -.5 + .05*np.cos(2*np.pi*np.arange(beads)/beads)
    p = np.zeros_like(q)
    p[:, 0, 0] = 20.
    initial = driver.initialize(State(q, p), 0)
    driver.step(initial.copy())  # Exclude one warmup from every timing.
    samples = []
    initial_energy = driver.energy(initial)
    for _ in range(repeats):
        state = initial.copy()
        model.calls = model.frames = 0
        hops = 0
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        for _ in range(steps):
            driver.step(state)
            hops += len(driver.last_hops)
        cpu_elapsed, wall_elapsed = time.process_time()-cpu_start, time.perf_counter()-wall_start
        calls, frames = model.calls, model.frames
        samples.append({
            "wall_seconds": wall_elapsed, "process_cpu_seconds": cpu_elapsed,
            "provider_calls": calls, "provider_frames": frames, "hop_events": hops,
            "absolute_energy_drift_hartree": abs(driver.energy(state)-initial_energy),
            "spin_norm_error": abs(state.spin@state.spin-1.),
        })
    model.provider.close()
    return {
        "beads": beads, "steps_per_repeat": steps, "repeats": repeats,
        "median_wall_ms_per_step": 1000*np.median([row["wall_seconds"] for row in samples])/steps,
        "median_cpu_ms_per_step": 1000*np.median([row["process_cpu_seconds"] for row in samples])/steps,
        "max_absolute_energy_drift_hartree": max(row["absolute_energy_drift_hartree"] for row in samples),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beads", nargs="+", type=positive_integer, default=[1, 8, 32])
    parser.add_argument("--steps", type=positive_integer, default=20)
    parser.add_argument("--repeats", type=positive_integer, default=3)
    args = parser.parse_args()
    result = {
        "method": "rp-mash-ba", "model": "tully1", "physical_converged": False,
        "platform": platform.platform(), "machine": platform.machine(),
        "logical_cpus_visible": os.cpu_count(), "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "dt_au": .1, "max_step_au": .025, "beta_hartree_inverse": 1000.,
        "masses_electron_mass": [2000.], "rtol": 1e-9, "atol": 1e-11,
        "measurements": [measure(beads, args.steps, args.repeats) for beads in args.beads],
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
