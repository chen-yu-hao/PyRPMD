"""Reproducible full-step timings for the centroid RP-IESH research variant.

The fixed cases have P=1,8,32, 12 metallic bath orbitals (13 total states), and
6 electrons. Each repeat restarts the same prepared ring and dynamics RNG.
Timings include validation, bead forces, nuclear propagation, electronic
midpoint quadrature and stochastic hopping. Initialization, energy/population
observations, checkpointing and result I/O are outside the timed region.

All detected BLAS/OpenMP thread pools and the nuclear kernel use one thread.
Run without other CPU-heavy jobs; these measurements do not establish physical
convergence or GPU/large-system performance. ``--native`` requires the compiled
nuclear kernel and never silently falls back to Python.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
        examples/nonadiabatic/rp_iesh_performance.py --steps 100 --repeats 5

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
        examples/nonadiabatic/rp_iesh_performance.py --native --steps 100 --repeats 5
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from pyrpmd.iesh import NewnsAndersonModel
from pyrpmd.nonadiabatic_sampling import sample_harmonic_ring
from pyrpmd.rp_iesh import RPIESHDynamics


SEED = 20261103
BEAD_COUNTS = (1, 8, 32)


def positive_integer(value):
    try:
        result = int(value)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
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
    return {
        "cpu_model": model, "platform": platform.platform(),
        "logical_cpus_visible": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": platform.python_version(), "numpy": np.__version__,
        "thread_environment": {key: os.environ.get(key)
                               for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "threadpools_during_measurement": threadpool_info(),
    }


def benchmark(*, steps=100, repeats=5, warmups=10, native=False):
    for name, value in (("steps", steps), ("repeats", repeats), ("warmups", warmups)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(native, bool):
        raise ValueError("native must be a bool; True requires the compiled nuclear kernel")
    native_request = "required" if native else False
    model = NewnsAndersonModel(nband=12)
    # Construct before threadpool_limits so it also sees any OpenMP runtime
    # loaded by the explicitly requested native extension.
    drivers = [RPIESHDynamics(model, [200.], 40., beads, .02, nelectrons=6,
                              seed=SEED, native=native_request, threads=1,
                              electronic_substeps=4)
               for beads in BEAD_COUNTS]
    results = []
    with threadpool_limits(limits=1):
        runtime = hardware_info()
        for driver in drivers:
            nuclear = sample_harmonic_ring(
                driver.masses, driver.beta, driver.nbeads, [.02], coordinates=[0],
                centers=[-.1], centroid_momenta=[4.], count=1, seed=SEED)[0]
            initial = driver.initialize(nuclear, occupied=np.arange(6))
            initial_rng = copy.deepcopy(driver.rng.bit_generator.state)
            warm_state = initial.copy()
            for _ in range(warmups):
                driver.step(warm_state)
            durations = []
            for _ in range(repeats):
                state = initial.copy()
                driver.rng.bit_generator.state = copy.deepcopy(initial_rng)
                started = time.perf_counter_ns()
                for _ in range(steps):
                    driver.step(state)
                durations.append((time.perf_counter_ns()-started)/1e9)
            per_step = np.asarray(durations)*1e6/steps
            results.append({
                "beads": driver.nbeads, "nstates": driver.nstates,
                "nband": model.nband, "nelectrons": driver.nelectrons,
                "nuclear_native": driver.native, "nuclear_threads": 1,
                "median_us_per_step": float(np.median(per_step)),
                "min_us_per_step": float(np.min(per_step)),
                "max_us_per_step": float(np.max(per_step)),
                "repeat_elapsed_seconds": durations,
                "final_hop_energy_change_hartree": state.hop_energy_change,
                "final_hop_energy_change_abs_hartree": state.hop_energy_change_abs,
            })
    return {
        "benchmark": "rp_iesh_complete_propagation_step",
        "method_variant": RPIESHDynamics.method_variant,
        "scope": "complete validated step and caller loop; excludes initialization, observations, checkpoint and I/O",
        "physical_converged": False, "native_requested": native_request,
        "seed": SEED, "steps_per_repeat": steps, "repeats": repeats, "warmup_steps": warmups,
        "model_identity": model.to_dict(),
        "dynamics": {"mass_electron_masses": [200.], "beta_inverse_hartree": 40.,
                     "dt_atomic_time": .02, "integrator": "cayley-bcb", "electronic_substeps": 4,
                     "gap_tolerance_hartree": 1e-10, "max_hop_probability": 1.},
        "initial": {"kind": "harmonic_ring", "frequencies_atomic": [.02],
                    "coordinates": [0], "centers_bohr": [-.1], "centroid_momenta_atomic": [4.],
                    "occupied": list(range(6))},
        "runtime": runtime, "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=positive_integer, default=100,
                        help="propagation steps per timed repeat (default: 100)")
    parser.add_argument("--repeats", type=positive_integer, default=5,
                        help="timed repeats of the same initial trajectory (default: 5)")
    parser.add_argument("--warmups", type=positive_integer, default=10,
                        help="untimed warmup steps per bead count (default: 10)")
    parser.add_argument("--native", action="store_true",
                        help="require the compiled nuclear kernel; default uses Python explicitly")
    args = parser.parse_args()
    print(json.dumps(benchmark(**vars(args)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
