"""Independent canonical double-well pulls, exact quadrature and Crooks/BAR.

Run from the repository root:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/steered_double_well.py

This atomic-unit model is a numerical validation example, not a molecular rate
calculation. It estimates the free-energy difference of the *restrained*
endpoints. Inverse-CDF initialization uses converged one-dimensional quadrature
to avoid interpreting a short equilibration trajectory as a canonical sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import time

import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson

from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.io import write_report_json
from pyrpmd.potentials.builtin import DoubleWellPotential
from pyrpmd.steered import SteeredBias, SteeredMD, crooks, jarzynski


def validate(replicas=128, steps=40, seed=731, output="steered-double-well"):
    output = Path(output)
    beta, stiffness, dt = 1.3, 1.5, 0.03
    endpoints = [-0.7, 0.3]
    grid = np.linspace(-4, 4, 32001)
    potential_energy = 0.4 * (grid**2 - 1)**2
    rng = np.random.default_rng(seed)
    work, partitions = [], []
    start_clock = time.perf_counter()
    for direction in range(2):
        center, target = endpoints[direction], endpoints[1 - direction]
        density = np.exp(-beta * (potential_energy + 0.5 * stiffness * (grid - center)**2))
        partitions.append(simpson(density, x=grid))
        cdf = cumulative_trapezoid(density, grid, initial=0)
        cdf /= cdf[-1]
        direction_work = []
        for replica in range(replicas):
            q = rng.normal(size=(1, 1, 3)) / np.sqrt(beta)
            q[0, 0, 0] = np.interp(rng.random(), cdf, grid)
            p = rng.normal(size=q.shape) / np.sqrt(beta)
            dynamics = Dynamics(DoubleWellPotential(height=0.4, confinement=1.0), [1.0], beta, 1, dt,
                                seed=np.random.SeedSequence([seed, direction, replica]), native=False,
                                potential_identity="double_well:height=0.4,width=1,confinement=1")
            bias = SteeredBias(Cartesian(0, 0), np.linspace(center, target, steps + 1)[:, None], stiffness)
            simulation = SteeredMD(dynamics, State(q, p), bias, thermostat="langevin", gamma=1.0,
                                   initial_equilibrium=True)
            checkpoint = output / "checkpoints" / f"{direction}_{replica:04d}.npz"
            if checkpoint.exists():
                simulation.load(checkpoint)
            result = simulation.run(checkpoint=checkpoint)
            direction_work.append(result["total_work"][-1])
        work.append(np.asarray(direction_work))
    elapsed = time.perf_counter() - start_clock
    result = crooks(*work, beta, seed=seed)
    exact = -np.log(partitions[1] / partitions[0]) / beta
    summary = {
        "model": "atomic-unit quartic double well with moving harmonic restraint",
        "replicas_per_direction": replicas, "steps_per_pull": steps, "seed": seed,
        "beta": beta, "timestep_au": dt, "stiffness_au": stiffness,
        "exact_restrained_endpoint_delta_free_energy": exact,
        "crooks": result.to_dict(), "jarzynski_forward": jarzynski(work[0], beta, seed=seed).to_dict(),
        "absolute_error": abs(result.delta_free_energy - exact),
        "forward_work": work[0], "reverse_work": work[1],
        "timing": {"seconds_including_checkpoints": elapsed,
                   "microseconds_per_step_including_checkpoints": 1e6 * elapsed / (2 * replicas * steps),
                   "threads": 1, "native": False, "platform": platform.platform()},
        "physical_converged": False,
        "note": "Statistical validation of this toy model does not establish molecular or bead convergence.",
    }
    write_report_json(output / "steered-reference.json", summary)
    print(f"Exact restrained endpoint delta F: {exact:.8f} hartree")
    print(f"Crooks/BAR: {result.delta_free_energy:.8f} +/- {result.standard_error:.8f} hartree")
    print(f"Forward/reverse overlap: {result.diagnostics['overlap']:.4f}")
    print(f"Saved {output / 'steered-reference.json'}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, default=128)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--output", default="steered-double-well")
    args = parser.parse_args()
    if args.replicas < 2 or args.steps < 1:
        parser.error("replicas must be at least two and steps must be positive")
    validate(args.replicas, args.steps, args.seed, args.output)
