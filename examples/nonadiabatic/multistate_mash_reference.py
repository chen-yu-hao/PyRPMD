"""Compare P=1 multistate MASH to the independent 1D Fourier reference.

This is a reproducible finite-packet comparison, not a converged scattering
probability. Both methods start from the same pure diabatic electronic state
and Gaussian nuclear wavepacket (MASH samples its positive Wigner function).
All inputs use atomic units. No optional plotting dependency is required.

Run from the repository root, for example::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
        examples/nonadiabatic/multistate_mash_reference.py --trajectories 128

The two reference refinements separately measure temporal and spatial
discretization sensitivity. SEM measures independent-trajectory sampling only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from pyrpmd.analysis.quantum_reference import SplitOperator1D
from pyrpmd.dynamics import State
from pyrpmd.mash import MultistateMASHDynamics
from pyrpmd.potentials.nonadiabatic_models import LinearVibronicCoupling, TullyModelOne


def comparison_model(name):
    if name == "tully1":
        return TullyModelOne()
    if name == "three_state_lvc":
        return LinearVibronicCoupling(
            [[0., .004, .001], [.004, .002, .004], [.001, .004, 0.]],
            [np.diag([.01, 0., -.01])], harmonic=[0.], coordinates=[0], natoms=1)
    raise ValueError("model must be tully1 or three_state_lvc")


def quantum_packet(model, *, points=512, dt=1., duration=240., mass=200.,
                   center=-3., width=.6, momentum=5., half_box=20.):
    grid = np.linspace(-half_box, half_box, points, endpoint=False)
    q = np.zeros((points, 1, 3))
    q[:, 0, 0] = grid
    matrix = model.evaluate(q).matrix
    psi = np.zeros((points, model.nstates), dtype=complex)
    psi[:, 0] = np.exp(-(grid-center)**2/(4*width**2)+1j*momentum*(grid-center))
    psi /= np.sqrt((grid[1]-grid[0])*np.sum(abs(psi)**2))
    steps = round(duration/dt)
    if not np.isclose(steps*dt, duration, atol=1e-12, rtol=0.):
        raise ValueError("duration must be an integral number of reference steps")
    return SplitOperator1D(grid, matrix, mass, dt).propagate(psi, steps, sample_stride=steps)


def compare(name, *, trajectories=128, seed=20261102, dt=4., duration=240.,
            points=512, reference_dt=1., max_step=None):
    if trajectories < 2:
        raise ValueError("at least two independent trajectories are needed for SEM")
    steps = round(duration/dt)
    if steps < 1 or not np.isclose(steps*dt, duration, atol=1e-12, rtol=0.):
        raise ValueError("duration must be a positive integral number of MASH steps")
    model = comparison_model(name)
    mass, center, width, momentum = 200., -3., .6, 5.
    reference_args = dict(duration=duration, mass=mass, center=center,
                          width=width, momentum=momentum)
    reference = quantum_packet(model, points=points, dt=reference_dt, **reference_args)
    temporal = quantum_packet(model, points=points, dt=reference_dt/2, **reference_args)
    spatial = quantum_packet(model, points=2*points, dt=reference_dt/2, **reference_args)
    children = np.random.SeedSequence(seed).spawn(trajectories)
    populations, initial_populations, energy_drift, hop_counts = [], [], [], []
    started = time.perf_counter()
    for child in children:
        nuclear_seed, electronic_seed = child.spawn(2)
        nuclear_rng = np.random.default_rng(nuclear_seed)
        q, p = np.zeros((1, 1, 3)), np.zeros((1, 1, 3))
        q[0, 0, 0] = nuclear_rng.normal(center, width)
        p[0, 0, 0] = nuclear_rng.normal(momentum, 1/(2*width))
        driver = MultistateMASHDynamics(
            model, [mass], 100., 1, dt, model.nstates, native=False,
            max_step=dt if max_step is None else max_step)
        state = driver.initialize(State(q, p), initial_state=0, basis="diabatic",
                                  rng=np.random.default_rng(electronic_seed))
        e0 = driver.energy(state)
        initial_populations.append(driver.populations(state, basis="diabatic"))
        count = 0
        for _ in range(steps):
            driver.step(state)
            count += len(driver.last_hops)
        populations.append(driver.populations(state, basis="diabatic"))
        energy_drift.append(driver.energy(state)-e0)
        hop_counts.append(count)
    elapsed = time.perf_counter()-started
    samples = np.asarray(populations)
    mean = samples.mean(axis=0)
    sem = samples.std(axis=0, ddof=1)/np.sqrt(trajectories)
    return {
        "model": name, "model_identity": model.to_dict(), "trajectories": trajectories,
        "seed": seed, "duration_au": duration, "dt_au": dt,
        "max_step_au": dt if max_step is None else max_step,
        "mass_electron": mass, "center_bohr": center, "position_std_bohr": width,
        "mean_momentum_au": momentum, "momentum_std_au": 1/(2*width),
        "initial_electronic_basis": "fixed_diabatic", "initial_electronic_state": 0,
        "population_basis": "fixed_diabatic", "estimator": "Runeson-Manolopoulos-2023",
        "mash_initial_mean": np.mean(initial_populations, axis=0).tolist(),
        "mash_final_mean": mean.tolist(), "mash_final_sem": sem.tolist(),
        "fourier_final_populations": spatial.populations[-1].tolist(),
        "absolute_population_error": abs(mean-spatial.populations[-1]).tolist(),
        "fourier_points": 2*points, "fourier_dt_au": reference_dt/2,
        "fourier_time_refinement_max_diff": float(np.max(abs(
            reference.populations[-1]-temporal.populations[-1]))),
        "fourier_space_refinement_max_diff": float(np.max(abs(
            temporal.populations[-1]-spatial.populations[-1]))),
        "fourier_max_edge_population": float(np.max(spatial.edge_population)),
        "fourier_max_norm_error": float(np.max(abs(spatial.norm-1.))),
        "mash_max_abs_final_energy_drift_hartree": float(np.max(np.abs(energy_drift))),
        "mash_total_events": int(np.sum(hop_counts)),
        "mash_elapsed_seconds": elapsed,
        "mash_microseconds_per_output_step": elapsed*1e6/(trajectories*steps),
        
        "limitations": ["finite nuclear packet and finite trajectory ensemble",
                        "pointwise SEM excludes method and discretization error",
                        "reference refinements do not establish all domain limits",
                        "P=1, real Hamiltonians; no ring-polymer or complex-SOC extension"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("tully1", "three_state_lvc", "both"), default="both")
    parser.add_argument("--trajectories", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--dt", type=float, default=4.)
    parser.add_argument("--duration", type=float, default=240.)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--reference-dt", type=float, default=1.)
    parser.add_argument("--max-step", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    names = ("tully1", "three_state_lvc") if args.model == "both" else (args.model,)
    results = [compare(name, trajectories=args.trajectories, seed=args.seed, dt=args.dt,
                       duration=args.duration, points=args.points,
                       reference_dt=args.reference_dt, max_step=args.max_step) for name in names]
    payload = json.dumps(results, indent=2, allow_nan=False)+"\n"
    if args.output:
        from pyrpmd.io import atomic_write_text
        atomic_write_text(args.output, payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
