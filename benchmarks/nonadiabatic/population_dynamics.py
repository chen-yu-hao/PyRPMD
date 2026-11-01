#!/usr/bin/env python3
"""Reproducible one-mode NRPMD versus quantum population time series.

Run from a checkout with PYTHONPATH=src, or after installing PyRPMD. The JSON
contains convergence differences, Monte Carlo SEMs, and method differences as
separate quantities. No closeness-to-quantum pass/fail criterion is imposed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy

from pyrpmd import __version__
from pyrpmd.analysis.quantum_reference import SplitOperator1D, harmonic_canonical_ensemble
from pyrpmd.nonadiabatic import NRPMDDynamics
from pyrpmd.nonadiabatic_ensemble import run_nonadiabatic_ensemble
from pyrpmd.nonadiabatic_sampling import sample_harmonic_ring
from pyrpmd.potentials.nonadiabatic_models import LinearVibronicCoupling


def _positive_int(value):
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return integer


def _positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def quantum_series(model, *, points, half_width, mass, omega, beta, dt, steps, stride):
    grid = np.linspace(-half_width, half_width, points, endpoint=False)
    thermal = harmonic_canonical_ensemble(grid, mass, omega, beta, tail_tolerance=1e-12)
    q = np.zeros((points, 1, 3)); q[:, 0, 0] = grid
    solver = SplitOperator1D(grid, model.evaluate(q).matrix, mass, dt)
    wavefunctions = np.zeros((len(thermal.weights), points, model.nstates), complex)
    wavefunctions[:, :, 0] = thermal.wavefunctions
    result = solver.propagate(wavefunctions, steps, weights=thermal.weights, sample_stride=stride)
    q2 = solver.dx*np.einsum("m,mx,x->", thermal.weights, abs(thermal.wavefunctions)**2, grid**2)
    return result, {"components": len(thermal.weights),
                    "finite_grid_omitted_thermal_weight": thermal.omitted_weight,
                    "initial_position_second_moment": float(q2)}


def run_benchmark(*, trajectories, beads, steps, dt, grid_points, half_width, sample_stride, seed):
    if trajectories < 2 or grid_points < 4 or seed < 0:
        raise ValueError("need at least two trajectories, four grid points, and a nonnegative seed")
    mass, omega, beta = 1., 1., 2.
    model = LinearVibronicCoupling(
        [[0., 0.15], [0.15, 0.3]], [[[0.35, 0.05], [0.05, -0.35]]],
        harmonic=[mass*omega**2], coordinates=[0], natoms=1)
    sequences = np.random.SeedSequence(seed).spawn(2)
    nuclear_seed, mapping_seed = (int(s.generate_state(1)[0]) for s in sequences)
    parents = sample_harmonic_ring([mass], beta, beads, [omega], count=trajectories, seed=nuclear_seed)

    def ensemble(timestep, refinement):
        return run_nonadiabatic_ensemble(
            parents,
            lambda index, rng: NRPMDDynamics(model, [mass], beta, beads, timestep, 2, native=False),
            steps=steps*refinement, sample_stride=sample_stride*refinement,
            seed=mapping_seed, initial_state=0, independent=True)

    nrpmd = ensemble(dt, 1)
    refined_nrpmd = ensemble(dt/2, 2)
    reference_options = dict(model=model, points=grid_points, half_width=half_width,
                             mass=mass, omega=omega, beta=beta)
    coarse_quantum, _ = quantum_series(**reference_options, dt=dt/4, steps=steps*4, stride=sample_stride*4)
    quantum, thermal = quantum_series(**reference_options, dt=dt/8, steps=steps*8, stride=sample_stride*8)
    fine_grid, _ = quantum_series(**{**reference_options, "points": 2*grid_points},
                                  dt=dt/8, steps=steps*8, stride=sample_stride*8)
    wide_box, _ = quantum_series(**{**reference_options, "points": 2*grid_points, "half_width": 2*half_width},
                                 dt=dt/8, steps=steps*8, stride=sample_stride*8)
    for result in (refined_nrpmd, coarse_quantum, quantum, fine_grid, wide_box):
        if not np.allclose(result.time, nrpmd.time, rtol=0., atol=1e-12):
            raise RuntimeError("benchmark observation times do not agree")

    paired = refined_nrpmd.populations-nrpmd.populations
    initial_q2 = np.array([np.mean(parent.q[:, 0, 0]**2) for parent in parents])
    ring_frequency = 2*beads/beta*np.sin(np.pi*np.arange(beads)/beads)
    finite_bead_q2 = float(np.sum(1/(omega**2+ring_frequency**2))/(beta*mass))
    quantum_q2 = float(1/(2*mass*omega*np.tanh(beta*omega/2)))
    maxdiff = lambda a, b: float(np.max(np.abs(a-b)))
    return {
        "schema": "pyrpmd.nonadiabatic_population_benchmark", "version": 1,
        "physical_converged": False,
        "software_versions": {"pyrpmd": __version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "units": {"length": "bohr", "mass": "electron_mass", "energy": "hartree", "time": "atomic_time"},
        "configuration": {"trajectories": trajectories, "beads": beads, "steps": steps,
                          "dt": dt, "sample_stride": sample_stride, "grid_points": grid_points,
                          "half_width": half_width, "mass": mass, "omega": omega, "beta": beta,
                          "seed": seed, "nuclear_seed": nuclear_seed, "mapping_seed": mapping_seed,
                          "model": model.to_dict()},
        "initial_state": {
            "definition": "canonical uncoupled harmonic nucleus times pure diabatic electronic state 0",
            "nrpmd_preparation": "independent finite-P primitive harmonic rings and focused MMST electronic variables",
            "reference_preparation": "finite Fourier-grid harmonic canonical density matrix",
            "analytic_quantum_position_second_moment": quantum_q2,
            "finite_bead_position_second_moment": finite_bead_q2,
            "finite_bead_minus_quantum_second_moment": finite_bead_q2-quantum_q2,
            "sample_position_second_moment": float(initial_q2.mean()),
            "sample_position_second_moment_sem": float(initial_q2.std(ddof=1)/np.sqrt(trajectories)),
            **thermal},
        "time": nrpmd.time.tolist(),
        "quantum_reference": {
            "method": "periodic Fourier-grid symmetric split operator; no absorber",
            "dt": dt/8, "populations": quantum.populations.tolist(),
            "max_component_norm_error": float(np.max(abs(quantum.component_norms-1))),
            "max_outer_five_percent_population": float(quantum.edge_population.max()),
            "numerical_differences": {
                "halved_dt_max_abs_population_change": maxdiff(quantum.populations, coarse_quantum.populations),
                "doubled_grid_max_abs_population_change": maxdiff(fine_grid.populations, quantum.populations),
                "doubled_box_same_dx_max_abs_population_change": maxdiff(wide_box.populations, quantum.populations)},
            "interpretation": "refinement differences estimate numerical sensitivity; they are not rigorous error bounds"},
        "nrpmd": {
            "mean_populations": nrpmd.mean_populations.tolist(),
            "population_sem": nrpmd.population_sem.tolist(),
            "population_estimator": "signed bead-averaged MMST; no clipping",
            "paired_halved_dt_mean_population_change": paired.mean(axis=0).tolist(),
            "paired_halved_dt_change_sem": (paired.std(axis=0, ddof=1)/np.sqrt(trajectories)).tolist(),
            "max_energy_drift_per_bead": float(np.max(abs(nrpmd.energy_drift))/beads),
            "max_mapping_norm_drift": float(np.max(abs(nrpmd.norm_drift))),
            "sem_interpretation": "pointwise Monte Carlo standard error across independent trajectories; not method uncertainty"},
        "nrpmd_minus_quantum_population": (nrpmd.mean_populations-quantum.populations).tolist(),
        "comparison_interpretation": (
            "The observed method difference still contains finite-bead preparation, propagation and Monte Carlo errors. "
            "A trajectory method is not required to equal the quantum result. Increase trajectories and beads and "
            "refine time step, grid and box before drawing physical conclusions."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="small reproducible plumbing and numerical-diagnostics run")
    parser.add_argument("--output", type=Path, help="also save the printed JSON to this file")
    for flag in ("trajectories", "beads", "steps", "grid-points", "sample-stride"):
        parser.add_argument("--"+flag, type=_positive_int)
    for flag in ("dt", "half-width"):
        parser.add_argument("--"+flag, type=_positive_float)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    defaults = dict(trajectories=16 if args.smoke else 128, beads=8 if args.smoke else 16,
                    steps=40 if args.smoke else 200, dt=0.05 if args.smoke else 0.025,
                    grid_points=48 if args.smoke else 96, half_width=8., sample_stride=5)
    config = {key: getattr(args, key) if getattr(args, key) is not None else value for key, value in defaults.items()}
    if config["trajectories"] < 2 or config["grid_points"] < 4 or args.seed < 0:
        parser.error("need at least two trajectories, four grid points, and a nonnegative seed")
    encoded = json.dumps(run_benchmark(**config, seed=args.seed), indent=2, allow_nan=False)+"\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
