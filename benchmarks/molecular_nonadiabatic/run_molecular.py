#!/usr/bin/env python3
"""Run explicit nonequilibrium NaH2 NRPMD/RPSH molecular smoke calculations.

The geometry is a specified interaction-region phase point, not a thermally or
rovibrationally prepared asymptotic collision. The two methods initialize their
documented different electronic bases. This is an executed interface example,
not a converged scattering-probability, reaction-rate or method-accuracy study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np

from pyrpmd import __version__
from pyrpmd.checkpoint import atomic_savez
from pyrpmd.constants import AMU_TO_AU, AU_TIME_FS
from pyrpmd.dynamics import State
from pyrpmd.io import write_json
from pyrpmd.nonadiabatic import NRPMDDynamics
from pyrpmd.nonadiabatic_ensemble import run_nonadiabatic_ensemble
from pyrpmd.potentials.diabatic import check_diabatic_potential
from pyrpmd.potentials.multistate_external import NaH2LEPS5G
from pyrpmd.surface_hopping import RPSHDynamics


def run_case(method="nrpmd", *, steps=100, trajectories=8, beads=4, seed=20261101):
    masses_amu = np.array([22.98976928, 1.00782503223, 1.00782503223])
    masses = masses_amu*AMU_TO_AU
    geometry = np.array([[3.2, .8, 0.], [-.85, 0., 0.], [.85, 0., 0.]])
    # Equal/opposite Na versus H2 impulses give zero total linear momentum.
    reduced_mass = masses[0]*(masses[1]+masses[2])/masses.sum()
    impulse = np.sqrt(2*reduced_mass*.01)*geometry[0]/np.linalg.norm(geometry[0])
    momentum = np.stack([-impulse, .5*impulse, .5*impulse])
    q = np.repeat(geometry[None], beads, axis=0)
    p = np.repeat(momentum[None], beads, axis=0)
    parents = tuple(State(q, p) for _ in range(trajectories))
    potential = NaH2LEPS5G()
    beta, dt = 1000., .2

    def factory(index, rng):
        if method == "nrpmd":
            return NRPMDDynamics(potential, masses, beta, beads, dt, 2, native=False)
        return RPSHDynamics(potential, masses, beta, beads, dt, 2, seed=rng,
                            native=False, rescaling="ring", frustrated_hops="keep")

    electronic = {"initial_state": 1} if method == "nrpmd" else {"active_surface": 1}
    result = run_nonadiabatic_ensemble(parents, factory, steps=steps,
                                     sample_stride=max(1, steps//10), seed=seed,
                                     independent=True, **electronic)
    initial = potential.evaluate(geometry[None])
    summary = {
        "format": "PyRPMD.NaH2.molecular_smoke.v1", "completed": True,
        "physical_converged": False, "method": method, "model": potential.to_dict(),
        "symbols": list(potential.symbols), "masses_amu": masses_amu.tolist(),
        "masses_electron": masses.tolist(), "positions_bohr": geometry.tolist(),
        "momenta_atomic": momentum.tolist(), "beta_hartree_inverse": beta,
        "dt_atomic": dt, "beads": beads, "trajectories": trajectories, "steps": steps,
        "seed": seed, "duration_fs": steps*dt*AU_TIME_FS,
        "initialization": {"nuclear": "specified collapsed-ring phase point, not a thermal ensemble",
                           "electronic": "fixed diabatic state 1" if method == "nrpmd" else "centroid adiabatic eigenstate 1"},
        "initial_matrix_hartree": initial.matrix[0].tolist(),
        "initial_adiabatic_energies_hartree": np.linalg.eigvalsh(initial.matrix[0]).tolist(),
        "matrix_gradient_check": check_diabatic_potential(potential, geometry[None], 2),
        "final_diabatic_population_mean": result.mean_populations[-1].tolist(),
        "final_diabatic_population_sem": None if result.population_sem is None else result.population_sem[-1].tolist(),
        "max_abs_sum_bead_energy_drift_hartree": float(np.max(np.abs(result.energy_drift))),
        "max_abs_electronic_norm_drift": float(np.max(np.abs(result.norm_drift))),
        "software": {"pyrpmd": __version__, "numpy": np.__version__, "python": platform.python_version()},
    }
    return summary, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("nrpmd", "rpsh", "both"), default="both")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--beads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/molecular_nonadiabatic"))
    args = parser.parse_args()
    if args.steps < 0 or args.trajectories < 1 or args.beads < 1:
        parser.error("steps must be nonnegative and trajectories/beads positive")
    methods = ("nrpmd", "rpsh") if args.method == "both" else (args.method,)
    for method in methods:
        summary, result = run_case(method, steps=args.steps, trajectories=args.trajectories, beads=args.beads)
        write_json(args.output/f"{method}.json", summary)
        arrays = {"time_atomic": result.time, "diabatic_density": result.diabatic_density,
                  "population_mean": result.mean_populations, "energy": result.energy,
                  "electronic_norm": result.electronic_norm}
        if result.population_sem is not None:
            arrays["population_sem"] = result.population_sem
        if method == "rpsh":
            arrays.update(active_surfaces=result.active_surfaces,
                          adiabatic_amplitude_populations=result.adiabatic_amplitude_populations)
        atomic_savez(args.output/f"{method}.npz", kind=summary["format"], **arrays)
        print(json.dumps({"method": method, "completed": True,
                          "final_populations": summary["final_diabatic_population_mean"],
                          "max_energy_drift": summary["max_abs_sum_bead_energy_drift_hartree"],
                          "physical_converged": False}, sort_keys=True))


if __name__ == "__main__":
    main()
