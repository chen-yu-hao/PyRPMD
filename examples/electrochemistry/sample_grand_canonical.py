"""Shared-N equilibrium sample on the analytic slab, compared with Gaussian theory.

The short default is a reproducible software demonstration, not a converged
electrode calculation. HMC/MC sweep numbers are never converted to seconds.
"""

import argparse
import json
from pathlib import Path
import numpy as np

from pyrpmd.electrochemistry import GrandCanonicalSampler, GrandCanonicalState
from pyrpmd.input import JobSpec
from pyrpmd.potentials import make_potential


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps", type=int, default=2000)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--nuclear-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7021)
    args = parser.parse_args(argv)
    if args.sweeps < 2 or args.burn_in < 0 or args.nuclear_steps < 1:
        parser.error("need sweeps >= 2, burn-in >= 0 and nuclear-steps >= 1")
    config, system = JobSpec.from_file(Path(__file__).with_name("fes2d.inp")).resolve()
    provider = make_potential(config["potential"], system)
    try:
        electrode = provider.obj
        model = electrode.model  # Already bound to this fixed cell by make_potential.
        sampler = GrandCanonicalSampler(
            model, system.masses, config["beta"], config["beads"], config["dt_au"],
            reservoir=electrode.reservoir, electron_bounds=electrode.electron_bounds,
            proposal_width=0.3, seed=args.seed, fixed_atoms=system.fixed_atoms,
            fixed_positions=system.geometry,
        )
        state = GrandCanonicalState(np.repeat(system.geometry[None], config["beads"], axis=0), 0.0)
        values = []
        for sweep in range(args.burn_in + args.sweeps):
            sampler.step(state, nuclear_steps=args.nuclear_steps, electron_moves=2)
            if sweep >= args.burn_in:
                z = float(state.q[:, 0, 2].mean() - state.q[:, 1:, 2].mean() - model.height)
                values.append([z, state.electron_number])
        values = np.asarray(values)
        stiffness = np.array([[model.vertical_k, -model.coupling], [-model.coupling, model.charging_energy]])
        exact_mean = np.linalg.solve(stiffness, [0.0, electrode.reservoir.chemical_potential_au])
        exact_covariance = np.linalg.inv(stiffness) / config["beta"]
        report = {
            "model": "analytic slab, shared electron number; no material fit",
            "ensemble": "shared_number_grand_canonical_ring_polymer",
            "physical_time_available": False, "physical_converged": False,
            "seed": args.seed, "beads": config["beads"], "temperature_K": config["temperature_K"],
            "burn_in_sweeps": args.burn_in, "saved_sweeps": args.sweeps,
            "observables": ["centroid_height_offset_bohr", "electron_number"],
            "sample_mean": values.mean(axis=0).tolist(),
            "sample_covariance": np.cov(values, rowvar=False).tolist(),
            "unbounded_gaussian_mean": exact_mean.tolist(),
            "unbounded_gaussian_covariance": exact_covariance.tolist(),
            "reference_note": "number bounds ±4 are many thermal widths away; finite sampling is correlated",
            "native_free_ring_kernel": sampler.native, "sampling": sampler.statistics,
        }
        print(json.dumps(report, indent=2, allow_nan=False))
        return report
    finally:
        provider.close()


if __name__ == "__main__":
    main()
