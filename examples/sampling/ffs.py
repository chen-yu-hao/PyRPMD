"""Direct FFS for an explicitly stochastic overdamped double well.

Run from a PyRPMD installation::

    python examples/sampling/ffs.py --replicates 4 --flux-steps 12000 --trials 400

All quantities use atomic units. This Euler--Maruyama model is overdamped
classical dynamics, not Hamiltonian MD or RPMD; its rates cannot be substituted
for a deterministic TIS/RPMD calculation. Equilibration, discretization and
independent-run convergence remain the user's responsibility.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from pyrpmd.dynamics import State
from pyrpmd.ffs import ForwardFluxSampler, combine_ffs_replicates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--flux-steps", type=int, default=12000)
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--seed", type=int, default=5821)
    args = parser.parse_args()
    if args.replicates < 2:
        parser.error("at least two independent replicates are needed for error diagnostics")
    beta, dt, mobility = 2.0, 0.005, 1.0

    def overdamped_step(state, rng):
        q = state.q
        force = -q.copy()
        force[:, :, 0] = -4 * q[:, :, 0] * (q[:, :, 0] ** 2 - 1)
        q += mobility * force * dt + np.sqrt(2 * mobility * dt / beta) * rng.normal(size=q.shape)
        # Momenta are unused zero placeholders in this explicitly overdamped
        # process; a phase-space process must preserve its real momenta here.

    runs = []
    streams = np.random.SeedSequence(args.seed).spawn(2 * args.replicates)
    for i in range(args.replicates):
        q = np.zeros((1, 1, 3))
        q[:, :, 0] = -1
        initial = State(q, np.zeros_like(q))
        equilibration_rng = np.random.default_rng(streams[2 * i])
        for _ in range(12000):
            overdamped_step(initial, equilibration_rng)
        ffs = ForwardFluxSampler(
            overdamped_step, lambda q: q[:, 0, 0].mean(),
            dt=dt, basin_a=-0.9, basin_b=0.9, interfaces=(-0.7, -0.3, 0.2, 0.6),
            dynamics_id=f"overdamped-em-quartic-harmonic-v1; beta={beta}; mobility={mobility}",
            coordinate_id="centroid-atom0-x-v1", flux_steps=args.flux_steps,
            trials_per_interface=args.trials, max_trial_steps=20000,
            seed=np.random.default_rng(streams[2 * i + 1]),
        )
        runs.append(ffs.run(initial))
        # To interrupt a run, use run(initial, n_operations=...) followed by
        # save_checkpoint(path); a matching new sampler can load_checkpoint
        # and call advance() to continue with identical noise and genealogy.
    report = {
        "individual_runs": [run.summary() for run in runs],
        "combined": combine_ffs_replicates(runs) if all(run.complete for run in runs) else None,
        "qualification": (
            "All replicates must contribute, including empirical zero rates. "
            "If any run never acquires A-history observation time, increase the flux run "
            "length for every replicate; incomplete runs are not silently dropped."
        ),
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
