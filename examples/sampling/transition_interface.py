"""Small TIS double-well demonstration; prints diagnostics, not a physical rate.

Run from the checkout after installation:
    python examples/sampling/transition_interface.py --samples 200 --burn-in 50
"""

import argparse
import json

import numpy as np

from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import PESResult
from pyrpmd.transition_interface import TransitionInterfaceSampler


class DoubleWell:
    """A separable teaching model in atomic units, not a molecular PES."""

    def evaluate(self, q):
        x = q[:, :, 0]
        energy = np.sum((x * x - 1) ** 2, axis=1)
        energy += 0.5 * np.sum(q[:, :, 1:] ** 2, axis=(1, 2))
        force = -q.copy()
        force[:, :, 0] = -4 * x * (x * x - 1)
        return PESResult(energy, force)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--burn-in", type=int, default=50)
    parser.add_argument("--beads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=832)
    args = parser.parse_args()
    dynamics = Dynamics(DoubleWell(), [1], 1.0, args.beads, 0.04, native=False)
    sampler = TransitionInterfaceSampler(
        dynamics, Cartesian(0), basin_a=-0.95, basin_b=0.95,
        interfaces=[-0.8, -0.4, 0.0, 0.4], max_path_steps=1500, seed=args.seed,
    )
    q, p = np.zeros((args.beads, 1, 3)), np.zeros((args.beads, 1, 3))
    q[:, 0, 0], p[:, 0, 0] = -0.6, 1.5
    path = sampler.generate_path(State(q, p), interface_index=3)
    results = sampler.sample_interfaces(
        [path] * len(sampler.interfaces), n_samples=args.samples, burn_in=args.burn_in,
    )
    summaries = [result.summary() for result in results]
    expected = np.exp(-((0.4**2 - 1)**2 - (0.8**2 - 1)**2))
    print(json.dumps({
        "model": "separable_double_well_teaching_example",
        "beads": args.beads,
        "ensembles": summaries,
        "classical_first_conditional_reference": float(expected) if args.beads == 1 else None,
        "rate_s_inv": None,
        "converged": False,
        "qualification": "Short path-sampling demonstration; an independent effective flux and convergence studies are required for a rate.",
    }, indent=2))


if __name__ == "__main__":
    main()
