"""RETIS with an internally sampled minus-ensemble flux, in atomic units.

    python examples/sampling/retis_minus.py --samples 300 --checkpoint retis.npz
    python examples/sampling/retis_minus.py --samples 300 --resume retis.npz

This separable quartic is a method demonstration, not molecular validation.
The analytic rate is classical; bead convergence does not establish exact
quantum dynamics. Increase samples, block size and path support independently.
"""

import argparse
import json

import numpy as np
from scipy.integrate import quad

from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import PESResult
from pyrpmd.retis import ReplicaExchangeTIS


class DoubleWell:
    def evaluate(self, q):
        x = q[:, :, 0]
        energy = np.sum((x*x - 1)**2, axis=1)
        energy += np.sum(q[:, :, 1:]**2, axis=(1, 2)) / 2
        force = -q.copy()
        force[:, :, 0] = -4*x*(x*x - 1)
        return PESResult(energy, force)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--burn-in", type=int, default=50)
    parser.add_argument("--block-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=981)
    parser.add_argument("--beads", type=int, default=1)
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    args = parser.parse_args()
    dynamics = Dynamics(DoubleWell(), [1.], 1., args.beads, .04, native=False)
    sampler = ReplicaExchangeTIS(
        dynamics, Cartesian(0), basin_a=-1., basin_b=1.,
        interfaces=[-1., -.5], max_path_steps=1000, seed=args.seed, minus=True,
        identity={"potential": "quartic-x-plus-harmonic-yz-v1", "coordinate": "centroid-x0-v1"},
    )
    if args.resume:
        sampler.load_checkpoint(args.resume)
        result = sampler.sample_interfaces(n_samples=args.samples)
    else:
        def point(x, momentum):
            q, p = np.zeros((args.beads, 1, 3)), np.zeros((args.beads, 1, 3))
            q[:, 0, 0], p[:, 0, 0] = x, momentum
            return State(q, p)
        positive = sampler.generate_path(point(-.3, 1.4), 1)
        negative = sampler.generate_minus_path(point(-1.1, -.7))
        result = sampler.sample_interfaces(
            [positive, positive], minus_path=negative,
            n_samples=args.samples, burn_in=args.burn_in)
    if args.checkpoint:
        sampler.save_checkpoint(args.checkpoint)
    report = result.summary(block_size=args.block_size)
    if args.beads == 1:
        partition = quad(lambda x: np.exp(-(x*x - 1)**2), -np.inf, np.inf)[0]
        report["exact_classical_flux_au"] = 2 / (np.sqrt(2*np.pi) * partition)
        report["exact_classical_rate_au"] = report["exact_classical_flux_au"] * np.exp(-1.)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
