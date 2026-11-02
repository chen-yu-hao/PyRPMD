"""PPTIS/REPPTIS quartic-well demonstration with exact classical references.

    python examples/sampling/pptis.py --samples 300 --burn-in 50 --checkpoint chain.npz
    python examples/sampling/pptis.py --samples 300 --resume chain.npz

The flux supplied here is analytic for this separable teaching potential. A
molecular calculation needs independently measured matching effective flux.
"""

import argparse
import json

import numpy as np
from scipy.integrate import quad

from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.io import jsonable
from pyrpmd.potentials import PESResult
from pyrpmd.pptis import PPTISSampler, ReplicaExchangePPTIS, combine_pptis_rate


class DoubleWell:
    def evaluate(self, q):
        x = q[:, :, 0]
        energy = np.sum((x * x - 1) ** 2, axis=1)
        energy += .5 * np.sum(q[:, :, 1:] ** 2, axis=(1, 2))
        force = -q.copy()
        force[:, :, 0] = -4 * x * (x * x - 1)
        return PESResult(energy, force)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--burn-in", type=int, default=50)
    parser.add_argument("--seed", type=int, default=325)
    parser.add_argument("--without-exchange", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    args = parser.parse_args()
    dynamics = Dynamics(DoubleWell(), [1.], 1., 1, .04, native=False)
    cls = PPTISSampler if args.without_exchange else ReplicaExchangePPTIS
    sampler = cls(dynamics, Cartesian(0), basin_a=-1., basin_b=1.,
                  interfaces=[-.8, -.4], max_path_steps=500, seed=args.seed,
                  identity={"potential": "quartic-x-plus-harmonic-yz-v1", "coordinate": "x0"})
    if args.resume:
        result = sampler.sample_interfaces(n_samples=args.samples, resume=sampler.load_checkpoint(args.resume))
    else:
        initial = []
        for i, center in enumerate(sampler.interfaces):
            q, p = np.zeros((1, 1, 3)), np.zeros((1, 1, 3))
            q[0, 0, 0], p[0, 0, 0] = center, 1.5
            initial.append(sampler.generate_path(State(q, p), i))
        result = sampler.sample_interfaces(initial, n_samples=args.samples, burn_in=args.burn_in)
    if args.checkpoint:
        sampler.save_checkpoint(args.checkpoint, result)
    partition = quad(lambda x: np.exp(-(x*x-1)**2), -np.inf, np.inf)[0]
    flux = {"basin_a": -1., "basin_b": 1., "first_interface": -.8, "dt_au": .04,
            "flux_au": 2*np.exp(-(.8**2-1)**2)/(np.sqrt(2*np.pi)*partition)}
    report = combine_pptis_rate(result, flux, block_size=50)
    report["exact_classical_rate_au"] = 2*np.exp(-1)/(np.sqrt(2*np.pi)*partition)
    report["flux_source"] = "analytic equilibrium flux for this separable teaching model"
    print(json.dumps(jsonable(report), indent=2))


if __name__ == "__main__":
    main()
