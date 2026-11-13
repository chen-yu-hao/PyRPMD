"""Restrained finite-temperature string with an analytically known entropic shift.

The hidden z coordinate integrates out to give
F(x,y) = (x*x-1)**2 + 5*y*y + 3*y/(2*beta) + constant.
The minimum free-energy valley has y=-3/(20*beta), whereas the PES valley has
y=0.  Small defaults are a runnable demonstration, not a convergence study.
"""

import argparse

import numpy as np

from pyrpmd.cv import Cartesian, CVStack
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.finite_temperature_string import RestrainedStringSampler, finite_temperature_string
from pyrpmd.potentials.base import PESResult


class EntropicDoubleWell:
    def evaluate(self, positions):
        x, y, z = np.asarray(positions)[:, 0].T
        coupling = np.exp(3 * y)
        energy = (x*x - 1)**2 + 5*y*y + .5*coupling*z*z
        force = np.column_stack((-4*x*(x*x-1), -10*y-1.5*coupling*z*z, -coupling*z))
        return PESResult(energy, force[:, None, :])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--sampling-steps", type=int, default=2000)
    parser.add_argument("--checkpoint", default="thermal-string.npz")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    beta = 2.
    x = np.linspace(-1., 1., 7)
    valley = -3 / (20 * beta)
    images = np.column_stack((x, valley + .15*np.sin(np.linspace(0., np.pi, len(x)))))
    dynamics, states = [], []
    for index, image in enumerate(images):
        dyn = Dynamics(EntropicDoubleWell(), [1.], beta, 1, .01, seed=100+index,
                       potential_identity="thermal-string-entropic-doublewell-v1")
        q = np.array([[[image[0], image[1], 0.]]])
        states.append(State(q, dyn.sample_momenta()))
        dynamics.append(dyn)
    cv = CVStack([Cartesian(0, 0), Cartesian(0, 1)])
    sampler = RestrainedStringSampler(dynamics, cv, states, stiffness=50.,
                                      equilibration_steps=500,
                                      sampling_steps=args.sampling_steps,
                                      sample_interval=2, block_size=50,
                                      gamma_centroid=2.)
    result = finite_temperature_string(images, sampler, step=.01, tolerance=.1,
                                       max_iterations=args.iterations,
                                       checkpoint=args.checkpoint, resume=args.resume)
    print("x [bohr], y [bohr], restrained free energy [hartree], block error [hartree]")
    print(np.column_stack((result.images, result.free_energy, result.free_energy_standard_error)))
    print(f"Infinite-stiffness thermal valley: y={valley:.6f} bohr")
    print(result.diagnostics)


if __name__ == "__main__":
    main()
