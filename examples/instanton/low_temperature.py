"""Deterministic 2D quartic model: estimates, thermal rate and ground splitting.

Validation model only, not a fitted molecular PES. All coordinates, masses and
Hessians use atomic units. No trajectory propagation or production files.
"""
import json

import numpy as np

from pyrpmd.instanton import (RingPolymerAction, crossover_temperature,
                             instanton_seed, optimize_instanton)
from pyrpmd.low_temperature import HarmonicState, harmonic_cross_checks, instanton_rate, print_cross_checks
from pyrpmd.potentials.base import PESResult, Potential
from pyrpmd.tunnelling import tunnelling_splitting


class Quartic(Potential):
    def evaluate(self, q):
        x, y = q[:, 0, 0], q[:, 0, 1]
        f = np.zeros_like(q)
        f[:, 0, 0] = -0.04*x*(x*x-1)
        f[:, 0, 1] = -0.02*y
        return PESResult(0.01*(x*x-1)**2+0.01*y*y, f)


def main():
    potential = Quartic()
    masses = np.array([1822.])
    mask = np.array([[True, True, False]])
    ts = np.zeros((1, 3))
    left, right = np.array([[-1., 0., 0.]]), np.array([[1., 0., 0.]])
    # These Hessians are analytic, showing how to reuse already computed data.
    saddle = HarmonicState.from_potential(potential, ts, masses, active_mask=mask,
                                          hessian=np.diag([-.04, .02]))
    reactant = HarmonicState.from_potential(potential, left, masses, active_mask=mask,
                                            hessian=np.diag([.08, .02]))
    product = HarmonicState.from_potential(potential, right, masses, active_mask=mask,
                                           hessian=np.diag([.08, .02]))
    temperature = 0.55*crossover_temperature(potential, ts, masses, active_mask=mask)
    print_cross_checks(harmonic_cross_checks(temperature, saddle, [reactant], products=[product]))
    seed = instanton_seed(ts, [[1., 0., 0.]], 128, .95)
    orbit = optimize_instanton(potential, masses, temperature, seed, active_mask=mask,
                              transition_state=ts, retain_hessian=True)
    action = RingPolymerAction(potential, masses, temperature, active_mask=mask)
    print(json.dumps(instanton_rate(action, orbit, [reactant]), indent=2, allow_nan=False))
    duration = 28/np.sqrt(.08/masses[0])
    tau = np.linspace(-duration/2, duration/2, 257)
    path = np.zeros((len(tau), 1, 3))
    path[:, 0, 0] = np.tanh(np.sqrt(.08/masses[0])*tau/2)
    path[0], path[-1] = left, right
    splitting = tunnelling_splitting(potential, masses, path, duration, active_mask=mask,
                                      equivalent_wells=True,
                                      endpoint_hessians=[reactant.hessian, product.hessian])
    splitting.pop("path_bohr")
    print(json.dumps(splitting, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
