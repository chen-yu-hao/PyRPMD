"""A deterministic ring-polymer API example; no equilibrium sampling claim."""

import numpy as np

from pyrpmd import Dynamics, State
from pyrpmd.constants import AMU_TO_AU, KB_AU
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.units import parse_quantity


def main():
    temperature = 300.0
    beads = 4
    masses = np.array([1.00782503223 * AMU_TO_AU])
    q = np.zeros((beads, 1, 3))
    q[:, 0, 0] = 0.5  # bohr; all beads initially coincide
    with HarmonicPotential(k=0.002) as potential:
        dynamics = Dynamics(
            potential, masses, beta=1.0 / (KB_AU * temperature),
            nbeads=beads, dt=parse_quantity("0.1 fs", "time"), seed=7,
            integrator="cayley-bcb", loop_backend="auto",
        )
        state = State(q, np.zeros_like(q))  # p is momentum, not velocity
        energy_before = dynamics.energy(state)
        dynamics.propagate(state, 100)
        print("steps:", state.step)
        print("centroid / bohr:", state.q.mean(axis=0))
        print("ring-Hamiltonian change / hartree:",
              dynamics.energy(state) - energy_before)
        print("backend counters:", dynamics.loop_stats)


if __name__ == "__main__":
    main()

