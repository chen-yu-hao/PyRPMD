"""Small deterministic additive-delta / RPC / r-RESPA API pilot in atomic units."""

import json

import numpy as np

from pyrpmd import DeltaLearningPotential
from pyrpmd.dynamics import State
from pyrpmd.multitime import MTSDynamics
from pyrpmd.potentials import HarmonicPotential


def main():
    model = DeltaLearningPotential(
        HarmonicPotential(k=0.002), HarmonicPotential(k=0.0002), owns_members=True,
    )
    try:
        split = model.split(nbeads=8, contracted_beads=1)
        dyn = MTSDynamics(split, [1837.0], beta=1000.0, nbeads=8,
                          dt=2.0, inner_steps=4, seed=221)
        rng = np.random.default_rng(221)
        state = State(rng.normal(scale=0.1, size=(8, 1, 3)), dyn.sample_momenta())
        before = dyn.energy(state)
        dyn.propagate(state, 100)
        after = dyn.energy(state)
        print(json.dumps({"outer_steps": state.step, "time_au": state.time,
                          "hamiltonian_initial_hartree": before,
                          "hamiltonian_final_hartree": after,
                          "hamiltonian_change_hartree": after - before,
                          "force_evaluations": dyn.force_evaluations,
                          "approximate_hamiltonian": split.metadata["approximate_hamiltonian"],
                          "converged": False}, indent=2, allow_nan=False))
    finally:
        model.close()


if __name__ == "__main__":
    main()
