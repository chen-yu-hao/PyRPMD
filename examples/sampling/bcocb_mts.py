"""Analytic CPU example; not a chemistry or equilibrium-convergence claim."""
import json
import numpy as np
from pyrpmd.dynamics import State
from pyrpmd.multitime import MTSDynamics
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.potentials.contraction import SplitPotential


def main():
    P = 8
    split = SplitPotential(HarmonicPotential(.9), HarmonicPotential(.1), P, 3)
    dynamics = MTSDynamics(
        split, [1.], 4., P, .04, inner_steps=4,
        integrator="normal-mode-bab", thermostat="pile-l",
        sampling_integrator="bcocb", seed=17,
    )
    state = State(np.zeros((P, 1, 3)), dynamics.sample_momenta())
    dynamics.propagate(state, 200)
    print(json.dumps({"outer_steps": state.step, "time_au": state.time,
                      "force_evaluations": dynamics.force_evaluations,
                      "note": "RPC changes the Hamiltonian; converge Pc and both timesteps."}, indent=2))


if __name__ == "__main__":
    main()
