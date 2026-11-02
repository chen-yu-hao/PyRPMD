"""Run direct adiabatic data dynamics on an analytical two-state model.

Replace ElectronicStructureModel.evaluate with a quantum-chemistry call that
returns the same quantities. This example validates the interface; it does not
claim an ab initio molecular benchmark. All quantities are atomic units.
"""
import argparse
import json

import numpy as np

from pyrpmd.adiabatic import (
    AdiabaticPythonPotential, AdiabaticResult, AdiabaticRPSHDynamics, QDNRPMDDynamics,
)
from pyrpmd.dynamics import State
from pyrpmd.mash import MASHDynamics


class ElectronicStructureModel:
    def evaluate(self, q, *, previous_q=None, previous_context=None):
        b, n, _ = q.shape
        matrix = np.zeros((b, 2, 2))
        scalar = .04*np.sum(q*q, axis=(1, 2))
        matrix[:, 0, 0], matrix[:, 1, 1] = scalar+.12*q[:, 0, 0], scalar-.12*q[:, 0, 0]
        matrix[:, 0, 1] = matrix[:, 1, 0] = .08
        derivative = np.zeros((b, 2, 2, n, 3))
        derivative[:, 0, 0] = derivative[:, 1, 1] = .08*q
        derivative[:, 0, 0, 0, 0] += .12
        derivative[:, 1, 1, 0, 0] -= .12
        energy, wavefunctions = np.linalg.eigh(matrix)
        operator_gradient = np.einsum("bsi,bstac,btj->bijac", wavefunctions, derivative, wavefunctions)
        gradient = np.einsum("biiac->biac", operator_gradient)
        gap = energy[:, None, :]-energy[:, :, None]
        gap[:, np.arange(2), np.arange(2)] = 1.
        nacs = operator_gradient/gap[..., None, None]
        nacs[:, np.arange(2), np.arange(2)] = 0.
        overlap = (None if previous_context is None else
                   previous_context.swapaxes(1, 2)@wavefunctions)
        return AdiabaticResult(energy, gradient, nacs, overlap, context=wavefunctions)


def run(method="qd-nrpmd", beads=1, steps=50):
    if method not in ("qd-nrpmd", "rpsh", "mash"):
        raise ValueError("method must be qd-nrpmd, rpsh or mash")
    provider = AdiabaticPythonPotential(ElectronicStructureModel())
    engine = {"qd-nrpmd": QDNRPMDDynamics, "rpsh": AdiabaticRPSHDynamics, "mash": MASHDynamics}[method]
    dynamics = engine(provider, [2.], 8., beads, .025, 2, native=False)
    rng = np.random.default_rng(13)
    nuclear = State(rng.normal(scale=.2, size=(beads, 1, 3)), rng.normal(size=(beads, 1, 3)))
    state = dynamics.initialize(nuclear, 0)
    before = dynamics.energy(state)
    for _ in range(steps):
        dynamics.step(state)
    result = {"method": method, "model": "analytical_two_state_interface_check", "beads": beads,
              "steps": steps, "time_au": state.time, "populations": dynamics.populations(state).tolist(),
              "hamiltonian_change_hartree": dynamics.energy(state)-before}
    provider.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("qd-nrpmd", "rpsh", "mash"), default="qd-nrpmd")
    parser.add_argument("--beads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(run(args.method, args.beads, args.steps), indent=2))
