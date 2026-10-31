"""Small deterministic SOC-NRPMD and real centroid-RPSH numerical examples.

Run ``python examples/nonadiabatic/soc_and_rpsh.py`` from an installed checkout.
The printed JSON describes a synthetic two-state model in atomic units. This
is neither a chemical scattering benchmark nor a thermal-rate calculation.
"""
import argparse
import json

import numpy as np

from pyrpmd.dynamics import State
from pyrpmd.nonadiabatic import DiabaticResult, NRPMDDynamics, NRPMDState
from pyrpmd.surface_hopping import RPSHDynamics


class SmoothTwoState:
    """Fixed diabatic basis; soc=0 selects a real avoided-crossing model."""
    def __init__(self, soc=0.):
        self.soc = float(soc)

    def evaluate(self, q):
        x = q[:, 0, 0]
        dtype = complex if self.soc else float
        matrix = np.zeros((len(q), 2, 2), dtype=dtype)
        gradient = np.zeros((*matrix.shape, *q.shape[1:]), dtype=dtype)
        diagonal = .12*np.tanh(x)
        slope = .12*(1.-np.tanh(x)**2)
        coupling = (.09+1j*self.soc)*np.exp(-x*x) if self.soc else .09*np.exp(-x*x)
        matrix[:, 0, 0], matrix[:, 1, 1] = diagonal, -diagonal
        matrix[:, 0, 1], matrix[:, 1, 0] = coupling, np.conj(coupling)
        gradient[:, 0, 0, 0, 0], gradient[:, 1, 1, 0, 0] = slope, -slope
        gradient[:, 0, 1, 0, 0] = -2*x*coupling
        gradient[:, 1, 0, 0, 0] = np.conj(-2*x*coupling)
        return DiabaticResult(matrix, gradient)


def run_demo(steps=200):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    beads, beta, dt, seed = 4, 4., .01, 2026
    masses = [2.]
    q = np.zeros((beads, 1, 3))
    q[:, 0, 0] = [-.4, -.38, -.42, -.39]
    p = np.zeros_like(q)
    p[:, 0, 0] = 1.2
    initial = State(q, p)

    soc_potential = SmoothTwoState(soc=.07)
    mapping = NRPMDState.from_nuclear(initial, 2, 0, seed)
    soc = NRPMDDynamics(soc_potential, masses, beta, beads, dt, 2, native=False)
    initial_energy = soc.energy(mapping)
    energy_errors = []
    for _ in range(steps):
        soc.step(mapping)
        energy_errors.append(soc.energy(mapping)-initial_energy)

    rpsh = RPSHDynamics(SmoothTwoState(), masses, beta, beads, dt, 2,
                        seed=seed, native=False, electronic_substeps=8,
                        rescaling="centroid", frustrated_hops="keep")
    hopping = rpsh.initialize(initial, active_surface=0)
    accepted, frustrated, hop_energy_changes = 0, 0, []
    for _ in range(steps):
        rpsh.step(hopping)
        if rpsh.last_hop is not None:
            accepted += int(rpsh.last_hop.accepted)
            frustrated += int(rpsh.last_hop.frustrated)
            hop_energy_changes.append(rpsh.last_hop.ring_energy_change)

    return {
        "purpose": "finite synthetic-model numerical demonstration",
        "physical_converged": False,
        "chemistry_validation": False,
        "settings": {"units": "atomic", "nbeads": beads, "masses": masses,
                     "beta": beta, "dt": dt, "steps": steps, "seed": seed,
                     "initial_nuclei": "specified unthermalized ring"},
        "soc_nrpmd": {
            "initialization": "focused diabatic state 0",
            "initial_energy": initial_energy,
            "final_energy": soc.energy(mapping),
            "max_absolute_energy_drift": float(np.max(np.abs(energy_errors))),
            "final_populations_signed": mapping.populations().tolist(),
            "population_sum": float(mapping.populations().sum()),
            "final_centroid": mapping.q.mean(axis=0).tolist(),
        },
        "rpsh": {
            "method": "centroid approximation, real matrix",
            "rescaling": rpsh.rescaling,
            "frustrated_hops": rpsh.frustrated_hops,
            "electronic_substeps": rpsh.electronic_substeps,
            "accepted_hops": accepted,
            "frustrated_hop_count": frustrated,
            "hop_full_ring_energy_changes": hop_energy_changes,
            "final_active_surface": hopping.active_surface,
            "final_adiabatic_amplitude_populations": rpsh.populations(hopping).tolist(),
            "amplitude_norm": float(np.vdot(hopping.coefficients, hopping.coefficients).real),
            "final_centroid": hopping.q.mean(axis=0).tolist(),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.steps), indent=2, allow_nan=False))
