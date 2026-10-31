"""Analytic charge-coupled slab for validation; no material/PCET fit is implied.

Reuses the existing periodic cosine PES for lateral motion. The height/charge
part is .5*k*z**2 + .5*a*N**2 - c*N*z, where z is relative to the fixed slab and
the chosen adsorption height. All input parameters and output arrays use AU.
"""

import numpy as np

from pyrpmd.electrochemistry import ElectronicPESResult
from pyrpmd.potentials import PeriodicCosinePotential


class ToyElectrode:
    metadata = {
        "supports_pbc": True, "thread_safety": "context_serial",
        "execution_device": "cpu", "model_scope": "analytic_validation_only",
    }

    def __init__(self, barrier=0.003, height=3.0, vertical_k=0.02, charging_energy=0.05, coupling=0.01):
        self.barrier, self.height, self.vertical_k, self.charging_energy, self.coupling = map(
            float, (barrier, height, vertical_k, charging_energy, coupling),
        )
        if (not np.isfinite([barrier, height, vertical_k, charging_energy, coupling]).all()
                or min(vertical_k, charging_energy) <= 0 or barrier < 0
                or vertical_k * charging_energy <= coupling**2):
            raise ValueError("toy electrode requires finite parameters and positive charge-height Hessian")
        self.base = None

    def bind_system(self, system):
        if system.natoms < 2 or system.pbc != (True, True, False):
            raise ValueError("toy electrode needs one mobile H, a fixed slab and pbc=[true,true,false]")
        if (not np.allclose(system.cell, np.diag(np.diag(system.cell)))
                or np.any(np.diag(system.cell) <= 0)):
            raise ValueError("toy electrode requires an orthorhombic positive cell")
        if set(system.fixed_atoms) != set(range(1, system.natoms)):
            raise ValueError("toy electrode requires atom 0 mobile and all slab atoms fixed")
        self.natoms = system.natoms
        self.base = PeriodicCosinePotential(system.cell, system.pbc, height=self.barrier, atoms=[0])

    def evaluate(self, q, electron_numbers):
        if self.base is None:
            raise ValueError("bind_system must supply the slab geometry first")
        if q.shape[1] != self.natoms:
            raise ValueError("toy electrode atom count changed")
        result = self.base.evaluate(q)
        z = q[:, 0, 2] - q[:, 1:, 2].mean(axis=1) - self.height
        n = electron_numbers
        energy = result.energy + 0.5 * self.vertical_k * z**2
        energy += 0.5 * self.charging_energy * n**2 - self.coupling * n * z
        force_z = -self.vertical_k * z + self.coupling * n
        forces = result.forces.copy()
        forces[:, 0, 2] += force_z
        forces[:, 1:, 2] -= force_z[:, None] / (self.natoms - 1)
        return ElectronicPESResult(energy, forces, self.charging_energy * n - self.coupling * z)


def build(**params):
    return ToyElectrode(**params)
