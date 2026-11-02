"""Analytic isolated q-TIP4P/F monomers for the f-QCMD example.

Only Eqs. (2)--(4) and the printed Table I parameters of Habershon, Markland
and Manolopoulos, JCP 131, 024501 (2009), DOI:10.1063/1.3167790, are used.
The intermolecular Coulomb/Lennard-Jones terms are absent. Multiple O,H,H
triples are independent gas-phase monomers, never a model of liquid water.

Coordinates, energies, forces, dipoles and masses use atomic units. This
small NumPy implementation has no OpenMM or other optional dependencies.
"""

from __future__ import annotations

import numpy as np

from pyrpmd.constants import AMU_TO_AU, ANGSTROM_TO_AU, BOHR_ANGSTROM, KCAL_MOL_TO_AU
from pyrpmd.potentials.base import PESResult, Potential, coordinates


PARAMETERS = {
    "D_r_kcal_mol": 116.09,
    "alpha_r_per_angstrom": 2.287,
    "r_eq_angstrom": 0.9419,
    "k_theta_kcal_mol_rad2": 87.85,
    "theta_eq_degrees": 107.4,
    "q_M_e": 1.1128,
    "gamma": 0.73612,
}
MONOMER_MASSES = np.array([15.99491461957, 1.00782503223, 1.00782503223]) * AMU_TO_AU


def geometry(copies=1):
    """Equilibrium O,H,H coordinates for independent, identically oriented copies."""
    if isinstance(copies, bool) or not isinstance(copies, int) or copies < 1:
        raise ValueError("copies must be a positive integer")
    angle = np.deg2rad(PARAMETERS["theta_eq_degrees"])
    radius = PARAMETERS["r_eq_angstrom"] * ANGSTROM_TO_AU
    q = radius * np.array([[0., 0., 0.], [1., 0., 0.],
                          [np.cos(angle), np.sin(angle), 0.]])
    # An explicit offset makes accidental interactions between copies obvious.
    return (q[None] + np.arange(copies)[:, None, None] * np.array([10., 0., 0.])).reshape(-1, 3)


def _molecular(q):
    q = coordinates(q)
    if q.shape[1] % 3:
        raise ValueError("q-TIP4P/F monomers require O,H,H triples")
    return q, q.reshape(len(q), -1, 3, 3)


def molecular_dipoles(q):
    """Per-monomer neutral virtual-site dipoles [batch,monomer,3], e*bohr."""
    _, mol = _molecular(q)
    bonds = mol[:, :, 1:] - mol[:, :, :1]
    return .5 * PARAMETERS["q_M_e"] * PARAMETERS["gamma"] * bonds.sum(axis=2)


class QTIP4PFMonomers(Potential):
    """Noninteracting flexible gas-phase monomers with analytic forces."""

    metadata = {
        "provider": "qtip4pf_noninteracting_monomers",
        "model_scope": "isolated_noninteracting_bent_triatomics",
        "thread_safety": "reentrant",
        "execution_device": "cpu",
        "native_batch": True,
        "translation_invariant": True,
        "rotation_invariant": True,
        "source_doi": "10.1063/1.3167790",
        "source_equations": [2, 3, 4],
        "source_parameter_table": "I",
        "parameters": PARAMETERS,
    }

    def bind_system(self, system):
        expected = ("O", "H", "H") * (system.natoms // 3)
        if tuple(system.symbols) != expected or not expected or np.any(system.pbc):
            raise ValueError("this example requires nonperiodic O,H,H monomers")

    def evaluate(self, q):
        q, mol = _molecular(q)
        bonds = mol[:, :, 1:] - mol[:, :, :1]
        radius = np.linalg.norm(bonds, axis=-1)
        if np.any(radius <= 0):
            raise ValueError("zero O-H bond length")
        direction = bonds / radius[..., None]
        cosine = np.sum(direction[:, :, 0] * direction[:, :, 1], axis=-1)
        cosine = np.clip(cosine, -1., 1.)
        sine = np.sqrt(1 - cosine * cosine)
        if np.any(sine <= 1e-8):
            raise ValueError("collinear water angle has a singular Cartesian derivative")

        alpha = PARAMETERS["alpha_r_per_angstrom"] * BOHR_ANGSTROM
        displacement = alpha * (radius - PARAMETERS["r_eq_angstrom"] * ANGSTROM_TO_AU)
        depth = PARAMETERS["D_r_kcal_mol"] * KCAL_MOL_TO_AU
        bond_energy = depth * (displacement**2 - displacement**3 + (7 / 12) * displacement**4)
        bond_derivative = depth * alpha * (2 * displacement - 3 * displacement**2
                                          + (7 / 3) * displacement**3)
        hydrogen_force = -bond_derivative[..., None] * direction

        angle_displacement = np.arccos(cosine) - np.deg2rad(PARAMETERS["theta_eq_degrees"])
        stiffness = PARAMETERS["k_theta_kcal_mol_rad2"] * KCAL_MOL_TO_AU
        angle_energy = .5 * stiffness * angle_displacement**2
        coefficient = stiffness * angle_displacement / sine
        hydrogen_force[:, :, 0] += (coefficient / radius[:, :, 0])[..., None] * (
            direction[:, :, 1] - cosine[..., None] * direction[:, :, 0])
        hydrogen_force[:, :, 1] += (coefficient / radius[:, :, 1])[..., None] * (
            direction[:, :, 0] - cosine[..., None] * direction[:, :, 1])
        force = np.empty_like(mol)
        force[:, :, 1:] = hydrogen_force
        force[:, :, 0] = -hydrogen_force.sum(axis=2)
        forces = force.reshape(q.shape)
        energy = bond_energy.sum(axis=(1, 2)) + angle_energy.sum(axis=1)
        # Each monomer has zero net force; subtracting its oxygen origin avoids
        # loss of precision when free centers of mass diffuse far from zero.
        relative = mol - mol[:, :, :1]
        virial = np.einsum("bmai,bmaj->bij", relative, force)
        return PESResult(energy, forces, virial=virial)


def build():
    """Python PES factory used by the companion input file."""
    return QTIP4PFMonomers()


def build_properties():
    """Return the ordinary q-TIP4P/F geometry-dependent dipole surface."""
    def evaluate(q):
        return {"dipole": molecular_dipoles(q).sum(axis=1)}
    return evaluate
