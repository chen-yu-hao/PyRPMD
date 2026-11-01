"""q-TIP4P/F from Habershon et al., JCP 131, 024501 (2009).

Independent implementation of Eqs. (1)--(4) and Table I, not copied from a
third-party force-field implementation. Physical coordinates are O,H,H per
molecule in atomic units. OpenMM supplies the virtual M sites and PME; PyRPMD
propagates only the three physical nuclei. This is a flexible model: adding
rigid bond constraints changes the model and removes its stretching spectrum.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from pyrpmd.constants import ANGSTROM_TO_AU, BOHR_ANGSTROM, KCAL_MOL_TO_AU


SOURCE = {
    "doi": "10.1063/1.3167790",
    "url": "https://arxiv.org/abs/1011.1047",
    "equations": [1, 2, 3, 4],
    "parameter_table": "I",
    "model": "q-TIP4P/F",
}
PARAMETERS = {
    "epsilon_kcal_mol": 0.1852,
    "sigma_angstrom": 3.1589,
    "q_M_e": 1.1128,
    "gamma": 0.73612,
    "D_r_kcal_mol": 116.09,
    "alpha_r_per_angstrom": 2.287,
    "r_eq_angstrom": 0.9419,
    "k_theta_kcal_mol_rad2": 87.85,
    "theta_eq_degrees": 107.4,
}
O_MASS_DA = 15.99491461957
H_MASS_DA = 1.00782503223
D_MASS_DA = 2.01410177812


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _count(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("molecules must be a positive integer")
    return value


def _coordinates(q):
    q = np.asarray(q)
    if (q.dtype.kind not in "fiu" or q.ndim != 3 or q.shape[-1] != 3
            or q.shape[1] == 0 or q.shape[1] % 3 or q.shape[0] == 0
            or not np.isfinite(q).all()):
        raise ValueError("q must be finite real [batch,3*molecules,3] in OHH order")
    return np.asarray(q, dtype=float).reshape(q.shape[0], -1, 3, 3)


def dipoles(q):
    """Total neutral-cell dipole [batch,3], e*bohr, for whole OHH molecules.

    Molecules must be internally unwrapped. Translating any whole molecule by
    a lattice vector leaves its dipole unchanged. Do not wrap each atom alone.
    """
    q = _coordinates(q)
    p = PARAMETERS
    # Subtract O first to avoid cancelling large absolute unwrapped positions.
    oh_sum = q[:, :, 1] + q[:, :, 2] - 2 * q[:, :, 0]
    return (0.5 * p["q_M_e"] * p["gamma"] * oh_sum).sum(axis=1)


def cluster_energy(q):
    """Independent NumPy, all-pairs nonperiodic energy oracle in hartree.

    No cutoffs, Ewald, or long-range correction; intended for force/virtual-site
    validation of isolated clusters, never as a periodic liquid potential.
    """
    q = _coordinates(q) * BOHR_ANGSTROM
    p = PARAMETERS
    oh = q[:, :, 1:] - q[:, :, :1]
    lengths = np.linalg.norm(oh, axis=-1)
    if np.any(lengths == 0):
        raise ValueError("zero O-H distance")
    cosine = np.sum(oh[:, :, 0] * oh[:, :, 1], axis=-1) / lengths.prod(axis=-1)
    theta = np.arccos(np.clip(cosine, -1.0, 1.0))
    x = p["alpha_r_per_angstrom"] * (lengths - p["r_eq_angstrom"])
    bond = p["D_r_kcal_mol"] * (x**2 - x**3 + 7.0 / 12.0 * x**4)
    angle = 0.5 * p["k_theta_kcal_mol_rad2"] * (
        theta - np.deg2rad(p["theta_eq_degrees"]))**2
    energy = (bond.sum(axis=(1, 2)) + angle.sum(axis=1)) * KCAL_MOL_TO_AU
    sites = np.empty_like(q)
    sites[:, :, 0] = (p["gamma"] * q[:, :, 0]
                      + (1 - p["gamma"]) * (q[:, :, 1] + q[:, :, 2]) / 2)
    sites[:, :, 1:] = q[:, :, 1:]
    charge = p["q_M_e"] * np.array([-1.0, 0.5, 0.5])
    for i, j in itertools.combinations(range(q.shape[1]), 2):
        r_oo = np.linalg.norm(q[:, i, 0] - q[:, j, 0], axis=-1)
        r = np.linalg.norm(sites[:, i, :, None] - sites[:, j, None, :], axis=-1)
        if np.any(r_oo == 0) or np.any(r == 0):
            raise ValueError("coincident intermolecular interaction sites")
        s6 = (p["sigma_angstrom"] / r_oo)**6
        energy += 4 * p["epsilon_kcal_mol"] * (s6**2 - s6) * KCAL_MOL_TO_AU
        energy += np.sum(charge[:, None] * charge[None, :] / (r * ANGSTROM_TO_AU), axis=(1, 2))
    return energy


def make_system(*, molecules, box_angstrom=None, cutoff_angstrom=9.0,
                ewald_tolerance=1e-6, dispersion_correction=False,
                hydrogen_mass_da=H_MASS_DA):
    """Return an OpenMM System with optional periodic PME and no constraints.

    ``box_angstrom`` is a cubic box side. No tail correction or switching is
    used by default. Match cutoff, tail and electrostatics conventions before
    comparing a periodic result with a reference. OpenMM particle order is
    O,H,H,M; its nonvirtual subsequence is exactly the PyRPMD O,H,H order.
    """
    molecules = _count(molecules)
    hydrogen_mass_da = _positive(hydrogen_mass_da, "hydrogen_mass_da")
    cutoff_angstrom = _positive(cutoff_angstrom, "cutoff_angstrom")
    ewald_tolerance = _positive(ewald_tolerance, "ewald_tolerance")
    if ewald_tolerance >= 1:
        raise ValueError("ewald_tolerance must be less than one")
    if not isinstance(dispersion_correction, bool):
        raise ValueError("dispersion_correction must be boolean")
    if box_angstrom is not None:
        box_angstrom = _positive(box_angstrom, "box_angstrom")
        if 2 * cutoff_angstrom >= box_angstrom:
            raise ValueError("cutoff must be strictly smaller than half the cubic box")
    import openmm as mm

    p = PARAMETERS
    system = mm.System()
    bonds = mm.CustomBondForce(
        "D*(x*x-x*x*x+(7.0/12.0)*x*x*x*x); x=alpha*(r-req)")
    for name, value in (("D", p["D_r_kcal_mol"] * 4.184),
                        ("alpha", p["alpha_r_per_angstrom"] * 10),
                        ("req", p["r_eq_angstrom"] / 10)):
        bonds.addGlobalParameter(name, value)
    angles = mm.HarmonicAngleForce()
    nonbonded = mm.NonbondedForce()
    nonbonded.setUseDispersionCorrection(dispersion_correction)
    nonbonded.setUseSwitchingFunction(False)
    if box_angstrom is not None:
        box_nm = box_angstrom / 10
        system.setDefaultPeriodicBoxVectors(mm.Vec3(box_nm, 0, 0),
                                            mm.Vec3(0, box_nm, 0),
                                            mm.Vec3(0, 0, box_nm))
        nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
        nonbonded.setCutoffDistance(cutoff_angstrom / 10)
        nonbonded.setEwaldErrorTolerance(ewald_tolerance)
        bonds.setUsesPeriodicBoundaryConditions(True)
        angles.setUsesPeriodicBoundaryConditions(True)
    else:
        nonbonded.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    for molecule in range(molecules):
        o, h1, h2, m = range(4 * molecule, 4 * molecule + 4)
        for mass in (O_MASS_DA, hydrogen_mass_da, hydrogen_mass_da, 0):
            system.addParticle(mass)
        system.setVirtualSite(m, mm.ThreeParticleAverageSite(
            o, h1, h2, p["gamma"], (1 - p["gamma"]) / 2, (1 - p["gamma"]) / 2))
        for h in (h1, h2):
            bonds.addBond(o, h, [])
        angles.addAngle(h1, o, h2, math.radians(p["theta_eq_degrees"]),
                        p["k_theta_kcal_mol_rad2"] * 4.184)
        for charge, epsilon in ((0, p["epsilon_kcal_mol"] * 4.184),
                                (p["q_M_e"] / 2, 0),
                                (p["q_M_e"] / 2, 0), (-p["q_M_e"], 0)):
            nonbonded.addParticle(charge, p["sigma_angstrom"] / 10, epsilon)
        for a, b in itertools.combinations((o, h1, h2, m), 2):
            nonbonded.addException(a, b, 0, 1, 0)
    system.addForce(bonds)
    system.addForce(angles)
    system.addForce(nonbonded)
    return system


def initial_grid(*, cells_per_axis=2, density_g_cm3=0.997, seed=541,
                 hydrogen_mass_da=H_MASS_DA):
    """Return unrelaxed random-oriented cubic OHH geometry (bohr) and box (A).

    This reproducible starting grid is not equilibrated liquid water. Sampling
    and liquid-structure checks must precede any physical time correlation.
    """
    from scipy.spatial.transform import Rotation
    from scipy.constants import Avogadro

    cells_per_axis = _count(cells_per_axis)
    density_g_cm3 = _positive(density_g_cm3, "density_g_cm3")
    hydrogen_mass_da = _positive(hydrogen_mass_da, "hydrogen_mass_da")
    molecules = cells_per_axis**3
    volume_a3 = molecules * (O_MASS_DA + 2 * hydrogen_mass_da) / Avogadro / density_g_cm3 * 1e24
    box = volume_a3**(1 / 3)
    theta = math.radians(PARAMETERS["theta_eq_degrees"])
    monomer = np.array([[0, 0, 0], [1, 0, 0], [math.cos(theta), math.sin(theta), 0]])
    monomer *= PARAMETERS["r_eq_angstrom"]
    rng = np.random.default_rng(seed)
    atoms = []
    for index in itertools.product(range(cells_per_axis), repeat=3):
        origin = (np.asarray(index) + .5) * box / cells_per_axis
        atoms.append(Rotation.random(random_state=rng).apply(monomer) + origin)
    return np.concatenate(atoms) * ANGSTROM_TO_AU, box
