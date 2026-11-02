"""Experimental compact Coulomb separation from a primary 2009 RPC paper.

Fanourgakis, Markland and Manolopoulos, JCP 131, 094102 (2009), Eq. 35,
DOI 10.1063/1.3216520, defines the polynomial used here. This is not asserted
to be identical to the earlier 2008 split used by the q-TIP4P/F paper.

The full Coulomb PME model is kept intact on the slow group; a compact short
potential is subtracted there and added on the full ring. Consequently the
uncontracted sum is the original Hamiltonian. Contraction remains an explicit
approximation whose ensemble and transport error require separate tests.
"""

from __future__ import annotations

import math
import numpy as np

from benchmarks.water.native_optimizations import NativeSplitRPMD, physical_indices
from benchmarks.water.qtip4pf import H_MASS_DA, O_MASS_DA, PARAMETERS, make_system
from benchmarks.water.split_reference import contraction_matrix, coulomb_constant
from pyrpmd.constants import AMU_TO_AU


SOURCE_DOI = "10.1063/1.3216520"
SOURCE_URL = "https://www.researchgate.net/publication/26798566_A_fast_path_integral_method_for_polarizable_force_fields"


def splitting_function(r_over_sigma):
    """The dimensionless compact f(x), equal to one for x >= 1."""
    x = np.asarray(r_over_sigma, dtype=float)
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("distance/splitting-length must be finite and nonnegative")
    inside = np.minimum(x, 1.)
    return (7/4 * inside - 21/4 * inside**5 + 7 * inside**6 - 5/2 * inside**7)


def compact_system(*, molecules, box_angstrom, splitting_angstrom=5., cutoff_angstrom=9.,
                   ewald_tolerance=1e-5, hydrogen_mass_da=H_MASS_DA):
    """Group 0 intra, 1 compact Coulomb, 2 full Coulomb minus compact, 3 LJ.

    All charge-site pairs within one molecule remain excluded from both short
    corrections. Keeping the complete original PME force in group 2 includes
    its own intramolecular exclusion correction, self and reciprocal terms.
    The splitting length is independent of the Ewald real-space cutoff.
    """
    import openmm as mm
    from openmm import unit
    if (not math.isfinite(splitting_angstrom) or not math.isfinite(cutoff_angstrom)
            or not 0 < splitting_angstrom <= cutoff_angstrom < box_angstrom / 2):
        raise ValueError("require 0 < splitting length <= real-space cutoff < half box")
    system = make_system(molecules=molecules, box_angstrom=box_angstrom,
                         cutoff_angstrom=cutoff_angstrom, ewald_tolerance=ewald_tolerance,
                         hydrogen_mass_da=hydrogen_mass_da, dispersion_correction=False)
    electrostatics = next(force for force in system.getForces() if isinstance(force, mm.NonbondedForce))
    charges = []
    for index in range(system.getNumParticles()):
        charge, sigma, _ = electrostatics.getParticleParameters(index)
        charges.append(float(charge / unit.elementary_charge))
        electrostatics.setParticleParameters(index, charge, sigma, 0)
    electrostatics.setForceGroup(2)
    electrostatics.setReciprocalSpaceForceGroup(2)
    coefficient = coulomb_constant()
    charged = {index for index, charge in enumerate(charges) if charge != 0}
    excluded = []
    for index in range(electrostatics.getNumExceptions()):
        first, second, charge, _, epsilon = electrostatics.getExceptionParameters(index)
        if float(charge / unit.elementary_charge**2) != 0 or float(epsilon / unit.kilojoule_per_mole) != 0:
            raise ValueError("compact q-TIP4P/F split requires fully excluded intramolecular nonbonded pairs")
        excluded.append((first, second))
    for sign, group in ((1, 1), (-1, 2)):
        short = mm.CustomNonbondedForce(
            f"{sign}*rpc_coulomb*q1*q2*(1-f)/r;"
            "f=1.75*x-5.25*x^5+7*x^6-2.5*x^7;x=r/rpc_sigma")
        short.addGlobalParameter("rpc_coulomb", coefficient)
        short.addGlobalParameter("rpc_sigma", splitting_angstrom / 10)
        short.addPerParticleParameter("q")
        for charge in charges:
            short.addParticle([charge])
        short.addInteractionGroup(charged, charged)
        for first, second in excluded:
            short.addExclusion(first, second)
        short.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        short.setCutoffDistance(splitting_angstrom / 10)
        short.setUseLongRangeCorrection(False)
        short.setUseSwitchingFunction(False)
        short.setForceGroup(group)
        system.addForce(short)
    lj = mm.CustomNonbondedForce("4*rpc_epsilon*((rpc_lj_sigma/r)^12-(rpc_lj_sigma/r)^6)")
    lj.addGlobalParameter("rpc_epsilon", PARAMETERS["epsilon_kcal_mol"] * 4.184)
    lj.addGlobalParameter("rpc_lj_sigma", PARAMETERS["sigma_angstrom"] / 10)
    for _ in range(system.getNumParticles()):
        lj.addParticle([])
    oxygens = set(range(0, system.getNumParticles(), 4))
    lj.addInteractionGroup(oxygens, oxygens)
    for first, second in excluded:
        lj.addExclusion(first, second)
    lj.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    lj.setCutoffDistance(cutoff_angstrom / 10)
    lj.setUseSwitchingFunction(False)
    lj.setUseLongRangeCorrection(False)
    lj.setForceGroup(3)
    system.addForce(lj)
    return system, {
        "splitting_function": "f(x)=7*x/4-21*x^5/4+7*x^6-5*x^7/2 for 0<=x<=1; f=1 for x>=1",
        "splitting_source": {"doi": SOURCE_DOI, "equation": "35", "author_uploaded_paper": SOURCE_URL},
        "identical_to_2008_qtip_RPC_claimed": False,
        "splitting_length_angstrom": splitting_angstrom,
        "PME_real_space_cutoff_angstrom": cutoff_angstrom,
        "LJ_cutoff_angstrom": cutoff_angstrom,
        "ewald_tolerance": ewald_tolerance,
        "coulomb_coefficient_kJ_mol_nm_e2": coefficient,
        "short_definition": "(1-f(r/sigma))*qiqj/r, intermolecular pairs only",
        "long_definition": "complete original Coulomb PME minus the same compact short potential",
        "full_ring_limit": "same original Coulomb PME and oxygen LJ Hamiltonian",
        "tail_correction": False, "switching": False, "constraints": [],
    }


class NativeCompactRPMD(NativeSplitRPMD):
    """Single-step native RPMD with only the compact long group contracted.

    This experimental reference inherits explicit per-copy state transfers.
    It supplies no validated liquid ensemble, transport or checkpoint protocol.
    """

    def __init__(self, *, molecules, box_angstrom, beads, long_copies, dt_fs=.1,
                 temperature_K=298., sampling=False, seed=901, platform="Reference", threads=1,
                 splitting_angstrom=5., cutoff_angstrom=9., ewald_tolerance=1e-5):
        import openmm as mm
        from openmm import unit
        contraction_matrix(beads, long_copies)
        if (not math.isfinite(dt_fs) or dt_fs <= 0 or not math.isfinite(temperature_K)
                or temperature_K <= 0 or platform not in ("Reference", "CPU")
                or isinstance(threads, bool) or not isinstance(threads, int) or threads < 1):
            raise ValueError("invalid declared native compact RPMD settings")
        self.mm, self.unit = mm, unit
        self.beads, self.molecules, self.dt_fs = beads, molecules, dt_fs
        self.masses = np.tile([O_MASS_DA, H_MASS_DA, H_MASS_DA], molecules) * AMU_TO_AU
        self.indices = physical_indices(molecules)
        self.system, self.model = compact_system(
            molecules=molecules, box_angstrom=box_angstrom, splitting_angstrom=splitting_angstrom,
            cutoff_angstrom=cutoff_angstrom, ewald_tolerance=ewald_tolerance)
        self.integrator = mm.RPMDIntegrator(beads, temperature_K, 10, dt_fs / 1000, {2: long_copies})
        self.integrator.setApplyThermostat(sampling)
        self.integrator.setRandomNumberSeed(seed)
        properties = {"Threads": str(threads), "DeterministicForces": "true"} if platform == "CPU" else {}
        self.context = mm.Context(self.system, self.integrator, mm.Platform.getPlatformByName(platform), properties)
