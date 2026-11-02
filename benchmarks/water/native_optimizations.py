"""Experimental native MTS/RPC maps; separate from the registered baseline.

Only the small-system deterministic map tests certify the implementation.
Neither contraction nor a larger outer time step is certified for diffusion.
"""

from __future__ import annotations

import math
import numpy as np

from benchmarks.water.native_reference import BOHR_NM, HARTREE_KJ_MOL, VELOCITY_NM_PS
from benchmarks.water.qtip4pf import H_MASS_DA, O_MASS_DA, PARAMETERS, make_system
from benchmarks.water.split_reference import contraction_matrix, split_system
from pyrpmd.constants import AMU_TO_AU, FS_TO_AU


def openmm_frequency_beta(temperature_K):
    """Atomic time beta matching OpenMM 8.6's documented source hbar value.

    This is a deterministic-map control, not a change of sampling temperature.
    """
    from scipy.constants import Avogadro
    from openmm import unit
    hbar = 1.054571628e-34 * Avogadro / (1000 * 1e-12)
    boltzmann = float(unit.MOLAR_GAS_CONSTANT_R / (unit.kilojoule_per_mole / unit.kelvin))
    return hbar * 1000 * FS_TO_AU / (boltzmann * temperature_K)


def physical_indices(molecules):
    return np.arange(4*molecules).reshape(molecules,4)[:,:3].ravel()


def extended_state(q,p,masses):
    n=len(masses)//3
    x,v=np.zeros((n,4,3)),np.zeros((n,4,3))
    x[:,:3]=np.asarray(q).reshape(n,3,3)*BOHR_NM
    x[:,3]=PARAMETERS["gamma"]*x[:,0]+(1-PARAMETERS["gamma"])*(x[:,1]+x[:,2])/2
    v[:,:3]=(np.asarray(p).reshape(-1,3)/masses[:,None]).reshape(n,3,3)*VELOCITY_NM_PS
    return x.reshape(-1,3),v.reshape(-1,3)


class DelayedSlowKickMTS:
    """Native classical rRESPA with the last slow half kick represented in v.

    Store v_engine = v_canonical - h F_slow(q)/(2m). Each custom step applies
    one full slow kick before the inner fast velocity-Verlet substeps. OpenMM
    refreshes virtual sites at the outer step end, before the next slow force.
    Fast intramolecular forces never depend on the virtual-site coordinates.
    """
    def __init__(self,*,molecules,box_angstrom,outer_dt_fs=.5,inner_dt_fs=.1,
                 cutoff_angstrom=9.,ewald_tolerance=1e-5,platform="Reference",threads=1):
        import openmm as mm
        from openmm import unit
        ratio=outer_dt_fs/inner_dt_fs
        if (not math.isfinite(ratio) or inner_dt_fs<=0 or ratio<1
                or not math.isclose(ratio,round(ratio),rel_tol=0,abs_tol=1e-12)):
            raise ValueError("outer time step must be a positive integer multiple of inner time step")
        if platform not in ("Reference","CPU"):
            raise ValueError("only Reference and CPU have been considered by this benchmark")
        self.mm,self.unit=mm,unit
        self.molecules,self.outer_dt_fs=molecules,outer_dt_fs
        self.masses=np.tile([O_MASS_DA,H_MASS_DA,H_MASS_DA],molecules)*AMU_TO_AU
        self.indices=physical_indices(molecules)
        self.system=make_system(molecules=molecules,box_angstrom=box_angstrom,
                                cutoff_angstrom=cutoff_angstrom,ewald_tolerance=ewald_tolerance)
        for force in self.system.getForces():
            force.setForceGroup(1 if isinstance(force,mm.NonbondedForce) else 0)
        self.integrator=mm.CustomIntegrator(outer_dt_fs/1000)
        self.integrator.addComputePerDof("v","v+dt*f1/m")
        count=round(ratio)
        for _ in range(count):
            self.integrator.addComputePerDof("v",f"v+0.5*dt*f0/({count}*m)")
            self.integrator.addComputePerDof("x",f"x+dt*v/{count}")
            self.integrator.addComputePerDof("v",f"v+0.5*dt*f0/({count}*m)")
        properties={"Threads":str(threads),"DeterministicForces":"true"} if platform=="CPU" else {}
        self.context=mm.Context(self.system,self.integrator,mm.Platform.getPlatformByName(platform),properties)

    def set_state(self,q,p):
        positions,velocities=extended_state(q,p,self.masses)
        self.context.setPositions(positions)
        self.context.computeVirtualSites()
        forces=np.asarray(self.context.getState(getForces=True,groups=2).getForces(asNumpy=True)
                          /(self.unit.kilojoule_per_mole/self.unit.nanometer))[self.indices]
        velocities[self.indices]-=.5*self.outer_dt_fs/1000*forces/(self.masses/AMU_TO_AU)[:,None]
        self.context.setVelocities(velocities)

    def get_state(self,*,energy=False):
        state=self.context.getState(getPositions=True,getVelocities=True,getForces=True,groups=2)
        q=np.asarray(state.getPositions(asNumpy=True)/self.unit.nanometer)[self.indices]/BOHR_NM
        velocity=np.asarray(state.getVelocities(asNumpy=True)/(self.unit.nanometer/self.unit.picosecond))[self.indices]
        force=np.asarray(state.getForces(asNumpy=True)/(self.unit.kilojoule_per_mole/self.unit.nanometer))[self.indices]
        velocity+=.5*self.outer_dt_fs/1000*force/(self.masses/AMU_TO_AU)[:,None]
        p=velocity/VELOCITY_NM_PS*self.masses[:,None]
        h=None
        if energy:
            potential=float(self.context.getState(getEnergy=True).getPotentialEnergy()/self.unit.kilojoule_per_mole)/HARTREE_KJ_MOL
            h=potential+float(np.sum(p*p/(2*self.masses[:,None])))
        return q[None],p[None],h

    def close(self):
        self.context=None;self.integrator=None


class NativeSplitRPMD:
    """Native OpenMM PILE/BAB propagation with explicitly grouped PME RPC.

    This does not perform MTS: every retained force is evaluated each dt. The
    smooth group includes the moved intramolecular Ewald exclusions. P and K
    comparisons remain necessary before assigning transport accuracy.
    """
    def __init__(self,*,molecules,box_angstrom,beads,long_copies,dt_fs=.1,
                 temperature_K=298.,sampling=False,seed=901,platform="Reference",threads=1,
                 lj_cutoff_angstrom=9.,electrostatic_cutoff_angstrom=5.,
                 lj_backend="custom",ewald_tolerance=1e-5):
        import openmm as mm
        from openmm import unit
        contraction_matrix(beads,long_copies)  # Reject undeclared even-K convention.
        self.mm,self.unit=mm,unit
        self.beads,self.molecules,self.dt_fs=beads,molecules,dt_fs
        self.masses=np.tile([O_MASS_DA,H_MASS_DA,H_MASS_DA],molecules)*AMU_TO_AU
        self.indices=physical_indices(molecules)
        self.system,self.model=split_system(molecules=molecules,box_angstrom=box_angstrom,
            lj_cutoff_angstrom=lj_cutoff_angstrom,electrostatic_cutoff_angstrom=electrostatic_cutoff_angstrom,
            ewald_tolerance=ewald_tolerance,lj_backend=lj_backend)
        self.integrator=mm.RPMDIntegrator(beads,temperature_K,10,dt_fs/1000,{2:long_copies})
        self.integrator.setApplyThermostat(sampling)
        self.integrator.setRandomNumberSeed(seed)
        properties={"Threads":str(threads),"DeterministicForces":"true"} if platform=="CPU" else {}
        self.context=mm.Context(self.system,self.integrator,mm.Platform.getPlatformByName(platform),properties)

    def set_state(self,q,p):
        q,p=np.asarray(q),np.asarray(p)
        if q.shape!=(self.beads,3*self.molecules,3) or p.shape!=q.shape:
            raise ValueError("q/p shape differs from declared ring and molecule count")
        for bead in range(self.beads):
            positions,velocities=extended_state(q[bead],p[bead],self.masses)
            self.integrator.setPositions(bead,positions)
            self.integrator.setVelocities(bead,velocities)

    def get_state(self):
        q,p=[],[]
        for bead in range(self.beads):
            state=self.integrator.getState(bead,getPositions=True,getVelocities=True)
            q.append(np.asarray(state.getPositions(asNumpy=True)/self.unit.nanometer)[self.indices]/BOHR_NM)
            velocity=np.asarray(state.getVelocities(asNumpy=True)/(self.unit.nanometer/self.unit.picosecond))[self.indices]
            p.append(velocity/VELOCITY_NM_PS*self.masses[:,None])
        return np.asarray(q),np.asarray(p)

    def close(self):
        self.context=None;self.integrator=None
