"""Experimental q-TIP4P/F force splitting and reversible RPMD MTS reference.

No production certification is implied. The uncontracted split is checked
against the audited full model; contraction defines an approximate Hamiltonian
whose accuracy must be established separately. LJ retains its own 9 A cutoff.
Electrostatic intramolecular exclusion corrections travel with the smooth PME
part when it is contracted, so intramolecular exclusions remain consistent.
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from benchmarks.water.campaign import FixedWaterCache
from benchmarks.water.native_reference import BOHR_NM,HARTREE_KJ_MOL
from benchmarks.water.qtip4pf import PARAMETERS,O_MASS_DA,H_MASS_DA,make_system
from pyrpmd.constants import AMU_TO_AU,BOHR_ANGSTROM,FS_TO_AU,KB_AU,KCAL_MOL_TO_AU
from pyrpmd.dynamics import Dynamics
from pyrpmd.potentials.base import PESResult


def intramolecular(q):
    """Analytic independent quartic-stretch/angle energy and physical forces.

    Input is whole OHH molecules, [bead, 3*molecule, 3], in bohr. Returns one
    energy per bead in hartree and forces in hartree/bohr. Internally wrapped
    molecules are not permitted; the campaign propagates unwrapped coordinates.
    """
    q=np.asarray(q,dtype=float)
    if q.ndim!=3 or not q.shape[0] or not q.shape[1] or q.shape[1]%3 or q.shape[-1]!=3 or not np.isfinite(q).all():
        raise ValueError("q must be finite [bead,3*molecule,3] OHH coordinates")
    molecular=q.reshape(len(q),-1,3,3)
    bond=molecular[:,:,1:]-molecular[:,:,:1]
    distance=np.linalg.norm(bond,axis=-1)
    if np.any(distance<=0):
        raise ValueError("zero bond length")
    direction=bond/distance[:,:,:,None]
    p=PARAMETERS
    alpha=p["alpha_r_per_angstrom"]*BOHR_ANGSTROM
    stretch=alpha*(distance-p["r_eq_angstrom"]/BOHR_ANGSTROM)
    D=p["D_r_kcal_mol"]*KCAL_MOL_TO_AU
    energy=D*(stretch**2-stretch**3+7/12*stretch**4).sum(axis=(1,2))
    derivative=D*alpha*(2*stretch-3*stretch**2+7/3*stretch**3)
    hydrogen_force=-derivative[:,:,:,None]*direction
    cosine=np.clip(np.sum(direction[:,:,0]*direction[:,:,1],axis=-1),-1,1)
    sine=np.sqrt(1-cosine**2)
    if np.any(sine<1e-8):
        raise ValueError("singular collinear water angle")
    delta=np.arccos(cosine)-np.deg2rad(p["theta_eq_degrees"])
    k=p["k_theta_kcal_mol_rad2"]*KCAL_MOL_TO_AU
    energy+=.5*k*np.sum(delta**2,axis=1)
    coefficient=k*delta/sine
    hydrogen_force[:,:,0]+=coefficient[:,:,None]*(direction[:,:,1]-cosine[:,:,None]*direction[:,:,0])/distance[:,:,0,None]
    hydrogen_force[:,:,1]+=coefficient[:,:,None]*(direction[:,:,0]-cosine[:,:,None]*direction[:,:,1])/distance[:,:,1,None]
    force=np.empty_like(molecular)
    force[:,:,1:]=hydrogen_force
    force[:,:,0]=-hydrogen_force.sum(axis=2)
    return energy,force.reshape(q.shape)


class IntramolecularPotential:
    def evaluate(self,q):
        energy,force=intramolecular(q)
        return PESResult(energy,force,np.zeros(len(q),dtype=int))


def contraction_matrix(beads,copies):
    """Real normal-mode contraction; odd contracted counts avoid a Nyquist choice."""
    if (isinstance(beads,bool) or isinstance(copies,bool) or not isinstance(beads,int)
            or not isinstance(copies,int) or not 1<=copies<=beads):
        raise ValueError("require integer 1 <= copies <= beads")
    if copies==beads:
        return np.eye(beads)
    if copies%2==0:
        raise ValueError("use odd contracted copy counts or the full ring")
    def modes(size):
        phase=np.arange(size)*2*np.pi/size
        rows=[np.full(size,1/np.sqrt(size))]
        for mode in range(1,(copies+1)//2):
            rows.extend([np.sqrt(2/size)*np.cos(mode*phase),np.sqrt(2/size)*np.sin(mode*phase)])
        return np.asarray(rows)
    return np.sqrt(copies/beads)*modes(copies).T@modes(beads)


def coulomb_constant():
    """Measure the engine's exact Coulomb coefficient with a unit-charge pair."""
    import openmm as mm
    from openmm import unit
    system=mm.System();system.addParticle(1);system.addParticle(1)
    force=mm.NonbondedForce();force.addParticle(1,1,0);force.addParticle(1,1,0)
    system.addForce(force)
    integrator=mm.VerletIntegrator(.001)
    context=mm.Context(system,integrator,mm.Platform.getPlatformByName("Reference"))
    try:
        context.setPositions([[0,0,0],[1,0,0]])
        return float(context.getState(getEnergy=True).getPotentialEnergy()/unit.kilojoule_per_mole)
    finally:
        del context,integrator


def split_system(*,molecules,box_angstrom,lj_cutoff_angstrom=9.,electrostatic_cutoff_angstrom=5.,
                 ewald_tolerance=1e-5,hydrogen_mass_da=H_MASS_DA,move_exclusions=True,
                 lj_backend="nonbonded"):
    """Separate identical LJ and Coulomb terms; move smooth intra exclusions.

    Groups: 0 intramolecular, 1 screened direct Coulomb, 2 smooth reciprocal
    Coulomb with intramolecular exclusions, 3 oxygen LJ. At full bead count,
    moving exclusions leaves the total potential and force unchanged.
    """
    import openmm as mm
    from openmm import unit
    if not 0<electrostatic_cutoff_angstrom<box_angstrom/2:
        raise ValueError("electrostatic cutoff must be positive and below half the box")
    if lj_backend not in ("nonbonded","custom"):
        raise ValueError("lj_backend must be nonbonded or custom")
    system=make_system(molecules=molecules,box_angstrom=box_angstrom,
                       cutoff_angstrom=lj_cutoff_angstrom,ewald_tolerance=ewald_tolerance,
                       hydrogen_mass_da=hydrogen_mass_da,dispersion_correction=False)
    electrostatics=next(force for force in system.getForces() if isinstance(force,mm.NonbondedForce))
    charges=[];lj_parameters=[]
    for particle in range(system.getNumParticles()):
        charge,sigma,epsilon=electrostatics.getParticleParameters(particle)
        charges.append(float(charge/unit.elementary_charge))
        lj_parameters.append((sigma,epsilon))
        electrostatics.setParticleParameters(particle,charge,sigma,0)
    electrostatics.setCutoffDistance(electrostatic_cutoff_angstrom/10)
    electrostatics.setForceGroup(1);electrostatics.setReciprocalSpaceForceGroup(2)
    if lj_backend=="custom":
        lj=mm.CustomNonbondedForce("4*eps_qtip*((sigma_qtip/r)^12-(sigma_qtip/r)^6)")
        lj.addGlobalParameter("eps_qtip",PARAMETERS["epsilon_kcal_mol"]*4.184)
        lj.addGlobalParameter("sigma_qtip",PARAMETERS["sigma_angstrom"]/10)
        for particle in range(system.getNumParticles()):
            lj.addParticle([])
        oxygens=set(range(0,system.getNumParticles(),4))
        lj.addInteractionGroup(oxygens,oxygens)
        lj.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        lj.setUseLongRangeCorrection(False)
    else:
        # A separate optimized, charge-free NonbondedForce is identical to the
        # oxygen interaction group and supports an independent 9 A LJ cutoff.
        lj=mm.NonbondedForce()
        for sigma,epsilon in lj_parameters:
            lj.addParticle(0,sigma,epsilon)
        lj.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
        lj.setUseDispersionCorrection(False)
    for exception in range(electrostatics.getNumExceptions()):
        first,second,charge,sigma,epsilon=electrostatics.getExceptionParameters(exception)
        if lj_backend=="custom":lj.addExclusion(first,second)
        else:lj.addException(first,second,0,sigma,0)
    lj.setCutoffDistance(lj_cutoff_angstrom/10)
    lj.setUseSwitchingFunction(False);lj.setForceGroup(3)
    system.addForce(lj)
    # Query the actual selected alpha; do not infer a backend's rounding or mesh
    # choice from its user-requested error tolerance.
    integrator=mm.VerletIntegrator(.001)
    context=mm.Context(system,integrator,mm.Platform.getPlatformByName("Reference"))
    try:
        alpha,nx,ny,nz=electrostatics.getPMEParametersInContext(context)
        alpha=float(alpha/(unit.nanometer**-1)) if hasattr(alpha,"unit") else float(alpha)
    finally:
        del context,integrator
    coefficient=coulomb_constant()
    if move_exclusions:
        for sign,group in ((1,1),(-1,2)):
            correction=mm.CustomBondForce(f"{sign}*coulomb_qtip*qq*erf(alpha_qtip*r)/r")
            correction.addGlobalParameter("coulomb_qtip",coefficient)
            correction.addGlobalParameter("alpha_qtip",alpha)
            correction.addPerBondParameter("qq")
            correction.setUsesPeriodicBoundaryConditions(True);correction.setForceGroup(group)
            for exception in range(electrostatics.getNumExceptions()):
                first,second,charge,sigma,epsilon=electrostatics.getExceptionParameters(exception)
                if float(charge/unit.elementary_charge**2)!=0 or float(epsilon/unit.kilojoule_per_mole)!=0:
                    raise ValueError("the audited q-TIP4P/F model requires fully excluded intramolecular nonbonded pairs")
                product=charges[first]*charges[second]
                if product:
                    correction.addBond(first,second,[product])
            system.addForce(correction)
    return system,{"lj_cutoff_angstrom":lj_cutoff_angstrom,"electrostatic_cutoff_angstrom":electrostatic_cutoff_angstrom,
                   "ewald_tolerance":ewald_tolerance,"alpha_per_nm":alpha,
                   "reference_mesh":[nx,ny,nz],"coulomb_coefficient_kJ_mol_nm_e2":coefficient,
                   "move_intramolecular_smooth_exclusions":move_exclusions,
                   "lj_backend":lj_backend,
                   "constraints":[],"tail_correction":False,"switching":False}


class GroupEvaluator:
    """Explicitly refresh virtual sites before every group force evaluation."""
    def __init__(self,system,platform="Reference",threads=1):
        import openmm as mm
        from openmm import unit
        self.mm,self.unit=mm,unit
        self.system=mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
        self.indices=np.asarray([i for i in range(system.getNumParticles()) if not system.isVirtualSite(i)])
        self.integrator=mm.VerletIntegrator(.001)
        properties={"Threads":str(threads),"DeterministicForces":"true"} if platform=="CPU" else {}
        self.context=mm.Context(self.system,self.integrator,mm.Platform.getPlatformByName(platform),properties)
        self.calls=0

    def evaluate(self,q,groups=-1):
        q=np.asarray(q,dtype=float)
        if q.ndim!=3 or q.shape[1:]!=(len(self.indices),3) or not np.isfinite(q).all():
            raise ValueError("group force input must be finite physical OHH bead coordinates")
        energies,forces=[],[]
        positions=np.zeros((self.system.getNumParticles(),3))
        for frame in q:
            positions[self.indices]=frame*BOHR_NM
            self.context.setPositions(positions);self.context.computeVirtualSites()
            state=self.context.getState(getEnergy=True,getForces=True,groups=groups)
            energies.append(float(state.getPotentialEnergy()/self.unit.kilojoule_per_mole)/HARTREE_KJ_MOL)
            values=np.asarray(state.getForces(asNumpy=True)/(self.unit.kilojoule_per_mole/self.unit.nanometer))
            forces.append(values[self.indices]*BOHR_NM/HARTREE_KJ_MOL)
            self.calls+=1
        return np.asarray(energies),np.asarray(forces)

    def close(self):
        self.context=None;self.integrator=None


class ContractedSlowPotential:
    """Coupled-bead slow Hamiltonian; never present it as independent PES batches."""
    def __init__(self,evaluator,beads,long_copies):
        self.evaluator=evaluator
        self.beads,self.long_copies=beads,long_copies
        self.transform=contraction_matrix(beads,long_copies)

    def evaluate(self,q):
        q=np.asarray(q)
        if q.shape[0]!=self.beads:
            raise ValueError("contraction belongs to one fixed bead count")
        if self.long_copies==self.beads:
            energy,force=self.evaluator.evaluate(q,groups=(1<<1)|(1<<2)|(1<<3))
            return float(energy.sum()),force
        short_energy,short_force=self.evaluator.evaluate(q,groups=(1<<1)|(1<<3))
        contracted=np.einsum("kp,pac->kac",self.transform,q)
        long_energy,long_force=self.evaluator.evaluate(contracted,groups=1<<2)
        mapped=(self.beads/self.long_copies)*np.einsum("kp,kac->pac",self.transform,long_force)
        energy=float(short_energy.sum()+self.beads/self.long_copies*long_energy.sum())
        return energy,short_force+mapped


class SplitMTS:
    """Reversible slow-half / fast-BAB-substeps / slow-half map for this audit."""
    def __init__(self,slow,masses,temperature_K,beads,outer_dt_fs=.5,inner_dt_fs=.1,seed=901):
        ratio=outer_dt_fs/inner_dt_fs
        if inner_dt_fs<=0 or ratio<1 or not np.isclose(ratio,round(ratio),atol=1e-12,rtol=0):
            raise ValueError("positive outer step must be an integer multiple of inner step")
        self.slow=slow;self.substeps=round(ratio);self.dt=outer_dt_fs*FS_TO_AU
        self.fast=Dynamics(FixedWaterCache(IntramolecularPotential()),masses,1/(KB_AU*temperature_K),
                           beads,inner_dt_fs*FS_TO_AU,seed=seed,native=True,threads=1,integrator="normal-mode-bab")
        self._cached_q=self._cached_slow=None

    def slow_value(self,q):
        if self._cached_q is not None and np.array_equal(q,self._cached_q):
            return self._cached_slow
        value=self.slow.evaluate(q)
        force=value[1].copy();force.flags.writeable=False
        self._cached_q=q.copy();self._cached_slow=(value[0],force)
        return self._cached_slow

    def step(self,state,*,sampling=False,centroid_friction_fs=100):
        q,p,step,clock=state.q.copy(),state.p.copy(),state.step,state.time
        rng=copy.deepcopy(self.fast.rng.bit_generator.state)
        try:
            state.p+=.5*self.dt*self.slow_value(state.q)[1]
            for _ in range(self.substeps):
                if sampling:
                    self.fast.sample_step(state,gamma_centroid=1/(centroid_friction_fs*FS_TO_AU))
                else:
                    self.fast.step(state)
            state.p+=.5*self.dt*self.slow_value(state.q)[1]
            if not np.isfinite(state.q).all() or not np.isfinite(state.p).all():
                raise FloatingPointError("nonfinite split-MTS state")
        except BaseException:
            state.q[:],state.p[:]=q,p;state.step,state.time=step,clock
            self.fast.rng.bit_generator.state=rng
            self._cached_q=self._cached_slow=None
            raise
        return state

    def energy(self,state):
        return self.fast.energy(state)+self.slow_value(state.q)[0]
