"""Matched dynamical-defect correction on a shared RPMDrate dividing surface.

Experimental model approximation: k_hybrid = k_QTST,full * (kappa_RP,full
+ kappa_Q,active - kappa_RP,active). It is NOT an exact identity for an
arbitrarily coupled active/bath Hamiltonian. A common scalar coordinate alone
is insufficient: equilibrium/reference flux, surface and units must match.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np
from scipy.linalg import eigh
from .rpmdrate_shared import _real, _positive, _digest


def _readonly(a):
    a=np.asarray(a);return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


def _times(time):
    t=_real(time,'time')
    if t.ndim!=1 or len(t)<2 or t[0]<0 or np.any(np.diff(t)<=0):
        raise ValueError('Increasing nonnegative time grid in atomic units required')
    return t


@dataclass(frozen=True)
class SharedRateContext:
    coordinate_id: str
    surface_id: str
    beta_au: float
    pes_identity: str
    reactant_normalization: str
    baseline_digest: str
    kinetic_mapping: str
    xi_star: float = 1.
    electronic_weight: float = 1.
    symmetry_factor: float = 1.
    time_unit: str = 'atomic_time'
    rate_unit: str = 'cm3 molecule^-1 s^-1'

    def __post_init__(self):
        for name in ('coordinate_id','surface_id','pes_identity','reactant_normalization','baseline_digest','kinetic_mapping'):
            if not isinstance(getattr(self,name),str) or not getattr(self,name).strip():
                raise ValueError(f'Explicit nonempty {name} required')
        for name in ('beta_au','electronic_weight','symmetry_factor'):_positive(getattr(self,name),name)
        xi=_real(self.xi_star,'xi_star')
        if xi.ndim!=0:raise ValueError('Scalar xi_star required')
        if self.time_unit!='atomic_time' or self.rate_unit!='cm3 molecule^-1 s^-1':
            raise ValueError('This workflow uses native gas-phase time/rate units only')

    @property
    def identity(self):return _digest(vars(self))

    def as_dict(self):return {**vars(self),'identity':self.identity}


@dataclass(frozen=True)
class MatchedFluxPair:
    """Independent matched active-subsystem block integrals, not child trajectories.

    numerator arrays [independent_block,time] must share the SAME equilibrium
    normalization. reference_flux[block] is the matched active RPMD positive
    flux; the SAME denominator normalizes both quantum and active RPMD curves.
    Block identities cannot be individual beads or correlated child pairs.
    No assumption of positive quantum flux or kappa<=1 is imposed.
    """
    context: SharedRateContext
    time_au: np.ndarray
    quantum_numerator: np.ndarray
    rpmd_numerator: np.ndarray
    reference_flux: np.ndarray
    independent_block_ids: tuple
    active_hamiltonian_id: str
    reference_flux_id: str
    equilibrium_normalization_id: str
    evidence: dict

    def __post_init__(self):
        if not isinstance(self.context,SharedRateContext):raise TypeError('SharedRateContext required')
        t=_times(self.time_au);q=_real(self.quantum_numerator,'quantum numerator');r=_real(self.rpmd_numerator,'RPMD numerator');d=_real(self.reference_flux,'reference flux')
        if q.ndim!=2 or q.shape!=r.shape or q.shape[1]!=len(t) or len(q)<2 or d.shape!=(len(q),):
            raise ValueError('Matched [independent blocks,time] numerators and block denominators required')
        if np.any(d<0) or d.sum()<=0 or np.any((d==0)&(np.any(q!=0,axis=1)|np.any(r!=0,axis=1))):
            raise ValueError('Invalid or unsupported reference flux')
        ids=tuple(self.independent_block_ids)
        if len(ids)!=len(q) or len(set(ids))!=len(q) or any(not isinstance(x,str) or not x for x in ids):
            raise ValueError('Unique explicit independent block IDs required')
        for name in ('active_hamiltonian_id','reference_flux_id','equilibrium_normalization_id'):
            if not isinstance(getattr(self,name),str) or not getattr(self,name).strip():raise ValueError('Missing matched active identity')
        if not isinstance(self.evidence,dict):raise ValueError('Evidence mapping required')
        for name,array in [('time_au',t),('quantum_numerator',q),('rpmd_numerator',r),('reference_flux',d)]:
            object.__setattr__(self,name,_readonly(array))
        object.__setattr__(self,'independent_block_ids',ids)
        # Make a private serialization copy so source dictionaries cannot change.
        object.__setattr__(self,'evidence',json.loads(json.dumps(self.evidence,allow_nan=False)))

    def curves(self):
        return (self.quantum_numerator.sum(0)/self.reference_flux.sum(),
                self.rpmd_numerator.sum(0)/self.reference_flux.sum())


def kubo_flux_side(hamiltonian, side_operator, beta_au, time_au, *, max_dimension=2048):
    """Exact finite-basis canonical Kubo flux-side kernel in an orthonormal basis.

    F = i[H,h]/hbar with hbar=1. Returns Cfs/Z and Cfs/Qr, NOT an RPMDrate
    kappa, a scattering transmission probability or a volumetric rate.
    A finite closed basis has recurrences; no irreversible plateau is presumed.
    For a FE solver, whiten its mass matrix AND transform its side operator
    before calling. Diagonalizing a PMF with a guessed mass is not supported.
    """
    H=np.asarray(hamiltonian,dtype=complex);h=np.asarray(side_operator,dtype=complex)
    if H.ndim!=2 or H.shape[0]!=H.shape[1] or h.shape!=H.shape or not 2<=len(H)<=max_dimension:
        raise ValueError('Matching square Hamiltonian and side operator within memory budget required')
    if not np.isfinite(H).all() or not np.isfinite(h).all() or not np.allclose(H,H.conj().T,atol=1e-12,rtol=1e-10) or not np.allclose(h,h.conj().T,atol=1e-12,rtol=1e-10):
        raise ValueError('Hermitian finite orthonormal-basis operators required')
    if not np.allclose(h@h,h,atol=1e-10,rtol=1e-10):raise ValueError('Side operator must be a projector')
    beta=_positive(beta_au,'beta');t=_times(time_au)
    E,U=eigh(H,check_finite=False);p=np.exp(-beta*(E-E[0]));p/=p.sum();he=U.conj().T@h@U
    i,j=np.triu_indices(len(E),1);gap=E[j]-E[i]
    coeff=2*(p[i]-p[j])*np.abs(he[i,j])**2/beta
    # Bound transient trigonometric work arrays even for large time/basis grids.
    kernel=np.zeros(len(t))
    chunk=2048
    for first in range(0,len(gap),chunk):
        take=slice(first,first+chunk)
        for start in range(0,len(t),256):
            ts=slice(start,start+256)
            kernel[ts]+=np.sin(np.outer(t[ts],gap[take]))@coeff[take]
    reactant=float(1-np.dot(p,he.diagonal().real))
    if reactant<=1e-12:raise ValueError('Unresolved reactant canonical population')
    F=1j*(H@h-h@H)
    return {'time_au':t,'canonical_flux_side':kernel,'reactant_population':reactant,
            'rate_curve_per_atomic_time':kernel/reactant,'flux_operator':F,
            'ground_energy_hartree':float(E[0]),'log_partition':float(-beta*E[0]+np.log(np.exp(-beta*(E-E[0])).sum())),
            'closed_finite_basis':True,'physical_converged':False,
            'is_rpmd_transmission_coefficient':False}


def shared_side_operator(shared, geometries):
    """Projector built from the native surface, not a separately chosen bond sign."""
    x=_real(geometries,'quantum grid geometries')
    if x.ndim!=3 or x.shape[1:]!=(len(shared.masses),3):raise ValueError('Grid geometries must be [basis,atoms,3]')
    sides=[shared.evaluate(q,require_smooth=True).surface>0 for q in x]
    return np.diag(np.asarray(sides,float))


def combine_shared_kappa(context, native_summary, time_au, full_rpmd_curve, pair, *,
                         plateau_start, relative_plateau_tolerance=.05,
                         bootstrap_samples=400, seed=713):
    """Keep native QTST; transfer a matched active dynamical defect only.

    The finite-time estimate remains visible even if a plateau/evidence check
    fails. `rate_cm3_s` is never promoted to a physically converged rate here.
    Bootstrap resamples whole active blocks jointly for Q and RP. Its reported
    error is conditional on native QTST and full-RPMD curve, not total error.
    """
    if not isinstance(context,SharedRateContext) or not isinstance(pair,MatchedFluxPair):
        raise TypeError('Shared context and matched active flux pair required')
    if context.identity!=pair.context.identity:raise ValueError('Different coordinate, dividing surface, temperature, PES or normalization')
    if _digest(native_summary)!=context.baseline_digest:raise ValueError('Native rate summary changed after binding')
    t=_times(time_au);base=_real(full_rpmd_curve,'native recrossing curve')
    if base.shape!=t.shape or not np.array_equal(t,pair.time_au):raise ValueError('Time grids must match exactly; no implicit interpolation')
    if native_summary.get('task')!='rate':raise ValueError('Only native bimolecular rate results are supported')
    pmf=native_summary.get('pmf',{});qtst=_positive(pmf.get('k_qtst_cm3_s'),'native QTST factor')
    if pmf.get('xi_star') is None or native_summary.get('kappa') is None:raise ValueError('PMF/recrossing result is incomplete')
    if type(plateau_start) is not int or not 0<=plateau_start<=len(t)-10:raise ValueError('At least ten plateau time samples required')
    if type(bootstrap_samples) is not int or bootstrap_samples<40:raise ValueError('Need at least 40 bootstrap draws')
    tol=_positive(relative_plateau_tolerance,'plateau tolerance')
    if not np.isclose(float(pmf['xi_star']),context.xi_star,atol=1e-12,rtol=0):
        raise ValueError('PMF-selected surface differs from the bound context')
    native_kappa=float(native_summary['kappa'])
    if not np.isfinite(native_kappa):raise ValueError('Finite native transmission required')
    if not np.isclose(base[plateau_start:].mean(),native_kappa,atol=1e-10,rtol=1e-7):
        raise ValueError('Chosen time tail is not the native reported plateau')
    q,r=pair.curves();delta=q-r;corrected=base+delta;tail=corrected[plateau_start:]
    rng=np.random.default_rng(seed);boots=[];defects=[];B=len(pair.reference_flux)
    for _ in range(bootstrap_samples):
        ix=rng.integers(B,size=B);den=pair.reference_flux[ix].sum()
        if den<=0:raise ValueError('Bootstrap has unsupported zero-flux blocks; no conditional discard')
        defects.append((pair.quantum_numerator[ix]-pair.rpmd_numerator[ix]).sum(0)/den)
        boots.append(float((base+defects[-1])[plateau_start:].mean()))
    defects=np.asarray(defects);boots=np.asarray(boots);kappa=float(tail.mean())
    centered=tail-kappa;bt=base[None,plateau_start:]+defects[:,plateau_start:]
    envelope=float(np.quantile(np.max(abs(bt-bt.mean(1,keepdims=True)-centered[None]),axis=1),.95))
    bound=float(np.max(abs(centered))+envelope)
    required=('same_active_hamiltonian','same_reactant_normalization','same_positive_reference_flux',
              'quantum_kinetic_and_measure','environment_coupling','independent_block_sampling')
    missing=[k for k in required if not isinstance(pair.evidence.get(k),dict) or pair.evidence[k].get('passed') is not True or not pair.evidence[k].get('evidence')]
    reasons=[]
    if native_summary.get('status')!='plateau_estimate':reasons.append('native_RPMD_not_ready')
    if kappa<=0:reasons.append('nonpositive_corrected_kappa')
    if bound>tol*abs(kappa):reasons.append('corrected_curve_not_flat_on_native_tail')
    reasons.extend('missing_'+k for k in missing)
    rate=qtst*kappa
    return {'format':'PyRPMD.shared-coordinate-quantum-correction.v1',
            'context':context.as_dict(),'approximation':'matched-active-normalized-flux-defect',
            'baseline_kappa':native_kappa,'active_quantum_kappa':float(q[plateau_start:].mean()),
            'active_rpmd_kappa':float(r[plateau_start:].mean()),'corrected_kappa_estimate':kappa,
            'k_qtst_cm3_s':qtst,'baseline_rate_estimate_cm3_s':qtst*native_kappa,
            'rate_estimate_cm3_s':rate,'rate_cm3_s':None,'physical_converged':False,
            'numerical_prerequisites_passed':not reasons,'withheld_reasons':reasons,
            'conditional_correction_standard_error_cm3_s':float(qtst*boots.std(ddof=1)),
            'uncertainty_scope':'active matched-block sampling only, native QTST and full RPMD held fixed',
            'plateau_start_index':plateau_start,'plateau_max_deviation_with_bootstrap_envelope':bound,
            'time_au':t.tolist(),'corrected_kappa_curve':corrected.tolist(),'delta_kappa_curve':delta.tolist(),
            'negative_estimate_retained':rate<0,'quantum_equals_RPMD_limit_error':float(np.max(abs(delta))) if np.array_equal(pair.quantum_numerator,pair.rpmd_numerator) else None,
            'evidence':pair.evidence,'exact_general_rate_claim':False}


@dataclass
class FiniteBasisFluxBackend:
    """Executable finite-basis Q backend with an explicit matched-RPMD provider.

    problem_factory(**workflow_arguments) must return a dict containing:
      hamiltonian, grid_geometries, rpmd_numerator[blocks,time],
      reference_flux[blocks], independent_block_ids, active_hamiltonian_id,
      reference_flux_id, equilibrium_normalization_id,
      quantum_to_common_measure, evidence.

    Both active numerators must refer to the SAME positive reference flux and
    common equilibrium measure. quantum_to_common_measure converts C_Q/Z_Q to
    that measure; no implicit partition-ratio assumption is made. All blocks
    must be equal-weight independent estimators. The quantum trace is exact
    only within the supplied finite orthonormal basis and stated Hamiltonian.

    This backend solves quantum real-time propagation spectrally; it does NOT
    construct an arbitrary active Hamiltonian or derive its environment closure.
    PMFs, guessed masses and bare transmission probabilities are not accepted
    in place of an explicit physical Hamiltonian and matched flux records.
    """
    problem_factory: object
    last_diagnostics: dict | None = None

    def __post_init__(self):
        if not callable(self.problem_factory):raise TypeError('An explicit matched active problem factory is required')

    def __call__(self, **workflow_arguments):
        context=workflow_arguments['context'];shared=workflow_arguments['coordinate']
        time=workflow_arguments['time_au']
        problem=self.problem_factory(**workflow_arguments)
        required={'hamiltonian','grid_geometries','rpmd_numerator','reference_flux',
                  'independent_block_ids','active_hamiltonian_id','reference_flux_id',
                  'equilibrium_normalization_id','quantum_to_common_measure','evidence'}
        if not isinstance(problem,dict) or set(problem)!=required:
            raise ValueError('Matched finite-basis problem fields differ: '+str(sorted(required)))
        if context.coordinate_id!=shared.coordinate_id or context.surface_id!=shared.surface_id:
            raise ValueError('Problem and native rate coordinates differ')
        h=shared_side_operator(shared,problem['grid_geometries'])
        result=kubo_flux_side(problem['hamiltonian'],h,context.beta_au,time)
        scale=_positive(problem['quantum_to_common_measure'],'quantum_to_common_measure')
        rp=_real(problem['rpmd_numerator'],'rpmd_numerator')
        if rp.ndim!=2 or rp.shape[1]!=len(time):raise ValueError('Expected active-RPMD [blocks,time] integrals')
        # Every block is an equal-weight estimator of the same canonical kernel.
        # Replicating this deterministic trace is not quantum Monte Carlo sampling.
        q=np.broadcast_to(scale*result['canonical_flux_side'],rp.shape).copy()
        ev=json.loads(json.dumps(problem['evidence'],allow_nan=False))
        self.last_diagnostics={
            'finite_basis_dimension':len(h),'quantum_to_common_measure':scale,
            'quantum_reactant_population':result['reactant_population'],
            'log_quantum_partition':result['log_partition'],
            'closed_finite_basis':True,'quantum_trace_sampling_error':0.,
            'quantum_discretization_error_included':False,
            'bath_feedback_constructed_automatically':False,'physical_converged':False,
            'side_operator_source':'same native RateSurface S>0 in every quantum grid geometry'}
        ev['finite_basis_calculation']=dict(self.last_diagnostics)
        return MatchedFluxPair(context,time,q,rp,problem['reference_flux'],
            tuple(problem['independent_block_ids']),problem['active_hamiltonian_id'],
            problem['reference_flux_id'],problem['equilibrium_normalization_id'],ev)


def build_active_slice_hamiltonian(chart, axes, physical_potential, *,
                                   environment_coordinates=None, max_basis=1024):
    """Build a 1--3D mass-lumped finite-element Hamiltonian using native xi first.

    All omitted chart coordinates are held fixed. The kinetic operator uses the
    *induced constrained-manifold metric*, including active cross derivatives,
    and its measure. This is an explicit frozen-coordinate model, NOT an exact
    partial trace, quantum PMF, or automatically consistent dynamic-bath closure.
    No extrinsic geometric-confinement potential is assumed or invented.

    physical_potential(grid_geometries[B,atoms,3]) -> energies[B] in hartree;
    use an actual molecular PES callback, never the native umbrella PMF.
    Dirichlet boundaries, tensor Q1 elements, 2-point Gauss per cell per axis.
    Dense output is bounded for the small finite-basis reference backend.
    """
    from itertools import product
    if not callable(physical_potential):raise TypeError('Explicit physical PES callback required')
    grids=tuple(_real(a,'active grid') for a in axes);d=len(grids)
    if not 1<=d<=3 or any(a.ndim!=1 or len(a)<4 or np.any(np.diff(a)<=0) for a in grids):
        raise ValueError('Need 1--3 increasing active axes with >=4 nodes')
    shape=tuple(len(a)-2 for a in grids);nb=int(np.prod(shape))
    if type(max_basis) is not int or max_basis<2 or nb>max_basis:
        raise ValueError('Finite-basis memory budget exceeded')
    bath=np.zeros(chart.dimension-d) if environment_coordinates is None else _real(environment_coordinates,'environment coordinates')
    if bath.shape!=(chart.dimension-d,):raise ValueError('Supply every omitted chart coordinate')
    def full(u):return np.r_[u,bath]
    interior=np.array(list(product(*(a[1:-1] for a in grids))))
    geometries=np.array([chart.forward(full(u)) for u in interior])
    K=np.zeros((nb,nb));mass=np.zeros(nb);vweight=np.zeros(nb)
    corners=np.array(list(product([0,1],repeat=d)),int)
    quadrature=np.array(list(product([.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)],repeat=d)))
    conditions=[]
    for cell in product(*(range(len(a)-1) for a in grids)):
        start=np.array([a[i] for a,i in zip(grids,cell)])
        width=np.array([a[i+1]-a[i] for a,i in zip(grids,cell)])
        us=start+quadrature*width
        xs=np.array([chart.forward(full(u)) for u in us])
        V=_real(physical_potential(xs),'physical PES energies')
        if V.shape!=(len(us),):raise ValueError('Physical PES must return one energy per physical geometry')
        ids=np.array(cell)[None]+corners-1;inside=np.all((ids>=0)&(ids<np.array(shape)),axis=1)
        global_ids=np.ravel_multi_index(ids[inside].T,shape)
        for loc,u,v in zip(quadrature,us,V):
            metric=chart.active_slice_metric(full(u),d);G=metric['inverse_mass'];rho=metric['density']
            conditions.append(float(np.linalg.cond(G)))
            factors=np.where(corners,loc,1-loc);phi=factors.prod(axis=1)
            grad=np.empty((len(corners),d))
            for j in range(d):
                temp=factors.copy();temp[:,j]=2*corners[:,j]-1
                grad[:,j]=temp.prod(axis=1)/width[j]
            weight=width.prod()*rho/(2**d)
            K[np.ix_(global_ids,global_ids)]+=.5*weight*(grad@G@grad.T)[np.ix_(inside,inside)]
            mass[global_ids]+=weight*phi[inside]
            vweight[global_ids]+=weight*phi[inside]*v
    if np.any(mass<=0):raise ValueError('Degenerate active measure')
    H=K/np.sqrt(mass[:,None]*mass[None])+np.diag(vweight/mass)
    h=shared_side_operator(chart.shared,geometries)
    return {'hamiltonian':H,'side_operator':h,'grid_geometries':geometries,
            'quadrature_mass':mass,'active_grid':interior,
            'diagnostics':{'active_dimensions':d,'basis_size':nb,
                'coordinate_id':chart.shared.coordinate_id,'surface_id':chart.shared.surface_id,
                'metric_condition_max':max(conditions),'dirichlet_boundary':True,
                'kinetic_model':'induced constrained-manifold Laplace-Beltrami',
                'mass_lumped_Q1':True,'physical_PES_callback':True,
                'environment_coordinates_fixed':bath.tolist(),
                'environment_feedback':False,'PMF_used_as_quantum_potential':False,
                'physical_converged':False}}
