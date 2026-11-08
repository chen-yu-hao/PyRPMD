"""Matched-surface quantum flux kernels and an experimental BC defect bridge.

The QM flux is (i/hbar)[H,h(S)] on the SAME physical dividing surface. A finite
closed quantum basis does not automatically have a rate plateau. Quantum Cfs
is NOT divided by its t=0 value (which is zero). All denominator mappings are
explicit; a QM transmission probability is never substituted for RPMD kappa.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from .shared_rate_coordinate import _real
from .recrossing import summarize_recrossing


def _time(time):
    t = _real(time, 'time')
    if t.ndim != 1 or len(t) < 2 or t[0] < 0 or np.any(np.diff(t) <= 0):
        raise ValueError('Increasing nonnegative physical time grid required')
    return t


def assemble_shared_side(reference, axes, chart):
    """Weak h(S) operator for a 1--3D Q1 chart with first coordinate xi.

    The surface must be a mesh hyperplane; quadrature uses the same full metric
    measure as the Hamiltonian. Native S>0 and xi>xi_star agree only on the
    positive denominator patch enforced by the chart. No nodal step/mass
    multiplication that breaks Hermiticity is used.
    """
    from itertools import product
    axes=tuple(_real(a,'axis') for a in axes);d=len(axes)
    if not 1<=d<=3 or any(a.ndim!=1 or len(a)<4 or np.any(np.diff(a)<=0) for a in axes):
        raise ValueError('Need 1--3 increasing finite grids')
    shape=tuple(len(a)-2 for a in axes);n=int(np.prod(shape))
    if reference.vectors.shape[0]!=n:raise ValueError('Grid does not match the quantum reference')
    hits=np.flatnonzero(np.isclose(axes[0],chart.shared.xi_star,rtol=0,atol=1e-12))
    if len(hits)!=1 or not 0<hits[0]<len(axes[0])-1:
        raise ValueError('Put the dividing surface on an interior FE node')
    cells=np.array(list(product(range(int(hits[0]),len(axes[0])-1),
        *(range(len(a)-1) for a in axes[1:]))))
    starts=np.column_stack([a[cells[:,j]] for j,a in enumerate(axes)])
    widths=np.column_stack([np.diff(a)[cells[:,j]] for j,a in enumerate(axes)])
    corners=np.array(list(product([0,1],repeat=d)));nv=len(corners)
    local=np.zeros((len(cells),nv,nv))
    for p in product([.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)],repeat=d):
        p=np.asarray(p);_,rho=chart.geometry(starts+widths*p)
        v=np.prod(np.where(corners,p,1-p),axis=1)
        local+=np.prod(widths,axis=1)[:,None,None]*rho[:,None,None]*np.outer(v,v)/(2**d)
    node=cells[:,None,:]+corners[None,:,:]-1
    inside=np.all((node>=0)&(node<np.array(shape)),axis=2)
    ids=np.zeros((len(cells),nv),int);ids[inside]=np.ravel_multi_index(node[inside].T,shape)
    keep=inside[:,:,None]&inside[:,None,:]
    row=np.broadcast_to(ids[:,:,None],local.shape)[keep];col=np.broadcast_to(ids[:,None,:],local.shape)[keep]
    Hside=coo_matrix((local[keep],(row,col)),shape=(n,n)).tocsr();U=reference.vectors
    if not np.allclose(U.conj().T@(reference.mass_matrix@U),np.eye(len(reference.energies)),atol=1e-8):
        raise ValueError('Reference eigenvectors are not normalized in the FE mass measure')
    return U.conj().T@(Hside@U)


def assemble_slice_side(reference, xi_nodes, path):
    """One-dimensional convenience wrapper for the shared weak side operator."""
    return assemble_shared_side(reference,[xi_nodes],path)


def kubo_flux_side(energies, side_in_energy_basis, time, *, beta, hbar=1., max_states=2048):
    """C_fs^K(t)/Z for a supplied finite closed Hamiltonian and h(S).

    Degenerate-energy terms vanish in the commutator, including arbitrary
    rotations inside degenerate eigenspaces. No wavefunction renormalization,
    clipping, plateau extrapolation or reactant normalization is hidden here.
    """
    E=_real(energies,'energies'); t=_time(time); h=np.asarray(side_in_energy_basis,complex)
    if (E.ndim!=1 or not 2<=len(E)<=max_states or np.any(np.diff(E)<0)
            or h.shape!=(len(E),len(E)) or not np.isfinite(h).all()
            or not np.allclose(h,h.conj().T,rtol=1e-10,atol=1e-12)
            or not np.isfinite([beta,hbar]).all() or min(beta,hbar)<=0):
        raise ValueError('Invalid finite quantum Hamiltonian/side/temperature')
    eigen=np.linalg.eigvalsh(h)
    if eigen.min() < -1e-9 or eigen.max() > 1+1e-9:
        raise ValueError('Projected side must be a positive contraction')
    weight=np.exp(-beta*(E-E[0])); p=weight/weight.sum()
    i,j=np.triu_indices(len(E),1); gap=E[j]-E[i]
    amplitude=2*(p[i]-p[j])*np.abs(h[i,j])**2/(beta*hbar)
    correlation=np.zeros(len(t))
    for start in range(0,len(gap),4096):
        correlation+=np.sin(np.outer(t,gap[start:start+4096]/hbar))@amplitude[start:start+4096]
    return {'time_au':t,'flux_side_per_partition':correlation,
            'reactant_population':float(1-np.dot(p,h.diagonal().real)),
            'log_partition':float(np.log(weight.sum())-beta*E[0]),
            'normalization':'total finite-model thermal partition Z, NOT RPMD positive flux',
            'finite_closed_basis':True,'physical_converged':False}


@dataclass(frozen=True)
class MatchedFluxBlocks:
    """Common native parent measure: same ordered IDs, time and positive flux.

    Quantum providers must map their conditional traces into this measure.
    Merely attaching matching metadata is not a physical proof of that map.
    """
    identity: str
    time_au: np.ndarray
    parent_ids: np.ndarray
    numerator: np.ndarray
    denominator: np.ndarray
    provenance: str

    def __post_init__(self):
        t=_time(self.time_au); ids=np.asarray(self.parent_ids)
        num=_real(self.numerator,'numerator');den=_real(self.denominator,'denominator')
        if (ids.ndim!=2 or ids.shape[1]!=2 or ids.dtype.kind not in 'iu' or np.any(ids<0)
                or len(set(map(tuple,ids)))!=len(ids) or num.shape!=(len(ids),len(t))
                or den.shape!=(len(ids),) or np.any(den<0) or den.sum()<=0
                or np.any(num[den==0]!=0) or not isinstance(self.identity,str) or not self.identity
                or not isinstance(self.provenance,str) or not self.provenance):
            raise ValueError('Invalid/missing matched parent flux evidence')
        for name,value in [('time_au',t),('parent_ids',ids),('numerator',num),('denominator',den)]:
            copy=np.frombuffer(np.ascontiguousarray(value).tobytes(),dtype=value.dtype).reshape(value.shape)
            object.__setattr__(self,name,copy)


def apply_shared_flux_correction(full, active_rp, active_q, *, k_qtst_cm3_s,
                                  plateau_fraction=.5, plateau_tolerance=.1,
                                  bootstrap=256, seed=1938, evidence=None):
    """Experimental Cfs defect at fixed RPMD QTST and common positive flux.

    kappa_mix = sum(N_full + N_Q,active - N_RP,active)/sum(D_full).
    All parent alignment/normalization checks are strict. Resampling uses native
    chain/block statistics; active estimates enter the block-length diagnostics.
    QTST uncertainty is NOT included in the reported conditional standard error.
    This function does not establish transferability or general exactness.
    """
    if not all(isinstance(x,MatchedFluxBlocks) for x in (full,active_rp,active_q)):
        raise TypeError('Explicit MatchedFluxBlocks required, not a transmission number')
    if isinstance(bootstrap,(bool,np.bool_)) or not isinstance(bootstrap,(int,np.integer)) or bootstrap<0:
        raise ValueError('bootstrap must be a nonnegative integer')
    for x in (active_rp,active_q):
        if x.identity!=full.identity or not all(np.array_equal(getattr(x,k),getattr(full,k)) for k in ('time_au','parent_ids','denominator')):
            raise ValueError('Coordinate, parent ordering, time or positive-flux normalization mismatch')
    if not np.isfinite(k_qtst_cm3_s) or k_qtst_cm3_s<0:
        raise ValueError('Finite nonnegative native QTST rate required')
    num=full.numerator+active_q.numerator-active_rp.numerator
    children=[{'chain':int(ids[0]),'parent':int(ids[1]),'pair':0,'numerator':n,'denominator':float(d)}
              for ids,n,d in zip(full.parent_ids,num,full.denominator)]
    extra={tuple(ids):[float(np.mean(rp)),float(np.mean(q))]
           for ids,rp,q in zip(full.parent_ids,active_rp.numerator,active_q.numerator)}
    stats=summarize_recrossing(children,child_steps=len(full.time_au),plateau_fraction=plateau_fraction,
        plateau_tolerance=plateau_tolerance,bootstrap=bootstrap,rng=np.random.default_rng(seed),
        extra_parent_observables=extra)
    required=('physical_core_hamiltonian','kinetic_metric','conditional_initial_measure',
              'common_positive_flux','environment_coupling','quantum_numerics')
    evidence={} if evidence is None else dict(evidence)
    reasons=[name for name in required if not isinstance(evidence.get(name),dict)
             or evidence[name].get('passed') is not True or not isinstance(evidence[name].get('record'),str)
             or not evidence[name]['record']]
    if not stats['plateau_detected']:reasons.append('no_resolved_plateau')
    kappa=float(stats['kappa']); conditional=stats['standard_error']
    return {'identity':full.identity,'time_au':full.time_au,'kappa_curve':stats['curve'],
        'baseline_kappa_curve':full.numerator.sum(0)/full.denominator.sum(),
        'kappa_estimate':kappa,'k_qtst_cm3_s':float(k_qtst_cm3_s),
        'rate_estimate_cm3_s':float(k_qtst_cm3_s*kappa),
        'rate_cm3_s':None,'conditional_sampling_SE_cm3_s':None if conditional is None else float(abs(k_qtst_cm3_s)*conditional),
        'negative_rate':bool(kappa<0),'plateau_detected':bool(stats['plateau_detected']),
        'plateau_diagnostics':stats['plateau_diagnostics'],'recrossing_uncertainty':stats['uncertainty'],
        'parent_block_lengths':stats['parent_block_lengths'],
        'status':'research_estimate' if not reasons else 'requires_validation',
        'unmet_gates':reasons,'evidence':evidence,'physical_converged':False,
        'approximation':'matched conditional Cfs defect; fixed native QTST',
        'uncertainty_scope':'native parent-chain/block sampling, conditional on fixed QTST and supplied core mapping',
        'provenance':[full.provenance,active_rp.provenance,active_q.provenance]}


def quantum_traces_to_blocks(reference_blocks, traces, scales, *, normalization_record):
    """Explicit provider adapter from Cfs/Z traces to common parent numerators.

    scales must be derived by the provider from the conditional initial measure
    and the SAME reference positive flux. This function neither derives those
    weights nor approves a bath approximation. No t=0 division is permitted.
    """
    values=_real(traces,'conditional quantum traces');factors=_real(scales,'trace scales')
    if (values.shape!=reference_blocks.numerator.shape or factors.shape!=reference_blocks.denominator.shape
            or np.any(factors<0) or not isinstance(normalization_record,str) or not normalization_record):
        raise ValueError('Explicit per-parent trace scales and their provenance required')
    return MatchedFluxBlocks(reference_blocks.identity,reference_blocks.time_au,
        reference_blocks.parent_ids,values*factors[:,None],reference_blocks.denominator,
        'quantum traces mapped to native positive-flux measure: '+normalization_record)


@dataclass(frozen=True)
class SharedQuantumCore:
    """Finite thermal quantum kernel bound to one native surface/temperature."""
    identity: str
    beta: float
    hbar: float
    energies: np.ndarray
    side: np.ndarray
    diagnostics: dict

    def flux_side(self,time):
        result=kubo_flux_side(self.energies,self.side,time,beta=self.beta,hbar=self.hbar)
        result.update(identity=self.identity,core_diagnostics=dict(self.diagnostics))
        return result


def solve_shared_core(chart,axes,potential,*,potential_identity,states=32,hbar=1.):
    """Build the actual quantum kernel using the same native xi and RateSurface.

    The supplied XiSlice/XiAffineChart fixes omitted Cartesian directions. This
    is a finite restricted Hamiltonian, NOT a PMF, conditional influence action,
    or automatic quantum/RPMD coupling. A provider must supply that approximation
    and its conditional positive-flux map separately.
    """
    from .shared_rate_coordinate import XiSlice,XiAffineChart
    from .curvilinear_quantum import solve_curvilinear
    if not isinstance(chart,(XiSlice,XiAffineChart)):
        raise TypeError('An explicitly supported shared native xi chart is required')
    identity=chart.shared.identity
    if potential_identity!=chart.shared.pes_identity:
        raise ValueError('Quantum core potential identity differs from the native contract')
    ref=solve_curvilinear(axes,chart.geometry,lambda u:chart.potential(u,potential),states=states,hbar=hbar)
    side=assemble_shared_side(ref,axes,chart)
    if chart.shared.identity!=identity:raise ValueError('Coordinate changed during quantum construction')
    E=np.frombuffer(np.ascontiguousarray(ref.energies).tobytes(),dtype=ref.energies.dtype)
    hs=np.frombuffer(np.ascontiguousarray(side).tobytes(),dtype=side.dtype).reshape(side.shape)
    return SharedQuantumCore(identity,chart.shared.beta,float(hbar),E,hs,
        {**ref.diagnostics,'coordinate':'native RPMDrate xi','side':'native RateSurface > 0',
         'potential_identity':potential_identity,'restriction':'fixed omitted Cartesian directions',
         'environment_coupling_implemented':False,'physical_converged':False})
