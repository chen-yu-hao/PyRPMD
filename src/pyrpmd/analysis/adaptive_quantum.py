"""Pilot-only active spaces and harmonic-reservoir response embedding.

Inputs are mass-weighted physical coordinates, not bead-index normal modes.
Selection is a heuristic, never an error bound. The reservoir closure is exact
only for quadratic dynamics and is not an entangled quantum/bath propagator.
All validation/production decisions must use independent data.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import null_space
from scipy.integrate import trapezoid

@dataclass(frozen=True)
class Selection:
    """Candidate physical subspace and explicit budget/failure diagnostics."""
    basis: np.ndarray
    nonlinear_rank: int
    selected_dimension: int
    nonlinear_eigenvalues: np.ndarray
    reservoir_scores: np.ndarray
    reservoir_frequencies: np.ndarray
    reservoir_basis: np.ndarray
    reservoir_coupling: np.ndarray
    unresolved_score: float
    status: str

def _real(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all():
        raise ValueError(f'{name} must contain finite real numbers')
    return np.asarray(raw, dtype=float)

def select_active_subspace(positions, forces, hessians, observable_direction, *,
                           max_dimension=3, nonlinear_fraction=.99,
                           entrainment_tolerance=.05, nonlinear_floor=1e-10):
    """Select nonlinear force directions, then strongly entrained bath modes.

    The force residual is relative to an affine least-squares fit. Its range
    detects even nonlinear couplings such as x*y**2 that linear correlation
    misses. Observable closure is enforced before a harmonic loading test.
    ``scores = ||L_j||**2 / omega_j**4`` is dimensionless but NOT an accuracy
    certificate. A budget-exceeded result must not be silently truncated.
    """
    z = _real(positions, 'positions'); f = _real(forces, 'forces')
    h = _real(hessians, 'hessians'); a = _real(observable_direction, 'observable')
    if z.ndim != 2 or f.shape != z.shape or len(z) < z.shape[1] + 2:
        raise ValueError('Need matched [samples,physical_dimension] arrays')
    N, D = z.shape
    if h.shape != (N,D,D) or a.shape != (D,) or np.linalg.norm(a) == 0:
        raise ValueError('Compatible Hessians and nonzero observable required')
    if not np.allclose(h, h.swapaxes(-1,-2), atol=1e-8, rtol=1e-8):
        raise ValueError('Hessians must be symmetric; no silent projection')
    if (isinstance(max_dimension, (bool,np.bool_)) or int(max_dimension) != max_dimension
            or max_dimension < 1 or not np.isfinite(nonlinear_fraction)
            or not 0 < nonlinear_fraction <= 1 or not np.isfinite(entrainment_tolerance)
            or entrainment_tolerance < 0 or not np.isfinite(nonlinear_floor) or nonlinear_floor < 0):
        raise ValueError('Invalid selection controls')
    max_dimension = min(int(max_dimension), D)
    design = np.column_stack([np.ones(N),z])
    residual = f - design @ np.linalg.lstsq(design,f,rcond=None)[0]
    covariance = residual.T @ residual / N
    vals,U = np.linalg.eigh(covariance); order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order],0); U = U[:,order]
    scale = np.mean(np.sum((f-f.mean(0))**2,axis=1))
    if vals.sum() <= nonlinear_floor * max(scale,1e-30):
        nnon=0; core=np.empty((D,0))
    else:
        nnon=min(D,int(np.searchsorted(np.cumsum(vals)/vals.sum(),nonlinear_fraction)+1))
        core=U[:,:nnon]
    rest=a-core@(core.T@a)
    if np.linalg.norm(rest)>1e-8*np.linalg.norm(a):
        core=np.column_stack([core,rest/np.linalg.norm(rest)])
    if core.shape[1]==1 and core[:,0]@a<0: core[:,0]*=-1
    if core.shape[1]>max_dimension:
        return Selection(core,nnon,core.shape[1],vals,np.array([]),np.array([]),
                         np.empty((D,0)),np.empty((0,core.shape[1])),float('inf'),
                         'dimension_budget_exceeded')
    W=null_space(core.T); hm=h.mean(0); K=W.T@hm@W
    freq2,R=np.linalg.eigh(K)
    if freq2.size and freq2.min()<=0:
        return Selection(core,nnon,core.shape[1],vals,np.array([]),np.array([]),W,
                         np.empty((0,core.shape[1])),float('inf'),'nonpositive_reservoir_curvature')
    modes=W@R; L=modes.T@hm@core
    scores=np.sum(L*L,axis=1)/freq2**2 if len(freq2) else np.array([])
    order=np.argsort(scores)[::-1]
    scores=scores[order]; freq=np.sqrt(freq2[order]); modes=modes[:,order]; L=L[order]
    m=0
    while scores[m:].sum()>entrainment_tolerance and core.shape[1]+m<max_dimension: m+=1
    basis=np.column_stack([core,modes[:,:m]]); remaining=float(scores[m:].sum())
    status='candidate' if remaining<=entrainment_tolerance else 'residual_loading_exceeds_budget'
    if basis.shape[1]==D: status='full_dimension_required_no_reduction'
    return Selection(basis,nnon,basis.shape[1],vals,scores,freq,modes,L,remaining,status)

def passive_reservoir_self_energy(frequencies, damping, stiffness, cross_coupling):
    """Sigma(s)=L.T [K^-1-(K+s^2 I)^-1] L, s=eta-i*omega.

    The active Hamiltonian MUST use the statically relaxed potential; otherwise
    the static counterterm is counted twice. Nonlinear system/reservoir coupling
    cannot be represented merely by inserting an averaged Hessian here.
    """
    w=_real(frequencies,'frequencies'); K=_real(stiffness,'stiffness')
    L=_real(cross_coupling,'cross_coupling'); eta=_real(damping,'damping')
    if (w.ndim!=1 or len(w)<1 or np.any(w<0) or np.any(np.diff(w)<=0)
            or eta.ndim!=0 or eta<=0 or K.ndim!=2 or K.shape[0]!=K.shape[1]
            or L.ndim!=2 or L.shape[0]!=K.shape[0]):
        raise ValueError('Invalid reservoir dimensions, frequency grid or damping')
    if not np.allclose(K,K.T): raise ValueError('Reservoir stiffness must be symmetric')
    if K.shape[0]==0: return np.zeros((len(w),L.shape[1],L.shape[1]),complex)
    val,R=np.linalg.eigh(K)
    if val.min()<=0: raise ValueError('Reservoir stiffness must be positive definite')
    C=R.T@L; s=float(eta)-1j*w
    return np.einsum('ja,wj,jb->wab',C,1/val[None,:]-1/(val[None,:]+s[:,None]**2),C)

def passive_embed(quantum_active_response, frequencies, damping, stiffness, cross_coupling):
    """Embed a matched active coordinate response; no clipping or renormalizing.

    Uses (I+Q Sigma)^-1 Q rather than explicitly inverting Q. The result is
    a closure approximation for an anharmonic active Hamiltonian.
    """
    Q=np.asarray(quantum_active_response,complex); scalar=Q.ndim==1
    if scalar: Q=Q[:,None,None]
    S=passive_reservoir_self_energy(frequencies,damping,stiffness,cross_coupling)
    if Q.shape!=S.shape or not np.isfinite(Q).all():
        raise ValueError('Matched active observable basis required')
    out=np.linalg.solve(np.eye(Q.shape[-1])[None,:,:]+Q@S,Q)
    return out[:,0,0] if scalar else out

def inverse_defect(full_rp, active_rp, active_q):
    """Scalar matched-observable inverse defect; singular input is rejected."""
    F,R,Q=map(lambda x:np.asarray(x,complex),(full_rp,active_rp,active_q))
    if F.ndim!=1 or F.shape!=R.shape or R.shape!=Q.shape:
        raise ValueError('Same scalar observable and frequency grid required')
    if any(not np.isfinite(v).all() or np.any(v==0) for v in (F,R,Q)):
        raise ValueError('Nonfinite or singular response; choose an explicit regularizer')
    den=1/F+1/Q-1/R
    if np.any(den==0): raise ValueError('Singular corrected inverse response')
    return 1/den

def spectral_diagnostics(chi, frequencies):
    """Absorption sign for integral exp((i*w-eta)t) chi(t) dt.

    A negative real part of chi is NOT a negative absorption diagnostic.
    """
    response=np.asarray(chi,complex); w=_real(frequencies,'frequencies')
    if response.ndim!=1 or response.shape!=w.shape or len(w)<2 or np.any(np.diff(w)<=0):
        raise ValueError('Matching scalar spectrum and increasing frequencies required')
    if not np.isfinite(response).all():
        return {'negative_absorption_fraction':float('nan'),'minimum_absorption':float('nan'),
                'finite':False}
    S=response.imag; denominator=trapezoid(abs(S),w)
    return {'negative_absorption_fraction':float(trapezoid(np.maximum(-S,0),w)/max(denominator,1e-30)),
            'minimum_absorption':float(S.min()),'finite':True}

def matched_rate_defect(full_rp, active_rp, active_q, *, beta, observable, units, mode='additive'):
    """Experimental plateau-rate defect transfer, not a general rate theorem.

    Caller must establish the SAME dividing surface, reactant normalization,
    temperature and units for all three rates. Returns negative results unchanged
    so a failed transfer cannot be hidden. Uncertainty must be propagated from
    whole replicas and from the quantum reference separately.

    ``mode='additive'`` (default) returns ``full + q - rp``; ``mode='ratio'``
    returns ``full * q / rp``, which is invariant to a prefactor shared by
    both active-space rates and is the transfer used by
    :mod:`pyrpmd.analysis.subspace_correction`.
    """
    values=_real([full_rp,active_rp,active_q,beta],'rates and beta')
    if values.shape!=(4,) or values[3]<=0 or np.any(values[:3]<0) or not observable or not units:
        raise ValueError('Positive beta, nonnegative matched rates, observable and units required')
    if mode=='additive':
        corrected=float(values[0]+values[2]-values[1]); label='matched-plateau-additive-rate-defect'
    elif mode=='ratio':
        if values[1]<=0: raise ValueError('ratio mode requires a positive active-space RPMD rate')
        corrected=float(values[0]*values[2]/values[1]); label='matched-plateau-multiplicative-rate-defect'
    else:
        raise ValueError("mode must be 'additive' or 'ratio'")
    return {'rate':corrected,'negative_rate':corrected<0,'beta':float(beta),'observable':str(observable),
            'units':str(units),'approximation':label,'exact_general_rate_claim':False}

__all__=['Selection','select_active_subspace','passive_reservoir_self_energy',
         'passive_embed','inverse_defect','spectral_diagnostics','matched_rate_defect']
