"""Geometry-aware, pilot-only molecular quantum active-space candidates.

A local mass-weighted Eckart frame removes external translations/rotations.
The selector is goal-oriented and includes nonlinear force residuals; it does
not infer precision from bead covariance alone. It is NOT a global curvilinear
quantum Hamiltonian builder, nor a proof that a selected core is sufficient.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import null_space


def _real(a, name):
    a = np.asarray(a)
    if a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite and real')
    return np.asarray(a, float)


def internal_basis(reference, masses, *, relative_tolerance=1e-9):
    """Orthonormal mass-weighted complement of rigid translations and rotations.

    Linear molecules have five rigid directions; nonlinear molecules six.
    Rotations do not remove physical bending. The returned basis columns use
    flattened atomic Cartesian components and mass-weighted displacements.
    """
    if not np.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError('relative_tolerance must lie strictly between zero and one')
    x = _real(reference, 'reference'); m = _real(masses,'masses')
    if x.ndim!=2 or x.shape[1]!=3 or m.shape!=(len(x),) or np.any(m<=0) or len(x)<2:
        raise ValueError('Expected reference [atoms,3] and positive atomic masses')
    x=x-np.sum(m[:,None]*x,axis=0)/m.sum();sm=np.sqrt(m)
    rigid=[]
    for a in np.eye(3):rigid.append((sm[:,None]*np.broadcast_to(a,x.shape)).ravel())
    for a in np.eye(3):rigid.append((sm[:,None]*np.cross(np.broadcast_to(a,x.shape),x)).ravel())
    R=np.column_stack(rigid)
    # Normalize nonzero columns before comparing translational/rotational rank.
    sizes=np.linalg.norm(R,axis=0);keep=sizes>1e-14
    R=R[:,keep]/sizes[keep]
    B=null_space(R.T,rcond=relative_tolerance)
    return B,x,3*len(x)-B.shape[1]


def align_molecular_frames(positions, forces, reference, masses):
    """Mass-weighted proper Kabsch alignment, rotating forces by the same map."""
    q=_real(positions,'positions');f=_real(forces,'forces');ref=_real(reference,'reference');m=_real(masses,'masses')
    if q.ndim!=3 or q.shape!=f.shape or q.shape[1:]!=ref.shape or ref.shape!=(len(m),3) or np.any(m<=0):
        raise ValueError('Need matched [frames,atoms,3] coordinates/forces')
    target=ref-(m[:,None]*ref).sum(0)/m.sum()
    centered=q-(q*m[None,:,None]).sum(1,keepdims=True)/m.sum()
    rotated=np.empty_like(q);rf=np.empty_like(f);rotations=np.empty((len(q),3,3))
    for i in range(len(q)):
        U,s,Vt=np.linalg.svd(centered[i].T@(m[:,None]*target))
        sign=np.ones(3);sign[-1]=np.sign(np.linalg.det(U@Vt))
        R=(U*sign)@Vt;rotations[i]=R;rotated[i]=centered[i]@R;rf[i]=f[i]@R
    return rotated,rf,rotations,target


@dataclass(frozen=True)
class MolecularSelection:
    basis_mass_weighted: np.ndarray
    reference: np.ndarray
    selected_dimension: int
    internal_dimension: int
    rigid_dimension: int
    status: str
    nonlinear_eigenvalues: np.ndarray
    nonlinear_retained_fraction: float
    partner_scores: np.ndarray
    unresolved_partner_score: float
    diagnostics: dict

    def as_dict(self):
        return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in vars(self).items()}


def select_molecular_subspace(positions, forces, masses, *, reference, hessians,
                              beta, observable_gradient, max_dimension=3,
                              nonlinear_fraction=.98, partner_tolerance=.05,
                              relative_floor=1e-8):
    """Select a LOCAL physical subspace from independent molecular pilot data.

    Hessians are Cartesian positive second derivatives of the PES, not forces.
    Stable Hessian modes weight the nonlinear residual by the harmonic quantum
    excess x*coth(x)-1, x=beta*omega/2; negative curvature is treated as a
    tunnelling warning, not a stable oscillator. Observable closure and a
    harmonic bath-loading diagnostic follow. The thresholds are heuristics.
    No full quantum result, product labels or production rate enters selection.
    """
    q=_real(positions,'positions');f=_real(forces,'forces');m=_real(masses,'masses');H=_real(hessians,'hessians')
    a=_real(observable_gradient,'observable_gradient')
    if (isinstance(max_dimension,(bool,np.bool_)) or not np.isscalar(max_dimension)
            or not np.isfinite(max_dimension) or int(max_dimension)!=max_dimension or max_dimension<1):
        raise ValueError('Invalid dimension budget')
    if (not np.isfinite([beta,nonlinear_fraction,partner_tolerance,relative_floor]).all()
            or beta<=0 or not 0<nonlinear_fraction<=1 or partner_tolerance<0 or relative_floor<0):
        raise ValueError('Invalid selection controls')
    B,ref,rigid=internal_basis(reference,m)
    N,D=len(q),B.shape[1];nc=3*len(m)
    if N<max(20,2*D+2) or H.shape!=(N,nc,nc) or a.shape!=(len(m),3) or np.linalg.norm(a)==0:
        raise ValueError('Insufficient/mismatched pilot, Cartesian Hessians or observable')
    if not np.allclose(H,H.swapaxes(1,2),rtol=1e-7,atol=1e-9):raise ValueError('Hessians must be symmetric')
    aligned,af,R,ref=align_molecular_frames(q,f,ref,m)
    sm=np.repeat(np.sqrt(m),3)
    z=((aligned-ref).reshape(N,nc)*sm)@B
    ff=(af.reshape(N,nc)/sm)@B
    hh=np.empty((N,D,D))
    for i in range(N):
        rot=np.kron(np.eye(len(m)),R[i].T)
        h=rot@H[i]@rot.T
        hh[i]=B.T@(h/sm[:,None]/sm[None,:])@B
    hm=hh.mean(0);lam,V=np.linalg.eigh(hm)
    x=.5*beta*np.sqrt(abs(lam));excess=np.empty_like(x)
    small=x<1e-3;excess[small]=x[small]**2/3;excess[~small]=x[~small]/np.tanh(x[~small])-1
    # Negative curvature is not assigned a harmonic quantum-statistical weight.
    excess[lam<0]=np.maximum(excess[lam<0],1.)
    Q=(V*np.sqrt(excess/(1+excess)))@V.T
    design=np.column_stack([np.ones(N),z]);residual=ff-design@np.linalg.lstsq(design,ff,rcond=1e-10)[0]
    C=Q@(residual.T@residual/N)@Q
    vals,U=np.linalg.eigh(C);order=np.argsort(vals)[::-1];vals=np.maximum(vals[order],0);U=U[:,order]
    total_var=np.mean(np.sum((ff-ff.mean(0))**2,axis=1))
    if vals.sum()<=relative_floor*max(total_var,1e-30):nnon=0
    else:nnon=min(D,1+int(np.searchsorted(np.cumsum(vals)/vals.sum(),nonlinear_fraction)))
    core=U[:,:nnon]
    a_int=B.T@(a.ravel()/sm);anorm=np.linalg.norm(a_int)
    if anorm<=1e-12*np.linalg.norm(a.ravel()/sm):raise ValueError('Observable lies in rigid-motion space')
    residual_a=a_int-core@(core.T@a_int)
    if np.linalg.norm(residual_a)>1e-8*anorm:core=np.column_stack([core,residual_a/np.linalg.norm(residual_a)])
    W=null_space(core.T);partner=np.array([]);remaining=0.;status='candidate'
    if W.shape[1]:
        eig,rot=np.linalg.eigh(W.T@hm@W);modes=W@rot;coupling=modes.T@hm@core
        if np.any(eig<=0):
            # Do not pretend a residual unstable mode is a passive harmonic bath.
            partner=np.where(eig>0,np.sum(coupling*coupling,axis=1)/np.maximum(eig,1e-30)**2,np.inf)
        else:partner=np.sum(coupling*coupling,axis=1)/eig**2
        idx=np.argsort(partner)[::-1];partner=partner[idx];modes=modes[:,idx]
        add=0
        while partner[add:].sum()>partner_tolerance and core.shape[1]+add<max_dimension:add+=1
        core=np.column_stack([core,modes[:,:add]]);remaining=float(partner[add:].sum())
    if core.shape[1]>max_dimension:status='dimension_budget_exceeded'
    elif remaining>partner_tolerance:status='residual_coupling_exceeds_budget'
    elif core.shape[1]>=D:status='full_internal_space_required_no_reduction'
    # The pilot's local affine-fit failure and geometry span remain explicit.
    dref=np.linalg.norm(ref*np.sqrt(m[:,None]));span=np.linalg.norm((aligned-ref)*np.sqrt(m)[None,:,None],axis=(1,2))/max(dref,1e-20)
    retained=float(vals[:nnon].sum()/vals.sum()) if vals.sum()>0 else 1.
    return MolecularSelection(B@core,ref,core.shape[1],D,rigid,status,vals,retained,partner,remaining,
        {'pilot_frames':N,'beta_au':float(beta),'nonlinear_rank':nnon,'quantum_excess_weights':excess.tolist(),
         'mean_internal_curvatures_au':lam.tolist(),'force_residual_fraction':float(np.mean(np.sum(residual**2,axis=1))/max(total_var,1e-30)),
         'relative_geometry_span_max':float(span.max()),'nonlinear_fraction_threshold':float(nonlinear_fraction),
         'partner_tolerance':float(partner_tolerance),'physical_converged':False,'uses_full_quantum_answer':False,
         'scope':'local aligned Cartesian candidate; no globally valid nonlinear quantum kinetic operator is implied'})
