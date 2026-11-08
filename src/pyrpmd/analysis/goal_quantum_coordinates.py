"""Goal-oriented candidate quantum coordinates; never a certificate of exactness.

Screen nonlinear forces that communicate with the requested observable, rather
than unioning every anharmonic direction. Independent group holdout is mandatory.
Harmonic partner coupling is reported separately, not equated with RPMD error.
Nonlinear transformations must additionally pass a kinetic/measure audit.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import factorial
import numpy as np
from scipy.linalg import null_space
from scipy.interpolate import LSQUnivariateSpline


def _real(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all():
        raise ValueError(f'{name} must be finite real data')
    return np.asarray(raw, dtype=float)


def _features(x, degree, max_features=4096):
    """Multinomial normalization preserves the polynomial kernel under rotations."""
    n, d = x.shape
    terms = [()]
    for k in range(1, degree + 1):
        terms.extend(combinations_with_replacement(range(d), k))
        if len(terms) > max_features:
            raise ValueError('Polynomial feature budget exceeded; use a local chart')
    if n*len(terms)*max(d,1)>20_000_000:
        raise ValueError('Feature memory budget exceeded; use a smaller local pilot')
    F = np.ones((n, len(terms))); derivative = np.zeros((n, len(terms), d))
    for col, inds in enumerate(terms[1:], 1):
        powers = np.bincount(inds, minlength=d)
        weight = np.sqrt(factorial(len(inds)) / np.prod([factorial(int(p)) for p in powers]))
        F[:, col] = weight * np.prod(x[:, inds], axis=1)
        for j in range(d):
            if powers[j]:
                reduced = list(inds); reduced.remove(j)
                derivative[:, col, j] = weight*powers[j]*np.prod(x[:, reduced], axis=1)
    return F, derivative, np.array([len(t) for t in terms])



def _flexible_core_refit(z, force, core, target, groups, train_ids, test, *, seed):
    """Nested-group RBF fit, nonlinear only within the proposed core.

    This tests the proposed residual-bath simplification; it does not assume it.
    Width selection uses an INNER split. The outer test is never used to tune.
    Farthest-point centers and scalar distances are rotation equivariant.
    """
    rng=np.random.default_rng(seed+91);ids=rng.permutation(train_ids)
    inner=np.isin(groups,ids[:max(3,int(.8*len(ids)))]);validation=np.isin(groups,ids[max(3,int(.8*len(ids))):])
    if not validation.any():return np.inf,{'status':'insufficient_inner_groups'}
    latent=z@core;pool=latent[inner];ncent=min(96,max(12,len(pool)//8))
    centers=[];idx=int(np.argmax(np.sum(pool*pool,axis=1)));dist=np.full(len(pool),np.inf)
    for _ in range(ncent):
        centers.append(pool[idx]);dist=np.minimum(dist,np.sum((pool-pool[idx])**2,axis=1));idx=int(np.argmax(dist))
    centers=np.asarray(centers);dd=np.sum((centers[:,None]-centers[None,:])**2,axis=2);np.fill_diagonal(dd,np.inf)
    spacing=max(float(np.sqrt(np.median(dd.min(1)))),1e-3)
    distances=np.sum((latent[:,None]-centers[None,:])**2,axis=2)
    goal=target/np.linalg.norm(target,axis=1)[:,None]
    def error(pred,truth):
        g=truth@goal.T;err=(pred-truth)@goal.T
        a=float(np.max(np.sqrt(np.mean(err**2,axis=0))/np.maximum(np.std(g,axis=0),1e-30)))
        b=float(np.sqrt(np.mean(((pred-truth)@core)**2))/max(np.std(truth@core),1e-30))
        return max(a,b)
    candidates=[]
    for factor in [1.,2.,4.]:
        width=factor*spacing;design=np.column_stack([np.ones(len(z)),z,np.exp(-distances/(2*width*width))])
        reg=np.zeros(design.shape[1]);reg[1+z.shape[1]:]=1e-8
        lhs=design[inner].T@design[inner]+np.diag(reg)
        coef=np.linalg.lstsq(lhs,design[inner].T@force[inner],rcond=1e-12)[0]
        candidates.append((error(design[validation]@coef,force[validation]),width,design,reg))
    best=min(candidates,key=lambda x:x[0]);inner_error,width,design,reg=best
    train=np.isin(groups,train_ids);lhs=design[train].T@design[train]+np.diag(reg)
    coef=np.linalg.lstsq(lhs,design[train].T@force[train],rcond=1e-12)[0]
    return error(design[test]@coef,force[test]),{'kernel_centers':ncent,'width':width,
        'inner_validation_relative_error':inner_error,'outer_test_used_for_tuning':False,
        'hypothesis':'nonlinear active core plus affine full-coordinate force'}

@dataclass(frozen=True)
class GoalSelection:
    basis_mass_weighted: np.ndarray
    required_dimension: int
    status: str
    diagnostics: dict

    def as_dict(self):
        return dict(basis_mass_weighted=self.basis_mass_weighted.tolist(),
                    required_dimension=self.required_dimension, status=self.status,
                    diagnostics=self.diagnostics, quantum_propagation_approved=False)


def screen_goal_subspace(positions, forces, masses, target_covectors, groups, *,
                         max_dimension=3, degree=3, retained_fraction=.99,
                         nonlinear_floor=1e-6, fit_tolerance=.10, flexible_refit=True, seed=8128):
    """Learn linear collective candidates in mass-weighted input coordinates.

    Inputs are physical independent Cartesian/internal *linear* coordinates;
    masses are their positive diagonal kinetic masses. target_covectors is
    [targets,D], defining A = a dot x. For nonlinear coordinates, transform forces
    and audit the FULL kinetic metric separately; this function is only a force
    screen, not an automatic prescription for a reduced kinetic operator.
    groups identify independent chains/paths; beads must share a group.
    """
    q=_real(positions,'positions'); f=_real(forces,'forces'); m=_real(masses,'masses')
    a=_real(target_covectors,'target_covectors'); groups=np.asarray(groups)
    if q.ndim!=2 or f.shape!=q.shape or m.shape!=(q.shape[1],) or np.any(m<=0):
        raise ValueError('Need matching [frames,D] coordinates/forces and positive masses')
    n,d=q.shape; a=np.atleast_2d(a)
    if a.shape[1]!=d or np.any(np.linalg.norm(a,axis=1)==0) or groups.shape!=(n,):
        raise ValueError('Missing target or independent group identifiers')
    if (isinstance(max_dimension,bool) or int(max_dimension)!=max_dimension or not 1<=max_dimension<=3
        or degree not in (2,3) or not 0<retained_fraction<=1
        or not np.isfinite([nonlinear_floor,fit_tolerance]).all()
        or nonlinear_floor<=0 or fit_tolerance<=0):
        raise ValueError('Invalid screening controls')
    ids=np.unique(groups)
    if len(ids)<8:raise ValueError('At least eight independent pilot groups required')
    ids=np.random.default_rng(seed).permutation(ids)
    train=np.isin(groups,ids[:max(4,int(.75*len(ids)))]); test=~train
    sm=np.sqrt(m); z=(q-q[train].mean(0))*sm; ff=f/sm
    scale=float(np.sqrt(np.mean(z[train]**2)))
    if scale<=1e-14:raise ValueError('Pilot has no coordinate span')
    z/=scale
    F,D,orders=_features(z,degree)
    if train.sum()<2*F.shape[1]:raise ValueError('Insufficient independent pilot feature support')
    coef=np.linalg.lstsq(F[train],ff[train],rcond=1e-11)[0]
    prediction=F@coef
    nonlinear=np.einsum('nfj,fo->noj',D[:,orders>=2],coef[orders>=2])/scale
    total=np.einsum('nfj,fo->noj',D,coef)/scale
    _,sv,Vt=np.linalg.svd(a/sm,full_matrices=False)
    core=Vt[sv>sv[0]*1e-10].T
    ranks=[]
    for _ in range(d):
        selected=np.einsum('ok,noj->nkj',core,nonlinear[train])
        C=np.einsum('nki,nkj->ij',selected,selected)/train.sum()
        W=null_space(core.T)
        if not W.shape[1]:break
        eig,U=np.linalg.eigh(W.T@C@W);order=np.argsort(eig)[::-1]
        eig=np.maximum(eig[order],0);U=U[:,order]
        total_size=float(np.mean(np.sum(np.einsum('ok,noj->nkj',core,total[train])**2,axis=(1,2))))
        cutoff=max(nonlinear_floor*total_size,(1-retained_fraction)*np.trace(C),1e-24)
        ranks.append(eig.tolist())
        if eig.sum()<=cutoff:break
        need=int(np.searchsorted(np.cumsum(eig),eig.sum()-cutoff)+1)
        core=np.column_stack((core,W@U[:,:need]))
    err=(prediction[test]-ff[test])@core; target_f=ff[test]@core
    fit=float(np.sqrt(np.mean(err**2))/max(np.sqrt(np.mean((target_f-target_f.mean(0))**2)),1e-30))
    # Do not let a large spectator force conceal a bad fit to the original goal.
    goal=a/sm; goal=goal/np.linalg.norm(goal,axis=1)[:,None]
    gf=ff[test]@goal.T; ge=(prediction[test]-ff[test])@goal.T
    goal_errors=np.sqrt(np.mean(ge*ge,axis=0))/np.maximum(np.std(gf,axis=0),1e-30)
    fit=max(fit,float(goal_errors.max()))
    polynomial_error=fit;refit={}
    if flexible_refit and fit>fit_tolerance and core.shape[1]<=max_dimension:
        fit,refit=_flexible_core_refit(z,ff,core,a/sm,groups,ids[:max(4,int(.75*len(ids)))],test,seed=seed)
    W=null_space(core.T); mean_j=total[train].mean(0)
    # The quantity below diagnoses a possible harmonic bath partner, not quantum error.
    load=float(np.linalg.norm(W.T@mean_j@core)/max(np.linalg.norm(mean_j@core),1e-30))
    status='candidate_requires_dynamical_probes'
    if core.shape[1]>max_dimension:status='dimension_budget_exceeded'
    elif fit>fit_tolerance:status='pilot_force_fit_unreliable'
    return GoalSelection(core,core.shape[1],status,{
        'holdout_relative_force_error':fit,'polynomial_holdout_error':polynomial_error,'flexible_refit':refit,'held_out_groups':int(len(ids)-len(np.unique(groups[train]))),
        'pilot_groups':int(len(ids)),'nonlinear_residual_spectra':ranks,
        'harmonic_partner_loading':load,'retained_fraction':retained_fraction,
        'nonlinear_floor':nonlinear_floor,'fit_tolerance':fit_tolerance,
        'uses_full_quantum_answer':False,'is_error_bound':False,
        'scope':'target-directed nonlinear-force screen; coupling/kinetic validation still required'})


@dataclass
class GraphCoordinateChart:
    """Invertible full-dimensional graph chart in mass-weighted Cartesian space.

    z = [s, b+f(s)]. The first coordinate is a progress variable; its path is
    curved in Cartesian space. f is learned without a quantum reference.
    All coordinates are retained in the map; dimensional reduction is separate.
    """
    origin: np.ndarray
    rotation: np.ndarray
    masses: np.ndarray
    functions: tuple
    support: tuple

    def forward(self,u):
        u=_real(u,'chart coordinates');z=u.copy()
        if u.shape[-1]!=len(self.masses):raise ValueError('Wrong chart dimension')
        if np.any(u[...,0]<self.support[0]) or np.any(u[...,0]>self.support[1]):
            raise ValueError('Extrapolation outside learned progress support is forbidden')
        for j,fn in enumerate(self.functions,1):z[...,j]+=fn(u[...,0])
        return self.origin+(z@self.rotation.T)/np.sqrt(self.masses)

    def inverse(self,x):
        z=(_real(x,'positions')-self.origin)*np.sqrt(self.masses)@self.rotation
        u=z.copy()
        if np.any(z[...,0]<self.support[0]) or np.any(z[...,0]>self.support[1]):
            raise ValueError('Extrapolation outside learned progress support is forbidden')
        for j,fn in enumerate(self.functions,1):u[...,j]-=fn(z[...,0])
        return u

    def jacobian(self,u):
        u=_real(u,'chart coordinates');self.forward(u)
        d=len(self.masses);J=np.broadcast_to(np.eye(d),u.shape[:-1]+(d,d)).copy()
        for j,fn in enumerate(self.functions,1):J[...,j,0]=fn.derivative()(u[...,0])
        return np.einsum('ij,...jk->...ik',self.rotation,J)/np.sqrt(self.masses)[:,None]

    def generalized_force(self,u,cartesian_force):
        return np.einsum('...ia,...i->...a',self.jacobian(u),_real(cartesian_force,'forces'))


def fit_graph_chart(positions,masses,progress_covector,*,knots=6):
    q=_real(positions,'positions');m=_real(masses,'masses');a=_real(progress_covector,'progress_covector')
    if q.ndim!=2 or m.shape!=(q.shape[1],) or a.shape!=m.shape or np.any(m<=0) or np.linalg.norm(a)==0:
        raise ValueError('Invalid chart training data')
    if isinstance(knots,bool) or int(knots)!=knots or knots<0:raise ValueError('Invalid knot count')
    t=a/np.sqrt(m);t/=np.linalg.norm(t);R=np.column_stack((t,null_space(t[None])))
    origin=q.mean(0);z=(q-origin)*np.sqrt(m)@R;order=np.argsort(z[:,0]);z=z[order]
    if len(np.unique(z[:,0]))<2*knots+8:raise ValueError('Insufficient progress support')
    interior=np.quantile(z[:,0],np.linspace(0,1,knots+2)[1:-1])
    functions=tuple(LSQUnivariateSpline(z[:,0],z[:,j],interior,k=3,ext=2) for j in range(1,q.shape[1]))
    return GraphCoordinateChart(origin,R,m,functions,(float(z[0,0]),float(z[-1,0])))


def assess_promotion(delta_blocks,reference_scale,*,tolerance=.05,zscore=1.96,
                     kinetic_complete=False,matched_observable=False):
    """Guarded decision from independent-block *matched target* promotion changes.

    delta_blocks compares predicted target curves before/after promoting a mode
    or a declared joint mode group. It must include the effect of the RPMD bath,
    not just unrelated isolated-subsystem observables. No full quantum truth is
    required. This pointwise-normal envelope is a diagnostic, not a simultaneous
    confidence bound or a guarantee for untested directions/temperatures.
    """
    delta=_real(delta_blocks,'delta_blocks')
    if delta.ndim!=2 or len(delta)<4 or delta.shape[1]<1:
        raise ValueError('Need at least four independent blocks and one target point')
    if not np.isfinite([reference_scale,tolerance,zscore]).all() or min(reference_scale,tolerance,zscore)<=0:
        raise ValueError('Positive target scale/tolerance/zscore required')
    mean=delta.mean(0);se=delta.std(0,ddof=1)/np.sqrt(len(delta))
    lower=float(np.maximum(np.abs(mean)-zscore*se,0).max()/reference_scale)
    upper=float((np.abs(mean)+zscore*se).max()/reference_scale)
    if not kinetic_complete or not matched_observable:status='insufficient_physical_evidence'
    elif lower>tolerance:status='promote_to_quantum_core'
    elif upper<tolerance:status='no_resolved_need_on_this_probe'
    else:status='sampling_or_model_uncertain'
    return {'status':status,'lower_diagnostic':lower,'upper_diagnostic':upper,
            'tolerance':float(tolerance),'is_error_bound':False,'physical_converged':False}


def screen_chart_subspace(coordinates,generalized_forces,geometry,target_covectors,groups,**controls):
    """Local metric-whitened force screen in a supplied FULL nonlinear chart.

    geometry supplies contravariant inverse mass and density, not a flat-mass
    fiction. Whitening uses the median-point tangent metric. Its variation is
    reported and NOT removed from any subsequent quantum Hamiltonian.
    The supplied target is linear in this chart; non-injective feature maps
    such as y**2 alone must not be presented as coordinate charts.
    """
    u=_real(coordinates,'coordinates');fu=_real(generalized_forces,'generalized_forces')
    if u.ndim!=2 or fu.shape!=u.shape:raise ValueError('Mismatched chart data')
    center=np.median(u,axis=0);G0,rho0=geometry(center[None])
    G,rho=geometry(u);G=_real(G,'inverse mass');rho=_real(rho,'density')
    d=u.shape[1]
    if G.shape!=(len(u),d,d) or rho.shape!=(len(u),) or np.any(rho<=0):raise ValueError('Invalid chart geometry')
    if not np.allclose(G,G.swapaxes(1,2)) or np.any(np.linalg.eigvalsh(G)<=0):raise ValueError('Nonpositive chart metric')
    ev,U=np.linalg.eigh(G0[0])
    if np.any(ev<=0):raise ValueError('Nonpositive reference metric')
    L=(U/np.sqrt(ev))@U.T;Li=(U*np.sqrt(ev))@U.T
    z=(u-center)@L.T;fz=fu@Li
    target=np.atleast_2d(_real(target_covectors,'target'))@Li
    r=screen_goal_subspace(z,fz,np.ones(d),target,groups,**controls)
    var=L[None]@G@L.T[None]-np.eye(d)[None]
    out=r.as_dict();out['basis_in_chart_coordinates']=(Li@r.basis_mass_weighted).tolist()
    out['diagnostics'].update({'metric_relative_variation_max':float(np.max(np.linalg.norm(var,axis=(1,2)))/np.sqrt(d)),
        'density_range':[float(rho.min()),float(rho.max())],
        'scope':'local tangent-metric force screen; full variable metric required for propagation'})
    return out


@dataclass
class ArcLengthChart:
    """Nonlinear mass-weighted arc length along a learned graph path.

    Full-dimensional map (ell,b) <-> x. Off-diagonal kinetic couplings remain;
    unit arc-length tangent mass does NOT justify discarding them.
    """
    graph: GraphCoordinateChart
    to_progress: object
    to_length: object

    @property
    def masses(self):return self.graph.masses

    def forward(self,u):
        u=_real(u,'arc coordinates');v=u.copy();v[...,0]=self.to_progress(u[...,0])
        if not np.isfinite(v).all():raise ValueError('Outside learned arc-length support')
        return self.graph.forward(v)

    def inverse(self,x):
        v=self.graph.inverse(x);u=v.copy();u[...,0]=self.to_length(v[...,0]);return u

    def jacobian(self,u):
        u=_real(u,'arc coordinates');v=u.copy();v[...,0]=self.to_progress(u[...,0])
        J=self.graph.jacobian(v);J[...,0]*=self.to_progress.derivative()(u[...,0])[...,None]
        return J


def arc_length_chart(graph,*,nodes=2049):
    from scipy.integrate import cumulative_trapezoid
    from scipy.interpolate import PchipInterpolator
    if isinstance(nodes,bool) or int(nodes)!=nodes or nodes<32:raise ValueError('Arc table too small')
    s=np.linspace(*graph.support,nodes);u=np.zeros((nodes,len(graph.masses)));u[:,0]=s
    J=graph.jacobian(u);speed=np.sqrt(np.sum(graph.masses[None,:]*J[:,:,0]**2,axis=1))
    ell=cumulative_trapezoid(speed,s,initial=0)
    return ArcLengthChart(graph,PchipInterpolator(ell,s,extrapolate=False),PchipInterpolator(s,ell,extrapolate=False))


def choose_core_from_probes(candidates,evidence,*,max_dimension=3):
    """Select the smallest candidate with all explicitly required probes passed.

    candidates: dicts with name, dimension, required_probes, target_id.
    evidence: {probe_name: {delta_blocks, reference_scale, target_id,
                           kinetic_complete, matched_observable}}.
    Joint promotions must be included explicitly to test cooperative effects.
    Incomplete evidence never changes a candidate into a validated core.
    """
    if isinstance(max_dimension,bool) or int(max_dimension)!=max_dimension or not 1<=max_dimension<=3:
        raise ValueError('Invalid quantum budget')
    ledger={}
    for c in sorted(candidates,key=lambda v:v['dimension']):
        if not isinstance(c['dimension'],int) or c['dimension']<1:raise ValueError('Invalid candidate dimension')
        if c['dimension']>max_dimension:continue
        checks={};required=tuple(c['required_probes'])
        if not required:checks['coverage']={'status':'insufficient_physical_evidence'}
        for name in required:
            p=evidence.get(name)
            if p is None or p.get('target_id')!=c['target_id']:
                checks[name]={'status':'insufficient_physical_evidence'};continue
            checks[name]=assess_promotion(p['delta_blocks'],p['reference_scale'],
                tolerance=p.get('tolerance',.05),kinetic_complete=p.get('kinetic_complete',False),
                matched_observable=p.get('matched_observable',False))
        ledger[c['name']]=checks
        if all(x['status']=='no_resolved_need_on_this_probe' for x in checks.values()):
            return {'selected_candidate':c['name'],'dimension':c['dimension'],
                'status':'provisionally_sufficient_on_supplied_probes','ledger':ledger,'physical_converged':False}
    return {'selected_candidate':None,'dimension':None,'status':'no_core_approved','ledger':ledger,'physical_converged':False}
