"""Full-chart kinetic metrics and a small 1--3D variational reference solver.

H = -hbar^2/(2*rho) d_a(rho G^{ab} d_b) + V, with norm int rho |psi|^2.
No flat-mass substitution, coordinate freezing, bath elimination or J>0 claim.
Dirichlet tensor-product Q1 finite elements retain all mixed derivative terms.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


def _real(x,name):
    a=np.asarray(x)
    if a.dtype.kind not in 'fiu' or not np.isfinite(a).all():raise ValueError(f'Invalid {name}')
    return np.asarray(a,float)


def transform_geometry(jacobian,base_inverse_mass,base_density):
    """Pull back a full-dimensional coordinate map x=F(u).

    base_density is explicit. In rotation-reduced molecular coordinates it
    generally is NOT sqrt(det(g)); substituting that expression is forbidden.
    """
    J=_real(jacobian,'Jacobian');G=_real(base_inverse_mass,'inverse mass');rho=_real(base_density,'density')
    if J.ndim<2 or J.shape[-2]!=J.shape[-1] or G.shape!=J.shape or rho.shape!=J.shape[:-2]:
        raise ValueError('Full square charts and matched geometry shapes required')
    if np.any(rho<=0) or not np.allclose(G,G.swapaxes(-1,-2),atol=1e-12,rtol=1e-10):
        raise ValueError('Need positive measure and symmetric inverse mass')
    if np.any(np.linalg.eigvalsh(G)<=0) or np.any(np.linalg.cond(J)>1e10):
        raise ValueError('Singular chart or nonpositive kinetic metric')
    inv=np.linalg.inv(J)
    return inv@G@inv.swapaxes(-1,-2),rho*np.abs(np.linalg.det(J))


def cartesian_geometry(chart,u):
    u=_real(u,'coordinates');J=chart.jacobian(u);d=J.shape[-1]
    G=np.broadcast_to(np.diag(1/np.asarray(chart.masses)),J.shape)
    return transform_geometry(J,G,np.ones(u.shape[:-1]))


def audit_chart(chart,points,*,step=1e-5):
    u=_real(points,'points')
    if u.ndim!=2 or step<=0 or not np.isfinite(step):raise ValueError('Invalid audit points/step')
    x=chart.forward(u);recovered=chart.inverse(x);J=chart.jacobian(u)
    numerical=np.empty_like(J)
    for j in range(u.shape[1]):
        h=np.zeros(u.shape[1]);h[j]=step
        numerical[...,j]=(chart.forward(u+h)-chart.forward(u-h))/(2*step)
    G,rho=cartesian_geometry(chart,u)
    g=np.linalg.inv(G);d=u.shape[1]
    cross=0. if d==1 else float(np.max(np.abs(g[:,0,1:])/np.sqrt(g[:,0,0,None]*np.diagonal(g,axis1=1,axis2=2)[:,1:])))
    return {'roundtrip_max':float(np.max(np.abs(recovered-u))),
            'jacobian_relative_max':float(np.max(np.abs(J-numerical))/max(np.max(np.abs(J)),1e-30)),
            'inverse_mass_condition_max':float(np.linalg.cond(G).max()),
            'density_min':float(rho.min()),'density_max':float(rho.max()),
            'normalized_core_environment_metric_cross_max':cross,
            'reduction_approved':False,'scope':'full chart, NOT a reduced quantum bath Hamiltonian'}


def triatomic_bond_geometry(r1,r2,theta,masses):
    """J=0 triatomic bond lengths/angle geometry after removing COM and rotation.

    Atom order A-B-C; r1=AB, r2=BC, theta=ABC. G=B M^-1 B^T computed from
    Cartesian covectors; volume rho proportional to r1^2*r2^2*sin(theta).
    The theta chart excludes collinearity; use an angular basis there.
    """
    r1,r2,theta=np.broadcast_arrays(_real(r1,'r1'),_real(r2,'r2'),_real(theta,'theta'))
    m=_real(masses,'masses')
    if m.shape!=(3,) or np.any(m<=0) or np.any(r1<=0) or np.any(r2<=0):raise ValueError('Invalid masses/distances')
    if np.any(theta<=1e-5) or np.any(theta>=np.pi-1e-5):raise ValueError('Singular collinear angular chart')
    e1=np.stack([np.ones_like(theta),np.zeros_like(theta),np.zeros_like(theta)],axis=-1)
    e2=np.stack([np.cos(theta),np.sin(theta),np.zeros_like(theta)],axis=-1)
    B=np.zeros(theta.shape+(3,3,3))
    B[...,0,0,:]=e1;B[...,0,1,:]=-e1
    B[...,1,2,:]=e2;B[...,1,1,:]=-e2
    a=(np.cos(theta)[...,None]*e1-e2)/(r1*np.sin(theta))[...,None]
    c=(np.cos(theta)[...,None]*e2-e1)/(r2*np.sin(theta))[...,None]
    B[...,2,0,:]=a;B[...,2,2,:]=c;B[...,2,1,:]=-a-c
    G=np.einsum('...aij,...bij,i->...ab',B,B,1/m)
    return G,r1*r1*r2*r2*np.sin(theta)


def triatomic_transfer_geometry(u,masses):
    """Full nonlinear molecular chart: u=(log(r1/r2),log(r1+r2),theta).

    q1 is a dimensionless signed bond-transfer coordinate, q2 collective stretch,
    q3 bending. J=0 only; finite-J Coriolis/rotation couplings are NOT included.
    """
    from scipy.special import expit
    u=_real(u,'internal chart')
    if u.shape[-1]!=3:raise ValueError('Three internal coordinates required')
    L=np.exp(u[...,1]);s=expit(u[...,0]);r1=L*s;r2=L*(1-s)
    G,rho=triatomic_bond_geometry(r1,r2,u[...,2],masses)
    J=np.zeros(u.shape[:-1]+(3,3));v=L*s*(1-s)
    J[...,0,0]=v;J[...,0,1]=r1;J[...,1,0]=-v;J[...,1,1]=r2;J[...,2,2]=1
    return transform_geometry(J,G,rho)


@dataclass(frozen=True)
class CurvilinearEigenResult:
    energies: np.ndarray
    vectors: np.ndarray
    mass_matrix: object
    diagnostics: dict


@dataclass(frozen=True)
class CurvilinearMatrices:
    """Q1 finite-element matrices on the interior (Dirichlet) nodes.

    ``hamiltonian`` and ``mass`` are sparse; ``extra`` holds one sparse matrix
    ``int rho w(u) phi_i phi_j`` per weight function passed to
    :func:`assemble_curvilinear`.  ``nodes`` are the interior node
    coordinates ``[dof, d]`` in the order of the matrix rows.
    """
    hamiltonian: object
    mass: object
    extra: dict
    nodes: np.ndarray
    shape: tuple
    diagnostics: dict


def assemble_curvilinear(axes, geometry, potential, *, hbar=1., max_dof=20000, weights=None):
    """Assemble the weak form of ``-hbar^2/(2 rho) d_a(rho G^ab d_b) + V``.

    ``weights`` maps names to callables ``w(points) -> [N]``; each yields an
    additional measure-weighted mass-like matrix (absorbing potentials, a
    side indicator evaluated per cell, ...).  Two Gauss points per cell per
    axis, as in :func:`solve_curvilinear`.
    """
    axes=tuple(_real(a,'axis') for a in axes);d=len(axes)
    if not 1<=d<=3 or any(a.ndim!=1 or len(a)<4 or np.any(np.diff(a)<=0) for a in axes):
        raise ValueError('Need 1--3 increasing grids with >=4 nodes each')
    shape=tuple(len(a)-2 for a in axes);ndof=int(np.prod(shape))
    if not np.isfinite(hbar) or hbar<=0 or ndof>max_dof:
        raise ValueError('Invalid hbar or memory budget exceeded')
    weights=dict(weights or {})
    cells=np.array(list(product(*(range(len(a)-1) for a in axes))),int)
    widths=np.column_stack([np.diff(a)[cells[:,j]] for j,a in enumerate(axes)])
    starts=np.column_stack([a[cells[:,j]] for j,a in enumerate(axes)])
    corners=np.array(list(product([0,1],repeat=d)),int);nc=len(cells);nv=len(corners)
    H=np.zeros((nc,nv,nv));M=np.zeros_like(H);W={k:np.zeros_like(H) for k in weights}
    rho_min=np.inf;rho_max=0.;cond_max=0.;potential_min=np.inf
    for loc in product([.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)],repeat=d):
        loc=np.asarray(loc);points=starts+widths*loc
        G,rho=geometry(points);G=_real(G,'metric');rho=_real(rho,'measure');V=_real(potential(points),'potential')
        potential_min=min(potential_min,float(V.min()))
        if G.shape!=(nc,d,d) or rho.shape!=(nc,) or V.shape!=(nc,):raise ValueError('Wrong provider output shape')
        if np.any(rho<=0) or not np.allclose(G,G.transpose(0,2,1),rtol=1e-9,atol=1e-12):raise ValueError('Invalid geometry')
        eig=np.linalg.eigvalsh(G)
        if np.any(eig<=0):raise ValueError('Nonpositive kinetic metric')
        cond_max=max(cond_max,float(np.max(eig[:,-1]/eig[:,0])))
        rho_min=min(rho_min,float(rho.min()));rho_max=max(rho_max,float(rho.max()))
        phi=np.prod(np.where(corners,loc,1-loc),axis=1)
        grad=np.empty((nc,nv,d))
        for j in range(d):
            factors=np.where(corners,loc,1-loc).copy();factors[:,j]=2*corners[:,j]-1
            grad[:,:,j]=np.prod(factors,axis=1)[None,:]/widths[:,j,None]
        weight=np.prod(widths,axis=1)*rho/(2**d)
        S=np.outer(phi,phi)
        M+=weight[:,None,None]*S
        H+=weight[:,None,None]*(.5*hbar*hbar*np.einsum('nai,nij,nbj->nab',grad,G,grad)+V[:,None,None]*S)
        for name,function in weights.items():
            value=_real(function(points),name)
            if value.shape!=(nc,):raise ValueError(f'weight {name} must return one value per point')
            W[name]+=(weight*value)[:,None,None]*S
    node=cells[:,None,:]+corners[None,:,:]-1
    inside=np.all((node>=0)&(node<np.array(shape)),axis=2)
    ids=np.zeros((nc,nv),int)
    ids[inside]=np.ravel_multi_index(node[inside].T,shape)
    valid=inside[:,:,None]&inside[:,None,:]
    row=np.broadcast_to(ids[:,:,None],H.shape)[valid];col=np.broadcast_to(ids[:,None,:],H.shape)[valid]
    def sparse(A):
        return coo_matrix((A[valid],(row,col)),shape=(ndof,ndof)).tocsr()
    interior=np.stack(np.meshgrid(*(a[1:-1] for a in axes),indexing='ij'),axis=-1).reshape(-1,d)
    return CurvilinearMatrices(sparse(H),sparse(M),{k:sparse(v) for k,v in W.items()},interior,shape,
        {'dof':ndof,'dimension':d,'dirichlet_boundary':True,'density_min':rho_min,'density_max':rho_max,
         'metric_condition_max':cond_max,'potential_min':potential_min})


def solve_curvilinear(axes,geometry,potential,*,states=6,hbar=1.,max_dof=20000):
    """Small full-coordinate reference, not a generic reduced reaction solver.

    geometry(points) -> (inverse_mass[N,D,D], density[N]); potential -> [N].
    All mixed terms and the supplied measure enter the weak form. Eigenvectors
    are normalized by the FE mass matrix. Two Gauss points per cell per axis.
    """
    axes=tuple(_real(a,'axis') for a in axes)
    ndof=int(np.prod([len(a)-2 for a in axes]))
    if (isinstance(states,bool) or int(states)!=states or not 1<=states<ndof-1
        or not np.isfinite(hbar) or hbar<=0 or ndof>max_dof):
        raise ValueError('Invalid states/hbar or memory budget exceeded')
    matrices=assemble_curvilinear(axes,geometry,potential,hbar=hbar,max_dof=max_dof)
    Hs,Ms=matrices.hamiltonian,matrices.mass
    E,U=eigsh(Hs,k=states,M=Ms,sigma=matrices.diagnostics['potential_min']-1.,which='LM',tol=1e-10,
              v0=np.random.default_rng(732).normal(size=ndof))
    order=np.argsort(E);E=E[order];U=U[:,order]
    residual=Hs@U-(Ms@U)*E[None,:]
    info=dict(matrices.diagnostics);info.pop('potential_min')
    info.update({'eigen_residual_max':float(np.linalg.norm(residual,axis=0).max()),
                 'mass_orthogonality_error':float(np.max(abs(U.T@(Ms@U)-np.eye(states)))),
                 'physical_converged':False,'no_bath_elimination':True})
    return CurvilinearEigenResult(E,U,Ms,info)
