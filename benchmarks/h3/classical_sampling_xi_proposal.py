"""Exact radial independence proposal for an H3 incoming-channel umbrella.

Prototype only: this does not modify or resume the frozen mixing sampler.
It changes the Jacobi R centroid radius while preserving its direction,
the r path and all zero-centroid ring deviations. The Gaussian umbrella
proposal cancels the umbrella in the MH ratio; the radial volume Jacobian
must remain. The construction applies at any bead count for equal masses.
"""
from __future__ import annotations

import numpy as np

from benchmarks.h3.classical_sampling import coordinate_metric


def incoming_radius(xi,bond,cosine,*,rinf=30.):
    """Unique physical inverse on xi<1, without a quadratic cancellation."""
    xi,bond,cosine=np.broadcast_arrays(xi,bond,cosine)
    if (not np.isfinite([xi,bond,cosine]).all() or np.any(xi>=1)
            or np.any(bond<=0) or np.any(np.abs(cosine)>1)):
        raise ValueError('finite incoming xi<1, positive bond and |cosine|<=1 required')
    u=np.abs(cosine);a=1-xi;b=xi*bond+a*rinf
    aa=2*xi-1;bb=2*a*b-xi*xi*bond*u;cc=xi*xi*bond*bond/4-b*b
    # The quadratic discriminant is xi² times this expression. Computing
    # B²-4AC directly loses precision near the reactant surface xi=0.
    discriminant_factor=(2*b-a*bond*u)**2-aa*bond*bond*(1-u*u)
    if np.any(discriminant_factor < -1e-10):raise ValueError('no physical incoming root')
    radius=-2*cc/(bb+xi*np.sqrt(np.maximum(0.,discriminant_factor)))
    return radius


def log_radial_jacobian(radius,bond,cosine,*,rinf=30.):
    """log(R² |dR/dxi|) for Cartesian centroid volume d³R."""
    _,derivative,_=coordinate_metric(radius,bond,np.abs(cosine),rinf=rinf)
    return 2*np.log(radius)-np.log(np.abs(derivative))


def propose(rng,R,r,centers,stiffness,beta,*,rinf=30.,center_limit=.82,xi_bounds=(-.2,.9)):
    """Return candidate R, log-Jacobian ratio and eligibility mask.

    The calling sampler evaluates beta*mean(V)-log(fs) and its physical
    domain guard. It accepts with log(U) < base_old-base_new+log_J_ratio.
    Proposals outside the fixed interval are rejected, not renormalized.
    States outside that interval are unchanged by this particular kernel;
    other reversible moves can connect them. No result of this function is
    accepted automatically.
    """
    R=np.asarray(R,float);r=np.asarray(r,float);centers=np.asarray(centers,float)
    if (R.shape!=r.shape or R.ndim!=3 or R.shape[-1]!=3 or len(R)!=len(centers)
            or not np.isfinite(R).all() or not np.isfinite(r).all() or beta<=0
            or not xi_bounds[0]<xi_bounds[1]<1):
        raise ValueError('valid Jacobi path arrays and fixed incoming interval required')
    k=np.broadcast_to(np.asarray(stiffness,float),centers.shape)
    if not np.isfinite(k).all() or np.any(k<=0):raise ValueError('positive stiffness required')
    Rc=R.mean(axis=1);rc=r.mean(axis=1)
    radius=np.linalg.norm(Rc,axis=1);bond=np.linalg.norm(rc,axis=1)
    if np.any(radius<=0) or np.any(bond<=0):raise ValueError('nonzero centroid vectors required')
    cosine=np.clip(np.sum(Rc*rc,axis=1)/(radius*bond),-1.,1.)
    old_xi,_,_=coordinate_metric(radius,bond,np.abs(cosine),rinf=rinf)
    wanted=centers+rng.normal(size=len(centers))/np.sqrt(beta*k)
    lower,upper=xi_bounds
    eligible=(centers<=center_limit)&(old_xi>lower)&(old_xi<upper)&(wanted>lower)&(wanted<upper)
    # Invalid proposals are self-loops; avoid evaluating an undefined inverse.
    trial_xi=np.where(eligible,wanted,np.minimum(old_xi,.5))
    new_radius=incoming_radius(trial_xi,bond,cosine,rinf=rinf)
    new_radius=np.where(eligible,new_radius,radius)
    candidate=R+((new_radius/radius-1)[:,None]*Rc)[:,None,:]
    log_ratio=(log_radial_jacobian(new_radius,bond,cosine,rinf=rinf)
               -log_radial_jacobian(radius,bond,cosine,rinf=rinf))
    return candidate,np.where(eligible,log_ratio,0.),eligible,old_xi,wanted
