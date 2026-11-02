"""Independent inverse-coordinate, volume and detailed-balance checks."""
import numpy as np
from scipy.stats import norm

from benchmarks.h3.classical_sampling import coordinate_metric
from benchmarks.h3.classical_sampling_xi_proposal import incoming_radius,log_radial_jacobian,propose
from benchmarks.h3.classical_tst import MASS_AU
from benchmarks.h3.mixing_sampling import free_ring,spring_action
from pyrpmd.constants import KB_AU


def test_incoming_inverse_including_zero_half_and_negative_coordinates():
    x,r,u=np.meshgrid([-.15,-.001,0.,.1,.5,.82,.9],[.8,1.4,2.,5.],[0.,.2,.7,1.],indexing='ij')
    R=incoming_radius(x,r,u)
    actual,_,_=coordinate_metric(R,r,u)
    np.testing.assert_allclose(actual,x,atol=3e-12,rtol=0)


def test_radial_volume_jacobian_agrees_with_finite_difference_inverse():
    x=np.linspace(-.1,.9,47);r=np.linspace(1.1,2.,47);u=np.linspace(0,1,47)
    R=incoming_radius(x,r,u);epsilon=1e-6
    derivative=(incoming_radius(x+epsilon,r,u)-incoming_radius(x-epsilon,r,u))/(2*epsilon)
    np.testing.assert_allclose(np.exp(log_radial_jacobian(R,r,u)),R*R*np.abs(derivative),rtol=2e-8)


def test_centroid_radial_proposal_preserves_free_ring_action():
    rng=np.random.default_rng(99107);beta=1/(KB_AU*300.);n=48;beads=32
    R=free_ring(rng,(n,beads,3),beta,2*MASS_AU/3)
    r=free_ring(rng,(n,beads,3),beta,MASS_AU/2)
    centers=np.linspace(0,.8,n);radius=incoming_radius(centers,1.4,.3)
    R[:,:,0]+=radius[:,None]
    r[:,:,0]+=.3*1.4;r[:,:,1]+=1.4*np.sqrt(1-.3**2)
    candidate,logjac,eligible,_,wanted=propose(rng,R,r,centers,2.,beta)
    assert eligible.all()
    np.testing.assert_allclose(candidate-candidate.mean(axis=1,keepdims=True),R-R.mean(axis=1,keepdims=True),atol=2e-14)
    np.testing.assert_allclose(spring_action(candidate,beta,2*MASS_AU/3),spring_action(R,beta,2*MASS_AU/3),atol=2e-12,rtol=1e-13)
    old_radius=np.linalg.norm(R.mean(axis=1),axis=1);new_radius=np.linalg.norm(candidate.mean(axis=1),axis=1)
    x,_,_=coordinate_metric(new_radius,1.4,.3)
    np.testing.assert_allclose(x,wanted,atol=1e-12)
    np.testing.assert_allclose(logjac,log_radial_jacobian(new_radius,1.4,.3)-log_radial_jacobian(old_radius,1.4,.3),atol=1e-12)


def test_gaussian_umbrella_cancellation_matches_cartesian_proposal_density():
    rng=np.random.default_rng(72281);beta=1/(KB_AU*300.)
    center=np.linspace(0,.8,100);k=np.linspace(2.,30.,100);sd=1/np.sqrt(beta*k)
    old=center+sd*rng.normal(size=100);new=center+sd*rng.normal(size=100)
    bond=np.linspace(1.2,1.7,100);cosine=np.linspace(.05,.95,100)
    Rold=incoming_radius(old,bond,cosine);Rnew=incoming_radius(new,bond,cosine)
    Jold=log_radial_jacobian(Rold,bond,cosine);Jnew=log_radial_jacobian(Rnew,bond,cosine)
    # Independent arbitrary base action: the cancellation cannot rely on a
    # particular PES, flat entrance channel, paper rate, or bead count.
    base_old=.03*Rold**2+.1*np.sin(Rold);base_new=.03*Rnew**2+.1*np.sin(Rnew)
    log_target_change=base_old-base_new-.5*beta*k*((new-center)**2-(old-center)**2)
    log_proposal_reverse=norm.logpdf(old,loc=center,scale=sd)-Jold
    log_proposal_forward=norm.logpdf(new,loc=center,scale=sd)-Jnew
    direct=log_target_change+log_proposal_reverse-log_proposal_forward
    np.testing.assert_allclose(direct,base_old-base_new+Jnew-Jold,atol=2e-13,rtol=1e-13)
