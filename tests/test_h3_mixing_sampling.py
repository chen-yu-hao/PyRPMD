"""Independent identities for the equilibrium H3 sampling proposals."""
import numpy as np
import pytest

from benchmarks.h3.classical_tst import MASS_AU
from benchmarks.h3.mixing_sampling import (
    cartesian, coordinate_metric, free_ring, initial_paths, jacobi,
    random_rotations, remaining_action, run, spring_action,
)
from pyrpmd.constants import KB_AU
from pyrpmd.cv import CVStack, RPMDrateCoordinate

BETA = 1/(KB_AU*300.)


def production_coordinate():
    return RPMDrateCoordinate([[0],[1,2]],np.full(3,MASS_AU),
        np.array([[-1.7570000251,0,0],[0,0,0],[1.7570000251,0,0]]),
        [[0,1]],[[1,2]],30.,equivalent_paths=[[0,2,1]])


def test_vectorized_coordinate_and_metric_match_production_gradient():
    rng=np.random.default_rng(98120)
    R=rng.normal(size=(160,3));R*=rng.uniform(1.8,35,size=(160,1))/np.linalg.norm(R,axis=1)[:,None]
    r=rng.normal(size=(160,3));r*=rng.uniform(1.2,4,size=(160,1))/np.linalg.norm(r,axis=1)[:,None]
    actual,logfs,valid,_=coordinate_metric(R,r,BETA)
    assert valid.all()
    cv=production_coordinate()
    for q,x,l in zip(cartesian(R,r),actual,logfs):
        wanted,g=cv.centroid_value_gradient(q)
        expected=.5*np.log(np.sum(g*g/MASS_AU)/(2*np.pi*BETA))
        assert x==pytest.approx(wanted,rel=2e-13,abs=2e-13)
        assert l==pytest.approx(expected,rel=2e-13,abs=2e-13)


@pytest.mark.parametrize('beads',[1,8,128])
def test_initial_centroids_match_every_declared_umbrella(beads):
    centers=np.linspace(-.06,1.10,59)
    R,r=initial_paths(centers,beads)
    xi,_,valid,_=coordinate_metric(R.mean(axis=1),r.mean(axis=1),BETA)
    assert valid.all()
    np.testing.assert_allclose(xi,centers,atol=1e-14,rtol=0)


@pytest.mark.parametrize('beads',[2,8,128])
def test_free_ring_covariance_matches_discrete_normal_modes(beads):
    rng=np.random.default_rng(6120+beads)
    mass=2*MASS_AU/3
    paths=free_ring(rng,(4000,beads,3),BETA,mass)
    np.testing.assert_allclose(paths.mean(axis=1),0,atol=1e-14)
    modes=np.fft.rfft(paths,axis=1,norm='ortho')
    k=np.arange(1,beads//2+1)
    expected=BETA/(4*mass*beads*np.sin(np.pi*k/beads)**2)
    observed=np.mean(np.abs(modes[:,1:])**2,axis=(0,2))
    np.testing.assert_allclose(observed,expected,rtol=.045,atol=0)
    expected_action=3*(beads-1)/2
    # The sum has 4000*3*(P-1) Gaussian squared degrees of freedom.
    # Use its derived five-sigma sampling bound, not a fitted tolerance.
    relative_bound=5*np.sqrt(2/(4000*3*(beads-1)))
    assert spring_action(paths,BETA,mass).mean()==pytest.approx(expected_action,rel=relative_bound)


def test_jacobi_map_and_spring_mass_factors():
    rng=np.random.default_rng(7812)
    R=free_ring(rng,(20,32,3),BETA,2*MASS_AU/3)
    r=free_ring(rng,(20,32,3),BETA,MASS_AU/2)
    q=cartesian(R,r)
    R2,r2=jacobi(q)
    np.testing.assert_allclose(R2,R,atol=1e-15)
    np.testing.assert_allclose(r2,r,atol=1e-15)
    np.testing.assert_allclose(q.mean(axis=-2),0,atol=1e-15)
    delta=q-np.roll(q,1,axis=1)
    cartesian_action=MASS_AU*32/(2*BETA)*np.sum(delta*delta,axis=(1,2,3))
    np.testing.assert_allclose(cartesian_action,
        spring_action(R,BETA,2*MASS_AU/3)+spring_action(r,BETA,MASS_AU/2),rtol=2e-15)


def test_pcn_transition_ratio_cancels_spring_prior():
    rng=np.random.default_rng(19121);alpha=.37;rho=np.sqrt(1-alpha**2)
    mass=MASS_AU/2
    u=free_ring(rng,(50,16,3),BETA,mass)
    v=rho*u+alpha*free_ring(rng,u.shape,BETA,mass)
    # Gaussian proposal precision is spring precision / alpha**2.
    log_reverse_minus_forward=(spring_action(v-rho*u,BETA,mass)
        -spring_action(u-rho*v,BETA,mass))/alpha**2
    np.testing.assert_allclose(log_reverse_minus_forward,
        spring_action(v,BETA,mass)-spring_action(u,BETA,mass),atol=5e-13,rtol=1e-12)


def test_rotation_preserves_spring_action_and_has_unit_jacobian():
    rng=np.random.default_rng(9128)
    Q=random_rotations(rng,80)
    np.testing.assert_allclose(Q@Q.transpose(0,2,1),np.broadcast_to(np.eye(3),(80,3,3)),atol=2e-15)
    np.testing.assert_allclose(np.linalg.det(Q),1,atol=2e-15)
    x=free_ring(rng,(80,16,3),BETA,MASS_AU)
    rotated=np.einsum('nij,npj->npi',Q,x)
    np.testing.assert_allclose(spring_action(rotated,BETA,MASS_AU),spring_action(x,BETA,MASS_AU),rtol=2e-15)


def test_adjacent_exchange_acceptance_uses_only_umbrella_difference():
    xi=np.array([.12,.13]);centers=np.array([.1,.15]);stiffness=2.
    old=.5*BETA*stiffness*np.sum((xi-centers)**2)
    new=.5*BETA*stiffness*np.sum((xi[::-1]-centers)**2)
    actual=BETA*stiffness*(centers[1]-centers[0])*(xi[0]-xi[1])
    assert actual==pytest.approx(old-new,abs=2e-15)


def test_remaining_action_matches_production_bias_energy_for_quantum_ring():
    from types import SimpleNamespace
    from pyrpmd.bias import CompositeBias, FluxMetricBias, UmbrellaBias
    class Harmonic:
        def evaluate(self,q):
            return SimpleNamespace(energy=.01*np.sum(q*q,axis=(1,2)),status=np.zeros(len(q),int))
    centers=np.array([.1,.6,.98,1.02]);k=np.array([2.,5.,30.,30.])
    R,r=initial_paths(centers,16)
    rng=np.random.default_rng(123)
    R+=.1*free_ring(rng,R.shape,BETA,2*MASS_AU/3)
    r+=.1*free_ring(rng,r.shape,BETA,MASS_AU/2)
    provider=Harmonic()
    action,*_=remaining_action(provider,R,r,BETA,centers,k)
    coordinate=production_coordinate();cv=CVStack([coordinate])
    for i,q in enumerate(cartesian(R,r)):
        bias=CompositeBias(UmbrellaBias(cv,[centers[i]],[[k[i]]]),
                           FluxMetricBias(coordinate,np.full(3,MASS_AU),BETA))
        expected=BETA*(provider.evaluate(q).energy.sum()+bias.energy_force(q)[0])/len(q)
        assert action[i]==pytest.approx(expected,abs=3e-10)


def test_global_moves_cannot_jump_to_disconnected_coordinate_component():
    R=np.array([[10.,0.,0.]])
    r=np.array([[100.,0.,0.]])
    _,_,valid,_=coordinate_metric(R,r,BETA)
    assert not valid[0]


def test_real_pes_checkpoint_resume_is_exact(tmp_path):
    from pathlib import Path
    library=Path(__file__).resolve().parents[1]/'benchmarks/h3/build/libbkmp2.so'
    if not library.is_file(): pytest.skip('optional built BKMP2 library unavailable')
    options=dict(library=library,beads=8,seed=7621,warmup=10,production=40,
        centers=np.array([.1,.8,.98,1.02]),stiffness=np.array([2.,10.,30.,30.]),
        diagnostic_interval=3,checkpoint_interval=17)
    uninterrupted=tmp_path/'full';restarted=tmp_path/'restart'
    run(uninterrupted,**options)
    run(restarted,stop_after=23,**options)
    run(restarted,resume=True,**options)
    with np.load(uninterrupted/'checkpoint.npz',allow_pickle=False) as a, \
         np.load(restarted/'checkpoint.npz',allow_pickle=False) as b:
        assert set(a.files)==set(b.files)
        for key in a.files: np.testing.assert_array_equal(a[key],b[key],err_msg=key)
