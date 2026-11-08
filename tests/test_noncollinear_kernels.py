"""Optional Numba benchmark tests; core molecular API tests do not require Numba."""
from pathlib import Path
import sys
import numpy as np
import pytest
pytest.importorskip('numba')
pytestmark=pytest.mark.campaign
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'benchmarks/noncollinear_quantum'))
from leps import MU_IN,MU_BOND,C_IN,MB,MH,MC,MASSES,radial_one
from dynamics import jacobi_force,jacobi_energy,angular_momentum,propagate
from cayley import setup,free_step
from hyperspherical import helicity_kinetics,sine_fd_kinetic,angular_setup

def test_jacobi_force_finite_difference():
    q=np.array([[[.1,.5,2.7],[1.2,-.5,6.2]]]);f=np.empty_like(q);assert jacobi_force(q,f,0.)
    h=1e-5
    for k in range(6):
        d=np.eye(6)[k].reshape(q.shape)*h
        diff=(jacobi_energy(q+d,np.zeros_like(q),0)-jacobi_energy(q-d,np.zeros_like(q),0))/(2*h)
        assert diff==pytest.approx(-f.ravel()[k],abs=3e-10,rel=3e-7)


@pytest.mark.parametrize('P',[1,2,3,8,32])
def test_cayley_spring_map(P):
    rng=np.random.default_rng(3);m=np.array([2.,3.]);q=rng.normal(size=(P,2,3));p=rng.normal(size=q.shape);w2=3.;dt=.13
    def e():return np.sum(p*p/(2*m[None,:,None]))+.5*w2*np.sum(m[None,:,None]*(q-np.roll(q,1,axis=0))**2)
    e0=e();q0=q.copy();p0=p.copy();plan=setup(P,dt,w2);free_step(q,p,m,dt,w2,plan)
    assert e()==pytest.approx(e0,rel=1e-12)
    free_step(q,p,m,-dt,w2,setup(P,-dt,w2));np.testing.assert_allclose(q,q0,atol=1e-12);np.testing.assert_allclose(p,p0,atol=1e-12)


@pytest.mark.parametrize('J,jmax',[(0,4),(1,5),(3,6),(7,8)])
def test_helicity_coriolis_eigenvalues(J,jmax):
    for parity in (-1,1):
        pairs,j2,l2=helicity_kinetics(J,jmax,parity)
        for j in range(jmax+1):
            ids=[i for i,(K,jj) in enumerate(pairs) if j==jj]
            expected=[l*(l+1) for l in range(abs(J-j),J+j+1) if (-1)**(j+l)==parity]
            np.testing.assert_allclose(np.linalg.eigvalsh(l2[np.ix_(ids,ids)]),expected,atol=1e-10)


def test_hyperangular_dirichlet_kinetic():
    n=64;length=np.pi/2;K=sine_fd_kinetic(n,length).toarray()
    np.testing.assert_allclose(K,K.T,atol=1e-13)
    eigen=np.linalg.eigvalsh(K)
    np.testing.assert_allclose(eigen[:5],(np.arange(1,6)*np.pi/length)**2,rtol=1e-7)


def test_angular_quadrature_and_physical_three_coordinates():
    a,u,w,Y,idx,T,pairs=angular_setup(nalpha=20,alpha_max=.4,jmax=6,J=2,parity=1,nquad=12)
    assert len(a)==20 and len(u)==12 and T.shape==(20*len(pairs),20*len(pairs))
    for B in Y.values():np.testing.assert_allclose(B.T@B,np.eye(B.shape[1]),atol=1e-12)



def test_generic_cartesian_matches_independent_jacobi_map():
    from pyrpmd.analysis.molecular_collision import propagate_cartesian
    from pilot_selection import cartesian_force
    rng=np.random.default_rng(763);P=4;beta=800.;dt=.5
    q=np.broadcast_to(np.array([[.1,.3,2.7],[1.3,.4,7.0]]),(2,P,2,3)).copy()
    q+=rng.normal(size=q.shape)*.005
    p=rng.normal(size=q.shape)*.3
    def convert(q,p):
        x=np.zeros(q.shape[:2]+(3,3));v=np.zeros_like(x)
        x[:,:,0]=-MH/(MB+MH)*q[:,:,0];x[:,:,1]=MB/(MB+MH)*q[:,:,0];x[:,:,2]=q[:,:,1]
        x-=(x*MASSES[None,None,:,None]).sum(2,keepdims=True)/MASSES.sum()
        v[:,:,0]=-p[:,:,0]-MB/(MB+MH)*p[:,:,1]
        v[:,:,1]=p[:,:,0]-MH/(MB+MH)*p[:,:,1];v[:,:,2]=p[:,:,1]
        return x,v
    def pes(frames):
        vals=[cartesian_force(x) for x in frames]
        return np.array([v[0] for v in vals]),np.array([v[1] for v in vals])
    x,v=convert(q,p)
    generic=propagate_cartesian(x,v,MASSES,beta=beta,dt=dt,steps=25,potential=pes,
        classifier=lambda q,v:np.full(len(q),-1),check_stride=5)
    for i in range(len(q)):
        f=np.empty_like(q[i]);assert jacobi_force(q[i],f,(P/beta)**2)
        for step in range(25):
            p[i]+=.5*dt*f;q[i]+=dt*p[i]/np.array([MU_BOND,MU_IN])[None,:,None]
            assert jacobi_force(q[i],f,(P/beta)**2);p[i]+=.5*dt*f
    xx,vv=convert(q,p)
    np.testing.assert_allclose(generic['positions'],xx,atol=5e-13)
    np.testing.assert_allclose(generic['momenta'],vv,atol=5e-12)


def test_scalar_leps_matches_vector_energy_at_noncollinear_geometries():
    from leps import energy
    rng=np.random.default_rng(553)
    for i in range(40):
        x=rng.normal(size=(3,3))*2+np.array([[0,0,0],[0,0,3],[1,0,6]])
        d=np.array([np.linalg.norm(x[a]-x[b]) for a,b in [(0,1),(1,2),(0,2)]])
        assert radial_one(d)[0]==pytest.approx(energy(d),abs=1e-12)
