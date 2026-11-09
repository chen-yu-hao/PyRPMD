"""Fast implementation checks, not a declaration of physical convergence."""
from pathlib import Path
import sys
import numpy as np
import pytest
pytest.importorskip('numba')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/anharmonic_quantum'))
sys.path.insert(0,str(ROOT/'src/pyrpmd/analysis'))
from adaptive_quantum import select_active_subspace,passive_embed,spectral_diagnostics,matched_rate_defect
from models import Model,force_one,energy_one
from sampling import hmc_samples,fermi_samples
from dynamics import tangent_force,response_blocks
from quantum import morse_quantum,eckart_scattering
import numba
numba.set_num_threads(2)

@pytest.mark.parametrize('kind',['morse','fermi','eckart'])
@pytest.mark.parametrize('dim',[1,2,3])
def test_forces_are_energy_gradients(kind,dim):
    m=Model(kind,dim);z=np.random.default_rng(4).normal(scale=.3,size=(7,dim));eps=1e-6
    for j in range(dim):
        dz=np.zeros(dim);dz[j]=eps
        np.testing.assert_allclose(m.force(z)[:,j],-(m.potential(z+dz)-m.potential(z-dz))/(2*eps),atol=2e-8)
    f=np.empty_like(z);force_one(z,m.code,m.coupling,0.,f)
    np.testing.assert_allclose(f,m.force(z),atol=1e-13)
    np.testing.assert_allclose(energy_one(z,np.zeros_like(z),m.code,m.coupling,0),m.potential(z).sum(),atol=1e-12)

def test_hessian_and_tangent():
    m=Model('fermi');q=np.random.default_rng(1).normal(size=(8,3));dq=np.random.default_rng(2).normal(size=q.shape)
    df=np.empty_like(q);tangent_force(q,dq,1,.15,0.,df)
    np.testing.assert_allclose(df,-np.einsum('nij,nj->ni',m.hessian(q),dq),atol=2e-8)

def test_morse_analytic_ground_states():
    _,records=morse_quantum(n=160)
    assert max(x['ground_energy_error'] for x in records)<1e-8

def test_eckart_analytic_and_flux_conservation():
    E=np.linspace(.1,2.0,8);_,T,meta=eckart_scattering(E,1,dq=.05)
    exact=np.sinh(np.pi*np.sqrt(2*E))**2/(np.sinh(np.pi*np.sqrt(2*E))**2+np.cosh(np.pi*np.sqrt(1.75))**2)
    assert abs(T-exact).max()<.001
    assert meta['max_unitarity_defect']<1e-7

def test_hmc_constraint():
    q,meta=hmc_samples(Model('eckart'),8,16,40,2,2,832,centroid=.3)
    np.testing.assert_allclose(q[...,0].mean(-1),.3,atol=1e-12)
    assert meta['independent_samples'] is False

def test_fermi_conditional_sampling_identity():
    q,_,_=fermi_samples(Model('fermi'),16,4096,833)
    # Gaussian conditional x integration implies <x>=-g/omega_x^2 <y^2>.
    assert abs(q[...,0].mean()+.35/4*(q[...,1]**2).mean())<.015

def test_rotated_even_nonlinear_selector():
    m=Model('fermi');z=np.random.default_rng(22).normal(size=(1000,3));f=m.force(z);h=m.hessian(z)
    R=np.linalg.qr(np.random.default_rng(27).normal(size=(3,3)))[0]
    sel=select_active_subspace(z@R.T,f@R.T,np.einsum('ia,nab,jb->nij',R,h,R),R[:,0],nonlinear_fraction=.999)
    assert sel.selected_dimension==2
    P=R.T@sel.basis@sel.basis.T@R
    np.testing.assert_allclose(P,np.diag([1,1,0]),atol=1e-10)

def test_morse_budget_not_silently_truncated():
    m=Model('morse');z=np.random.default_rng(31).normal(scale=.4,size=(1000,3))
    r=select_active_subspace(z,m.force(z),m.hessian(z),np.ones(3),max_dimension=2)
    assert r.status=='dimension_budget_exceeded'
    assert r.selected_dimension==3

def test_linear_harmonic_tangent_limit():
    # y=0 is an invariant trajectory: the x tangent is exactly harmonic.
    q=np.zeros((32,8,2));p=np.zeros_like(q)
    r=response_blocks(q,p,1,0.,4.,.005,400,4).mean(0)[:,0]
    t=np.arange(len(r))*.02
    np.testing.assert_allclose(r,.5*np.sin(2*t),atol=1.1e-5)

@pytest.mark.parametrize('bad',[0,True,1.5])
def test_invalid_dimension_budget(bad):
    m=Model('fermi');q=np.random.default_rng(33).normal(size=(20,3))
    with pytest.raises(ValueError):select_active_subspace(q,m.force(q),m.hessian(q),[1,0,0],max_dimension=bad)
