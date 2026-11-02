"""Independent BCOCB square-root, noise, invariant measure and boundary tests."""
import importlib.util
import math
from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

torch = pytest.importorskip("torch")
from pyrpmd.torch_backend import TorchDynamics, TorchPES, TorchState, TorchRPCTerm, TorchSplitPotential

_spec = importlib.util.spec_from_file_location(
    "bcocb_validation", Path(__file__).parents[1]/"benchmarks"/"torch_bcocb_validation.py")
v = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v)


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def engine(P=8, beta=8., dt=.1, mass=1., stiffness=1., **kwargs):
    return TorchDynamics(TorchPES(v.obs.Oscillator(stiffness, 0.)), [mass], beta, P, dt,
                         integrator=v.BCOCB, **kwargs)


@pytest.mark.parametrize("P,k", [(1,0), (5,1), (8,4), (16,7)])
@pytest.mark.parametrize("mass", [1.,7.])
@pytest.mark.parametrize("dt", [.02,.5])
def test_principal_square_root_not_half_timestep_cayley(P,k,mass,dt):
    beta=2.; w=v.obs.frequencies(P,beta)[k]
    basis=np.cos(2*np.pi*k*np.arange(P)/P);basis/=np.linalg.norm(basis)
    s=v.obs.state_1d(np.array([basis,np.zeros(P)]),np.array([np.zeros(P),basis]))
    d=engine(P,beta,dt,mass)
    q,p=d._free(s.q,s.p,half=True)
    actual=np.stack([q[:,:,0,0].numpy()@basis,p[:,:,0,0].numpy()@basis])
    _,_,_,expected=v.independent_step(w,dt,mass)
    np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-12)
    A=np.array([[0.,1/mass],[-mass*w*w,0.]])
    full=np.linalg.solve(np.eye(2)-dt*A/2,np.eye(2)+dt*A/2)
    np.testing.assert_allclose(actual@actual,full,atol=3e-12,rtol=3e-12)
    if w>0 and dt==.5:
        wrong=np.linalg.solve(np.eye(2)-dt*A/4,np.eye(2)+dt*A/4)
        assert np.max(abs(actual-wrong))>1e-3


@pytest.mark.parametrize("P,k", [(1,0),(5,1),(8,4),(16,7)])
@pytest.mark.parametrize("mass", [1.,7.])
@pytest.mark.parametrize("gamma_centroid", [.3,2.7])
def test_exact_conditional_mean_and_noise_location(monkeypatch,P,k,mass,gamma_centroid):
    dt,beta,stiffness=.17,3.,2.
    basis=np.cos(2*np.pi*k*np.arange(P)/P);basis/=np.linalg.norm(basis)
    q=np.zeros((3,P));p=q.copy();q[0]=basis;p[1]=basis
    state=v.obs.state_1d(q,p)
    noise=torch.zeros_like(state.p);noise[2,:,0,0]=torch.from_numpy(basis)
    calls=[]
    def fixed_noise(shape,**kwargs):
        calls.append(tuple(shape)); return noise.clone()
    monkeypatch.setattr(torch,"randn",fixed_noise)
    d=engine(P,beta,dt,mass,stiffness,thermostat="pile-l",gamma_centroid=gamma_centroid)
    d.step(state)
    w=v.obs.frequencies(P,beta)[k]
    L,Q,G,_=v.independent_step(w,dt,mass,stiffness,P,beta,gamma_centroid if k==0 else 2*w)
    actual=np.stack([state.q[:,:,0,0].numpy()@basis,state.p[:,:,0,0].numpy()@basis])
    np.testing.assert_allclose(actual,np.column_stack([L,G]),atol=3e-12,rtol=3e-12)
    assert len(calls)==1 # exactly one full central OU step
    assert d.force_evaluations=={"fast":2,"slow":0}


@pytest.mark.parametrize("mass,stiffness", [(1.,.5),(1.,2.),(7.,2.)])
@pytest.mark.parametrize("w", [0.,.7,3.,17.,100.])
@pytest.mark.parametrize("dt", [.025,.17,.5])
@pytest.mark.parametrize("gamma", [.3,2.7])
def test_published_configuration_and_momentum_covariance(mass,stiffness,w,dt,gamma):
    L,Q,_,_=v.independent_step(w,dt,mass,stiffness,gamma=gamma)
    assert max(abs(np.linalg.eigvals(L)))<1
    actual=solve_discrete_lyapunov(L,Q)
    expected=v.paper_covariance(w,dt,mass,stiffness)
    scale=np.sqrt(np.outer(np.diag(expected),np.diag(expected)))
    assert np.max(abs(actual-expected)/scale)<2e-9
    assert np.max(abs(expected-L@expected@L.T-Q)/scale)<2e-11
    assert expected[1,1]<mass # P/beta=1; stored momenta are NOT Maxwellian


@pytest.mark.parametrize("P", [1,5,16])
@pytest.mark.parametrize("mass", [1.,7.])
def test_cold_start_full_covariance_not_preloaded_equilibrium(P,mass):
    n,beta,dt,steps=4096,2.,.2,40
    finite=[]
    for k,w in enumerate(v.obs.frequencies(P,beta)):
        L,Q,_,_=v.independent_step(w,dt,mass,1.,P,beta,1. if k==0 else 2*w)
        C=solve_discrete_lyapunov(L,Q);Ln=np.linalg.matrix_power(L,steps)
        finite.append(C-Ln@C@Ln.T)
    zero=np.zeros((n,P));s=v.obs.state_1d(zero,zero)
    d=engine(P,beta,dt,mass,thermostat="pile-l",seed=431);d.propagate(s,steps)
    q,p=s.q[:,:,0,0].numpy(),s.p[:,:,0,0].numpy()
    for values,indices in ((np.mean(q*q,1),(0,0)),(np.mean(p*p,1),(1,1)),(np.mean(q*p,1),(0,1))):
        stat=v.obs.summary(values); target=np.mean([C[indices] for C in finite])
        assert abs(stat["mean"]-target)<6*stat["se"]


@pytest.mark.parametrize("P", [1,5,16])
def test_no_thermostat_is_exactly_existing_bcb_with_no_rng_consumption(P):
    q,p=v.obs.harmonic_iid(4,P,3.,np.random.default_rng(3))
    a,b=v.obs.state_1d(q,p),v.obs.state_1d(q,p)
    old=TorchDynamics(TorchPES(v.obs.Oscillator()),[1.],3.,P,.02,integrator="cayley-bcb")
    new=TorchDynamics(TorchPES(v.obs.Oscillator()),[1.],3.,P,.02,integrator=v.BCOCB)
    rng=new.generator.get_state().clone()
    old.propagate(a,50);new.propagate(b,50)
    assert torch.equal(a.q,b.q) and torch.equal(a.p,b.p)
    assert torch.equal(rng,new.generator.get_state())
    assert a.step==b.step==50 and a.time==b.time


@pytest.mark.parametrize("thermostat", ["pile-l","trpmd"])
def test_block_partition_and_force_work(thermostat):
    q,p=v.obs.harmonic_iid(4,5,3.,np.random.default_rng(3))
    a,b=v.obs.state_1d(q,p),v.obs.state_1d(q,p)
    d1=engine(5,3.,thermostat=thermostat,seed=22)
    d2=engine(5,3.,thermostat=thermostat,seed=22)
    d1.propagate(a,7)
    for _ in range(7):d2.step(b)
    assert torch.equal(a.q,b.q) and torch.equal(a.p,b.p)
    assert torch.equal(d1.generator.get_state(),d2.generator.get_state())
    assert d1.force_evaluations["fast"]==8 and d2.force_evaluations["fast"]==14


def test_trpmd_centroid_and_resonant_internal_mode():
    P,n,beta,dt=8,2048,8.,math.pi/2
    q=np.zeros((n,P));p=np.ones_like(q)*.2;s=v.obs.state_1d(q,p)
    d=engine(P,beta,dt,stiffness=0.,thermostat="trpmd",seed=1729);d.propagate(s,100)
    np.testing.assert_allclose(s.q[:,:,0,0].mean(1),100*dt*.2,atol=3e-12,rtol=0)
    np.testing.assert_allclose(s.p[:,:,0,0].mean(1),.2,atol=3e-14,rtol=0)
    values=torch.fft.rfft(s.q,dim=1,norm="ortho")[:,-1,0,0].real.square().numpy()
    stat=v.obs.summary(values);assert abs(stat["mean"]-.25)<6*stat["se"]


@pytest.mark.parametrize("failure", ["raise","nan"])
def test_middle_noise_and_state_roll_back_on_failure(failure):
    class Fail(v.obs.Oscillator):
        def __init__(self):super().__init__(1.,0.);self.calls=0;self.enabled=True
        def forward(self,q):
            self.calls+=1
            if self.enabled and self.calls==3:
                if failure=="raise":raise RuntimeError("failure after central OU")
                return q.sum((1,2))*float("nan"),q*float("nan")
            return super().forward(q)
    model=Fail();d=TorchDynamics(TorchPES(model),[1.],3.,5,.1,integrator=v.BCOCB,thermostat="pile-l",seed=9)
    q,p=v.obs.harmonic_iid(4,5,3.,np.random.default_rng(3));s=v.obs.state_1d(q,p)
    q0,p0=s.q.clone(),s.p.clone();r0=d.generator.get_state().clone()
    with pytest.raises((RuntimeError,FloatingPointError)):d.propagate(s,3)
    assert torch.equal(s.q,q0) and torch.equal(s.p,p0) and s.step==0 and s.time==0
    assert torch.equal(d.generator.get_state(),r0)
    model.enabled=False;d.propagate(s,3)
    ref=engine(5,3.,.1,thermostat="pile-l",seed=9);expected=v.obs.state_1d(q,p);ref.propagate(expected,3)
    assert torch.equal(s.q,expected.q) and torch.equal(s.p,expected.p)


def test_no_implicit_host_array_or_scalar_reads_in_hot_loop(monkeypatch):
    d=engine(5,3.,thermostat="pile-l");s=v.obs.state_1d(np.zeros((2,5)),np.zeros((2,5)))
    item=torch.Tensor.item;reads=[]
    def counted(x,*a,**kw):reads.append(x.numel());return item(x,*a,**kw)
    def forbidden(*a,**kw):raise AssertionError("hot loop host transfer")
    monkeypatch.setattr(torch.Tensor,"item",counted)
    for name in ("cpu","numpy","tolist"):monkeypatch.setattr(torch.Tensor,name,forbidden)
    d.propagate(s,9)
    assert reads==[1]


def test_rpc_and_split_forces_are_all_in_the_bcocb_kicks(monkeypatch):
    P,beta,dt,C=5,3.,.17,3
    fast=TorchRPCTerm(TorchPES(v.obs.Oscillator(.9,0.)),P)
    slow=TorchRPCTerm(TorchPES(v.obs.Oscillator(.1,0.)),P,C)
    d=TorchDynamics(TorchSplitPotential([fast],[slow]),[1.],beta,P,dt,integrator=v.BCOCB,thermostat="pile-l")
    from scipy.linalg import sqrtm,expm
    I=np.eye(P);spring=(P/beta)**2*(2*I-np.roll(I,1,1)-np.roll(I,-1,1))
    # Independently constructed odd-Pc Fourier projection.
    T=(1+2*np.cos(2*np.pi*(np.arange(C)[:,None]/C-np.arange(P)[None,:]/P)))/P
    K=.9*I+.1*(P/C)*T.T@T
    A=np.block([[0*I,I],[-spring,0*I]]);eye=np.eye(2*P)
    S=np.real_if_close(sqrtm(np.linalg.solve(eye-dt*A/2,eye+dt*A/2)))
    B=np.block([[I,0*I],[-dt*K/2,I]])
    eig,U=np.linalg.eigh(spring);gamma=2*np.sqrt(np.maximum(eig,0));gamma[0]=1.
    O=np.block([[I,0*I],[0*I,(U*np.exp(-dt*gamma))@U.T]])
    phase=np.eye(2*P);s=v.obs.state_1d(phase[:,:P],phase[:,P:])
    monkeypatch.setattr(torch,"randn",lambda shape,**kw:torch.zeros(shape,dtype=kw["dtype"],device=kw["device"]))
    d.step(s)
    actual=np.concatenate([s.q[:,:,0,0],s.p[:,:,0,0]],axis=1).T
    np.testing.assert_allclose(actual,B@S@O@S@B,atol=1e-12,rtol=1e-12)
    assert d.force_evaluations=={"fast":2,"slow":2}


def test_bcocb_accepts_explicit_mts_but_rejects_mutation():
    terms=TorchSplitPotential([TorchRPCTerm(TorchPES(v.obs.Oscillator(.9,0.)),5)],
                             [TorchRPCTerm(TorchPES(v.obs.Oscillator(.1,0.)),5,1)])
    mts = TorchDynamics(terms,[1.],3.,5,.1,inner_steps=2,integrator=v.BCOCB)
    assert mts.inner_steps == 2
    d=engine(5,3.,thermostat="pile-l");d.integrator="cayley-bcb"
    s=v.obs.state_1d(np.zeros((2,5)),np.zeros((2,5)))
    with pytest.raises(ValueError,match="immutable"):d.step(s)


@pytest.mark.parametrize("P", [8,32,128,256])
def test_no_harmonic_primitive_kinetic_divergence(P):
    beta,dt=8.,.5;omega=v.obs.frequencies(P,beta)
    variances=np.array([solve_discrete_lyapunov(*v.independent_step(w,dt,P=P,beta=beta)[:2])[0,0] for w in omega])
    target=P/beta/(1+omega**2)
    np.testing.assert_allclose(variances,target,atol=3e-12,rtol=3e-10)
    kinetic=P/(2*beta)-np.dot(omega**2,variances)/(2*P)
    exact=P/(2*beta)-np.dot(omega**2,target)/(2*P)
    assert kinetic==pytest.approx(exact,abs=3e-9)
    assert kinetic>0


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA hardware/runtime unavailable")
def test_cuda_bcocb_stays_on_device_and_samples():
    n,P=4096,8
    d=TorchDynamics(TorchPES(v.obs.Oscillator(1.,0.),device="cuda"),[1.],8.,P,.2,
                    integrator=v.BCOCB,thermostat="pile-l",seed=31)
    s=v.obs.state_1d(np.zeros((n,P)),np.zeros((n,P)),"cuda")
    d.propagate(s,500)
    assert s.q.is_cuda and s.p.is_cuda and all(x.is_cuda for x in d._half_abc)
    values=s.q[:,:,0,0].square().mean(1).cpu().numpy();stat=v.obs.summary(values)
    target=np.mean(P/8/(1+v.obs.frequencies(P,8.)**2))
    assert abs(stat["mean"]-target)<6*stat["se"]


@pytest.mark.parametrize("P", [1,5,8])
def test_coupled_unequal_mass_drift_noise_and_configurational_invariant(monkeypatch,P):
    from scipy.linalg import sqrtm
    beta,dt=3.,.3
    masses=np.array([1.,7.]);K=np.array([[1.,-.3],[-.3,2.]])
    class Coupled(torch.nn.Module):
        metadata={"batch_independent":True}
        def forward(self,q):
            x=q[:,:,0];k=torch.as_tensor(K,dtype=q.dtype,device=q.device)
            g=x@k;f=torch.zeros_like(q);f[:,:,0]=-g
            return .5*(x*g).sum(1),f
    D=2*P;I=np.eye(D);Z=0*I;mass=np.diag(np.tile(masses,P));inv=np.linalg.inv(mass)
    ring=(P/beta)**2*(2*np.eye(P)-np.roll(np.eye(P),1,1)-np.roll(np.eye(P),-1,1))
    spring=np.kron(ring,np.eye(2))@mass;physical=np.kron(np.eye(P),K)
    A=np.block([[Z,inv],[-spring,Z]]);eye=np.eye(2*D)
    S=np.real_if_close(sqrtm(np.linalg.solve(eye-dt*A/2,eye+dt*A/2)))
    B=np.block([[I,Z],[-dt*physical/2,I]])
    w2,U=np.linalg.eigh(ring);gamma=2*np.sqrt(np.maximum(w2,0));gamma[0]=1.
    decay=np.kron((U*np.exp(-dt*gamma))@U.T,np.eye(2))
    O=np.block([[I,Z],[Z,decay]])
    noise=np.kron((U*np.sqrt(-np.expm1(-2*dt*gamma)))@U.T,np.diag(np.sqrt(masses*P/beta)))
    L=B@S@O@S@B;G=B@S@np.vstack([Z,noise]);Q=G@G.T
    phase=np.zeros((3*D,2*D));phase[:2*D]=np.eye(2*D)
    q=torch.zeros((3*D,P,2,3),dtype=torch.float64);p=q.clone()
    q[:,:,:,0]=torch.from_numpy(phase[:,:D].reshape(3*D,P,2))
    p[:,:,:,0]=torch.from_numpy(phase[:,D:].reshape(3*D,P,2))
    noise_beads=torch.zeros_like(q);noise_beads[2*D:,:,:,0]=torch.from_numpy(I.reshape(D,P,2))
    monkeypatch.setattr(torch,"randn",lambda shape,**kwargs:noise_beads.clone())
    d=TorchDynamics(TorchPES(Coupled()),masses,beta,P,dt,integrator=v.BCOCB,thermostat="pile-l")
    s=TorchState(q,p);d.step(s)
    actual=np.concatenate([s.q[:,:,:,0].reshape(3*D,D),s.p[:,:,:,0].reshape(3*D,D)],axis=1).T
    np.testing.assert_allclose(actual,np.column_stack([L,G]),atol=2e-12,rtol=2e-12)
    stationary=solve_discrete_lyapunov(L,Q)
    np.testing.assert_allclose(stationary[:D,:D],(P/beta)*np.linalg.inv(spring+physical),atol=2e-11,rtol=2e-11)


def test_harmonic_stability_limit_is_not_removed_by_bcocb():
    # Explicitly preserve the physical-force step limit; finite checks cannot
    # substitute for convergence of an unknown model's local curvature.
    for w in (0.,1.,50.):
        L,*_=v.independent_step(w,2.1,gamma=1.)
        assert max(abs(np.linalg.eigvals(L)))>1
    with pytest.raises(ValueError,match="stability"):
        v.paper_covariance(3.,2.)


def test_stochastic_linear_mean_and_covariance_converge_at_second_order():
    from scipy.linalg import expm
    w,m,k,gamma,P,beta,T=3.,2.,1.3,1.7,8,3.,1.
    A=np.array([[0.,1/m],[-(m*w*w+k),-gamma]])
    continuous=np.diag([P/(beta*(k+m*w*w)),m*P/beta])
    exact_mean=expm(T*A);exact_cov=continuous-exact_mean@continuous@exact_mean.T
    errors=[]
    for dt in (.05,.025,.0125):
        L,Q,_,_=v.independent_step(w,dt,m,k,P,beta,gamma)
        C=solve_discrete_lyapunov(L,Q);Ln=np.linalg.matrix_power(L,round(T/dt))
        covariance=C-Ln@C@Ln.T
        errors.append((np.linalg.norm(Ln-exact_mean),np.linalg.norm(covariance-exact_cov)))
    ratios=np.array(errors[:-1])/np.array(errors[1:])
    assert np.all((ratios>3.8)&(ratios<4.2))
