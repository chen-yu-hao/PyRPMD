"""Mixed real-time/sampling contexts, constrained PILE, and middle BZP NPT."""
from pathlib import Path
import copy
import importlib.util
import json
import math
import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.engine import ForceSplitting, PropagationPlan
from pyrpmd.potentials import FreePotential, HarmonicPotential, PESResult
from pyrpmd.potentials.contraction import SplitPotential

spec = importlib.util.spec_from_file_location('mixed_reference', Path(__file__).parents[1] / 'benchmarks/mixed_sampling_validation.py')
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)

PAIRS = [('normal-mode-bab','bcocb','cayley-bcb'), ('cayley-bcb','baoab','normal-mode-bab')]


def initial(P=5):
    rng = np.random.default_rng(42)
    return State(rng.normal(scale=.1,size=(P,3,3)),rng.normal(scale=.2,size=(P,3,3)))


@pytest.mark.parametrize('base,method,target', PAIRS)
@pytest.mark.parametrize('P', [1,3,4,8,17])
@pytest.mark.parametrize('native', [False,True])
def test_selected_map_matches_dedicated_context_and_does_not_replace_nve(base, method, target, P, native):
    if native: pytest.importorskip('pyrpmd._native')
    kwargs=dict(masses=[1.,2.,5.], beta=3., nbeads=P, dt=.025, seed=37, native=native)
    d=Dynamics(HarmonicPotential(.6),integrator=base,**kwargs)
    reference=Dynamics(HarmonicPotential(.6),integrator=target,**kwargs)
    x=initial(P);y=x.copy()
    for _ in range(12):
        d.sample_step(x,method=method);reference.sample_step(y)
    np.testing.assert_array_equal(x.q,y.q);np.testing.assert_array_equal(x.p,y.p)
    assert d.rng.bit_generator.state==reference.rng.bit_generator.state
    assert d.integrator_name==base and d.integrator.scheme==base
    z=x.copy(); nve=Dynamics(HarmonicPotential(.6),integrator=base,**kwargs)
    d.step(x);nve.step(z)
    np.testing.assert_array_equal(x.q,z.q);np.testing.assert_array_equal(x.p,z.p)
    assert len(d._sampling_kernels)==1


@pytest.mark.parametrize('base,method,target', PAIRS)
@pytest.mark.parametrize('P', [1,5,8])
@pytest.mark.parametrize('scale', [0.,.6])
def test_mixed_real_time_trpmd_matches_actual_map_including_zero_friction(base, method, target, P, scale):
    d=Dynamics(HarmonicPotential(.4),[1.,2.,5.],4.,P,.03,integrator=base,native=False,seed=23)
    r=Dynamics(HarmonicPotential(.4),[1.,2.,5.],4.,P,.03,integrator=target,native=False,seed=23)
    a=initial(P);b=a.copy();rng=copy.deepcopy(d.rng.bit_generator.state)
    for _ in range(8):
        d.trpmd_step(a,method=method,friction_scale=scale)
        r.trpmd_step(b,friction_scale=scale)
    np.testing.assert_allclose(a.q,b.q,atol=2e-15,rtol=2e-14)
    np.testing.assert_allclose(a.p,b.p,atol=2e-15,rtol=2e-14)
    if scale==0 or P==1:
        assert d.rng.bit_generator.state==rng
    # Internal-mode noise does not directly thermostat the centroid.
    np.testing.assert_allclose(a.q.mean(0),b.q.mean(0),atol=2e-15)


@pytest.mark.parametrize('method', ['bcocb','baoab'])
@pytest.mark.parametrize('P', [1,3,4,8])
@pytest.mark.parametrize('native', [False,True])
def test_fragment_pile_against_independent_mass_metric_projection(method, P, native):
    if native: pytest.importorskip('pyrpmd._native')
    masses=np.array([1.,2.,5.]);beta,dt,k,gamma=3.,.03,.6,.7
    d=Dynamics(HarmonicPotential(k),masses,beta,P,dt,integrator='normal-mode-bab',native=native,seed=17)
    state=initial(P);fragments=[[0,1],[2]]
    data=d._fragment_data(state,fragments)
    data=[(ix,m,total,np.zeros(3)) for ix,m,total,_ in data]
    noise=np.random.default_rng(17).normal(size=state.p.shape)
    q,p,projection=ref.projected_pile_step(state.q,state.p,masses,beta,dt,k,fragments,noise,method,gamma)
    d.sample_step(state,method=method,gamma_centroid=gamma,fragment_data=data)
    np.testing.assert_allclose(state.q,q,atol=2e-14,rtol=1e-12)
    np.testing.assert_allclose(state.p,p,atol=2e-14,rtol=1e-12)
    phase=np.r_[state.q.ravel(),state.p.ravel()]
    np.testing.assert_allclose(projection@phase,phase,atol=2e-14,rtol=0)


@pytest.mark.parametrize('method', ['bcocb','baoab'])
def test_fragment_pile_block_partition_and_restart_share_one_fixed_anchor(method):
    masses=np.array([1.,2.,5.]);fragments=[[0,1],[2]]
    reference=np.array([[-2.,0.,.2],[-1.8,0.,-.1],[2.,.1,.4]])
    def dynamics(): return Dynamics(HarmonicPotential(.2),masses,3.,5,.03,integrator='normal-mode-bab',seed=17,native=False)
    args=dict(thermostat='pile-l',method=method,gamma_centroid=.7,fragments=fragments,fragment_reference=reference,resample=False)
    d=dynamics();r=dynamics();a=initial();b=a.copy()
    d.thermalize(a,15,**args)
    r.thermalize(b,7,**args);saved=r.operator_state_dict()
    resumed=dynamics();resumed.load_operator_state_dict(saved)
    resumed.thermalize(b,8,**args)
    np.testing.assert_array_equal(a.q,b.q);np.testing.assert_array_equal(a.p,b.p)
    assert d.operator_state_dict()==resumed.operator_state_dict()
    for ix in fragments:
        expected=np.average(reference[ix],weights=masses[ix],axis=0)
        np.testing.assert_allclose(np.sum(a.q[:,ix]*masses[None,ix,None],axis=(0,1))/(5*masses[ix].sum()),expected,atol=1e-14)
        np.testing.assert_allclose(a.p[:,ix].sum((0,1)),0,atol=1e-13)


@pytest.mark.parametrize('method', ['bcocb','baoab'])
def test_fragment_force_failure_restores_state_rng_and_clock(method):
    class Fail(HarmonicPotential):
        def __init__(self):super().__init__(.2);self.calls=0;self.failure=False
        def evaluate(self,q):
            self.calls+=1
            if self.failure and self.calls%2==0: raise RuntimeError('fragment endpoint failure')
            return super().evaluate(q)
    pes=Fail();d=Dynamics(pes,[1.,2.,5.],3.,5,.04,native=False,integrator='normal-mode-bab')
    state=initial();data=d._fragment_data(state,[[0,1],[2]])
    d.sample_step(state,method=method,fragment_data=data)
    before=state.copy();rng=copy.deepcopy(d.rng.bit_generator.state);pes.failure=True
    with pytest.raises(RuntimeError,match='fragment endpoint'): d.sample_step(state,method=method,fragment_data=data)
    np.testing.assert_array_equal(state.q,before.q);np.testing.assert_array_equal(state.p,before.p)
    assert (state.step,state.time)==(before.step,before.time) and d.rng.bit_generator.state==rng


def test_paired_legacy_operator_checkpoint_key_is_not_reinterpreted():
    pytest.importorskip('pyrpmd._thermal')
    d=Dynamics(HarmonicPotential(.3),[1.,2.,5.],3.,5,.04,native=False,seed=8)
    a=initial();spec={'mode':'pile-l','gamma_centroid':.7}
    d.sample_step(a,thermostat=spec)
    saved=d.operator_state_dict();key=json.dumps(spec,sort_keys=True)
    assert list(saved['sampling'])==[key]
    r=Dynamics(HarmonicPotential(.3),[1.,2.,5.],3.,5,.04,native=False,seed=9)
    r.load_operator_state_dict(saved);b=a.copy()
    d.sample_step(a,thermostat=spec)
    r.sample_step(b,thermostat=spec,method='bcocb')
    np.testing.assert_array_equal(a.q,b.q);np.testing.assert_array_equal(a.p,b.p)
    assert d.operator_state_dict()==r.operator_state_dict()


@pytest.mark.parametrize('method', ['bcocb','baoab'])
@pytest.mark.parametrize('M', [2,3,5])
def test_cpu_split_composition_means_against_independent_matrix(method, M, monkeypatch):
    pytest.importorskip('pyrpmd._thermal')
    from pyrpmd.engine import _Composition
    P=5;split=SplitPotential(HarmonicPotential(.9),HarmonicPotential(.1),P,3)
    d=Dynamics(split,[1.7],3.,P,.14,native=False,integrator='normal-mode-bab',
               thermostat='pile-l',sampling_integrator=method,force_splitting=ForceSplitting(M))
    durations=[]
    def mean_bath(self,q,p,h,step):
        U,labels=ref.real_modes(P);g=4*P/3.*np.sin(np.pi*np.array([x[1] for x in labels])/P);g[0]=1.
        p[:]=np.einsum('ij,jna->ina',(U*np.exp(-g*h))@U.T,p);durations.append(h)
    monkeypatch.setattr(_Composition,'_bath',mean_bath)
    states=[]
    for basis in np.eye(2*P):
        q=np.zeros((P,1,3));p=np.zeros_like(q);q[:,0,0]=basis[:P];p[:,0,0]=basis[P:]
        state=State(q,p);d.step(state);states.append(np.r_[state.q[:,0,0],state.p[:,0,0]])
    L,_,_=ref.mts_gaussian_map(P,3.,1.7,.14,M,contracted=3,method=method)
    np.testing.assert_allclose(np.array(states).T,L,atol=2e-13,rtol=3e-13)
    assert len(durations)==2*P*M and set(durations)=={d.dt/M}
    assert d.force_evaluations=={'fast':2*P*(M+1),'slow':4*P}


@pytest.mark.parametrize('base,method,target', PAIRS)
def test_mixed_bzp_python_matches_dedicated_sampler_and_rolls_back_helpers(base, method, target):
    from test_barostat import make
    b,d,s,system,obj=make(friction=.5)
    other_b,other_d,other,_,_=make(friction=.5)
    def dyn(provider, scheme):return Dynamics(provider,system.masses,4.,4,.02,integrator=scheme,native=False,seed=91)
    d=dyn(d.potential,base);other_d=dyn(other_d.potential,target)
    for _ in range(12):
        b.step(s,d,method=method);other_b.step(other,other_d)
    np.testing.assert_array_equal(s.q,other.q);np.testing.assert_array_equal(s.p,other.p)
    assert b.snapshot()==other_b.snapshot()
    before=s.copy();saved=d.operator_state_dict();piston=b.snapshot()
    obj.fail_at=obj.calls+3
    with pytest.raises(RuntimeError,match='injected'): b.step(s,d,method=method)
    np.testing.assert_array_equal(s.q,before.q);np.testing.assert_array_equal(s.p,before.p)
    assert b.snapshot()==piston and d.operator_state_dict()==saved


@pytest.mark.parametrize('method', ['bcocb','baoab'])
@pytest.mark.parametrize('P', [1,3,8])
def test_native_middle_bzp_matches_python_piston_sequence_deterministically(method,P):
    pytest.importorskip('pyrpmd._thermal')
    from pyrpmd.native_ensemble import NativeEnsemble
    from test_barostat import make
    b,d,state,system,_=make(P,dt=.005,free=True)
    b.momentum=0
    target='cayley-bcb' if method=='bcocb' else 'normal-mode-bab'
    d=Dynamics(d.potential,d.masses,4.,P,.005,integrator=target,native=False)
    e=NativeEnsemble(state.q,d.masses,4.,.005,p=state.p,cell=system.cell,
                     thermostat={'mode':'pile-l','gamma_centroid':0.,'pile_scale':0.},
                     # Native piston inertia is 3*N*P/beta*tau^2; legacy BZP uses
                     # 3*N/beta*tau^2. Match inertia, not the nominal time constant.
                     barostat={'mode':'bzp','pressure':.02,'tau':2./math.sqrt(P),'gamma':0.},
                     integrator='normal-mode-bab',sampling_integrator=method)
    for _ in range(40): b.step(state,d,thermostat=None)
    e.run(40)
    result=e.state()
    np.testing.assert_allclose(result['q'],state.q,atol=3e-13,rtol=2e-13)
    np.testing.assert_allclose(result['p'],state.p,atol=3e-13,rtol=2e-13)
    np.testing.assert_allclose(result['cell'],b.cell,atol=3e-13,rtol=2e-13)
    assert result['cell_p'][0,0]==pytest.approx(b.momentum,abs=1e-12)
    assert e.splitting=='BZP-'+method.upper()


@pytest.mark.parametrize('method',['bcocb','baoab'])
def test_native_middle_bzp_checkpoint_fused_parity_and_stale_extension_guard(method,tmp_path,monkeypatch):
    native=pytest.importorskip('pyrpmd._thermal')
    from pyrpmd.native_ensemble import NativeEnsemble
    options=dict(q=np.random.default_rng(1).normal(size=(5,2,3))*.1,masses=[1.,2.],beta=3.,dt=.015,
                 cell=np.eye(3)*3,thermostat='pile-l',barostat={'mode':'bzp','pressure':.03,'tau':2.,'gamma':.5},
                 integrator='normal-mode-bab',sampling_integrator=method,seed=17,
                 potential={'kind':'periodic_cosine','heights':[.1,.2,.3]})
    a=NativeEnsemble(**options);b=NativeEnsemble(**options);unfused=NativeEnsemble(**options,fuse_modes=False)
    expected=a.run(33,stride=3)
    first=b.run(13,stride=3);b.save(tmp_path/'middle.npz')
    resumed=NativeEnsemble(**{**options,'seed':81});resumed.load(tmp_path/'middle.npz')
    second=resumed.run(20,stride=3)
    np.testing.assert_array_equal(np.r_[first['samples'],second['samples']],expected['samples'])
    assert a.state()['digest']==resumed.state()['digest']
    unfused.run(33)
    np.testing.assert_allclose(a.state()['q'],unfused.state()['q'],atol=2e-13,rtol=2e-13)
    np.testing.assert_allclose(a.state()['p'],unfused.state()['p'],atol=2e-13,rtol=2e-13)
    assert a.state()['rng']==unfused.state()['rng']
    legacy=NativeEnsemble(**{k:v for k,v in options.items() if k!='sampling_integrator'})
    with pytest.raises(ValueError,match='physics/backend'): legacy.load(tmp_path/'middle.npz')
    monkeypatch.setattr(native,'MIDDLE_BZP_API',0)
    with pytest.raises(RuntimeError,match='rebuild'):NativeEnsemble(**options)


def test_middle_pressure_map_rejects_unvalidated_piston_and_duplicate_piston():
    from test_barostat import make
    for mode in ['mttk','scr']:
        with pytest.raises(ValueError,match='isotropic BZP'):
            PropagationPlan('normal-mode-bab','pile-l',{'mode':mode,'pressure':.02,**({'compressibility':1.} if mode=='scr' else {})},sampling_integrator='bcocb')
    b,d,state,_,_=make()
    attached=Dynamics(d.potential,d.masses,4.,4,.02,native=False,barostat=b)
    with pytest.raises(ValueError,match='second piston'): b.step(state,attached)


@pytest.mark.parametrize('thermostat', ['pile-l','trpmd'])
def test_configured_mixed_bath_checkpoint_restart_and_wrong_plan_rejection(thermostat):
    pytest.importorskip('pyrpmd._thermal')
    def dyn(method): return Dynamics(HarmonicPotential(.5),[1.,2.,5.],3.,5,.04,
                                    integrator='normal-mode-bab',thermostat=thermostat,
                                    sampling_integrator=method,native=False,seed=9)
    d=dyn('bcocb');state=initial();d.propagate(state,7)
    before=state.copy();saved=d.operator_state_dict();d.propagate(state,11)
    r=dyn('bcocb');r.load_operator_state_dict(saved);r.propagate(before,11)
    np.testing.assert_array_equal(state.q,before.q);np.testing.assert_array_equal(state.p,before.p)
    assert d.operator_state_dict()==r.operator_state_dict()
    with pytest.raises(ValueError,match='parameters differ'):dyn('baoab').load_operator_state_dict(saved)


@pytest.mark.parametrize('method', ['bcocb','baoab'])
def test_public_mts_adapter_selected_middle_and_atomic_failure(method):
    from pyrpmd.multitime import MTSDynamics
    class Failing(HarmonicPotential):
        fail=False
        def evaluate(self,q):
            if self.fail: raise RuntimeError('slow endpoint')
            return super().evaluate(q)
    slow=Failing(.1)
    split=SplitPotential(HarmonicPotential(.9),slow,5,3)
    kw=dict(potential=split,masses=[1.,2.,5.],beta=3.,nbeads=5,dt=.06,native=False,
            integrator='normal-mode-bab',thermostat='pile-l',sampling_integrator=method,seed=81)
    adapter=MTSDynamics(**kw,inner_steps=3)
    common=Dynamics(**kw,force_splitting=ForceSplitting(3))
    a=initial();b=a.copy();adapter.propagate(a,11);common.propagate(b,11)
    np.testing.assert_array_equal(a.q,b.q);np.testing.assert_array_equal(a.p,b.p)
    saved=adapter.operator_state_dict();before=a.copy();slow.fail=True
    with pytest.raises(RuntimeError,match='slow endpoint'):adapter.propagate(a,4)
    np.testing.assert_array_equal(a.q,before.q);np.testing.assert_array_equal(a.p,before.p)
    assert adapter.operator_state_dict()==saved


@pytest.mark.parametrize('method', ['bcocb','baoab'])
def test_native_middle_bzp_nonzero_force_virial_against_python_sequence(method):
    from pyrpmd.native_ensemble import NativeEnsemble
    from test_barostat import make
    P=3;b,d,state,system,_=make(P,dt=.003,free=False);b.momentum=0.
    _,other,_,_,obj=make(P,dt=.003,free=False)
    target='cayley-bcb' if method=='bcocb' else 'normal-mode-bab'
    d=Dynamics(d.potential,d.masses,4.,P,.003,integrator=target,native=False)
    e=NativeEnsemble(state.q,d.masses,4.,.003,p=state.p,cell=system.cell,
                     thermostat={'mode':'pile-l','gamma_centroid':0.,'pile_scale':0.},
                     barostat={'mode':'bzp','pressure':.02,'tau':2./math.sqrt(P),'gamma':0.},
                     integrator='normal-mode-bab',sampling_integrator=method,
                     potential=other.potential,allow_python=True,potential_identity='fractional-well-v1')
    for _ in range(25):b.step(state,d,thermostat=None)
    e.run(25);result=e.state()
    np.testing.assert_allclose(result['q'],state.q,atol=3e-13,rtol=2e-13)
    np.testing.assert_allclose(result['p'],state.p,atol=3e-13,rtol=2e-13)
    np.testing.assert_allclose(result['cell'],b.cell,atol=3e-13,rtol=2e-13)
    assert result['cell_p'][0,0]==pytest.approx(b.momentum,abs=1e-12)
    # Failure after the first dilation/physical kick must undo every piston,
    # provider-cell, force-cache and RNG mutation, not merely q and p.
    before=e.state();oldcell=other.potential.cell.copy()
    obj.fail_at=obj.calls+2
    with pytest.raises((RuntimeError,ValueError),match='injected'):e.run(3)
    assert e.state()['digest']==before['digest']
    np.testing.assert_array_equal(other.potential.cell,oldcell)
