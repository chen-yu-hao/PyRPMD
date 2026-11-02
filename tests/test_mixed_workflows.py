"""Selected splitting, phase handoff, and checkpoint identity end to end."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.dynamics import Dynamics, State


@pytest.mark.parametrize('method', ['bcocb','baoab'])
@pytest.mark.parametrize('failure', ['burn','launch','save'])
def test_fragment_pile_parent_recovery_and_launch_refresh(tmp_path, monkeypatch, method, failure):
    from test_drpmd import config, system
    from pyrpmd.workflows import drpmd
    from pyrpmd.workflows.drpmd_persistence import _load_npz
    c=config();c.update(beads=5,dt_au=.003)
    c['outcomes']['bond_cutoffs_au']={'H-H':1.0}
    c['dynamics']={'integrator':'normal-mode-bab','loop_backend':'python'}
    c['sampling'].update(thermostat='pile-l',integrator=method,gamma_centroid=.7,equilibration_steps=9,interval_steps=5)
    c['output']['checkpoint_steps']=4
    sys=system();sys.update(symbols=['H','H','H'],masses=np.array([1.,2.,3.]),
                           geometry=np.array([[-.2,0,-2.],[.1,0,-2.],[0,0,2.]]),fragments=[[0,1],[2]])
    def task(path):return dict(config=c,system=sys,parent_id=0,trajectory_ids=[0,2,5],physics_hash='mixed-pile-test',output=str(path),sample_only=True)
    drpmd._parent_task(task(tmp_path/'reference'))
    original_launch=drpmd.launch_collision;original_save=drpmd._atomic_npz;original_thermalize=Dynamics.thermalize
    calls=[]
    def launch(*args,**kwargs):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('injected launch')
        return original_launch(*args,**kwargs)
    def save(path,metadata,**arrays):
        if Path(path).name=='parent-000000.npz' and metadata['next_index']==1:
            raise OSError('injected save')
        return original_save(path,metadata,**arrays)
    def burn(self,state,*args,**kwargs):
        result=original_thermalize(self,state,*args,**kwargs)
        if state.step==8:raise RuntimeError('injected burn')
        return result
    with monkeypatch.context() as p:
        if failure=='launch':p.setattr(drpmd,'launch_collision',launch)
        if failure=='save':p.setattr(drpmd,'_atomic_npz',save)
        if failure=='burn':p.setattr(Dynamics,'thermalize',burn)
        with pytest.raises((RuntimeError,OSError),match='injected'):
            drpmd._parent_task(task(tmp_path/'resumed'))
    c['resume']=True;c['output']['checkpoint_steps']=3
    drpmd._parent_task(task(tmp_path/'resumed'))
    for trajectory_id in [0,2,5]:
        name=f'states/initial-{trajectory_id:08d}.npz'
        a,x=_load_npz(tmp_path/'reference'/name);b,y=_load_npz(tmp_path/'resumed'/name)
        assert a==b
        for key in x:np.testing.assert_array_equal(x[key],y[key])
        launch_meta=a['launch']
        assert launch_meta['momentum_rng_stream']==[17,6,trajectory_id]
        assert launch_meta['parent_sampling_integrator']==method
        # Recover the Maxwell draw before the collision COM velocity/rotation.
        rng=np.random.default_rng(np.random.SeedSequence([17,6,trajectory_id]))
        momenta=rng.normal(size=x['thermal_p'].shape)*np.sqrt(sys['masses'][None,:,None]*5)
        parent=State(x['thermal_q'],momenta)
        for fragment in sys['fragments']:
            ids=list(fragment);m=sys['masses'][ids]
            parent.p[:,ids]-=m[None,:,None]*parent.p[:,ids].mean(0).sum(0)[None,None,:]/m.sum()
            np.testing.assert_allclose(x['thermal_p'][:,ids].sum((0,1)),0,atol=2e-13)
        expected,_=original_launch(parent,sys['masses'],sys['fragments'],1.,.5,4.,
                                  np.random.default_rng(np.random.SeedSequence([17,2,trajectory_id])))
        np.testing.assert_allclose(expected.p,x['p'],atol=3e-14,rtol=2e-14)
        assert not np.array_equal(x['thermal_p'],parent.p)
    a,x=_load_npz(tmp_path/'reference/checkpoints/parent-000000.npz')
    b,y=_load_npz(tmp_path/'resumed/checkpoints/parent-000000.npz')
    assert a['operators']==b['operators'] and a['rng']==b['rng']
    for key in x:np.testing.assert_array_equal(x[key],y[key])


@pytest.mark.parametrize('method',['bcocb','baoab'])
def test_fragment_pile_standard_frontend_and_actual_scattering(tmp_path,method):
    from test_checkpoint import collision_job
    from pyrpmd.input import JobSpec
    from pyrpmd.runner import run,resume_run
    data=collision_job();data['beads']=3
    data['dynamics']={'integrator':'normal-mode-bab','timestep':'0.01 fs'}
    data['sampling'].update(thermostat='pile-l',integrator=method,gamma_centroid='0.1 fs^-1',equilibration_steps=5,interval_steps=2)
    data['collision']['trajectories']=2
    resolved,system=JobSpec(data).resolve()
    assert resolved['sampling']['integrator']==method
    result=run(data,output=tmp_path/'run')
    replay=resume_run(tmp_path/'run')
    assert result['counts']==replay['counts'] and result['trajectories']==2


@pytest.mark.parametrize('base,method', [('normal-mode-bab','bcocb'),('cayley-bcb','baoab')])
@pytest.mark.parametrize('failure_at_boundary',[False,True])
def test_mixed_diffusion_phase_draw_is_once_and_transactional(tmp_path,monkeypatch,base,method,failure_at_boundary):
    from test_materials_workflows import config,system,task,assert_same_stream
    import pyrpmd.workflows.materials as module
    from pyrpmd.potentials.builtin import HarmonicPotential
    c=config();c['dynamics']['integrator']=base;c['sampling']['integrator']=method
    reference=module._sample_replica(task(c,tmp_path/'reference'))
    original=module.quantum_observables
    def failure(state,*args,**kwargs):
        if state.step==5:raise RuntimeError('phase observation failure')
        return original(state,*args,**kwargs)
    if failure_at_boundary:
        with monkeypatch.context() as p:
            p.setattr(module,'quantum_observables',failure)
            with pytest.raises(RuntimeError,match='phase observation'):
                module._sample_replica(task(c,tmp_path/'actual'))
        prefix=module._read_stream(c,system(),tmp_path/'actual',0)
        assert prefix['elapsed']==4
    actual=module._sample_replica(task(c,tmp_path/'actual'))
    assert_same_stream(reference,actual)
    rng=np.random.default_rng(np.random.SeedSequence([412,1301,0]))
    d=Dynamics(HarmonicPotential(.4),system().masses,2.,3,.03,seed=rng,native=False,integrator=base,loop_backend='python')
    state=State(np.repeat(system().geometry[None],3,axis=0),d.sample_momenta())
    for _ in range(5):d.sample_step(state,gamma_centroid=.6,method=method)
    state.p[:]=d.sample_momenta();d.propagate(state,30)
    np.testing.assert_array_equal(actual['state'].q,state.q)
    np.testing.assert_array_equal(actual['state'].p,state.p)
    assert actual['rng']==rng.bit_generator.state
    assert_same_stream(actual,module._sample_replica(task(c,tmp_path/'actual')))


@pytest.mark.parametrize('base,method',[('normal-mode-bab','bcocb'),('cayley-bcb','baoab')])
@pytest.mark.parametrize('piston',[None,'bzp'])
def test_mixed_native_pimd_actual_input_journal_and_restart(tmp_path,monkeypatch,base,method,piston):
    pytest.importorskip('pyrpmd._thermal')
    from test_native_pimd_followup import config,read_records
    from pyrpmd.runner import run,resume_run
    from pyrpmd.workflows import native_pimd as workflow
    c=config(barostat=piston);c['dynamics']['integrator']=base;c['sampling']['integrator']=method
    reference=run(c,output=tmp_path/'reference')
    assert reference['sampling_splitting'].lower()==(('bzp-' if piston else '')+method)
    original=workflow.write_json
    def fail(path,data):
        if Path(path).name=='journal.json' and data['blocks'][-1]['end']==29:
            raise OSError('injected journal failure')
        return original(path,data)
    with monkeypatch.context() as p:
        p.setattr(workflow,'write_json',fail)
        with pytest.raises(OSError,match='journal failure'):run(c,output=tmp_path/'resumed')
    actual=resume_run(tmp_path/'resumed')
    np.testing.assert_array_equal(read_records(reference['files']['journals'][0]),read_records(actual['files']['journals'][0]))
    assert reference['estimators']==actual['estimators']
    assert resume_run(tmp_path/'resumed')['estimators']==actual['estimators']


@pytest.mark.parametrize('method',['bcocb','baoab'])
def test_mixed_recrossing_trpmd_selection_and_identity(tmp_path,monkeypatch,method):
    from test_trpmd_rates import child_task
    from pyrpmd.workflows import rate
    task=child_task(tmp_path);task['config']['dynamics']={'integrator':'normal-mode-bab'}
    task['config']['recrossing']['integrator']=method
    calls=[];original=Dynamics.trpmd_step
    def record(self,state,*a,**k):calls.append((self.integrator_name,k['method']));return original(self,state,*a,**k)
    with monkeypatch.context() as p:
        p.setattr(Dynamics,'trpmd_step',record);result=rate._child_task(task)
    assert len(calls)==24 and set(calls)=={('normal-mode-bab',method)}
    replay=rate._child_task(task);np.testing.assert_array_equal(replay['numerator'],result['numerator'])
    task['config']['recrossing']['integrator']='baoab' if method=='bcocb' else 'bcocb'
    with pytest.raises(ValueError,match='checkpoint physics'):rate._child_task(task)


@pytest.mark.parametrize('method',['rpmd','trpmd'])
@pytest.mark.parametrize('failure_step',[9,10,11,24])
def test_mixed_spectroscopy_refresh_and_real_time_replay(tmp_path,monkeypatch,method,failure_step):
    from test_spectroscopy_workflow import job
    import pyrpmd.workflows.spectroscopy as w
    c,system=job(tmp_path,method);c['sampling']['integrator']='bcocb';c['sampling']['replicas']=1
    c['output']['checkpoint_steps']=5
    if method=='trpmd':c['spectroscopy']['integrator']='bcocb'
    reference=w.run_spectroscopy(c,system,tmp_path/'reference')
    original=Dynamics.sample_step if failure_step<=10 else (Dynamics.trpmd_step if method=='trpmd' else Dynamics.step)
    name='sample_step' if failure_step<=10 else ('trpmd_step' if method=='trpmd' else 'step')
    def fail(self,state,*args,**kwargs):
        result=original(self,state,*args,**kwargs)
        if state.step==failure_step:raise RuntimeError('injected spectroscopy failure')
        return result
    with monkeypatch.context() as p:
        p.setattr(Dynamics,name,fail)
        with pytest.raises(RuntimeError,match='spectroscopy failure'):
            w.run_spectroscopy(c,system,tmp_path/'resumed')
    actual=w.run_spectroscopy(c,system,tmp_path/'resumed')
    for key in ('q','p','rng','dipole','polarizability','sample_steps'):
        with load_checkpoint(reference['files']['checkpoints'][0]) as a,load_checkpoint(actual['files']['checkpoints'][0]) as b:
            np.testing.assert_array_equal(a[key],b[key])
    options=w.resolve_spectroscopy(c,system)
    rng=np.random.default_rng(np.random.SeedSequence([57,2503,0]))
    from pyrpmd.potentials import HarmonicPotential
    sampler,production=w._engines(c,system,HarmonicPotential(.3),options,rng)
    state=State(np.repeat(system.geometry[None],8,axis=0),sampler.sample_momenta())
    for _ in range(10):sampler.sample_step(state,method='bcocb',gamma_centroid=.7)
    state.p[:]=production.sample_momenta()
    for _ in range(80):
        if method=='trpmd':production.trpmd_step(state,method='bcocb')
        else:production.step(state)
    with load_checkpoint(actual['files']['checkpoints'][0]) as saved:
        np.testing.assert_array_equal(saved['p'],state.p);np.testing.assert_array_equal(saved['q'],state.q)
        assert json.loads(str(saved['rng']))==rng.bit_generator.state
    changed=copy.deepcopy(c);changed['sampling']['integrator']='baoab'
    with pytest.raises(ValueError,match='physics'):w.reanalyze_spectroscopy(changed,system,tmp_path/'resumed')


def test_mixed_spectroscopy_frontend_keeps_pacmd_mass_contract(tmp_path):
    from test_spectroscopy_workflow import raw_job
    from pyrpmd.spectroscopy_input import resolve_spectroscopy_job
    raw=raw_job(tmp_path);raw['sampling']['integrator']='bcocb'
    with pytest.raises(ValueError,match='PA-CMD'):resolve_spectroscopy_job(raw,base_directory=tmp_path)
    raw['spectroscopy'].update(method='trpmd',integrator='bcocb');raw['spectroscopy'].pop('target_frequency')
    c,s=resolve_spectroscopy_job(raw,base_directory=tmp_path)
    assert c['sampling']['integrator']=='bcocb' and c['spectroscopy']['integrator']=='bcocb'
    c['spectroscopy']['method']='rpmd'
    from pyrpmd.workflows.spectroscopy import resolve_spectroscopy
    with pytest.raises(ValueError,match='only to TRPMD'):resolve_spectroscopy(c,s)
