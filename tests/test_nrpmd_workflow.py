"""NRPMD collisions preserve electronic state, weights and chain provenance."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pyrpmd.dynamics import State
from pyrpmd.nonadiabatic import DiabaticResult, NRPMDState, NRPMDDynamics
from pyrpmd.workflows import drpmd


MATRIX_SOURCE = '''
import numpy as np
from pyrpmd.nonadiabatic import DiabaticResult
class Matrix:
    def evaluate(self,q):
        d=q[:,1]-q[:,0]
        r=np.linalg.norm(d,axis=1)
        z=np.maximum(1-(r/2)**2,0)
        c=0.2*z*z
        v=np.zeros((len(q),2,2)); v[:,0,1]=v[:,1,0]=c
        g=np.zeros((len(q),2,2,2,3))
        dc=-0.2*z[:,None]*d
        g[:,0,1,1]=g[:,1,0,1]=dc
        g[:,0,1,0]=g[:,1,0,0]=-dc
        return DiabaticResult(v,g)
def make(system=None): return Matrix()
'''


def setup_case(tmp_path, *, beads=1):
    path = tmp_path / 'matrix.py'
    path.write_text(MATRIX_SOURCE)
    system = {'symbols':['H','H'], 'masses':np.ones(2),
              'geometry':np.array([[0.,0.,-2.],[0.,0.,2.]]), 'fragments':[[0],[1]]}
    config = {
        'task':'drpmd','beta':1., 'temperature_K':1/drpmd.KB,'beads':beads,
        'dt_au':.05, 'seed':17,'native':False, 'system':{'separation_au':4.},
        'potential':{'provider':'diabatic_python','factory':str(path)+':make','derivative':'gradient'},
        'nonadiabatic':{'nstates':2,'initial_state':0,'state_labels':['g','e'],
                       'asymptotic_coupling_tolerance':1e-8},
        'sampling':{'parents':2,'equilibration_steps':4,'interval_steps':2,'andersen_interval':2},
        'collision':{'trajectories':4,'bmax_au':.2,'max_steps':200,'sampling':'fixed_energy','energy_au':1.},
        'outcomes':{'bond_cutoffs_au':{'H-H':.1},'channels':{},'exit_distance_au':4.,
                    'check_interval':1,'persistence_steps':2},
        'output':{'trajectory_stride':3,'checkpoint_steps':10},
        'parallel':{'backend':'serial'},
    }
    return config,system


def assert_same_states(a,b):
    for i in range(4):
        with np.load(a/f'trajectories/trajectory-{i:08d}.npz') as x, np.load(b/f'trajectories/trajectory-{i:08d}.npz') as y:
            for key in ['final_q','final_p','final_mapping_q','final_mapping_p','electronic_populations','energy_per_bead']:
                np.testing.assert_array_equal(x[key],y[key])


def test_coupled_collision_and_complete_resume(tmp_path):
    config,system=setup_case(tmp_path)
    first=drpmd.run_drpmd(config,system,tmp_path/'run')
    assert first['counts']['nonreactive']==4
    assert first['nonadiabatic']['final_estimates_available']
    assert any(abs(r['electronic_populations'][1]) > .01 for r in first['records'])
    for r in first['records']:
        assert sum(r['electronic_populations']) == pytest.approx(1.)
    config['resume']=True
    again=drpmd.run_drpmd(config,system,tmp_path/'run')
    assert first['records']==again['records']
    assert first['nonadiabatic']==again['nonadiabatic']
    assert (tmp_path/'run/results/electronic_channels.csv').exists()
    with np.load(tmp_path/'run/states/initial-00000000.npz') as data:
        assert data['mapping_q'].shape==(1,2)


@pytest.mark.parametrize('interruption', [30, 200])
def test_interrupted_resume_preserves_electrons_and_parent_clock(tmp_path, interruption):
    config,system=setup_case(tmp_path)
    # Terminal checkpoints also cover crash between final classification and result JSON.
    expected=drpmd.run_drpmd(config,system,tmp_path/'full')
    save=drpmd._atomic_npz
    tripped=False
    def interrupt(path,metadata,**arrays):
        nonlocal tripped
        save(path,metadata,**arrays)
        if not tripped and metadata['format']=='PyRPMD.drpmd.trajectory-checkpoint.v1' and (
            metadata['done']==interruption or (interruption==200 and metadata['status']=='nonreactive')):
            tripped=True
            if interruption == 30:
                separation = np.linalg.norm(arrays['q'][:,1] - arrays['q'][:,0], axis=1)
                assert np.any(separation < 2.), "Restart test must interrupt inside the coupling region"
            raise KeyboardInterrupt('persisted')
    with patch.object(drpmd,'_atomic_npz',side_effect=interrupt),pytest.raises(KeyboardInterrupt):
        drpmd.run_drpmd(config,system,tmp_path/'resumed')
    config['resume']=True
    config['dynamics']={'block_steps':7}
    actual=drpmd.run_drpmd(config,system,tmp_path/'resumed')
    assert actual['nonadiabatic']==expected['nonadiabatic']
    assert actual['parent_thermal_steps']==expected['parent_thermal_steps']
    assert_same_states(tmp_path/'full',tmp_path/'resumed')


def test_process_collisions_match_serial(tmp_path):
    config,system=setup_case(tmp_path,beads=3)
    expected=drpmd.run_drpmd(config,system,tmp_path/'serial')
    config['parallel']={'backend':'process','trajectory_workers':2}
    actual=drpmd.run_drpmd(config,system,tmp_path/'process')
    assert actual['nonadiabatic']==expected['nonadiabatic']
    assert_same_states(tmp_path/'serial',tmp_path/'process')


def records():
    return [
        {'status':'product','channel':'p','parent_id':0,'electronic_populations':[1.2,-.2], 'launch':{'speed_au':2.}},
        {'status':'nonreactive','channel':None,'parent_id':0,'electronic_populations':[.4,.6], 'launch':{'speed_au':3.}},
        {'status':'product','channel':'p','parent_id':1,'electronic_populations':[-.1,1.1], 'launch':{'speed_au':4.}},
    ]


@pytest.mark.parametrize('sampling',['fixed_energy','thermal_flux','maxwell'])
def test_signed_joint_statistics_keep_weights_and_shared_covariance(tmp_path,sampling):
    config,system=setup_case(tmp_path)
    config['collision']['sampling']=sampling
    summary=drpmd.summarize(records(),config,system)
    na=summary['nonadiabatic']
    assert na['channels']['p']['electronic_sum_error']==pytest.approx(0,abs=1e-20)
    assert np.asarray(na['parent_cluster_bootstrap_covariance']).shape==(6,6)
    point = 'cross_section_bohr2' if sampling=='fixed_energy' else 'rate_cm3_molecule_s'
    assert sum(na['channels']['p'][point])==pytest.approx(summary['channels']['p'][point])
    # Repeated aggregate and chemical channel estimates share identical draws.
    cov=np.asarray(na['parent_cluster_bootstrap_covariance'])
    np.testing.assert_allclose(cov[:2,:2],cov[-2:,-2:])
    assert na['channels']['p']['parent_cluster_bootstrap95'][0][0]<0


def test_unknown_withholds_final_signed_results_and_preserves_denominator(tmp_path):
    config,system=setup_case(tmp_path)
    rec=records(); rec[1]['status']='unresolved'
    summary=drpmd.summarize(rec,config,system)['nonadiabatic']
    assert not summary['final_estimates_available']
    assert summary['channels']['p']['cross_section_bohr2'] is None
    assert summary['parent_cluster_bootstrap_covariance'] is None
    assert sum(summary['channels']['p']['signed_observed_contributions'])==pytest.approx(2*np.pi*.2**2/3)


@pytest.mark.parametrize('name',['nonreactive','total_reaction'])
def test_reserved_product_labels_fail_explicitly(tmp_path,name):
    config,system=setup_case(tmp_path)
    rec=records(); rec[0]['channel']=name
    with pytest.raises(ValueError,match='reserved'):
        drpmd.summarize(rec,config,system)


def test_direct_api_rejects_unsupported_backend(tmp_path):
    config,system=setup_case(tmp_path)
    config['parallel']={'backend':'process','mode':'bead','bead_workers':2}
    with pytest.raises(ValueError,match='trajectory-process'):
        drpmd.run_drpmd(config,system,tmp_path/'bad')


class PersistentCoupling:
    def evaluate(self,q):
        x=q[:,0,0]
        t=np.tanh(5*x)
        v=np.zeros((len(q),2,2)); v[:,0,1]=v[:,1,0]=.1*(1+t)
        g=np.zeros((len(q),2,2,*q.shape[1:])); g[:,0,1,0,0]=g[:,1,0,0,0]=.5*(1-t*t)
        return DiabaticResult(v,g)


def direct_case(beads=1):
    q=np.zeros((beads,1,3)); q[:,0,0]=-2
    p=np.zeros_like(q); p[:,0,0]=1
    state=NRPMDState.from_nuclear(State(q,p),2,0,np.random.default_rng(9))
    dyn=NRPMDDynamics(PersistentCoupling(),[1.],1.,beads,.05,2,native=False)
    config={'beads':beads,'dt_au':.05,'nonadiabatic':{'nstates':2,'asymptotic_coupling_tolerance':1e-8},
            'collision':{'max_steps':100},'outcomes':{'check_interval':1,'persistence_steps':0},
            'output':{'checkpoint_steps':10,'trajectory_stride':1}}
    return dyn,state,config


def test_geometric_exit_waits_for_electronic_decoupling(tmp_path):
    dyn,state,config=direct_case()
    actual=drpmd._trajectory(dyn,state,lambda s:'p' if s.q[0,0,0]>0 else None,
                              config,tmp_path,0,0,{},'test',False)
    assert actual['status']=='unresolved'
    assert actual['steps']==100
    assert dyn.asymptotic_coupling(state)>1e-3


def test_launch_coupling_checks_every_bead(tmp_path):
    dyn,state,config=direct_case(beads=2)
    state.q[1,0,0]=2
    actual=drpmd._trajectory(dyn,state,None,config,tmp_path,0,0,{},'test',False)
    assert actual['status']=='failed'
    assert actual['steps']==0
    assert 'asymptotically decoupled' in actual['reason']


def test_classifier_failure_rolls_back_electronic_and_nuclear_step(tmp_path):
    dyn,state,config=direct_case()
    expected=state.copy(); dyn.step(expected)
    def classifier(s):
        if s.step==2:
            raise RuntimeError('classifier unavailable')
    actual=drpmd._trajectory(dyn,state,classifier,config,tmp_path,0,0,{},'test',False)
    assert actual['status']=='failed' and actual['steps']==1
    with np.load(tmp_path/'trajectories/trajectory-00000000.npz') as data:
        for key in ['q','p','mapping_q','mapping_p']:
            np.testing.assert_array_equal(data['final_'+key],getattr(expected,key))


def test_checkpoint_requires_electronic_state(tmp_path):
    dyn,state,config=direct_case()
    save=drpmd._atomic_npz
    def interrupt(path,metadata,**arrays):
        if metadata['format']=='PyRPMD.drpmd.trajectory-checkpoint.v1':
            arrays.pop('mapping_p')
        save(path,metadata,**arrays)
        raise KeyboardInterrupt('persisted corrupt checkpoint')
    with patch.object(drpmd,'_atomic_npz',side_effect=interrupt),pytest.raises(KeyboardInterrupt):
        drpmd._trajectory(dyn,state,None,config,tmp_path,0,0,{},'test',False)
    with pytest.raises(ValueError,match='mapping state'):
        drpmd._trajectory(dyn,state,None,config,tmp_path,0,0,{},'test',True)


def test_parent_uses_selected_diagonal_and_children_focus_selected_state(tmp_path):
    config,system=setup_case(tmp_path)
    config['nonadiabatic']['initial_state']=1
    path=tmp_path/'matrix.py'
    path.write_text(MATRIX_SOURCE.replace('v[:,0,1]=v[:,1,0]=c',
                                         'v[:,0,1]=v[:,1,0]=c; v[:,0,0]=.2; v[:,1,1]=.7'))
    constructor=drpmd.Dynamics
    sampled=[]
    def capture(potential,*args,**kwargs):
        sampled.append(potential.evaluate(system['geometry'][None]).energy.copy())
        return constructor(potential,*args,**kwargs)
    with patch.object(drpmd,'Dynamics',side_effect=capture):
        drpmd.run_drpmd(config,system,tmp_path/'selected')
    assert len(sampled)==2
    for value in sampled:
        np.testing.assert_allclose(value,[.7])
    for i in range(4):
        with np.load(tmp_path/f'selected/states/initial-{i:08d}.npz') as data:
            populations=.5*(data['mapping_q']**2+data['mapping_p']**2-1)
            np.testing.assert_allclose(populations,[[0.,1.]],atol=1e-14)


@pytest.mark.parametrize('corruption',['clock','trace'])
def test_checkpoint_rejects_inconsistent_clock_or_trace(tmp_path,corruption):
    dyn,state,config=direct_case()
    save=drpmd._atomic_npz
    def interrupt(path,metadata,**arrays):
        if metadata['format']=='PyRPMD.drpmd.trajectory-checkpoint.v1':
            if corruption=='clock':
                metadata=dict(metadata,done=metadata['done']+1)
            else:
                arrays['trace_populations']=arrays['trace_populations'][:-1]
        save(path,metadata,**arrays)
        raise KeyboardInterrupt('persisted corrupt checkpoint')
    with patch.object(drpmd,'_atomic_npz',side_effect=interrupt),pytest.raises(KeyboardInterrupt):
        drpmd._trajectory(dyn,state,None,config,tmp_path,0,0,{},'test',False)
    with pytest.raises(ValueError,match='clock|traces'):
        drpmd._trajectory(dyn,state,None,config,tmp_path,0,0,{},'test',True)
