"""Shared native RPMDrate geometry/flux integration; no physical rate certificate."""
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from pyrpmd.cv import RPMDrateCoordinate, Distance
from pyrpmd.system import System
from pyrpmd.potentials import PESResult
from pyrpmd.analysis.shared_rate_coordinate import SharedRateCoordinate, XiSlice
from pyrpmd.analysis.shared_quantum_flux import (
    assemble_slice_side, kubo_flux_side, MatchedFluxBlocks, apply_shared_flux_correction,
)
from pyrpmd.analysis.curvilinear_quantum import solve_curvilinear
from pyrpmd.workflows.shared_rate_quantum import (
    export_native_reference, apply_provider, run_shared_rate,
)


def shared(*, beta=5., xi_star=1., rinf=10., masses=None, bonds=None):
    q=np.array([[-1.5,0,0],[0,0,0],[1.5,0,0.]])
    m=np.full(3,2.) if masses is None else np.asarray(masses)
    c=RPMDrateCoordinate([[0],[1,2]],m,q,[[0,1]],[[1,2]],rinf)
    return SharedRateCoordinate(c,beta=beta,xi_star=xi_star,pes_identity='test-PES-v1',normalization_id='test-reference-v1'),q


def blocks(value=.6):
    ids=np.array([(i,j) for i in range(4) for j in range(6)])
    t=np.arange(1,81)*.02;den=np.linspace(.4,2.,len(ids))
    num=den[:,None]*np.full((len(ids),len(t)),value)
    return MatchedFluxBlocks('same',t,ids,num,den,'test conditional flux')


def test_native_values_surface_and_gradient_are_reused():
    s,q=shared();q[1]+=[.2,.1,.05]
    c=s._coordinate;d=s.geometry(q);v,g=c.centroid_value_gradient(q)
    assert d['xi']==v
    np.testing.assert_array_equal(d['xi_gradient'],g)
    S,gS=c.dividing_surface(1.).centroid_value_gradient(q)
    assert d['surface']==S
    np.testing.assert_array_equal(d['surface_gradient'],gS)
    assert s.side(q)==float(S>0)
    assert d['surface_positive_flux_speed']==pytest.approx(np.sqrt(np.sum(gS*gS/s.masses[:,None])/(2*np.pi*s.beta)))


def test_asymptote_zero_and_transition_one():
    s,q=shared();assert s.geometry(q)['xi']==1
    # COM(1,2)=.75: put atom 0 at .75-rinf.
    q[0,0]=.75-10
    assert s.geometry(q)['xi']==pytest.approx(0)


def test_nonlinear_centroid_not_bead_average():
    s,q=shared();beads=np.array([q,q]);beads[0,1,0]=.3;beads[1,1,0]=-.3
    assert s.centroid_geometry(beads)['xi']==1
    assert abs(np.mean([s.geometry(x)['xi'] for x in beads])-1)>1e-4


def test_translation_rotation_and_isotopic_metric():
    from scipy.spatial.transform import Rotation
    s,q=shared();r=Rotation.from_rotvec([.2,.3,-.5]).as_matrix();a=s.geometry(q);b=s.geometry(q@r+2)
    assert a['xi']==pytest.approx(b['xi']);assert a['xi_inverse_mass']==pytest.approx(b['xi_inverse_mass'])
    np.testing.assert_allclose(a['xi_gradient']@r,b['xi_gradient'],atol=1e-14)
    heavier,_=shared(masses=[4,4,4]);assert heavier.geometry(q)['xi_inverse_mass']==pytest.approx(a['xi_inverse_mass']/2)
    assert heavier.identity!=s.identity


@pytest.mark.parametrize('key,value',[('beta',6.),('xi_star',.9),('normalization_id','other'),('pes_identity','other')])
def test_identity_mutation_is_rejected(key,value):
    s,q=shared();setattr(s,key,value)
    with pytest.raises(ValueError,match='mutated'):s.geometry(q)


def test_mutating_original_coordinate_not_shared_snapshot():
    s,q=shared();c=s._coordinate
    copied=SharedRateCoordinate(c,beta=5,xi_star=1,pes_identity='p',normalization_id='n')
    old=copied.identity;c.rinf=11
    assert copied.identity==old


def test_wrong_ratio_side_cannot_be_used():
    s,q=shared(rinf=.5);q[1,0]=-.1
    # Negative denominator reverses xi ordering, but physical S keeps its sign.
    s0,_,_,s1,_,_=s._coordinate.surfaces(q)
    assert s0-s1<0
    assert s.side(q)!=float(s._coordinate.centroid_value_gradient(q)[0]>1)
    with pytest.raises(ValueError,match='positive'):s.geometry(q)


def test_branch_tie_rejected_for_quantum_chart():
    s,q=shared();c=RPMDrateCoordinate([[0],[1,2]],s.masses,q,[[0,1],[0,1]],[[1,2],[1,2]],10)
    ss=SharedRateCoordinate(c,beta=5,xi_star=1,pes_identity='p',normalization_id='n')
    with pytest.raises(ValueError,match='seam'):ss.geometry(q)


def test_auxiliary_metric_and_rank():
    s,q=shared();data=s.active_metric(q,[Distance([0,2])]);assert data['inverse_mass_block'].shape==(2,2)
    assert not data['full_kinetic_metric_available']
    with pytest.raises(ValueError,match='Dependent'):s.active_metric(q,[s._coordinate])


def test_slice_induced_mass_jacobian():
    s,q=shared();v=np.zeros_like(q);v[1,0]=-1
    path=XiSlice(s,q,v,[-.7,.7]);xi=np.array([.9,1,1.1]);xyz,m=path.map(xi)
    h=1e-5;qp,_=path.map(xi+h);qm,_=path.map(xi-h)
    np.testing.assert_allclose(m,np.sum(s.masses[None,:,None]*((qp-qm)/(2*h))**2,axis=(1,2)),rtol=1e-8)
    for x,u in zip(xyz,xi):assert s.geometry(x)['xi']==pytest.approx(u,abs=1e-12)
    with pytest.raises(ValueError):path.map([10.])


def test_two_level_flux_against_independent_matrix_trace():
    E=np.array([0.,1.3]);h=np.array([[.5,.5],[.5,.5]],complex);beta=2.7;hb=.8;t=np.linspace(0,3,101)
    actual=kubo_flux_side(E,h,t,beta=beta,hbar=hb)
    H=np.diag(E);F=1j*(H@h-h@H)/hb;p=np.exp(-beta*E);p/=p.sum()
    K=np.empty((2,2),complex)
    for i in range(2):
        for j in range(2):K[i,j]=F[i,j]*(p[i] if i==j else (p[i]-p[j])/(beta*(E[j]-E[i])))
    independent=[]
    for time in t:
        U=expm(1j*H*time/hb);independent.append(np.trace(K@U@h@U.conj().T).real)
    np.testing.assert_allclose(actual['flux_side_per_partition'],independent,atol=1e-14)
    assert actual['flux_side_per_partition'][0]==0
    assert actual['reactant_population']==pytest.approx(.5)


def test_flux_energy_offset_and_degenerate_rotation():
    E=np.array([0.,0.,1.]);h=np.ones((3,3))/3;t=np.linspace(0,4,41)
    a=kubo_flux_side(E,h,t,beta=3)
    b=kubo_flux_side(E+100,h,t,beta=3)
    U=np.array([[.6,-.8,0],[.8,.6,0],[0,0,1]])
    c=kubo_flux_side(E,U.T@h@U,t,beta=3)
    np.testing.assert_allclose(a['flux_side_per_partition'],b['flux_side_per_partition'],atol=1e-14)
    np.testing.assert_allclose(a['flux_side_per_partition'],c['flux_side_per_partition'],atol=1e-14)


@pytest.mark.parametrize('h',[np.eye(2)*2,np.array([[0,1],[0,0]]),np.eye(2)*np.nan])
def test_invalid_quantum_side(h):
    with pytest.raises(ValueError):kubo_flux_side([0,1],h,[0,1],beta=1)


def test_zero_defect_and_full_replacement_limits():
    f=blocks(.6);a=blocks(.3);q=blocks(.9)
    r=apply_shared_flux_correction(f,a,a,k_qtst_cm3_s=2,bootstrap=40)
    np.testing.assert_allclose(r['kappa_curve'],.6,atol=1e-14)
    r=apply_shared_flux_correction(f,f,q,k_qtst_cm3_s=2,bootstrap=40)
    np.testing.assert_allclose(r['kappa_curve'],.9,atol=1e-14)
    assert r['rate_estimate_cm3_s']==pytest.approx(1.8)
    assert r['rate_cm3_s'] is None and not r['physical_converged']


def test_ratio_of_flux_sums_not_average_ratios():
    f=blocks();n=f.numerator.copy();n[:12]=0;f=replace(f,numerator=n)
    r=apply_shared_flux_correction(f,f,f,k_qtst_cm3_s=1,bootstrap=40)
    assert r['kappa_estimate']==pytest.approx(n.sum(0)[0]/f.denominator.sum())
    assert abs(r['kappa_estimate']-np.mean(n[:,0]/f.denominator))>.05


@pytest.mark.parametrize('field',['identity','parent_ids','denominator','time_au'])
def test_mismatched_flux_inputs_rejected(field):
    f=blocks();change={'identity':'wrong','parent_ids':f.parent_ids[::-1],
        'denominator':f.denominator*2,'time_au':f.time_au*2}
    other=replace(f,**{field:change[field]})
    with pytest.raises(ValueError,match='mismatch'):apply_shared_flux_correction(f,f,other,k_qtst_cm3_s=1)


def test_negative_correction_not_clipped():
    f=blocks(.2);a=blocks(.8);q=blocks(.1)
    r=apply_shared_flux_correction(f,a,q,k_qtst_cm3_s=1,bootstrap=40)
    assert r['negative_rate'] and r['rate_estimate_cm3_s']==pytest.approx(-.5)


def test_finite_closed_quantum_cannot_fake_plateau():
    f=blocks(1);q=replace(f,numerator=f.denominator[:,None]*np.sin(f.time_au*10)[None])
    r=apply_shared_flux_correction(f,f,q,k_qtst_cm3_s=1,bootstrap=40)
    assert not r['plateau_detected'] and 'no_resolved_plateau' in r['unmet_gates']


def test_parent_arrays_are_owned_immutable():
    f=blocks()
    with pytest.raises(ValueError):f.numerator[0,0]=9
    with pytest.raises(ValueError):f.numerator.flags.writeable=True


def smoke_job():
    system=System(('H','H','H'),np.array([2.,2.,2.]),np.array([[-1.5,0,0],[0,0,0],[1.5,0,0.]]),((0,),(1,2)))
    config={'task':'rate','beta':5.,'temperature_K':300.,'beads':2,'dt_au':.003,'seed':17,'native':False,
        'potential':{'provider':'free'},'sampling':{'equilibration_steps':50,'production_steps':200,'sample_interval':2,'andersen_interval':20},
        'reaction':{'fragments':[[0],[1,2]],'forming_bonds':[[0,1]],'breaking_bonds':[[1,2]],'rinf':4.},
        'umbrella':{'centers':np.linspace(0,1,11).tolist(),'kappa':10.},'analysis':{'bins':30,'bootstrap':2},
        'parallel':{'backend':'serial'},'output':{'checkpoint_steps':200},
        'recrossing':{'dividing_surface':1.,'parent_replicas':2,'parent_samples':3,'equilibration_steps':20,'parent_interval_steps':5,
                     'child_pairs':1,'child_steps':40,'bootstrap':20}}
    return config,system


def test_real_native_workflow_and_noop_provider(tmp_path):
    config,system=smoke_job();captured={}
    def noop(ref):
        captured['reference']=ref
        return {'active_rp':ref.full_flux,'active_quantum':ref.full_flux,'evidence':{}}
    report=run_shared_rate(config,system,tmp_path,quantum_provider=noop,bootstrap=40)
    native=json.loads((tmp_path/'rpmd/summary.json').read_text())
    assert report['kappa_estimate']==pytest.approx(native['kappa'],rel=1e-13)
    assert report['k_qtst_cm3_s']==native['pmf']['k_qtst_cm3_s']
    assert captured['reference'].parent_snapshots.shape==(6,2,3,3)
    assert report['status']=='requires_validation' and report['rate_cm3_s'] is None
    assert (tmp_path/'rpmd/shared_coordinate.json').exists()
    changed=copy.deepcopy(config);changed['beta']*=2
    with pytest.raises(ValueError,match='identity changed'):export_native_reference(changed,system,tmp_path/'rpmd')
    with pytest.raises(ValueError,match='overwrite'):apply_provider(captured['reference'],noop,tmp_path/'quantum')


@pytest.mark.parametrize('dimension',[2,3])
def test_shared_xi_auxiliary_chart_retains_mixed_kinetic_terms(dimension):
    from pyrpmd.analysis.shared_rate_coordinate import XiAffineChart
    from pyrpmd.analysis.shared_quantum_flux import assemble_shared_side
    s,q=shared();directions=np.zeros((dimension,3,3));directions[0,1,0]=-1
    for j in range(1,dimension):directions[j,1,j]=1
    chart=XiAffineChart(s,q,directions,[-.7,.7])
    points=np.array([[.94]+[.12]*(dimension-1),[1.06]+[-.13]*(dimension-1)])
    x,J=chart._mapped(points);h=1e-6
    for j in range(dimension):
        delta=np.eye(dimension)[j]*h
        np.testing.assert_allclose((chart.map(points+delta)-chart.map(points-delta))/(2*h),J[:,j],atol=2e-8)
    for p,v in zip(x,points):assert s.geometry(p)['xi']==pytest.approx(v[0],abs=1e-12)
    G,rho=chart.geometry(points);g=np.linalg.inv(G)
    assert np.max(abs(g[:,0,1:]))>1e-4
    np.testing.assert_allclose(rho,np.sqrt(np.linalg.det(g)),rtol=1e-12)
    axes=[np.linspace(.95,1.05,7)]+[np.linspace(-.2,.2,6)]*(dimension-1)
    def potential(u):
        x=chart.map(u)
        return .5*np.sum(x[:,1]**2,axis=1)
    ref=solve_curvilinear(axes,chart.geometry,potential,states=3)
    side=assemble_shared_side(ref,axes,chart)
    np.testing.assert_allclose(side,side.conj().T,atol=1e-12)
    assert np.linalg.eigvalsh(side).min()>-1e-10
    assert np.linalg.eigvalsh(side).max()<1+1e-10


def test_side_matrix_requires_a_surface_mesh_plane():
    s,q=shared();v=np.zeros_like(q);v[1,0]=-1;path=XiSlice(s,q,v,[-.7,.7])
    nodes=np.linspace(.95,1.05,10)
    ref=solve_curvilinear([nodes],path.geometry,lambda u:np.zeros(len(u)),states=3)
    with pytest.raises(ValueError,match='interior'):assemble_slice_side(ref,nodes,path)


def test_quantum_trace_to_parent_mapping_is_explicit():
    from pyrpmd.analysis.shared_quantum_flux import quantum_traces_to_blocks
    f=blocks();t=f.time_au
    trace=kubo_flux_side([0,1],np.full((2,2),.5),t,beta=2)['flux_side_per_partition']
    traces=np.tile(trace,(len(f.parent_ids),1));scale=f.denominator*3
    q=quantum_traces_to_blocks(f,traces,scale,normalization_record='synthetic controlled-unit-test scale, not a physical bath map')
    r=apply_shared_flux_correction(f,f,q,k_qtst_cm3_s=2,bootstrap=40)
    np.testing.assert_allclose(r['kappa_curve'],3*trace)
    assert r['rate_cm3_s'] is None and 'common_positive_flux' in r['unmet_gates']
    with pytest.raises(ValueError):quantum_traces_to_blocks(f,traces,scale,normalization_record='')


def test_parent_source_tampering_detected(tmp_path):
    from pyrpmd.workflows.rate import run_rate
    config,system=smoke_job();run_rate(config,system,tmp_path)
    file=tmp_path/'recrossing.npz'
    with np.load(file) as z:values=dict(z)
    values['parent_numerator']=values['parent_numerator']+.01
    np.savez(file,**values)
    with pytest.raises(ValueError,match='complete child'):export_native_reference(config,system,tmp_path)


def test_native_config_mismatch_rejected_on_first_export(tmp_path):
    from pyrpmd.workflows.rate import run_rate
    config,system=smoke_job();run_rate(config,system,tmp_path)
    config['potential']={'provider':'harmonic','params':{'k':1.}}
    with pytest.raises(ValueError,match='physics/identity'):export_native_reference(config,system,tmp_path)


def test_nonnative_provider_does_not_accept_a_transmission_number(tmp_path):
    config,system=smoke_job()
    with pytest.raises(TypeError):run_shared_rate(config,system,tmp_path,quantum_provider=.8)


def test_quantum_core_bound_to_surface_temperature_and_pes():
    from pyrpmd.potentials import HarmonicPotential
    from pyrpmd.analysis.shared_quantum_flux import solve_shared_core
    s,q=shared();v=np.zeros_like(q);v[1,0]=-1;chart=XiSlice(s,q,v,[-.7,.7])
    axes=[np.linspace(.95,1.05,31)]
    core=solve_shared_core(chart,axes,HarmonicPotential(k=1,center=q),potential_identity=s.pes_identity,states=8)
    result=core.flux_side(np.linspace(0,1,10))
    assert core.identity==s.identity and core.beta==s.beta
    assert result['flux_side_per_partition'][0]==0 and not result['physical_converged']
    assert not core.diagnostics['environment_coupling_implemented']
    with pytest.raises(ValueError):core.side[0,0]=1
    with pytest.raises(ValueError,match='potential identity'):
        solve_shared_core(chart,axes,HarmonicPotential(),potential_identity='wrong')
