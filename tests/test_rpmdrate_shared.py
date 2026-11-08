from pathlib import Path
import sys
import json
from dataclasses import replace
from types import SimpleNamespace, ModuleType
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'benchmarks/rpmdrate_shared'))
from coordinate_oracle import FormulaOracle
from pyrpmd.analysis.rpmdrate_shared import SharedRPMDrateCoordinate,LocalXiChart,_digest
from pyrpmd.analysis.rpmdrate_quantum import SharedRateContext,MatchedFluxPair,kubo_flux_side,shared_side_operator,combine_shared_kappa
from pyrpmd.workflows.rate_quantum import bind_rate_context,run_shared_quantum_rate


def coordinate(xi_star=1.):
    ref=np.array([[-3.,0,0],[0.,.2,0],[3.,0,0]])
    oracle=FormulaOracle([[0],[1,2]],[10.,1.,10.],ref,[[0,1]],[[1,2]],12.)
    return SharedRPMDrateCoordinate(oracle,xi_star),ref


def context(shared,summary):
    return SharedRateContext(shared.coordinate_id,shared.surface_id,4.,'PES-sha-abc','same reactant partition',
        _digest(summary),'physical xi vs centroid xi; matched thermal active Hamiltonian')


def baseline(shared,n=51):
    t=np.linspace(.1,5,n);curve=np.full(n,.7)
    summary={'task':'rate','status':'plateau_estimate','kappa':.7,'pmf':{'xi_star':shared.xi_star,'k_qtst_cm3_s':2e-12}}
    return t,curve,summary


def pair(ctx,t,delta=.1,nblocks=16):
    r=np.ones((nblocks,len(t)))*.5;q=r+delta;d=np.ones(nblocks)
    ev={k:{'passed':True,'evidence':'synthetic matched test, not a physical production assertion'} for k in
        ('same_active_hamiltonian','same_reactant_normalization','same_positive_reference_flux','quantum_kinetic_and_measure','environment_coupling','independent_block_sampling')}
    return MatchedFluxPair(ctx,t,q,r,d,tuple(f'chain-{i}' for i in range(nblocks)),'H-active','RP-flux-active','active canonical Qr',ev)


def test_native_formula_limits():
    s,q=coordinate();p=s.evaluate(q)
    assert p.xi==pytest.approx(1.) and p.surface==pytest.approx(0.)
    z=q.copy();com=(s.masses[1:]*1.)@z[1:]/s.masses[1:].sum()
    z[0]=com+np.array([-12.,0,0]);assert s.evaluate(z).xi==pytest.approx(0.,abs=1e-12)


def test_same_side_orientation_on_both_sides():
    s,q=coordinate();left=q.copy();right=q.copy();left[1,0]-=.4;right[1,0]+=.4
    assert s.side(left)==1. and s.side(right)==0.
    for x in (left,right):
        p=s.evaluate(x);assert p.surface==pytest.approx(p.denominator*(p.xi-1))


def test_gradients_finite_difference_and_symmetries():
    s,q=coordinate();q=q+np.array([[.1,.1,0],[.2,.1,.1],[0,-.1,.1]])
    p=s.evaluate(q);eps=1e-6
    for j in range(q.size):
        d=np.eye(q.size)[j].reshape(q.shape)*eps
        assert (s.evaluate(q+d).xi-s.evaluate(q-d).xi)/(2*eps)==pytest.approx(p.gradient.ravel()[j],rel=1e-6,abs=1e-9)
    np.testing.assert_allclose(p.gradient.sum(0),0,atol=1e-12)
    np.testing.assert_allclose(np.cross(q,p.gradient).sum(0),0,atol=1e-12)


def test_centroid_is_not_bead_average():
    s,q=coordinate();ring=np.stack([q.copy(),q.copy()]);ring[0,1,0]=-.8;ring[1,1,0]=.8
    value,J=s.ring_value_gradient(ring)
    assert value==pytest.approx(s.evaluate(ring.mean(0)).xi)
    assert abs(value-np.mean([s.evaluate(x).xi for x in ring]))>1e-5
    np.testing.assert_allclose(J[0],s.evaluate(q).gradient/2)


def test_ring_flux_uses_centroid_velocity_no_extra_bead_mass():
    s,q=coordinate();ring=np.repeat(q[None],8,axis=0);p=np.ones_like(ring);p[:,:,0]*=np.array([1.,2.,3.])
    expected=np.sum(s.evaluate(q).surface_gradient*p.mean(0)/s.masses[:,None])
    assert s.centroid_flux_velocity(ring,p)==pytest.approx(expected)


def test_physical_metric_not_ring_scaled():
    s,q=coordinate();G=s.physical_mass_metric(q)
    assert G[0,0]==pytest.approx(np.sum(s.evaluate(q).gradient**2/s.masses[:,None]))
    with pytest.raises(ValueError):s.physical_mass_metric(q,[s.evaluate(q).gradient])


def test_mutated_coordinate_rejected():
    s,q=coordinate();s.coordinate.rinf+=1
    with pytest.raises(ValueError):s.evaluate(q)


def test_mutated_surface_rejected():
    s,q=coordinate();s._native_surface.xi+=.1
    with pytest.raises(ValueError):s.evaluate(q)


def test_denominator_domain_rejected():
    s,q=coordinate();s.coordinate.rinf=.1
    s=SharedRPMDrateCoordinate(s.coordinate,1.)
    with pytest.raises(ValueError):s.evaluate(q)


def test_equivalent_branch_seam_rejected_only_for_quantum():
    s,q=coordinate();s.coordinate.paths.append(s.coordinate.paths[0])
    s=SharedRPMDrateCoordinate(s.coordinate,1.)
    s.evaluate(q)
    with pytest.raises(ValueError):s.evaluate(q,require_smooth=True)


def test_chart_roundtrip_and_exact_xi():
    s,q=coordinate();c=LocalXiChart(s,q,radius=2.)
    u=np.r_[1.01,np.linspace(-.03,.04,8)]
    x=c.forward(u);assert s.evaluate(x).xi==pytest.approx(u[0],abs=2e-12)
    np.testing.assert_allclose(c.inverse(x),u,atol=3e-11)


def test_chart_jacobian_and_full_metric():
    s,q=coordinate();c=LocalXiChart(s,q,radius=3.)
    u=np.r_[1.005,np.linspace(-.02,.03,8)];J=c.jacobian(u);h=1e-6
    for j in range(len(u)):
        d=np.eye(len(u))[j]*h
        np.testing.assert_allclose(J[:,j],(c.forward(u+d)-c.forward(u-d)).ravel()/(2*h),rtol=1e-5,atol=1e-7)
    G,rho=c.geometry(u);mass=np.repeat(s.masses,3)
    np.testing.assert_allclose(G,np.linalg.inv(J.T@(mass[:,None]*J)),atol=1e-10)
    assert rho>0


def test_projected_mass_is_not_frozen_slice_mass():
    s,q=coordinate();c=LocalXiChart(s,q,radius=10)
    u=np.r_[1.06,np.linspace(.1,.3,8)];x=c.forward(u)
    G=s.physical_mass_metric(x)[0,0];sliceG=c.active_slice_metric(u,1)['inverse_mass'][0,0]
    assert abs(sliceG/G-1)>1e-5
    assert c.active_slice_metric(u,1)['bath_feedback_included'] is False


def test_chart_cannot_extrapolate():
    s,q=coordinate();c=LocalXiChart(s,q,radius=.1)
    with pytest.raises(ValueError):c.forward(np.r_[2.,np.zeros(8)])


def test_shared_projector_matches_surface_not_atom_label():
    s,q=coordinate();grid=np.repeat(q[None],31,axis=0);grid[:,1,0]=np.linspace(-.5,.5,31)
    h=shared_side_operator(s,grid)
    np.testing.assert_array_equal(np.diag(h),[s.side(x) for x in grid])


def test_quantum_flux_hermitian_and_zero_time():
    H=np.array([[1.,-.4],[-.4,1.]]);h=np.diag([0.,1.]);t=np.linspace(0,2,100)
    r=kubo_flux_side(H,h,4,t)
    np.testing.assert_allclose(r['flux_operator'],r['flux_operator'].conj().T,atol=1e-14)
    assert r['canonical_flux_side'][0]==0. and r['canonical_flux_side'][1]>0
    assert r['reactant_population']==pytest.approx(.5)
    assert not r['is_rpmd_transmission_coefficient']


def test_quantum_flux_energy_zero_invariance():
    H=np.array([[1.,-.4],[-.4,1.]]);h=np.diag([0.,1.]);t=np.linspace(0,3,100)
    a=kubo_flux_side(H,h,4,t);b=kubo_flux_side(H+100*np.eye(2),h,4,t)
    np.testing.assert_allclose(a['canonical_flux_side'],b['canonical_flux_side'],atol=1e-13)


def test_quantum_flux_against_direct_kubo_trace():
    from scipy.linalg import eigh
    H=np.array([[.2,.1,.07],[.1,.6,.05],[.07,.05,1.]]);h=np.diag([0.,1.,1.]);beta=3.;t=np.array([0.,.2,.7])
    r=kubo_flux_side(H,h,beta,t);E,U=eigh(H);p=np.exp(-beta*E);p/=p.sum();he=U.T@h@U
    F=U.T@(1j*(H@h-h@H))@U
    expected=[]
    for time in t:
        result=0j
        for i in range(3):
            for j in range(3):
                k=p[i] if abs(E[i]-E[j])<1e-12 else (p[i]-p[j])/(beta*(E[j]-E[i]))
                result+=k*F[i,j]*he[j,i]*np.exp(1j*(E[j]-E[i])*time)
        expected.append(result.real)
    np.testing.assert_allclose(r['canonical_flux_side'],expected,atol=1e-14)


@pytest.mark.parametrize('badH,badh',[(np.array([[1,1],[0,1]]),np.diag([0,1])),(np.eye(2),np.diag([.5,1]))])
def test_quantum_operators_rejected(badH,badh):
    with pytest.raises(ValueError):kubo_flux_side(badH,badh,3,[0,1])


def test_zero_defect_recovers_native_rpmd():
    s,_=coordinate();t,curve,sm=baseline(s);ctx=context(s,sm);pa=pair(ctx,t,0.)
    r=combine_shared_kappa(ctx,sm,t,curve,pa,plateau_start=20)
    assert r['rate_estimate_cm3_s']==pytest.approx(sm['pmf']['k_qtst_cm3_s']*sm['kappa'])
    assert r['quantum_equals_RPMD_limit_error']==0 and r['conditional_correction_standard_error_cm3_s']<1e-25
    assert r['rate_cm3_s'] is None and not r['physical_converged']


def test_all_active_algebraic_recovery():
    s,_=coordinate();t,curve,sm=baseline(s);ctx=context(s,sm)
    pa=pair(ctx,t);pa=replace(pa,rpmd_numerator=np.ones((16,len(t)))*.7,quantum_numerator=np.ones((16,len(t)))*1.3)
    r=combine_shared_kappa(ctx,sm,t,curve,pa,plateau_start=20)
    assert r['corrected_kappa_estimate']==pytest.approx(1.3)
    assert r['numerical_prerequisites_passed'] and not r['exact_general_rate_claim']


def test_reference_flux_ratio_not_mean_of_ratios():
    s,_=coordinate();t,curve,sm=baseline(s);ctx=context(s,sm);pa=pair(ctx,t,nblocks=2)
    pa=replace(pa,reference_flux=np.array([1.,9.]),quantum_numerator=np.ones((2,len(t))),rpmd_numerator=np.zeros((2,len(t))))
    q,r=pa.curves();np.testing.assert_allclose(q,.2)


def test_numerators_are_immutable():
    s,_=coordinate();t,curve,sm=baseline(s);pa=pair(context(s,sm),t)
    with pytest.raises(ValueError):pa.quantum_numerator[0,0]=99


@pytest.mark.parametrize('field,value',[('beta_au',5.),('surface_id','wrong-surface'),('coordinate_id','wrong-coordinate'),('pes_identity','other-PES'),('reactant_normalization','different-Qr'),('kinetic_mapping','bad-mass')])
def test_mismatched_correction_rejected(field,value):
    s,_=coordinate();t,curve,sm=baseline(s);ctx=context(s,sm);pa=pair(replace(ctx,**{field:value}),t)
    with pytest.raises(ValueError):combine_shared_kappa(ctx,sm,t,curve,pa,plateau_start=20)


def test_changed_baseline_cannot_be_reused():
    s,_=coordinate();t,curve,sm=baseline(s);ctx=context(s,sm);pa=pair(ctx,t);sm['pmf']['k_qtst_cm3_s']*=2
    with pytest.raises(ValueError):combine_shared_kappa(ctx,sm,t,curve,pa,plateau_start=20)


def test_missing_kinetic_evidence_blocks_publication():
    s,_=coordinate();t,curve,sm=baseline(s);ctx=context(s,sm);pa=pair(ctx,t);ev=dict(pa.evidence);ev.pop('quantum_kinetic_and_measure');pa=replace(pa,evidence=ev)
    r=combine_shared_kappa(ctx,sm,t,curve,pa,plateau_start=20)
    assert not r['numerical_prerequisites_passed']
    assert 'missing_quantum_kinetic_and_measure' in r['withheld_reasons']


def test_quantum_oscillations_are_not_hidden_by_zero_drift():
    s,_=coordinate();t,curve,sm=baseline(s);ctx=context(s,sm);pa=pair(ctx,t)
    pa=replace(pa,quantum_numerator=pa.quantum_numerator+.3*np.sin(20*t)[None])
    r=combine_shared_kappa(ctx,sm,t,curve,pa,plateau_start=20)
    assert 'corrected_curve_not_flat_on_native_tail' in r['withheld_reasons']


def test_negative_correction_retained_not_clipped():
    s,_=coordinate();t,curve,sm=baseline(s);ctx=context(s,sm);pa=pair(ctx,t,delta=-2.)
    r=combine_shared_kappa(ctx,sm,t,curve,pa,plateau_start=20)
    assert r['rate_estimate_cm3_s']<0 and r['negative_estimate_retained']


def test_native_surface_binding_requires_same_xi():
    s,_=coordinate();t,curve,sm=baseline(s);sm['pmf']['xi_star']=.9
    with pytest.raises(ValueError):bind_rate_context(s,{'beta':4},sm,pes_identity='x',reactant_normalization='y',kinetic_mapping='z')


def test_workflow_preserves_native_result_and_calls_same_coordinate(tmp_path,monkeypatch):
    s,ref=coordinate();t,curve,sm=baseline(s)
    native=ModuleType('pyrpmd.workflows.rate'); artifacts=ModuleType('pyrpmd.rate_artifacts')
    prepare=ModuleType('pyrpmd.workflows.prepare');prepare.prepared_system=lambda system,record:SimpleNamespace(prepared=True)
    monkeypatch.setitem(sys.modules,'pyrpmd.workflows.prepare',prepare)
    def native_run(config,system,output):
        output.mkdir(parents=True);np.savez(output/'recrossing.npz',time_au=t,kappa=curve)
        (output/'summary.json').write_text(json.dumps(sm))
        (output/'preparation.json').write_text(json.dumps({'ready':True,'reaction':{'forming_bonds':[[0,1]]}}));return sm
    native.run_rate=native_run;artifacts.active_rate_directory=lambda path,summary:path
    monkeypatch.setitem(sys.modules,'pyrpmd.workflows.rate',native);monkeypatch.setitem(sys.modules,'pyrpmd.rate_artifacts',artifacts)
    monkeypatch.setattr(SharedRPMDrateCoordinate,'from_rate_job',classmethod(lambda cls,cfg,sys,star:s))
    def backend(**kw):
        assert kw['coordinate'] is s and kw['coordinate'].evaluate(ref).xi==pytest.approx(1)
        assert kw['system'].prepared and kw['config']['reaction']['forming_bonds']==[[0,1]]
        return pair(kw['context'],kw['time_au'],delta=.1)
    result=run_shared_quantum_rate({'task':'rate','beta':4.},SimpleNamespace(),tmp_path,backend,pes_identity='toy',reactant_normalization='same-Qr',kinetic_mapping='toy-active',plateau_start=20)
    assert result['rate_estimate_cm3_s']==pytest.approx(1.6e-12)
    assert json.loads((tmp_path/'rpmd/summary.json').read_text())==sm
    assert (tmp_path/'active_flux_pair.npz').is_file() and (tmp_path/'shared_coordinate.json').is_file()


def test_native_pyrpmd_coordinate_parity_when_installed():
    native=pytest.importorskip('pyrpmd.cv',reason='Full upstream package is not present in the isolated patch-only environment')
    o,ref=coordinate();actual=native.RPMDrateCoordinate([[0],[1,2]],[10.,1.,10.],ref,[[0,1]],[[1,2]],12.)
    shared=SharedRPMDrateCoordinate(actual,1.)
    for q in ref[None]+np.random.default_rng(3).normal(scale=.05,size=(30,3,3)):
        a=shared.evaluate(q);b=o.evaluate(q)
        assert a.xi==pytest.approx(b.xi,abs=1e-13);np.testing.assert_allclose(a.gradient,b.gradient,atol=1e-13)


def test_context_numeric_dividing_surface_mismatch():
    s,_=coordinate();t,curve,sm=baseline(s);ctx=replace(context(s,sm),xi_star=.9)
    with pytest.raises(ValueError):combine_shared_kappa(ctx,sm,t,curve,pair(ctx,t),plateau_start=20)


@pytest.mark.parametrize('cfg',[{'task':'rate','recrossing':{'dynamics':'trpmd'}},
    {'task':'rate','rate':{'recrossing':{'dynamics':'trpmd'}}}])
def test_workflow_rejects_top_level_and_nested_trpmd(cfg,tmp_path):
    with pytest.raises(ValueError,match='RPMD'):run_shared_quantum_rate(cfg,None,tmp_path,lambda **kw:None,
        pes_identity='PES',reactant_normalization='Qr',kinetic_mapping='map',plateau_start=2)


def test_finite_basis_backend_really_propagates_and_uses_same_side():
    from pyrpmd.analysis.rpmdrate_quantum import FiniteBasisFluxBackend
    s,ref=coordinate();t,curve,sm=baseline(s);ctx=context(s,sm)
    grid=np.stack([ref.copy(),ref.copy()]);grid[:,1,0]=[-.3,.3]
    H=np.array([[.1,-.2],[-.2,.3]])
    template=pair(ctx,t)
    problem=dict(hamiltonian=H,grid_geometries=grid,rpmd_numerator=template.rpmd_numerator,
        reference_flux=template.reference_flux,independent_block_ids=template.independent_block_ids,
        active_hamiltonian_id='physical two-site H',reference_flux_id='common positive flux',
        equilibrium_normalization_id='common canonical measure',quantum_to_common_measure=2.3,evidence=template.evidence)
    backend=FiniteBasisFluxBackend(lambda **kw:problem)
    result=backend(context=ctx,coordinate=s,time_au=t)
    reference=kubo_flux_side(H,shared_side_operator(s,grid),4,t)
    np.testing.assert_allclose(result.quantum_numerator[0],2.3*reference['canonical_flux_side'],atol=1e-14)
    assert not backend.last_diagnostics['bath_feedback_constructed_automatically']
    assert result.context.identity==ctx.identity
    problem.pop('quantum_to_common_measure')
    with pytest.raises(ValueError):backend(context=ctx,coordinate=s,time_au=t)


def test_backend_failure_preserves_native_run(tmp_path,monkeypatch):
    s,ref=coordinate();t,curve,sm=baseline(s)
    native=ModuleType('pyrpmd.workflows.rate');prepare=ModuleType('pyrpmd.workflows.prepare')
    artifacts=ModuleType('pyrpmd.rate_artifacts')
    prepare.prepared_system=lambda system,record:system
    def native_run(c,sy,out):
        out.mkdir(parents=True);np.savez(out/'recrossing.npz',time_au=t,kappa=curve)
        (out/'summary.json').write_text(json.dumps(sm));(out/'preparation.json').write_text(json.dumps({'ready':True,'reaction':{}}));return sm
    native.run_rate=native_run;artifacts.active_rate_directory=lambda path,summary:path
    for name,obj in [('pyrpmd.workflows.rate',native),('pyrpmd.workflows.prepare',prepare),('pyrpmd.rate_artifacts',artifacts)]:monkeypatch.setitem(sys.modules,name,obj)
    monkeypatch.setattr(SharedRPMDrateCoordinate,'from_rate_job',classmethod(lambda cls,*args:s))
    def broken(**kw):raise RuntimeError('quantum solver failed')
    with pytest.raises(RuntimeError):run_shared_quantum_rate({'task':'rate','beta':4.},SimpleNamespace(),tmp_path,broken,
        pes_identity='test',reactant_normalization='same',kinetic_mapping='map',plateau_start=20)
    assert json.loads((tmp_path/'rpmd/summary.json').read_text())==sm
    assert not (tmp_path/'quantum_correction.json').exists()


@pytest.mark.parametrize('d',[1,2,3])
def test_active_grid_builds_actual_shared_xi_hamiltonian(d):
    from pyrpmd.analysis.rpmdrate_quantum import build_active_slice_hamiltonian
    s,ref=coordinate();chart=LocalXiChart(s,ref,radius=2.)
    axes=[np.linspace(.995,1.005,5)]+[np.linspace(-.03,.03,5) for _ in range(d-1)]
    potential=lambda x:.1*np.sum((x-ref)**2,axis=(1,2))
    data=build_active_slice_hamiltonian(chart,axes,potential)
    H=data['hamiltonian'];h=data['side_operator']
    assert H.shape==(3**d,3**d) and np.isfinite(H).all()
    np.testing.assert_allclose(H,H.T,atol=1e-10)
    assert np.linalg.eigvalsh(H).min()>0
    assert not data['diagnostics']['environment_feedback']
    for u,x in zip(data['active_grid'],data['grid_geometries']):assert s.evaluate(x).xi==pytest.approx(u[0],abs=3e-12)
    kernel=kubo_flux_side(H,h,1.,[0.,.01,.02])
    assert kernel['canonical_flux_side'][0]==0.


def test_active_grid_memory_guard_and_pes_contract():
    from pyrpmd.analysis.rpmdrate_quantum import build_active_slice_hamiltonian
    s,ref=coordinate();chart=LocalXiChart(s,ref,radius=2.);axes=[np.linspace(.99,1.01,7)]
    with pytest.raises(ValueError):build_active_slice_hamiltonian(chart,axes,lambda x:np.zeros(len(x)),max_basis=2)
    with pytest.raises(ValueError):build_active_slice_hamiltonian(chart,axes,lambda x:0.)
