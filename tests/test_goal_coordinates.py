from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from pyrpmd.analysis.goal_quantum_coordinates import screen_goal_subspace,fit_graph_chart,GraphCoordinateChart,assess_promotion
from pyrpmd.analysis.curvilinear_quantum import transform_geometry,cartesian_geometry,audit_chart,solve_curvilinear,triatomic_bond_geometry,triatomic_transfer_geometry


def fixture(n=600):
    q=np.random.default_rng(42).normal(size=(n,3))
    f=-q.copy();f[:,0]-=.35*q[:,1]**2;f[:,1]-=.7*q[:,0]*q[:,1]+.02*q[:,1]**3
    f[:,2]-=50*q[:,2]**3
    return q,f,np.arange(n)


def test_fermi_even_partner_without_spectator():
    q,f,g=fixture();r=screen_goal_subspace(q,f,np.ones(3),[1,0,0],g)
    assert r.required_dimension==2
    assert np.linalg.norm(r.basis_mass_weighted@r.basis_mass_weighted.T-np.diag([1,1,0]))<1e-10
    assert r.status=='candidate_requires_dynamical_probes'


def test_anharmonic_spectator_not_automatically_quantum():
    q,f,g=fixture();f=-q-4*q**3
    r=screen_goal_subspace(q,f,np.ones(3),[1,0,0],g)
    assert r.required_dimension==1


def test_collective_target_all_three_needed():
    q,f,g=fixture();r=screen_goal_subspace(q,-q-4*q**3,np.ones(3),[1,1,1],g)
    assert r.required_dimension==3


def test_rotation_equivariance():
    q,f,g=fixture();R=np.linalg.qr(np.random.default_rng(8).normal(size=(3,3)))[0]
    a=screen_goal_subspace(q,f,np.ones(3),[1,0,0],g)
    b=screen_goal_subspace(q@R,f@R,np.ones(3),np.array([1,0,0])@R,g)
    assert np.linalg.norm(R@b.basis_mass_weighted@b.basis_mass_weighted.T@R.T-a.basis_mass_weighted@a.basis_mass_weighted.T)<1e-9


def test_budget_failure_does_not_truncate():
    q,f,g=fixture();r=screen_goal_subspace(q,f,np.ones(3),[1,0,0],g,max_dimension=1)
    assert r.required_dimension==2 and r.status=='dimension_budget_exceeded'


def test_harmonic_linear_target_not_nqe_size_rank():
    q,_,g=fixture();K=np.diag([1,100,1000])
    r=screen_goal_subspace(q,-q@K,np.ones(3),[1,0,0],g)
    assert r.required_dimension==1
    assert max(sum(v) for v in r.diagnostics['nonlinear_residual_spectra'])<1e-14


@pytest.mark.parametrize('bad',[0,4,True,1.5])
def test_invalid_budget(bad):
    q,f,g=fixture()
    with pytest.raises(ValueError):screen_goal_subspace(q,f,np.ones(3),[1,0,0],g,max_dimension=bad)


def test_reject_missing_independent_groups():
    q,f,g=fixture()
    with pytest.raises(ValueError):screen_goal_subspace(q,f,np.ones(3),[1,0,0],np.zeros(len(q)))


def test_bad_force_fit_is_not_a_success():
    q,f,g=fixture();f[:,0]=np.sin(20*q[:,0])
    r=screen_goal_subspace(q,f,np.ones(3),[1,0,0],g)
    assert r.status=='pilot_force_fit_unreliable'


def graph(alpha=.6):
    t=np.linspace(-6,6,30);fn=CubicSpline(t,alpha*t*t)
    return GraphCoordinateChart(np.zeros(2),np.eye(2),np.ones(2),(fn,),(-6.,6.))


def test_graph_roundtrip_metric_and_forces():
    c=graph();u=np.random.default_rng(9).normal(size=(200,2));a=audit_chart(c,u)
    assert a['roundtrip_max']<1e-12 and a['jacobian_relative_max']<1e-9
    G,rho=cartesian_geometry(c,u);p=1.2*u[:,0]
    assert np.max(abs(G[:,0,0]-1))<1e-12
    assert np.max(abs(G[:,0,1]+p))<1e-12
    assert np.max(abs(G[:,1,1]-(1+p*p)))<1e-12
    assert np.max(abs(rho-1))<1e-12
    assert not a['reduction_approved']
    x=c.forward(u);force=-x;fu=c.generalized_force(u,force)
    h=1e-5
    for j in range(2):
        delta=np.eye(2)[j]*h
        numeric=-(np.sum(c.forward(u+delta)**2,axis=1)-np.sum(c.forward(u-delta)**2,axis=1))/(4*h)
        assert np.max(abs(fu[:,j]-numeric))<1e-7


def test_chart_learning_and_no_extrapolation():
    s=np.linspace(-2,2,500);q=np.column_stack([s,.7*s*s,np.sin(s)])
    c=fit_graph_chart(q,np.ones(3),[1,0,0],knots=8);u=c.inverse(q)
    assert np.std(u[:,1:])<2e-4
    with pytest.raises(ValueError):c.forward([[10,0,0]])


def test_folded_square_map_rejected():
    J=np.array([[[0.,0.],[0.,1.]]])
    with pytest.raises(ValueError):transform_geometry(J,np.eye(2)[None],np.ones(1))


def test_induced_1d_mass_is_not_full_G_block():
    c=graph();u=np.array([[1.,0.]]);G,_=cartesian_geometry(c,u)
    g=np.linalg.inv(G)
    assert abs(1/g[0,0,0]-G[0,0,0])>.5


def test_triatomic_measure_is_not_sqrt_metric_det():
    G,rho=triatomic_bond_geometry(np.array([2.7]),np.array([3.6]),np.array([2.8]),[79.,1.,35.])
    assert np.linalg.eigvalsh(G).min()>0
    assert abs(rho[0]-1/np.sqrt(np.linalg.det(G[0])))>1


def test_triatomic_nonlinear_transfer_jacobian_measure():
    r1,r2,th=2.7,3.6,2.8;u=np.array([[np.log(r1/r2),np.log(r1+r2),th]])
    G,rho=triatomic_transfer_geometry(u,[79,1,35])
    assert np.linalg.eigvalsh(G).min()>0
    assert abs(rho[0]-r1**3*r2**3*np.sin(th))<1e-8


def test_collinear_chart_not_silently_flattened():
    with pytest.raises(ValueError):triatomic_bond_geometry(2.,3.,np.pi,[79,1,35])


def test_quantum_harmonic_1d_full_metric():
    class Reparam:
        masses=np.ones(1)
        def forward(self,u):return u+.3*np.tanh(u)
        def jacobian(self,u):return (1+.3/np.cosh(u)**2)[...,None]
    c=Reparam()
    out=solve_curvilinear([np.linspace(-7,7,220)],lambda u:cartesian_geometry(c,u),lambda u:.5*c.forward(u)[:,0]**2,states=3)
    assert np.max(abs(out.energies-np.array([.5,1.5,2.5])))<.006
    assert out.diagnostics['mass_orthogonality_error']<1e-10


def test_2d_full_metric_cross_terms():
    c=graph(.4)
    r=solve_curvilinear([np.linspace(-5,5,35),np.linspace(-6,6,39)],lambda u:cartesian_geometry(c,u),lambda u:.5*np.sum(c.forward(u)**2,axis=1),states=3)
    assert np.max(abs(r.energies-[1,2,2]))<.08


def test_three_dimensional_solver_smoke():
    def geom(u):return np.broadcast_to(np.eye(3),(len(u),3,3)),np.ones(len(u))
    r=solve_curvilinear([np.linspace(-4,4,12)]*3,geom,lambda u:.5*np.sum(u*u,axis=1),states=2)
    assert abs(r.energies[0]-1.5)<.15


def test_promotion_detects_needed_even_partner():
    v=np.random.default_rng(1).normal(.3,.02,size=(32,10))
    d=assess_promotion(v,1,kinetic_complete=True,matched_observable=True)
    assert d['status']=='promote_to_quantum_core'


def test_small_change_with_big_uncertainty_is_not_pass():
    x=np.zeros((8,10));x[:4]=1;x[4:]=-1
    assert assess_promotion(x,1,kinetic_complete=True,matched_observable=True)['status']=='sampling_or_model_uncertain'


def test_no_quantum_propagation_approval_without_metric():
    x=np.zeros((8,10))
    assert assess_promotion(x,1)['status']=='insufficient_physical_evidence'


def test_residual_can_stay_rpmd_only_on_tested_probe():
    x=np.random.default_rng(1).normal(0,.001,size=(32,10))
    assert assess_promotion(x,1,kinetic_complete=True,matched_observable=True)['status']=='no_resolved_need_on_this_probe'


def test_arc_length_coordinate_is_nonlinear_and_invertible():
    from pyrpmd.analysis.goal_quantum_coordinates import arc_length_chart
    c=arc_length_chart(graph());u=np.array([[5.,0.],[6.,.2],[7.,-.1]])
    a=audit_chart(c,u,step=1e-6)
    assert a['roundtrip_max']<1e-7 and a['jacobian_relative_max']<1e-5
    J=c.jacobian(u)
    assert np.max(abs(np.sum(J[:,:,0]**2,axis=1)-1))<1e-4
    assert a['normalized_core_environment_metric_cross_max']>.5


def test_goal_chart_whitening_invariant_to_units():
    from pyrpmd.analysis.goal_quantum_coordinates import screen_chart_subspace
    q,f,g=fixture();S=np.diag([2.,.3,5.]);Si=np.linalg.inv(S)
    geom=lambda u:(np.broadcast_to(Si@Si,(len(u),3,3)),np.ones(len(u))*np.linalg.det(S))
    r=screen_chart_subspace(q@Si,f@S,geom,np.array([1,0,0])@S,g)
    B=np.asarray(r['basis_in_chart_coordinates'])
    assert np.linalg.norm((S@B)@(S@B).T-np.diag([1,1,0]))<1e-9


def test_missing_joint_probe_blocks_approval():
    from pyrpmd.analysis.goal_quantum_coordinates import choose_core_from_probes
    c=[{'name':'q','dimension':1,'target_id':'flux','required_probes':['y','z','yz']}]
    ev={k:dict(delta_blocks=np.zeros((8,3)),reference_scale=1,target_id='flux',kinetic_complete=True,matched_observable=True) for k in ['y','z']}
    assert choose_core_from_probes(c,ev)['selected_candidate'] is None


def test_choose_smallest_core_only_after_all_probes():
    from pyrpmd.analysis.goal_quantum_coordinates import choose_core_from_probes
    c=[{'name':'q','dimension':1,'target_id':'x','required_probes':['y']},{'name':'qy','dimension':2,'target_id':'x','required_probes':['z']}]
    ev={k:dict(delta_blocks=np.full((8,3),v),reference_scale=1,target_id='x',kinetic_complete=True,matched_observable=True) for k,v in [('y',.3),('z',.001)]}
    assert choose_core_from_probes(c,ev)['dimension']==2


def test_wrong_target_does_not_pass_gate():
    from pyrpmd.analysis.goal_quantum_coordinates import choose_core_from_probes
    c=[{'name':'q','dimension':1,'target_id':'flux','required_probes':['y']}]
    ev={'y':dict(delta_blocks=np.zeros((8,3)),reference_scale=1,target_id='position',kinetic_complete=True,matched_observable=True)}
    assert choose_core_from_probes(c,ev)['status']=='no_core_approved'
