"""Fast software/analytic gates. None certifies physical rate convergence."""
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'benchmarks/noncollinear_quantum'))
from pyrpmd.analysis.bimolecular_scattering import (
    sample_collision_flux,thermal_impact_rate,partial_wave_thermal_rate,
    rovibrational_partition,matched_bimolecular_rate_defect,KB_AU,RATE_AU_TO_CM3_S)
from pyrpmd.analysis.bimolecular_active_space import internal_basis,align_molecular_frames,select_molecular_subspace
from pyrpmd.analysis.molecular_collision import assemble_collision,propagate_cartesian


def test_flux_normalization_and_energy_measure():
    f=sample_collision_flux(100000,300,1000,6,seed=45,proposal_temperature_ratio=2)
    assert abs(f.importance_weight.mean()-1)<.015
    assert abs(np.mean(f.energy*f.importance_weight)*f.beta-2)<.025
    assert abs(np.mean(f.impact_parameter**2)-18)<.12


def test_hard_sphere_flux_rate():
    f=sample_collision_flux(100000,300,1000,6,seed=91)
    r=thermal_impact_rate((f.impact_parameter<3).astype(int),f.importance_weight,temperature=300,reduced_mass=1000,bmax=6)
    exact=np.pi*9*np.sqrt(8*KB_AU*300/(np.pi*1000))
    assert abs(r['rate_lower_au']/exact-1)<.02
    assert r['rate_lower_cm3_molecule_s']==r['rate_lower_au']*RATE_AU_TO_CM3_S


def test_unknown_trajectories_are_not_deleted():
    r=thermal_impact_rate([1,0,-1,-2],[1,2,3,4],temperature=300,reduced_mass=1,bmax=1)
    assert r['rate_upper_au']/r['rate_lower_au']==pytest.approx(8)
    assert r['invalid']==r['unfinished']==1


@pytest.mark.parametrize('kwargs',[{'bmax':0},{'temperature':-1},{'reduced_mass':np.nan}])
def test_flux_bad_scalars(kwargs):
    d=dict(temperature=300,reduced_mass=1,bmax=1);d.update(kwargs)
    with pytest.raises(ValueError):sample_collision_flux(10,**d)


def test_quantum_rate_energy_zero_invariance():
    E=np.linspace(0,.03,1001);N=(E>.003)[None,:].astype(float)
    d=dict(temperature=300,reduced_mass=1000,internal_partition=2,parity_summed=True)
    a=partial_wave_thermal_rate(E,N,[0],**d);b=partial_wave_thermal_rate(E+.2,N,[0],reference_energy=.2,**d)
    assert a['rate_au']==pytest.approx(b['rate_au'],rel=1e-12)


def test_partial_wave_degeneracy():
    E=np.linspace(0,.04,301);N=np.ones((3,len(E)))
    d=dict(temperature=600,reduced_mass=1000,internal_partition=1,parity_summed=True)
    a=partial_wave_thermal_rate(E,N,[0,1,2],**d);b=partial_wave_thermal_rate(E,N[:1],[0],**d)
    assert a['rate_au']/b['rate_au']==pytest.approx(9)


def test_missing_J_or_parity_is_rejected():
    d=dict(temperature=300,reduced_mass=1,internal_partition=1)
    with pytest.raises(ValueError):partial_wave_thermal_rate([0,1],np.ones((2,2)),[0,2],parity_summed=True,**d)
    with pytest.raises(ValueError):partial_wave_thermal_rate([0,1],np.ones((1,2)),[0],**d)


def test_rotational_partition():
    assert rovibrational_partition([.1,.2],[1,3],300,reference_energy=.1)==pytest.approx(1+3*np.exp(-.1/(KB_AU*300)))


def test_mismatched_rate_metadata_rejected():
    d=dict(reaction='A+B',temperature_K=300,unit='cm^3 molecule^-1 s^-1',spatial_dimension=3,
        reactant_normalization='thermal bound',electronic_weight=1,symmetry_factor=1,observable='product flux',rate=2.,
        pes_identity='test',angular_momentum_coverage='all impact parameters',state_population_convention='canonical')
    assert matched_bimolecular_rate_defect(d,d,dict(d,rate=3))['rate']==3
    with pytest.raises(ValueError):matched_bimolecular_rate_defect(d,dict(d,unit='bohr/au'),d)
    with pytest.raises(ValueError):matched_bimolecular_rate_defect(d,dict(d,observable='position'),d)
    with pytest.raises(ValueError):matched_bimolecular_rate_defect(d,d,dict(d,angular_momentum_coverage='J=0'))


def test_translation_rotation_projector():
    x=np.array([[0,0,0],[1,.5,0],[2,0,0.]])
    B,centered,rank=internal_basis(x,[2,1,3]);assert rank==6;assert B.shape==(9,3)
    np.testing.assert_allclose(B.T@B,np.eye(3),atol=1e-12)
    B,_,rank=internal_basis([[0,0,0],[1,0,0],[2,0,0]],[2,1,3]);assert rank==5;assert B.shape==(9,4)


def test_force_alignment_is_covariant():
    rng=np.random.default_rng(44);ref=rng.normal(size=(5,3));F=rng.normal(size=(5,3));rot=Rotation.random(random_state=rng).as_matrix()
    q=np.array([ref@rot+[2,3,4]]);f=np.array([F@rot]);m=np.arange(1,6)
    aq,af,_,target=align_molecular_frames(q,f,ref,m)
    np.testing.assert_allclose(aq[0],target,atol=1e-12);np.testing.assert_allclose(af[0],F,atol=1e-12)


def test_selector_global_rotation_invariance():
    rng=np.random.default_rng(11);ref=np.array([[0,0,0],[1,.6,0],[2,0,.2],[.5,.2,1.]])
    m=np.array([10,1,9,6.]);B,ref,_=internal_basis(ref,m);D=B.shape[1];sm=np.repeat(np.sqrt(m),3)
    z=rng.normal(size=(120,D))*.03;K=np.diag(np.arange(1,D+1));g=np.array([15.,0,0,0,0,0])
    q=ref+(z@B.T/sm).reshape(120,4,3);f=(-((z@K)+4*g*z**3)@B.T*sm).reshape(q.shape)
    H=np.array([(B@(K+np.diag(12*g*a*a))@B.T)*sm[:,None]*sm[None,:] for a in z]);a=(B[:,0]*sm).reshape(4,3)
    kwargs=dict(reference=ref,hessians=H,beta=3,observable_gradient=a,max_dimension=3)
    r=select_molecular_subspace(q,f,m,**kwargs)
    R=Rotation.random(random_state=rng).as_matrix();T=np.kron(np.eye(4),R.T)
    rr=select_molecular_subspace(q@R+2,f@R,m,reference=ref@R+2,hessians=T@H@T.T,beta=3,observable_gradient=a@R,max_dimension=3)
    assert r.selected_dimension==rr.selected_dimension
    left=T@r.basis_mass_weighted
    np.testing.assert_allclose(left@left.T,rr.basis_mass_weighted@rr.basis_mass_weighted.T,atol=2e-7)


def test_assembly_arbitrary_atom_count_and_total_com():
    rng=np.random.default_rng(11);N,P=12,4;ma=np.array([12.,1,1]);mb=np.array([16.,1.])
    def fragment(m):
        q=rng.normal(size=(N,P,len(m),3));q-=(q*m[None,None,:,None]).sum(2,keepdims=True)/m.sum()
        p=rng.normal(size=q.shape);p-=p.sum(2,keepdims=True)*m[None,None,:,None]/m.sum()
        return q,p
    a,b=fragment(ma),fragment(mb);q,p,flux=assemble_collision(a,b,ma,mb,temperature=300,bmax=4,launch_R=12)
    m=np.r_[ma,mb]
    np.testing.assert_allclose((q*m[None,None,:,None]).sum(2),0,atol=2e-13)
    np.testing.assert_allclose(p.sum(2),0,atol=2e-13)
    R=(q[:,:,3:]*mb[None,None,:,None]).sum(2)/mb.sum()-(q[:,:,:3]*ma[None,None,:,None]).sum(2)/ma.sum()
    np.testing.assert_allclose(R.mean(1)[:,0],flux.impact_parameter,atol=1e-13)
    np.testing.assert_allclose(R.mean(1)[:,2],12,atol=1e-13)


def test_generic_integrator_multidimensional_free_particle():
    rng=np.random.default_rng(5);q=rng.normal(size=(3,1,7,3));p=rng.normal(size=q.shape);m=np.arange(1,8.)
    def potential(x):return np.zeros(len(x)),np.zeros_like(x)
    r=propagate_cartesian(q,p,m,beta=3,dt=.01,steps=20,potential=potential,classifier=lambda q,v:np.full(len(q),-1))
    np.testing.assert_allclose(r['positions'],q+.2*p/m[None,None,:,None],atol=1e-13)
    np.testing.assert_allclose(r['max_energy_drift_per_trajectory'],0,atol=1e-13)


def test_unvalidated_quantum_reference_cannot_be_promoted():
    from types import SimpleNamespace
    from pyrpmd.analysis.bimolecular_workflow import assemble_correction
    d=dict(reaction='A+B',temperature_K=300,unit='cm^3 molecule^-1 s^-1',spatial_dimension=3,
        reactant_normalization='thermal',electronic_weight=1,symmetry_factor=1,observable='flux',rate=2.,
        pes_identity='test',angular_momentum_coverage='all J',state_population_convention='canonical')
    candidate=SimpleNamespace(status='candidate',selected_dimension=2)
    with pytest.raises(ValueError):assemble_correction(candidate,d,d,d,sampling_covariance=np.eye(3),quantum_reference_gates={})
    gates={k:{'passed':True,'evidence':'unit-test-only synthetic evidence'} for k in
        ('active_hamiltonian_metric','reactant_normalization','same_observable','quantum_grid','quantum_time_or_scattering','angular_momentum_sum')}
    result=assemble_correction(candidate,d,d,dict(d,rate=3),sampling_covariance=np.eye(3),quantum_reference_gates=gates)
    assert result['rate']==3 and result['sampling_standard_error']==pytest.approx(np.sqrt(3))
    assert not result['physical_converged']
    with pytest.raises(ValueError):assemble_correction(SimpleNamespace(status='full_internal_space_required_no_reduction'),d,d,d,sampling_covariance=np.eye(3),quantum_reference_gates=gates)


def test_end_to_end_arbitrary_fragment_workflow():
    from pyrpmd.analysis.molecular_collision import run_bimolecular_scattering
    n=4000;ma=np.array([12.,1.,1.]);mb=np.array([16.,1.])
    a=np.zeros((n,1,3,3));b=np.zeros((n,1,2,3))
    # Free fragments plus a synthetic unit capture disk isolate phase-space
    # normalization; this is not a real polyatomic reaction calculation.
    def potential(x):return np.zeros(len(x)),np.zeros_like(x)
    def classifier(x,v):
        R=(x[:,3:]*mb[None,:,None]).sum(1)/mb.sum()-(x[:,:3]*ma[None,:,None]).sum(1)/ma.sum()
        return (np.linalg.norm(R[:,:2],axis=1)<1.).astype(int)
    result=run_bimolecular_scattering((a,a),(b,b),ma,mb,temperature=300,bmax=2,
        launch_R=8,potential=potential,classifier=classifier,dt=.1,steps=1,
        provenance={'pes_identity':'zero-potential synthetic unit test','reactant_preparation':'zero internal coordinates'},seed=55)
    mu=ma.sum()*mb.sum()/(ma.sum()+mb.sum());expected=np.pi*np.sqrt(8*KB_AU*300/(np.pi*mu))*RATE_AU_TO_CM3_S
    assert result['rate']['rate_lower_cm3_molecule_s']==pytest.approx(expected,rel=.08)
    assert result['rate']['atoms']==5 and not result['rate']['physical_converged']
