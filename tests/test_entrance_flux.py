"""Scientific regression guards for the H3 asymptotic entrance-flux bridge."""
import numpy as np
import pytest
from pyrpmd.analysis.entrance import (
    free_h3_entrance_flux,bond_entrance_observables,entrance_flux_ratio,
    h3_entrance_from_centroid_samples,bond_pmf_rate_from_entrance_bridge,
)
M=1837.1526474396187
BETA=210.51668320265807
IDENTITY="BKMP2_1500K_P16_r01r12_common_joint_umbrella_20261031"


def fixture():
    n=400
    return dict(
        reference_coordinate=np.r_[np.full(200,10.),np.full(200,7.)],
        target_coordinates=np.r_[np.full((200,1),5.),np.zeros((200,1))],
        weights=np.ones(n),reference_speed=np.ones(n),
        target_speed=np.ones((n,1)),target_topology=np.ones((n,1),bool),
        reference_reactant_mask=np.ones(n,bool),
        reference_radius=10.,reference_width=.2,target_width=.1,
        canonical_identity=IDENTITY,measure="joint_unbiased_canonical",
        connected=True,masses_au=[M]*3,beta=BETA,min_effective=20)


def test_free_entrance_matches_independent_rpmdrate_prefactor():
    k=free_h3_entrance_flux(10,BETA,[M]*3)
    assert k["reference_rate_au"]==pytest.approx(.9873012308830208,rel=1e-12)
    assert free_h3_entrance_flux(20,BETA,[M]*3)["reference_rate_au"]==pytest.approx(4*k["reference_rate_au"])
    assert free_h3_entrance_flux(10,BETA,[M]*3,electronic_weight=0)["reference_rate_au"]==0
    with pytest.raises(ValueError): free_h3_entrance_flux(0,BETA,[M]*3)


def test_exact_bond_gradient_half_maxwell_and_third_bond_topology():
    q=np.array([[-1.4,0,0],[0,0,0],[1.4,0,0]])
    o=bond_entrance_observables(q,[M]*3,BETA)
    assert o["R_bohr"]==pytest.approx(2.1)
    assert o["sigma_bohr"][0]==pytest.approx(0)
    assert list(o["target_topology"])==[True,False]
    assert o["target_speed"][0]==pytest.approx(np.sqrt(6/M/(2*np.pi*BETA)))
    assert bond_entrance_observables(np.repeat(q[None],16,axis=0),[M]*3,BETA)["reference_speed"]==pytest.approx(o["reference_speed"])


def test_same_ensemble_shell_ratio_and_gauge_invariance():
    kw=fixture()
    out=entrance_flux_ratio(**kw)
    assert out["flux_ratio_each"]==pytest.approx([2.0])
    assert out["total_qtst_au"]==pytest.approx(2*.9873012308830208)
    assert out["effective_reference_samples"]==pytest.approx(200)
    assert out["effective_target_samples"]==pytest.approx([200])
    assert not out["converged"]
    kw["weights"]*=1e-300
    with np.errstate(under="ignore"):
        assert entrance_flux_ratio(**kw)["total_qtst_au"]==pytest.approx(out["total_qtst_au"])


def test_reject_unmatched_measures_and_disconnected_umbrellas():
    kw=fixture()
    kw["measure"]="cv_density"
    with pytest.raises(ValueError,match="joint_unbiased"):
        entrance_flux_ratio(**kw)
    kw["measure"]="joint_unbiased_canonical";kw["connected"]=False
    with pytest.raises(ValueError,match="joint_unbiased"):
        entrance_flux_ratio(**kw)
    kw["connected"]=True;kw["canonical_identity"]="short"
    with pytest.raises(ValueError,match="canonical_identity"):
        entrance_flux_ratio(**kw)


def test_fail_closed_on_missing_or_low_ess_entrance():
    kw=fixture();kw["reference_coordinate"][:]=8.
    with pytest.raises(ValueError,match="missing entrance"):
        entrance_flux_ratio(**kw)
    kw=fixture();kw["target_coordinates"][:]=2.
    with pytest.raises(ValueError,match="missing entrance"):
        entrance_flux_ratio(**kw)
    kw=fixture();kw["weights"][1:200]=1e-10
    with pytest.raises(ValueError,match="insufficient effective"):
        entrance_flux_ratio(**kw)


def test_reject_overlapping_exit_sectors():
    kw=fixture()
    kw["target_coordinates"]=np.tile(kw["target_coordinates"],(1,2))
    kw["target_speed"]=np.ones((400,2))
    kw["target_topology"]=np.ones((400,2),bool)
    with pytest.raises(ValueError,match="overlapping"):
        entrance_flux_ratio(**kw)


def test_finite_bond_pmf_checkpoint_ensemble_has_no_Rinf_support():
    q=np.repeat(np.array([[[-1.4,0,0],[0,0,0],[1.4,0,0]]]),64,axis=0)
    with pytest.raises(ValueError,match="missing entrance"):
        h3_entrance_from_centroid_samples(q,np.ones(64),[M]*3,BETA,
            radius_bohr=10,reference_width_bohr=.2,target_width_bohr=.1,
            canonical_identity=IDENTITY,connected=True,min_effective=2)


def test_bridge_anchors_within_same_fes_without_box_volume():
    kw=fixture();bridge=entrance_flux_ratio(**kw)
    grid=np.zeros((4,4))
    edges=[np.linspace(0,2,5)]*2
    diagonal=[[.25,.25],[1.75,1.75]]
    result=bond_pmf_rate_from_entrance_bridge(grid,edges,beta=BETA,
        curve=diagonal,positive_speed=.001,
        anchor_curve=diagonal,anchor_positive_speed=.001,
        bridge=bridge,canonical_identity=IDENTITY)
    assert result["k_tst_au"]==pytest.approx(bridge["rate_each_au"][0])
    assert not result["converged"]
    with pytest.raises(ValueError,match="same canonical"):
        bond_pmf_rate_from_entrance_bridge(grid,edges,beta=BETA,
            curve=diagonal,positive_speed=.001,
            anchor_curve=diagonal,anchor_positive_speed=.001,
            bridge=bridge,canonical_identity="not_matching")
    with pytest.raises(ValueError,match="anchor must"):
        bond_pmf_rate_from_entrance_bridge(grid,edges,beta=BETA,
            curve=diagonal,positive_speed=.001,
            anchor_curve=[[0,1],[1,1]],anchor_positive_speed=.001,
            bridge=bridge,canonical_identity=IDENTITY)


def test_reactant_identity_is_required_for_reference_shell():
    kw=fixture()
    kw["reference_reactant_mask"][:]=False
    with pytest.raises(ValueError,match="missing entrance"):
        entrance_flux_ratio(**kw)
    kw=fixture()
    kw["reference_reactant_mask"]=None
    with pytest.raises(ValueError,match="reference_reactant_mask"):
        entrance_flux_ratio(**kw)
