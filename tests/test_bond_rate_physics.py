"""Analytical two-dimensional flux and signed channel checks."""
import numpy as np
import pytest
from pyrpmd.analysis.bond_rate import bond_2d_qtst,bond_2d_signed_rate


def test_exact_flat_fes_normalization():
    f=np.zeros((2,2)); edges=[np.array([0,.5,1.]),np.array([0,.5,1.])]
    rt=bond_2d_qtst(f,edges,beta=1,curve=[[0,0],[1,1]],positive_speed=.2,
        reference_curve=[[0,.5],[1,.5]],reference_speed=.2,
        entrance_rate_au=3,surface_id="same",reference_surface_id="same")
    assert rt["k_tst_au"]==pytest.approx(3*np.sqrt(2),rel=1e-12)
    assert rt["converged"] is False


def test_reference_is_required_and_identical():
    f=np.zeros((2,2));e=[np.array([0,.5,1.])]*2
    kw=dict(beta=1,curve=[[0,0],[1,1]],positive_speed=.2,
        reference_curve=[[0,.5],[1,.5]],reference_speed=.2,
        entrance_rate_au=3,surface_id="a",reference_surface_id="b")
    with pytest.raises(ValueError,match="matching"):
        bond_2d_qtst(f,e,**kw)
    kw["reference_surface_id"]="a";kw["entrance_rate_au"]=None
    with pytest.raises(ValueError,match="bridge"):
        bond_2d_qtst(f,e,**kw)


def test_signed_two_channel_result_and_identity_guard():
    static={"k_tst_au":2.0,"surface_identity":"bond"}
    t=np.arange(1.,101.)
    numerator=np.zeros((4,100,2));numerator[:,:,0]=.3;numerator[:,:,1]=.2
    den=np.ones(4);ids=np.array([[0,0],[0,1],[1,0],[1,1]])
    r=bond_2d_signed_rate(static,t,numerator,den,ids,["left","right"],
                          surface_identity="bond",bootstrap=0)
    assert r["dynamics"]["total_kappa"]==pytest.approx(.5)
    assert r["dynamics"]["branching_fraction_estimates"]==pytest.approx([.6,.4])
    assert r["rate_cm3_s"] is None
    with pytest.raises(ValueError,match="do not match"):
        bond_2d_signed_rate(static,t,numerator,den,ids,["left","right"],surface_identity="xi")
