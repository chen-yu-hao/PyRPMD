"""Numerical checks for the H3 bond-diagonal flux metric."""
import numpy as np
import pytest
from pyrpmd.analysis.bond_flux import bond_diagonal_flux_metric,bond_difference_gradient
M=1837.1526474396187
BETA=210.51668320265807

def test_collinear_half_maxwell():
    q=np.array([[-1.4,0,0],[0,0,0],[1.4,0,0]])
    r=bond_diagonal_flux_metric(q,[M]*3,BETA)
    assert r["sigma_bohr"]==pytest.approx(0,abs=1e-12)
    assert r["inverse_normal_mass_au"]==pytest.approx(3/M)
    assert r["positive_normal_speed_bohr_per_au_time"]==pytest.approx(np.sqrt(3/(2*np.pi*M*BETA)))

def test_bend_changes_metric():
    q=np.array([[0.,0,0],[1,0,0],[1,1,0]])
    assert bond_diagonal_flux_metric(q,[M]*3,BETA)["inverse_normal_mass_au"]==pytest.approx(2/M)

def test_no_spurious_bead_factor():
    q=np.array([[-1.,0,0],[0,0,0],[1,0,0]])
    assert bond_diagonal_flux_metric(q,[M]*3,BETA)==bond_diagonal_flux_metric(np.broadcast_to(q,(16,3,3)),[M]*3,BETA)

def test_finite_difference_gradient():
    q=np.array([[-1.2,0.1,0],[0,0,0],[0.8,-0.6,0]])
    _,g=bond_difference_gradient(q)
    for i in range(3):
        for j in range(3):
            plus=q.copy();minus=q.copy()
            plus[i,j]+=1e-6;minus[i,j]-=1e-6
            vp,_=bond_difference_gradient(plus)
            vm,_=bond_difference_gradient(minus)
            assert g[i,j]==pytest.approx((vp-vm)/2e-6,abs=1e-8)

def test_invalid_masses_and_degenerate_bonds():
    with pytest.raises(ValueError,match="undefined"):
        bond_diagonal_flux_metric(np.zeros((3,3)),[M]*3,BETA)
    with pytest.raises(ValueError,match="masses"):
        bond_diagonal_flux_metric(np.eye(3),[M,0,M],BETA)
