from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from pyrpmd.analysis.curvilinear_quantum import solve_curvilinear
from pyrpmd.analysis.curvilinear_propagation import propagate_curvilinear


def reference():
    def geometry(u):
        jac=1+.3/np.cosh(u[:,0])**2
        return (1/jac**2)[:,None,None],jac
    return solve_curvilinear([np.linspace(-7,7,100)],geometry,
        lambda u:.5*(u[:,0]+.3*np.tanh(u[:,0]))**2,states=4)


def test_propagation_conserves_noncartesian_norm():
    r=reference();psi=(r.vectors[:,0]+1j*r.vectors[:,1])/np.sqrt(2)
    out=propagate_curvilinear(r,psi,np.linspace(0,20,101),hbar=1)
    assert out['norm_drift']<1e-12 and out['projection_loss']<1e-12
    assert np.max(abs(out['wavefunctions'][0]-psi))<1e-12


def test_missing_eigenstates_are_not_renormalized_away():
    r=reference();psi=np.random.default_rng(44).normal(size=r.vectors.shape[0]);psi/=np.sqrt(psi@(r.mass_matrix@psi))
    with pytest.raises(ValueError,match='loses too much'):propagate_curvilinear(r,psi,[0,1],hbar=1)


def test_wrong_euclidean_normalization_rejected():
    r=reference();psi=r.vectors[:,0];psi/=np.linalg.norm(psi)
    with pytest.raises(ValueError,match='mass measure'):propagate_curvilinear(r,psi,[0,1],hbar=1)
