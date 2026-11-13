"""adaptive_quantum checks that need neither numba nor the example models."""
import numpy as np
import pytest

from pyrpmd.analysis.adaptive_quantum import matched_rate_defect, passive_embed, spectral_diagnostics


def test_harmonic_embedding_matches_schur_complement():
    w=np.linspace(0,3,100);s=.1-1j*w;K=np.array([[2.2]]);L=np.array([[-.4]])
    relaxed=1.3-L[0,0]**2/K[0,0];Q=1/(s*s+relaxed)
    actual=passive_embed(Q,w,.1,K,L)
    full=np.array([[1.3,-.4],[-.4,2.2]])
    expected=np.linalg.inv(s[:,None,None]**2*np.eye(2)[None]+full)[:,0,0]
    np.testing.assert_allclose(actual,expected,atol=1e-12)

def test_absorption_not_real_part():
    w=np.linspace(.1,3,100);chi=1/(1+(.05-1j*w)**2)
    assert np.any(chi.real<0)
    assert spectral_diagnostics(chi,w)['negative_absorption_fraction']==0

def test_passive_embed_rejects_invalid_bath():
    with pytest.raises(ValueError):passive_embed(np.ones(3),[0,1,2],.1,[[-1]],[[.1]])

def test_rate_defect_limits_and_failure_visibility():
    kw=dict(beta=8.,observable='centroid(q)>0',units='length/time')
    assert matched_rate_defect(2.,1.,1.,**kw)['rate']==2.
    assert matched_rate_defect(2.,2.,3.,**kw)['rate']==3.
    r=matched_rate_defect(1.,3.,1.,**kw)
    assert r['rate']==-1. and r['negative_rate']


def test_rate_defect_ratio_mode_limits():
    kw = dict(beta=8., observable='centroid(q)>0', units='length/time', mode='ratio')
    assert matched_rate_defect(2., 1., 1., **kw)['rate'] == 2.
    assert matched_rate_defect(2., 2., 3., **kw)['rate'] == 3.
    # A prefactor shared by both active-space rates cancels.
    assert matched_rate_defect(2., 1e-6, 3e-6, **kw)['rate'] == pytest.approx(6.)
    assert matched_rate_defect(1., 3., 1., **kw)['negative_rate'] is False
    with pytest.raises(ValueError):
        matched_rate_defect(1., 0., 1., **kw)
    with pytest.raises(ValueError):
        matched_rate_defect(1., 1., 1., beta=1., observable='x', units='u', mode='other')
