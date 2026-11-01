"""Independent geometry, proposal and flux-normalization checks."""
from types import SimpleNamespace

import numpy as np
from scipy.integrate import quad
from scipy.stats import truncnorm,expon

from benchmarks.h3.classical_sampling_surface import (
    surface_positions,surface_coordinate,proposal,proposal_log_density,
    surface_moments,paired_flux_values,MU_R,SIGMA_R,SCALE_Y,
)
from benchmarks.h3.classical_tst import MASS_AU


class FlatPotential:
    def evaluate(self,q):
        return SimpleNamespace(energy=np.zeros(len(q)),status=np.zeros(len(q),dtype=int))


def test_surface_coordinates_and_mass_metric_match_production():
    rng=np.random.default_rng(90172)
    rt=np.column_stack((rng.uniform(.6,4.,100),rng.uniform(1.001,1.999,100)))
    q=surface_positions(rt)
    np.testing.assert_allclose(np.linalg.norm(q[:,0]-q[:,1],axis=1),rt[:,0],rtol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(q[:,1]-q[:,2],axis=1),rt[:,0],rtol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(q[:,0]-q[:,2],axis=1),rt[:,0]*rt[:,1],rtol=1e-14)
    surface=surface_coordinate()
    for frame,(_,t) in zip(q,rt):
        value,gradient=surface.centroid_value_gradient(frame)
        np.testing.assert_allclose(value,0.,atol=1e-14)
        np.testing.assert_allclose(np.sum(gradient*gradient)/MASS_AU,(2+t*t)/MASS_AU,rtol=1e-13)


def test_truncated_independence_proposal_density_matches_scipy():
    rt=proposal(np.random.default_rng(7192),1000)
    assert ((rt[:,0]>=.5)&(rt[:,0]<=8)&(rt[:,1]>=1)&(rt[:,1]<=2)).all()
    expected=(truncnorm.logpdf(rt[:,0],(.5-MU_R)/SIGMA_R,(8-MU_R)/SIGMA_R,loc=MU_R,scale=SIGMA_R)
              +expon.logpdf(2-rt[:,1],scale=SCALE_Y)-np.log(expon.cdf(1,scale=SCALE_Y)))
    np.testing.assert_allclose(proposal_log_density(rt),expected,atol=3e-14,rtol=1e-13)


def test_flat_surface_measure_moments_match_closed_form_and_adaptive_quadrature():
    moments=surface_moments(FlatPotential(),order=64)
    low,high=.5,8.
    r1=5/6*(high**6-low**6)/(high**5-low**5)
    r2=5/7*(high**7-low**7)/(high**5-low**5)
    normalization=(6**1.5-3**1.5)/3
    t1=quad(lambda t:t*t*np.sqrt(2+t*t),1,2,epsabs=1e-13)[0]/normalization
    t2=((6**2.5/5-2*6**1.5/3)-(3**2.5/5-2*3**1.5/3))/normalization
    np.testing.assert_allclose(moments,[r1,t1,r2,t2,0.],rtol=2e-14,atol=1e-13)


def test_antithetic_maxwell_flux_estimator_has_unit_ballistic_transmission():
    times=np.arange(1,10)
    for velocity in [-.7,.2,1.1]:
        values=np.stack((velocity*times,-velocity*times))
        numerator,denominator=paired_flux_values(velocity,.37,values)
        np.testing.assert_allclose(numerator/denominator,1.,rtol=0,atol=0)
        # Returning both children to the same side produces zero net flux.
        reflected=np.ones_like(values)
        numerator,_=paired_flux_values(velocity,.37,reflected)
        np.testing.assert_array_equal(numerator,0.)
