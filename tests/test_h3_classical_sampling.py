"""Independent physical checks for the classical Jacobi sampling reference."""
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.h3.classical_sampling import (
    cartesian, coordinate_metric, radial_roots, flux_density,
    umbrella_moments, gaussian_ui_from_moments,
)
from benchmarks.h3.classical_tst import MASS_AU, quadrature
from pyrpmd.analysis.fes import umbrella_integration
from pyrpmd.constants import AU_RATE_TO_CM3_S, KB_AU
from pyrpmd.cv import RPMDrateCoordinate


class FlatPotential:
    def evaluate(self, q):
        return SimpleNamespace(energy=np.zeros(len(q)), status=np.zeros(len(q), dtype=int))


def test_jacobi_metric_matches_cartesian_coordinate():
    cv = RPMDrateCoordinate([[0],[1,2]], np.full(3,MASS_AU),
                            np.array([[-1.757,0,0],[0,0,0],[1.757,0,0]]),
                            [[0,1]], [[1,2]], 30., equivalent_paths=[[0,2,1]])
    rng = np.random.default_rng(79241)
    for rc, r, u in zip(rng.uniform(2.8,35,30), rng.uniform(1.2,2.1,30), rng.uniform(.05,.95,30)):
        q = cartesian(rc,r,u)
        x, derivative, metric = coordinate_metric(rc,r,u)
        expected, gradient = cv.centroid_value_gradient(q)
        np.testing.assert_allclose(x, expected, atol=1e-14, rtol=1e-13)
        np.testing.assert_allclose(metric, np.sum(gradient**2)/MASS_AU, atol=1e-16, rtol=1e-12)
        epsilon = 1e-4
        finite = (coordinate_metric(rc+epsilon,r,u)[0]-coordinate_metric(rc-epsilon,r,u)[0])/(2*epsilon)
        np.testing.assert_allclose(derivative, finite, atol=2e-11, rtol=1e-8)


@pytest.mark.parametrize('xi',[-.1,-.05,-.001,0.,.1,.5,.9,1.,1.015,1.05,1.1])
def test_inverse_coordinate_retains_only_unsquared_roots(xi):
    r,u=np.meshgrid(np.linspace(.5,8.,37),np.linspace(.01,.99,23),indexing='ij')
    roots,valid=radial_roots(xi,r,u)
    assert valid.any()
    for rc,mask in zip(roots,valid):
        if mask.any():
            actual=coordinate_metric(rc[mask],r[mask],u[mask])[0]
            np.testing.assert_allclose(actual,xi,atol=2e-10,rtol=0)


def test_all_forward_roots_survive_inverse_mapping():
    rng=np.random.default_rng(9342)
    for _ in range(500):
        r=float(rng.uniform(.5,8.)); u=float(rng.uniform(.01,.99)); rc=float(rng.uniform(.1,35.))
        near=np.sqrt(rc*rc+r*r/4-rc*r*u)
        if 30-rc-r+near <= 0:
            continue
        x=coordinate_metric(rc,r,u)[0]
        roots,valid=radial_roots(x,r,u)
        assert np.any(np.isclose(roots[valid],rc,atol=1e-8,rtol=1e-9))


def test_flat_reactant_density_has_exact_cartesian_measure():
    beta=1/(KB_AU*300.)
    rho,_=flux_density(FlatPotential(),[0.],radial_order=24,angular_order=24)
    expected=30**2*(8.**3-.5**3)/3/np.sqrt(2*np.pi*beta*(2*MASS_AU/3))
    np.testing.assert_allclose(rho[0],expected,rtol=2e-14)


def test_flat_conditional_ratio_matches_independent_surface_integral():
    beta=1/(KB_AU*300.)
    rho,_=flux_density(FlatPotential(),[0.,1.],radial_order=64,angular_order=64)
    prefactor=4*np.pi*30**2/np.sqrt(2*np.pi*beta*(2*MASS_AU/3))
    rate=prefactor*rho[1]/rho[0]*AU_RATE_TO_CM3_S
    independent=quadrature(FlatPotential(),order=64)['classical_tst_cm3_molecule_s']
    np.testing.assert_allclose(rate,independent,rtol=1e-12)


def test_umbrella_moments_of_exact_gaussian_density(monkeypatch):
    # PyRPMD supports NumPy 1.26, before NumPy exposed this integration API.
    monkeypatch.delattr(np, 'trapezoid', raising=False)
    grid=np.linspace(-3,3,30001)
    centers=np.array([-.3,0.,.7])
    beta=2.; intrinsic=4.; stiffness=30.
    density=np.exp(-.5*beta*intrinsic*(grid-.2)**2)
    means,variances=umbrella_moments(grid,density,centers,beta=beta,stiffness=stiffness)
    np.testing.assert_allclose(means,(stiffness*centers+intrinsic*.2)/(stiffness+intrinsic),atol=2e-15)
    np.testing.assert_allclose(variances,1/(beta*(stiffness+intrinsic)),atol=2e-15)


def test_moment_ui_matches_production_estimator_and_exact_harmonic_pmf():
    beta=2.; stiffness=30.; intrinsic=4.
    centers=np.linspace(-1,1,101)
    means=(stiffness*centers+intrinsic*.2)/(stiffness+intrinsic)
    variances=np.full_like(means,1/(beta*(stiffness+intrinsic)))
    grid=np.linspace(-.5,.5,501)
    actual,gradient=gaussian_ui_from_moments(means,variances,centers,grid,beta=beta,stiffness=stiffness)
    expected=.5*intrinsic*(grid-.2)**2
    np.testing.assert_allclose(actual,expected-expected[0],atol=1e-14)
    # Two values encode moments exactly; this checks the estimator, not sampling.
    values=[np.array([mu-np.sqrt(var),mu+np.sqrt(var)]) for mu,var in zip(means,variances)]
    production,_=umbrella_integration(values,centers,np.full_like(centers,stiffness),beta,grid)
    np.testing.assert_allclose(actual,production-production[0],atol=1e-14)
