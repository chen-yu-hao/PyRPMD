"""Independent analytic checks of direct H3 classical transition-state integration."""
import numpy as np
import pytest

from benchmarks.h3.classical_tst import quadrature
from pyrpmd.constants import KB_AU
from pyrpmd.potentials import FreePotential


def test_free_bounded_channel_matches_closed_form_integral():
    # Flat PES on this finite radial domain has an elementary integral. This
    # checks both equivalent-channel factors and both coordinate Jacobians.
    temperature, mass, lower, upper = 450.0, 1900.0, .5, 4.0
    result = quadrature(FreePotential(), temperature=temperature, mass=mass,
                        order=32, domain=(lower, upper))
    radial_numerator = (upper**5-lower**5)/5
    radial_denominator = (upper**3-lower**3)/3
    angular = (6**1.5-3**1.5)/3
    expected = 4*np.pi*radial_numerator/radial_denominator*angular*np.sqrt(KB_AU*temperature/(2*np.pi*mass))
    assert result["rate_au"] == pytest.approx(expected, rel=2e-14)


def test_mass_and_temperature_scalings_of_flat_channel():
    base = quadrature(FreePotential(), order=16)
    isotope = quadrature(FreePotential(), order=16, mass=4*base["equal_nuclear_mass_au"])
    hotter = quadrature(FreePotential(), order=16, temperature=1200.)
    assert isotope["rate_au"] == pytest.approx(base["rate_au"]/2, rel=2e-14)
    assert hotter["rate_au"] == pytest.approx(2*base["rate_au"], rel=2e-14)


@pytest.mark.parametrize("t", [1.05, 1.5, 1.95])
def test_analytic_transition_flux_metric_matches_cartesian_finite_differences(t):
    r, mass = 1.8, 1837.0
    cosine = 1-t*t/2
    q = np.array([[r, 0., 0.], [0., 0., 0.], [r*cosine, r*np.sqrt(1-cosine*cosine), 0.]])
    def s(x):
        return np.linalg.norm(x[1]-x[2])-np.linalg.norm(x[0]-x[1])
    gradient = np.empty((3, 3))
    for index in np.ndindex(q.shape):
        a, b = q.copy(), q.copy()
        a[index] += 1e-5; b[index] -= 1e-5
        gradient[index] = (s(a)-s(b))/2e-5
    assert np.sum(gradient**2)/mass == pytest.approx((2+t*t)/mass, rel=5e-11)


@pytest.mark.parametrize("kwargs", [{"domain":(.2,8.)}, {"domain":(.5,.5)},
                                     {"temperature":0.}, {"mass":float("nan")},
                                     {"order":True}, {"order":3}])
def test_invalid_or_unphysical_integration_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        quadrature(FreePotential(), **kwargs)
