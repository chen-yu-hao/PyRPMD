"""Two-section harmonic Q against an independent cyclic Gaussian determinant."""
import math

import numpy as np
import pytest

from pyrpmd.golden_rule_partition import harmonic_segment_log_partition


def cyclic_gaussian_log_partition(omega, mass, n0, n1, tau0, tau1):
    steps = [tau0/n0]*n0 + [tau1/n1]*n1
    p = len(steps)
    h = np.zeros((p, p))
    for j, dtau in enumerate(steps):
        k = (j+1) % p
        # Add each physical link separately, including both links when P=2.
        h[j, j] += mass/dtau + 0.5*mass*omega**2*dtau
        h[k, k] += mass/dtau + 0.5*mass*omega**2*dtau
        h[j, k] -= mass/dtau
        h[k, j] -= mass/dtau
    sign, logdet = np.linalg.slogdet(h)
    assert sign == 1
    return 0.5*(p*math.log(mass)-sum(math.log(dtau) for dtau in steps)-logdet)


@pytest.mark.parametrize("n0,n1,tau0,tau1", [(1, 1, 0.2, 1.8), (1, 3, 0.7, 2.1),
                                             (2, 5, 1.8, 0.3), (3, 2, 3.0, 0.2), (8, 1, 0.4, 2.0)])
def test_matched_partition_matches_direct_cyclic_gaussian_determinant(n0, n1, tau0, tau1):
    frequencies = [0.7, 1.3, 2.2]
    expected = math.log(3) + sum(cyclic_gaussian_log_partition(w, 2.7, n0, n1, tau0, tau1)
                                for w in frequencies)
    actual = harmonic_segment_log_partition(frequencies, n0, n1, tau0, tau1, electronic_degeneracy=3)
    assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("n0,n1,step", [(1, 1, 0.1), (3, 8, 0.5), (17, 32, 2.0)])
def test_equal_time_steps_recover_uniform_primitive_partition(n0, n1, step):
    omega = 1.7
    a = (n0+n1)*math.asinh(omega*step/2)
    expected = -(a+math.log1p(-math.exp(-2*a)))
    actual = harmonic_segment_log_partition([omega], n0, n1, n0*step, n1*step)
    assert actual == pytest.approx(expected, abs=1e-12)


def test_extreme_temperature_frequency_products_remain_in_logarithmic_range():
    # omega*dt would overflow, although log Q remains modest and representable.
    omega, step = 1e200, 1e200
    large = harmonic_segment_log_partition([omega], 7, 11, 7*step, 11*step)
    expected = -18*(math.log(omega)+math.log(step))
    assert math.isfinite(large)
    assert large == pytest.approx(expected, abs=1e-9)
    # omega*dt would underflow: recover the classical 1/(beta*omega) limit.
    omega, step = 1e-250, 1e-250
    small = harmonic_segment_log_partition([omega], 7, 11, 7*step, 11*step)
    assert small == pytest.approx(-math.log(18*step)-math.log(omega), abs=1e-12)


@pytest.mark.parametrize("frequencies", [[0.0], [-1.0], [math.inf], [True], [[1.0]]])
def test_invalid_harmonic_frequencies_are_rejected(frequencies):
    with pytest.raises(ValueError, match="positive finite angular"):
        harmonic_segment_log_partition(frequencies, 2, 3, 1.0, 2.0)
