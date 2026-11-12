"""Independent 2-D thermal-string reference: continuous Müller--Brown samples.

This checks a fixed path's finite-restraint free energy F_K, not optimization of
that path or convergence to the unrestrained Müller--Brown potential.  Samples
come from exact rejection sampling, not MD.  A separate quadrature measures the
path-integration error; block statistics quantify sampling error only.
"""

import numpy as np
from numpy.polynomial.hermite import hermgauss

from pyrpmd.analysis.mbar import solve_mbar
from pyrpmd.finite_temperature_string import (
    finite_temperature_string,
    restrained_string_statistics,
)


def _potential(points):
    """Standard Müller--Brown surface scaled by 0.01 in atomic energy units."""
    dx = points[..., 0, None] - np.array([1.0, 0.0, -0.5, -1.0])
    dy = points[..., 1, None] - np.array([0.0, 0.5, 1.5, 1.0])
    exponent = (np.array([-1.0, -1.0, -6.5, 0.7]) * dx**2
                + np.array([0.0, 0.0, 11.0, 0.6]) * dx * dy
                + np.array([-10.0, -10.0, -6.5, 0.7]) * dy**2)
    return (np.array([-2.0, -1.0, -1.7, 0.15]) * np.exp(exponent)).sum(axis=-1)


def _path(t):
    return np.stack((-0.558 + 1.181 * t,
                     1.442 - 1.414 * t - 0.25 * np.sin(np.pi * t)), axis=-1)


def _quadrature(images, beta, stiffness, order):
    """Integrate the continuous restrained distribution on all of R^2.

    The harmonic umbrella is the Gauss--Hermite weight.  Its normalization is
    common to all images, so it cancels in F_K(z)-F_K(z_0).  This reference does
    not call the string reducer or integrate a sampled force.
    """
    nodes, weights = hermgauss(order)
    offsets = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), axis=-1)
    offsets *= np.sqrt(2.0 / (beta * stiffness))
    product_weights = weights[:, None] * weights[None, :] / np.pi
    free_energy, gradients = [], []
    for image in images:
        points = image + offsets
        density = product_weights * np.exp(-beta * (_potential(points) + 4.7))
        partition = density.sum()
        mean = np.sum(points * density[..., None], axis=(0, 1)) / partition
        free_energy.append(-np.log(partition) / beta)
        gradients.append(stiffness * (image - mean))
    free_energy = np.asarray(free_energy)
    return free_energy - free_energy[0], np.asarray(gradients)


def _independent_samples(images, beta, stiffness, count, rng):
    """Exact continuous rejection sampling with a rigorous global envelope.

    The first three Gaussian quadratic forms are negative definite.  Their
    contributions are therefore >= -2, -1, -1.7; the fourth is positive.
    Hence V >= -4.7 everywhere.  A harmonic-umbrella Gaussian proposal accepted
    with exp[-beta*(V+4.7)] has exactly the desired restrained density.
    """
    samples = []
    for image in images:
        pieces, retained = [], 0
        while retained < count:
            proposal = image + rng.normal(size=(32768, 2)) / np.sqrt(beta * stiffness)
            acceptance = np.exp(-beta * (_potential(proposal) + 4.7))
            assert np.all(acceptance <= 1.0)
            accepted = proposal[rng.random(len(proposal)) < acceptance]
            pieces.append(accepted)
            retained += len(accepted)
        samples.append(np.concatenate(pieces)[:count])
    return np.asarray(samples)


def _trapezoid_profile(images, gradients):
    increments = np.sum(np.diff(images, axis=0) * (gradients[1:] + gradients[:-1]) / 2, axis=1)
    return np.concatenate(([0.0], np.cumsum(increments)))


def test_muller_brown_thermal_string_matches_quadrature_and_umbrella_mbar():
    beta, stiffness, sample_count = 0.7, 8.0, 20000
    images = _path(np.linspace(0.0, 1.0, 17))
    exact, exact_gradient = _quadrature(images, beta, stiffness, 128)
    lower_order, lower_gradient = _quadrature(images, beta, stiffness, 96)
    # Integration of the continuous restrained measure is converged independently
    # of the stochastic test and is much more accurate than its sampling error.
    np.testing.assert_allclose(exact, lower_order, rtol=0, atol=2e-6)
    np.testing.assert_allclose(exact_gradient, lower_gradient, rtol=0, atol=2e-5)

    refined_images = _path(np.linspace(0.0, 1.0, 65))
    _, refined_gradient = _quadrature(refined_images, beta, stiffness, 128)
    coarse_profile = _trapezoid_profile(images, exact_gradient)
    refined_profile = _trapezoid_profile(refined_images, refined_gradient)[::4]
    discretization_error = np.abs(coarse_profile - exact)
    assert discretization_error.max() < 0.01
    assert np.max(np.abs(refined_profile - exact)) < discretization_error.max() / 8

    values = _independent_samples(images, beta, stiffness, sample_count,
                                  np.random.default_rng(11029))
    # Cartesian x,y and unit mass have the identity CV metric.  Frames and
    # windows are independent; contiguous blocks still exercise the public
    # reducer's normal covariance and correlation-diagnostic path.
    metrics = np.broadcast_to(np.eye(2), values.shape[:2] + (2, 2))
    statistics = restrained_string_statistics(images, values, metrics, stiffness,
                                               block_size=100)

    def sampler(requested_images):
        np.testing.assert_array_equal(requested_images, images)
        return statistics

    # A large tolerance evaluates the chosen geometry without moving it.  The
    # geometric stop is not evidence of a relaxed path or physical convergence.
    result = finite_temperature_string(images, sampler, tolerance=1e6, max_iterations=1)
    assert result.iterations == 0
    assert result.statistics.correlation_resolved
    np.testing.assert_array_equal(result.images, images)
    assert np.all(np.abs(result.free_energy - coarse_profile)
                  <= 4 * result.free_energy_standard_error + 2e-6)
    assert np.all(np.abs(result.free_energy - exact)
                  <= 4 * result.free_energy_standard_error + discretization_error + 2e-6)

    # Existing umbrella MBAR estimates the same restrained state partition
    # functions.  Independent batches provide an uncertainty on their mean
    # without importing a second MBAR implementation or inventing an SEM.
    mbar_profiles = []
    for batch in np.split(values, 20, axis=1):
        joined = batch.reshape(-1, 2)
        reduced_biases = beta * stiffness / 2 * np.sum(
            (joined[None] - images[:, None])**2, axis=-1)
        fit = solve_mbar(reduced_biases, np.full(len(images), batch.shape[1]),
                         return_state_weights=False)
        assert fit.converged
        assert len(np.unique(fit.components)) == 1
        mbar_profiles.append((fit.free_energies - fit.free_energies[0]) / beta)
    mbar_profiles = np.asarray(mbar_profiles)
    mbar_mean = mbar_profiles.mean(axis=0)
    mbar_error = mbar_profiles.std(axis=0, ddof=1) / np.sqrt(len(mbar_profiles))
    assert np.all(np.abs(mbar_mean - exact) <= 4 * mbar_error + 2e-6)
    # Both estimates reuse samples, so summing their SEMs is conservative and
    # does not assume independence between the string and MBAR estimators.
    assert np.all(np.abs(result.free_energy - mbar_mean)
                  <= 4 * (result.free_energy_standard_error + mbar_error) + discretization_error + 2e-6)

    # A finite K smooths the potential substantially here.  Agreement with F_K
    # must never be reported as recovery of the unsmoothed Cartesian PMF V.
    unsmoothed = _potential(images) - _potential(images[:1])
    assert np.max(np.abs(exact - unsmoothed)) > 0.1
