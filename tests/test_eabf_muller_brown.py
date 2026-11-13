"""CZAR conditional-distribution oracle on the 2D Muller--Brown surface.

This is an estimator acceptance test, not an eABF trajectory or mixing test.
For a fixed, integrable auxiliary bias B(lambda)=-b.lambda and diagonal
harmonic coupling kappa, the exact conditional distribution is
lambda|z ~ N(z+b/kappa, diag(1/(beta*kappa))). Thus
E[kappa*(lambda-z)|z]=b and rho_tilde(z) is proportional to
exp[-beta*(V_MB(z)-b.z)]. Tensor Gauss--Legendre quadrature supplies bin
populations and an independent unbiased bin-integrated PMF. Large integer
oracle counts remove count-rounding noise; they are not MD observations or
evidence of statistical precision. CZAR's differentiation/projection error
is checked separately from MBAR's bootstrap sampling uncertainty.

Umbrella samples are independent draws from the continuous reference
distributions by rejection sampling, not samples from a discretized grid.
All windows share the same finite rectangular domain; its normalization
changes only the free-energy gauge within that domain.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss

from pyrpmd.abf import ExtendedABF
from pyrpmd.analysis.fes import reconstruct_fes
from pyrpmd.cv import CVStack, Cartesian


_BETA = 1.0
_TILT = np.array([0.35, -0.25])
_BOUNDS = np.array([[-1.25, 0.75], [0.0, 1.8]])


def _energy(points):
    """Four Gaussian terms; original MB amplitudes scaled by 0.01 hartree."""
    displacement = np.asarray(points)[..., None, :] - np.array(
        [[1.0, 0.0], [0.0, 0.5], [-0.5, 1.5], [-1.0, 1.0]]
    )
    x, y = displacement[..., 0], displacement[..., 1]
    exponent = (np.array([-1.0, -1.0, -6.5, 0.7]) * x**2
                + np.array([0.0, 0.0, 11.0, 0.6]) * x * y
                + np.array([-10.0, -10.0, -6.5, 0.7]) * y**2)
    return np.sum(np.array([-2.0, -1.0, -1.7, 0.15]) * np.exp(exponent), axis=-1)


def _bin_density(edges, *, order):
    """Integrate biased and unbiased density independently in every 2D bin."""
    axes = [(edge[:-1] + edge[1:]) / 2 for edge in edges]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    widths = np.array([np.diff(edge)[0] for edge in edges])
    nodes, weights = leggauss(order)
    unbiased = np.zeros(centers.shape[:-1])
    biased = np.zeros_like(unbiased)
    for i, x in enumerate(nodes):
        for j, y in enumerate(nodes):
            points = centers + 0.5 * widths * [x, y]
            density = np.exp(-_BETA * _energy(points))
            weight = weights[i] * weights[j] / 4
            unbiased += weight * density
            biased += weight * density * np.exp(_BETA * (points @ _TILT))
    # These are bin-averaged densities. Multiplication by the constant cell
    # area produces bin masses; the normalization below cancels that area.
    return unbiased, biased


def _czar_oracle(bins):
    edges = tuple(np.linspace(lo, hi, bins + 1) for lo, hi in _BOUNDS)
    unbiased, biased = _bin_density(edges, order=8)
    check, _ = _bin_density(edges, order=5)
    quadrature_error = np.max(np.abs(np.log(unbiased / check))) / _BETA
    assert quadrature_error < 1e-8
    model = ExtendedABF(
        CVStack([Cartesian(0, "x"), Cartesian(0, "y")]), edges, _BETA, [0.0, 0.8],
        coupling=[4.0, 7.0], mass=[2.0, 3.0], friction=0.2,
    )
    # Analytically integrated conditional lambda populations, deliberately
    # supplied as an oracle rather than claimed to result from dynamics.
    model.physical_grid.count[:] = np.rint(biased / biased.sum() * 1e12).astype(np.int64)
    model.physical_grid.mean[:] = _TILT
    result = model.fes_result(min_count=100)
    exact = -np.log(unbiased) / _BETA
    assert result.metadata["estimator"] == "czar"
    assert np.isnan(result.uncertainty).all()
    assert result.mask[[0, -1], :].all() and result.mask[:, [0, -1]].all()
    assert not result.mask[1:-1, 1:-1].any()
    return result, exact, quadrature_error


def _independent_umbrella_samples(rng, center, stiffness, size):
    """Exact IID continuous reference draws using a proven rejection bound.

    The three negative-amplitude Gaussian quadratic forms are nonpositive,
    so each exponential is <=1; the fourth term and umbrella are nonnegative.
    Consequently V_MB+W >= -2-1-1.7=-4.7 everywhere, supplying a true bound.
    """
    samples = []
    remaining = size
    while remaining:
        points = rng.uniform(_BOUNDS[:, 0], _BOUNDS[:, 1], size=(100000, 2))
        total_energy = _energy(points) + 0.5 * stiffness * np.sum((points - center)**2, axis=-1)
        log_acceptance = -_BETA * (total_energy + 4.7)
        assert np.max(log_acceptance) <= 0.0
        accepted = points[np.log(rng.random(len(points))) < log_acceptance][:remaining]
        samples.append(accepted)
        remaining -= len(accepted)
    return np.concatenate(samples)


def test_czar_muller_brown_matches_bin_quadrature_and_independent_umbrella_mbar():
    coarse, exact, quadrature_error = _czar_oracle(24)
    fine, exact_fine, _ = _czar_oracle(48)
    numerical_errors = []
    for result, reference, bound in ((coarse, exact, 0.06), (fine, exact_fine, 0.02)):
        error = result.free_energy - reference
        middle = tuple(size // 2 for size in reference.shape)
        error -= error[middle]
        assert np.nanmax(np.abs(error)) < bound
        numerical_errors.append(np.sqrt(np.nanmean(error**2)))
    # Halving both grid spacings resolves the expected O(h^2) finite-bin /
    # centered-difference / trapezoidal-projection error. It is not an SEM.
    assert numerical_errors[1] < 0.35 * numerical_errors[0]

    centers = np.array([[-0.75, 0.35], [0.25, 0.35], [-0.75, 1.35], [0.25, 1.35]])
    stiffness = 1.8
    rng = np.random.default_rng(41623)
    samples = [_independent_umbrella_samples(rng, center, stiffness, 6000) for center in centers]
    mbar = reconstruct_fes(
        samples, centers, stiffness * np.eye(2), _BETA, bins=coarse.edges,
        ranges=_BOUNDS.tolist(), min_count=10, bootstrap=24, seed=491,
    )
    assert mbar["connected"]
    assert mbar["bootstrap_replicates"] == 24
    reference = tuple(mbar["bootstrap_reference_bin"])
    assert not coarse.mask[reference]
    supported = ~coarse.mask & ~mbar["mask"] & np.isfinite(mbar["standard_error"])
    assert supported.sum() >= 350
    exact_relative = exact - exact[reference]
    czar_relative = coarse.free_energy - coarse.free_energy[reference]
    mbar_relative = mbar["free_energy"] - mbar["free_energy"][reference]
    # Bootstrap errors already use this same fixed reference. Their sampling
    # error must not be mixed with CZAR's independently measured grid bias.
    grid_error = np.abs(czar_relative - exact_relative) + 2 * quadrature_error
    assert np.nanmax(grid_error) < 0.1
    statistical_error = 5 * mbar["standard_error"]
    # A simultaneous five-SEM guard is used for hundreds of grid comparisons;
    # it is a seeded regression test, not a calibrated confidence certificate.
    assert np.all(np.abs(mbar_relative[supported] - exact_relative[supported])
                  < statistical_error[supported] + 2 * quadrature_error)
    assert np.all(np.abs(czar_relative[supported] - mbar_relative[supported])
                  < statistical_error[supported] + grid_error[supported])
