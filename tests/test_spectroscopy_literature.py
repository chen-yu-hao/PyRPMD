"""Independent numerical checks on the literature benchmark's reference tools."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import eigh_tridiagonal

from pyrpmd.analysis.spectroscopy import relative_ir_spectrum
from pyrpmd.cmd import fit_quadratic_centroid_pmf


@pytest.fixture(scope="module")
def benchmark():
    path = Path(__file__).resolve().parents[1] / "benchmarks/tier2_literature/spectroscopy.py"
    spec = importlib.util.spec_from_file_location("pyrpmd_spectroscopy_literature", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_harmonic_quantum_reference_and_kubo_sum_rule(benchmark):
    reference = benchmark.exact_reference(60, quartic=0.)
    times = np.linspace(0, 15, 301)
    assert reference["ground_energy"] == pytest.approx(.5, abs=1e-14)
    assert reference["fundamental"] == pytest.approx(1., abs=1e-14)
    assert reference["c0"] == pytest.approx(1 / 5., abs=2e-14)
    assert reference["q2"] == pytest.approx(.5 / np.tanh(2.5), abs=2e-14)
    assert reference["curvature"] == pytest.approx(-1 / 5., abs=2e-14)
    np.testing.assert_allclose(benchmark.exact_correlation(reference, times), np.cos(times) / 5., atol=2e-14)


def test_quartic_basis_reference_agrees_with_independent_real_space_solver(benchmark):
    reference = benchmark.exact_reference(100)
    more = benchmark.exact_reference(160)
    # An independent second-order real-space Dirichlet discretization, not the
    # oscillator matrices used by the report. Boundary tails are negligible.
    levels = []
    for intervals in (4000, 8000):
        grid = np.linspace(-8., 8., intervals + 1)[1:-1]
        dx = 16. / intervals
        diagonal = 1 / dx ** 2 + .5 * grid ** 2 + .1 * grid ** 4
        offdiagonal = np.full(len(grid) - 1, -.5 / dx ** 2)
        levels.append(eigh_tridiagonal(diagonal, offdiagonal, select="i", select_range=(0, 3), eigvals_only=True))
    # Richardson extrapolation removes the known O(dx^2) finite-difference
    # error; finite-basis and finite-grid convergence are distinct checks.
    extrapolated = (4 * levels[1] - levels[0]) / 3
    np.testing.assert_allclose(reference["energies"][:4], extrapolated, rtol=0, atol=2e-8)
    assert reference["fundamental"] == pytest.approx(more["fundamental"], abs=2e-11)
    assert reference["c0"] == pytest.approx(more["c0"], abs=2e-12)
    assert reference["curvature"] == pytest.approx(-.2, abs=2e-11)
    # Cao--Voth report a ground-state increase of 12% rounded to an integer.
    assert round(100 * (reference["ground_energy"] / .5 - 1)) == 12


def test_exact_initial_ring_sampler_has_correct_classical_and_centroid_moments(benchmark):
    rng = np.random.default_rng(42172)
    sampled, acceptance = benchmark.canonical_rings(1, 80_000, rng)
    boltzmann = lambda x: np.exp(-5 * (.5 * x * x + .1 * x ** 4))
    normalization = quad(boltzmann, -np.inf, np.inf)[0]
    expected = quad(lambda x: x * x * boltzmann(x), -np.inf, np.inf)[0] / normalization
    observed = sampled[0] ** 2
    error = np.std(observed, ddof=1) / np.sqrt(len(observed))
    assert abs(observed.mean() - expected) < 5 * error
    assert .7 < acceptance <= 1
    rings, _ = benchmark.canonical_rings(16, 20_000, rng)
    centroid = rings.mean(axis=0)
    centroid_force = (-rings - .4 * rings ** 3).mean(axis=0)
    products = centroid * centroid_force
    # Integration by parts in the centroid coordinate gives this exact finite-P
    # relation for the unbiased distribution, independently of the anharmonicity.
    error = np.std(products, ddof=1) / np.sqrt(len(products))
    assert abs(products.mean() + .2) < 5 * error


def test_windowed_exact_spectrum_matches_harmonic_quadrature_ensemble(benchmark):
    frames, dt, nfft = 911, .02, 4096
    reference = benchmark.exact_reference(40, quartic=0.)
    omega, expected = benchmark.expected_windowed_ir(reference, frames, dt, nfft)
    t = np.arange(frames) * dt
    dipoles = np.zeros((2, frames, 1, 3))
    dipoles[0, :, 0, 0] = np.sqrt(2 / 5.) * np.cos(t)
    dipoles[1, :, 0, 0] = np.sqrt(2 / 5.) * np.sin(t)
    actual = relative_ir_spectrum(dipoles, dt, beta=5., normalize=False, nfft=nfft)
    np.testing.assert_allclose(actual["angular_frequency_au"], omega)
    np.testing.assert_allclose(actual["unnormalized_relative_ir"], expected, rtol=3e-13, atol=1e-13)


def test_quadratic_force_matching_recovers_exact_harmonic_centroid_pmf(benchmark):
    values, _ = benchmark.canonical_rings(16, 1024 * 3, np.random.default_rng(1943), quartic=0.)
    rings = values.reshape(16, 1024, 1, 3).transpose(1, 0, 2, 3)
    fitted = fit_quadratic_centroid_pmf(rings, -rings, beta=5.)
    np.testing.assert_allclose(fitted.potential.hessian, np.eye(3), atol=2e-14)
    assert fitted.force_rmse < 2e-14


@pytest.fixture(scope="module")
def static_reference():
    path = Path(__file__).resolve().parents[1] / "benchmarks/tier2_literature/spectroscopy_static.py"
    spec = importlib.util.spec_from_file_location("pyrpmd_spectroscopy_static_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_transfer_static_p1_reduces_to_independent_classical_quadrature(static_reference):
    result = static_reference.finite_bead_covariance(1, points=301)
    boltzmann = lambda x: np.exp(-5 * (.5 * x * x + .1 * x ** 4))
    expected = quad(lambda x: x * x * boltzmann(x), -np.inf, np.inf)[0] / quad(boltzmann, -np.inf, np.inf)[0]
    assert result["bead_variance"] == pytest.approx(expected, abs=2e-12)
    assert result["centroid_variance"] == pytest.approx(expected, abs=2e-9)


def test_transfer_static_has_grid_and_bead_convergence(benchmark, static_reference):
    exact = benchmark.exact_reference(100)
    values = [static_reference.finite_bead_covariance(p, points=201) for p in (8, 16, 32)]
    finer = static_reference.finite_bead_covariance(32, points=401)
    assert abs(values[-1]["centroid_variance"] - finer["centroid_variance"]) < 2e-8
    for key, reference in (("centroid_variance", exact["c0"]), ("bead_variance", exact["q2"])):
        errors = [abs(row[key] - reference) for row in values]
        assert 3 < errors[0] / errors[1] < 5
        assert 3 < errors[1] / errors[2] < 5
