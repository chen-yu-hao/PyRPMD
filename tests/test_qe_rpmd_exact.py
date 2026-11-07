"""QE-RPMD exact quantum regression tests.

These tests deliberately use PyRPMD's existing sparse SpinBosonHEOM as the
reference implementation.  The generic DVR solver is checked against its
reduced density dynamics with the same Hamiltonian, coupling operator,
Matsubara coefficients and hierarchy depth; no fitted parameters are adjusted
to make the comparison agree.
"""

import numpy as np

from pyrpmd.analysis.dvr_heom import DVRSystemHEOM
from pyrpmd.analysis.heom import SpinBosonHEOM
from pyrpmd.analysis.spectral_density import (
    analyze_kubo_tcf,
    fit_exponential_correlation,
    harmonic_bath_kubo_tcf,
    spectral_density_from_kubo,
)


def test_generic_dvr_heom_matches_existing_spin_boson_heom():
    """The generic multi-exponential equations reproduce the established solver."""
    reference = SpinBosonHEOM(
        epsilon=0.4,
        delta=0.8,
        reorganization_energy=0.12,
        cutoff=1.3,
        beta=1.7,
        max_depth=2,
        matsubara_terms=1,
        terminator=False,
        coupling_scale=0.65,
    )
    times = np.linspace(0.0, 3.0, 31)
    rho0 = np.array([[0.7, 0.15j], [-0.15j, 0.3]], dtype=complex)
    expected = reference.propagate(rho0, times)

    generic = DVRSystemHEOM(
        grid=np.array([-1.0, 1.0]),
        potential=np.zeros(2),
        coupling=np.diag(reference.coupling).real,
        beta=reference.beta,
        exponents=tuple(zip(reference.coefficients, reference.frequencies)),
        depth=reference.max_depth,
        hamiltonian=reference.hamiltonian,
    )
    actual = generic.propagate(times, rho0, rtol=2e-11, atol=2e-13)

    np.testing.assert_allclose(actual.rho, expected.density, rtol=0.0, atol=3e-9)
    assert actual.success
    assert actual.trace_error < 2e-10
    assert actual.hermiticity_error < 2e-10


def test_generic_dvr_heom_accepts_callable_potential_coupling_and_counterterm():
    """The DVR constructor evaluates arbitrary system functions consistently."""
    grid = np.linspace(-2.0, 2.0, 9)
    exponents = ((0.08 - 0.03j, 1.1), (0.02, 3.2))
    solver = DVRSystemHEOM(
        grid,
        potential=lambda x: 0.5 * x**2,
        coupling=lambda x: x + 0.2 * x**2,
        beta=2.0,
        exponents=exponents,
        counterterm=0.15,
        depth=2,
    )
    rho0 = np.zeros((grid.size, grid.size), dtype=complex)
    rho0[grid.size // 2, grid.size // 2] = 1.0
    result = solver.propagate(np.linspace(0.0, 0.2, 5), rho0)
    assert result.success
    assert result.rho.shape == (5, grid.size, grid.size)
    assert np.all(np.isfinite(result.rho))
    assert result.trace_error < 2e-9


def test_drude_harmonic_kubo_pipeline_recovers_input_spectral_density():
    """P2 analytic harmonic fixture: recovered J is below one percent error."""
    beta, lam, gamma = 2.0, 0.4, 0.7
    time = np.linspace(0.0, 100.0, 10001)
    # This is the Kubo force TCF for the continuum Drude fixture.
    kubo = 2.0 * lam / beta * np.exp(-gamma * time)
    frequencies = np.linspace(0.02, 8.0, 1200)
    omega, recovered = spectral_density_from_kubo(
        time, kubo, beta, frequencies=frequencies, window=None
    )
    target = 2.0 * lam * gamma * omega / (omega**2 + gamma**2)
    relative = np.abs(recovered - target) / target
    assert float(np.max(relative)) < 1.0e-2


def test_harmonic_mode_kubo_fixture_matches_analytic_mode_sum():
    beta = 1.8
    time = np.linspace(0.0, 12.0, 101)
    modes = np.array([0.7, 1.9, 3.1])
    couplings = np.array([0.4, -0.2, 0.3])
    masses = np.array([1.0, 1.4, 0.8])
    expected = sum(
        c**2 / (beta * mass * omega**2) * np.cos(omega * time)
        for omega, c, mass in zip(modes, couplings, masses)
    )
    np.testing.assert_allclose(
        harmonic_bath_kubo_tcf(
            time, modes, couplings, beta=beta, masses=masses
        ),
        expected,
        rtol=0.0,
        atol=1e-14,
    )


def test_exponential_fit_reports_residual_for_fixed_poles():
    time = np.linspace(0.0, 20.0, 1001)
    poles = np.array([0.35 - 1.7j, 0.35 + 1.7j, 1.1 + 0.0j])
    amplitudes = np.array([0.4, 0.4, 0.2])
    correlation = np.exp(-np.outer(time, poles)) @ amplitudes
    fit = fit_exponential_correlation(
        time,
        correlation,
        n_terms=3,
        initial_exponents=poles,
        max_nfev=500,
    )
    assert fit.success
    assert fit.relative_l2_error < 1e-10
    assert fit.rms_error < 1e-10


def test_full_kubo_to_correlation_pipeline_records_fit_error():
    time = np.linspace(0.0, 20.0, 401)
    kubo = np.exp(-0.8 * time)
    result = analyze_kubo_tcf(
        time,
        kubo,
        beta=2.0,
        frequencies=np.linspace(0.02, 6.0, 400),
        correlation_time=np.linspace(0.0, 8.0, 161),
        n_terms=2,
        initial_exponents=np.array([0.8 + 0.0j, 2.0 + 0.0j]),
        window=None,
        max_nfev=300,
    )
    assert result.frequencies.shape == (400,)
    assert result.correlation.shape == (161,)
    assert np.isfinite(result.fit.relative_l2_error)
    assert np.isfinite(result.fit.absolute_l2_error)
