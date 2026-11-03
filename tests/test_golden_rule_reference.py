"""Independent spectral integrals and propagators check the GR reference helper."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "golden_rule_reference", _ROOT / "examples" / "instanton" / "golden_rule_reference.py",
)
reference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reference)


@pytest.mark.parametrize("time", [0.2, 1.0, 2.0])
def test_gamma_correlation_matches_original_spectral_integral(time):
    # Evaluate the published spectral-density expression directly, without
    # Gamma functions or a finite bath.  J=(pi/2)*alpha*w*exp(-w), hbar=1.
    alpha, beta, bias = 5.0, 2.0, 2.0
    real = quad(
        lambda omega: 4 * alpha * math.exp(-omega) * math.sin(omega * time / 2)**2
        / (omega * math.tanh(beta * omega / 2)),
        0, np.inf, epsabs=1e-11, epsrel=1e-12, limit=300,
    )[0]
    imaginary = quad(
        lambda omega: 2 * alpha * math.exp(-omega) * math.sin(omega * time) / omega,
        0, np.inf, epsabs=1e-11, epsrel=1e-12, limit=300,
    )[0] - bias * time
    analytic = reference.ohmic_log_correlation(time, beta, 1.0, 10.0, bias)
    assert analytic.real == pytest.approx(-real, abs=3e-11)
    assert analytic.imag == pytest.approx(-imaginary, abs=3e-11)


def test_gamma_correlation_has_exact_partition_endpoints_and_complex_phase():
    # Tr[e^-beta*H0]/Z0=1 and Tr[e^-beta*H1]/Z0=exp(beta*bias).
    assert reference.ohmic_log_correlation(0.0, 2.0, 1.0, 10.0, 2.0) == pytest.approx(0, abs=1e-13)
    assert reference.ohmic_log_correlation(-2j, 2.0, 1.0, 10.0, 2.0) == pytest.approx(4, abs=1e-13)
    value = reference.ohmic_log_correlation(0.5 - 0.7j, 2.0, 1.0, 10.0, 2.0)
    reverse = reference.ohmic_log_correlation(-0.5 - 0.7j, 2.0, 1.0, 10.0, 2.0)
    assert reverse == pytest.approx(value.conjugate(), abs=1e-13)
    assert abs(value.imag) > 0.05


def test_exact_continuum_rate_is_contour_independent_and_differs_from_sci():
    report = reference.ohmic_continuum_reference(2.0, 1.0, 10.0, 2.0, 1e-3)
    # An independent real-axis integration has neither the optimized contour
    # nor the action/curvature scaling used by the helper's quadrature.
    direct, error = quad(
        lambda time: np.exp(
            reference.ohmic_log_correlation(time, 2.0, 1.0, 10.0, 2.0)
        ).real,
        0, np.inf, epsabs=1e-12, epsrel=1e-11, limit=300,
    )
    assert error < 1e-11
    assert report["exact_rate"] == pytest.approx(2e-6 * direct, rel=1e-10)
    assert report["exact_rate"] == pytest.approx(5.54458032131393e-8, rel=1e-11)
    assert report["sci_rate"] == pytest.approx(5.46453487243166e-8, rel=1e-11)
    assert -0.01444 < report["sci_relative_error_vs_exact"] < -0.01443
    assert report["quadrature"]["relative_error_estimate"] < 1e-9
    assert report["action_gradient"] == pytest.approx(0, abs=1e-12)


def test_harmonic_action_matches_exact_mehler_propagator_gaussian_integral():
    # Two independently assembled 2x2 Mehler kernels, integrated over their
    # shared endpoints, provide a continuum oracle without the closed S(tau).
    mass, omega, displacement, beta, tau, bias = 1.7, 0.8, 1.2, 2.3, 0.7, 0.9
    energy = 2 * mass * omega**2 * displacement**2
    hessian = np.zeros((2, 2))
    linear = np.zeros(2)
    constant = -bias * tau
    log_amplitude = 0.0
    for duration, centre in ((beta - tau, -displacement), (tau, displacement)):
        matrix = mass * omega / math.sinh(omega * duration) * np.array([
            [math.cosh(omega * duration), -1.0],
            [-1.0, math.cosh(omega * duration)],
        ])
        centre_vector = np.full(2, centre)
        hessian += matrix
        linear += matrix @ centre_vector
        constant += 0.5 * centre_vector @ matrix @ centre_vector
        log_amplitude += 0.5 * math.log(
            mass * omega / (2 * math.pi * math.sinh(omega * duration))
        )
    gaussian_action = constant - 0.5 * linear @ np.linalg.solve(hessian, linear)
    gaussian_log_correlation = (
        log_amplitude + math.log(2 * math.pi) - 0.5 * np.linalg.slogdet(hessian)[1]
        - gaussian_action
    )
    log_partition = -math.log(2 * math.sinh(beta * omega / 2))
    action = reference.harmonic_action(tau, beta, [omega], [energy], bias)[0]
    assert action == pytest.approx(gaussian_action, abs=2e-14)
    assert gaussian_log_correlation - log_partition == pytest.approx(-action, abs=2e-14)


def test_harmonic_saddle_has_closed_single_mode_time_and_classical_marcus_limit():
    beta, omega, energy, bias, coupling = 4.0, 0.5, 4.5, 1.5, 1e-3
    expected_tau = beta / 2 - math.asinh(
        bias / energy * math.sinh(beta * omega / 2)
    ) / omega
    result = reference.harmonic_spin_boson_reference(
        beta, [omega], [energy], bias, coupling,
    )
    assert result["tau"] == pytest.approx(expected_tau, abs=1e-13)
    classical = reference.harmonic_spin_boson_reference(
        0.001, [0.6], [4.0], 1.0, 0.002,
    )
    assert classical["sci_rate"] == pytest.approx(classical["marcus_rate"], rel=2e-8)
    assert classical["exact_quantum_rate"] is None


@pytest.mark.parametrize("nmodes", [4, 8])
def test_ohmic_bath_has_correct_spectral_moments_and_independent_discretization(nmodes):
    bath = reference.discretize_ohmic_bath(nmodes, 1.0, 10.0, mass=1822.0)
    frequencies = np.array(bath["frequencies"])
    energies = np.array(bath["reorganization_energies"])
    for order in range(5):
        # Integral of exp(-x)*x**order is order!, independently of the
        # quadrature construction and of the harmonic SCI implementation.
        assert np.dot(energies, frequencies**order) == pytest.approx(
            10 * math.factorial(order), rel=2e-13,
        )
    np.testing.assert_allclose(
        2 * np.array(bath["masses"]) * frequencies**2 * np.array(bath["displacements"])**2,
        energies, rtol=3e-15,
    )
    result = reference.harmonic_spin_boson_reference(
        2.0, frequencies, energies, 2.0, 1e-3,
    )
    expected = {4: 5.479066142978905e-8, 8: 5.4643393621491504e-8}
    assert result["sci_rate"] == pytest.approx(expected[nmodes], rel=2e-12)


def test_reference_energy_scaling_coupling_scaling_and_json_report():
    unscaled = reference.ohmic_continuum_reference(2.0, 1.0, 10.0, 2.0, 1e-3)
    report = reference.reference_report()
    scaled = report["continuous_bath"]
    assert scaled["exact_rate"] == pytest.approx(unscaled["exact_rate"] * 1e-3, rel=1e-12)
    assert scaled["action_over_hbar"] == pytest.approx(unscaled["action_over_hbar"], abs=1e-12)
    larger_coupling = reference.ohmic_continuum_reference(2.0, 1.0, 10.0, 2.0, 2e-3)
    assert larger_coupling["exact_rate"] == pytest.approx(4 * unscaled["exact_rate"], rel=1e-13)
    assert report["finite_baths"][1]["bath_discretization_relative_error"] < 0
    json.dumps(report, allow_nan=False)


def test_reference_rejects_nonphysical_parameters_and_unsupported_time_saddles():
    with pytest.raises(ValueError, match="normal regime"):
        reference.harmonic_spin_boson_reference(2, [1], [10], 10, 1e-3)
    with pytest.raises(ValueError, match="normal regime"):
        reference.ohmic_continuum_reference(2, 1, 10, 11, 1e-3)
    with pytest.raises(ValueError, match="thermal strip"):
        reference.ohmic_log_correlation(1j, 2, 1, 10, 2)
    with pytest.raises(ValueError, match="positive integer"):
        reference.discretize_ohmic_bath(True, 1, 10)
    with pytest.raises(ValueError, match="matching reorganization"):
        reference.harmonic_spin_boson_reference(2, [1], [-10], 2, 1e-3)
