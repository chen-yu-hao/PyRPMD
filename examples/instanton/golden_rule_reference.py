"""Independent analytic oracles for golden-rule instanton validation.

No production instanton, potential, or partition-function implementation is
imported here.  The harmonic coordinate integrals are analytic; a remaining
time saddle is still an approximation to the quantum golden-rule rate.

Primary sources:

* Richardson, Bauer and Thoss, J. Chem. Phys. 143, 134115 (2015),
  doi:10.1063/1.4932361, Sec. V B, Eqs. (60)--(69).
  https://arxiv.org/pdf/1508.04919
* Heller and Richardson, J. Chem. Phys. 152, 034106 (2020),
  doi:10.1063/1.5137823, Eqs. (28)--(35).
  https://arxiv.org/pdf/1911.06730

The Ohmic Gamma-function expression below is obtained independently by
integrating the spectral-density formula in those papers for
J(w) = (pi/2) * alpha * hbar * w * exp(-w/wc).
It is exact for this continuous bath, including its finite exponential cutoff.
It is not an exact smooth rate for a finite set of isolated oscillators, whose
unbroadened golden-rule spectrum consists of delta functions.

Conventions: V0 = sum_j m_j*w_j**2*(x_j+zeta_j)**2/2 and
V1 = sum_j m_j*w_j**2*(x_j-zeta_j)**2/2 - bias.  Positive bias favors
products.  Coupling is the off-diagonal Hamiltonian matrix element, not twice
that element.  Frequencies are angular frequencies.  Atomic units are obtained
by setting hbar=1 and supplying all quantities in atomic units.

Run this file directly to print a reference report, including four- and
eight-mode discretizations, without invoking the production optimizer.
"""
from __future__ import annotations

import json
import math
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq
from scipy.special import digamma, gammaln, loggamma, polygamma, roots_laguerre


SOURCES = {
    "harmonic_action_and_time_saddle": {
        "doi": "10.1063/1.4932361",
        "url": "https://arxiv.org/pdf/1508.04919",
        "location": "Sec. V B; Eqs. (60)-(69); PDF pages 12-13",
    },
    "spectral_density_exact_rate_and_marcus": {
        "doi": "10.1063/1.5137823",
        "url": "https://arxiv.org/pdf/1911.06730",
        "location": "Sec. V A; Eqs. (28)-(35); PDF pages 11-12",
    },
    "gamma_correlation": (
        "Independent integration of the cited exact spectral-density formula "
        "for an Ohmic bath with an exponential cutoff; no scaling-limit or "
        "high-temperature approximation."
    ),
}


def _positive(name, value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _finite(name, value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normal_regime(bias, reorganization_energy):
    if abs(bias) >= reorganization_energy:
        raise ValueError(
            "this reference saddle requires the strict normal regime "
            "abs(bias) < reorganization_energy"
        )


def _log_sinh_positive(value):
    """Stable log(sinh(x)) for nonnegative real arrays, including x=0."""
    value = np.asarray(value, dtype=float)
    with np.errstate(divide="ignore"):
        return value + np.log(-np.expm1(-2 * value)) - math.log(2)


def _log_cosh(value):
    magnitude = np.abs(value)
    return magnitude + np.log1p(np.exp(-2 * magnitude)) - math.log(2)


def marcus_rate(beta, reorganization_energy, bias, coupling, *, hbar=1.0):
    """Classical golden-rule rate, valid in normal and inverted regimes.

    The reactant minimum has zero energy and the product minimum has energy
    -bias.  A positive, finite coupling is required by this validation helper.
    """
    beta = _positive("beta", beta)
    reorganization_energy = _positive("reorganization_energy", reorganization_energy)
    bias = _finite("bias", bias)
    coupling = _positive("coupling", coupling)
    hbar = _positive("hbar", hbar)
    log_rate = (
        2 * math.log(coupling) - math.log(hbar)
        + 0.5 * math.log(math.pi * beta / reorganization_energy)
        - beta * (reorganization_energy - bias)**2 / (4 * reorganization_energy)
    )
    return math.exp(log_rate)


def _harmonic_parameters(frequencies, reorganization_energies):
    frequencies = np.asarray(frequencies, dtype=float)
    energies = np.asarray(reorganization_energies, dtype=float)
    if (
        frequencies.ndim != 1 or frequencies.size == 0
        or energies.shape != frequencies.shape
        or not np.all(np.isfinite(frequencies))
        or not np.all(np.isfinite(energies))
        or np.any(frequencies <= 0) or np.any(energies < 0)
        or not 0 < float(energies.sum()) < math.inf
    ):
        raise ValueError(
            "frequencies must be positive and finite; matching reorganization "
            "energies must be nonnegative, finite, and have a positive sum"
        )
    return frequencies, energies


def harmonic_action(tau, beta, frequencies, reorganization_energies, bias, *, hbar=1.0):
    """Return the continuum harmonic action and its first two tau derivatives.

    C(-i*tau)/Z0 = exp(-S(tau)/hbar) is an exact harmonic correlation
    function.  The time saddle applied to it is only a semiclassical rate.
    The hyperbolic expressions are evaluated without large-frequency overflow.
    """
    beta = _positive("beta", beta)
    hbar = _positive("hbar", hbar)
    bias = _finite("bias", bias)
    frequencies, energies = _harmonic_parameters(frequencies, reorganization_energies)
    tau = _finite("tau", tau)
    duration = beta * hbar
    if not 0 <= tau <= duration:
        raise ValueError("tau must lie in [0, beta*hbar]")
    half = frequencies * duration / 2
    difference = frequencies * (duration / 2 - tau)
    denominator = _log_sinh_positive(half)
    # cosh(B)-cosh(B-w*tau) = 2*sinh(w*tau/2)*sinh(w*(b-tau)/2).
    action_terms = np.exp(
        math.log(2) + _log_sinh_positive(frequencies * tau / 2)
        + _log_sinh_positive(frequencies * (duration - tau) / 2)
        - denominator
    )
    derivative_terms = np.sign(difference) * np.exp(
        _log_sinh_positive(np.abs(difference)) - denominator
    )
    curvature_terms = np.exp(_log_cosh(difference) - denominator)
    return (
        float(-bias * tau + np.sum(energies / frequencies * action_terms)),
        float(-bias + np.sum(energies * derivative_terms)),
        float(-np.sum(energies * frequencies * curvature_terms)),
    )


def harmonic_spin_boson_reference(
    beta, frequencies, reorganization_energies, bias, coupling, *, hbar=1.0,
):
    """Analytic continuum SCI result for a finite displaced harmonic bath.

    This function deliberately does not label its result an exact quantum rate.
    Nuclear Gaussian fluctuations and the reactant partition function are exact;
    only the remaining real-time integral is approximated by its saddle.
    """
    beta = _positive("beta", beta)
    hbar = _positive("hbar", hbar)
    coupling = _positive("coupling", coupling)
    bias = _finite("bias", bias)
    frequencies, energies = _harmonic_parameters(frequencies, reorganization_energies)
    reorganization_energy = float(energies.sum())
    _normal_regime(bias, reorganization_energy)
    duration = beta * hbar
    fraction = brentq(
        lambda value: harmonic_action(
            value * duration, beta, frequencies, energies, bias, hbar=hbar,
        )[1],
        0.0, 1.0, xtol=1e-14,
    )
    tau = fraction * duration
    action, gradient, curvature = harmonic_action(
        tau, beta, frequencies, energies, bias, hbar=hbar,
    )
    if not math.isfinite(curvature) or curvature >= 0:
        raise ValueError("the harmonic time saddle has no resolved negative curvature")
    log_rate = (
        2 * math.log(coupling / hbar)
        + 0.5 * math.log(2 * math.pi * hbar / -curvature) - action / hbar
    )
    log_partition = -float(np.sum(
        _log_sinh_positive(frequencies * duration / 2) + math.log(2)
    ))
    return {
        "model": "finite_displaced_harmonic_bath",
        "approximation": "continuum_semiclassical_time_saddle",
        "nmodes": int(frequencies.size),
        "tau": tau,
        "tau_fraction": fraction,
        "action": action,
        "action_over_hbar": action / hbar,
        "action_gradient": gradient,
        "action_curvature": curvature,
        "log_reactant_partition": log_partition,
        "log_sci_rate": log_rate,
        "sci_rate": math.exp(log_rate),
        "marcus_rate": marcus_rate(
            beta, reorganization_energy, bias, coupling, hbar=hbar,
        ),
        "exact_quantum_rate": None,
        "exact_rate_limitation": (
            "An isolated finite harmonic bath has a delta-function spectrum. "
            "No smooth exact rate or artificial broadening is assigned."
        ),
    }


def _ohmic_parameters(beta, cutoff_frequency, reorganization_energy, bias, hbar):
    beta = _positive("beta", beta)
    cutoff = _positive("cutoff_frequency", cutoff_frequency)
    energy = _positive("reorganization_energy", reorganization_energy)
    bias = _finite("bias", bias)
    hbar = _positive("hbar", hbar)
    duration = beta * hbar
    alpha = energy / (2 * hbar * cutoff)
    return duration, cutoff, alpha, 1 / (duration * cutoff), bias, hbar


def ohmic_log_correlation(
    time, beta, cutoff_frequency, reorganization_energy, bias, *, hbar=1.0,
):
    """Exact log(C(z)/Z0) for the continuous exponential-cutoff Ohmic bath.

    z is complex real time.  The original real-time contour has Im(z)=0;
    the saddle contour has z=t-i*tau.  The closed thermal strip
    -beta*hbar <= Im(z) <= 0 is supported.  Complex logarithms retain the
    analytic phase: replacing them with log(abs(...)) would change the rate.
    """
    duration, cutoff, alpha, a, bias, hbar = _ohmic_parameters(
        beta, cutoff_frequency, reorganization_energy, bias, hbar,
    )
    time = np.asarray(time, dtype=complex)
    if (
        not np.all(np.isfinite(time))
        or np.any(time.imag > 0) or np.any(time.imag < -duration)
    ):
        raise ValueError("complex time must be finite and inside the thermal strip")
    value = (
        1j * bias * time / hbar - 2 * alpha * np.log1p(1j * cutoff * time)
        + 2 * alpha * (
            loggamma(1 + a + 1j * time / duration)
            + loggamma(1 + a - 1j * time / duration) - 2 * gammaln(1 + a)
        )
    )
    return complex(value) if value.ndim == 0 else value


def ohmic_action(
    tau, beta, cutoff_frequency, reorganization_energy, bias, *, hbar=1.0,
):
    """Exact imaginary-time Ohmic action and its first two derivatives."""
    duration, cutoff, alpha, a, bias, hbar = _ohmic_parameters(
        beta, cutoff_frequency, reorganization_energy, bias, hbar,
    )
    tau = _finite("tau", tau)
    if not 0 <= tau <= duration:
        raise ValueError("tau must lie in [0, beta*hbar]")
    plus, minus = 1 + a + tau / duration, 1 + a - tau / duration
    action = -bias * tau + 2 * alpha * hbar * (
        math.log1p(cutoff * tau)
        - gammaln(plus) - gammaln(minus) + 2 * gammaln(1 + a)
    )
    gradient = -bias + 2 * alpha * hbar * (
        cutoff / (1 + cutoff * tau)
        - (digamma(plus) - digamma(minus)) / duration
    )
    curvature = -2 * alpha * hbar * (
        cutoff**2 / (1 + cutoff * tau)**2
        + (polygamma(1, plus) + polygamma(1, minus)) / duration**2
    )
    return float(action), float(gradient), float(curvature)


def ohmic_continuum_reference(
    beta, cutoff_frequency, reorganization_energy, bias, coupling, *,
    hbar=1.0, quadrature_rtol=1e-10,
):
    """Compute exact-continuum GR, continuum SCI, and classical Marcus rates.

    The exact rate is quadrature of the full correlation function, not its
    Gaussian expansion.  The time contour is shifted to the normal-regime
    saddle and scaled by its Gaussian width to resolve very small rates.
    The returned quadrature uncertainty is an estimate, not a rigorous bound.
    """
    beta = _positive("beta", beta)
    coupling = _positive("coupling", coupling)
    hbar = _positive("hbar", hbar)
    reorganization_energy = _positive("reorganization_energy", reorganization_energy)
    cutoff_frequency = _positive("cutoff_frequency", cutoff_frequency)
    bias = _finite("bias", bias)
    quadrature_rtol = _positive("quadrature_rtol", quadrature_rtol)
    if quadrature_rtol < 1e-13 or quadrature_rtol >= 0.1:
        raise ValueError("quadrature_rtol must lie in [1e-13, 0.1)")
    _normal_regime(bias, reorganization_energy)
    duration = beta * hbar
    fraction = brentq(
        lambda value: ohmic_action(
            value * duration, beta, cutoff_frequency, reorganization_energy,
            bias, hbar=hbar,
        )[1],
        0.0, 1.0, xtol=1e-14,
    )
    tau = fraction * duration
    action, gradient, curvature = ohmic_action(
        tau, beta, cutoff_frequency, reorganization_energy, bias, hbar=hbar,
    )
    if not math.isfinite(curvature) or curvature >= 0:
        raise ValueError("the Ohmic time saddle has no resolved negative curvature")
    width = math.sqrt(hbar / -curvature)

    def scaled_integrand(value):
        log_value = ohmic_log_correlation(
            width * value - 1j * tau, beta, cutoff_frequency,
            reorganization_energy, bias, hbar=hbar,
        )
        return float(np.exp(log_value + action / hbar).real)

    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        integral, error = quad(
            scaled_integrand, 0, np.inf, epsabs=quadrature_rtol / 10,
            epsrel=quadrature_rtol, limit=300,
        )
    if not math.isfinite(integral) or integral <= 0:
        raise ValueError("exact correlation quadrature did not yield a positive finite rate")
    log_exact = (
        2 * math.log(coupling / hbar) + math.log(2 * width * integral)
        - action / hbar
    )
    log_sci = (
        2 * math.log(coupling / hbar)
        + 0.5 * math.log(2 * math.pi * hbar / -curvature) - action / hbar
    )
    return {
        "model": "continuous_ohmic_bath_with_finite_exponential_cutoff",
        "tau": tau,
        "tau_fraction": fraction,
        "action": action,
        "action_over_hbar": action / hbar,
        "action_gradient": gradient,
        "action_curvature": curvature,
        "log_exact_rate": log_exact,
        "exact_rate": math.exp(log_exact),
        "log_sci_rate": log_sci,
        "sci_rate": math.exp(log_sci),
        "sci_relative_error_vs_exact": math.expm1(log_sci - log_exact),
        "marcus_rate": marcus_rate(
            beta, reorganization_energy, bias, coupling, hbar=hbar,
        ),
        "quadrature": {
            "contour_tau": tau,
            "time_scale": width,
            "scaled_half_integral": integral,
            "scaled_absolute_error_estimate": error,
            "relative_error_estimate": error / integral,
            "error_estimate_is_rigorous_bound": False,
            "requested_relative_tolerance": quadrature_rtol,
        },
    }


def discretize_ohmic_bath(nmodes, cutoff_frequency, reorganization_energy, *, mass=1.0):
    """Gauss-Laguerre discretization of the continuous Ohmic spectral measure.

    Since dLambda = Lambda*exp(-x) dx for x=omega/omega_c, quadrature gives
    omega_j=omega_c*x_j and Lambda_j=Lambda*w_j.  Each coordinate's displacement
    is sqrt(Lambda_j/(2*m*omega_j**2)).  Increasing nmodes converges the bath
    independently of the ring-polymer bead discretization.
    """
    if isinstance(nmodes, (bool, np.bool_)) or not isinstance(nmodes, (int, np.integer)) or nmodes < 1:
        raise ValueError("nmodes must be a positive integer")
    cutoff = _positive("cutoff_frequency", cutoff_frequency)
    energy = _positive("reorganization_energy", reorganization_energy)
    mass = _positive("mass", mass)
    nodes, weights = roots_laguerre(int(nmodes))
    frequencies = cutoff * nodes
    energies = energy * weights
    displacements = np.sqrt(energies / (2 * mass * frequencies**2))
    return {
        "nmodes": int(nmodes),
        "quadrature": "Gauss-Laguerre",
        "frequencies": frequencies.tolist(),
        "reorganization_energies": energies.tolist(),
        "displacements": displacements.tolist(),
        "masses": [mass] * int(nmodes),
        "nodes": nodes.tolist(),
        "weights": weights.tolist(),
    }


def benchmark_parameters(nmodes=4, *, energy_scale=1e-3, mass=1.0):
    """A reproducible normal-regime benchmark; beta*omega_c=2, beta*Lambda=20.

    For energy_scale=1e-3 in atomic units, beta=2000, omega_c=0.001,
    Lambda=0.01, bias=0.002 and coupling=1e-6.  The atomic mass can be chosen
    independently: the displacements change so that the spectral density does
    not.  Four and eight modes are inexpensive documented discretizations.
    """
    scale = _positive("energy_scale", energy_scale)
    return {
        "beta": 2 / scale,
        "cutoff_frequency": scale,
        "reorganization_energy": 10 * scale,
        "bias": 2 * scale,
        "coupling": 1e-3 * scale,
        "hbar": 1.0,
        "bath": discretize_ohmic_bath(nmodes, scale, 10 * scale, mass=mass),
    }


def reference_report():
    """JSON-ready reference results; neither a bead study nor a production run."""
    parameters = benchmark_parameters()
    arguments = {key: value for key, value in parameters.items() if key != "bath"}
    continuum = ohmic_continuum_reference(**arguments)
    baths = []
    for nmodes in (4, 8):
        bath = benchmark_parameters(nmodes)["bath"]
        result = harmonic_spin_boson_reference(
            parameters["beta"], bath["frequencies"], bath["reorganization_energies"],
            parameters["bias"], parameters["coupling"],
        )
        baths.append({
            "parameters": bath,
            "reference": result,
            "bath_discretization_relative_error": math.expm1(
                result["log_sci_rate"] - continuum["log_sci_rate"]
            ),
        })
    return {
        "sources": SOURCES,
        "parameters": arguments,
        "continuous_bath": continuum,
        "finite_baths": baths,
        "scope": (
            "Independent analytic references only. Bath convergence, bead "
            "convergence and the semiclassical approximation error are separate."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(reference_report(), indent=2, allow_nan=False))
