"""Reproducible instanton/quantum cross-check for a symmetric Eckart barrier.

Run: PYTHONPATH=src python examples/instanton/eckart.py
This is an analytic 1D validation potential, not a fitted chemical PES.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.integrate import quad

from pyrpmd.constants import ATOMIC_TIME_S, BOHR_M, KB_AU
from pyrpmd.instanton import crossover_temperature, instanton_rate_1d, instanton_seed
from pyrpmd.potentials.base import PESResult, Potential


class EckartPotential(Potential):
    def __init__(self, height=0.01, inverse_width=0.5):
        self.height = height
        self.inverse_width = inverse_width

    def evaluate(self, q):
        t = np.tanh(self.inverse_width * q[:, 0, 0])
        v = self.height * (1 - t**2)
        force = np.zeros_like(q)
        force[:, 0, 0] = 2 * self.inverse_width * v * t
        return PESResult(v, force)


def run_validation(beads=(32, 64, 128)):
    mass = np.array([1822.0])
    active = np.array([[True, False, False]])
    potential = EckartPotential()
    ts = np.zeros((1, 3))
    crossover = crossover_temperature(potential, ts, mass, active_mask=active)
    temperature = crossover / 2
    beta = 1 / (KB_AU * temperature)
    v0, a = potential.height, potential.inverse_width
    partition = np.sqrt(mass[0] / (2 * np.pi * beta))  # per bohr

    # Continuum abbreviated action W(E)=2*pi*sqrt(2m)/a*(sqrt(V0)-sqrt(E)).
    constant = 2 * np.pi**2 * mass[0] / a**2
    analytic_action = 2 * np.pi / a * np.sqrt(2 * mass[0] * v0) - constant / beta
    analytic_energy = constant / beta**2
    analytic_derivative = -2 * constant / beta**3
    prefactor = np.sqrt(-analytic_derivative / (2 * np.pi))
    analytic_rate = prefactor * np.exp(-analytic_action) / partition * BOHR_M / ATOMIC_TIME_S

    # Exact Eckart transmission, independently thermally integrated. The formula
    # is Eq. (70) in arXiv:2509.01454, with its length a replaced here by 1/a.
    gamma = np.pi * np.sqrt(2 * mass[0] * v0) / a
    z = np.sqrt(gamma**2 - np.pi**2 / 4)
    log_cosh = z + np.log1p(np.exp(-2 * z)) - np.log(2)

    def scaled_integrand(energy):
        u = gamma * np.sqrt(energy / v0)
        if u == 0:
            return 0.0
        log_sinh = u + np.log(-np.expm1(-2 * u)) - np.log(2)
        log_transmission = 2 * log_sinh - np.logaddexp(2 * log_sinh, 2 * log_cosh)
        return float(np.exp(log_transmission - beta * energy + analytic_action))

    integral, integral_error = quad(scaled_integrand, 0, max(10 * v0, 80 / beta),
                                    epsabs=1e-12, epsrel=1e-10,
                                    points=[analytic_energy, v0])
    exact_rate = integral / (2 * np.pi) * np.exp(-analytic_action) / partition * BOHR_M / ATOMIC_TIME_S
    cases = []
    for nbeads in beads:
        seed = instanton_seed(ts, [[1.0, 0.0, 0.0]], nbeads, 2.5)
        result = instanton_rate_1d(potential, mass, temperature, seed,
                                  active_mask=active, reactant_partition=partition,
                                  normalization="per_bohr", transition_state=ts)
        orbit = result["instanton"]
        cases.append({
            "beads": nbeads,
            "action_hbar": orbit["action_hbar"],
            "absolute_action_discretization_error": abs(orbit["action_hbar"] - analytic_action),
            "energy_hartree": orbit["energy_hartree"],
            "rate_m_per_s": result["rate"],
            "relative_rate_discretization_error": result["rate"] / analytic_rate - 1,
            "relative_difference_from_exact_quantum": result["rate"] / exact_rate - 1,
            "gradient_max": orbit["gradient_max"],
            "negative_modes": orbit["negative_modes"],
            "phase_mode_relative_residual": orbit["phase_mode_relative_residual"],
            "beta_derivative_relative_difference": result["beta_derivative_relative_difference"],
            "numerical_converged": result["numerical_converged"],
        })
    return {
        "model": "V(x)=V0*sech(a*x)^2; one active Cartesian coordinate",
        "mass_electron_masses": float(mass[0]),
        "barrier_hartree": v0,
        "inverse_width_bohr_inverse": a,
        "crossover_temperature_kelvin": crossover,
        "temperature_kelvin": temperature,
        "rate_normalization": "one-dimensional flux velocity in m/s, not cm^3 molecule^-1 s^-1",
        "analytic_continuum_instanton_action_hbar": analytic_action,
        "analytic_continuum_instanton_energy_hartree": analytic_energy,
        "analytic_continuum_instanton_rate_m_per_s": analytic_rate,
        "exact_quantum_rate_m_per_s": exact_rate,
        "quantum_quadrature_relative_error_estimate": integral_error / abs(integral),
        "continuum_instanton_relative_semiclassical_error": analytic_rate / exact_rate - 1,
        "cases": cases,
        "physical_converged": False,
        "scope": "Analytic numerical validation; not a real-reaction rate or exact-dynamics replacement",
        "references": ["https://doi.org/10.1063/1.4943866",
                       "https://arxiv.org/html/2509.01454v1#A3"],
    }


if __name__ == "__main__":
    print(json.dumps(run_validation(), indent=2, allow_nan=False))
