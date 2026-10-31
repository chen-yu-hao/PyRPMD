"""Reproducible analytic + sampled checks; this is a model, not a molecular PES.

Run from the checkout:
  python examples/quantum_statistics/harmonic_validation.py --output report.json
The optional --steps changes Monte Carlo effort, not the deterministic trace table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyrpmd.constants import KB_AU
from pyrpmd.path_integral import (
    OpenPathAction, PathIntegralAction, sample_open_path, sample_path_integral,
)
from pyrpmd.potentials.builtin import HarmonicPotential
from pyrpmd.quantum_statistics import (
    energy_heat_capacity, harmonic_log_partition, isotope_fractionation,
    isotope_substitution_free_energy, momentum_distribution_from_open_paths,
    quantum_kinetic_energy,
)


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--burn-in", type=int, default=3000)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    # One distinguishable nucleus in an isotropic confining harmonic potential.
    mass, omega, beta = 1837.0, 0.005, 1600.0
    temperature = 1 / (KB_AU * beta)
    stiffness = mass * omega**2
    exact = harmonic_log_partition(beta, [omega])
    table = []
    for p in (4, 8, 16, 32, 64, 128):
        table.append({"beads": p,
                      "primitive_absolute_logZ_error": abs(harmonic_log_partition(beta, [omega], beads=p) - exact),
                      "TI_absolute_logZ_error": abs(harmonic_log_partition(beta, [omega], beads=p, method="ti") - exact)})
    settings = dict(steps=args.steps, burn_in=args.burn_in, stride=args.stride,
                    centroid_step_bohr=0.1, internal_mix=0.5)
    phases, isotope_results = {}, {}
    for name, scale, seed in (("A", 1.0, 710), ("B", 0.25, 711)):
        action = PathIntegralAction(HarmonicPotential(stiffness * scale), [mass], temperature, method="ti")
        sample = sample_path_integral(action, np.zeros((16, 1, 3)), seed=seed, **settings)
        isotope = isotope_substitution_free_energy(sample, [2 * mass])
        exact_log_ratio = harmonic_log_partition(beta, [omega * np.sqrt(scale / 2)] * 3) - harmonic_log_partition(beta, [omega * np.sqrt(scale)] * 3)
        isotope_results[name] = isotope
        frequency = omega * np.sqrt(scale)
        x = beta * frequency
        exact_energy = 1.5 * frequency / np.tanh(x / 2)
        exact_heat = 3 * x**2 * np.exp(-x) / (-np.expm1(-x))**2
        dbeta = beta * 1e-4
        logs = [harmonic_log_partition(b, [frequency] * 3, beads=16, method="ti")
                for b in (beta - dbeta, beta, beta + dbeta)]
        phases[name] = {"heat_and_energy": energy_heat_capacity(sample),
                        "scaled_heat_and_energy": energy_heat_capacity(sample, estimator="scaled_coordinates"),
                        "kinetic_energy": quantum_kinetic_energy(sample),
                        "isotope": isotope,
                        "exact_quantum_log_isotope_partition_ratio": exact_log_ratio,
                        "exact_quantum_internal_energy_hartree": float(exact_energy),
                        "exact_quantum_heat_capacity_over_kB": float(exact_heat),
                        "finite_bead_TI_reference_internal_energy_hartree": -(logs[2] - logs[0]) / (2 * dbeta),
                        "finite_bead_TI_reference_heat_capacity_over_kB": beta**2 * (logs[2] - 2 * logs[1] + logs[0]) / dbeta**2,
                        "acceptance": sample.acceptance}
    # Preserve poor-overlap diagnostics in intentionally small smoke runs.
    fractionation = isotope_fractionation(isotope_results["A"], isotope_results["B"], allow_poor_overlap=True)
    fractionation["exact_quantum_log_alpha_A_over_B"] = phases["A"]["exact_quantum_log_isotope_partition_ratio"] - phases["B"]["exact_quantum_log_isotope_partition_ratio"]
    open_action = OpenPathAction(HarmonicPotential(stiffness), [mass], temperature, atom=0, axis=0)
    open_sample = sample_open_path(open_action, np.zeros((33, 1, 3)), seed=712, **settings)
    momentum = momentum_distribution_from_open_paths(open_sample, np.linspace(-12, 12, 241))
    momentum["exact_quantum_momentum_variance_au"] = mass * omega / (2 * np.tanh(beta * omega / 2))
    momentum["exact_quantum_end_to_end_variance_bohr2"] = 2 * np.tanh(beta * omega / 2) / (mass * omega)
    momentum["sample_end_to_end_second_moment_bohr2"] = float(np.mean(open_sample.displacements**2))
    momentum["acceptance"] = open_sample.acceptance
    result = {"model": "one_nucleus_isotropic_harmonic_confinement",
              "temperature_K": temperature, "beta_times_omega": beta * omega,
              "mass_au": mass, "omega_au": omega, "sampling": settings,
              "trace_convergence_per_Cartesian_mode": table,
              "phases": phases, "fractionation": fractionation, "momentum": momentum,
              "physical_converged": False,
              "limitations": ["harmonic analytic validation only",
                              "TI accelerates closed traces, not general density observables",
                              "momentum uses primitive open paths and has kernel and finite-grid bias",
                              "conditional errors do not establish mixing or bead convergence of molecular observables"]}
    text = json.dumps(result, default=json_value, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(args.output)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
