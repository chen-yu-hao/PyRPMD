#!/usr/bin/env python3
"""Reproduce a finite-P harmonic PIMD equilibrium spot check.

Run from the repository root with its installed Python environment::

    .venv/bin/python benchmarks/materials/harmonic_validation.py \
        --output benchmarks/materials/harmonic_validation.json

This is an independently known statistical reference, not a gating stochastic
unit test or a validation of a material-specific force field or transport rate.
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path

import numpy as np

import pyrpmd
from pyrpmd.system import System
from pyrpmd.workflows.materials import run_materials


COMMAND = (
    ".venv/bin/python benchmarks/materials/harmonic_validation.py "
    "--output benchmarks/materials/harmonic_validation.json"
)


def experiment_config():
    return {
        "task": "pimd", "beta": 4.0, "beads": 16, "dt_au": 0.04,
        "seed": 6703, "native": False,
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 1.0}},
        "sampling": {
            "equilibration_steps": 4000, "production_steps": 24000,
            "sample_interval": 10, "replicas": 4,
            "thermostat": "pile-l", "gamma_centroid": 1.0,
        },
        "dynamics": {"loop_backend": "python", "integrator": "cayley-bcb"},
        "output": {"checkpoint_steps": 100000},
    }


def build_report(statistics, elapsed_seconds):
    config = experiment_config()
    beta, beads, mass, spring = config["beta"], config["beads"], 1.0, 1.0
    # Each cyclic normal mode is an independent Gaussian with stiffness
    # k + 4*m*(P/beta)^2*sin(pi*j/P)^2 under exp[-beta*H_P/P].
    # The harmonic virial theorem gives <K_cv> = <V> also at finite P.
    mode_springs = spring + 4 * mass * (beads / beta) ** 2 * np.sin(np.pi * np.arange(beads) / beads) ** 2
    exact_potential = float(3 * spring / (2 * beta) * np.sum(1 / mode_springs))
    frequency = np.sqrt(spring / mass)
    continuum_energy = float(1.5 * frequency / np.tanh(beta * frequency / 2))
    comparisons = {}
    for name, exact in (
        ("potential_energy", exact_potential),
        ("kinetic_centroid_virial", exact_potential),
        ("kinetic_primitive", exact_potential),
        ("internal_energy_centroid_virial", 2 * exact_potential),
    ):
        estimate = statistics[name]
        comparisons[name] = {
            **estimate,
            "finite_P_exact": exact,
            "difference_in_between_replica_standard_errors": (
                (estimate["mean"] - exact) / estimate["standard_error_between_replicas"]
            ),
        }
    return {
        "schema": "pyrpmd.materials-harmonic-validation",
        "schema_version": 1,
        "command": COMMAND,
        "description": "Four independent PIMD streams for a synthetic three-dimensional harmonic oscillator.",
        "physical_converged": False,
        "environment": {
            "pyrpmd": pyrpmd.__version__, "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "units": {
            "energy": "hartree", "mass": "electron_mass", "position": "bohr",
            "time": "atomic_time", "beta": "hartree^-1", "force_constant": "hartree/bohr^2",
        },
        "system": {"atoms": 1, "dimensions": 3, "mass": mass, "force_constant": spring, "periodic": False},
        "configuration": config,
        "hamiltonian_convention": "sum_beads_beta_over_P",
        "reference": {
            "finite_P_potential_energy": exact_potential,
            "finite_P_kinetic_energy": exact_potential,
            "finite_P_internal_energy": 2 * exact_potential,
            "continuum_quantum_internal_energy": continuum_energy,
            "classical_internal_energy": 3 / beta,
            "finite_P_formula": "<V> = 3*k/(2*beta) * sum_j [k + 4*m*(P/beta)^2*sin(pi*j/P)^2]^-1; <K> = <V>",
            "continuum_formula": "<E> = (3*omega/2) coth(beta*omega/2), omega=sqrt(k/m), hbar=1",
        },
        "results": comparisons,
        "wall_seconds": float(elapsed_seconds),
        "limitations": [
            "A synthetic equilibrium check does not validate hydrogen migration or a material-specific potential.",
            "Standard errors use four independent replica means; they are noisy statistical estimates, not rigorous error bounds.",
            "This single timestep and bead count does not certify discretization or equilibration convergence.",
            "The finite-P reference is distinct from the continuum quantum value; their difference is not Monte Carlo error.",
            "No diffusive rate, real-time correlation, exchange statistics, or periodic winding sector is tested here.",
            "The reported deviations are diagnostics without a pass/fail significance threshold.",
        ],
    }


def run_validation():
    system = System(("H",), np.array([1.0]), np.zeros((1, 3)))
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="pyrpmd-harmonic-validation-") as output:
        result = run_materials(experiment_config(), system, output)
    return build_report(result["estimators"], time.monotonic() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report destination.")
    args = parser.parse_args()
    report = run_validation()
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
