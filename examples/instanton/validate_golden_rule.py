"""Reproduce golden-rule bead, bath, and semiclassical error comparisons.

This executes the production workflow on genuine two-state matrix providers.
The companion golden_rule_reference.py supplies independently derived analytic
references and a full continuous-bath quantum golden-rule time integral.
No finite undamped bath is assigned an artificial smooth exact quantum rate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import runpy
import subprocess
import time

import numpy as np
import scipy

from pyrpmd.constants import AMU_TO_AU, ATOMIC_TIME_S, KB_AU
from pyrpmd.golden_rule_adapters import diabatic_surfaces
from pyrpmd.io import write_json
from pyrpmd.low_temperature import HarmonicState
from pyrpmd.potentials.nonadiabatic_models import DiscreteSpinBoson
from pyrpmd.workflows.instanton import run_instanton


_REFERENCE_PATH = Path(__file__).with_name("golden_rule_reference.py")
_REFERENCE = runpy.run_path(str(_REFERENCE_PATH))
_COUNTERS = ("force_calls", "force_configurations", "physical_hessian_evaluations",
             "finite_difference_hessian_evaluations", "hessian_provider_calls",
             "hessian_provider_configurations")


def build_case(nmodes, intervals=(8, 16, 32, 64), *, energy_scale=1e-3,
               mass=AMU_TO_AU, beta=None, relative_tolerance=0.01):
    """Construct the published-convention Ohmic discretization without a fit."""
    parameters = _REFERENCE["benchmark_parameters"](nmodes, energy_scale=energy_scale, mass=mass)
    if beta is not None:
        parameters["beta"] = float(beta)
    bath = parameters["bath"]
    omega = np.asarray(bath["frequencies"])
    lambdas = np.asarray(bath["reorganization_energies"])
    mode_mass = np.asarray(bath["masses"])
    # The matrix model uses U +/- model.bias +/- c.Q; here model.bias=epsilon/2.
    # Its reorganization energy is 2*c_j^2/(m_j*omega_j^2).
    linear = np.sqrt(lambdas * mode_mass * omega**2 / 2)
    matrix = DiscreteSpinBoson(omega, linear, masses=mode_mass,
                               bias=parameters["bias"] / 2,
                               tunnelling=parameters["coupling"])
    surface0, surface1 = diabatic_surfaces(matrix, coupling_hartree=parameters["coupling"])
    mask = np.zeros((nmodes, 3), dtype=bool)
    mask[:, 0] = True
    minimum = np.zeros((nmodes, 3))
    minimum[:, 0] = -np.asarray(bath["displacements"])
    reactant = HarmonicState.from_potential(
        surface0, minimum, matrix.nuclear_masses,
        hessian=np.diag(mode_mass * omega**2), active_mask=mask)
    levels = [(int(p), int(p)) for p in intervals]
    initial = np.zeros((levels[0][0] + 1, nmodes, 3))
    model_identity = matrix.metadata["model_identity"]
    arguments = {
        "potential": (surface0, surface1), "masses": matrix.nuclear_masses,
        "initial_path": (initial, initial.copy()), "mode": "golden_rule",
        "temperature_kelvin": 1 / (KB_AU * parameters["beta"]),
        "active_mask": mask, "reactants": [reactant],
        "energy_reference_hartree": reactant.energy_hartree,
        "bead_counts": levels,
        "potential_identity": {
            "surface0": {"matrix_model": model_identity, "state": 0},
            "surface1": {"matrix_model": model_identity, "state": 1},
        },
        "golden_rule_options": {"coupling_hartree": parameters["coupling"],
                                "partition_discretization": "matched",
                                "gradient_tolerance": 1e-8,
                                "energy_tolerance_hartree": 1e-10},
        "convergence_relative_tolerance": relative_tolerance,
    }
    return arguments, parameters


def _validation_view(report):
    """Keep all grid results and identities, omitting large trajectory arrays."""
    result = deepcopy(report)
    for row in result["rows"]:
        orbit = row["orbit"]
        orbit.pop("path0_bohr")
        orbit.pop("path1_bohr")
        spectrum = orbit.pop("coordinate_hessian_eigenvalues")
        orbit["coordinate_spectrum_summary"] = {
            "count": len(spectrum), "minimum": min(spectrum), "maximum": max(spectrum)}
        history = orbit["optimizer_diagnostics"].pop("history")
        orbit["optimizer_diagnostics"]["maximum_fixed_time_gradient"] = max(
            entry["gradient_max"] for entry in history)
    result["validation_record_scope"] = (
        "All scan points, prefactors, partitions, residuals, costs and input identities. "
        "Paths, full spectra and per-iteration histories are reproducible with run_instanton "
        "and are omitted from this compact validation record."
    )
    return result


def run_case(nmodes, intervals=(8, 16, 32, 64), *, energy_scale=1e-3,
             mass=AMU_TO_AU, beta=None, relative_tolerance=0.01):
    arguments, parameters = build_case(nmodes, intervals, energy_scale=energy_scale,
                                        mass=mass, beta=beta, relative_tolerance=relative_tolerance)
    bath = parameters["bath"]
    analytic = _REFERENCE["harmonic_spin_boson_reference"](
        parameters["beta"], bath["frequencies"], bath["reorganization_energies"],
        parameters["bias"], parameters["coupling"])
    continuum = _REFERENCE["ohmic_continuum_reference"](
        **{key: value for key, value in parameters.items() if key != "bath"})
    started = time.perf_counter()
    study = run_instanton(**arguments)
    elapsed = time.perf_counter() - started
    log_rate = study["final_rate"]["log_rate_au"]
    costs = {key: sum(row["orbit"]["optimizer_diagnostics"][key] for row in study["rows"])
             for key in _COUNTERS}
    return {
        "parameters": parameters, "temperature_kelvin": arguments["temperature_kelvin"],
        "finite_bath_analytic_sci": analytic, "continuous_bath_reference": continuum,
        "production_rate_au": math.exp(log_rate),
        "production_rate_si": study["final_rate"]["rate"],
        "production_rate_unit": "s^-1",
        "bead_error_vs_finite_bath_sci": math.expm1(log_rate - analytic["log_sci_rate"]),
        "bath_error_vs_continuous_sci": math.expm1(analytic["log_sci_rate"] - continuum["log_sci_rate"]),
        "continuous_sci_error_vs_exact_gr": continuum["sci_relative_error_vs_exact"],
        "combined_error_vs_continuous_exact_gr": math.expm1(log_rate - continuum["log_exact_rate"]),
        "production_error_vs_marcus": math.expm1(log_rate - math.log(analytic["marcus_rate"])),
        "total_workflow_cost": costs, "elapsed_seconds": elapsed,
        "study": _validation_view(study),
    }


def _provenance():
    root = Path(__file__).resolve().parents[2]
    relative_sources = (
        "src/pyrpmd/golden_rule.py", "src/pyrpmd/golden_rule_adapters.py",
        "src/pyrpmd/golden_rule_partition.py", "src/pyrpmd/workflows/golden_rule.py",
        "src/pyrpmd/workflows/instanton.py", "examples/instanton/validate_golden_rule.py",
        "examples/instanton/golden_rule_reference.py",
    )
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        head, dirty = None, None
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    except (OSError, StopIteration):
        cpu = platform.processor() or None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": head, "git_worktree_dirty": dirty,
        "source_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                          for name in relative_sources},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "platform": platform.platform(),
                    "cpu_model": cpu, "logical_cpus": os.cpu_count(),
                    "threads": {name: os.environ.get(name) for name in
                                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
    }


def validate(*, modes=(4, 8), intervals=(8, 16, 32, 64), relative_tolerance=0.01,
             include_classical_limit=True):
    cases = [run_case(n, intervals, relative_tolerance=relative_tolerance) for n in modes]
    classical = (run_case(1, (4, 8, 16), beta=40, relative_tolerance=relative_tolerance)
                 if include_classical_limit else None)
    return {
        "schema": "PyRPMD.golden-rule-validation.v1", "provenance": _provenance(),
        "sources": _REFERENCE["SOURCES"], "cases": cases,
        "classical_limit_case": classical,
        "rate_unit_au": "inverse atomic time", "atomic_time_seconds": ATOMIC_TIME_S,
        
        "scope": (
            "Normal-regime, constant weak real electronic coupling and constrained bath coordinates. "
            "Bead, bath and semiclassical errors are reported separately. The exact GR reference "
            "belongs to the continuous Ohmic bath, never to an isolated finite oscillator set."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--intervals", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument("--skip-classical-limit", action="store_true")
    parser.add_argument("--output", default="golden-rule-validation.json")
    args = parser.parse_args(argv)
    result = validate(modes=args.modes, intervals=args.intervals,
                      relative_tolerance=args.relative_tolerance,
                      include_classical_limit=not args.skip_classical_limit)
    write_json(args.output, result, callables=False)
    print(json.dumps({"output": str(args.output), "cases": [
        {"modes": case["parameters"]["bath"]["nmodes"],
         "final_intervals": case["study"]["final_intervals"],
         "discretization_converged": case["study"]["discretization_converged"],
         "rate_s^-1": case["production_rate_si"],
         "bead_error": case["bead_error_vs_finite_bath_sci"],
         "bath_error": case["bath_error_vs_continuous_sci"],
         "semiclassical_error": case["continuous_sci_error_vs_exact_gr"]}
        for case in result["cases"]]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
