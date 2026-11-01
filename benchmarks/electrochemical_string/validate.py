#!/usr/bin/env python3
"""Deterministic coupling checks and a fresh PCET/archive-reference comparison.

Not a reproduction of a published combined constant-potential NRPMD rate.
Use --output FILE; no network access, plotting library, or fitting is required.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from pyrpmd.adaptive_string import adaptive_mean_force_string
from pyrpmd.electrochemistry import ElectronReservoir, ElectronicPESResult, AdiabaticElectrodePotential
from pyrpmd.electrochemistry.diabatic import StateIndependentElectrodeDiabaticPotential
from pyrpmd.nonadiabatic import DiabaticResult, NRPMDState
from pyrpmd.thermal_nrpmd import ThermalNRPMDDynamics


class LocalMatrix:
    """Synthetic AU model, unrelated to the archived PCET comparison."""
    def __init__(self, g=.15):
        self.g = g
    def evaluate(self, q):
        v = np.zeros((len(q), 2, 2))
        grad = np.zeros((len(q), 2, 2, *q.shape[1:]))
        for s in range(2):
            sign = 1-2*s
            v[:, s, s] = .35*np.sum(q*q, axis=(1, 2)) + sign*(self.g*q[:, 0, 0]+.06)
            grad[:, s, s] = .7*q
            grad[:, s, s, 0, 0] += sign*self.g
        v[:, 0, 1] = v[:, 1, 0] = .2
        return DiabaticResult(v, grad)


class Charging:
    def evaluate(self, q, electron_numbers):
        n = electron_numbers
        force = np.zeros_like(q)
        force[:, 0, 0] = .4*n
        return ElectronicPESResult(.65*n*n-.4*q[:, 0, 0]*n,
                                   force, 1.3*n-.4*q[:, 0, 0])


def model(g=.15, voltage=0.):
    bath = ElectronReservoir(voltage, 0., .23, "synthetic-model-not-SHE")
    electrode = AdiabaticElectrodePotential(Charging(), bath, electron_bounds=(-20, 20),
                                           chemical_potential_tolerance_au=1e-12)
    return StateIndependentElectrodeDiabaticPotential(LocalMatrix(g), electrode)


def vector(s):
    return np.concatenate([s.q.ravel(), s.p.ravel(), s.mapping_q.ravel(), s.mapping_p.ravel()])


def ode_reference(y0, time):
    # Independently coded Hamilton equations for the explicit model, not an
    # adapter call and not a reference quantum-mechanical dynamics algorithm.
    def rhs(t, y):
        q, p = y[:12].reshape(4, 1, 3), y[12:24].reshape(4, 1, 3)
        x, z = y[24:32].reshape(4, 2), y[32:].reshape(4, 2)
        n = (.4*q[:, 0, 0]+.23)/1.3
        f = -.7*q
        f[:, 0, 0] += .4*n - .075*(x[:, 0]**2+z[:, 0]**2-x[:, 1]**2-z[:, 1]**2)
        f -= (4/3)**2*(2*q-np.roll(q, 1, axis=0)-np.roll(q, -1, axis=0))
        gap = .15*q[:, 0, 0]+.06
        vx = np.column_stack((gap*x[:, 0]+.2*x[:, 1], .2*x[:, 0]-gap*x[:, 1]))
        vz = np.column_stack((gap*z[:, 0]+.2*z[:, 1], .2*z[:, 0]-gap*z[:, 1]))
        return np.concatenate([p.ravel(), f.ravel(), vz.ravel(), -vx.ravel()])
    result = solve_ivp(rhs, (0., time), y0, method="DOP853", rtol=2e-12, atol=2e-14)
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[:, -1]


def deterministic_checks():
    potential = model()
    rng = np.random.default_rng(812)
    state = NRPMDState(rng.normal(size=(4, 1, 3))*.2, rng.normal(size=(4, 1, 3))*.2,
                      mapping_q=rng.normal(size=(4, 2)), mapping_p=rng.normal(size=(4, 2)))
    solved = potential.evaluate_with_electrode(state.q)
    expected_n = (.4*state.q[:, 0, 0]+.23)/1.3
    eps = 1e-5
    fd_error = 0.
    for c in range(3):
        qp, qm = state.q.copy(), state.q.copy()
        qp[:, 0, c] += eps
        qm[:, 0, c] -= eps
        fd = (potential.evaluate(qp).matrix-potential.evaluate(qm).matrix)/(2*eps)
        fd_error = max(fd_error, float(np.max(np.abs(fd-solved.diabatic.gradients[:, :, :, 0, c]))))
    exact = ode_reference(vector(state), 2.)
    trajectories = []
    for dt in (.04, .02, .01):
        dyn = ThermalNRPMDDynamics(potential, [1.], 3., 4, dt, nstates=2, native=False)
        s = state.copy()
        e0 = dyn.energy(s)
        maximum_drift = 0.
        for _ in range(round(2./dt)):
            dyn.step(s)
            maximum_drift = max(maximum_drift, abs(dyn.energy(s)-e0))
        trajectories.append({"dt_au": dt, "time_au": 2.,
            "max_phase_space_error_vs_independent_ode": float(np.max(np.abs(vector(s)-exact))),
            "max_hamiltonian_drift_au": maximum_drift})
    # Constant matrix limit: finite phase quadrature gives an exact electronic
    # initial projector while the nuclear ring still moves on the electrode PES.
    uncoupled = model(g=0.)
    dt, steps = .02, 100
    times = np.arange(steps+1)*dt
    values = np.zeros(steps+1)
    for a in np.arange(4)*np.pi/2:
        for b in np.arange(4)*np.pi/2:
            angles = np.array([a, b])
            x = np.sqrt([3., 1.])*np.cos(angles)
            p = np.sqrt([3., 1.])*np.sin(angles)
            s = NRPMDState(state.q.copy(), state.p.copy(),
                mapping_q=np.tile(x, (4, 1)), mapping_p=np.tile(p, (4, 1)))
            dyn = ThermalNRPMDDynamics(uncoupled, [1.], 3., 4, dt, nstates=2, native=False)
            values[0] += s.populations()[1]/16
            for j in range(steps):
                dyn.step(s)
                values[j+1] += s.populations()[1]/16
    reference = .2**2/(.2**2+.06**2)*np.sin(np.sqrt(.2**2+.06**2)*times)**2
    path = np.column_stack((np.linspace(-1, 1, 9), .15*np.sin(np.linspace(0, np.pi, 9))))
    def oracle(x):
        return np.column_stack((4*x[:, 0]*(x[:, 0]**2-1), 20*x[:, 1]))
    string = adaptive_mean_force_string(path, oracle, step=.02, tolerance=1e-5, max_iterations=500)
    return {
        "scope": "synthetic state-independent electrode; numerical/limiting-case validation, not a physical rate",
        "model_au": {"beta": 3., "mass": 1., "beads": 4, "k": .7, "charging_curvature": 1.3,
                     "charge_nuclear_coupling": .4, "mu": .23, "diabatic_slope": .15, "coupling": .2, "bias": .12},
        "max_number_error": float(np.max(np.abs(solved.electrode.electron_numbers-expected_n))),
        "max_chemical_potential_residual_au": float(np.max(np.abs(solved.electrode.chemical_potentials_au-.23))),
        "max_matrix_gradient_fd_error_au": fd_error,
        "integrator_vs_independent_classical_ode": trajectories,
        "constant_matrix_quantum_limit": {"max_population_error": float(np.max(np.abs(values-reference))),
            "reference": "analytic coherent two-level transition, not a rate", "time_au": times.tolist(),
            "nrpmd_population": values.tolist(), "exact_population": reference.tolist()},
        "string": {"scope": "separate analytic Euclidean-gradient check; NOT sampled electrochemical quantum PMF",
            "converged": string.converged, "updates": string.iterations, "images": len(string.images),
            "residual": string.maximum_perpendicular_force,
            "max_interior_y_error": float(np.max(np.abs(string.images[1:-1, 1])))},
    }


def pcet_archived_reference_comparison():
    directory = ROOT / "docs/validation/pcet-model-2026-10-30"
    source = directory / "reference_results.json"
    archived = json.loads(source.read_text())
    spec = importlib.util.spec_from_file_location("pcet_validation_harness", directory / "compare_pypcet.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    fresh = harness.run_pyrpmd(1601, 24)
    reference = archived["reference_runs"]["fgh_512_states_24"]
    cases = []
    maximum = 0.
    for case in fresh["cases"]:
        ref = next(r for r in reference["cases"] if r["isotope"] == case["isotope"] and
                   r["potential_minus_reference_V"] == case["potential_minus_reference_V"])
        result = dict(case)
        for direction in ("reduction", "oxidation"):
            key = direction + "_rate_s"
            error = 100*(case[key]/ref[key]-1)
            result["reference_"+key] = ref[key]
            result[direction+"_relative_difference_percent"] = error
            maximum = max(maximum, abs(error))
        cases.append(result)
    kmax = max(abs(100*(a[key]/b[key]-1)) for a, b in zip(fresh["isotope_effects"], reference["isotope_effects"])
               for key in ("reduction_KIE_H_over_D", "oxidation_KIE_H_over_D"))
    return {"scope": "fresh PyRPMD golden-rule PCET vs archived author-code outputs; NOT the new NRPMD adapter",
            "reference_reexecuted": False, "reference_commit": archived["reference"]["commit"],
            "archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "parameters": archived["parameters"], "fresh_grid_points": 1601, "fresh_states": 24,
            "max_rate_difference_percent": maximum, "max_KIE_difference_percent": kmax,
            "cases": cases, "fresh_isotope_effects": fresh["isotope_effects"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "physical_converged": False, "combined_rate_available": False,
        "deterministic": deterministic_checks(), "pcet": pcet_archived_reference_comparison()}
    d = result["deterministic"]
    errors = [row["max_phase_space_error_vs_independent_ode"] for row in d["integrator_vs_independent_classical_ode"]]
    checks = {
        "charging": d["max_number_error"] < 1e-11 and d["max_chemical_potential_residual_au"] < 1e-11,
        "matrix_gradients": d["max_matrix_gradient_fd_error_au"] < 1e-9,
        "second_order": all(3.8 < a/b < 4.2 for a, b in zip(errors[:-1], errors[1:])),
        "constant_matrix_limit": d["constant_matrix_quantum_limit"]["max_population_error"] < 5e-12,
        "string": d["string"]["converged"] and d["string"]["max_interior_y_error"] < 1e-5,
        "archived_pcet_regression": result["pcet"]["max_rate_difference_percent"] < .07,
    }
    result["numerical_checks"] = checks
    files = ["src/pyrpmd/adaptive_string.py", "src/pyrpmd/electrochemistry/diabatic.py", "src/pyrpmd/thermal_nrpmd.py",
             "src/pyrpmd/nonadiabatic.py", "src/pyrpmd/pcet.py", "benchmarks/electrochemical_string/validate.py"]
    result["source_sha256"] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    if not all(checks.values()):
        raise RuntimeError(f"Numerical validation failed; inspect {args.output}")
    print(json.dumps({"output": str(args.output), "checks": result["deterministic"]["string"],
        "max_PCET_rate_difference_percent": result["pcet"]["max_rate_difference_percent"]}, indent=2))


if __name__ == "__main__":
    main()
