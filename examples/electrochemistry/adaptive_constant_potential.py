"""Adaptive reaction-path sampling at constant potential with thermal NRPMD.

This completely analytic, uncalibrated model exercises the coupled interfaces.
Its units are bohr, hartree, electron masses and atomic time.  The short sample
budgets do not establish physical convergence, a rate constant or an electrode
current.  Monte Carlo sweeps and path-update iterations have no physical time.

Run from the repository root, for example::

    python examples/electrochemistry/adaptive_constant_potential.py \
        --potential-V -0.3 --beads 2 --output runs/adaptive-electrode.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyrpmd.cv import Cartesian, CVStack
from pyrpmd.electrochemistry import (
    AdiabaticElectrodePotential, ConstantPotentialDiabaticPotential,
    ElectronReservoir, ElectronicPESResult,
)
from pyrpmd.nonadiabatic import DiabaticResult
from pyrpmd.thermal_nrpmd import ThermalNRPMDDynamics


class CurvedTwoStateModel:
    """Confined curved valley with a real, smooth two-state coupling.

    One toy particle has coordinates (x,y,z).  The common valley is
    0.3*(x²-1)² + 2*[y-0.6*(1-x²)]² + 0.4*z².  The diagonal splitting is
    ±(0.04+0.10*x); the coupling is 0.08*exp(-0.3*x²).  Charging is absent
    here and is supplied exactly once by CommonChargingModel below.
    """

    def evaluate(self, q):
        if q.shape[1:] != (1, 3):
            raise ValueError("the toy model requires one particle with three coordinates")
        x, y, z = q[:, 0].T
        displacement = y - 0.6*(1-x*x)
        common = 0.3*(x*x-1)**2 + 2.0*displacement**2 + 0.4*z*z
        gradient = np.empty_like(q)
        gradient[:, 0, 0] = 1.2*x*(x*x-1) + 4.8*x*displacement
        gradient[:, 0, 1] = 4.0*displacement
        gradient[:, 0, 2] = 0.8*z
        gap = 0.04 + 0.10*x
        coupling = 0.08*np.exp(-0.3*x*x)
        matrix = np.empty((len(q), 2, 2))
        matrix[:, 0, 0], matrix[:, 1, 1] = common+gap, common-gap
        matrix[:, 0, 1] = matrix[:, 1, 0] = coupling
        gradients = np.zeros((len(q), 2, 2, 1, 3))
        gradients[:, 0, 0] = gradients[:, 1, 1] = gradient
        gradients[:, 0, 0, 0, 0] += 0.10
        gradients[:, 1, 1, 0, 0] -= 0.10
        gradients[:, 0, 1, 0, 0] = gradients[:, 1, 0, 0, 0] = -0.6*x*coupling
        return DiabaticResult(matrix, gradients)


class CommonChargingModel:
    """E_charge=(N-0.3*x-0.08*y)²/(2*1.2), common to both electronic states.

    The model is strictly convex in continuous excess electron number N.  Its
    optimum at reservoir chemical potential mu is N*=0.3*x+0.08*y+1.2*mu.
    It contains no second copy of the nuclear valley or diabatic splitting.
    """

    def evaluate(self, q, electron_numbers):
        center = 0.3*q[:, 0, 0] + 0.08*q[:, 0, 1]
        derivative = (electron_numbers-center)/1.2
        energy = 0.6*derivative*derivative
        forces = np.zeros_like(q)
        forces[:, 0, 0] = 0.3*derivative
        forces[:, 0, 1] = 0.08*derivative
        return ElectronicPESResult(energy, forces, derivative)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _charge_diagnostics(potential, q, *, scope):
    result = potential.solve(q)
    return {
        "scope": scope,
        "independent_configurations": len(q),
        "minimum_electron_number": float(result.electron_numbers.min()),
        "maximum_electron_number": float(result.electron_numbers.max()),
        "maximum_chemical_potential_residual_hartree": float(
            np.max(np.abs(result.chemical_potential_residuals_au))),
        "solver_iterations": result.electrode.iterations,
    }


def _real_time_report(parents, potential, masses, beta, beads, *, quick, min_average_sign):
    """Launch detached parents with the same potential and no umbrella or bath."""
    dt, steps = 0.01, (12 if quick else 60)
    dynamics = ThermalNRPMDDynamics(potential, masses, beta, beads, dt, 2, native=False)
    count = len(parents.states)
    centroids = np.empty((count, steps+1, 3))
    populations = np.empty((count, steps+1, 2))
    energies = np.empty((count, steps+1))
    for index, parent in enumerate(parents.states):
        state = parent.copy()
        for frame in range(steps+1):
            if frame:
                dynamics.step(state)
            centroids[index, frame] = state.q.mean(axis=0)[0]
            populations[index, frame] = state.populations()
            energies[index, frame] = dynamics.energy(state)
    signs = parents.signs
    average_sign = float(signs.mean())
    reasons = [] if average_sign >= min_average_sign else ["resampled_average_sign_below_threshold"]
    resolved = not reasons
    denominator = float(signs.sum())
    mean_centroids = np.einsum("n,ntd->td", signs, centroids)/denominator if resolved else None
    mean_populations = np.einsum("n,nts->ts", signs, populations)/denominator if resolved else None
    return {
        "available": True,
        "estimator": "resampled_absolute_MBAR_parents_with_signed_observables",
        "bias_applied": False, "thermostat_applied": False,
        "same_constant_potential_as_sampling": True,
        "parents": count, "parent_average_sign": average_sign,
        "parent_sampling_diagnostics": parents.sampling_diagnostics,
        "observables_available": resolved, "unavailable_reasons": reasons,
        "time_au": (np.arange(steps+1)*dt).tolist(),
        "signed_centroid_mean_bohr": mean_centroids,
        "signed_MMST_population_mean": mean_populations,
        "population_note": "Signed mapping estimators are not clipped probabilities.",
        "initial_sum_bead_propagation_hamiltonian_hartree": energies[:, 0],
        "maximum_absolute_hamiltonian_drift_hartree": float(
            np.max(np.abs(energies-energies[:, :1]))),
        "energy_note": "Propagation Hamiltonian diagnostic, not a thermal quantum energy estimator.",
        "charge": _charge_diagnostics(potential, np.concatenate([s.q for s in parents.states]),
                                      scope="resampled_parent_beads"),
         "rate_constant": None,
    }


def run(*, potential_V=-0.3, beads=2, quick=False, seed=20261101):
    from pyrpmd.adaptive_nrpmd import AdaptivePathOptions, run_adaptive_nrpmd_path

    if isinstance(beads, bool) or not isinstance(beads, (int, np.integer)) or beads < 1:
        raise ValueError("beads must be a positive integer")
    masses, beta = [1.5], 3.0
    reservoir = ElectronReservoir(
        potential_V=potential_V, reference_potential_V=0.0,
        chemical_potential_at_reference_au=0.1,
        potential_reference="analytic-model-reference; no SHE/RHE calibration",
    )
    electrode = AdiabaticElectrodePotential(
        CommonChargingModel(), reservoir, electron_bounds=(-5.0, 5.0),
        chemical_potential_tolerance_au=1e-10,
    )
    potential = ConstantPotentialDiabaticPotential(CurvedTwoStateModel(), electrode)
    cv = CVStack([Cartesian(0, axis=0, name="x"), Cartesian(0, axis=1, name="y")])
    images = np.zeros((5, beads, 1, 3))
    images[:, :, 0, 0] = np.linspace(-1, 1, len(images))[:, None]
    initial_centers = np.stack([cv.value(image) for image in images])
    options = AdaptivePathOptions(
        max_iterations=2 if quick else 4,
        pilot_samples=64 if quick else 128,
        pilot_burn_in=32 if quick else 100, pilot_stride=2,
        production_samples=48 if quick else 256,
        production_burn_in=32 if quick else 120, production_stride=1 if quick else 3,
        relaxation=0.5, max_step=0.15, path_tolerance=0.03,
        min_average_sign=0.1, min_effective_samples=10.0, overlap_threshold=0.03,
        nuclear_step=0.25, centroid_step=0.2,
    )
    result = run_adaptive_nrpmd_path(
        potential, masses, beta, images, 2, cv, stiffness=[5.0, 5.0],
        options=options, cv_scales=[1.0, 1.0], seed=seed,
    )
    center_configurations = np.zeros((len(result.centers), 1, 3))
    center_configurations[:, 0, :2] = result.centers
    report = {
        "schema_version": 1,
        "model": "analytic_curved_two_state_valley_with_common_convex_charging",
        "units": {"position": "bohr", "energy": "hartree", "mass": "electron_mass",
                  "time": "atomic_time", "beta": "hartree^-1", "potential": "volt"},
        "beta_au": beta, "masses_au": masses, "beads": beads, "seed": seed,
        "quick": bool(quick),
        "reservoir": {
            "potential_V": reservoir.potential_V,
            "reference_potential_V": reservoir.reference_potential_V,
            "chemical_potential_at_reference_au": reservoir.chemical_potential_at_reference_au,
            "chemical_potential_au": reservoir.chemical_potential_au,
            "potential_reference": reservoir.potential_reference,
        },
        "model_scope": [
            "state-independent scalar charging on the full diabatic matrix",
            "one electronic minimum per nuclear bead at a fixed reservoir chemical potential",
            "signed thermal SEO sampling followed by frozen-umbrella MBAR reweighting",
            "approximate NRPMD real-time propagation without an electronic reservoir response time",
        ],
        "initial_centers_bohr": initial_centers,
        "frozen_centers_bohr": result.centers,
        "path_history": result.history,
        "sampling_summary": result.summary(),
        "charge_at_frozen_centers": _charge_diagnostics(
            potential, center_configurations, scope="collapsed_rings_at_frozen_CV_centers"),
        "sampling_iterations_have_physical_time": False,
         "rate_constant": None,
    }
    unavailable = list(result.reweighted.unavailable_reasons)
    if unavailable:
        report["signed_equilibrium_CV_mean_bohr"] = None
        report["real_time"] = {
            "available": False, "unavailable_reasons": unavailable,
            "note": "Production reweighting did not resolve the requested diagnostics; no parents were launched.",
             "rate_constant": None,
        }
    else:
        report["signed_equilibrium_CV_mean_bohr"] = result.reweighted.expectation(result.reweighted.cv_values)
        parents = result.reweighted.resample_parents(8 if quick else 32, seed=seed+1)
        report["real_time"] = _real_time_report(
            parents, potential, masses, beta, beads,
            quick=quick, min_average_sign=options.min_average_sign,
        )
    return _jsonable(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the complete JSON report")
    parser.add_argument("--potential-V", type=float, default=-0.3)
    parser.add_argument("--beads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--quick", action="store_true", help="small software smoke-test sampling budget")
    args = parser.parse_args(argv)
    if args.beads < 1:
        parser.error("--beads must be a positive integer")
    report = run(potential_V=args.potential_V, beads=args.beads, quick=args.quick, seed=args.seed)
    content = json.dumps(report, indent=2, allow_nan=False)+"\n"
    if args.output is None:
        print(content, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
