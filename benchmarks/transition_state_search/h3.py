"""Validate endpoint and approximate-TS preparation on the real BKMP2 H3 PES.

Build the checksum-pinned PES with ``python benchmarks/h3/build_pes.py`` first.
Run from the repository root with an installed PyRPMD, or with PYTHONPATH=src::

    python benchmarks/transition_state_search/h3.py --output h3.json

This benchmark checks the electronic saddle, its entrance channel, and the
geometry handed to rate sampling. It does not estimate or validate an RPMD
rate. The paper comparison is deliberately limited to quantities reported in
Boothroyd et al., J. Chem. Phys. 104, 7139 (1996), Sec. IV D,
https://doi.org/10.1063/1.471430. Imaginary frequencies below are independently
computed references, not claimed published values.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
import scipy
from scipy.optimize import root_scalar

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.h3.pes import BKMP2Potential, DEFAULT_LIBRARY, SOURCE, verify
from pyrpmd.constants import AMU_TO_AU, AU_ENERGY_KCAL_MOL, WAVENUMBER_TO_AU
from pyrpmd.input import JobSpec
from pyrpmd.io import jsonable
from pyrpmd.runner import prepare
from pyrpmd.system import system_from_config
from pyrpmd.workflows.fes import resolve_umbrellas
from pyrpmd.workflows.prepare import numerical_hessian, prepare_rate
from pyrpmd.workflows.rate import _prepare as prepare_for_run

MASS_AMU = 1.00782503223
MASS_AU = MASS_AMU * AMU_TO_AU
PAPER = {
    "citation": (
        "A. I. Boothroyd, W. J. Keogh, P. G. Martin, M. R. Peterson, "
        "A refined H3 potential energy surface, J. Chem. Phys. 104, 7139–7152 (1996)"
    ),
    "doi": "10.1063/1.471430",
    "url": "https://www.cita.utoronto.ca/~boothroy/papers/bkmp2pre.pdf",
    "location": "Section IV D, Saddle point properties; preprint printed page 12",
    "saddle_neighbor_distance_bohr": 1.757,
    "classical_barrier_millihartree": 15.32,
    "classical_barrier_kcal_mol": 9.613,
    "precision_note": "Comparisons use half of the last reported decimal place, not exact equality.",
    "imaginary_frequency_reference": None,
}


def endpoints(h2_distance, separation=6.0):
    """Atom-mapped collinear A+BC and AB+C; each has zero center of mass."""
    reactant = np.zeros((3, 3))
    product = np.zeros((3, 3))
    reactant[:, 0] = [-separation, 0.0, h2_distance]
    product[:, 0] = [-h2_distance, 0.0, separation]
    return reactant - reactant.mean(0), product - product.mean(0)


def independent_reference(potential):
    """Bracket scalar radial-gradient roots; do not call either TS optimizer."""
    h2 = root_scalar(
        lambda r: potential.radial([r, 99.0, 99.0])[1][0],
        bracket=(1.2, 1.6), xtol=1e-13,
    )

    def symmetric_gradient(r):
        gradient = potential.radial([r, r, 2 * r])[1]
        return float(gradient[0] + gradient[1] + 2 * gradient[2])

    saddle = root_scalar(symmetric_gradient, bracket=(1.5, 2.0), xtol=1e-13)
    if not h2.converged or not saddle.converged:
        raise AssertionError("Independent bracketed radial-gradient root did not converge")
    q = np.array([[-saddle.root, 0, 0], [0, 0, 0], [saddle.root, 0, 0]])
    energy = float(potential.evaluate(q[None]).energy[0])
    asymptote = potential.radial([h2.root, 99.0, 99.0])[0]
    # Equal-mass antisymmetric stretch, normalized in Cartesian space. Its
    # curvature follows from energies, independently of force-difference Hessians.
    mode = np.zeros_like(q)
    mode[:, 0] = np.array([1.0, -2.0, 1.0]) / np.sqrt(6.0)
    convergence = []
    for step in (0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003):
        plus = potential.evaluate((q + step * mode)[None]).energy[0]
        minus = potential.evaluate((q - step * mode)[None]).energy[0]
        curvature = float((plus - 2 * energy + minus) / step**2)
        convergence.append({
            "energy_difference_step_bohr": step,
            "cartesian_mode_curvature_hartree_per_bohr2": curvature,
            "imaginary_frequency_magnitude_cm_minus1": float(
                np.sqrt(-curvature / MASS_AU) / WAVENUMBER_TO_AU
            ),
        })
    hessian_convergence = []
    for step in (1e-3, 3e-4, 1e-4, 3e-5):
        eigenvalues = np.linalg.eigvalsh(numerical_hessian(potential, q, step) / MASS_AU)
        hessian_convergence.append({
            "force_difference_step_bohr": step,
            "mass_weighted_eigenvalues_au": eigenvalues.tolist(),
            "negative_modes_below_minus_1e_minus8": int(np.count_nonzero(eigenvalues < -1e-8)),
            "imaginary_frequency_magnitude_cm_minus1": float(
                np.sqrt(-eigenvalues[0]) / WAVENUMBER_TO_AU
            ),
        })
    force = float(np.max(np.abs(potential.evaluate(q[None]).forces)))
    assert force < 1e-10
    assert all(c["negative_modes_below_minus_1e_minus8"] == 1 for c in hessian_convergence)
    assert abs(convergence[-1]["imaginary_frequency_magnitude_cm_minus1"]
               - hessian_convergence[-1]["imaginary_frequency_magnitude_cm_minus1"]) < 0.01
    barrier = energy - asymptote
    paper_checks = {
        "geometry_error_bohr": float(abs(saddle.root - PAPER["saddle_neighbor_distance_bohr"])),
        "geometry_rounding_tolerance_bohr": 0.0005,
        "barrier_error_millihartree": float(abs(1000 * barrier - PAPER["classical_barrier_millihartree"])),
        "barrier_rounding_tolerance_millihartree": 0.005,
        "barrier_error_kcal_mol": float(abs(barrier * AU_ENERGY_KCAL_MOL - PAPER["classical_barrier_kcal_mol"])),
        "barrier_rounding_tolerance_kcal_mol": 0.0005,
    }
    paper_checks["passed_at_published_precision"] = bool(
        paper_checks["geometry_error_bohr"] <= 0.0005
        and paper_checks["barrier_error_millihartree"] <= 0.005
        and paper_checks["barrier_error_kcal_mol"] <= 0.0005
    )
    assert paper_checks["passed_at_published_precision"]
    return {
        "method": "Brent root of radial analytic gradients in the H2 and symmetric H3 subspaces",
        "h2_equilibrium_distance_bohr": float(h2.root),
        "h2_asymptotic_other_pair_distances_bohr": [99.0, 99.0],
        "h2_asymptotic_energy_hartree": float(asymptote),
        "saddle_geometry_bohr": q.tolist(),
        "saddle_neighbor_distance_bohr": float(saddle.root),
        "saddle_energy_hartree": energy,
        "saddle_max_force_hartree_per_bohr": force,
        "electronic_barrier_hartree": float(barrier),
        "electronic_barrier_kcal_mol": float(barrier * AU_ENERGY_KCAL_MOL),
        "energy_zero": "Three separated H atoms; barrier relative to optimized isolated H2 + H",
        "zero_point_energy_included": False,
        "isotope_mass_amu": MASS_AMU,
        "isotope_mass_electron_masses": float(MASS_AU),
        "energy_curvature_convergence": convergence,
        "force_hessian_convergence": hessian_convergence,
        "paper_comparison": paper_checks,
    }


def job(reactant, product, library, options):
    return {
        "task": "rate", "temperature": "1500 K", "beads": 1,
        "system": {
            "symbols": ["H", "H", "H"], "masses_amu": [MASS_AMU] * 3,
            "geometry_unit": "bohr", "multiplicity": 2,
            "reactant": np.asarray(reactant).tolist(), "product": np.asarray(product).tolist(),
        },
        "potential": {
            "provider": "python", "factory": str(ROOT / "benchmarks/h3/pes.py") + ":build",
            "params": {"library": str(Path(library).resolve())},
        },
        "reaction": {"rinf": "10 bohr", "preparation": copy.deepcopy(options)},
        "dynamics": {"timestep": "0.1 fs", "native": False},
        "parallel": {"backend": "serial"},
    }


def assess(potential, result, reference, *, reverse=False, geometry_tolerance=3e-4,
           force_tolerance=1e-7):
    q = np.asarray(result["transition_state"])
    measured = potential.evaluate(q[None])
    distances = np.array([np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[1] - q[2])])
    distance_error = float(np.max(np.abs(distances - reference["saddle_neighbor_distance_bohr"])))
    energy = float(measured.energy[0])
    eigenvalues = np.linalg.eigvalsh(numerical_hessian(potential, q, 1e-4) / MASS_AU)
    imaginary = float(np.sqrt(-eigenvalues[0]) / WAVENUMBER_TO_AU)
    frequency_reference = reference["energy_curvature_convergence"][-1]["imaginary_frequency_magnitude_cm_minus1"]
    forming = [[1, 2]] if reverse else [[0, 1]]
    breaking = [[0, 1]] if reverse else [[1, 2]]
    expected_fragments = [[0, 1], [2]] if reverse else [[0], [1, 2]]
    channel_ok = (
        result["reaction"].get("forming_bonds") == forming
        and result["reaction"].get("breaking_bonds") == breaking
        and result["reaction"].get("fragments") == expected_fragments
    )
    return {
        "ready": bool(result["ready"]),
        "transition_state_bohr": q.tolist(),
        "neighbor_distances_bohr": distances.tolist(),
        "maximum_neighbor_distance_error_bohr": distance_error,
        "distance_tolerance_bohr": geometry_tolerance,
        "energy_hartree": energy,
        "energy_error_hartree": abs(energy - reference["saddle_energy_hartree"]),
        "electronic_barrier_kcal_mol": (energy - reference["h2_asymptotic_energy_hartree"]) * AU_ENERGY_KCAL_MOL,
        "max_force_hartree_per_bohr": float(np.max(np.abs(measured.forces))),
        "force_tolerance_hartree_per_bohr": force_tolerance,
        "negative_mass_weighted_modes": int(np.count_nonzero(eigenvalues < -1e-8)),
        "imaginary_frequency_magnitude_cm_minus1": imaginary,
        "imaginary_frequency_error_vs_energy_difference_cm_minus1": abs(imaginary - frequency_reference),
        "stricter_frequency_target_cm_minus1": 0.2,
        "stricter_frequency_target_met": bool(abs(imaginary - frequency_reference) < 0.2),
        "channel_matches_labeled_endpoints": channel_ok,
        "forming_bonds": result["reaction"].get("forming_bonds"),
        "breaking_bonds": result["reaction"].get("breaking_bonds"),
        "fragments": result["reaction"].get("fragments"),
        "diagnostics": result["diagnostics"],
        "passed": bool(result["ready"] and channel_ok and distance_error < geometry_tolerance
                       and abs(energy - reference["saddle_energy_hartree"]) < 2e-8
                       and np.count_nonzero(eigenvalues < -1e-8) == 1
                       and np.max(np.abs(measured.forces)) <= force_tolerance),
    }


def validate(library=DEFAULT_LIBRARY):
    started = time.perf_counter()
    potential = BKMP2Potential(library)
    adapter = verify(library)
    reference = independent_reference(potential)
    h2 = reference["h2_equilibrium_distance_bohr"]
    cases = []
    matrix = [
        ("defaults", 6.0, 7, 1e-4, 1e-5, 0.1),
        ("three_images", 4.0, 3, 1e-4, 1e-7, 0.1),
        ("five_images", 4.0, 5, 1e-4, 1e-7, 0.1),
        ("seven_images", 6.0, 7, 1e-4, 1e-7, 0.1),
        ("even_eight_images", 6.0, 8, 1e-4, 1e-7, 0.1),
        ("eleven_images", 6.0, 11, 1e-4, 1e-7, 0.1),
        ("neb_tolerance_1e_minus5", 6.0, 7, 1e-5, 1e-7, 0.1),
        ("neb_tolerance_1e_minus6", 6.0, 7, 1e-6, 1e-7, 0.1),
        ("longer_asymptote", 8.0, 9, 1e-5, 1e-7, 0.1),
        ("weaker_springs", 6.0, 7, 1e-5, 1e-7, 0.05),
        ("stronger_springs", 6.0, 7, 1e-5, 1e-7, 0.2),
    ]
    for name, separation, images, neb_tolerance, ts_tolerance, spring in matrix:
        reactant, product = endpoints(h2, separation)
        controls = {
            "gradient_tolerance": ts_tolerance,
            "neb": {"images": images, "gradient_tolerance": neb_tolerance, "spring_constant": spring},
        }
        config = job(reactant, product, library, controls)
        system = system_from_config(config["system"])
        tic = time.perf_counter()
        result = prepare_rate(config, system, potential, numerical=True)
        row = assess(potential, result, reference, force_tolerance=ts_tolerance)
        row.update(name=name, parameters=controls, endpoint_separation_bohr=separation,
                   neb_iterations=result["neb"]["iterations"],
                   refinement_iterations=result.get("refinement", {}).get("iterations"),
                   neb_max_force=result["neb"]["max_force"],
                   initial_endpoint_energies_hartree=potential.evaluate(np.array([reactant, product])).energy.tolist(),
                   elapsed_seconds=time.perf_counter() - tic)
        cases.append(row)

    reactant, product = endpoints(h2)
    controls = {"gradient_tolerance": 1e-7, "neb": {"gradient_tolerance": 1e-5}}
    rotation, _ = np.linalg.qr(np.random.default_rng(38291).normal(size=(3, 3)))
    variants = [
        ("reversed_endpoints", product, reactant, controls, True),
        ("rotated_translated_endpoints", reactant @ rotation + [2, -3, 0.5],
         product @ rotation + [2, -3, 0.5], controls, False),
    ]
    path = np.linspace(reactant, product, 7)
    path[:, :, 1] += np.sin(np.linspace(0, np.pi, 7))[:, None] * np.array([-0.1, 0.2, -0.1])
    bent_controls = copy.deepcopy(controls)
    bent_controls["neb"]["initial_path"] = path.tolist()
    variants.append(("bent_initial_path", reactant, product, bent_controls, False))
    for name, reactant_case, product_case, settings, reverse in variants:
        config = job(reactant_case, product_case, library, settings)
        result = prepare_rate(config, system_from_config(config["system"]), potential, numerical=True)
        row = assess(potential, result, reference, reverse=reverse)
        row.update(name=name, parameters=settings, neb_iterations=result["neb"]["iterations"],
                   refinement_iterations=result.get("refinement", {}).get("iterations"),
                   neb_max_force=result["neb"]["max_force"])
        cases.append(row)

    perturbations = []
    exact = np.asarray(reference["saddle_geometry_bohr"])
    # Keep entrance fragments explicit to disambiguate the identical H channels,
    # but let preparation infer the bonds from actual downhill trajectories.
    for amplitude in (0.02, 0.05, 0.1):
        for seed in (731, 902, 1873):
            perturbation = np.random.default_rng(seed).normal(size=(3, 3))
            perturbation -= perturbation.mean(0)
            perturbation *= amplitude / np.max(np.linalg.norm(perturbation, axis=1))
            spec = {"symbols": ["H"] * 3, "masses_amu": [MASS_AMU] * 3,
                    "geometry": (exact + perturbation).tolist(), "geometry_unit": "bohr", "multiplicity": 2}
            settings = {"refine": True, "gradient_tolerance": 1e-7}
            config = {"system": spec, "reaction": {"fragments": [[0], [1, 2]], "preparation": settings}}
            result = prepare_rate(config, system_from_config(spec), potential, numerical=True)
            row = assess(potential, result, reference)
            row.update(seed=seed, maximum_initial_atom_displacement_bohr=amplitude,
                       initial_geometry_bohr=spec["geometry"],
                       initial_max_force_hartree_per_bohr=float(np.max(np.abs(potential.evaluate(np.array(spec["geometry"])[None]).forces))),
                       refinement_iterations=result.get("refinement", {}).get("iterations"))
            perturbations.append(row)

    # Exercise the public prepare route and the exact preparation function run_rate
    # calls, without claiming this electronic benchmark performs rate sampling.
    config = job(reactant, product, library, controls)
    public = prepare(config, numerical=True)
    resolved, input_system = JobSpec.from_mapping(config).resolve()
    with tempfile.TemporaryDirectory(prefix="pyrpmd-h3-ts-") as directory:
        run_config, run_system = prepare_for_run(resolved, input_system, directory, True)
        saved = json.loads((Path(directory) / "preparation.json").read_text())
        cv, _, _, _ = resolve_umbrellas(run_config, run_system)
        public_q = np.asarray(public["system"]["geometry"])
        handoff = {
            "public_prepare_ready": public["ready"],
            "public_prepare_vs_run_preparation_max_difference_bohr": float(np.max(np.abs(public_q - run_system.geometry))),
            "saved_preparation_vs_run_geometry_max_difference_bohr": float(np.max(np.abs(np.asarray(saved["transition_state"]) - run_system.geometry))),
            "prepared_geometry_differs_from_input_endpoint": bool(np.max(np.abs(run_system.geometry - input_system.geometry)) > 0.1),
            "rpmdrate_coordinate_at_prepared_ts": float(cv.value(run_system.geometry[None])[0]),
            "scope": "Public prepare and the preparation stage called by run_rate; no sampling in this benchmark",
        }
        handoff["passed"] = bool(
            handoff["public_prepare_ready"]
            and handoff["public_prepare_vs_run_preparation_max_difference_bohr"] == 0.0
            and handoff["saved_preparation_vs_run_geometry_max_difference_bohr"] == 0.0
            and handoff["prepared_geometry_differs_from_input_endpoint"]
            and abs(handoff["rpmdrate_coordinate_at_prepared_ts"] - 1.0) < 1e-10
        )
    return jsonable({
        "schema": "pyrpmd.transition-state-search.h3.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": bool(adapter["passed"] and all(row["passed"] for row in cases + perturbations) and handoff["passed"]),
        "scope": "Electronic saddle/entrance-channel and geometry-handoff validation; no RPMD rate convergence claim",
        "source": SOURCE, "paper_reference": PAPER,
        "provenance": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                       "platform": platform.platform(), "build": potential.provenance,
                       "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "adapter_verification": adapter,
        "independent_reference": reference,
        "endpoint_cases": cases,
        "perturbed_ts_cases": perturbations,
        "frequency_accuracy_summary": {
            "target_cm_minus1": 0.2,
            "cases_meeting_target": sum(row["stricter_frequency_target_met"] for row in cases + perturbations),
            "total_cases": len(cases) + len(perturbations),
            "cases_missing_target": [row.get("name", str((row.get("maximum_initial_atom_displacement_bohr"), row.get("seed"))))
                                     for row in cases + perturbations if not row["stricter_frequency_target_met"]],
            "scope": (
                "Additional accuracy diagnostic, separate from force, saddle-index, and published-precision checks. "
                "Default TS force tolerance is not a frequency-convergence criterion; tighten it for spectra. "
                "All finite-difference step results are retained, and no paper frequency is asserted."
            ),
        },
        "workflow_geometry_handoff": handoff,
        "elapsed_seconds": time.perf_counter() - started,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.library)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "endpoint_cases": len(report["endpoint_cases"]),
                      "perturbed_ts_cases": len(report["perturbed_ts_cases"]),
                      "elapsed_seconds": report["elapsed_seconds"], "output": str(args.output)}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
