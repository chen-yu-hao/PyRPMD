"""Small, independently referenced CBE/PES-2008 adapter checks, never a rate run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.constants import N_A

from pyrpmd.constants import BOHR_ANGSTROM, HARTREE_J, AU_ENERGY_KCAL_MOL
from pyrpmd.potentials.base import validate_result

from benchmarks.hch4.pes import CBE2008Potential, DEFAULT_LIBRARY, HERE, SOURCE

GEOMETRIES = json.loads((HERE / "geometries.json").read_text())


def paper_geometries(separation_bohr=50.0):
    """Construct the rounded Table III CH4 minimum and C3v saddle, in bohr."""
    table = GEOMETRIES["paper_table_III"]
    reactant = np.zeros((6, 3))
    tetra = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    reactant[[0, 2, 3, 4]] = tetra * (table["methane_ch_angstrom"] / np.sqrt(3))
    reactant[5, 2] = separation_bohr * BOHR_ANGSTROM
    saddle = np.zeros((6, 3))
    theta = np.deg2rad(table["saddle_h_c_transfer_angle_degrees"])
    length = table["saddle_spectator_ch_angstrom"]
    for atom, phi in zip([0, 2, 4], np.deg2rad([0, 120, 240])):
        saddle[atom] = length * np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    saddle[3, 2] = table["saddle_breaking_ch_angstrom"]
    saddle[5, 2] = table["saddle_breaking_ch_angstrom"] + table["saddle_forming_hh_angstrom"]
    return reactant / BOHR_ANGSTROM, saddle / BOHR_ANGSTROM


def finite_difference_forces(potential, points, epsilon):
    force = np.empty_like(points)
    for atom in range(6):
        for axis in range(3):
            plus = points.copy()
            minus = points.copy()
            plus[:, atom, axis] += epsilon
            minus[:, atom, axis] -= epsilon
            ep = validate_result(potential.evaluate(plus), plus.shape).energy.copy()
            em = validate_result(potential.evaluate(minus), minus.shape).energy
            force[:, atom, axis] = -(ep - em)/(2*epsilon)
    return force


def verify(library=DEFAULT_LIBRARY):
    started = time.perf_counter()
    potential = CBE2008Potential(library)
    reactant, saddle = paper_geometries()
    far_reactant, _ = paper_geometries(80.0)
    official = np.asarray(GEOMETRIES["official_input"]["coordinates"]) / BOHR_ANGSTROM
    rng = np.random.default_rng(2026110106)
    # Fixed, small perturbations selected before observing energies or forces.
    points = np.concatenate([official[None], saddle[None], reactant[None],
                             official[None] + rng.normal(scale=0.025, size=(16, 6, 3))])
    observed = validate_result(potential.evaluate(points), points.shape)
    legacy = SOURCE["legacy_constants"]
    expected_gradient_scale = (legacy["gradient_100000_j_mol_per_angstrom_to_hartree_per_bohr"]
                               / (legacy["energy_100000_j_mol_to_hartree"] * legacy["bohr_to_angstrom"]))
    fd_results = []
    for epsilon in (1e-4, 3e-5, 1e-5, 3e-6):
        numerical = finite_difference_forces(potential, points, epsilon)
        delta = observed.forces - numerical
        residual = observed.forces - expected_gradient_scale*numerical
        passed = bool(np.all(np.abs(delta) <= 3e-8 + 5e-6*np.abs(numerical)))
        fd_results.append({
            "epsilon_bohr": epsilon,
            "maximum_absolute_error_hartree_per_bohr": float(np.max(np.abs(delta))),
            "rms_error_hartree_per_bohr": float(np.sqrt(np.mean(delta**2))),
            "observed_least_squares_gradient_scale": float(np.sum(observed.forces*numerical)/np.sum(numerical**2)),
            "maximum_residual_after_known_upstream_scale": float(np.max(np.abs(residual))),
            "raw_gradient_gate": passed,
        })
    barrier = (observed.energy[1]-observed.energy[2])*AU_ENERGY_KCAL_MOL
    target = GEOMETRIES["paper_table_III"]["barrier_kcal_mol"]
    tolerance = GEOMETRIES["paper_table_III"]["predeclared_barrier_tolerance_kcal_mol"]
    tail_error = abs(float(potential.evaluate(far_reactant[None]).energy[0]-observed.energy[2]))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    moved = potential.evaluate(points @ rotation + [4.2, -1.8, 3.1])
    perm = [2, 1, 4, 0, 3, 5]  # only the four equivalent methane H atoms
    reordered = potential.evaluate(points[:, perm])
    net_force = np.max(np.abs(np.sum(observed.forces, axis=1)))
    torque = np.max(np.abs(np.sum(np.cross(points-points[:, 1:2], observed.forces), axis=1)))
    symmetry_errors = {
        "translation_rotation_energy": float(np.max(abs(moved.energy-observed.energy))),
        "translation_rotation_force": float(np.max(abs(moved.forces-observed.forces@rotation))),
        "methane_h_permutation_energy": float(np.max(abs(reordered.energy-observed.energy))),
        "methane_h_permutation_force": float(np.max(abs(reordered.forces-observed.forces[:, perm]))),
        "net_force": float(net_force), "net_torque": float(torque),
    }
    singles = [potential.evaluate(point[None]) for point in points]
    scalar_exact = (np.array_equal(observed.energy, np.concatenate([r.energy for r in singles]))
                    and np.array_equal(observed.forces, np.concatenate([r.forces for r in singles])))
    # PREPOT rescales COMMON-block coefficients in-place. A new Python object
    # must not call it a second time for the same loaded shared library.
    second = CBE2008Potential(library).evaluate(points)
    reinitialize_exact = (np.array_equal(second.energy, observed.energy)
                          and np.array_equal(second.forces, observed.forces))
    mixed_points = np.concatenate([np.zeros((1, 6, 3)), points[:1]])
    mixed = potential.evaluate(mixed_points)
    invalid_guard = bool(mixed.status[0] == 1 and mixed.status[1] == 0
                         and mixed.energy[1] == observed.energy[0])
    modern_energy_factor = 1e5/(N_A*HARTREE_J)
    gates = {
        "paper_barrier_coarse_identity": abs(float(barrier)-target) <= tolerance,
        "asymptotic_50_to_80_bohr": tail_error < 1e-10,
        "raw_gradient_with_declared_legacy_rounding": all(r["raw_gradient_gate"] for r in fd_results),
        "symmetries_and_momentum": all(v < 3e-10 for v in symmetry_errors.values()),
        "batch_scalar_exact": bool(scalar_exact), "second_instance_does_not_rescale": bool(reinitialize_exact),
        "invalid_frame_does_not_poison_batch": invalid_guard,
    }
    return {
        "schema": "pyrpmd.hch4.pes-verification.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Electronic model identity and adapter verification only; no H+CH4 thermal rate or RPMD convergence is claimed.",
        "passed": bool(all(gates.values())), "gates": gates,
        "build": potential.provenance,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "seconds": time.perf_counter()-started,
                    "source_files": {str(p.relative_to(HERE.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in [HERE/"source.json", HERE/"geometries.json", HERE/"wrapper.f90", HERE/"pes.py", Path(__file__)]}},
        "paper_barrier": {"doi": "10.1063/1.3132223", "table": "III, PES column",
                          "reference_kcal_mol": target, "actual_kcal_mol": float(barrier),
                          "difference_kcal_mol": float(barrier)-target, "tolerance_kcal_mol": tolerance,
                          "reactant_energy_hartree": float(observed.energy[2]),
                          "saddle_energy_hartree": float(observed.energy[1]),
                          "reactant_tail_50_to_80_bohr_energy_change": tail_error},
        "units": {"declared": "upstream bohr / Hartree / Hartree per bohr, without rescaling",
                  "coordinate_conversion_used_for_paper_angstrom_geometries": BOHR_ANGSTROM,
                  "legacy_energy_conversion_100000_j_mol": legacy["energy_100000_j_mol_to_hartree"],
                  "modern_energy_conversion_100000_j_mol": modern_energy_factor,
                  "legacy_energy_factor_relative_difference": legacy["energy_100000_j_mol_to_hartree"]/modern_energy_factor-1,
                  "expected_gradient_energy_scale_ratio": expected_gradient_scale,
                  "policy": "No correction applied. Scale-residual values diagnose the source constants, not a replacement force."},
        "finite_difference": {"predeclared_absolute_tolerance": 3e-8, "predeclared_relative_tolerance": 5e-6,
                              "tests": fd_results},
        "symmetry_errors_atomic_units": symmetry_errors,
        "replay": {"seed": 2026110106, "points_bohr": points.tolist(), "energies_hartree": observed.energy.tolist(),
                   "forces_hartree_per_bohr": observed.forces.tolist(), "statuses": observed.status.tolist(),
                   "point_order": "official input; reconstructed paper saddle; reconstructed paper CH4+H at50bohr;16 fixed random perturbations of official input"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.library)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "gates": report["gates"],
                      "barrier": report["paper_barrier"], "finite_difference": report["finite_difference"],
                      "units": report["units"]}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
