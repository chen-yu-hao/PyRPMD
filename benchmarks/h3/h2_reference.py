"""Numerical j=0 H2 vibrational levels on the verified BKMP2 asymptote.

Run from the repository root, for example::

    PYTHONPATH=src python benchmarks/h3/h2_reference.py

This solves for the reduced radial wavefunction u(r), normalized with measure
dr, using -d²/(2*mu*dr²) + V(r). It is an isolated-diatom numerical reference,
not an RPMD trajectory or a calculation of reactive-scattering probabilities.
The third H nucleus is held at equal pair distances of 99 bohr in the PES.
The positive-r box uses Dirichlet walls, checked by an independent box change.
Second-order finite differences are refined four times; reported reference
energies use Richardson extrapolation and retain the unextrapolated results.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.h3.pes import BKMP2Potential, DEFAULT_LIBRARY, SOURCE
from pyrpmd.constants import AU_TIME_FS, EV_TO_AU, WAVENUMBER_TO_AU


REDUCED_MASS = 918.5763237198094
OTHER_DISTANCE = 99.0
GRID_STEPS = (0.004, 0.002, 0.001, 0.0005)
BASE_DOMAIN = (0.2, 8.0)
CHECK_DOMAIN = (0.25, 10.0)
PAPER_WAVENUMBER = 4407.0
ENERGY_TOLERANCE_CM = 1e-4


def radial_values(potential, r):
    """Batch V(r01=r, r12=99, r02=99) using an exact isosceles geometry."""
    r = np.asarray(r, dtype=float)
    if r.ndim != 1 or np.any(r < 0.2) or np.any(r >= 2 * OTHER_DISTANCE):
        raise ValueError("Radial grid must lie in the guarded BKMP2 positive-r domain")
    q = np.zeros((len(r), 3, 3))
    q[:, 0, 0] = -r / 2
    q[:, 1, 0] = r / 2
    q[:, 2, 2] = np.sqrt(OTHER_DISTANCE**2 - (r / 2)**2)
    result = potential.evaluate(q)
    if np.any(result.status) or not np.isfinite(result.energy).all():
        raise RuntimeError("BKMP2 rejected a radial-grid geometry")
    return result.energy.copy()


def harmonic_reference(potential, reduced_mass):
    minimum = minimize_scalar(
        lambda r: potential.radial([r, OTHER_DISTANCE, OTHER_DISTANCE])[0],
        bounds=(1.2, 1.6), method="bounded", options={"xatol": 1e-13},
    )
    if not minimum.success:
        raise RuntimeError("Could not locate the BKMP2 asymptotic H2 minimum")
    curvature = []
    for step in (0.001, 0.0003, 0.0001, 0.00003, 0.00001):
        left = potential.radial([minimum.x-step, OTHER_DISTANCE, OTHER_DISTANCE])[1][0]
        right = potential.radial([minimum.x+step, OTHER_DISTANCE, OTHER_DISTANCE])[1][0]
        force_constant = float((right-left) / (2*step))
        omega = float(np.sqrt(force_constant / reduced_mass))
        curvature.append({
            "step_bohr": step, "force_constant_hartree_per_bohr2": force_constant,
            "omega_au": omega, "wavenumber_cm_minus1": omega / WAVENUMBER_TO_AU,
        })
    selected = curvature[-2]
    return {
        "r_eq_bohr": float(minimum.x), "minimum_absolute_hartree": float(minimum.fun),
        "omega_au": selected["omega_au"],
        "wavenumber_cm_minus1": selected["wavenumber_cm_minus1"],
        "period_fs": 2*np.pi / selected["omega_au"] * AU_TIME_FS,
        "curvature_step_bohr": selected["step_bohr"],
        "curvature_convergence": curvature,
        "comparison_harmonic_wavenumber_cm_minus1": PAPER_WAVENUMBER,
        "comparison_value_scope": "Requested literature harmonic-model parameter, not an anharmonic vibrational spacing.",
    }


def solve_grid(potential, minimum, reduced_mass, domain, spacing, levels):
    intervals = int(round((domain[1]-domain[0]) / spacing))
    r = np.linspace(domain[0], domain[1], intervals+1)
    actual_spacing = float(r[1]-r[0])
    if not np.isclose(actual_spacing, spacing, rtol=1e-10, atol=1e-14):
        raise ValueError("Grid spacing must divide the selected radial box")
    potential_absolute = radial_values(potential, r)
    potential_relative = potential_absolute - minimum
    kinetic_diagonal = 1 / (reduced_mass * actual_spacing**2)
    diagonal = potential_relative[1:-1] + kinetic_diagonal
    off_diagonal = np.full(len(diagonal)-1, -kinetic_diagonal/2)
    energies, vectors = eigh_tridiagonal(
        diagonal, off_diagonal, select="i", select_range=(0, levels-1),
        lapack_driver="stebz", tol=1e-13, check_finite=True,
    )
    # LAPACK uses sum(phi**2)=1. u=phi/sqrt(dr) uses the physical radial dr measure.
    vectors *= np.where(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(levels)] < 0, -1.0, 1.0)
    wavefunctions = np.zeros((levels, len(r)))
    wavefunctions[:, 1:-1] = vectors.T / np.sqrt(actual_spacing)
    overlap = actual_spacing * wavefunctions @ wavefunctions.T
    norms = np.diag(overlap)
    residual = diagonal[:, None] * vectors - vectors * energies[None]
    residual[:-1] += off_diagonal[:, None] * vectors[1:]
    residual[1:] += off_diagonal[:, None] * vectors[:-1]
    residual_norm = np.linalg.norm(residual, axis=0)
    nodes = []
    for wavefunction in wavefunctions:
        core = wavefunction[np.abs(wavefunction) > 1e-7 * np.max(np.abs(wavefunction))]
        nodes.append(int(np.count_nonzero(core[1:] * core[:-1] < 0)))
    meta = {
        "domain_bohr": list(domain), "spacing_bohr": actual_spacing,
        "grid_points_including_walls": len(r), "interior_matrix_dimension": len(diagonal),
        "boundary": "Dirichlet u(r_min)=u(r_max)=0",
        "energies_relative_to_minimum_hartree": energies.tolist(),
        "energies_relative_to_minimum_cm_minus1": (energies/WAVENUMBER_TO_AU).tolist(),
        "normalization_integrals_dr": norms.tolist(),
        "maximum_overlap_error": float(np.max(np.abs(overlap-np.eye(levels)))),
        "eigensolver_residual_l2_hartree": residual_norm.tolist(),
        "nodes_above_relative_amplitude_1e_minus7": nodes,
        "lower_tail_probability_r_below_0_4_bohr": (
            actual_spacing*np.sum(wavefunctions[:, r < 0.4]**2, axis=1)).tolist(),
        "upper_tail_probability_r_above_6_bohr": (
            actual_spacing*np.sum(wavefunctions[:, r > 6.0]**2, axis=1)).tolist(),
    }
    return meta, {
        "r_bohr": r, "potential_absolute_hartree": potential_absolute,
        "potential_relative_hartree": potential_relative,
        "u_radial": wavefunctions, "probability_density_dr": wavefunctions**2,
        "R_radial": wavefunctions/r[None], "grid_energies_hartree": energies,
    }


def harmonic_ring_energies(omega, name):
    """Canonical harmonic primitive-energy estimator, not H_P/P.

    E_P = (1/beta) sum_k omega²/(omega²+omega_k²)
        = (omega/2)*coth(P*asinh(beta*omega/(2P)))/sqrt(1+(beta*omega/(2P))²).
    The continuum result is (omega/2)*coth(beta*omega/2).
    """
    beads = 32
    rows = []
    for beta in (200.0, 300.0, 400.0):
        x = beta*omega/(2*beads)
        ratio = 1 / (np.tanh(beads*np.arcsinh(x))*np.sqrt(1+x*x))
        continuum = 1 / np.tanh(beta*omega/2)
        ring_frequencies = 2*beads/beta * np.sin(np.pi*np.arange(beads)/beads)
        independent_sum = float(np.sum(omega**2/(omega**2+ring_frequencies**2))/beta)
        if not np.isclose(independent_sum, omega/2*ratio, rtol=1e-13, atol=1e-15):
            raise AssertionError("Finite-P energy closed form disagrees with the ring-mode sum")
        rows.append({
            "model": name, "beta_au": beta, "beads": beads, "omega_au": omega,
            "finite_P_energy_hartree": float(omega/2*ratio),
            "finite_P_energy_over_harmonic_ZPE": float(ratio),
            "continuum_energy_hartree": float(omega/2*continuum),
            "continuum_energy_over_harmonic_ZPE": float(continuum),
            "finite_P_relative_error_to_continuum": float(ratio/continuum-1),
        })
    return rows


def calculate(library=DEFAULT_LIBRARY, *, reduced_mass=REDUCED_MASS, levels=3):
    if not np.isfinite(reduced_mass) or reduced_mass <= 0 or not 1 <= levels <= 5:
        raise ValueError("Need a positive reduced mass and 1 to 5 low vibrational levels")
    potential = BKMP2Potential(library)
    harmonic = harmonic_reference(potential, reduced_mass)
    minimum = harmonic["minimum_absolute_hartree"]
    grids, arrays = [], None
    for step in GRID_STEPS:
        meta, arrays = solve_grid(potential, minimum, reduced_mass, BASE_DOMAIN, step, levels)
        grids.append(meta)
    domain_meta, _ = solve_grid(potential, minimum, reduced_mass, CHECK_DOMAIN, GRID_STEPS[-1], levels)
    energies = [np.asarray(item["energies_relative_to_minimum_hartree"]) for item in grids]
    extrapolated = (4*energies[-1]-energies[-2])/3
    previous_extrapolated = (4*energies[-2]-energies[-3])/3
    extrapolation_change = np.abs(extrapolated-previous_extrapolated)
    grid_error = np.abs(extrapolated-energies[-1])
    domain_change = np.abs(np.asarray(domain_meta["energies_relative_to_minimum_hartree"])-energies[-1])
    residual = np.asarray(grids[-1]["eigensolver_residual_l2_hartree"])
    estimated_error = np.maximum.reduce([extrapolation_change, domain_change, residual])
    normalization_error = max(item["maximum_overlap_error"] for item in [*grids, domain_meta])
    passed = bool(
        np.max(estimated_error)/WAVENUMBER_TO_AU < ENERGY_TOLERANCE_CM
        and normalization_error < 1e-10
        and all(item["nodes_above_relative_amplitude_1e_minus7"] == list(range(levels))
                for item in [*grids, domain_meta])
    )
    omega = harmonic["omega_au"]
    paper_omega = PAPER_WAVENUMBER * WAVENUMBER_TO_AU
    rows = []
    for v, energy in enumerate(extrapolated):
        probability = arrays["probability_density_dr"][v]
        r = arrays["r_bohr"]
        mean_r = float(trapezoid(r*probability, r))
        rows.append({
            "v": v, "j": 0,
            "energy_above_well_minimum_hartree": float(energy),
            "energy_above_well_minimum_cm_minus1": float(energy/WAVENUMBER_TO_AU),
            "energy_above_well_minimum_eV": float(energy/EV_TO_AU),
            "absolute_energy_hartree": float(energy+minimum),
            "spacing_from_previous_hartree": None if v == 0 else float(energy-extrapolated[v-1]),
            "spacing_from_previous_cm_minus1": None if v == 0 else float((energy-extrapolated[v-1])/WAVENUMBER_TO_AU),
            "harmonic_BKMP2_energy_cm_minus1": float(omega*(v+0.5)/WAVENUMBER_TO_AU),
            "harmonic_4407_energy_cm_minus1": float(PAPER_WAVENUMBER*(v+0.5)),
            "anharmonic_minus_harmonic_BKMP2_cm_minus1": float((energy-omega*(v+0.5))/WAVENUMBER_TO_AU),
            "anharmonic_minus_harmonic_4407_cm_minus1": float((energy-paper_omega*(v+0.5))/WAVENUMBER_TO_AU),
            "estimated_numerical_error_cm_minus1": float(estimated_error[v]/WAVENUMBER_TO_AU),
            "finest_grid_to_extrapolation_cm_minus1": float(grid_error[v]/WAVENUMBER_TO_AU),
            "mean_r_bohr": mean_r,
            "standard_deviation_r_bohr": float(np.sqrt(trapezoid((r-mean_r)**2*probability, r))),
        })
    report = {
        "schema": "pyrpmd.h3.h2-radial-reference.v1", "passed": passed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "j=0 reduced-radial Schrodinger equation; second-order symmetric tridiagonal finite differences; Richardson energy extrapolation",
        "normalization": "integral |u(r)|^2 dr = 1; R(r)=u(r)/r; radial probability density is |u(r)|^2, without an additional r^2 factor",
        "scope": "Low H2 vibrational levels of the BKMP2 isolated-diatom asymptote, using finite 99-bohr spectator distances. No scattering or nuclear-spin/electronic degeneracy factors.",
        "energy_zero": "Optimized asymptotic H2 well minimum; original BKMP2 absolute zero is three separated H atoms",
        "reduced_mass_atomic": reduced_mass,
        "spectator_pair_distances_bohr": [OTHER_DISTANCE, OTHER_DISTANCE],
        "radial_pair_order": ["r01", "r12", "r02"],
        "harmonic_reference": harmonic,
        "levels": rows,
        "convergence": {
            "grids": grids, "domain_check": domain_meta,
            "extrapolation": "E(h->0) = (4 E(h/2)-E(h))/3 using the two finest base-box grids",
            "extrapolation_change_cm_minus1": (extrapolation_change/WAVENUMBER_TO_AU).tolist(),
            "same_spacing_domain_change_cm_minus1": (domain_change/WAVENUMBER_TO_AU).tolist(),
            "estimated_error_definition": "max(change of successive Richardson extrapolations, same-spacing box change, finest-matrix residual norm); this is a numerical consistency estimate, not a bound on PES or mass-model accuracy",
            "acceptance_tolerance_cm_minus1": ENERGY_TOLERANCE_CM,
            "maximum_overlap_error": normalization_error,
        },
        "harmonic_ring_model": {
            "definition": "Canonical harmonic primitive physical-energy estimator; not the conserved ring-polymer Hamiltonian H_P/P and not the anharmonic levels above",
            "ratio_denominator": "harmonic zero-point energy omega/2",
            "finite_P_formula": "coth(P*asinh(beta*omega/(2*P)))/sqrt(1+(beta*omega/(2*P))**2)",
            "continuum_formula": "coth(beta*omega/2)",
            "finite_P_cross_checked_against_mode_sum": True,
            "rows": harmonic_ring_energies(omega, "BKMP2 curvature") + harmonic_ring_energies(paper_omega, "4407 cm^-1 harmonic model"),
        },
        "provenance": {
            "potential_source": SOURCE, "potential_build": potential.provenance,
            "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "numpy": np.__version__, "scipy": scipy.__version__, "python": sys.version,
            "platform": platform.platform(), "lapack_driver": "stebz", "eigensolver_absolute_tolerance_hartree": 1e-13,
        },
        "array_conventions": {
            "wavefunction_grid": "finest base-box grid; energies of its eigenvectors are grid_energies_hartree",
            "u_radial_shape": "[v, radial_grid_point]",
            "reference_energies_hartree": "Richardson-extrapolated energies relative to the well minimum, not the finite-grid eigenvalues",
        },
    }
    arrays.update(reference_energies_hartree=extrapolated, vibrational_quantum_numbers=np.arange(levels),
                  reduced_mass_atomic=np.array(reduced_mass), minimum_absolute_hartree=np.array(minimum),
                  harmonic_omega_au=np.array(omega))
    return report, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output-prefix", type=Path, default=ROOT/"runs/state_selected_reproduction/h2_reference")
    parser.add_argument("--reduced-mass", type=float, default=REDUCED_MASS)
    parser.add_argument("--levels", type=int, default=3)
    args = parser.parse_args()
    report, arrays = calculate(args.library, reduced_mass=args.reduced_mass, levels=args.levels)
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    with prefix.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["levels"][0]))
        writer.writeheader()
        writer.writerows(report["levels"])
    np.savez_compressed(prefix.with_suffix(".npz"), **arrays)
    print(json.dumps({"passed": report["passed"], "levels": report["levels"],
                      "convergence": {key: value for key, value in report["convergence"].items()
                                      if key not in ("grids", "domain_check")},
                      "output_prefix": str(prefix)}, indent=2))
    if not report["passed"]:
        raise SystemExit("H2 numerical-reference convergence gate failed; inspect saved diagnostics")


if __name__ == "__main__":
    main()
