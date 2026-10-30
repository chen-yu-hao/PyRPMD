#!/usr/bin/env python3
"""Compare the local PCET model against an unchanged official pyPCET checkout.

This is a reproducible validation harness, not a runtime pyPCET dependency.
It imports the external implementation; no external implementation is vendored.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Avoid creating cache files in the external checkout and oversubscribing its
# small dense FGH eigensolves.  Explicit caller settings retain precedence.
sys.dont_write_bytecode = True
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import scipy
from scipy.integrate import quad_vec, simpson

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from pyrpmd.constants import (  # noqa: E402
    ANGSTROM_TO_AU,
    ATOMIC_TIME_S,
    EV_TO_AU,
    KB_AU,
)
from pyrpmd.pcet import (  # noqa: E402
    ISOTOPE_MASSES_AU,
    golden_rule_rates,
    solve_proton,
)


REFERENCE_SHA = "031a679eab50e210aa7f76698ef5f302de31a460"
POTENTIALS = (-0.2, 0.0, 0.2)
TEMPERATURE = 300.0
LAMBDA_EV = 0.7
COUPLING_EV = 0.001
DENSITY_PER_EV = 1.0
DOMAIN_ANGSTROM = (-2.0, 2.0)


def git(checkout, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(checkout), *arguments], text=True
    ).strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def oxidized_potential(position_angstrom):
    return 0.5 * 4.0 * (np.asarray(position_angstrom) + 0.2) ** 2


def reduced_potential(position_angstrom):
    return 0.5 * 5.0 * (np.asarray(position_angstrom) - 0.2) ** 2


def summarize_cases(cases):
    result = []
    for potential in POTENTIALS:
        h = next(case for case in cases if case["isotope"] == "H" and case["potential_minus_reference_V"] == potential)
        d = next(case for case in cases if case["isotope"] == "D" and case["potential_minus_reference_V"] == potential)
        result.append({
            "potential_minus_reference_V": potential,
            "reduction_KIE_H_over_D": h["reduction_rate_s"] / d["reduction_rate_s"],
            "oxidation_KIE_H_over_D": h["oxidation_rate_s"] / d["oxidation_rate_s"],
        })
    return result


def run_pyrpmd(points, states):
    grid_angstrom = np.linspace(*DOMAIN_ANGSTROM, points)
    grid = grid_angstrom * ANGSTROM_TO_AU
    cases = []
    spectra_metadata = {}
    for isotope in ("H", "D"):
        mass = ISOTOPE_MASSES_AU[isotope]
        oxidized = solve_proton(grid, oxidized_potential(grid_angstrom) * EV_TO_AU, mass, states, energy_origin_au=0.0)
        reduced = solve_proton(grid, reduced_potential(grid_angstrom) * EV_TO_AU, mass, states, energy_origin_au=0.0)
        spectra_metadata[isotope] = {
            "mass_electron_masses": mass,
            "oxidized_ground_energy_eV": float(oxidized.energies_au[0] / EV_TO_AU),
            "reduced_ground_energy_eV": float(reduced.energies_au[0] / EV_TO_AU),
            "max_retained_boundary_probability": float(max(np.max(oxidized.boundary_probabilities), np.max(reduced.boundary_probabilities))),
        }
        for potential in POTENTIALS:
            rates = golden_rule_rates(
                oxidized, reduced,
                temperature_K=TEMPERATURE,
                driving_force_au=potential * EV_TO_AU,
                reorganization_au=LAMBDA_EV * EV_TO_AU,
                coupling_au=COUPLING_EV * EV_TO_AU,
                density_per_hartree=DENSITY_PER_EV / EV_TO_AU,
                quadrature_rtol=1e-11,
                quadrature_atol=1e-13,
            )
            cases.append({
                "isotope": isotope,
                "potential_minus_reference_V": potential,
                "reduction_rate_s": rates["reduction_rate_s"],
                "oxidation_rate_s": rates["oxidation_rate_s"],
                "detailed_balance_log_residual": rates["detailed_balance_log_residual"],
            })
    return {
        "grid_points": points,
        "retained_states_per_surface": states,
        "spectra": spectra_metadata,
        "cases": cases,
        "isotope_effects": summarize_cases(cases),
    }


def run_reference(reference_module, electrochemistry, points, states):
    cases = []
    spectra_metadata = {}
    for isotope in ("H", "D"):
        mass = ISOTOPE_MASSES_AU[isotope]
        arguments = dict(
            DeltaG=0.0, Lambda=LAMBDA_EV, Vel=COUPLING_EV,
            NStates=states, NGridPot=points,
            rmin=DOMAIN_ANGSTROM[0], rmax=DOMAIN_ANGSTROM[1],
        )
        forward = reference_module.pyPCET(oxidized_potential, reduced_potential, **arguments)
        reverse = reference_module.pyPCET(reduced_potential, oxidized_potential, **arguments)
        for model in (forward, reverse):
            model.calc_proton_vibrational_states(mass=mass)
            model.calc_proton_overlap_matrix()
        forward_zpe_difference = float(forward.ProdProtonEnergyLevels[0] - forward.ReacProtonEnergyLevels[0])
        reverse_zpe_difference = float(reverse.ProdProtonEnergyLevels[0] - reverse.ReacProtonEnergyLevels[0])
        spectra_metadata[isotope] = {
            "mass_electron_masses": mass,
            "oxidized_ground_energy_eV": float(forward.ReacProtonEnergyLevels[0]),
            "reduced_ground_energy_eV": float(forward.ProdProtonEnergyLevels[0]),
            "forward_DeltaG_ground_state_correction_eV": forward_zpe_difference,
            "reverse_DeltaG_ground_state_correction_eV": reverse_zpe_difference,
        }
        for potential in POTENTIALS:
            def integrand(electrode_energy):
                # pyPCET builds dG_uv from level energies relative to each
                # ground state.  Therefore its DeltaG must include DeltaZPE.
                forward.set_parameters(DeltaG=potential + forward_zpe_difference - electrode_energy)
                reverse.set_parameters(DeltaG=-potential + reverse_zpe_difference + electrode_energy)
                reduction = forward.calculate(mass=mass, T=TEMPERATURE, reuse_saved_proton_states=True)
                oxidation = reverse.calculate(mass=mass, T=TEMPERATURE, reuse_saved_proton_states=True)
                filled = electrochemistry.Fermi_distribution(electrode_energy, T=TEMPERATURE)
                # f(-eps) == 1-f(eps), without subtracting nearly equal values.
                empty = electrochemistry.Fermi_distribution(-electrode_energy, T=TEMPERATURE)
                return DENSITY_PER_EV * np.asarray([filled * reduction, empty * oxidation])

            integrated, error, info = quad_vec(
                integrand, -4.0, 4.0,
                epsabs=1e-7, epsrel=1e-10, points=[-1.0, 0.0, 1.0],
                norm="max", full_output=True,
            )
            if not info.success or not np.isfinite(integrated).all() or np.any(integrated <= 0):
                raise RuntimeError("The external reference rate integral did not converge")
            # Separately verify the chosen energy range for the final reference.
            tail = np.zeros(2)
            if points == 512 and states == 24:
                for left, right in ((-6.0, -4.0), (4.0, 6.0)):
                    extra, _ = quad_vec(integrand, left, right, epsabs=1e-10, epsrel=1e-10)
                    tail += extra
            cases.append({
                "isotope": isotope,
                "potential_minus_reference_V": potential,
                "reduction_rate_s": float(integrated[0]),
                "oxidation_rate_s": float(integrated[1]),
                "electrode_energy_bound_eV": 4.0,
                "quadrature_absolute_error_estimate_s": float(error),
                "quadrature_evaluations": int(info.neval),
                "extra_4_to_6_eV_tail_fraction": [float(value) for value in tail / integrated],
                "tail_check_performed": points == 512 and states == 24,
            })
    return {
        "grid_points": points,
        "retained_states_per_surface": states,
        "spectra": spectra_metadata,
        "cases": cases,
        "isotope_effects": summarize_cases(cases),
    }


def compare(left, right, *, left_name, right_name):
    rate_differences = []
    kie_differences = []
    for left_case, right_case in zip(left["cases"], right["cases"], strict=True):
        assert (left_case["isotope"], left_case["potential_minus_reference_V"]) == (right_case["isotope"], right_case["potential_minus_reference_V"])
        for direction in ("reduction", "oxidation"):
            key = f"{direction}_rate_s"
            rate_differences.append({
                "isotope": left_case["isotope"],
                "potential_minus_reference_V": left_case["potential_minus_reference_V"],
                "direction": direction,
                "relative_difference": left_case[key] / right_case[key] - 1.0,
            })
    for left_case, right_case in zip(left["isotope_effects"], right["isotope_effects"], strict=True):
        for direction in ("reduction", "oxidation"):
            key = f"{direction}_KIE_H_over_D"
            kie_differences.append({
                "potential_minus_reference_V": left_case["potential_minus_reference_V"],
                "direction": direction,
                "relative_difference": left_case[key] / right_case[key] - 1.0,
            })
    return {
        "left": left_name,
        "right": right_name,
        "relative_difference_definition": "left / right - 1",
        "max_absolute_relative_rate_difference": max(abs(row["relative_difference"]) for row in rate_differences),
        "max_absolute_relative_KIE_difference": max(abs(row["relative_difference"]) for row in kie_differences),
        "rate_differences": rate_differences,
        "KIE_differences": kie_differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("reference_results.json"))
    args = parser.parse_args()
    reference = args.reference_checkout.resolve()
    if git(reference, "rev-parse", "HEAD") != REFERENCE_SHA:
        raise ValueError(f"Reference checkout must be pinned to {REFERENCE_SHA}")
    if git(reference, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Reference tracked files must be unchanged")
    core_files = ("src/pyrpmd/pcet.py", "src/pyrpmd/constants.py")
    reference_files = ("pyPCET/pyPCET.py", "pyPCET/FGH_1D.py", "pyPCET/units.py", "pyPCET/electrochemistry.py")
    core_hashes = {name: digest(ROOT / name) for name in core_files}
    reference_hashes = {name: digest(reference / name) for name in reference_files}
    sys.path.insert(0, str(reference))
    reference_module = importlib.import_module("pyPCET.pyPCET")
    reference_units = importlib.import_module("pyPCET.units")
    electrochemistry = importlib.import_module("pyPCET.electrochemistry")
    import numba

    # The original module falls back to scipy.integrate.simpson as simps,
    # but calls it with positional x.  New SciPy requires x as a keyword.
    # This shim changes only the calling syntax; no formula or constant changes.
    reference_module.simps = lambda y, x: simpson(y, x=x)

    started = time.monotonic()
    own_runs, reference_runs = {}, {}
    for points, states in ((801, 16), (1601, 16), (1601, 24)):
        name = f"fd_{points}_states_{states}"
        own_runs[name] = run_pyrpmd(points, states)
        print(f"Completed PyRPMD {name}", flush=True)
    for points, states in ((256, 16), (512, 16), (512, 24)):
        name = f"fgh_{points}_states_{states}"
        reference_runs[name] = run_reference(reference_module, electrochemistry, points, states)
        print(f"Completed external pyPCET {name}", flush=True)
    comparisons = {
        "independent_implementation": compare(own_runs["fd_1601_states_24"], reference_runs["fgh_512_states_24"], left_name="PyRPMD finite differences 1601/24", right_name="official pyPCET FGH 512/24"),
        "pyrpmd_grid_refinement": compare(own_runs["fd_1601_states_16"], own_runs["fd_801_states_16"], left_name="FD 1601/16", right_name="FD 801/16"),
        "pyrpmd_state_refinement": compare(own_runs["fd_1601_states_24"], own_runs["fd_1601_states_16"], left_name="FD 1601/24", right_name="FD 1601/16"),
        "reference_grid_refinement": compare(reference_runs["fgh_512_states_16"], reference_runs["fgh_256_states_16"], left_name="FGH 512/16", right_name="FGH 256/16"),
        "reference_state_refinement": compare(reference_runs["fgh_512_states_24"], reference_runs["fgh_512_states_16"], left_name="FGH 512/24", right_name="FGH 512/16"),
    }
    if core_hashes != {name: digest(ROOT / name) for name in core_files}:
        raise RuntimeError("PyRPMD sources changed during the comparison; rerun against stable sources")
    if reference_hashes != {name: digest(reference / name) for name in reference_files}:
        raise RuntimeError("External reference sources changed during the comparison")
    result = {
        "scope": "Fixed-geometry wide-band local golden-rule PCET teaching model; not reproduction of a molecular or atomistic electrode benchmark",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "reference": {
            "repository": "https://github.com/shsgroup/pyPCET",
            "commit": REFERENCE_SHA,
            "tracked_files_unchanged": not bool(git(reference, "status", "--porcelain", "--untracked-files=no")),
            "source_sha256": reference_hashes,
            "sources_unchanged_during_run": True,
            "compatibility_shim": "pyPCET.pyPCET.simps = lambda y, x: scipy.integrate.simpson(y, x=x); call syntax only",
            "mass_override": "Explicit PyRPMD neutral atomic isotope mass supplied through pyPCET's public mass argument; default proton masses not used",
            "driving_force_mapping": "Forward DeltaG = U-U_ref + (E_R0-E_O0) - epsilon; reverse uses swapped surfaces and DeltaG = -(U-U_ref) + (E_O0-E_R0) + epsilon. All values eV; pyPCET itself uses excitation energies relative to each ground state.",
        },
        "pyrpmd": {
            "base_commit": git(ROOT, "rev-parse", "HEAD"),
            "working_tree_dirty": bool(git(ROOT, "status", "--porcelain")),
            "source_sha256": core_hashes,
            "sources_unchanged_during_run": True,
            "harness_sha256": digest(__file__),
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "numba": numba.__version__},
        "parameters": {
            "oxidized_force_constant_eV_per_A2": 4.0,
            "reduced_force_constant_eV_per_A2": 5.0,
            "oxidized_center_A": -0.2,
            "reduced_center_A": 0.2,
            "temperature_K": TEMPERATURE,
            "reorganization_eV": LAMBDA_EV,
            "electronic_coupling_eV": COUPLING_EV,
            "electrode_DOS_per_eV": DENSITY_PER_EV,
            "well_minimum_driving_force_at_reference_eV": 0.0,
            "potentials_minus_reference_V": POTENTIALS,
            "proton_domain_A": DOMAIN_ANGSTROM,
        },
        "constants": {
            "pyrpmd": {"kB_eV_per_K": KB_AU / EV_TO_AU, "hbar_eV_s": ATOMIC_TIME_S / EV_TO_AU, "hartree_eV": 1.0 / EV_TO_AU, "bohr_angstrom": 1.0 / ANGSTROM_TO_AU},
            "reference_unmodified": {"kB_eV_per_K": reference_units.kB, "hbar_eV_s": reference_units.hbar, "hartree_eV": reference_units.Ha2eV, "bohr_angstrom": reference_units.Bohr2A},
            "note": "The official reference uses rounded constants; these are deliberately retained, so raw cross-code differences include that convention difference.",
        },
        "pyrpmd_runs": own_runs,
        "reference_runs": reference_runs,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    comparison = comparisons["independent_implementation"]
    print(json.dumps({key: comparison[key] for key in ("max_absolute_relative_rate_difference", "max_absolute_relative_KIE_difference")}, indent=2))
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
