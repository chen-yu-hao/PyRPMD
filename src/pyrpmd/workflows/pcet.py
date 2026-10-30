"""System-free electrochemical PCET scans with explicit local normalization.

The workflow combines a quantum one-coordinate proton spectrum with the
nonadiabatic golden-rule electrode model. It does not invoke the bimolecular
RPMD rate prefactor or interpret an electrode potential as a thermostat.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from ..constants import EV_TO_AU
from ..io import fingerprint, jsonable, write_json
from ..pcet import ISOTOPE_MASSES_AU, golden_rule_rates, solve_proton
from ..pcet_input import PCETModelIdentity, sample_proton_potential


_RATE_FIELDS = ("reduction_rate_s", "oxidation_rate_s")


def _validate(config, model):
    if config.get("task") != "pcet" or not isinstance(model, PCETModelIdentity):
        raise ValueError("PCET requires a resolved PCET input and PCETModelIdentity")
    pcet = config["pcet"]
    if pcet.get("method") != "golden_rule" or pcet.get("normalization") != "local_first_order":
        raise ValueError("Unsupported PCET method or rate normalization")
    if pcet["electrode"].get("model") != "wide_band":
        raise ValueError("PCET currently supports only a wide-band electrode")
    parallel = config.get("parallel", {})
    if (parallel.get("backend", "serial") != "serial" or parallel.get("mode", "serial") != "serial"
            or parallel.get("trajectory_workers", 1) != 1 or parallel.get("trajectory_groups", 1) != 1):
        raise ValueError("This PCET model workflow supports serial execution only")
    proton = pcet["proton"]
    x = model.coordinate_bohr
    if (len(x) != proton["grid_points"] or x[0] != proton["grid_min"]
            or x[-1] != proton["grid_max"]):
        raise ValueError("Resolved PCET proton grid disagrees with its model identity")
    if not 1 <= proton["states"] <= len(x) - 2:
        raise ValueError("PCET proton state count exceeds the interior grid size")


def _solve_pair(model, isotope, states):
    mass = ISOTOPE_MASSES_AU[isotope]
    return (
        solve_proton(model.coordinate_bohr, model.oxidized_hartree, mass, states,
                     energy_origin_au=model.oxidized_energy_origin_au),
        solve_proton(model.coordinate_bohr, model.reduced_hartree, mass, states,
                     energy_origin_au=model.reduced_energy_origin_au),
    )


def _spectrum_summary(spectrum):
    return {
        "mass_au": float(spectrum.mass_au),
        "energies_au": spectrum.energies_au,
        "energy_origin_au": float(spectrum.energy_origin_au),
        "boundary_probabilities": spectrum.boundary_probabilities,
        "grid_points": len(spectrum.grid_au),
        "states": len(spectrum.energies_au),
        "wavefunction_normalization": "discrete_sum_of_squared_column_equals_one",
    }


def prepare_pcet(config, model, *, numerical=False):
    """Describe the local model; optionally diagonalize its proton Hamiltonians."""
    _validate(config, model)
    pcet = config["pcet"]
    result = {
        "method": "golden_rule", "normalization": "local_first_order", "rate_unit": "s^-1",
        "model": "one_local_complex_one_proton_wide_band_electrode",
        "isotopes": pcet["isotopes"], "temperatures_K": config["temperature_K"],
        "potentials_V": pcet["electrode"]["potentials_V"],
        "potential_reference": pcet["electrode"]["potential_reference"],
        "scan_points": len(config["temperature_K"]) * len(pcet["electrode"]["potentials_V"]) * len(pcet["isotopes"]),
        "proton_grid_points": pcet["proton"]["grid_points"],
        "proton_states": pcet["proton"]["states"],
        "physical_applicability_unverified": True,
    }
    if numerical:
        result["proton_spectra"] = {}
        for isotope in pcet["isotopes"]:
            oxidized, reduced = _solve_pair(model, isotope, pcet["proton"]["states"])
            result["proton_spectra"][isotope] = {
                "oxidized": _spectrum_summary(oxidized), "reduced": _spectrum_summary(reduced),
            }
    return result


def _refined_grid(config, model):
    proton = config["pcet"]["proton"]
    coordinate = np.linspace(model.coordinate_bohr[0], model.coordinate_bohr[-1], 2 * len(model.coordinate_bohr) - 1)
    _, oxidized, ox_origin = sample_proton_potential(proton["oxidized"], coordinate, name="pcet.proton.oxidized")
    _, reduced, red_origin = sample_proton_potential(proton["reduced"], coordinate, name="pcet.proton.reduced")
    if ox_origin != model.oxidized_energy_origin_au or red_origin != model.reduced_energy_origin_au:
        raise ValueError("PCET potential energy reference changed during grid refinement")
    return PCETModelIdentity(coordinate, oxidized, reduced, ox_origin, red_origin)


def _rates(pair, pcet, temperature, driving):
    numerics = pcet["numerics"]
    return golden_rule_rates(
        *pair, temperature_K=temperature, driving_force_au=driving,
        reorganization_au=pcet["reorganization_au"], coupling_au=pcet["coupling_au"],
        density_per_hartree=pcet["electrode"]["density_per_hartree"],
        quadrature_rtol=numerics["quadrature_rtol"], quadrature_atol=numerics["quadrature_atol"],
    )


def _compare_rates(reference, refined, numerics):
    comparisons = {}
    for field in _RATE_FIELDS:
        first, second = float(reference[field]), float(refined[field])
        difference = abs(first - second)
        log_field = "log_" + field
        first_log, second_log = reference.get(log_field), refined.get(log_field)
        log_difference = None
        if first_log is not None and second_log is not None:
            if not math.isfinite(first_log) or not math.isfinite(second_log):
                raise ValueError("PCET convergence comparison requires finite log rates")
            delta = float(second_log) - float(first_log)
            # The relative error uses the larger rate as its denominator.
            # Logs preserve it even when both ordinary rates underflow, and
            # scaling the electronic coupling must not change convergence.
            relative = -math.expm1(-abs(delta))
            log_difference = delta if math.isfinite(delta) else None
            passed = relative <= numerics["relative_tolerance"]
            criterion = "relative_difference_from_log_rates"
        elif first_log is None and second_log is None:
            if first != 0.0 or second != 0.0:
                raise ValueError("PCET nonzero rates must retain their finite logarithms")
            relative = 0.0
            passed = difference <= numerics["absolute_tolerance_s"]
            criterion = "absolute_difference_for_exact_zero_rates"
        else:
            # An exactly zero channel sum in only one basis cannot establish
            # convergence to a positive physical rate, however small it is.
            relative = 1.0
            passed = False
            criterion = "exact_zero_and_positive_rate_disagree"
        comparisons[field] = {
            "reference": first, "refined": second, "absolute_difference_s": difference,
            "relative_difference": relative,
            "log_rate_difference": log_difference, "criterion": criterion,
            "passed": bool(passed),
        }
    return {"passed": all(value["passed"] for value in comparisons.values()), "rates": comparisons}


def _convergence(base, grid_result, state_result, pair, pcet):
    numerics = pcet["numerics"]
    population = {}
    for name, spectrum in zip(("oxidized", "reduced"), pair):
        p = np.asarray(base[name + "_population"])
        boundary = np.asarray(spectrum.boundary_probabilities)
        population[name] = {
            "highest_retained_state_population": float(p[-1]),
            "thermal_boundary_probability": float(np.dot(p, boundary)),
        }
    reactive = {}
    for direction in ("reduction", "oxidation"):
        fractions = np.asarray(base[direction + "_channel_fractions"])
        # Both matrices have oxidized-state rows and reduced-state columns.
        # Weight both surfaces by the actual channel flux, including excited
        # final proton states that have negligible equilibrium population.
        reactive[direction] = {}
        for name, spectrum, weights in (
            ("oxidized", pair[0], fractions.sum(axis=1)),
            ("reduced", pair[1], fractions.sum(axis=0)),
        ):
            reactive[direction][name] = {
                "rate_weighted_boundary_probability": float(weights @ spectrum.boundary_probabilities),
                "highest_retained_state_rate_fraction": float(weights[-1]),
            }
    if not numerics["convergence_check"]:
        return {
            "checked": False, "passed": False, "proton_population_diagnostics": population,
            "reactive_proton_diagnostics": reactive,
            "unchecked": ["grid_spacing", "proton_state_truncation", "proton_grid_extent"],
        }
    grid = _compare_rates(base, grid_result, numerics)
    states = _compare_rates(base, state_result, numerics) if state_result is not None else {
        "passed": False, "reason": "all_interior_grid_states_already_used_cannot_extend_basis",
    }
    # This is a diagnostic of the supplied finite coordinate interval, not a
    # substitute for an independent calculation on a wider physical interval.
    edge_tolerance = min(1e-3, numerics["relative_tolerance"])
    populations_passed = all(
        row["highest_retained_state_population"] <= edge_tolerance
        and row["thermal_boundary_probability"] <= edge_tolerance
        for row in population.values()
    )
    reactive_passed = all(
        row["rate_weighted_boundary_probability"] <= edge_tolerance
        and row["highest_retained_state_rate_fraction"] <= edge_tolerance
        for direction in reactive.values() for row in direction.values()
    )
    return {
        "checked": True,
        "passed": bool(grid["passed"] and states["passed"] and populations_passed and reactive_passed),
        "grid_spacing": grid, "proton_state_truncation": states,
        "proton_population_diagnostics": population,
        "proton_population_tolerance": edge_tolerance,
        "proton_population_checks_passed": populations_passed,
        "reactive_proton_diagnostics": reactive,
        "reactive_proton_checks_passed": reactive_passed,
        "unchecked": ["independent_proton_grid_extent_scan", "physical_model_applicability"],
    }


def _checksum(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _load_case(path, physics, ident):
    if not path.is_file():
        return None
    saved = json.loads(path.read_text())
    if (saved.get("schema") != "pyrpmd.pcet-case" or type(saved.get("schema_version")) is not int
            or saved["schema_version"] != 1 or saved.get("physics_fingerprint") != physics
            or saved.get("case_id") != ident or "result" not in saved
            or saved.get("sha256") != _checksum(saved["result"])):
        raise ValueError(f"PCET saved scan-point identity or integrity failed: {path}")
    return saved["result"]


def _save_case(path, physics, ident, result):
    write_json(path, {
        "schema": "pyrpmd.pcet-case", "schema_version": 1,
        "physics_fingerprint": physics, "case_id": ident,
        "result": result, "sha256": _checksum(result),
    })


def _atomic_arrays(path, arrays):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_rates(path, rows):
    fields = [
        "temperature_K", "potential_V", "potential_reference", "isotope", "driving_force_au",
        "reduction_rate_s", "oxidation_rate_s", "log_reduction_rate_s", "log_oxidation_rate_s",
        "numerical_convergence",
    ]
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _isotope_effects(rows):
    by_condition = {}
    for row in rows:
        by_condition.setdefault((row["temperature_K"], row["potential_V"]), {})[row["isotope"]] = row
    result = []
    for (temperature, potential), isotopes in by_condition.items():
        if "H" not in isotopes or "D" not in isotopes:
            continue
        entry = {"temperature_K": temperature, "potential_V": potential, "definition": "k_H/k_D"}
        for direction in ("reduction", "oxidation"):
            hlog, dlog = [isotopes[iso][f"log_{direction}_rate_s"] for iso in ("H", "D")]
            log_ratio = hlog - dlog if hlog is not None and dlog is not None else None
            ratio = None
            if log_ratio is not None:
                if math.log(np.nextafter(0.0, 1.0)) <= log_ratio <= math.log(np.finfo(float).max):
                    ratio = math.exp(log_ratio)
            entry[direction + "_KIE"] = ratio
            entry["log_" + direction + "_KIE"] = log_ratio
        result.append(entry)
    return result


def _run(config, model, output):
    _validate(config, model)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pcet = config["pcet"]
    numerics = pcet["numerics"]
    states = pcet["proton"]["states"]
    physics = fingerprint(config, model)
    pairs, refined_pairs, extended_pairs = {}, {}, {}
    summaries, arrays = {}, {}
    refined_model = _refined_grid(config, model) if numerics["convergence_check"] else None
    extended_states = min(2 * states, len(model.coordinate_bohr) - 2)
    for isotope in pcet["isotopes"]:
        pairs[isotope] = _solve_pair(model, isotope, states)
        summaries[isotope] = {}
        for name, spectrum in zip(("oxidized", "reduced"), pairs[isotope]):
            summaries[isotope][name] = _spectrum_summary(spectrum)
            prefix = isotope + "_" + name + "_"
            for field in ("grid_au", "potential_au", "energies_au", "wavefunctions", "boundary_probabilities"):
                arrays[prefix + field] = getattr(spectrum, field)
        if numerics["convergence_check"]:
            refined_pairs[isotope] = _solve_pair(refined_model, isotope, states)
            if extended_states > states:
                extended_pairs[isotope] = _solve_pair(model, isotope, extended_states)
    _atomic_arrays(output / "proton_wavefunctions.npz", arrays)
    write_json(output / "proton_spectra.json", {
        "schema": "pyrpmd.pcet-spectra", "schema_version": 1,
        "spectra": summaries, "arrays_file": "proton_wavefunctions.npz",
        "energy_reference": "each_diabatic_proton_potential_minimum_excluding_proton_zero_point_energy",
    })
    details, rows = [], []
    electrode = pcet["electrode"]
    for t_index, temperature in enumerate(config["temperature_K"]):
        for u_index, potential in enumerate(electrode["potentials_V"]):
            driving = pcet["driving_force_au"] + (potential - electrode["reference_potential_V"]) * EV_TO_AU
            for isotope in pcet["isotopes"]:
                ident = f"T{t_index:04d}_U{u_index:04d}_{isotope}"
                checkpoint = output / "checkpoints" / ("pcet-" + ident + ".json")
                result = _load_case(checkpoint, physics, ident) if config.get("resume") else None
                if result is None:
                    rates = _rates(pairs[isotope], pcet, temperature, driving)
                    grid_rates = _rates(refined_pairs[isotope], pcet, temperature, driving) if numerics["convergence_check"] else None
                    state_rates = _rates(extended_pairs[isotope], pcet, temperature, driving) if isotope in extended_pairs else None
                    convergence = _convergence(rates, grid_rates, state_rates, pairs[isotope], pcet)
                    result = {
                        "case_id": ident, "temperature_K": temperature, "potential_V": potential,
                        "potential_reference": electrode["potential_reference"], "isotope": isotope,
                        "driving_force_au": driving, "rates": rates, "numerical_checks": convergence,
                        "numerical_convergence": convergence["passed"],
                    }
                    _save_case(checkpoint, physics, ident, result)
                details.append(result)
                rows.append({
                    **{key: result[key] for key in ("case_id", "temperature_K", "potential_V", "potential_reference", "isotope", "driving_force_au", "numerical_convergence")},
                    **{key: result["rates"][key] for key in (*_RATE_FIELDS, "log_reduction_rate_s", "log_oxidation_rate_s")},
                })
    _write_rates(output / "rates.csv", rows)
    write_json(output / "pcet_details.json", {
        "schema": "pyrpmd.pcet-details", "schema_version": 1,
        "physics_fingerprint": physics, "cases": details,
    })
    numerical_convergence = bool(rows and all(row["numerical_convergence"] for row in rows))
    summary = {
        "schema": "pyrpmd.pcet-summary", "schema_version": 1, "task": "pcet",
        "status": "model_rates_computed", "complete": True,
        "method": "golden_rule", "normalization": "local_first_order", "rate_unit": "s^-1",
        "model": "one_local_complex_one_proton_wide_band_electrode",
        "temperature_K": config["temperature_K"], "potentials_V": electrode["potentials_V"],
        "potential_reference": electrode["potential_reference"],
        "reference_potential_V": electrode["reference_potential_V"],
        "isotopes": pcet["isotopes"], "completed_scan_points": len(rows),
        "numerical_convergence": numerical_convergence,
        "physical_applicability_unverified": True,
        "convergence_note": "Numerical checks compare grid spacing and proton-state truncation for this supplied one-dimensional model. They do not validate weak coupling, the classical environment, the chosen coordinate interval, or a real electrode material.",
        "rates": rows, "kinetic_isotope_effects": _isotope_effects(rows),
        "files": {
            "rates": str(output / "rates.csv"), "details": str(output / "pcet_details.json"),
            "spectra": str(output / "proton_spectra.json"), "wavefunctions": str(output / "proton_wavefunctions.npz"),
        },
    }
    write_json(output / "summary.json", summary)
    return summary


def run_pcet(config, model, output):
    """Evaluate a deterministic PCET scan on the coordinating process."""
    from ..parallel import root_call

    return root_call(_run, config, model, output)
