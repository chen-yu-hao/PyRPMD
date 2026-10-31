"""Reproducible temperature, isotope and numerical-convergence rate studies.

Each case runs the existing checked rate workflow with an independent random
stream. A study never promotes a withheld pilot rate into a physical rate.
The convergence test is an equivalence test on the *tested* refinements;
failure to distinguish two noisy estimates is not evidence of convergence.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import tomllib
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from .constants import KB_AU, KCAL_MOL_TO_AU
from .input import InputError, JobSpec
from .io import fingerprint, jsonable, referenced_files, write_json
from .system import System
from .units import parse_quantity, time_steps
from .validation import checked_integer

_AXES = {"beads": None, "timestep": "time", "separation": "length", "bmax": "length"}
_RATE_TASKS = ("rate", "site_rate", "surface_rate", "drpmd")


def _positive_array(values, name):
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real")
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not a.size or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError(f"{name} must be a nonempty vector of positive finite values")
    return a


def arrhenius_fit(temperatures, rates, *, standard_errors=None, rate_unit="s^-1"):
    """Fit ln(k)=ln(A)-Ea/(kB*T), with errors in ln(k) when supplied.

The input rate unit is retained by A; Ea is reported per molecule in hartree
and per mole in kJ/mol. Supplied errors are absolute standard errors and the
log transformation uses the small-error delta approximation. This model does
not assume that low-temperature tunnelling rates actually obey Arrhenius law.
"""
    t = _positive_array(temperatures, "temperatures")
    k = _positive_array(rates, "rates")
    if t.shape != k.shape or len(t) < 2 or len(np.unique(t)) != len(t):
        raise ValueError("Arrhenius fit requires matching rates at at least two distinct temperatures")
    if not isinstance(rate_unit, str) or not rate_unit.strip():
        raise ValueError("rate_unit must be a nonempty string")
    y, x = np.log(k), 1 / t
    center, scale = float(x.mean()), float(np.ptp(x))
    design = np.column_stack((np.ones(len(t)), (x - center) / scale))
    if standard_errors is None:
        sigma = np.ones(len(t))
    else:
        errors = _positive_array(standard_errors, "standard_errors")
        if errors.shape != k.shape:
            raise ValueError("standard_errors must match rates")
        sigma = errors / k
        if not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("Relative rate errors are outside finite floating-point range")
    weighted = design / sigma[:, None]
    coefficient, _, rank, _ = np.linalg.lstsq(weighted, y / sigma, rcond=None)
    if rank != 2:
        raise ValueError("Temperature grid is numerically rank deficient")
    slope = float(coefficient[1] / scale)
    ln_a = float(coefficient[0] - slope * center)
    ea = -KB_AU * slope
    predicted = design @ coefficient
    residual = y - predicted
    covariance = np.linalg.inv(weighted.T @ weighted)
    if standard_errors is None:
        covariance = covariance * float(residual @ residual) / (len(t) - 2) if len(t) > 2 else None
    transform = np.array([[1., -center / scale], [0., -KB_AU / scale]])
    covariance = None if covariance is None else transform @ covariance @ transform.T
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "model": "ln(k) = ln(A) - Ea/(kB*T)", "rate_unit": rate_unit,
        "temperatures_K": t.tolist(), "rates": k.tolist(),
        "ln_A": ln_a, "A": math.exp(ln_a) if ln_a < math.log(np.finfo(float).max) else None,
        "activation_energy_hartree": float(ea),
        "activation_energy_kJ_mol": float(ea / KCAL_MOL_TO_AU * 4.184),
        "ln_A_standard_error": None if covariance is None else float(np.sqrt(max(0., covariance[0, 0]))),
        "activation_energy_standard_error_hartree": None if covariance is None else float(np.sqrt(max(0., covariance[1, 1]))),
        "parameter_covariance_ln_A_Ea_hartree": None if covariance is None else covariance.tolist(),
        "residuals_ln_k": residual.tolist(),
        "r_squared_ln_k": None if total == 0 else float(1 - residual @ residual / total),
        "degrees_of_freedom": len(t) - 2,
        "uncertainty_model": "independent log-rate errors (delta approximation)" if standard_errors is not None else "unweighted residual variance",
        "physical_converged": False,
        "note": "Inspect residuals and temperature range; an Arrhenius fit does not validate rates or exclude tunnelling curvature.",
    }


def kinetic_isotope_effect(numerator, denominator, *, numerator_error=None, denominator_error=None):
    """Independent-rate ratio k(numerator)/k(denominator), with log-delta CI.

The named numerator/denominator convention avoids silently swapping normal
and inverse isotope effects. Correlated simulations require a joint bootstrap
outside this independent-stream helper.
"""
    a, b = _positive_array([numerator, denominator], "isotope rates")
    if (numerator_error is None) != (denominator_error is None):
        raise ValueError("Provide both isotope rate errors, or neither")
    log_ratio = float(np.log(a) - np.log(b))
    if abs(log_ratio) > math.log(np.finfo(float).max):
        raise ValueError("Isotope rate ratio is outside finite floating-point range")
    ratio = float(np.exp(log_ratio))
    error, interval = None, None
    if numerator_error is not None:
        e = np.asarray([numerator_error, denominator_error], float)
        if not np.isfinite(e).all() or np.any(e < 0):
            raise ValueError("Rate standard errors must be nonnegative and finite")
        log_error = float(np.hypot(e[0] / a, e[1] / b))
        bounds = log_ratio + np.array([-1., 1.]) * 1.959963984540054 * log_error
        error = ratio * log_error
        if not np.isfinite(error) or np.max(np.abs(bounds)) > math.log(np.finfo(float).max):
            raise ValueError("Isotope uncertainty is outside finite floating-point range")
        interval = np.exp(bounds).tolist()
    return {"ratio": ratio, "log_ratio": log_ratio, "standard_error": error,
            "confidence_interval_95": interval, "convention": "k_numerator / k_denominator",
            "uncertainty_model": "independent rates; first-order log-delta approximation"}


_integer = partial(checked_integer, error_type=InputError)


def _quantity(value, dimension, name):
    result = parse_quantity(value, dimension)
    if result <= 0:
        raise InputError(f"{name} must be positive")
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass
class StudyCase:
    """One resolved, independently seeded calculation in a rate study."""

    case_id: str
    config: dict
    system: System
    parameters: dict
    replica: int

    def to_dict(self):
        return {"case_id": self.case_id, "parameters": self.parameters, "replica": self.replica,
                "seed": self.config["seed"], "physics_fingerprint": fingerprint(self.config, self.system),
                "masses_au": self.system.masses.tolist()}


def _study_spec(value, base_directory=None):
    base = Path(base_directory or ".").resolve()
    if isinstance(value, (str, Path)):
        path = Path(value).resolve()
        base = path.parent
        if path.suffix == ".json":
            value = json.loads(path.read_text())
        elif path.suffix in (".toml", ".inp"):
            value = tomllib.loads(path.read_text())
        else:
            raise InputError("Study input must be JSON or TOML")
    if not isinstance(value, dict):
        raise InputError("Study must be a mapping")
    allowed = {"schema_version", "job", "temperatures", "isotopologues", "convergence", "replicates",
               "seed", "design", "relative_tolerance", "absolute_tolerance", "confidence_z", "kie_reference"}
    unknown = set(value) - allowed
    if unknown:
        raise InputError("Unknown study fields: " + ", ".join(sorted(unknown)))
    if _integer(value.get("schema_version", 1), "schema_version") != 1:
        raise InputError("Study schema_version must be 1")
    if "job" not in value:
        raise InputError("Study requires a base job path or mapping")
    job = value["job"]
    if isinstance(job, JobSpec):
        spec = job
    elif isinstance(job, (str, Path)):
        spec = JobSpec.from_file(base / job)
    else:
        spec = JobSpec.from_mapping(job, base_directory=base)
    return value, spec


def _preserve_durations(config, old_dt, new_dt):
    """Hold physical lengths and observation/thermostat cadences fixed."""
    groups = {
        "sampling": ("equilibration_steps", "production_steps", "interval_steps", "sample_interval", "andersen_interval"),
        "collision": ("max_steps",), "outcomes": ("persistence_steps", "check_interval"),
        "output": ("checkpoint_steps", "trajectory_stride"),
        "recrossing": ("equilibration_steps", "parent_interval_steps", "child_steps"),
    }
    for name, keys in groups.items():
        table = config.get(name, {})
        for key in keys:
            if key in table:
                table[key] = time_steps(float(table[key]) * old_dt, new_dt, allow_zero=key == "equilibration_steps")
    sampling = config["sampling"]
    sampling["dt_au"] = sampling["timestep"] = new_dt
    if "thermostat_params" in sampling and "interval_steps" in sampling["thermostat_params"]:
        sampling["thermostat_params"]["interval_steps"] = sampling["andersen_interval"]
    config["dt_au"] = new_dt
    config.setdefault("dynamics", {})["timestep"] = new_dt


def plan_rate_study(study, *, base_directory=None):
    """Resolve all cases without running a potential or writing output.

Convergence axes: beads, timestep, separation (Rinf for rate, R0 for drpmd),
and bmax (drpmd only). Default design varies one axis around the finest
settings of the others. ``design='factorial'`` tests interactions explicitly.
Numerical quantities follow PyRPMD atomic units; strings accept explicit units.
"""
    raw, spec = _study_spec(study, base_directory)
    original, system = spec.resolve()
    task = original["task"]
    if task not in _RATE_TASKS:
        raise InputError("Rate studies require rate, site_rate, surface_rate or drpmd")
    if task == "drpmd" and original["collision"].get("sampling") == "fixed_energy":
        raise InputError("Fixed-energy cross sections are not thermal rates; Arrhenius/KIE studies require thermal collision sampling")
    if original["parallel"].get("backend") == "mpi":
        raise InputError("The study coordinator is serial; use process trajectory workers within each case")
    if original.get("adaptive_umbrella"):
        raise InputError("Controlled rate studies require fixed umbrella windows; adaptive windows change the numerical comparison")
    if task != "drpmd":
        from .workflows.rate import _recrossing_options
        recrossing_raw = deepcopy(original.get("recrossing", original.get("rate", {}).get("recrossing", {})))
        options = _recrossing_options(original)
        original.setdefault("rate", {}).pop("recrossing", None)
        if task in ("site_rate", "surface_rate"):
            original.pop("rate", None)
        original["recrossing"] = {**recrossing_raw, **options}
    temps = [_quantity(v, "temperature", "temperature") for v in raw.get("temperatures", [original["temperature_K"]])]
    if not temps or len(set(temps)) != len(temps):
        raise InputError("temperatures must be nonempty and distinct")
    isotopes = raw.get("isotopologues", {"reference": []})
    if not isinstance(isotopes, dict) or not isotopes:
        raise InputError("isotopologues must map unique names to lists of atom/mass substitutions")
    isotope_masses = {}
    for label, changes in isotopes.items():
        if not isinstance(label, str) or not label.strip() or not isinstance(changes, (list, tuple)):
            raise InputError("Each isotopologue needs a nonempty name and an atom/mass list")
        masses, seen = system.masses.copy(), set()
        for change in changes:
            if not isinstance(change, dict) or set(change) != {"atom", "mass"}:
                raise InputError("Isotope substitutions require exactly atom and mass")
            atom = _integer(change["atom"], "isotope atom", 0)
            if atom >= system.natoms or atom in seen:
                raise InputError("Isotope atom must be in range and not repeated")
            seen.add(atom)
            masses[atom] = _quantity(change["mass"], "mass", "isotope mass")
        isotope_masses[label] = masses
    kie_reference = raw.get("kie_reference", next(iter(isotopes)))
    if kie_reference not in isotopes:
        raise InputError("kie_reference must name an isotopologue")
    axes_raw = raw.get("convergence", {})
    if not isinstance(axes_raw, dict) or set(axes_raw) - set(_AXES):
        raise InputError("convergence accepts beads, timestep, separation and bmax")
    if "separation" in axes_raw and task not in ("rate", "drpmd"):
        raise InputError("separation applies only to gas-phase rate/drpmd")
    if "bmax" in axes_raw and task != "drpmd":
        raise InputError("bmax applies only to drpmd collision rates")
    axes = {}
    for axis, values in axes_raw.items():
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            raise InputError(f"Convergence axis {axis} needs at least two values")
        parsed = [_integer(v, axis) if axis == "beads" else _quantity(v, _AXES[axis], axis) for v in values]
        if len(set(parsed)) != len(parsed):
            raise InputError(f"Convergence axis {axis} contains duplicate physical values")
        axes[axis] = sorted(parsed, reverse=axis == "timestep")
    defaults = {"beads": original["beads"], "timestep": original["dt_au"]}
    if task == "rate":
        reaction = original["reaction"]
        defaults["separation"] = parse_quantity(reaction.get("rinf_au", reaction.get("rinf", reaction.get("Rinf", "12 angstrom"))), "length")
    elif task == "drpmd":
        defaults.update(separation=original["system"]["separation_au"], bmax=original["collision"]["bmax_au"])
    finest = {**defaults, **{name: values[-1] for name, values in axes.items()}}
    design = raw.get("design", "one_at_a_time")
    if design == "one_at_a_time":
        settings = [finest] + [{**finest, axis: value} for axis, values in axes.items() for value in values[:-1]]
    elif design == "factorial":
        names = list(axes)
        settings = [{**defaults, **dict(zip(names, values))} for values in itertools.product(*(axes[n] for n in names))]
    else:
        raise InputError("design must be one_at_a_time or factorial")
    replicates = _integer(raw.get("replicates", 1), "replicates")
    seed = _integer(raw.get("seed", original["seed"]), "seed", 0)
    rtol = float(raw.get("relative_tolerance", .05))
    atol = float(raw.get("absolute_tolerance", 0.))
    z = float(raw.get("confidence_z", 1.959963984540054))
    if not np.isfinite([rtol, atol, z]).all() or min(rtol, atol) < 0 or rtol + atol <= 0 or z <= 0:
        raise InputError("Study tolerances must be finite, nonnegative and not both zero; confidence_z must be positive")
    cases = []
    for temperature, label, setting, replica in itertools.product(temps, isotope_masses, settings, range(replicates)):
        config = deepcopy(original)
        identity = {"temperature_K": temperature, "isotopologue": label, **setting}
        token = _digest({"parameters": identity, "replica": replica})
        seed_words = np.random.SeedSequence([seed, *[int(token[i:i + 8], 16) for i in range(0, 32, 8)]]).generate_state(4)
        case_seed = sum(int(word) << (32 * i) for i, word in enumerate(seed_words))
        config.update(temperature=temperature, temperature_K=temperature, beta=1 / (KB_AU * temperature),
                      beads=setting["beads"], seed=case_seed)
        _preserve_durations(config, original["dt_au"], setting["timestep"])
        isotope_system = System.from_dict(system.to_dict())
        isotope_system.masses[:] = isotope_masses[label]
        for key in ("masses_amu", "isotopes"):
            config["system"].pop(key, None)
        config["system"]["masses"] = isotope_system.masses.tolist()
        if task == "rate":
            config["reaction"].update(rinf=setting["separation"], rinf_au=setting["separation"])
            config["reaction"].pop("Rinf", None)
        elif task == "drpmd":
            if setting["separation"] <= setting["bmax"]:
                raise InputError("Every collision case requires separation > bmax")
            config["system"].update(separation=setting["separation"], separation_au=setting["separation"])
            config["collision"].update(separation_au=setting["separation"], bmax=setting["bmax"], bmax_au=setting["bmax"])
            # Parent thermalization locks the fragment COMs at system.geometry.
            # Move intact fragments as well as the launch-distance setting,
            # using isotope masses, so an R0 scan changes both physical stages.
            fragments = [np.asarray(f, dtype=int) for f in isotope_system.fragments]
            source_centers = [np.average(system.geometry[f], axis=0, weights=system.masses[f]) for f in fragments]
            direction = source_centers[1] - source_centers[0]
            separation_norm = float(np.linalg.norm(direction))
            if separation_norm <= 0:
                raise InputError("Collision parent fragments need distinct initial centers of mass")
            direction /= separation_norm
            fragment_masses = [float(isotope_system.masses[f].sum()) for f in fragments]
            origin = np.average(isotope_system.geometry, axis=0, weights=isotope_system.masses)
            targets = [origin - fragment_masses[1] / sum(fragment_masses) * setting["separation"] * direction,
                       origin + fragment_masses[0] / sum(fragment_masses) * setting["separation"] * direction]
            for fragment, target_center in zip(fragments, targets):
                center = np.average(isotope_system.geometry[fragment], axis=0, weights=isotope_system.masses[fragment])
                isotope_system.geometry[fragment] += target_center - center
            config["system"].update(geometry=isotope_system.geometry.tolist(), symbols=list(isotope_system.symbols), geometry_unit="bohr")
            if "bmax" in axes:
                config["collision"]["bmax_unvalidated"] = False
            # Hold product classification radius fixed while testing launch distance.
        cases.append(StudyCase("case-" + token[:20], config, isotope_system, identity, replica))
    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids):
        raise InputError("Duplicate study case identities")
    return {"schema": "pyrpmd.rate-study", "schema_version": 1, "task": task,
            "design": design, "axes": axes, "finest_settings": finest,
            "temperatures_K": temps, "isotopologues": list(isotope_masses), "kie_reference": kie_reference,
            "replicates": replicates, "relative_tolerance": rtol, "absolute_tolerance": atol,
            "confidence_z": z, "cases": cases,
            "scope": "numerical equivalence on tested grid; independent case random streams; physical durations fixed"}


def _extract_rate(summary, task):
    if task == "drpmd":
        entry = summary.get("total_reaction", {})
        rate = entry.get("rate_cm3_molecule_s")
        # A percentile interval alone does not identify a standard error.
        return rate, None, "cm^3 molecule^-1 s^-1"
    suffix, unit = {"rate": ("cm3_s", "cm^3 molecule^-1 s^-1"),
                    "surface_rate": ("s_inv", "s^-1"), "site_rate": ("s", "s^-1")}[task]
    return summary.get("rate_" + suffix), summary.get("rate_standard_error_" + suffix), unit


def _combine(records):
    valid = [r for r in records if r.get("rate") is not None and np.isfinite(r["rate"]) and r["rate"] > 0]
    if len(valid) != len(records):
        return {"rate": None, "standard_error": None, "available": False, "reason": "withheld, nonpositive or invalid case rate", "replicates": len(records)}
    rates = np.asarray([r["rate"] for r in records])
    variances = []
    if len(rates) >= 2:
        variances.append(float(np.var(rates, ddof=1) / len(rates)))
    errors = [r.get("standard_error") for r in records]
    if all(e is not None and np.isfinite(e) and e >= 0 for e in errors):
        variances.append(float(np.sum(np.square(errors)) / len(rates) ** 2))
    return {"rate": float(rates.mean()), "standard_error": math.sqrt(max(variances)) if variances else None,
            "available": True, "replicates": len(rates),
            "uncertainty": "maximum of between-replica and within-run mean variance; independent replicas"}


def analyze_rate_study(plan, records):
    """Combine completed cases; fit temperatures/KIE only at finest settings."""
    expected = {c.case_id: c for c in plan["cases"]}
    unit = "s^-1" if plan["task"] in ("site_rate", "surface_rate") else "cm^3 molecule^-1 s^-1"
    if len(records) != len(expected) or {r.get("case_id") for r in records} != set(expected):
        raise ValueError("Study records must contain each planned case exactly once")
    grouped = defaultdict(list)
    for r in records:
        if "rate_unit" in r and r["rate_unit"] != unit:
            raise ValueError(f"Study record rate_unit must be {unit!r}; convert units explicitly before analysis")
        case = expected[r["case_id"]]
        grouped[json.dumps(case.parameters, sort_keys=True)].append(r)
    rows = [{"parameters": json.loads(key), **_combine(values)} for key, values in grouped.items()]
    lookup = {json.dumps(row["parameters"], sort_keys=True): row for row in rows}
    def row_at(temperature, isotope, settings):
        return lookup.get(json.dumps({"temperature_K": temperature, "isotopologue": isotope, **settings}, sort_keys=True))
    fits, kies, checks = {}, [], []
    finest = plan["finest_settings"]
    for isotope in plan["isotopologues"]:
        selection = [row_at(t, isotope, finest) for t in plan["temperatures_K"]]
        if len(selection) >= 2 and all(r is not None and r["available"] for r in selection):
            errors = [r["standard_error"] for r in selection]
            try:
                fits[isotope] = arrhenius_fit(plan["temperatures_K"], [r["rate"] for r in selection],
                    standard_errors=errors if all(e is not None and e > 0 for e in errors) else None, rate_unit=unit)
            except (ValueError, np.linalg.LinAlgError) as error:
                fits[isotope] = {"available": False, "reason": str(error)}
        else:
            fits[isotope] = {"available": False, "reason": "Need supported positive rates at two or more temperatures"}
    for temperature in plan["temperatures_K"]:
        ref = row_at(temperature, plan["kie_reference"], finest)
        for isotope in plan["isotopologues"]:
            if isotope == plan["kie_reference"]:
                continue
            other = row_at(temperature, isotope, finest)
            entry = {"temperature_K": temperature, "numerator": plan["kie_reference"], "denominator": isotope}
            if ref and other and ref["available"] and other["available"]:
                errors = {"numerator_error": ref["standard_error"], "denominator_error": other["standard_error"]}
                try:
                    entry.update(kinetic_isotope_effect(ref["rate"], other["rate"], **(errors if all(e is not None for e in errors.values()) else {})))
                except ValueError as error:
                    entry.update(ratio=None, reason=str(error))
            else:
                entry.update(ratio=None, reason="withheld or missing case rate")
            kies.append(entry)
        for isotope, (axis, values) in itertools.product(plan["isotopologues"], plan["axes"].items()):
            # Test the last two refinements (or the single available pair).
            for coarse, fine in zip(values[-3:-1] if len(values) >= 3 else values[:1], values[-2:] if len(values) >= 3 else values[1:]):
                a = row_at(temperature, isotope, {**finest, axis: coarse})
                b = row_at(temperature, isotope, {**finest, axis: fine})
                check = {"temperature_K": temperature, "isotopologue": isotope, "axis": axis,
                         "coarse": coarse, "fine": fine, "status": "inconclusive", "passed": False}
                if a and b and a["available"] and b["available"] and a["standard_error"] is not None and b["standard_error"] is not None:
                    difference = abs(a["rate"] - b["rate"])
                    margin = plan["confidence_z"] * math.hypot(a["standard_error"], b["standard_error"])
                    tolerance = plan["absolute_tolerance"] + plan["relative_tolerance"] * abs(b["rate"])
                    passed = difference + margin <= tolerance
                    check.update(difference=difference, uncertainty_margin=margin, tolerance=tolerance,
                                 passed=passed, status="equivalent_on_tested_pair" if passed else "not_established")
                else:
                    check["reason"] = "Positive accepted rates and full standard errors (or independent replicas) are required"
                checks.append(check)
    return {"schema": "pyrpmd.rate-study.results", "schema_version": 1, "rate_unit": unit,
            "rows": rows, "arrhenius": fits, "kinetic_isotope_effects": kies, "convergence_checks": checks,
            "numerically_converged_on_tested_grid": bool(checks) and all(c["passed"] for c in checks),
            "physical_converged": False,
            "notes": ["Only accepted workflow rates are fitted; diagnostic rate_estimate values are never substituted.",
                      "Convergence requires |difference| + z*SE(difference) <= atol + rtol*|finer rate|.",
                      "A tested grid does not establish sampling, PMF-domain, recrossing-time, classifier or PES validity.",
                      "Two-level axes and one-at-a-time designs provide limited evidence; use more levels/factorial cases to investigate interactions."]}


def run_rate_study(study, *, output, resume=False, base_directory=None):
    """Run/resume a complete study, preserving per-case workflow checkpoints."""
    from .runner import run_resolved
    plan = plan_rate_study(study, base_directory=base_directory)
    target = Path(output).resolve()
    public = {key: value for key, value in plan.items() if key != "cases"}
    public["cases"] = [case.to_dict() for case in plan["cases"]]
    files = {}
    for case in plan["cases"]:
        files.update(referenced_files(case.config))
    identity = _digest({"plan": public, "files": files})
    manifest = target / "study.json"
    if resume:
        if not manifest.is_file() or json.loads(manifest.read_text()).get("identity") != identity:
            raise InputError("Study resume requires the identical plan, system and referenced files")
    elif target.exists() and any(target.iterdir()):
        raise InputError("Study output is not empty; use resume with the same plan")
    target.mkdir(parents=True, exist_ok=True)
    if not resume:
        write_json(manifest, {**public, "identity": identity, "files": files})
    records = []
    for case in plan["cases"]:
        destination = target / case.case_id
        case_resume = resume and (destination / "manifest.json").is_file()
        result = run_resolved(case.config, case.system, output=destination, resume=case_resume)
        rate, error, unit = _extract_rate(result, plan["task"])
        record = {**case.to_dict(), "rate": rate, "standard_error": error, "rate_unit": unit,
                  "workflow_status": result.get("status"), "output": str(destination)}
        records.append(record)
        write_json(target / "progress.json", {"completed": len(records), "total": len(plan["cases"]), "records": records})
    report = analyze_rate_study(plan, records)
    report["cases"] = records
    write_json(target / "results.json", report)
    with (target / "rates.csv").open("w", newline="", encoding="utf-8") as stream:
        keys = ["case_id", "temperature_K", "isotopologue", "replica", "beads", "timestep", "separation", "bmax", "rate", "standard_error", "rate_unit"]
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for record in records:
            row = {**record, **record["parameters"]}
            writer.writerow({key: row.get(key) for key in keys})
    return report
