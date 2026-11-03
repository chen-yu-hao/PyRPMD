"""Deterministic molecular instanton rate and ground-doublet convergence studies.

This workflow uses atomic units and never infers PES identity, reactant states,
symmetry numbers, or physical accuracy from a converged stationary path.  It
has its own versioned scientific identity and does not change RPMD identities
or checkpoint formats.  JSON reports are atomic results, not restart files.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import inspect
import json
import math
from pathlib import Path
import runpy

import numpy as np
from scipy.interpolate import CubicSpline

from ..instanton import (InstantonError, RingPolymerAction, _mask_and_masses,
                        crossover_temperature, optimize_instanton)
from ..io import file_sha256, fingerprint_payload, jsonable, write_json
from ..low_temperature import HarmonicState, _compatible, instanton_rate
from ..potentials.base import coordinates
from ..tunnelling import tunnelling_splitting
from ..validation import finite_positive, finite_real, nonnegative_integer, positive_integer


_SCHEMA = "PyRPMD.instanton-workflow.v1"
_CONVERGENCE_CRITERION = {"successive_refinements": 2,
                          "minimum_bead_refinement_ratio": 1.5,
                          "minimum_duration_refinement_ratio": 1.1}


def _portable(value, name):
    """Strict JSON, with no repr/callable fallback in a scientific identity."""
    def validate_keys(item):
        # jsonable normally stringifies mapping keys. Scientific identities
        # must reject collisions such as {1: a, "1": b} before that conversion.
        if is_dataclass(item) and not isinstance(item, type):
            item = asdict(item)
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ValueError(f"{name} mappings must have string keys at every nesting level")
            for child in item.values():
                validate_keys(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                validate_keys(child)

    validate_keys(value)
    try:
        converted = jsonable(value, callables=False, dataclasses="asdict", to_dict="ignore")
        json.dumps(converted, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite, JSON-serializable scientific data") from exc
    return converted


def _mapping(value, name):
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    return dict(value)


def _options(function, supplied, excluded, name, overrides=None):
    """Materialize every scientific default in the versioned run identity."""
    defaults = {
        key: parameter.default
        for key, parameter in inspect.signature(function).parameters.items()
        if key not in excluded and parameter.default is not inspect.Parameter.empty
    }
    value = {} if supplied is None else _mapping(supplied, name)
    unknown = set(value) - set(defaults)
    if unknown:
        raise ValueError(f"unknown or workflow-owned {name}: {sorted(unknown)}")
    defaults.update(overrides or {})
    defaults.update(value)
    return defaults


def _counts(values, *, even):
    if isinstance(values, (str, bytes)):
        raise ValueError("bead_counts must contain at least three increasing integers")
    try:
        counts = [positive_integer(value, "bead_counts entry") for value in values]
    except TypeError as exc:
        raise ValueError("bead_counts must contain at least three increasing integers") from exc
    if (len(counts) < 3 or any(p < 8 or (even and p % 2) for p in counts)
            or any(b <= a for a, b in zip(counts[:-1], counts[1:]))):
        qualifier = "even " if even else ""
        raise ValueError(f"bead_counts must contain at least three strictly increasing {qualifier}integers >= 8")
    return counts


def _validate_options(optimizer=None, rate=None, splitting=None):
    """Reject malformed numerical controls before stationarity/PES checks."""
    for options in (optimizer, rate, splitting):
        if options is None:
            continue
        for name in ("hessian_step_bohr", "gradient_tolerance", "minimum_spread_bohr",
                     "trust_radius_bohr", "phase_tolerance"):
            if name in options:
                options[name] = finite_positive(options[name], name)
        if "max_iterations" in options:
            options["max_iterations"] = positive_integer(options["max_iterations"], "max_iterations")
    if optimizer is not None:
        if optimizer["method"] not in ("root", "bofill"):
            raise ValueError("optimizer method must be root or bofill")
        optimizer["hessian_update_interval"] = nonnegative_integer(
            optimizer["hessian_update_interval"], "hessian_update_interval")
        if optimizer["method"] == "root" and (optimizer["hessian_update_interval"]
                                               or optimizer["trust_radius_bohr"] != 0.3):
            raise ValueError("Hessian update and trust radius options require method='bofill'")
    if rate is not None:
        if not isinstance(rate["rigid_body"], (bool, np.bool_)):
            raise ValueError("rigid_body must be boolean")
        for name in ("symmetry_number", "electronic_degeneracy", "reaction_degeneracy"):
            rate[name] = positive_integer(rate[name], name)


def _provider_identity(potential, declared):
    identity = _mapping(declared, "potential_identity")
    if not identity:
        raise ValueError("potential_identity must explicitly identify the PES and its scientific parameters")
    if not callable(getattr(potential, "evaluate", None)):
        raise ValueError("potential must implement evaluate(x) in atomic units")
    sources = {}
    for name in ("evaluate", "hessian"):
        method = getattr(potential, name, None)
        if callable(method):
            try:
                filename = inspect.getsourcefile(method)
            except (TypeError, OSError):
                filename = None
            if filename and Path(filename).is_file():
                sources[name] = file_sha256(filename)
    return _portable({"declared": identity,
                      "provider_class": f"{type(potential).__module__}:{type(potential).__qualname__}",
                      "provider_source_sha256": sources,
                      "metadata": getattr(potential, "metadata", {}),
                      "hessian_provider": callable(getattr(potential, "hessian", None))},
                     "potential_identity")


def _closed_resample(path, nbeads):
    """Interpolate in imaginary time, preserving link reflection exactly."""
    path = coordinates(path)
    old = len(path)
    if old < 8 or old % 2:
        raise ValueError("a thermal initial_path needs an even number of beads >= 8")
    link = np.allclose(path, path[::-1], rtol=1e-9, atol=1e-9)
    bead = np.allclose(path, path[(-np.arange(old)) % old], rtol=1e-9, atol=1e-9)
    if not link and not bead:
        raise ValueError("thermal initial_path must be folded under link or bead reflection")
    if old == nbeads and link:
        return path.copy()
    # Coordinates of old knot zero relative to the turning point.  A cubic
    # periodic interpolation is only a seed; all new beads are reoptimized.
    offset = 0.5 / old if link else 0.0
    spline = CubicSpline(np.arange(old + 1) / old,
                         np.concatenate((path, path[:1])), axis=0, bc_type="periodic")
    sampled = spline(((np.arange(nbeads) + 0.5) / nbeads - offset) % 1.0)
    return 0.5 * (sampled + sampled[::-1])


def _open_resample(path, old_duration, duration, intervals):
    """Refine a kink on physical imaginary time; longer paths extend its tails."""
    source = np.linspace(-old_duration / 2, old_duration / 2, len(path))
    target = np.linspace(-duration / 2, duration / 2, intervals + 1)
    flat = np.asarray(path).reshape(len(path), -1)
    sampled = np.column_stack([np.interp(target, source, column) for column in flat.T])
    sampled = sampled.reshape(intervals + 1, *path.shape[1:])
    sampled[0], sampled[-1] = path[0], path[-1]
    return sampled


def _convergence(log_values, relative_tolerance, resolutions, minimum_ratio):
    """Require the last two refinements, not a single accidentally small change."""
    differences = [abs(float(b - a)) for a, b in zip(log_values[:-1], log_values[1:])]
    changes = [math.expm1(delta) if delta < 700 else None for delta in differences]
    ratios = [float(b/a) for a, b in zip(resolutions[:-1], resolutions[1:])]
    meaningful = len(ratios) >= 2 and all(ratio >= minimum_ratio*(1-1e-12)
                                         for ratio in ratios[-2:])
    threshold = math.log1p(relative_tolerance)
    passed = meaningful and all(delta <= threshold for delta in differences[-2:])
    return {"relative_tolerance": relative_tolerance,
            "relative_changes": changes, "absolute_log_changes": differences,
            "converged": bool(passed), "required_successive_refinements": 2,
            "refinement_ratios": ratios, "minimum_refinement_ratio": minimum_ratio,
            "sufficient_refinement_span": bool(meaningful),
            "relative_change_definition": "max(value_a,value_b)/min(value_a,value_b)-1"}


def run_instanton(potential, masses, initial_path, *, bead_counts, potential_identity,
                  mode="rate", temperature_kelvin=None, reactants=None,
                  active_mask=None, transition_state=None, energy_reference_hartree=0.0,
                  optimizer_options=None, rate_options=None, imaginary_times_au=None,
                  splitting_options=None, golden_rule_options=None, convergence_relative_tolerance=0.01,
                  output=None):
    """Run a complete deterministic convergence study and optionally save JSON.

    ``mode='rate'`` optimizes a folded thermal ring at each strictly increasing
    even bead count (at least three), evaluates the full Gaussian prefactor and
    harmonic reactant partitions, and compares the final two rate refinements.
    ``reactants`` contains prepared :class:`HarmonicState` objects.  Isolated
    gas molecules use ``rate_options={'rigid_body': True, ...}``; frozen models
    use an explicit ``active_mask``.  The final Hessian is reused for the rate.

    ``mode='splitting'`` studies open kinks on a grid of at least three durations
    and three time steps.  ``bead_counts`` then counts INTERVALS at the shortest
    duration, so paths have one extra endpoint node.  Longer durations scale
    the interval counts by their duration ratio, which must give integer
    counts.  Duration comparisons therefore keep the time step exactly fixed.
    ``initial_path`` belongs to the shortest duration.  Assert equivalent
    wells explicitly in ``splitting_options={'equivalent_wells': True}``.

    ``mode='golden_rule'`` uses two ordinary diabatic-surface potentials and two
    directed open paths. ``bead_counts`` then contains (P0,P1) INTERVAL pairs.
    The second path closes the first: path0[0]=path1[-1], path0[-1]=path1[0].
    Supply separate ``potential_identity['surface0']`` and ``['surface1']``
    records plus a constant ``golden_rule_options['coupling_hartree']``. Both
    segment grids and their optimized imaginary-time allocation are studied;
    see :func:`pyrpmd.workflows.golden_rule.run_golden_rule_study` for scope.

    ``potential_identity`` is a required nonempty mapping identifying the PES,
    its parameters, model files/checksums and any remote model version.  Code
    and metadata are recorded too, but cannot infer hidden external parameters.
    Every resolved numerical/scientific option enters the separate v1 identity.
    ``output`` is an optional JSON FILE, atomically replaced after success.
    No RNG or optimizer restart state is created.  ``physical_converged`` always
    remains false: grid convergence is not a validation of the PES, harmonic
    reactant approximation or leading semiclassical theory.
    """
    if mode == "golden_rule":
        if any(value is not None for value in (transition_state, optimizer_options, rate_options,
                                               imaginary_times_au, splitting_options)):
            raise ValueError("golden_rule mode accepts golden_rule_options, not thermal TS/optimizer/rate or splitting settings")
        from .golden_rule import run_golden_rule_study
        return run_golden_rule_study(
            potential, masses, initial_path, bead_counts=bead_counts,
            potential_identity=potential_identity, temperature_kelvin=temperature_kelvin,
            reactants=reactants, active_mask=active_mask,
            energy_reference_hartree=energy_reference_hartree,
            golden_rule_options=golden_rule_options,
            convergence_relative_tolerance=convergence_relative_tolerance, output=output)
    if mode not in ("rate", "splitting"):
        raise ValueError("mode must be rate, splitting or golden_rule")
    if golden_rule_options is not None:
        raise ValueError("golden_rule_options requires mode='golden_rule'")
    q = coordinates(initial_path).copy()
    mask, _ = _mask_and_masses(q.shape[1:], masses, active_mask)
    masses = np.array(masses, dtype=float, copy=True)
    if not np.allclose(q[:, ~mask], q[0, ~mask], rtol=0, atol=1e-12):
        raise ValueError("inactive coordinates must be identical on every path node")
    counts = _counts(bead_counts, even=mode == "rate")
    if (counts[-1] - (mode == "splitting")) * int(mask.sum()) > 4096:
        raise ValueError("dense instanton studies are limited to 4096 active bead/interior coordinates")
    tolerance = finite_positive(convergence_relative_tolerance, "convergence_relative_tolerance")
    if tolerance >= 1:
        raise ValueError("convergence_relative_tolerance must be smaller than one")
    reference = finite_real(energy_reference_hartree, "energy_reference_hartree")
    provider = _provider_identity(potential, potential_identity)
    identity = {"schema": _SCHEMA, "mode": mode, "potential": provider,
                "masses": masses, "initial_path_bohr": q, "active_mask": mask,
                "bead_counts": counts, "convergence_relative_tolerance": tolerance,
                "convergence_criterion": dict(_CONVERGENCE_CRITERION)}

    if mode == "rate":
        if imaginary_times_au is not None or splitting_options is not None:
            raise ValueError("rate mode does not accept imaginary_times_au or splitting_options")
        temperature = finite_positive(temperature_kelvin, "temperature_kelvin")
        if not isinstance(reactants, (tuple, list)) or not reactants or not all(
                isinstance(state, HarmonicState) for state in reactants):
            raise ValueError("rate mode requires a nonempty list of HarmonicState reactants")
        optimizer = _options(optimize_instanton, optimizer_options,
                             {"active_mask", "transition_state", "energy_reference_hartree",
                              "raise_on_failure", "retain_hessian"}, "optimizer_options",
                             {"method": "bofill"})
        rate = _options(instanton_rate, rate_options,
                        {"action_hessian"}, "rate_options",
                        {"gradient_tolerance": optimizer["gradient_tolerance"]})
        _validate_options(optimizer=optimizer, rate=rate)
        if rate["rigid_body"] and not mask.all():
            raise ValueError("rigid_body requires all Cartesian coordinates active")
        if not rate["rigid_body"] and rate["symmetry_number"] != 1:
            raise ValueError("constrained paths do not have a rotational symmetry number")
        _compatible(q[0], masses, mask, rate["rigid_body"], reactants)
        for state in reactants:
            state.spectrum()  # Validate all reactant partitions before orbit PES calls.
        seed = _closed_resample(q, counts[0])
        ts = None if transition_state is None else coordinates(np.asarray(transition_state)[None])[0]
        if ts is not None and (ts.shape != q.shape[1:] or not np.allclose(
                ts[~mask], q[0, ~mask], rtol=0, atol=1e-12)):
            raise ValueError("transition_state must match the path shape and frozen coordinates")
        identity.update(temperature_kelvin=temperature,
                        reactants=[asdict(state) for state in reactants],
                        transition_state_bohr=ts, energy_reference_hartree=reference,
                        optimizer_options=dict(optimizer, raise_on_failure=True, retain_hessian=True),
                        rate_options=rate)
    else:
        if (temperature_kelvin is not None or reactants is not None or transition_state is not None
                or optimizer_options is not None or rate_options is not None or reference != 0.0):
            raise ValueError("splitting mode accepts durations and splitting_options, not thermal rate settings")
        if len(q) < 9:
            raise ValueError("an open initial_path needs at least nine nodes including endpoints")
        try:
            durations = [finite_positive(t, "imaginary_times_au entry") for t in imaginary_times_au]
        except TypeError as exc:
            raise ValueError("splitting mode requires at least three increasing imaginary_times_au") from exc
        if len(durations) < 3 or any(b <= a for a, b in zip(durations[:-1], durations[1:])):
            raise ValueError("imaginary_times_au must contain at least three strictly increasing durations")
        interval_grid = []
        for duration in durations:
            ratio = duration / durations[0]
            if not math.isfinite(ratio) or (counts[-1] * ratio - 1) * int(mask.sum()) > 4096:
                raise ValueError("dense splitting Hessian is limited to 4096 interior coordinates")
            proposed = np.array(counts, dtype=float) * ratio
            rounded = np.rint(proposed)
            if not np.allclose(proposed, rounded, rtol=1e-12, atol=1e-9):
                raise ValueError("durations must give integer scaled bead_counts to keep duration scans at fixed dtau")
            interval_grid.append([int(p) for p in rounded])
        splitting = _options(tunnelling_splitting, splitting_options,
                             {"active_mask"}, "splitting_options")
        _validate_options(splitting=splitting)
        if splitting["equivalent_wells"] is not True:
            raise ValueError("splitting_options must explicitly assert equivalent_wells=True")
        if (interval_grid[-1][-1] - 1) * int(mask.sum()) > 4096:
            raise ValueError("dense splitting Hessian is limited to 4096 interior coordinates")
        identity.update(imaginary_times_au=durations, interval_grid=interval_grid,
                        splitting_options=splitting)

    identity = _portable(identity, "scientific_identity")
    report = {"schema": _SCHEMA, "mode": mode,
              "physics_fingerprint": fingerprint_payload(identity), "scientific_identity": identity,
              "rows": [], "physical_converged": False,
              "convergence_scope": "stationarity_and_requested_discretization_only",
              "limitations": ["leading semiclassical theory; not exact quantum dynamics",
                              "PES, Hessian-step and stationary-tolerance validation remain independent",
                              "harmonic reactant and classical rigid-rotor approximations require validation"
                              if mode == "rate" else "one equivalent-well ground doublet; no free-rotor or multiwell splitting pattern"]}

    if mode == "rate":
        action = RingPolymerAction(potential, masses, temperature, active_mask=mask,
                                   energy_reference_hartree=reference,
                                   hessian_step_bohr=optimizer["hessian_step_bohr"])
        tc = None
        if ts is not None:
            tc = crossover_temperature(potential, ts, masses, active_mask=mask,
                                       hessian_step_bohr=optimizer["hessian_step_bohr"])
            if temperature >= tc:
                raise InstantonError(f"thermal instanton requires temperature below crossover ({tc:.8g} K)")
        report["crossover_temperature_kelvin"] = tc
        for p in counts:
            seed = _closed_resample(seed, p)
            try:
                orbit = optimize_instanton(potential, masses, temperature, seed,
                                           active_mask=mask, energy_reference_hartree=reference,
                                           retain_hessian=True, raise_on_failure=True, **optimizer)
                row = instanton_rate(action, orbit, reactants, **rate)
            except (ValueError, InstantonError) as exc:
                raise InstantonError(f"rate study failed at P={p}: {exc}") from exc
            report["rows"].append({"nbeads": p, "orbit": orbit.to_dict(), "rate": row})
            seed = orbit.path_bohr
            # The report contains modes and paths, not the dense matrix. Release
            # it before the next level allocates a larger fluctuation Hessian.
            del orbit
        check = _convergence([row["rate"]["log_rate_au"] for row in report["rows"]], tolerance,
                             counts, _CONVERGENCE_CRITERION["minimum_bead_refinement_ratio"])
        report.update(bead_convergence=check, numerical_converged=True,
                      discretization_converged=check["converged"], converged=check["converged"],
                      final_rate=report["rows"][-1]["rate"])
    else:
        seed, previous_duration = q, durations[0]
        for index, duration in enumerate(durations):
            row_seed = _open_resample(seed, previous_duration, duration, interval_grid[index][0])
            coarse_path = None
            for intervals in interval_grid[index]:
                row_seed = _open_resample(row_seed, duration, duration, intervals)
                try:
                    row = tunnelling_splitting(potential, masses, row_seed, duration,
                                               active_mask=mask, **splitting)
                except (ValueError, InstantonError) as exc:
                    raise InstantonError(f"splitting study failed at duration={duration:g}, intervals={intervals}: {exc}") from exc
                report["rows"].append({"imaginary_time_au": duration, "nintervals": intervals,
                                       "dtau_au": duration / intervals, "splitting": row})
                row_seed = np.asarray(row["path_bohr"])
                if coarse_path is None:
                    coarse_path = row_seed
            seed, previous_duration = coarse_path, duration
        grid = [report["rows"][i * len(counts):(i + 1) * len(counts)] for i in range(len(durations))]
        grid_checks = [_convergence([row["splitting"]["log_splitting_hartree"] for row in level], tolerance,
                                    [row["nintervals"] for row in level],
                                    _CONVERGENCE_CRITERION["minimum_bead_refinement_ratio"])
                       for level in grid]
        duration_checks = [_convergence([level[j]["splitting"]["log_splitting_hartree"] for level in grid], tolerance,
                                        durations, _CONVERGENCE_CRITERION["minimum_duration_refinement_ratio"])
                           for j in range(len(counts))]
        # Two finest time steps must both resolve the duration limit.  The grid
        # check must pass at every duration used in the two final comparisons.
        passed = (all(check["converged"] for check in grid_checks[-3:])
                  and all(check["converged"] for check in duration_checks[-2:]))
        report.update(grid_convergence=grid_checks, duration_convergence=duration_checks,
                      duration_comparisons_at_fixed_dtau=True, numerical_converged=True,
                      discretization_converged=bool(passed), converged=bool(passed),
                      final_splitting=report["rows"][-1]["splitting"])
    if output is not None:
        write_json(output, report)
    return report


def load_instanton_input(path):
    """Read a trusted Python file whose build() returns run_instanton kwargs.

    A Python input can construct a real molecular PES and analytic Hessian
    provider without hiding it behind a unit-converting adapter.  All arguments
    and returned PES data must already use the documented atomic units.
    """
    path = Path(path).resolve()
    if path.suffix != ".py":
        raise ValueError("instanton INPUT must be a trusted .py file defining build()")
    before = file_sha256(path)
    namespace = runpy.run_path(str(path))
    build = namespace.get("build")
    if not callable(build):
        raise ValueError("instanton INPUT must define a callable build()")
    mapping = _mapping(build(), "instanton build() result")
    if file_sha256(path) != before:
        raise ValueError("instanton input changed while build() was executing")
    unknown = set(mapping) - set(inspect.signature(run_instanton).parameters)
    if unknown:
        raise ValueError(f"unknown instanton input fields: {sorted(unknown)}")
    if "output" in mapping:
        raise ValueError("select the report path with --output, not inside scientific build() input")
    identity = _mapping(mapping.get("potential_identity"), "potential_identity")
    if not identity:
        raise ValueError("potential_identity must explicitly identify the PES and its scientific parameters")
    if mapping.get("mode") == "golden_rule":
        from .golden_rule import surface_identities
        declarations = surface_identities(identity)
        mapping["potential_identity"] = {
            f"surface{index}": {"declared": declaration, "input_sha256": before}
            for index, declaration in enumerate(declarations)
        }
    else:
        mapping["potential_identity"] = {"declared": identity, "input_sha256": before}
    return mapping


def run_instanton_input(path, output=None):
    """Execute a Python build() input; return the same report as the Python API."""
    return run_instanton(**load_instanton_input(path), output=output)


def _command(args):
    report = run_instanton_input(args.input, args.output)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


def register_parser(commands):
    """Register the standalone command with no changes to the RPMD input schema."""
    parser = commands.add_parser("instanton", help="Run thermal-rate, ground-splitting or golden-rule instanton convergence studies")
    parser.add_argument("input", help="Trusted Python .py file defining build(); values use atomic units")
    parser.add_argument("--output", help="Atomically written JSON report file")
    parser.set_defaults(_instanton_handler=_command)
    return parser
