"""Independent two-segment discretization studies for golden-rule instantons.

This workflow is deliberately separate from the versioned adiabatic rate and
splitting identities. It covers a constant electronic coupling between two
bound or Cartesian-constrained diabatic surfaces in the normal regime.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import inspect
import math

import numpy as np

from ..constants import KB_AU
from ..golden_rule_partition import harmonic_segment_log_partition
from ..instanton import InstantonError, _mask_and_masses
from ..io import fingerprint_payload, write_json
from ..low_temperature import HarmonicState, _compatible
from ..potentials.base import coordinates
from ..validation import finite_positive, finite_real, positive_integer
from .instanton import (_convergence, _mapping, _options, _portable,
                        _provider_identity)


_SCHEMA = "PyRPMD.golden-rule-instanton-workflow.v1"
_CRITERION = {"successive_refinements": 2, "minimum_segment_refinement_ratio": 1.5,
              "require_joint_and_both_independent_segment_scans": True}


def surface_identities(value):
    """Require distinct explicit identity records for both diabatic surfaces."""
    identity = _mapping(value, "golden-rule potential_identity")
    if set(identity) != {"surface0", "surface1"}:
        raise ValueError("golden-rule potential_identity requires exactly surface0 and surface1 mappings")
    declarations = []
    for name in ("surface0", "surface1"):
        declaration = _mapping(identity[name], f"potential_identity.{name}")
        if not declaration:
            raise ValueError(f"potential_identity.{name} must explicitly identify that PES and its parameters")
        declarations.append(_portable(declaration, f"potential_identity.{name}"))
    return declarations


def _interval_levels(values):
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError("golden-rule bead_counts must contain at least three (P0,P1) interval pairs")
    try:
        rows = list(values)
    except TypeError as exc:
        raise ValueError("golden-rule bead_counts must contain at least three (P0,P1) interval pairs") from exc
    levels = []
    for row in rows:
        if isinstance(row, (str, bytes, Mapping)):
            raise ValueError("each golden-rule bead_counts entry must be a pair of interval counts")
        try:
            pair = tuple(row)
        except TypeError as exc:
            raise ValueError("each golden-rule bead_counts entry must be a pair of interval counts") from exc
        if len(pair) != 2:
            raise ValueError("each golden-rule bead_counts entry must be a pair of interval counts")
        pair = tuple(positive_integer(p, "segment interval count") for p in pair)
        if min(pair) < 2:
            raise ValueError("each golden-rule segment needs at least two intervals")
        levels.append(pair)
    if len(levels) < 3 or any(b[0] <= a[0] or b[1] <= a[1]
                              for a, b in zip(levels[:-1], levels[1:])):
        raise ValueError("golden-rule bead_counts needs at least three pairs increasing strictly on both surfaces")
    return levels


def _resample_segment(path, intervals):
    """Interpolate only a seed; preserve both hopping endpoints exactly."""
    source = np.linspace(0.0, 1.0, len(path))
    target = np.linspace(0.0, 1.0, intervals + 1)
    flat = path.reshape(len(path), -1)
    result = np.column_stack([np.interp(target, source, column) for column in flat.T])
    result = result.reshape(intervals + 1, *path.shape[1:])
    result[0], result[-1] = path[0], path[-1]
    return result


def _scan_points(levels):
    """m diagonal levels plus both independent axes cost 3m-2 optimizations."""
    last0, last1 = levels[-1]
    points = list(levels)
    for point in [(p0, last1) for p0, _ in levels] + [(last0, p1) for _, p1 in levels]:
        if point not in points:
            points.append(point)
    return points


def _resolve_options(options, optimizer, rate):
    supplied = _mapping(options, "golden_rule_options")
    allowed_optimizer = {
        name for name, p in inspect.signature(optimizer).parameters.items()
        if p.default is not inspect.Parameter.empty
        and name not in {"active_mask", "energy_reference_hartree", "raise_on_failure"}
    }
    allowed = allowed_optimizer | {"coupling_hartree", "log_reactant_partition", "reaction_degeneracy",
                                   "partition_discretization"}
    unknown = set(supplied) - allowed
    if unknown:
        raise ValueError(f"unknown or workflow-owned golden_rule_options: {sorted(unknown)}")
    if "coupling_hartree" not in supplied:
        raise ValueError("golden_rule_options requires the constant coupling_hartree")
    numerical = _options(optimizer, {k: v for k, v in supplied.items() if k in allowed_optimizer},
                         {"active_mask", "energy_reference_hartree", "raise_on_failure"},
                         "golden_rule_options")
    for name in ("hessian_step_bohr", "gradient_tolerance", "energy_tolerance_hartree",
                 "tau_fraction_tolerance"):
        numerical[name] = finite_positive(numerical[name], name)
    for name in ("max_iterations", "max_tau_iterations"):
        numerical[name] = positive_integer(numerical[name], name)
    numerical["initial_tau_fraction"] = finite_positive(numerical["initial_tau_fraction"], "initial_tau_fraction")
    try:
        bounds = tuple(finite_positive(value, "tau_fraction_bounds entry")
                       for value in numerical["tau_fraction_bounds"])
    except TypeError as exc:
        raise ValueError("tau_fraction_bounds must contain two finite fractions") from exc
    if len(bounds) != 2 or not 0 < bounds[0] < bounds[1] < 1:
        raise ValueError("tau_fraction_bounds must satisfy 0 < lower < upper < 1")
    numerical["tau_fraction_bounds"] = bounds
    if not bounds[0] < numerical["initial_tau_fraction"] < bounds[1]:
        raise ValueError("initial_tau_fraction must lie strictly inside tau_fraction_bounds")
    if numerical["regime"] != "normal":
        raise ValueError("golden-rule workflow currently requires regime='normal'; inverted and activationless limits are unsupported")
    if not isinstance(numerical["rigid_body"], (bool, np.bool_)) or numerical["rigid_body"]:
        raise ValueError("golden-rule workflow requires rigid_body=False; free-molecule rigid modes are unsupported")
    kinetic = _options(rate, {"reaction_degeneracy": supplied.get("reaction_degeneracy", 1)},
                       set(), "golden_rule_rate_options")
    kinetic["reaction_degeneracy"] = positive_integer(kinetic["reaction_degeneracy"], "reaction_degeneracy")
    kinetic["coupling_hartree"] = finite_real(supplied["coupling_hartree"], "coupling_hartree")
    if kinetic["coupling_hartree"] == 0:
        raise ValueError("coupling_hartree must be a nonzero real constant")
    declared_partition = supplied.get("log_reactant_partition")
    if "log_reactant_partition" in supplied:
        declared_partition = finite_real(declared_partition, "log_reactant_partition")
    partition_mode = supplied.get("partition_discretization", "matched")
    if partition_mode not in ("matched", "continuous", "external"):
        raise ValueError("partition_discretization must be matched, continuous or external")
    return numerical, kinetic, declared_partition, partition_mode


def _reactant_partition(reactants, declared, partition_mode, *, masses, mask, geometry, beta, reference):
    if reactants is None:
        if declared is None:
            raise ValueError("golden-rule rate needs one harmonic reactant or an explicit log_reactant_partition")
        if partition_mode != "external":
            raise ValueError("explicit log_reactant_partition requires partition_discretization='external'")
        return float(declared), None, {"source": "explicit_caller_log_partition",
                                  "partition_discretization": partition_mode,
                                  "log_reactant_partition": float(declared),
                                  "normalization": "dimensionless_bound_or_constrained",
                                  "energy_reference_hartree": reference,
                                  "caller_must_use_the_action_energy_zero": True}
    if declared is not None:
        raise ValueError("reactants and explicit log_reactant_partition are mutually exclusive")
    if partition_mode == "external":
        raise ValueError("partition_discretization='external' requires explicit log_reactant_partition, without reactants")
    if not isinstance(reactants, (list, tuple)) or len(reactants) != 1 or not isinstance(reactants[0], HarmonicState):
        raise ValueError("golden-rule workflow requires exactly one bound/constrained HarmonicState reactant")
    state = reactants[0]
    if state.rigid_body:
        raise ValueError("golden-rule workflow does not support free-molecule rigid modes or gas reactant partitions")
    _compatible(geometry, masses, mask, False, reactants)
    frequencies = np.sqrt(state.spectrum()[0])
    identity = {"source": "harmonic_reactant", "partition_discretization": partition_mode,
                "state": asdict(state), "frequencies_au": frequencies,
                "normalization": "dimensionless_bound_or_constrained",
                "energy_reference_hartree": reference,
                "finite_P_reactant_normalization": partition_mode == "matched"}
    if partition_mode == "matched":
        identity["partition_algorithm"] = "two_section_primitive_harmonic_transfer_v1"
        return None, frequencies, identity
    log_qr = state.log_partition(beta) - beta * (state.energy_hartree - reference)
    identity.update(log_reactant_partition=float(log_qr),
                    partition_diagnostics=state.partition_diagnostics(beta))
    return float(log_qr), frequencies, identity


def run_golden_rule_study(potential, masses, initial_path, *, bead_counts,
                          potential_identity, temperature_kelvin, reactants=None,
                          active_mask=None, energy_reference_hartree=0.0,
                          golden_rule_options=None, convergence_relative_tolerance=0.01,
                          output=None):
    """Optimize the imaginary-time split on every joint and independent grid.

    ``potential=(surface0,surface1)`` contains ordinary atomic-unit energy/force
    providers, including adapters returned by :mod:`pyrpmd.golden_rule_adapters`.
    ``initial_path=(path0,path1)`` contains directed open segments with
    ``path0[0]==path1[-1]`` and ``path0[-1]==path1[0]``. A ``(P0,P1)`` row in
    ``bead_counts`` counts INTERVALS, so each segment has Pi+1 nodes.

    At least three levels are required. The workflow computes the joint
    refinement and refines each segment independently with the other at its
    finest requested grid (3m-2 optimizations for m levels). Imaginary time is
    reoptimized at every point. All three convergence checks must pass; two
    successive refinements of each segment must span a factor of at least 1.5.
    This does not establish weak-coupling, semiclassical or PES accuracy.

    Harmonic reactants default to ``partition_discretization='matched'`` in
    ``golden_rule_options``: the primitive harmonic partition uses the same
    two interval counts and the final optimized times as each path. An explicit
    ``'continuous'`` choice uses continuum harmonic reactants. A caller-supplied
    ``log_reactant_partition`` instead requires ``'external'`` and no reactant
    states; its physical normalization and energy origin remain the caller's
    responsibility. Every choice is recorded in the scientific identity.
    """
    from ..golden_rule import golden_rule_rate, optimize_golden_rule_instanton

    if not isinstance(potential, (tuple, list)) or len(potential) != 2:
        raise ValueError("golden-rule potential must contain exactly two energy/force providers")
    if not isinstance(initial_path, (tuple, list)) or len(initial_path) != 2:
        raise ValueError("golden-rule initial_path must contain the two directed open segments")
    paths = tuple(coordinates(path).copy() for path in initial_path)
    if paths[0].shape[1:] != paths[1].shape[1:] or min(map(len, paths)) < 3:
        raise ValueError("golden-rule paths must have matching atom shapes and at least three nodes each")
    if not (np.allclose(paths[0][0], paths[1][-1], rtol=0, atol=1e-12)
            and np.allclose(paths[0][-1], paths[1][0], rtol=0, atol=1e-12)):
        raise ValueError("golden-rule paths must close in opposite directions at both hopping endpoints")
    mask, _ = _mask_and_masses(paths[0].shape[1:], masses, active_mask)
    masses = np.asarray(masses, dtype=float).copy()
    if any(not np.allclose(path[:, ~mask], paths[0][0, ~mask], rtol=0, atol=1e-12) for path in paths):
        raise ValueError("inactive coordinates must be identical on both golden-rule segments")
    levels = _interval_levels(bead_counts)
    if sum(levels[-1]) * int(mask.sum()) > 4096:
        raise ValueError("dense golden-rule Hessian is limited to 4096 independent active coordinates")
    points = _scan_points(levels)
    temperature = finite_positive(temperature_kelvin, "temperature_kelvin")
    denominator = KB_AU * temperature
    if denominator == 0.0:
        raise ValueError("temperature_kelvin produces a nonfinite inverse temperature in atomic units")
    beta = 1.0 / denominator
    if not math.isfinite(beta):
        raise ValueError("temperature_kelvin produces a nonfinite inverse temperature in atomic units")
    reference = finite_real(energy_reference_hartree, "energy_reference_hartree")
    tolerance = finite_positive(convergence_relative_tolerance, "convergence_relative_tolerance")
    if tolerance >= 1:
        raise ValueError("convergence_relative_tolerance must be smaller than one")
    declarations = surface_identities(potential_identity)
    providers = [_provider_identity(pes, declaration) for pes, declaration in zip(potential, declarations)]
    numerical, kinetic, declared, partition_mode = _resolve_options(
        golden_rule_options, optimize_golden_rule_instanton, golden_rule_rate)
    missing = object()
    for index, surface in enumerate(potential):
        known_coupling = getattr(surface, "coupling_hartree", missing)
        if known_coupling is not missing:
            known_coupling = finite_real(known_coupling, f"surface{index}.coupling_hartree")
            if abs(known_coupling) != abs(kinetic["coupling_hartree"]):
                raise ValueError(f"surface{index} declared coupling magnitude differs from golden_rule_options.coupling_hartree")
    partition, frequencies, partition_identity = _reactant_partition(
        reactants, declared, partition_mode, masses=masses, mask=mask,
        geometry=paths[0][0], beta=beta, reference=reference)
    identity = _portable({"schema": _SCHEMA, "mode": "golden_rule", "surfaces": providers,
                          "masses": masses, "initial_path0_bohr": paths[0], "initial_path1_bohr": paths[1],
                          "active_mask": mask, "temperature_kelvin": temperature,
                          "energy_reference_hartree": reference, "segment_interval_levels": levels,
                          "calculation_grid": points,
                          "optimizer_options": numerical, "failure_policy": "raise",
                          "rate_options": kinetic, "reactant_partition": partition_identity,
                          "convergence_relative_tolerance": tolerance,
                          "convergence_criterion": dict(_CRITERION)}, "golden-rule scientific_identity")
    report = {"schema": _SCHEMA, "mode": "golden_rule", "scientific_identity": identity,
              "physics_fingerprint": fingerprint_payload(identity), "rows": [],
              "convergence_scope": "stationarity_and_requested_segment_discretizations_only",
              "limitations": ["constant coupling, weak-coupling golden-rule and leading semiclassical approximations",
                              "normal regime; inverted and activationless limits are unsupported",
                              "bound/constrained reactant partition; no free-molecule rigid modes or gas rates",
                              "PES, Hessian-step, tolerance and reactant approximations require independent validation"]}
    solved = {}
    for p0, p1 in points:
        source = None
        if solved:
            source = min(solved, key=lambda key: abs(math.log(key[0] / p0)) + abs(math.log(key[1] / p1)))
            seed_paths = solved[source]["paths"]
            fraction = solved[source]["tau_fraction"]
        else:
            seed_paths = paths
            fraction = numerical["initial_tau_fraction"]
        seed0 = _resample_segment(seed_paths[0], p0)
        seed1 = _resample_segment(seed_paths[1], p1)
        settings = dict(numerical, initial_tau_fraction=fraction)
        try:
            orbit = optimize_golden_rule_instanton(
                potential[0], potential[1], masses, temperature, seed0, seed1,
                active_mask=mask, energy_reference_hartree=reference, **settings)
            if partition_mode == "matched":
                state = reactants[0]
                partition = harmonic_segment_log_partition(
                    frequencies, p0, p1, orbit.tau0_au, orbit.tau1_au,
                    electronic_degeneracy=state.electronic_degeneracy)
                partition -= beta * (state.energy_hartree-reference)
            rate = golden_rule_rate(orbit, log_reactant_partition=partition, **kinetic)
        except (ValueError, InstantonError) as exc:
            raise InstantonError(f"golden-rule study failed at intervals=({p0},{p1}): {exc}") from exc
        row = {"nintervals0": p0, "nintervals1": p1,
               "dtau0_au": orbit.tau0_au / p0, "dtau1_au": orbit.tau1_au / p1,
               "seed_intervals": None if source is None else list(source),
               "initial_tau_fraction_used": fraction,
               "reactant_partition": {"partition_discretization": partition_mode,
                                        "log_reactant_partition": float(partition),
                                        "energy_reference_hartree": reference,
                                        "normalization": "dimensionless_bound_or_constrained",
                                        "matched_to_optimized_segment_grid": partition_mode == "matched"},
               "orbit": orbit.to_dict(), "rate": rate}
        report["rows"].append(row)
        solved[(p0, p1)] = {"paths": (orbit.path0_bohr, orbit.path1_bohr),
                            "tau_fraction": orbit.tau1_au / (orbit.tau0_au + orbit.tau1_au),
                            "row": row}
        del orbit
    minimum = _CRITERION["minimum_segment_refinement_ratio"]

    def check(keys, column):
        return _convergence([solved[key]["row"]["rate"]["log_rate_au"] for key in keys],
                            tolerance, [key[column] for key in keys], minimum)

    joint = check(levels, 0)
    other_joint = check(levels, 1)
    joint["refinement_ratios"] = {"surface0": joint["refinement_ratios"],
                                  "surface1": other_joint["refinement_ratios"]}
    joint["sufficient_refinement_span"] &= other_joint["sufficient_refinement_span"]
    joint["converged"] &= other_joint["converged"]
    final0, final1 = levels[-1]
    segment0 = check([(p0, final1) for p0, _ in levels], 0)
    segment1 = check([(final0, p1) for _, p1 in levels], 1)
    passed = bool(joint["converged"] and segment0["converged"] and segment1["converged"])
    report.update(joint_convergence=joint, surface0_convergence=segment0,
                  surface1_convergence=segment1, numerical_converged=True,
                  discretization_converged=passed, converged=passed,
                  final_rate=solved[(final0, final1)]["row"]["rate"],
                  final_intervals=[final0, final1], number_of_optimizations=len(points),
                  imaginary_time_reoptimized_at_every_grid=True)
    if output is not None:
        write_json(output, report)
    return report
