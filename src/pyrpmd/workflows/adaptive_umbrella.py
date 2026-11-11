"""Pilot-only adaptive placement followed by an independent, frozen production.

The algorithm refines the supplied 1D/2D/3D domain; it does not discover arbitrary
unspecified basins. Missing overlap is repaired by midpoint insertion along
the shortest gaps between overlap components. Pilot length grows only when
effective sample size is insufficient, subject to an explicit cap. Optional
pilot-only stiffness tuning balances centering and overlap in each window;
production uses the frozen window-specific springs. Every decision is persisted.
An optional, explicitly bounded coverage grid also inserts windows into cells
with insufficient local reweighting ESS: graph connectivity alone cannot
guarantee coverage of a two-dimensional free-energy surface.

Only frozen-production streams enter the final MBAR/UI analysis. Adaptive
pilot samples cannot be passed off as samples from the frozen umbrella set.
"""

from __future__ import annotations

import copy
import hashlib
from math import prod
import json
from collections.abc import Mapping
from functools import partial
from pathlib import Path

import numpy as np

from ..analysis.fes import (
    CV_CORRELATION_OBSERVABLES,
    cv_statistical_inefficiency,
    solve_mbar,
)
from ..analysis.mbar_problem import umbrella_problem
from ..bias import displacement
from ..parallel import root_call, run_tasks
from .fes import resolve_umbrellas, _sample_window
from .common import (_fingerprint, checked_integer, write_json, jsonable, _dimension,
                     fingerprint_matches)
from ..units import parse_quantity

LEGACY_OVERLAP_METRIC = "mbar_symmetric_transition"
NORMALIZED_OVERLAP_METRIC = "normalized_mbar_cosine"


ADAPTIVE_FIELDS = {"enabled", "adapt_stiffness", "pilot_steps", "pilot_equilibration_steps", "max_rounds",
                   "max_windows", "max_pilot_steps", "target_overlap", "min_effective_samples", "min_spacing",
                   "coverage_bins", "coverage_ranges", "min_bin_effective_samples", "target_coverage",
                   "continuation"}


def adaptive_options(config):
    table = config.get("adaptive_umbrella", {})
    if not isinstance(table, Mapping):
        raise ValueError("adaptive_umbrella must be a table")
    unknown = set(table) - ADAPTIVE_FIELDS
    if unknown:
        raise ValueError("Unknown adaptive_umbrella fields: " + ", ".join(sorted(unknown)))
    enabled = table.get("enabled", bool(table))
    if not isinstance(enabled, bool):
        raise ValueError("adaptive_umbrella.enabled must be boolean")
    adapt_stiffness = table.get("adapt_stiffness", False)
    if not isinstance(adapt_stiffness, bool):
        raise ValueError("adaptive_umbrella.adapt_stiffness must be boolean")
    continuation = table.get("continuation", False)
    if not isinstance(continuation, bool):
        raise ValueError("adaptive_umbrella.continuation must be boolean")
    options = {"enabled": enabled, "adapt_stiffness": adapt_stiffness}
    if continuation:
        options["continuation"] = True
    for key, default, minimum in (
        ("pilot_steps", 1000, 1),
        ("pilot_equilibration_steps", config.get("sampling", {}).get("equilibration_steps", 0), 0),
        ("max_rounds", 8, 1),
        ("max_windows", 128, 1),
    ):
        options[key] = checked_integer(table.get(key, default), "adaptive_umbrella." + key, minimum)
    options["max_pilot_steps"] = checked_integer(
        table.get("max_pilot_steps", 16 * options["pilot_steps"]),
        "adaptive_umbrella.max_pilot_steps", options["pilot_steps"])
    for key, default in (("target_overlap", 0.03), ("min_effective_samples", 20.0),
                         ("min_spacing", 1e-6), ("min_bin_effective_samples", 5.0),
                         ("target_coverage", 0.95)):
        value = table.get(key, default)
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("adaptive_umbrella." + key + " must be a positive number")
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("adaptive_umbrella." + key + " must be positive and finite")
        options[key] = value
    if options["target_overlap"] >= 1:
        raise ValueError("adaptive_umbrella.target_overlap must be below one")
    if options["target_coverage"] > 1:
        raise ValueError("adaptive_umbrella.target_coverage must not exceed one")
    bins = table.get("coverage_bins")
    ranges = table.get("coverage_ranges")
    if (bins is None) != (ranges is None):
        raise ValueError("adaptive_umbrella.coverage_bins and coverage_ranges must be supplied together")
    if bins is not None:
        if not isinstance(bins, (list, tuple, np.ndarray)) or len(bins) not in (1, 2, 3):
            raise ValueError("adaptive_umbrella.coverage_bins must list one to three positive integers")
        bins = [checked_integer(n, "adaptive_umbrella.coverage_bins") for n in bins]
        if prod(bins) > 1_000_000:
            raise ValueError("adaptive_umbrella.coverage_bins must not exceed one million cells")
        if not isinstance(ranges, (list, tuple, np.ndarray)) or len(ranges) != len(bins):
            raise ValueError("adaptive_umbrella.coverage_ranges dimension must match coverage_bins")
        if any(not isinstance(pair, (list, tuple, np.ndarray)) or len(pair) != 2 for pair in ranges):
            raise ValueError("adaptive_umbrella.coverage_ranges must contain [lower, upper] pairs")
        options.update(coverage_bins=bins, coverage_ranges=[list(pair) for pair in ranges])
    if enabled:
        if continuation and (options["pilot_equilibration_steps"] < 1
                             or config.get("sampling", {}).get("equilibration_steps", 0) < 1):
            raise ValueError("umbrella continuation requires positive pilot and production equilibration_steps")
        if config.get("task", "fes") not in ("fes", "rate"):
            raise ValueError("adaptive_umbrella is supported only for FES and rate PMF workflows")
        interval = checked_integer(config.get("sampling", {}).get("sample_interval",
                    config.get("sampling", {}).get("interval_steps", 1)), "sampling.sample_interval")
        if options["pilot_steps"] // interval < 2:
            raise ValueError("adaptive_umbrella.pilot_steps must include at least two saved samples")
    return options


def _resolve_coverage_options(options, cv):
    """Coverage bounds use canonical CV units, as do analysis.ranges."""
    if "coverage_bins" not in options:
        return options
    if len(options["coverage_bins"]) != cv.dimension:
        raise ValueError("adaptive_umbrella.coverage_bins dimension must match selected CVs")
    ranges = [[parse_quantity(value, _dimension(unit)) for value in pair]
              for pair, unit in zip(options["coverage_ranges"], cv.units)]
    for (lower, upper), period in zip(ranges, cv.periods):
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError("adaptive_umbrella.coverage_ranges requires finite lower < upper")
        if period is not None and upper - lower > period * (1 + 1e-12):
            raise ValueError("adaptive_umbrella.coverage_ranges must span at most one CV period")
    return {**options, "coverage_ranges": ranges}


def validate_adaptive_umbrella(config, system):
    """Early input validation; TS-only rate CVs are resolved after preparation."""
    options = adaptive_options(config)
    if not options["enabled"]:
        return
    if config.get("task") == "rate" and not config.get("cvs"):
        return
    cv, _, centers, stiffness = resolve_umbrellas(config, system)
    _resolve_coverage_options(options, cv)
    if len(centers) > options["max_windows"]:
        raise ValueError("initial umbrella count exceeds adaptive_umbrella.max_windows")
    matrices = _window_matrices(stiffness, len(centers), cv.dimension)
    for i, c in enumerate(centers):
        delta = displacement(centers[:i], c, cv.periods)
        if len(delta) and np.any(np.einsum("ni,nij,nj->n", delta, (matrices[:i] + matrices[i]) / 2, delta) * config["beta"] < options["min_spacing"]**2):
            raise ValueError("adaptive umbrella initial centers are duplicate or closer than min_spacing")


def _window_matrices(stiffness, count, dimension):
    return np.broadcast_to(np.asarray(stiffness, float), (count, dimension, dimension)).copy()


def interpolate_stiffness(centers, matrices, targets, periods):
    """Positive matrix interpolation for newly inserted windows, in CV units."""
    centers, targets = np.asarray(centers, float), np.asarray(targets, float)
    matrices = _window_matrices(matrices, len(centers), centers.shape[1])
    metric = matrices.mean(axis=0)
    result = []
    for target in targets:
        delta = displacement(centers, target, periods)
        distance = np.einsum("ni,ij,nj->n", delta, metric, delta)
        nearest = np.argsort(distance)[:2]
        if distance[nearest[0]] < 1e-24 or len(nearest) == 1:
            result.append(matrices[nearest[0]])
        else:
            weight = 1 / np.sqrt(distance[nearest])
            result.append(np.average(matrices[nearest], axis=0, weights=weight))
    return np.asarray(result)


def _seed(config, phase, round_index=0):
    return int(np.random.SeedSequence([int(config.get("seed", 0)), 823, phase, round_index]).generate_state(1)[0])


def _stage_config(config, centers, stiffness, seed, *, pilot=None):
    stage = copy.deepcopy(config)
    stage.pop("adaptive_umbrella", None)
    stage["seed"] = seed
    umbrella = stage.setdefault("umbrella", {})
    for key in ("grid", "centers", "start", "stop", "count", "kappa", "stiffness",
                "regions", "windows", "force_constant", "force_matrix"):
        umbrella.pop(key, None)
    umbrella["centers_au"] = np.asarray(centers).tolist()
    umbrella["stiffness_au"] = np.asarray(stiffness).tolist()
    if pilot is not None:
        stage["sampling"]["equilibration_steps"] = pilot["equilibration_steps"]
        stage["sampling"]["production_steps"] = pilot["production_steps"]
        stage.setdefault("analysis", {})["bootstrap"] = 0
        stage["analysis"]["bootstrap_replicates"] = 0
        stage["analysis"]["bootstrap_samples"] = 0
    return stage


def _components(count, edges):
    parent = list(range(count))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        if a == b:
            return False
        parent[max(a, b)] = min(a, b)
        return True
    for i, j in edges:
        union(i, j)
    return find, union


def _plan_overlap_metric(plan):
    """A saved plan owns its diagnostic, including legacy transition scores."""
    if plan.get("schema_version", 1) < 3:
        return LEGACY_OVERLAP_METRIC
    metric = plan.get("overlap_metric")
    if metric != NORMALIZED_OVERLAP_METRIC:
        raise ValueError("unsupported adaptive umbrella overlap metric")
    return metric


def _overlap_scores(overlap, metric):
    """Compare MBAR weight columns without dilution by the number of states.

    O_ij = N_j (W_i . W_j).  Dividing its symmetric score by the geometric
    mean of the diagonals gives the cosine of the two weight columns.  It is
    one for identical sampled distributions, even with many nearby windows
    or unequal sample counts, and zero for disjoint empirical support.  This
    remains a local sampling diagnostic, not an equilibration certificate.
    """
    overlap = np.asarray(overlap, float)
    scores = np.sqrt(np.maximum(0, overlap * overlap.T))
    if metric == LEGACY_OVERLAP_METRIC:
        return scores
    if metric != NORMALIZED_OVERLAP_METRIC:
        raise ValueError("unsupported adaptive umbrella overlap metric")
    diagonal = np.diag(overlap)
    normalization = np.sqrt(diagonal[:, None] * diagonal[None, :])
    return np.clip(np.divide(scores, normalization, out=np.zeros_like(scores),
                             where=normalization > 0), 0, 1)


def _overlap_diagnostic(overlap, target, metric):
    scores = _overlap_scores(overlap, metric)
    count = len(scores)
    pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
    strong = [(i, j) for i, j in pairs if scores[i, j] >= target]
    find, _ = _components(count, strong)
    labels = [find(i) for i in range(count)]
    _, tree_union = _components(count, [])
    tree_scores = []
    for i, j in sorted(pairs, key=lambda p: (-scores[p[0], p[1]], p)):
        if tree_union(i, j):
            tree_scores.append(float(scores[i, j]))
    return {
        "overlap_metric": metric,
        "overlap_connected_at_target": len(set(labels)) == 1,
        "overlap_components": labels,
        "overlap": np.asarray(overlap).tolist(),
        "symmetric_overlap": scores.tolist(),
        "spanning_overlap": min(tree_scores, default=1.0),
    }, scores, pairs, strong


def _coverage_diagnostic(samples, weights, frame_inefficiency, periods, options):
    """Local Kish ESS with a conservative per-stream CV correlation factor.

    This is a pilot support diagnostic, not a free-energy error estimate: it
    does not account for uncertainty in fitted MBAR offsets or unseen basins.
    Every declared grid cell has equal importance, including zero-count cells.
    """
    edges = [np.linspace(lower, upper, n + 1)
             for n, (lower, upper) in zip(options["coverage_bins"], options["coverage_ranges"])]
    values = np.array(samples, copy=True)
    for axis, period in enumerate(periods):
        if period is not None:
            lower = edges[axis][0]
            values[:, axis] = lower + (values[:, axis] - lower) % period
    counts, _ = np.histogramdd(values, edges)
    mass, _ = np.histogramdd(values, edges, weights=weights)
    variance, _ = np.histogramdd(values, edges, weights=weights**2 * frame_inefficiency)
    ess = np.divide(mass**2, variance, out=np.zeros_like(mass), where=variance > 0)
    supported = ess >= options["min_bin_effective_samples"]
    fraction = float(supported.mean())
    return {
        "bins": options["coverage_bins"], "ranges": options["coverage_ranges"],
        "effective_samples": ess.tolist(), "sample_counts": counts.astype(int).tolist(),
        "min_bin_effective_samples": options["min_bin_effective_samples"],
        "fraction": fraction, "target": options["target_coverage"],
        "target_met": bool(fraction >= options["target_coverage"]),
        "note": "Local Kish ESS corrected by per-stream CV autocorrelation; not a physical convergence certificate.",
    }, edges, ess, supported


def diagnose_and_refine(streams, centers, stiffness, beta, periods, options, *,
                        overlap_metric=NORMALIZED_OVERLAP_METRIC):
    """Deterministic refinement from actual pilot samples, independently testable."""
    # Grouping visits the results once per window; executor.map may be a
    # one-shot iterator, so consume it only here and retain all pilot streams.
    streams = list(streams)
    centers = np.asarray(centers, float)
    count = len(centers)
    matrices = _window_matrices(stiffness, count, centers.shape[1])
    metric = matrices.mean(axis=0)
    groups = [[s for s in streams if s["window"] == i] for i in range(count)]
    if any(not group for group in groups):
        raise ValueError("pilot results must contain every umbrella window")
    samples = [np.concatenate([s["samples"] for s in group]) for group in groups]
    stream_g = [[cv_statistical_inefficiency(s["samples"], periods) for s in group] for group in groups]
    ess = [sum(len(s["samples"]) / g for s, g in zip(group, factors))
           for group, factors in zip(groups, stream_g)]
    joined = np.concatenate(samples)
    reduced = umbrella_problem(joined, centers, matrices, beta, periods)
    mbar = solve_mbar(reduced, [len(x) for x in samples], return_state_weights=False)
    if not mbar.converged:
        raise RuntimeError(f"adaptive pilot MBAR failed its equation residual: {mbar.residual:g}")
    overlap_report, scores, pairs, strong = _overlap_diagnostic(
        mbar.overlap, options["target_overlap"], overlap_metric)
    _, union = _components(count, strong)
    connected = overlap_report["overlap_connected_at_target"]
    missing = []
    # Select shortest geometric links *between* sampled-overlap components,
    # preserving periodic distances and mixed-CV units via beta*K.
    def separation(pair):
        i, j = pair
        delta = displacement(centers[j], centers[i], periods)
        return float(beta * delta @ ((matrices[i] + matrices[j]) / 2) @ delta)
    for i, j in sorted(pairs, key=lambda pair: (separation(pair), pair)):
        if union(i, j):
            missing.append((i, j))
    proposed = centers.tolist()
    inserted = []
    rejected = []
    for i, j in missing:
        midpoint = centers[i] + 0.5 * displacement(centers[j], centers[i], periods)
        delta = displacement(np.asarray(proposed), midpoint, periods)
        distances = np.sqrt(np.maximum(0, beta * np.einsum("ni,ij,nj->n", delta, metric, delta)))
        reason = None
        if len(proposed) >= options["max_windows"]:
            reason = "max_windows"
        elif distances.min() <= options["min_spacing"]:
            reason = "min_spacing"
        if reason:
            rejected.append({"between": [i, j], "reason": reason})
        else:
            proposed.append(midpoint.tolist())
            inserted.append({"between": [i, j], "center": midpoint.tolist(), "overlap": float(scores[i, j])})
    coverage = None
    if "coverage_bins" in options:
        frame_g = np.concatenate([np.full(len(s["samples"]), g)
                                  for group, factors in zip(groups, stream_g)
                                  for s, g in zip(group, factors)])
        coverage, edges, bin_ess, supported = _coverage_diagnostic(
            joined, mbar.weights, frame_g, periods, options)
        if not coverage["target_met"]:
            # Put at most one candidate at each deficient cell's center. If a
            # center already exists, subsequent pilots get longer instead of
            # piling up almost-identical windows. The user controls spatial
            # resolution separately from the final analysis histogram.
            candidates = [tuple(index) for index in np.argwhere(~supported)]
            candidates.sort(key=lambda index: (bin_ess[index], index))
            for index in candidates:
                center = np.array([(edge[i] + edge[i + 1]) / 2 for edge, i in zip(edges, index)])
                delta = displacement(np.asarray(proposed), center, periods)
                distances = np.sqrt(np.maximum(0, beta * np.einsum("ni,ij,nj->n", delta, metric, delta)))
                if distances.min() <= options["min_spacing"]:
                    reason = "min_spacing"
                elif len(proposed) >= options["max_windows"]:
                    reason = "max_windows"
                else:
                    reason = None
                detail = {"reason": "coverage", "bin": list(index), "center": center.tolist(),
                          "effective_samples": float(bin_ess[index])}
                if reason:
                    rejected.append({**detail, "reason": reason, "criterion": "coverage"})
                else:
                    proposed.append(center.tolist())
                    inserted.append(detail)
    enough = bool(np.all(np.asarray(ess) >= options["min_effective_samples"]))
    return {
        "target_met": bool(connected and enough and (coverage is None or coverage["target_met"])),
        **overlap_report,
        "effective_samples_per_window": ess, "effective_samples_target_met": enough,
        "effective_sample_observables": CV_CORRELATION_OBSERVABLES,
        "samples_per_window": [len(x) for x in samples], "mbar_residual": float(mbar.residual),
        "inserted": inserted, "rejected_insertions": rejected,
        "next_centers": proposed,
        **({"coverage": coverage} if coverage is not None else {}),
    }


def _plan_path(output):
    return Path(output) / "adaptive-plan.json"


def _plan_checksum(plan):
    payload = {key: value for key, value in plan.items() if key != "checksum"}
    return hashlib.sha256(json.dumps(jsonable(payload), sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _save_plan(output, plan):
    plan["checksum"] = _plan_checksum(plan)
    write_json(_plan_path(output), plan)


def extend_adaptive_budget(output, limits):
    """Explicitly reopen an exhausted pilot plan; never change frozen data.

    The run's input/physics fingerprint stays unchanged. Monotonic budget
    overrides are recorded in the checksummed plan, which owns the resolved
    pilot schedule. All completed round checkpoints remain authoritative.
    """
    fields = {"max_rounds", "max_windows", "max_pilot_steps"}
    if not isinstance(limits, Mapping) or not limits or set(limits) - fields:
        raise ValueError("adaptive budget accepts only max_rounds, max_windows and max_pilot_steps")
    path = _plan_path(output)
    plan = json.loads(path.read_text())
    if plan.get("checksum") != _plan_checksum(plan):
        raise ValueError("adaptive umbrella plan checksum mismatch")
    if plan["status"] != "incomplete":
        raise ValueError("adaptive budget can only extend an incomplete pilot plan")
    options = dict(plan["options"])
    for key, value in limits.items():
        if key not in options:
            raise ValueError(f"saved adaptive plan does not support {key}")
        options[key] = checked_integer(value, "adaptive_umbrella." + key, options[key])
    if options == plan["options"] or options["max_rounds"] <= plan["next_round"]:
        raise ValueError("increase the exhausted budget and allow at least one further pilot round")
    plan.setdefault("budget_extensions", []).append({
        "after_rounds": plan["next_round"], "previous": plan["options"], "updated": options,
        "previous_failure_reason": plan.get("failure_reason"),
    })
    plan["options"] = options
    last = plan["rounds"][-1]
    plan["centers"] = last["next_centers"]
    if "stiffness_update" in last:
        update = last["stiffness_update"]
        plan["stiffness"] = update["next_stiffness"]
        if plan.get("schema_version", 1) >= 4:
            plan["initial_stiffness"] = update["next_initial_stiffness"]
        for key in ("stiffness_scale", "stiffness_floor_scale"):
            if key in update:
                plan[key] = update[key]
    if plan.get("schema_version", 1) >= 2:
        plan["next_pilot_steps"] = _next_pilot_steps(last["pilot"]["production_steps"], last, options)
    plan["status"] = "adapting"
    plan.pop("failure_reason", None)
    _save_plan(output, plan)


def _open_plan(config, system, output, centers, stiffness, options, create=True):
    path = _plan_path(output)
    identity = _fingerprint(config, system, {"adaptive_umbrella_schema": 1})
    if path.exists():
        plan = json.loads(path.read_text())
        if plan.get("checksum") != _plan_checksum(plan):
            raise ValueError("adaptive umbrella plan checksum mismatch")
        if not fingerprint_matches(plan.get("fingerprint"), config, system, {"adaptive_umbrella_schema": 1}):
            raise ValueError("adaptive umbrella plan physics differs; choose a fresh output directory")
        if plan.get("schema_version", 1) not in (1, 2, 3, 4):
            raise ValueError("unsupported adaptive umbrella plan schema version")
        _plan_overlap_metric(plan)
        return plan
    if not create:
        raise ValueError("no saved adaptive umbrella plan is available for reanalysis")
    plan = {"schema_version": 3, "fingerprint": identity, "status": "adapting", "next_round": 0,
            "overlap_metric": NORMALIZED_OVERLAP_METRIC,
            "centers": centers.tolist(), "stiffness": stiffness.tolist(), "options": options,
            "next_pilot_steps": options["pilot_steps"],
            "rounds": [], "production_seed": _seed(config, 2), "pilot_samples_used_in_production": 0}
    if options.get("adapt_stiffness") or np.asarray(stiffness).ndim == 3:
        plan.update(schema_version=4, stiffness_policy="per_window_moments",
                    stiffness=_window_matrices(stiffness, len(centers), centers.shape[1]).tolist())
        plan["initial_stiffness"] = copy.deepcopy(plan["stiffness"])
    _save_plan(output, plan)
    return plan


def _next_pilot_steps(current, report, options):
    """Refining placement alone must not double every trajectory's duration."""
    coverage_needs_sampling = (
        not report.get("coverage", {}).get("target_met", True)
        and not report.get("inserted")
    )
    if report["effective_samples_target_met"] and not coverage_needs_sampling:
        return current
    # Short-trace ESS is too noisy to extrapolate a fractional increase: that
    # can cost an extra round of equilibration and a complete repeated pilot.
    return min(options["max_pilot_steps"], 2 * current)


def _stiffness_decision(streams, centers, stiffness, beta, periods, report, plan):
    """Tune one spring from discarded pilots, using dimensionless distances.

    A factor of two bounds each update, and the total scale stays within
    [1/16, 16] of the initial spring. Once a spring fails centering, the next
    stronger spring is a permanent floor: subsequent softening cannot cycle
    back to that failed value. Window insertion still repairs remaining gaps.
    """
    centers, stiffness = np.asarray(centers), np.asarray(stiffness)
    scale = plan.get("stiffness_scale", 1.)
    floor = plan.get("stiffness_floor_scale", 1 / 16)
    shifts, widths = [], []
    for i, center in enumerate(centers):
        neighbors = displacement(np.delete(centers, i, axis=0), center, periods)
        if not len(neighbors):
            continue
        spacing = np.sqrt(np.min(beta * np.einsum("ni,ij,nj->n", neighbors, stiffness, neighbors)))
        samples = np.concatenate([s["samples"] for s in streams if s["window"] == i])
        delta = displacement(samples, center, periods)
        mean = delta.mean(axis=0)
        shifts.append(float(np.sqrt(beta * mean @ stiffness @ mean) / spacing))
        centered = delta - mean
        widths.append(float(np.sqrt(np.mean(beta * np.einsum(
            "ni,ij,nj->n", centered, stiffness, centered))) / spacing))
    enough = report["effective_samples_target_met"]
    centered = not shifts or max(shifts) <= .5
    next_scale, reason = scale, "centering_and_width_acceptable"
    if not enough:
        reason = "insufficient_pilot_effective_samples"
    elif not centered:
        next_scale = min(16., 2 * scale)
        floor = max(floor, next_scale)
        reason = "increase_for_center_displacement" if next_scale > scale else "stiffness_limit"
    elif (widths and not report["overlap_connected_at_target"]
          and max(widths) < .3 and max(shifts) < .25):
        next_scale = max(floor, .5 * scale)
        reason = "soften_narrow_disconnected_windows" if next_scale < scale else "centering_floor"
    factor = next_scale / scale
    return {"changed": factor != 1., "factor": factor, "reason": reason,
            "centering_target_met": centered,
            "displacement_per_spacing": shifts, "width_per_spacing": widths,
            "next_stiffness": (stiffness * factor).tolist(),
            "stiffness_scale": next_scale, "stiffness_floor_scale": floor}


def _window_stiffness_decision(streams, centers, stiffness, beta, periods, report, plan):
    """Infer local curvature from biased moments, and tune each spring alone.

    In one dimension, F''(mean) = 1/(beta*variance) - K. Targeting a width
    comparable to the neighboring spacing compensates a negative barrier
    curvature while allowing softer springs in flat regions. The mean shift
    supplies a separate lower bound on K in steep regions. Updates require
    effective pilot samples, are damped/bounded, and never change production.
    """
    centers = np.asarray(centers, float)
    count, dimension = centers.shape
    matrices = _window_matrices(stiffness, count, dimension)
    initial = _window_matrices(plan["initial_stiffness"], count, dimension)
    updated = matrices.copy()
    records = []
    for i, (center, matrix) in enumerate(zip(centers, matrices)):
        neighbors = displacement(np.delete(centers, i, axis=0), center, periods)
        spacing = (float(np.sqrt(np.min(np.einsum("ni,ij,nj->n", neighbors, matrix, neighbors))))
                   if len(neighbors) else float(np.sqrt(1 / beta)))
        values = np.concatenate([s["samples"] for s in streams if s["window"] == i])
        delta = displacement(values, center, periods)
        mean = delta.mean(axis=0)
        covariance = np.atleast_2d(np.cov(delta, rowvar=False, ddof=0))
        shift = float(np.sqrt(mean @ matrix @ mean) / spacing)
        width = float(np.sqrt(np.trace(matrix @ covariance)) / spacing)
        ess = report["effective_samples_per_window"][i]
        enough = ess >= plan["options"]["min_effective_samples"]
        centered = shift <= max(.5, 2 * width / np.sqrt(max(1., ess)))
        factor, curvature, reason = 1., None, "width_and_center_acceptable"
        if dimension == 1 and covariance[0, 0] > 0:
            curvature = float(1 / (beta * covariance[0, 0]) - matrix[0, 0])
        if not plan["options"].get("adapt_stiffness"):
            centered, reason = True, "fixed_window_spring"
        elif not enough:
            reason = "insufficient_pilot_effective_samples"
        elif not centered:
            factor, reason = min(2., max(1.25, shift / .35)), "increase_for_center_displacement"
        elif width > 1.5:
            factor, reason = min(2., (width / .8)**2), "stabilize_broad_window"
        elif width < .35:
            if curvature is not None:
                distance = spacing / np.sqrt(matrix[0, 0])
                desired = 1 / (beta * (.6 * distance)**2) - curvature
                # Do not soften below the restraint needed on a steep slope.
                desired = max(desired, matrix[0, 0] * shift / .4)
                factor = float(np.clip(desired / matrix[0, 0], .5, 1.))
            else:
                factor = .5
            reason = "soften_local_curvature" if factor < .8 else "slope_limits_softening"
            if factor >= .8:
                factor = 1.
        scale = float(np.trace(matrix) / np.trace(initial[i]))
        factor = float(np.clip(scale * factor, 1 / 64, 256) / scale)
        updated[i] = matrix * factor
        records.append({"window": i, "factor": factor, "reason": reason,
                        "centering_target_met": centered,
                        "displacement_per_spacing": shift, "width_per_spacing": width,
                        "effective_samples": float(ess), "local_curvature": curvature,
                        "stiffness_before": matrix.tolist(), "stiffness_after": updated[i].tolist()})
    targets = report["next_centers"]
    return {"policy": "per_window_moments", "windows": records,
            "changed": any(r["factor"] != 1 for r in records),
            "centering_target_met": all(r["centering_target_met"] for r in records),
            "next_stiffness": interpolate_stiffness(centers, updated, targets, periods).tolist(),
            "next_initial_stiffness": interpolate_stiffness(centers, initial, targets, periods).tolist()}


def _record_round(plan, streams, config, system, output, cv, options, pilot):
    streams = list(streams)
    report = diagnose_and_refine(streams, np.asarray(plan["centers"]), np.asarray(plan["stiffness"]),
                                config["beta"], cv.periods, options,
                                overlap_metric=_plan_overlap_metric(plan))
    stiffness_changed = False
    if options.get("adapt_stiffness", False) or plan.get("schema_version", 1) >= 4:
        tune = (_window_stiffness_decision if plan.get("schema_version", 1) >= 4
                else _stiffness_decision)
        update = tune(streams, plan["centers"], plan["stiffness"],
                                     config["beta"], cv.periods, report, plan)
        report["stiffness_update"] = update
        report["stiffness"] = copy.deepcopy(plan["stiffness"])
        stiffness_changed = update["changed"]
        report["target_met"] &= update["centering_target_met"] and not stiffness_changed
    round_index = plan["next_round"]
    decision = {"round": round_index, "seed": _seed(config, 1, round_index),
                "centers": plan["centers"], "pilot": pilot, **report}
    plan["rounds"].append(decision)
    plan["next_round"] += 1
    if plan.get("schema_version", 1) >= 2:
        plan["next_pilot_steps"] = _next_pilot_steps(pilot["production_steps"], report, options)
    if report["target_met"]:
        plan["status"] = "frozen"
        plan["target_met"] = True
    elif plan["next_round"] >= options["max_rounds"]:
        plan["status"] = "incomplete"
        targets = "overlap/effective-sample" + ("/coverage" if "coverage" in report else "")
        plan["failure_reason"] = "max_rounds before " + targets + " targets"
    elif ("stiffness_update" in report
          and report["stiffness_update"].get("reason") == "stiffness_limit"):
        plan["status"] = "incomplete"
        plan["failure_reason"] = "stiffness_limit before centering target"
    elif not report["overlap_connected_at_target"] and not report["inserted"] and not stiffness_changed:
        plan["status"] = "incomplete"
        plan["failure_reason"] = "no admissible midpoint: max_windows or min_spacing"
    elif (plan.get("schema_version", 1) >= 2
          and (not report["effective_samples_target_met"]
               or not report.get("coverage", {}).get("target_met", True))
          and pilot["production_steps"] >= options["max_pilot_steps"]
          and not report["inserted"] and not stiffness_changed):
        plan["status"] = "incomplete"
        target = ("effective-sample" if not report["effective_samples_target_met"] else "coverage")
        plan["failure_reason"] = "max_pilot_steps before " + target + " target"
    else:
        plan["centers"] = report["next_centers"]
        if "stiffness_update" in report:
            plan["stiffness"] = update["next_stiffness"]
            if plan.get("schema_version", 1) >= 4:
                plan["initial_stiffness"] = update["next_initial_stiffness"]
            for key in ("stiffness_scale", "stiffness_floor_scale"):
                if key in update:
                    plan[key] = update[key]
    _save_plan(output, plan)
    write_json(Path(output) / "adaptation" / f"round_{round_index:03d}" / "diagnostics.json", decision)
    return plan


def _production_diagnostic(result, plan, periods):
    """Check the frozen trajectories using the already computed MBAR fit.

    FES fits one state per stream; pilots pool replicas at the same window.
    Identical-bias rows have identical MBAR weight columns, so selecting one
    row and summing destination-replica columns exactly recovers window-level
    overlap without another solve. Frame weights and correlation factors keep
    the original stream order for independent production coverage checks.
    """
    data, streams = result["data"], result["streams"]
    options = plan["options"]
    count = len(plan["centers"])
    groups = [[i for i, stream in enumerate(streams) if stream["window"] == window]
              for window in range(count)]
    if any(not group for group in groups) or sum(map(len, groups)) != len(streams):
        raise ValueError("production results must contain every frozen umbrella window")
    stream_overlap = np.asarray(data["overlap"])
    overlap = np.array([[stream_overlap[group[0], other].sum() for other in groups]
                        for group in groups])
    report, _, _, _ = _overlap_diagnostic(
        overlap, options["target_overlap"], _plan_overlap_metric(plan))
    # The full stream-level matrix is already available in fes.npz/fes.h5.
    # Keep the summary compact rather than repeating two quadratic matrices.
    report.pop("overlap")
    report.pop("symmetric_overlap")
    stream_ess = np.asarray(data["effective_samples_per_stream"])
    window_ess = np.array([stream_ess[group].sum() for group in groups])
    enough = bool(np.all(window_ess >= options["min_effective_samples"]))
    coverage = None
    if "coverage_bins" in options:
        samples = np.concatenate([stream["samples"] for stream in streams])
        frame_g = np.repeat(data["statistical_inefficiency"],
                            [len(stream["samples"]) for stream in streams])
        coverage, _, _, _ = _coverage_diagnostic(
            samples, data["weights"], frame_g, periods, options)
        coverage["scope"] = "frozen_production"
    report.update(
        scope="frozen_production",
        effective_samples_per_window=window_ess.tolist(),
        effective_samples_target_met=enough,
        analysis_overlap_connected=bool(data["connected"]),
        target_met=bool(data["connected"] and report["overlap_connected_at_target"]
                        and enough and (coverage is None or coverage["target_met"])),
    )
    if coverage is not None:
        report["coverage"] = coverage
    return report


def _production_refinement(plan, production, periods):
    """Propose a fresh pilot from production evidence without changing this plan."""
    centers = np.asarray(plan["centers"], float)
    options = plan["options"]
    stiffness = np.asarray(plan.get("stiffness", np.eye(centers.shape[1])))
    labels = production["overlap_components"]
    groups = [(i, j) for i in range(len(centers)) for j in range(i + 1, len(centers))
              if labels[i] == labels[j]]
    _, union = _components(len(centers), groups)
    pairs = [(i, j) for i in range(len(centers)) for j in range(i + 1, len(centers))]
    def distance(pair):
        delta = displacement(centers[pair[1]], centers[pair[0]], periods)
        return float(delta @ stiffness @ delta), pair
    inserted = []
    for i, j in sorted(pairs, key=distance):
        if union(i, j):
            inserted.append((centers[i] + 0.5 * displacement(centers[j], centers[i], periods)).tolist())
    return {
        "scope": "next_run_only", "applied": False, "requires_fresh_output": True,
        "source": "frozen_production", "frozen_plan_modified": False,
        "suggested_additional_centers": inserted,
        "windows_needing_more_effective_samples": [
            i for i, ess in enumerate(production["effective_samples_per_window"])
            if ess < options["min_effective_samples"]],
        "coverage_refinement_needed": not production.get("coverage", {}).get("target_met", True),
        "diagnostic": production,
        "policy": "Start a new pilot plan; recheck window limits, overlap, ESS and declared coverage. Do not append windows to frozen production.",
    }


def _annotate(result, plan, output, production_steps=None, *, periods=None):
    summary = result["summary"]
    production = _production_diagnostic(result, plan, periods)
    # Independent production can mix worse than an accepted pilot. Recheck
    # all configured targets, including domain coverage, on these samples.
    summary["adaptive_umbrella"] = {
        "status": "frozen", "target_met": True, "pilot_rounds": len(plan["rounds"]),
        "target_scope": "discarded_adaptation_pilots", "pilot_target_met": True,
        "overlap_metric": production["overlap_metric"],
        "frozen_windows": len(plan["centers"]), "pilot_samples_used_in_production": 0,
        "stiffness_policy": plan.get("stiffness_policy", "shared"),
        "frozen_stiffness": plan.get("stiffness"),
        "production_seed": plan["production_seed"], "plan": str(_plan_path(output)),
        "production_target_met": production["target_met"],
        "production_overlap_connected_at_target": production["overlap_connected_at_target"],
        "production_overlap_components": production["overlap_components"],
        "production_spanning_overlap": production["spanning_overlap"],
        "production_analysis_overlap_connected": production["analysis_overlap_connected"],
        "production_effective_samples_per_window": production["effective_samples_per_window"],
        "production_effective_samples_target_met": production["effective_samples_target_met"],
        "final_pilot_steps": plan["rounds"][-1]["pilot"]["production_steps"],
        "production_steps_per_stream": production_steps,
        "target_note": "target_met is a compatibility alias for pilot_target_met; production_target_met independently checks frozen-production overlap/ESS and optional coverage. Neither certifies equilibration or physical convergence.",
    }
    if "coverage" in plan["rounds"][-1]:
        summary["adaptive_umbrella"]["coverage"] = plan["rounds"][-1]["coverage"]
        summary["adaptive_umbrella"]["coverage_scope"] = "discarded_adaptation_pilots"
        summary["adaptive_umbrella"]["pilot_coverage"] = plan["rounds"][-1]["coverage"]
    if "coverage" in production:
        summary["adaptive_umbrella"]["production_coverage"] = production["coverage"]
    summary["adaptive_umbrella"]["effective_sample_observables"] = plan["rounds"][-1].get(
        "effective_sample_observables", "Legacy saved pilot: CV only, or first circular harmonics")
    if "effective_sample_observables" not in plan["rounds"][-1]:
        summary["warnings"].append("This saved frozen plan used legacy coordinate-only pilot ESS; "
                                   "use a fresh output directory to evaluate amplitude-sensitive pilot criteria.")
    summary["warnings"].append("Adaptive pilots were discarded before fixed-umbrella production; pilot overlap/ESS is not a physical convergence certificate.")
    if production["overlap_metric"] == LEGACY_OVERLAP_METRIC:
        summary["warnings"].append("This saved plan retains legacy MBAR transition overlap scores; a fresh output directory uses normalized MBAR cosine overlap, whose thresholds have a different numerical meaning.")
    if not production["effective_samples_target_met"]:
        summary["warnings"].append("Frozen production has windows below adaptive_umbrella.min_effective_samples; increase sampling.production_steps and check independent replicas.")
    if not production["overlap_connected_at_target"]:
        summary["warnings"].append("Frozen production does not meet adaptive_umbrella.target_overlap; an accepted pilot does not establish production overlap.")
    if "coverage" in production and not production["coverage"]["target_met"]:
        summary["warnings"].append("Frozen production does not meet the declared adaptive umbrella coverage target; pilot coverage is reported separately and cannot establish production support.")
    if production_steps is not None and production_steps < plan["rounds"][-1]["pilot"]["production_steps"]:
        summary["warnings"].append("Production is shorter than the final accepted pilot; pilot success does not establish adequate production statistics.")
    if not production["target_met"]:
        refinement_path = Path(output) / "production-refinement.json"
        write_json(refinement_path, _production_refinement(plan, production, periods))
        summary["adaptive_umbrella"]["next_run_refinement"] = str(refinement_path)
    # The FES writer persists this annotated summary in all canonical files.
    # Writing only summary.json here would leave the NPZ metadata stale.
    return result


def _require_frozen(plan, output):
    if plan["status"] != "frozen":
        summary = {"task": "fes", "status": "adaptation_incomplete", "converged": False,
                   "failure_reason": plan.get("failure_reason", "pilot stage incomplete"),
                   "production_started": False, "plan": str(_plan_path(output))}
        if plan["rounds"]:
            last = plan["rounds"][-1]
            summary.update(pilot_rounds=len(plan["rounds"]), windows=len(plan["centers"]),
                           spanning_overlap=last["spanning_overlap"],
                           minimum_effective_samples=min(last["effective_samples_per_window"]),
                           targets=plan["options"])
        write_json(Path(output) / "summary.json", summary)
        raise RuntimeError("Adaptive umbrella targets were not reached; production was not started. " +
                           summary["failure_reason"] + ". Inspect " + str(_plan_path(output)))


def _save_continuation(config, system, output, initializations):
    write_json(Path(output) / "continuation.json", {
        "schema_version": 1,
        "fingerprint": _fingerprint(config, system, {"umbrella_continuation": 1}),
        "policy": "coordinates_only; independent replica RNG/momenta and full equilibration",
        "initializations": initializations,
    })


def _continuation_pilot(tasks, cv, config, system, output, executor):
    """Complete each neighboring pilot window before seeding the next one."""
    from .fes import _read_continuation

    initializations = root_call(_read_continuation, config, system, output) or {}
    pending = sorted({task["window"] for task in tasks})
    centers = {task["window"]: np.asarray(task["center"]) for task in tasks}
    stiffness = np.asarray(tasks[0]["stiffness"])
    reference = cv.value(np.repeat(system.geometry[None], config["beads"], axis=0))
    completed, sources = [], {}
    while pending:
        candidates = [(i, j) for i in pending for j in sources] if sources else [(i, None) for i in pending]
        def distance(pair):
            i, j = pair
            delta = displacement(centers[i], reference if j is None else centers[j], cv.periods)
            return float(delta @ stiffness @ delta), i, -1 if j is None else j
        window, source = min(candidates, key=distance)
        if str(window) not in initializations:
            initialization = {"source": "reference_geometry"}
            if source is not None:
                stream = sources[source]
                initialization = {
                    "source": "neighbor_pilot", "source_window": source,
                    "source_replica": stream["replica"], "source_center": centers[source].tolist(),
                    "source_checkpoint": str(Path(stream["checkpoint"]).relative_to(output)),
                    "source_elapsed_steps": config["sampling"]["equilibration_steps"] + config["sampling"]["production_steps"],
                    "q": np.asarray(stream["last_q"]).tolist(),
                }
            initializations[str(window)] = initialization
            root_call(_save_continuation, config, system, output, initializations)
        group = [{**task, "initialization": initializations[str(window)]}
                 for task in tasks if task["window"] == window]
        streams = (executor.map(_sample_window, group) if executor is not None else
                   run_tasks(_sample_window, group, config.get("parallel", {})))
        streams = root_call(list, streams)
        completed.extend(streams)
        sources[window] = min(streams, key=lambda stream: stream["replica"])
        pending.remove(window)
    return sorted(completed, key=lambda stream: (stream["window"], stream["replica"]))


def _production_continuation(config, system, frozen, plan, output):
    """Freeze final-pilot coordinate seeds before independent production starts."""
    from .fes import _make_tasks, _read_continuation, _read_window_checkpoint, _window_checkpoint

    if _read_continuation(frozen, system, output) is not None:
        return
    last = plan["rounds"][-1]
    stage = _stage_config(config, last["centers"], plan["stiffness"], last["seed"], pilot=last["pilot"])
    pilot_output = Path(output) / "adaptation" / f"round_{last['round']:03d}"
    cv, tasks = _make_tasks(stage, system, pilot_output)
    initializations = {}
    total = last["pilot"]["equilibration_steps"] + last["pilot"]["production_steps"]
    for task in tasks:
        if task["replica"] != 0:
            continue
        saved = _read_window_checkpoint(task, cv)
        if saved["elapsed"] != total:
            raise ValueError("continuation requires completed pilot checkpoints")
        initializations[str(task["window"])] = {
            "source": "frozen_pilot", "source_round": last["round"],
            "source_window": task["window"], "source_replica": 0,
            "source_center": np.asarray(task["center"]).tolist(), "source_elapsed_steps": total,
            "source_checkpoint": str(_window_checkpoint(task).relative_to(output)),
            "q": saved["state"].q.tolist(),
        }
    _save_continuation(frozen, system, output, initializations)


def sample_adaptive_umbrellas(config, system, output, executor=None):
    """Adapt pilot windows, freeze, then start independent production samples."""
    from . import validate_bond_constraint_workflow
    validate_bond_constraint_workflow(system, "adaptive_fes")
    from .sampling import validate_andersen_schedule
    validate_andersen_schedule(config, "adaptive_umbrella")
    from .fes import sample_umbrellas
    validate_adaptive_umbrella(config, system)
    options = adaptive_options(config)
    cv, specs, centers, stiffness = resolve_umbrellas(config, system)
    options = _resolve_coverage_options(options, cv)
    plan = root_call(_open_plan, config, system, str(output), centers, stiffness, options)
    # Saved plans own their resolved defaults and schedule, including v1's
    # geometric schedule. Upgrading the package must not change an active run.
    options = plan["options"]
    while plan["status"] == "adapting":
        index = plan["next_round"]
        pilot = {"equilibration_steps": options["pilot_equilibration_steps"],
                 "production_steps": plan.get("next_pilot_steps", options["pilot_steps"] * 2**index)}
        stage = _stage_config(config, plan["centers"], plan["stiffness"], _seed(config, 1, index), pilot=pilot)
        pilot_output = Path(output) / "adaptation" / f"round_{index:03d}"
        replicas = checked_integer(config.get("sampling", {}).get("replicas", 1), "sampling.replicas")
        matrices = _window_matrices(plan["stiffness"], len(plan["centers"]), cv.dimension)
        tasks = [{"config": stage, "system": system, "specs": specs, "window": i,
                  "replica": r, "center": center, "stiffness": matrices[i], "output": str(pilot_output)}
                 for i, center in enumerate(plan["centers"]) for r in range(replicas)]
        if options.get("continuation", False):
            streams = _continuation_pilot(tasks, cv, stage, system, pilot_output, executor)
        else:
            streams = (executor.map(_sample_window, tasks) if executor is not None else
                       run_tasks(_sample_window, tasks, config.get("parallel", {})))
        plan = root_call(_record_round, plan, streams, config, system, str(output), cv, options, pilot)
    root_call(_require_frozen, plan, str(output))
    frozen = _stage_config(config, plan["centers"], plan["stiffness"], plan["production_seed"])
    if options.get("continuation", False):
        root_call(_production_continuation, config, system, frozen, plan, output)
    postprocess = partial(_annotate, plan=plan, output=str(output), periods=cv.periods,
                          production_steps=frozen["sampling"]["production_steps"])
    return sample_umbrellas(frozen, system, output, executor=executor,
                            _postprocess=postprocess)


def run_adaptive_fes(config, system, output, executor=None, return_data=False):
    """Compose adaptive sampling with the generic production FES estimator."""
    from .fes import _analyze_streams

    sampled = sample_adaptive_umbrellas(config, system, output, executor=executor)
    return root_call(
        _analyze_streams, sampled["streams"], sampled["sampling_config"], system,
        sampled["output"], return_data, sampled["postprocess"],
    )


def reanalyze_adaptive_fes(config, system, output, return_data=False):
    """Analyze only the frozen production checkpoints, never pilot histories."""
    from .fes import reanalyze_fes
    options = adaptive_options(config)
    cv, _, centers, stiffness = resolve_umbrellas(config, system)
    options = _resolve_coverage_options(options, cv)
    plan = root_call(_open_plan, config, system, str(output), centers, stiffness, options, False)
    root_call(_require_frozen, plan, str(output))
    frozen = _stage_config(config, plan["centers"], plan["stiffness"], plan["production_seed"])
    postprocess = partial(_annotate, plan=plan, output=str(output), periods=cv.periods,
                          production_steps=frozen["sampling"]["production_steps"])
    return reanalyze_fes(frozen, system, output, return_data=return_data,
                        _postprocess=postprocess)
