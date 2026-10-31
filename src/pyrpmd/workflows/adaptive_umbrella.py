"""Pilot-only adaptive placement followed by an independent, frozen production.

The algorithm refines the supplied 1D/2D domain; it does not discover arbitrary
unspecified basins. Missing overlap is repaired by midpoint insertion along
the shortest gaps between overlap components. Pilot length grows only when
effective sample size is insufficient, subject to an explicit cap. Stiffness is kept
fixed. Every decision, seed and diagnostic is persisted before the next stage.
An optional, explicitly bounded coverage grid also inserts windows into cells
with insufficient local reweighting ESS: graph connectivity alone cannot
guarantee coverage of a two-dimensional free-energy surface.

Only frozen-production streams enter the final MBAR/UI analysis. Adaptive
pilot samples cannot be passed off as samples from the frozen umbrella set.
"""

from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping
import numpy as np

from ..analysis.fes import (solve_mbar, umbrella_reduced_potentials,
                            cv_statistical_inefficiency, CV_CORRELATION_OBSERVABLES)
from ..bias import displacement
from ..parallel import root_call, run_tasks
from .fes import (resolve_umbrellas, _fingerprint, _sample_window,
                  checked_integer, write_json, jsonable, _dimension)
from ..units import parse_quantity


ADAPTIVE_FIELDS = {"enabled", "pilot_steps", "pilot_equilibration_steps", "max_rounds",
                   "max_windows", "max_pilot_steps", "target_overlap", "min_effective_samples", "min_spacing",
                   "coverage_bins", "coverage_ranges", "min_bin_effective_samples", "target_coverage"}


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
    options = {"enabled": enabled}
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
        if not isinstance(bins, (list, tuple, np.ndarray)) or len(bins) not in (1, 2):
            raise ValueError("adaptive_umbrella.coverage_bins must list one or two positive integers")
        bins = [checked_integer(n, "adaptive_umbrella.coverage_bins") for n in bins]
        if bins[0] * (bins[1] if len(bins) == 2 else 1) > 1_000_000:
            raise ValueError("adaptive_umbrella.coverage_bins must not exceed one million cells")
        if not isinstance(ranges, (list, tuple, np.ndarray)) or len(ranges) != len(bins):
            raise ValueError("adaptive_umbrella.coverage_ranges dimension must match coverage_bins")
        if any(not isinstance(pair, (list, tuple, np.ndarray)) or len(pair) != 2 for pair in ranges):
            raise ValueError("adaptive_umbrella.coverage_ranges must contain [lower, upper] pairs")
        options.update(coverage_bins=bins, coverage_ranges=[list(pair) for pair in ranges])
    if enabled:
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
    for i, c in enumerate(centers):
        delta = displacement(centers[:i], c, cv.periods)
        if len(delta) and np.any(np.einsum("ni,ij,nj->n", delta, stiffness, delta) * config["beta"] < options["min_spacing"]**2):
            raise ValueError("adaptive umbrella initial centers are duplicate or closer than min_spacing")


def _seed(config, phase, round_index=0):
    return int(np.random.SeedSequence([int(config.get("seed", 0)), 823, phase, round_index]).generate_state(1)[0])


def _stage_config(config, centers, stiffness, seed, *, pilot=None):
    stage = copy.deepcopy(config)
    stage.pop("adaptive_umbrella", None)
    stage["seed"] = seed
    umbrella = stage.setdefault("umbrella", {})
    for key in ("grid", "centers", "start", "stop", "count", "kappa", "stiffness"):
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


def diagnose_and_refine(streams, centers, stiffness, beta, periods, options):
    """Deterministic refinement from actual pilot samples, independently testable."""
    # Grouping visits the results once per window; executor.map may be a
    # one-shot iterator, so consume it only here and retain all pilot streams.
    streams = list(streams)
    centers = np.asarray(centers, float)
    count = len(centers)
    groups = [[s for s in streams if s["window"] == i] for i in range(count)]
    if any(not group for group in groups):
        raise ValueError("pilot results must contain every umbrella window")
    samples = [np.concatenate([s["samples"] for s in group]) for group in groups]
    stream_g = [[cv_statistical_inefficiency(s["samples"], periods) for s in group] for group in groups]
    ess = [sum(len(s["samples"]) / g for s, g in zip(group, factors))
           for group, factors in zip(groups, stream_g)]
    joined = np.concatenate(samples)
    reduced = umbrella_reduced_potentials(joined, centers, np.broadcast_to(stiffness, (count,) + np.shape(stiffness)), beta, periods)
    mbar = solve_mbar(reduced, [len(x) for x in samples], return_state_weights=False)
    if not mbar.converged:
        raise RuntimeError(f"adaptive pilot MBAR failed its equation residual: {mbar.residual:g}")
    scores = np.sqrt(np.maximum(0, mbar.overlap * mbar.overlap.T))
    pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
    strong = [(i, j) for i, j in pairs if scores[i, j] >= options["target_overlap"]]
    find, union = _components(count, strong)
    labels = [find(i) for i in range(count)]
    connected = len(set(labels)) == 1
    missing = []
    # Select shortest geometric links *between* sampled-overlap components,
    # preserving periodic distances and mixed-CV units via beta*K.
    def separation(pair):
        i, j = pair
        delta = displacement(centers[j], centers[i], periods)
        return float(beta * delta @ stiffness @ delta)
    for i, j in sorted(pairs, key=lambda pair: (separation(pair), pair)):
        if union(i, j):
            missing.append((i, j))
    # Also record the bottleneck of the strongest spanning tree as a compact
    # overlap diagnostic; it is not a proof of global exploration/mixing.
    _, tree_union = _components(count, [])
    tree_scores = []
    for i, j in sorted(pairs, key=lambda p: (-scores[p[0], p[1]], p)):
        if tree_union(i, j):
            tree_scores.append(float(scores[i, j]))
    proposed = centers.tolist()
    inserted = []
    rejected = []
    for i, j in missing:
        midpoint = centers[i] + 0.5 * displacement(centers[j], centers[i], periods)
        delta = displacement(np.asarray(proposed), midpoint, periods)
        distances = np.sqrt(np.maximum(0, beta * np.einsum("ni,ij,nj->n", delta, stiffness, delta)))
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
                distances = np.sqrt(np.maximum(0, beta * np.einsum("ni,ij,nj->n", delta, stiffness, delta)))
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
        "overlap_connected_at_target": connected,
        "overlap_components": labels, "overlap": mbar.overlap.tolist(),
        "symmetric_overlap": scores.tolist(), "spanning_overlap": min(tree_scores, default=1.0),
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
        if plan.get("fingerprint") != identity:
            raise ValueError("adaptive umbrella plan physics differs; choose a fresh output directory")
        return plan
    if not create:
        raise ValueError("no saved adaptive umbrella plan is available for reanalysis")
    plan = {"schema_version": 2, "fingerprint": identity, "status": "adapting", "next_round": 0,
            "centers": centers.tolist(), "stiffness": stiffness.tolist(), "options": options,
            "next_pilot_steps": options["pilot_steps"],
            "rounds": [], "production_seed": _seed(config, 2), "pilot_samples_used_in_production": 0}
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


def _record_round(plan, streams, config, system, output, cv, options, pilot):
    report = diagnose_and_refine(streams, np.asarray(plan["centers"]), np.asarray(plan["stiffness"]),
                                config["beta"], cv.periods, options)
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
    elif not report["overlap_connected_at_target"] and not report["inserted"]:
        plan["status"] = "incomplete"
        plan["failure_reason"] = "no admissible midpoint: max_windows or min_spacing"
    elif (plan.get("schema_version", 1) >= 2
          and (not report["effective_samples_target_met"]
               or not report.get("coverage", {}).get("target_met", True))
          and pilot["production_steps"] >= options["max_pilot_steps"] and not report["inserted"]):
        plan["status"] = "incomplete"
        target = ("effective-sample" if not report["effective_samples_target_met"] else "coverage")
        plan["failure_reason"] = "max_pilot_steps before " + target + " target"
    else:
        plan["centers"] = report["next_centers"]
    _save_plan(output, plan)
    write_json(Path(output) / "adaptation" / f"round_{round_index:03d}" / "diagnostics.json", decision)
    return plan


def _annotate(result, plan, output, return_data, production_steps=None):
    summary = result["summary"] if return_data else result
    # Independent production can be shorter or mix worse than the accepted
    # pilot. Expose its own ESS instead of letting target_met imply otherwise.
    production_ess = np.asarray(summary["effective_samples_per_stream"])
    replicas = len(production_ess) // len(plan["centers"])
    window_ess = production_ess.reshape(len(plan["centers"]), replicas).sum(axis=1)
    production_target = bool(np.all(window_ess >= plan["options"]["min_effective_samples"]))
    summary["adaptive_umbrella"] = {
        "status": "frozen", "target_met": True, "pilot_rounds": len(plan["rounds"]),
        "frozen_windows": len(plan["centers"]), "pilot_samples_used_in_production": 0,
        "production_seed": plan["production_seed"], "plan": str(_plan_path(output)),
        "production_effective_samples_per_window": window_ess.tolist(),
        "production_effective_samples_target_met": production_target,
        "final_pilot_steps": plan["rounds"][-1]["pilot"]["production_steps"],
        "production_steps_per_stream": production_steps,
        "target_note": "Pilot overlap/ESS and optional coverage targets only; physical and production sampling convergence remain unverified.",
    }
    if "coverage" in plan["rounds"][-1]:
        summary["adaptive_umbrella"]["coverage"] = plan["rounds"][-1]["coverage"]
    summary["adaptive_umbrella"]["effective_sample_observables"] = plan["rounds"][-1].get(
        "effective_sample_observables", "Legacy saved pilot: CV only, or first circular harmonics")
    if "effective_sample_observables" not in plan["rounds"][-1]:
        summary["warnings"].append("This saved frozen plan used legacy coordinate-only pilot ESS; "
                                   "use a fresh output directory to evaluate amplitude-sensitive pilot criteria.")
    summary["warnings"].append("Adaptive pilots were discarded before fixed-umbrella production; pilot overlap/ESS is not a physical convergence certificate.")
    if not production_target:
        summary["warnings"].append("Frozen production has windows below adaptive_umbrella.min_effective_samples; increase sampling.production_steps and check independent replicas.")
    if production_steps is not None and production_steps < plan["rounds"][-1]["pilot"]["production_steps"]:
        summary["warnings"].append("Production is shorter than the final accepted pilot; pilot success does not establish adequate production statistics.")
    write_json(Path(output) / "summary.json", summary)
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


def run_adaptive_fes(config, system, output, executor=None, return_data=False):
    """Adapt pilot windows, freeze, then start independent production samples."""
    from .fes import run_fes
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
        tasks = [{"config": stage, "system": system, "specs": specs, "window": i,
                  "replica": r, "center": center, "stiffness": plan["stiffness"], "output": str(pilot_output)}
                 for i, center in enumerate(plan["centers"]) for r in range(replicas)]
        streams = (executor.map(_sample_window, tasks) if executor is not None else
                   run_tasks(_sample_window, tasks, config.get("parallel", {})))
        plan = root_call(_record_round, plan, streams, config, system, str(output), cv, options, pilot)
    root_call(_require_frozen, plan, str(output))
    frozen = _stage_config(config, plan["centers"], plan["stiffness"], plan["production_seed"])
    result = run_fes(frozen, system, output, executor=executor, return_data=return_data)
    return root_call(_annotate, result, plan, str(output), return_data, frozen["sampling"]["production_steps"])


def reanalyze_adaptive_fes(config, system, output, return_data=False):
    """Analyze only the frozen production checkpoints, never pilot histories."""
    from .fes import reanalyze_fes
    options = adaptive_options(config)
    cv, _, centers, stiffness = resolve_umbrellas(config, system)
    options = _resolve_coverage_options(options, cv)
    plan = root_call(_open_plan, config, system, str(output), centers, stiffness, options, False)
    root_call(_require_frozen, plan, str(output))
    frozen = _stage_config(config, plan["centers"], plan["stiffness"], plan["production_seed"])
    result = reanalyze_fes(frozen, system, output, return_data=return_data)
    return root_call(_annotate, result, plan, str(output), return_data, frozen["sampling"]["production_steps"])
