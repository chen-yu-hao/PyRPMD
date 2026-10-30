"""Pilot-only adaptive placement followed by an independent, frozen production.

The algorithm refines the supplied 1D/2D domain; it does not discover arbitrary
unspecified basins. Missing overlap is repaired by midpoint insertion along
the shortest gaps between overlap components. Pilot length doubles each round
to address insufficient effective sample size. The initial stiffness is kept
fixed. Every decision, seed and diagnostic is persisted before the next stage.

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

from ..analysis.fes import solve_mbar, umbrella_reduced_potentials, statistical_inefficiency
from ..bias import displacement
from ..parallel import root_call, run_tasks
from .fes import (resolve_umbrellas, _fingerprint, _sample_window,
                  checked_integer, write_json, jsonable)


ADAPTIVE_FIELDS = {"enabled", "pilot_steps", "pilot_equilibration_steps", "max_rounds",
                   "max_windows", "target_overlap", "min_effective_samples", "min_spacing"}


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
        ("max_rounds", 4, 1),
        ("max_windows", 64, 1),
    ):
        options[key] = checked_integer(table.get(key, default), "adaptive_umbrella." + key, minimum)
    for key, default in (("target_overlap", 0.03), ("min_effective_samples", 20.0), ("min_spacing", 1e-6)):
        value = table.get(key, default)
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("adaptive_umbrella." + key + " must be a positive number")
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("adaptive_umbrella." + key + " must be positive and finite")
        options[key] = value
    if options["target_overlap"] >= 1:
        raise ValueError("adaptive_umbrella.target_overlap must be below one")
    if enabled:
        if config.get("task", "fes") not in ("fes", "rate"):
            raise ValueError("adaptive_umbrella is supported only for FES and rate PMF workflows")
        interval = checked_integer(config.get("sampling", {}).get("sample_interval",
                    config.get("sampling", {}).get("interval_steps", 1)), "sampling.sample_interval")
        if options["pilot_steps"] // interval < 2:
            raise ValueError("adaptive_umbrella.pilot_steps must include at least two saved samples")
    return options


def validate_adaptive_umbrella(config, system):
    """Early input validation; TS-only rate CVs are resolved after preparation."""
    options = adaptive_options(config)
    if not options["enabled"]:
        return
    if config.get("task") == "rate" and not config.get("cvs"):
        return
    cv, _, centers, stiffness = resolve_umbrellas(config, system)
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


def diagnose_and_refine(streams, centers, stiffness, beta, periods, options):
    """Deterministic refinement from actual pilot samples, independently testable."""
    centers = np.asarray(centers, float)
    count = len(centers)
    groups = [[s for s in streams if s["window"] == i] for i in range(count)]
    if any(not group for group in groups):
        raise ValueError("pilot results must contain every umbrella window")
    samples = [np.concatenate([s["samples"] for s in group]) for group in groups]
    def inefficiency(values):
        # Circular CV traces must not appear artificially decorrelated merely
        # because they cross their coordinate branch cut.
        traces = []
        for j, period in enumerate(periods):
            if period is None:
                traces.append(values[:, j])
            else:
                angle = 2 * np.pi * values[:, j] / period
                traces.extend([np.sin(angle), np.cos(angle)])
        return max(statistical_inefficiency(x) for x in traces)
    ess = [sum(len(s["samples"]) / inefficiency(np.asarray(s["samples"])) for s in group) for group in groups]
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
    enough = bool(np.all(np.asarray(ess) >= options["min_effective_samples"]))
    return {
        "target_met": bool(connected and enough), "overlap_connected_at_target": connected,
        "overlap_components": labels, "overlap": mbar.overlap.tolist(),
        "symmetric_overlap": scores.tolist(), "spanning_overlap": min(tree_scores, default=1.0),
        "effective_samples_per_window": ess, "effective_samples_target_met": enough,
        "samples_per_window": [len(x) for x in samples], "mbar_residual": float(mbar.residual),
        "inserted": inserted, "rejected_insertions": rejected,
        "next_centers": proposed,
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
    plan = {"schema_version": 1, "fingerprint": identity, "status": "adapting", "next_round": 0,
            "centers": centers.tolist(), "stiffness": stiffness.tolist(), "options": options,
            "rounds": [], "production_seed": _seed(config, 2), "pilot_samples_used_in_production": 0}
    _save_plan(output, plan)
    return plan


def _record_round(plan, streams, config, system, output, cv, options, pilot):
    report = diagnose_and_refine(streams, np.asarray(plan["centers"]), np.asarray(plan["stiffness"]),
                                config["beta"], cv.periods, options)
    round_index = plan["next_round"]
    decision = {"round": round_index, "seed": _seed(config, 1, round_index),
                "centers": plan["centers"], "pilot": pilot, **report}
    plan["rounds"].append(decision)
    plan["next_round"] += 1
    if report["target_met"]:
        plan["status"] = "frozen"
        plan["target_met"] = True
    elif plan["next_round"] >= options["max_rounds"]:
        plan["status"] = "incomplete"
        plan["failure_reason"] = "max_rounds before overlap/effective-sample targets"
    elif not report["overlap_connected_at_target"] and not report["inserted"]:
        plan["status"] = "incomplete"
        plan["failure_reason"] = "no admissible midpoint: max_windows or min_spacing"
    else:
        plan["centers"] = report["next_centers"]
    _save_plan(output, plan)
    write_json(Path(output) / "adaptation" / f"round_{round_index:03d}" / "diagnostics.json", decision)
    return plan


def _annotate(result, plan, output, return_data):
    summary = result["summary"] if return_data else result
    summary["adaptive_umbrella"] = {
        "status": "frozen", "target_met": True, "pilot_rounds": len(plan["rounds"]),
        "frozen_windows": len(plan["centers"]), "pilot_samples_used_in_production": 0,
        "production_seed": plan["production_seed"], "plan": str(_plan_path(output)),
        "target_note": "Pilot overlap/ESS targets only; physical and production sampling convergence remain unverified.",
    }
    summary["warnings"].append("Adaptive pilots were discarded before fixed-umbrella production; pilot overlap/ESS is not a physical convergence certificate.")
    write_json(Path(output) / "summary.json", summary)
    return result


def _require_frozen(plan, output):
    if plan["status"] != "frozen":
        summary = {"task": "fes", "status": "adaptation_incomplete", "converged": False,
                   "failure_reason": plan.get("failure_reason", "pilot stage incomplete"),
                   "production_started": False, "plan": str(_plan_path(output))}
        write_json(Path(output) / "summary.json", summary)
        raise RuntimeError("Adaptive umbrella targets were not reached; production was not started. " +
                           summary["failure_reason"] + ". Inspect " + str(_plan_path(output)))


def run_adaptive_fes(config, system, output, executor=None, return_data=False):
    """Adapt pilot windows, freeze, then start independent production samples."""
    from .fes import run_fes
    validate_adaptive_umbrella(config, system)
    options = adaptive_options(config)
    cv, specs, centers, stiffness = resolve_umbrellas(config, system)
    plan = root_call(_open_plan, config, system, str(output), centers, stiffness, options)
    while plan["status"] == "adapting":
        index = plan["next_round"]
        pilot = {"equilibration_steps": options["pilot_equilibration_steps"],
                 "production_steps": options["pilot_steps"] * 2**index}
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
    return root_call(_annotate, result, plan, str(output), return_data)


def reanalyze_adaptive_fes(config, system, output, return_data=False):
    """Analyze only the frozen production checkpoints, never pilot histories."""
    from .fes import reanalyze_fes
    options = adaptive_options(config)
    _, _, centers, stiffness = resolve_umbrellas(config, system)
    plan = root_call(_open_plan, config, system, str(output), centers, stiffness, options, False)
    root_call(_require_frozen, plan, str(output))
    frozen = _stage_config(config, plan["centers"], plan["stiffness"], plan["production_seed"])
    result = reanalyze_fes(frozen, system, output, return_data=return_data)
    return root_call(_annotate, result, plan, str(output), return_data)
