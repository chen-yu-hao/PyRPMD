"""Bounded AUS → frozen PMF → recrossing feedback for an automatic rate.

Production extensions keep the original stream identities. A change to the
window Hamiltonians starts a new, independent production generation instead.
The durable next budget is written before sampling, so resume never requests
a shorter prefix from an already extended checkpoint.
"""

from __future__ import annotations

import copy
from functools import partial
import json
from pathlib import Path

import numpy as np

from ..input import rate_pmf_stage
from ..io import write_report_json
from ..parallel import root_call
from .common import _fingerprint
from .fes import resolve_umbrellas, sample_umbrellas
from .rate_analysis import _aggregate
from .rate_options import _recrossing_options
from .rate_pmf import _pmf_and_parent_tasks, _prepare


MIN_GROWTH_FACTOR = 1.25
MAX_GROWTH_FACTOR = 2.0


def _bounded_growth_factor(relative, target, *, fallback=MAX_GROWTH_FACTOR):
    """Choose a bounded budget multiplier from the observed relative error."""
    if (relative is None or target is None or not np.isfinite(relative)
            or not np.isfinite(target) or relative <= 0 or target <= 0):
        return MAX_GROWTH_FACTOR if fallback is None else float(fallback)
    return float(np.clip((relative / target) ** 2,
                         MIN_GROWTH_FACTOR, MAX_GROWTH_FACTOR))


def _state_path(output):
    return Path(output) / "auto-rate" / "state.json"


def _open_state(config, system, sampling, output):
    path = _state_path(output)
    identity = _fingerprint(config, system, {"auto_rate_schema": 1})
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("fingerprint") != identity:
            raise ValueError("automatic rate settings differ; use a fresh output directory")
        # Resume can change execution resources without changing a stream's
        # scientific identity or its saved adaptation/sampling budgets.
        for stored in (state["sampling_config"], state.get("frozen_config")):
            if stored is None:
                continue
            stored["parallel"] = copy.deepcopy(sampling.get("parallel", {}))
            for key in ("loop_backend", "block_steps"):
                if key in sampling.get("dynamics", {}):
                    stored.setdefault("dynamics", {})[key] = sampling["dynamics"][key]
        return state
    state = {
        "schema_version": 1, "fingerprint": identity, "phase": "pmf",
        "round": 0, "generation": 0, "history": [],
        "sampling_config": sampling, "sampling_directory": str(Path(output) / "umbrella"),
        "production_steps": sampling["sampling"]["production_steps"],
        "replicas": sampling["sampling"].get("replicas", 2),
    }
    write_report_json(path, state)
    return state


def _production_hook(state, system):
    from .adaptive_umbrella import _annotate, _open_plan, adaptive_options

    sampling = state["sampling_config"]
    cv, _, centers, stiffness = resolve_umbrellas(sampling, system)
    plan = _open_plan(sampling, system, state["sampling_directory"], centers,
                      stiffness, adaptive_options(sampling), create=False)
    return partial(_annotate, plan=plan, output=state["sampling_directory"],
                   periods=cv.periods, production_steps=state["production_steps"])


def _pmf_decision(pmf, state, config):
    """Distinguish missing spatial support, correlated time and sample count."""
    error = pmf.get("barrier_standard_error_hartree")
    log_error = (float(config["beta"] * error)
                 if error is not None and np.isfinite(error) and error >= 0 else None)
    # Reserve half the variance for each Bennett--Chandler factor. The final
    # joint bootstrap, rather than this allocation, decides rate acceptance.
    target = config["rate"]["auto"]["target_relative_error"] / np.sqrt(2.)
    decision = {"pmf_log_standard_error": log_error, "target": float(target)}
    uncertainty = pmf.get("uncertainty", {})
    supported = uncertainty.get("available") and not uncertainty.get("reasons")
    barriers = np.asarray(pmf.get("pmf_bootstrap_barriers", []), float)
    barrier = pmf.get("free_energy_barrier_hartree")
    barrier_relative = (float(error / abs(barrier))
                        if log_error is not None and barrier is not None
                        and np.isfinite(barrier) and barrier != 0 else None)
    barrier_target = config["rate"]["auto"].get("pmf_relative_error", .01)
    decision.update(pmf_relative_standard_error=barrier_relative,
                    target_pmf_relative_standard_error=barrier_target)
    relative = None
    if len(barriers) >= 2 and np.isfinite(barriers).all() and barrier is not None:
        # Use the actual nonlinear QTST bootstrap; beta*sigma(W) alone can
        # underestimate the error for an asymmetric barrier distribution.
        with np.errstate(over="ignore", invalid="ignore"):
            relative = float(np.std(np.exp(-config["beta"] * (barriers - barrier)), ddof=1))
        if not np.isfinite(relative):
            relative = None
    decision["qtst_relative_standard_error"] = relative
    if (pmf.get("window_anomalies") or
            any(not check["passed"] for check in pmf.get("stream_state_checks", []))):
        return {**decision, "action": "stop", "reason": "umbrella_stream_anomaly"}

    adaptive = pmf.get("adaptive_umbrella", {})
    frozen = state["frozen_config"]
    centers = np.asarray(frozen["umbrella"]["centers_au"], float)[:, 0]
    candidates = []
    support = pmf.get("support", {})
    histogram = support.get("histogram", {})
    grid = np.asarray(histogram.get("centers", []))
    for interval in support.get("search_endpoints", {}).get("masked_intervals", []):
        candidates.extend(grid[interval["first_bin"]:interval["last_bin"] + 1].tolist())
    for endpoint in (support.get("reference_endpoint"),
                     *support.get("observed_histogram", {}).values()):
        if endpoint and (endpoint.get("status") == "masked" or endpoint.get("supported") is False):
            candidates.extend(endpoint.get("bracket_centers", [endpoint["coordinate"]]))
    if (not candidates and pmf.get("ready") and supported
            and relative is not None and relative <= target
            and barrier_relative is not None and barrier_relative <= barrier_target):
        return {**decision, "action": "recrossing"}
    labels = adaptive.get("production_overlap_components", [])
    if len(labels) == len(centers) and len(set(labels)) > 1:
        order = np.argsort(centers)
        candidates.extend(float((centers[i] + centers[j]) / 2)
                          for i, j in zip(order[:-1], order[1:]) if labels[i] != labels[j])
    coverage = adaptive.get("production_coverage", {})
    if coverage and not coverage["target_met"]:
        lo, hi = coverage["ranges"][0]
        edges = np.linspace(lo, hi, coverage["bins"][0] + 1)
        candidates.extend(float((edges[i] + edges[i + 1]) / 2)
                          for i, ess in enumerate(coverage["effective_samples"])
                          if ess < coverage["min_bin_effective_samples"])
    limit = state["sampling_config"]["adaptive_umbrella"].get("max_windows", 128)
    new_centers = centers.tolist()
    for center in candidates:
        if len(new_centers) >= limit:
            break
        if min(abs(center - np.asarray(new_centers))) > 1e-8:
            new_centers.append(center)
    if len(new_centers) > len(centers):
        return {**decision, "action": "refine_windows", "centers": sorted(new_centers),
                "reason": "frozen_production_spatial_support"}

    # More replicas cannot create a time series long enough for a correlated
    # block bootstrap. First lengthen short streams; once these mix, add truly
    # independent trajectories to reduce the barrier variance.
    ess = np.asarray(pmf.get("effective_samples_per_stream", []), float)
    if (not supported or relative is None or not len(ess)
            or not np.isfinite(ess).all() or float(ess.min()) < 20):
        return {**decision, "action": "extend_time", "reason": "correlated_sample_support",
                "growth_factor": MAX_GROWTH_FACTOR}
    return {**decision, "action": "add_replicas", "reason": "pmf_precision",
            "growth_factor": _bounded_growth_factor(relative, target)}


def _advance(state, decision):
    state["round"] += 1
    if decision["action"] == "refine_windows":
        from .adaptive_umbrella import interpolate_stiffness

        state["generation"] += 1
        sampling = copy.deepcopy(state.pop("frozen_config"))
        old_centers = sampling["umbrella"]["centers_au"]
        springs = sampling["umbrella"]["stiffness_au"]
        sampling["umbrella"]["stiffness_au"] = interpolate_stiffness(
            old_centers, springs, [[x] for x in decision["centers"]], (None,),
        ).tolist()
        sampling["umbrella"]["centers_au"] = [[x] for x in decision["centers"]]
        sampling["adaptive_umbrella"] = copy.deepcopy(state["sampling_config"]["adaptive_umbrella"])
        sampling["seed"] = int(np.random.SeedSequence(
            [state["sampling_config"]["seed"], 947, state["generation"]]).generate_state(1)[0])
        sampling["sampling"].update(production_steps=state["production_steps"], replicas=state["replicas"])
        state["sampling_config"] = sampling
        state["sampling_directory"] = str(Path(state["sampling_directory"]).parents[0] /
                                           f"umbrella-generation-{state['generation']:03d}")
    elif decision["action"] == "extend_time":
        factor = float(np.clip(decision.get("growth_factor", MAX_GROWTH_FACTOR),
                               MIN_GROWTH_FACTOR, MAX_GROWTH_FACTOR))
        state["production_steps"] = max(
            state["production_steps"] + 1,
            int(np.ceil(state["production_steps"] * factor)),
        )
        decision["growth_factor"] = factor
    elif decision["action"] == "add_replicas":
        factor = float(np.clip(decision.get("growth_factor", MAX_GROWTH_FACTOR),
                               MIN_GROWTH_FACTOR, MAX_GROWTH_FACTOR))
        state["replicas"] = max(
            state["replicas"] + 1,
            int(np.ceil(state["replicas"] * factor)),
        )
        decision["growth_factor"] = factor


def _incomplete(pmf, config, output, options, reason, history):
    result = _aggregate([], pmf, config, output, options)
    result.update(status="auto_rate_incomplete", sampling_converged=False,
                  automatic_rate={"stage": "pmf", "reason": reason, "history": history})
    write_report_json(Path(output) / "summary.json", result)
    return result


def run_auto_rate(config, system, output, executor=None):
    from . import validate_bond_constraint_workflow
    from .auto_recrossing import run_auto_recrossing
    from .rate_subspace import finish_rate
    from .sampling import validate_andersen_schedule

    config = copy.deepcopy(config)
    validate_bond_constraint_workflow(system, "rate")
    validate_andersen_schedule(config, "rate")
    if any(system.pbc) or system.fixed_atoms:
        raise ValueError("automatic gas-phase rate does not support periodic/fixed-atom systems")
    config, system = root_call(_prepare, config, system, str(output), True)
    options = _recrossing_options(config)
    controls = config["rate"]["auto"]
    sampling = rate_pmf_stage(config, system) or config
    if config.get("output", {}).get("centroid_path"):
        sampling = copy.deepcopy(sampling)
        sampling.setdefault("output", {})["centroid_path"] = True
    state = root_call(_open_state, config, system, sampling, str(output))
    if state.get("result") is not None:
        return state["result"]

    while True:
        if "frozen_config" not in state:
            try:
                sampled = sample_umbrellas(state["sampling_config"], system,
                                           state["sampling_directory"], executor=executor)
            except RuntimeError:
                def exhausted():
                    path = Path(state["sampling_directory"]) / "adaptive-plan.json"
                    return path.exists() and json.loads(path.read_text()).get("status") == "incomplete"
                if not root_call(exhausted):
                    raise
                pmf = {"ready": False, "status": "adaptation_incomplete",
                       "warnings": ["AUS pilot budget exhausted; production and recrossing were not started."]}
                return root_call(_incomplete, pmf, config, str(output), options,
                                 "aus_budget_exhausted", state["history"])
            state["frozen_config"] = sampled["sampling_config"]
            root_call(write_report_json, _state_path(output), state)
        else:
            hook = root_call(_production_hook, state, system)
            sampled = sample_umbrellas(
                state["frozen_config"], system, state["sampling_directory"], executor=executor,
                production_steps=state["production_steps"], replicas=state["replicas"],
                _postprocess=hook,
            )
        pmf, parent_tasks = root_call(_pmf_and_parent_tasks, sampled, config, system,
                                     str(output), options, executor)
        decision = root_call(_pmf_decision, pmf, state, config)
        record = {"round": state["round"], "generation": state["generation"],
                  "production_steps": state["production_steps"], "replicas": state["replicas"],
                  "sampling_directory": state["sampling_directory"], "decision": decision, "pmf": pmf}
        # Replacing the current record after an interruption is deterministic;
        # past budgets are never replayed against longer checkpoints.
        state["history"] = state["history"][:state["round"]] + [record]
        root_call(write_report_json, Path(output) / "auto-rate" / "pmf" /
                  f"round-{state['round']:03d}.json", record)
        if decision["action"] == "recrossing":
            state["phase"] = "recrossing"
            root_call(write_report_json, _state_path(output), state)
            result = run_auto_recrossing(
                config, system, output, pmf, parent_tasks,
                target_relative_error=controls["target_relative_error"],
                max_rounds=controls["max_rounds"], executor=executor,
            )
            result.setdefault("automatic_rate", {})["pmf_history"] = state["history"]
            # Optional subspace analysis reads this source summary and binds
            # its identity. Publish it first and keep it immutable afterwards.
            root_call(write_report_json, Path(output) / "summary.json", result)
            result = root_call(finish_rate, config, str(output), result)
            state.update(phase="complete" if result.get("sampling_converged") else "incomplete", result=result)
            root_call(write_report_json, _state_path(output), state)
            return result
        if decision["action"] == "stop" or state["round"] + 1 >= controls["max_rounds"]:
            reason = decision.get("reason") if decision["action"] == "stop" else "pmf_budget_exhausted"
            result = root_call(_incomplete, pmf, config, str(output), options, reason, state["history"])
            state.update(phase="incomplete", result=result)
            root_call(write_report_json, _state_path(output), state)
            return result
        _advance(state, decision)
        root_call(write_report_json, _state_path(output), state)
