"""Bounded, restartable refinement of constrained-parent recrossing samples."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np

from ..io import fingerprint, report_jsonable, write_json, write_report_json
from ..parallel import root_call, run_tasks
from ..validation import checked_integer
from .rate_analysis import _aggregate
from .rate_options import _recrossing_dynamics, _recrossing_options, _recrossing_spec
from .rate_recrossing import _child_tasks, _dispatch_children, _parent_task


def _quality(result, target, kappa_target=.01):
    uncertainty = result.get("rate_uncertainty", {})
    support = uncertainty.get("recrossing_support", {})
    reasons = []
    if not result.get("plateau_detected", False):
        reasons.append("recrossing_plateau_unresolved")
    if support.get("independent_parent_chains", 0) < 2:
        reasons.append("fewer_than_two_independent_parent_chains")
    if support.get("nonzero_flux_parent_configurations", 0) < 20:
        reasons.append("fewer_than_twenty_nonzero_flux_parents")
    if not uncertainty.get("available", False) or uncertainty.get("reasons"):
        reasons.append("joint_rate_uncertainty_unavailable")
    if not result.get("pmf", {}).get("uncertainty", {}).get("available", False):
        reasons.append("pmf_uncertainty_unavailable")
    estimate = result.get("rate_estimate_cm3_s")
    published = result.get("rate_cm3_s")
    if published is None or not np.isfinite(published) or published <= 0:
        reasons.append("rate_publication_prerequisites_not_met")
    error = result.get("rate_standard_error_cm3_s")
    relative = None
    if estimate is not None and np.isfinite(estimate) and estimate > 0 and error is not None:
        if np.isfinite(error) and error >= 0:
            relative = float(error / estimate)
    if relative is None or relative > target:
        reasons.append("rate_relative_error_above_target")
    energy = result.get("maximum_endpoint_energy_error_kbt")
    if energy is None or not np.isfinite(energy) or energy > .1:
        reasons.append("child_energy_error_above_tolerance")
    kappa = result.get("kappa")
    kappa_error = result.get("kappa_standard_error")
    kappa_relative = None
    if (kappa is not None and np.isfinite(kappa) and kappa > 0
            and kappa_error is not None and np.isfinite(kappa_error) and kappa_error >= 0):
        kappa_relative = float(kappa_error / abs(kappa))
    if kappa_relative is None or kappa_relative > kappa_target:
        reasons.append("kappa_relative_error_above_target")
    if kappa is None or not np.isfinite(kappa) or kappa <= 0:
        reasons.append("nonpositive_transmission")
    elif kappa > 1 + max(.05, 3 * (result.get("kappa_standard_error") or 0)):
        reasons.append("transmission_above_one")
    return {"passed": not reasons, "reasons": reasons, "relative_standard_error": relative,
            "target_relative_standard_error": target,
            "kappa_relative_standard_error": kappa_relative,
            "target_kappa_relative_standard_error": kappa_target}


def _pair_allocation(children, options, kappa):
    """Estimate a cost-balanced pair count from nested flux residual variance.

    This is a pilot allocation, not a proof of a globally optimal allocation.
    Children of a parent remain one resampling unit in the rate estimator.
    """
    grouped = {}
    tail = max(2, int(np.ceil(options["child_steps"] * options["plateau_fraction"])))
    for child in children:
        residual = float(np.mean(child["numerator"][-tail:]) - kappa * child["denominator"])
        grouped.setdefault((child["chain"], child["parent"]), []).append(residual)
    groups = [np.asarray(values) for values in grouped.values() if len(values) >= 2]
    if len(groups) < 2:
        return max(2, options["child_pairs"])
    within = float(np.mean([values.var(ddof=1) for values in groups]))
    between = max(0., float(np.var([values.mean() for values in groups], ddof=1))
                  - float(np.mean([values.var(ddof=1) / len(values) for values in groups])))
    if within <= 0:
        return 1
    if between <= 0:
        return min(64, options["child_pairs"] * 2)
    parent_cost = options["parent_interval_steps"] + options["equilibration_steps"] / options["parent_samples"]
    optimum = np.sqrt(within * parent_cost / (between * 2 * options["child_steps"]))
    return max(1, min(64, int(np.ceil(optimum))))


def _next_options(result, children, options, target, kappa_target=.01):
    updated = dict(options)
    actions = []
    diagnostic = result.get("plateau_diagnostics", {})
    limit = diagnostic.get("absolute_tolerance", 0.)
    variation = diagnostic.get("maximum_tail_deviation", 0.)
    radius = diagnostic.get("bootstrap_95_envelope_radius")
    drift = abs(result.get("plateau_drift") or 0.)
    drift_error = result.get("plateau_drift_standard_error")
    resolved_variation = radius is not None and variation - radius > limit
    resolved_drift = drift_error is not None and drift - 2 * drift_error > limit
    if (diagnostic.get("tail_samples", 0) < 10 or resolved_variation or resolved_drift
            or (result.get("kappa") or 0.) <= 0):
        updated["child_steps"] *= 2
        actions.append("extend_child_time")
    else:
        quality = _quality(result, target, kappa_target)
        ratios = [quality["relative_standard_error"], quality["kappa_relative_standard_error"]]
        scales = [value / limit for value, limit in zip(ratios, (target, kappa_target))
                  if value is not None]
        factor = max(2, min(4, int(np.ceil(max(scales, default=1.) ** 2))))
        preferred_pairs = _pair_allocation(children, options, result.get("kappa") or 0.)
        if preferred_pairs > options["child_pairs"]:
            updated["child_pairs"] = min(preferred_pairs, options["child_pairs"] * factor)
            actions.append("increase_child_pairs")
        else:
            updated["parent_samples"] *= factor
            actions.append("increase_parent_samples")
    support = result.get("rate_uncertainty", {}).get("recrossing_support", {})
    if support.get("nonzero_flux_parent_configurations", 0) < 20:
        updated["parent_samples"] = max(updated["parent_samples"], options["parent_samples"] * 2)
        actions.append("increase_parent_support")
    blocks = result.get("rate_uncertainty", {}).get("parent_block_lengths", {})
    correlation = max(blocks.values(), default=1)
    # A constant unit-transmission observable has a conservative whole-chain
    # block length, but contains no measured correlation to tune spacing from.
    if correlation > 1 and (result.get("kappa_standard_error") or 0.) > 0:
        updated["parent_interval_steps"] *= min(4, int(np.ceil(correlation)))
        actions.append("increase_parent_spacing")
    if (support.get("independent_parent_chains", 0) < 2
            or (correlation >= options["parent_samples"] and (result.get("kappa_standard_error") or 0.) > 0)):
        updated["parent_replicas"] = max(options["parent_replicas"], 2,
                                         min(16, options["parent_replicas"] * 2))
        if updated["parent_replicas"] > options["parent_replicas"]:
            actions.append("increase_parent_chains")
    return updated, actions


def _prepare_round(directory, config, system, options, pmf, parent_tasks, target):
    directory.mkdir(parents=True, exist_ok=True)
    identity = fingerprint(config, system, {
        "stage": "auto_recrossing", "options": options, "target_relative_error": target,
        "pmf": report_jsonable({key: value for key, value in pmf.items()
                                if key not in ("propagation", "propagation_parents", "warnings")}),
        "parent_origins": [task["initial_q"] for task in parent_tasks],
    })
    plan = {"fingerprint": identity, "config": config, "options": options}
    path = directory / "plan.json"
    if path.exists():
        if json.loads(path.read_text())["fingerprint"] != identity:
            raise ValueError("Automatic recrossing round belongs to another PMF, configuration or parent origin")
    else:
        write_json(path, plan)
    completed = directory / "report.json"
    if completed.exists():
        if not all((directory / name).is_file() for name in ("recrossing.npz", "recrossing.csv")):
            raise ValueError("Completed automatic recrossing round is missing its saved arrays")
        return json.loads(completed.read_text())
    return None


def _finish_round(children, parents, pmf, config, directory, options, target):
    round_pmf = dict(pmf, propagation_parents=[p.get("propagation", {}) for p in parents])
    result = _aggregate(children, round_pmf, config, str(directory), options)
    kappa_target = config.get("rate", {}).get("auto", {}).get("kappa_relative_error", .01)
    quality = _quality(result, target, kappa_target)
    updated, actions = (_next_options(result, children, options, target, kappa_target)
                        if not quality["passed"] else (options, []))
    result["sampling_converged"] = quality["passed"]
    result["auto_recrossing"] = {"quality": quality, "options": options,
                                 "next_options": updated, "actions": actions}
    if not quality["passed"]:
        result["rate_cm3_s"] = None
    result = report_jsonable(result)
    write_report_json(directory / "report.json", result)
    return result


def _publish(result, directory, output, rounds, max_rounds):
    result = deepcopy(result)
    passed = result["sampling_converged"]
    result["auto_recrossing"].update(
        rounds=rounds, max_rounds=max_rounds, directory=str(directory),
        status="converged" if passed else ("budget_exhausted" if rounds == max_rounds else "requires_action"),
        sampling_scope="PMF and recrossing statistical precision at fixed beads, timestep and Rinf",
    )
    if not passed:
        result["status"] = "auto_rate_" + result["auto_recrossing"]["status"]
        result["rate_cm3_s"] = None
    for name in ("recrossing.npz", "recrossing.csv"):
        temporary = output / (name + ".auto-tmp")
        shutil.copyfile(directory / name, temporary)
        temporary.replace(output / name)
    result.setdefault("files", {}).update(pmf=str(output / "rate_pmf.csv"),
                                          recrossing=str(output / "recrossing.csv"))
    write_report_json(output / "auto-rate" / "recrossing" / "summary.json", result)
    return result


def run_auto_recrossing(config, system, output, pmf, parent_tasks, *,
                        target_relative_error, max_rounds, executor=None):
    """Refine complete ensembles; rounds never pool duplicate seeded samples.

    Every changed budget gets its own checkpoints. Existing child checkpoints
    contain fluxes rather than terminal states, so longer trajectories are
    recomputed deterministically; completed rounds replay without propagation.
    """
    max_rounds = checked_integer(max_rounds, "rate.auto.max_rounds")
    if not np.isfinite(target_relative_error) or target_relative_error <= 0:
        raise ValueError("rate.auto.target_relative_error must be finite and positive")
    if not parent_tasks or not pmf.get("ready", False):
        raise ValueError("Automatic recrossing requires a supported PMF and parent origins")
    if _recrossing_dynamics(config)[0] != "rpmd":
        raise ValueError("Automatic rate recrossing requires RPMD child dynamics")
    output = Path(output)
    options = _recrossing_options(config)

    def dispatch(function, tasks):
        return (executor.map(function, tasks) if executor is not None
                else run_tasks(function, tasks, config.get("parallel", {})))

    for number in range(1, max_rounds + 1):
        directory = output / "auto-rate" / "recrossing" / f"round-{number:03d}"
        round_config = deepcopy(config)
        round_config.setdefault("rate", {}).pop("recrossing", None)
        round_config["recrossing"] = {**_recrossing_spec(config), **options}
        result = root_call(_prepare_round, directory, round_config, system, options,
                           pmf, parent_tasks, target_relative_error)
        if result is None:
            tasks = [dict(parent_tasks[chain % len(parent_tasks)], config=round_config, system=system,
                          options=options, output=str(directory), chain=chain)
                     for chain in range(options["parent_replicas"])]
            parents = root_call(list, dispatch(_parent_task, tasks))
            children_tasks = root_call(_child_tasks, parents, round_config, system,
                                       str(directory), options, pmf)
            children = root_call(list, _dispatch_children(children_tasks, round_config, dispatch))
            result = root_call(_finish_round, children, parents, pmf, round_config,
                               directory, options, target_relative_error)
        reasons = result["auto_recrossing"]["quality"]["reasons"]
        if (result["sampling_converged"] or "child_energy_error_above_tolerance" in reasons
                or "pmf_uncertainty_unavailable" in reasons or "transmission_above_one" in reasons):
            break
        options = result["auto_recrossing"]["next_options"]
    return root_call(_publish, result, directory, output, number, max_rounds)
