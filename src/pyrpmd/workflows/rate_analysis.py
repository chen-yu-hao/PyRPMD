"""Combine PMF and correlated recrossing estimates with rate diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..analysis.recrossing import summarize_recrossing
from ..constants import ATOMIC_TIME_S
from ..checkpoint import atomic_savez as _atomic_npz
from ..io import write_report_json as write_json
from .rate_options import _recrossing_dynamics


def _aggregate(children, pmf, config, output, options):
    child_dynamics, friction_scale = _recrossing_dynamics(config)
    thermostatted = child_dynamics == "trpmd"
    is_surface = config.get("task") == "surface_rate"
    task_name = "surface_rate" if is_surface else "rate"
    unit_key = "s_inv" if is_surface else "cm3_s"
    qtst_key = "k_qtst_" + unit_key
    private_pmf = ("pmf_bootstrap_barriers", "pmf_bootstrap_qtst")
    if not children:
        pmf_status = pmf.get("status", "insufficient_overlap")
        result = {
            "task": task_name,
            "status": (
                pmf_status if pmf_status != "ready" else "insufficient_recrossing"
            ),
            "converged": False,
            "rate_" + unit_key: None,
            "rate_estimate_" + unit_key: None,
            "rate_standard_error_" + unit_key: None,
            "kappa": None,
            "pmf": {k: v for k, v in pmf.items() if k not in private_pmf},
            "warnings": pmf["warnings"],
        }
        if is_surface:
            from ..analysis.surface import surface_rate_quality
            result["rate_quality"] = surface_rate_quality(
                pmf, {}, None, None, None,
                target=config.get("surface", {}).get("max_relative_standard_error", .2),
                dynamics=child_dynamics,
            )
        write_json(Path(output) / "summary.json", result)
        return result
    for child in children:
        if "xi_star" in child and child["xi_star"] != pmf.get("xi_star"):
            raise ValueError("recrossing child belongs to another dividing surface")
        if "pmf_method" in child and child["pmf_method"] != pmf.get("method", "ui"):
            raise ValueError("recrossing child belongs to another PMF method")
    if not pmf.get("ready", True) or pmf.get(qtst_key) is None:
        raise ValueError(
            "cannot combine recrossing trajectories with an unsupported PMF"
        )
    rng = np.random.default_rng(
        np.random.SeedSequence([int(config.get("seed", 0)), 503])
    )
    statistics = summarize_recrossing(
        children, rng=rng,
        **{key: options[key] for key in (
            "child_steps", "plateau_fraction", "plateau_tolerance", "bootstrap"
        )},
    )
    keys = statistics["parent_ids"]
    num, den = statistics["parent_numerator"], statistics["parent_denominator"]
    transmission, start = statistics["curve"], statistics["start"]
    estimate, drift = statistics["kappa"], statistics["drift"]
    boots = statistics["bootstrap_kappa"]
    stderr, drift_error = statistics["standard_error"], statistics["drift_standard_error"]
    curve_error, plateau = statistics["curve_standard_error"], statistics["plateau_detected"]
    recrossing_uncertainty = statistics["uncertainty"]
    candidate = pmf[qtst_key] * estimate
    total_boot = []
    barriers = pmf.get("pmf_bootstrap_barriers", [])
    qtst_draws = pmf.get("pmf_bootstrap_qtst", [])
    pmf_draws = qtst_draws if is_surface else barriers
    if is_surface and len(qtst_draws) >= 2 and recrossing_uncertainty["available"]:
        total_boot = [float(rng.choice(qtst_draws)) * kval for kval in boots]
    elif not is_surface and len(barriers) >= 2 and recrossing_uncertainty["available"]:
        for kval in boots:
            barrier = rng.choice(barriers)
            total_boot.append(
                pmf[qtst_key]
                * np.exp(
                    -config["beta"] * (barrier - pmf["free_energy_barrier_hartree"])
                )
                * kval
            )
    error = float(np.std(total_boot, ddof=1)) if len(total_boot) > 1 else None
    supported_draw_error = error
    conditional_error = (
        abs(pmf[qtst_key]) * stderr if stderr is not None else None
    )
    uncertainty_reasons = list(recrossing_uncertainty["reasons"])
    if len(pmf_draws) < 2:
        uncertainty_reasons.append("fewer_than_two_valid_pmf_bootstrap_draws")
    pmf_bootstrap = pmf.get("bootstrap", {})
    pmf_block_status = pmf_bootstrap.get("block_plan", {}).get("status")
    if pmf_block_status in ("limited_blocks", "unavailable"):
        uncertainty_reasons.append("pmf_bootstrap_" + pmf_block_status)
    if pmf_block_status == "unavailable":
        error = None
        supported_draw_error = None
    rejected_pmf_draws = max(0, pmf_bootstrap.get("attempted", 0) - pmf_bootstrap.get("valid", 0))
    if rejected_pmf_draws:
        uncertainty_reasons.append("pmf_bootstrap_contains_unsupported_draws")
        # A confidence interval conditional on complete histogram support is
        # not an unconditional joint interval. Preserve it under an explicit
        # conditional label, and withhold the primary uncertainty fields.
        error = None
    times = np.arange(1, len(transmission) + 1) * config["dt_au"]
    _atomic_npz(
        Path(output) / "recrossing.npz",
        time_au=times,
        kappa=transmission,
        standard_error=curve_error,
        parent_numerator=num,
        parent_denominator=den,
        parent_ids=np.asarray(keys),
        bootstrap_kappa=np.asarray(boots),
    )
    np.savetxt(
        Path(output) / "recrossing.csv",
        np.column_stack([times, transmission, curve_error]),
        delimiter=",",
        header="time_au,kappa,standard_error",
        comments="",
    )
    warnings = list(pmf["warnings"])
    if uncertainty_reasons:
        warnings.append(
            ("Complete rate uncertainty is unavailable: " if error is None
             else "Complete rate uncertainty has limited support: ")
            + "; ".join(uncertainty_reasons)
            + ". Recrossing-only conditional uncertainty holds QTST fixed; supported-draw uncertainty is conditional on complete PMF support."
        )
    if not plateau:
        warnings.append(
            "No stable positive recrossing plateau detected in the selected tail; extend child propagation/sampling."
        )
    max_drift = max(c["energy_drift"] for c in children)
    endpoint_errors = [
        c["endpoint_energy_error_hartree_per_bead"]
        for c in children
        if c.get("endpoint_energy_error_hartree_per_bead") is not None
    ]
    endpoint_error = max(endpoint_errors) if endpoint_errors else None
    energy_diagnostics_complete = len(endpoint_errors) == len(children)
    if not thermostatted and max_drift > 0.01:
        warnings.append(
            "Some child trajectories have >1% legacy scaled endpoint drift; this energy-zero-dependent field is informational."
        )
    if thermostatted:
        warnings.append(
            "TRPMD friction sensitivity calculation: the centroid is unthermostatted, "
            "but the internal modes exchange heat. Endpoint energy changes include "
            "thermostat heat and cannot by themselves diagnose integration error. "
            "Compare friction and timestep series; "
            "this estimate is not an RPMD rate."
        )
    elif not energy_diagnostics_complete:
        warnings.append(
            "Some older child checkpoints lack energy-zero-independent errors; rate publication is withheld until those children are recomputed."
        )
    elif endpoint_error * config["beta"] > 0.1:
        warnings.append(
            "Child endpoint energy error exceeds 0.1 kBT per bead; reduce timestep before publishing a rate estimate."
        )
    if len(keys) < 20:
        warnings.append(
            "Fewer than 20 parent configurations; uncertainty and plateau checks are preliminary."
        )
    if estimate > 1 + max(0.05, 3 * (stderr or 0)):
        warnings.append(
            "Estimated transmission exceeds one beyond sampling noise; inspect PMF/surface and equilibration."
        )
    valid = (
        not thermostatted
        and plateau
        and energy_diagnostics_complete
        and endpoint_error * config["beta"] <= 0.1
    )
    quality = None
    if is_surface:
        from ..analysis.surface import surface_rate_quality
        quality = surface_rate_quality(
            pmf, statistics, candidate, error,
            endpoint_error * config["beta"] if energy_diagnostics_complete else None,
            target=config.get("surface", {}).get("max_relative_standard_error", .2),
            dynamics=child_dynamics,
        )
        valid = quality["passed"]
        if not valid:
            warnings.append("Surface rate reporting prerequisites not met: "
                            + "; ".join(quality["reasons"])
                            + ". The finite-time point estimate is retained in rate_estimate_s_inv.")
    public_pmf = {k: v for k, v in pmf.items() if k not in private_pmf}
    result = {
        "task": task_name,
        "status": "plateau_estimate" if valid else "pilot_requires_convergence",
        "rate_" + unit_key: candidate if valid else None,
        "rate_estimate_" + unit_key: candidate,
        "rate_standard_error_" + unit_key: error,
        "rate_percentile_95_ci_" + unit_key: (
            np.quantile(total_boot, [0.025, 0.975]).tolist()
            if error is not None
            else None
        ),
        "conditional_rate_standard_error_" + unit_key: conditional_error,
        "rate_uncertainty": {
            "available": error is not None,
            "pmf_block_status": pmf_block_status,
            "reasons": uncertainty_reasons,
            "pmf_bootstrap_valid": len(pmf_draws),
            "recrossing_bootstrap_valid": len(boots),
            "conditional_quantity": "recrossing uncertainty with the QTST factor held fixed",
            "combination": "independent draws from the PMF and constrained-parent block bootstrap distributions, with the selected dividing surface fixed",
            "recrossing_scope": recrossing_uncertainty["scope"],
            "recrossing_support": recrossing_uncertainty,
            "parent_block_lengths": statistics["parent_block_lengths"],
        },
        "kappa": estimate,
        "kappa_standard_error": stderr,
        "plateau_detected": plateau,
        "plateau_diagnostics": statistics["plateau_diagnostics"],
        "plateau_start_au": float(times[start]),
        "plateau_drift": drift,
        "plateau_drift_standard_error": drift_error,
        "child_time_fs": float(times[-1] * ATOMIC_TIME_S / 1e-15),
        "parent_configurations": len(keys),
        "child_pairs": len(children),
        "child_trajectories": 2 * len(children),
        "maximum_scaled_energy_drift": max_drift,
        "maximum_endpoint_energy_error_hartree_per_bead": endpoint_error,
        "maximum_endpoint_energy_error_kbt": (
            endpoint_error * config["beta"] if endpoint_error is not None else None
        ),
        "energy_diagnostic_child_pairs": len(endpoint_errors),
        "pmf": public_pmf,
        "warnings": warnings,
        "converged": False,
        "convergence_note": (
            "A finite-time plateau does not establish bead, timestep, slab size, basin truncation or sampling convergence."
            if is_surface else "A detected time plateau is not bead, timestep, Rinf or statistical convergence."
        ),
        "files": {
            "pmf": str(Path(output) / ("surface_pmf.csv" if is_surface else "rate_pmf.csv")),
            "recrossing": str(Path(output) / "recrossing.csv"),
        },
    }
    if thermostatted:
        result.update(
            status="trpmd_sensitivity_estimate",
            dynamics="trpmd",
            friction_scale=friction_scale,
            centroid_friction=0.0,
            energy_conservation_applicable=False,
            maximum_endpoint_energy_change_kbt=(
                endpoint_error * config["beta"] if endpoint_error is not None else None
            ),
        )
        result["trpmd_rate_estimate_" + unit_key] = candidate
        # Do not publish thermostat heat under a numerical energy-error label.
        for name in ("maximum_scaled_energy_drift", "maximum_endpoint_energy_error_hartree_per_bead",
                     "maximum_endpoint_energy_error_kbt", "energy_diagnostic_child_pairs"):
            result.pop(name, None)
    if is_surface:
        result["rate_quality"] = quality
    result["rate_uncertainty"].update(
        pmf_bootstrap_requested=pmf_bootstrap.get("requested", 0),
        pmf_bootstrap_attempted=pmf_bootstrap.get("attempted", 0),
        pmf_bootstrap_rejected=rejected_pmf_draws,
        independence_assumption="Constrained-parent burn-in removes correlation with the umbrella endpoint used to initialize it; this is not established automatically.",
    )
    if rejected_pmf_draws:
        result["supported_draw_rate_standard_error_" + unit_key] = supported_draw_error
        result["supported_draw_rate_percentile_95_ci_" + unit_key] = (
            np.quantile(total_boot, [0.025, 0.975]).tolist()
            if supported_draw_error is not None else None
        )
        result["rate_uncertainty"]["supported_draw_condition"] = (
            "conditional on the bootstrap draw having complete PMF support "
            "and a valid estimator fit"
        )
    write_json(Path(output) / "summary.json", result)
    return result
