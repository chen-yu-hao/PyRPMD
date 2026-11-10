"""Gas-phase rate preparation, PMF estimates and parent initial configurations."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from ..analysis.fes import (
    _block_resample,
    _bootstrap_block_plan,
    _rate_bootstrap_statistics,
    umbrella_integration,
)
from ..constants import AU_RATE_TO_CM3_S
from ..input import rate_pmf_options, rate_pmf_stage
from ..propagation import combine_propagation_stats
from .prepare import prepare_rate, prepared_system
from ..checkpoint import atomic_savez as _atomic_npz
from ..io import write_report_json as write_json, report_jsonable as jsonable
from .fes import resolve_umbrellas
from .rate_constraints import rate_coordinate

FLUX_MEASURE = "delta(xi-x) * sqrt(grad(xi) M^-1 grad(xi)/(2*pi*beta))"


def _stream_state_checks(streams, coordinate):
    """Recompute xi from each stream's final ring and check its CV branch.

    The final centroid of every umbrella stream is evaluated with the same
    coordinate object used by the rate workflow. A stream on the far branch
    of xi=s0/(s0-s1) is not a sample of the requested window.
    """
    from ..cv import RPMDrateCoordinate

    if not isinstance(coordinate, RPMDrateCoordinate):
        return []
    checks = []
    for index, stream in enumerate(streams):
        q = stream.get("last_q")
        if q is None:
            continue
        record = {"stream": index, "center": float(np.asarray(stream["center"]).reshape(-1)[0])}
        for key in ("window", "replica"):
            if key in stream:
                record[key] = int(stream[key])
        if "window_id" in stream:
            record["window_id"] = str(stream["window_id"])
        try:
            branch = coordinate.branch(q)
        except (ValueError, FloatingPointError) as error:
            branch = {"branch": "unavailable", "regular": False, "error": str(error)}
        record.update(final_centroid=branch)
        samples = np.asarray(stream.get("samples", [])).reshape(-1)
        if len(samples):
            record["last_recorded_xi"] = float(samples[-1])
            record["stream_mean_xi"] = float(samples.mean())
        record["passed"] = bool(branch.get("regular"))
        if not record["passed"]:
            record["reason"] = ("rpmdrate_xi_beyond_singular_surface"
                                if branch["branch"] == "beyond_singular_surface"
                                else "final_centroid_xi_unavailable")
        checks.append(record)
    return checks


def _adaptive_production_gate(summary):
    """Pilot acceptance selects windows; only frozen production may pass."""
    adaptive = summary.get("adaptive_umbrella")
    if not adaptive or "production_target_met" not in adaptive:
        return None
    gate = {
        "name": "adaptive_frozen_production",
        "passed": bool(adaptive["production_target_met"]),
        "pilot_target_met": bool(adaptive.get("pilot_target_met", adaptive.get("target_met"))),
        "spanning_overlap": adaptive.get("production_spanning_overlap"),
        "overlap_connected_at_target": adaptive.get("production_overlap_connected_at_target"),
        "effective_samples_target_met": adaptive.get("production_effective_samples_target_met"),
        "effective_samples_per_window": adaptive.get("production_effective_samples_per_window"),
        "analysis_overlap_connected": adaptive.get("production_analysis_overlap_connected"),
    }
    coverage = adaptive.get("production_coverage")
    if coverage is not None:
        gate["coverage_target_met"] = bool(coverage.get("target_met"))
    return gate


def _prepare(config, system, output, return_system=False, *, _preparation=None, _write=True):
    rate_pmf_options(config)
    prepared = (prepare_rate(config, system, numerical=True)
                if _preparation is None else copy.deepcopy(_preparation))
    if _write:
        write_json(Path(output) / "preparation.json", prepared)
    if not prepared["ready"]:
        raise ValueError(
            "Reaction preparation is not ready: "
            + "; ".join(prepared["diagnostics"])
            + ". See preparation.json; inspect the reaction definition and TS validation/search results."
        )
    system = prepared_system(system, prepared)
    result = configure_rate_umbrella(config, system, prepared["reaction"])
    if rate_pmf_options(result)["source"] is not None:
        from .rate_source import read_source

        read_source(result, system)
    elif result["reaction"].get("definition") and rate_pmf_stage(result, system) is None:
        from .fes import validate_fes_request

        # In inference mode the coordinate was unavailable during structural
        # input preflight. Validate its complete sampling plan now, before any
        # production trajectory starts.
        validate_fes_request(result, system)
    return (result, system) if return_system else result


def configure_rate_umbrella(config, system, reaction):
    """Build and validate a rate window plan without evaluating the PES.

    The caller must supply an explicitly defined or already inferred reaction.
    Reaction definition and numerical TS validation are deliberately independent:
    this function performs no preparation, numerical check, or output I/O.
    """
    result = copy.deepcopy(config)
    result["reaction"] = copy.deepcopy(reaction)
    reaction = result["reaction"]
    rinf = reaction.get("rinf_au", reaction.get("rinf", reaction.get("Rinf", "12 angstrom")))
    result["cvs"] = [
        {
            "name": "xi",
            "kind": "rpmdrate_xi",
            "estimator": "centroid",
            "fragments": reaction["fragments"],
            "forming_bonds": reaction["forming_bonds"],
            "breaking_bonds": reaction["breaking_bonds"],
            "rinf": rinf,
        }
    ]
    if reaction.get("equivalent_paths"):
        result["cvs"][0]["equivalent_paths"] = reaction["equivalent_paths"]
    if result.get("analysis", {}).get("measure", "rate_flux") != "rate_flux":
        raise ValueError(
            "standard RPMD rate sampling requires analysis.measure='rate_flux'"
        )
    pmf_options = rate_pmf_options(result)
    if pmf_options["source"] is not None:
        return result
    stage = rate_pmf_stage(result, system)
    if stage is not None:
        from .fes_dispatch import validate_fes

        validate_fes(stage, system)
        if pmf_options.get("sampler", "umbrella") == "umbrella":
            _, _, centers, _ = resolve_umbrellas(stage, system)
            _check_pmf_domain(centers[:, 0], pmf_options)
            if stage.get("adaptive_umbrella"):
                from .adaptive_umbrella import validate_adaptive_umbrella

                validate_adaptive_umbrella(stage, system)
        return result
    umbrella = result.setdefault("umbrella", {})
    new_input = (result.get("schema_version", 1) >= 2 or reaction.get("definition")
                 or any(key in umbrella for key in ("coordinate", "regions", "windows", "force_constant", "force_matrix")))
    if new_input and umbrella.get("cvs", ["xi"]) != ["xi"]:
        raise ValueError("gas-phase rate umbrellas require the reaction coordinate xi")
    selectors = ("centers", "centers_au", "grid", "start", "regions", "windows")
    if new_input:
        selectors += ("stop", "count")
    if not any(k in umbrella for k in selectors):
        umbrella.update(start=-0.05, stop=1.05, count=23)
    if not any(k in umbrella for k in ("stiffness", "stiffness_au", "kappa", "force_constant", "force_matrix")):
        umbrella["kappa"] = 50.0 / result["beta"]
    umbrella["cvs"] = ["xi"]
    result.setdefault("analysis", {}).update(measure="rate_flux")
    result["analysis"].setdefault("bins", 100)
    result["analysis"].setdefault(
        "bootstrap",
        result["analysis"].get(
            "bootstrap_replicates", result["analysis"].get("bootstrap_samples", 20)
        ),
    )
    _, _, centers, _ = resolve_umbrellas(result, system)
    _check_pmf_domain(centers[:, 0], pmf_options)
    return result


def _check_pmf_domain(centers, options):
    centers = np.asarray(centers, float)
    if centers.min() > 0 or centers.max() < 1:
        raise ValueError("rate umbrellas must cover xi=0 (reactants) through xi=1 (TS)")
    if options["xi_star"] is not None and not 0 <= options["xi_star"] <= centers.max():
        raise ValueError("dividing surface outside sampled PMF range")
    if (
        options["search_range"] is not None
        and options["search_range"][1] > centers.max()
    ):
        raise ValueError(
            "rate.pmf_search_range extends beyond the umbrella-center range"
        )


def _histogram_endpoint_support(data, value):
    """Observed histogram support, used to qualify the Gaussian UI model."""
    edges = np.asarray(data["edges"][0])
    axis = 0.5 * (edges[:-1] + edges[1:])
    mask = np.asarray(data["mask"], bool).reshape(-1)
    index = int(np.searchsorted(axis, value))
    if index < len(axis) and np.isclose(axis[index], value, rtol=0, atol=1e-14):
        bracket = [index]
    elif 0 < index < len(axis):
        bracket = [index - 1, index]
    else:
        return {
            "coordinate": value,
            "supported": False,
            "reason": "outside_bin_centers",
        }
    supported = not mask[bracket].any()
    return {
        "coordinate": value,
        "supported": bool(supported),
        "reason": None if supported else "masked_endpoint_bracket",
        "bracket_centers": axis[bracket].tolist(),
        "raw_count": np.asarray(data["raw_count"]).reshape(-1)[bracket].tolist(),
    }


def _ui_rate_pmf(sampled, config, centers, samples, stiffness, options):
    """Retain the existing Gaussian UI estimator with explicit diagnostics."""
    grid = np.linspace(0.0, centers.max(), options["points"])
    curve, derivative = umbrella_integration(
        samples, centers, stiffness, config["beta"], grid
    )
    if not np.isfinite(curve).all() or not np.isfinite(derivative).all():
        raise FloatingPointError(
            "nonfinite Gaussian UI curve; inspect the umbrella streams"
        )
    interval = options["search_range"] or [0.0, float(centers.max())]
    eligible = np.flatnonzero((grid >= interval[0]) & (grid <= interval[1]))
    if not len(eligible):
        raise ValueError("rate.pmf_search_range contains no UI grid points")
    xi = options["xi_star"]
    if xi is None:
        xi = float(grid[eligible[np.argmax(curve[eligible])]])
    barrier = float(np.interp(xi, grid, curve) - curve[0])
    connected = bool(sampled["data"]["connected"])
    gs = np.asarray(sampled["data"]["statistical_inefficiency"])
    block_plan = _bootstrap_block_plan(np.asarray([len(s) for s in samples]), gs)
    rng = np.random.default_rng(np.random.SeedSequence([options["analysis_seed"], 401]))
    barriers, curves, records, endpoints, failures = [], [], [], [], []
    if connected:
        for draw in range(options["bootstrap"]):
            record = {"draw": draw, "valid": False, "reason": None}
            draws = [
                _block_resample(s, block, rng)
                for s, block in zip(samples, block_plan["block_lengths"])
            ]
            try:
                fb, _ = umbrella_integration(
                    draws, centers, stiffness, config["beta"], grid
                )
                rv, sv = float(fb[0]), float(np.interp(xi, grid, fb))
                if not np.isfinite(fb).all() or not np.isfinite(sv - rv):
                    raise FloatingPointError("nonfinite UI bootstrap curve")
                barriers.append(sv - rv)
                curves.append(fb - rv)
                endpoints.append([rv, sv])
                record.update(
                    valid=True,
                    reference_free_energy_hartree=rv,
                    surface_free_energy_hartree=sv,
                    barrier_hartree=sv - rv,
                )
            except (ValueError, FloatingPointError) as error:
                record.update(reason="ui_fit_failed", message=str(error))
                failures.append(
                    {key: value for key, value in record.items() if key != "valid"}
                )
            records.append(record)
    standard_error = (
        np.std(curves, axis=0, ddof=1)
        if len(curves) >= 2
        else np.full_like(curve, np.nan)
    )
    if block_plan["status"] == "unavailable":
        standard_error[:] = np.nan
    supported_standard_error = standard_error.copy()
    stats, supported_stats, uncertainty = _rate_bootstrap_statistics(
        barriers, requested=options["bootstrap"], attempted=len(records),
        block_status=block_plan["status"],
    )
    if failures:
        standard_error[:] = np.nan
    empirical = {
        "reference": _histogram_endpoint_support(sampled["data"], 0.0),
        "dividing_surface": _histogram_endpoint_support(sampled["data"], xi),
    }
    warnings = [
        "Gaussian UI models each biased window by its mean and variance; inspect non-Gaussian window distributions or compare with MBAR."
    ]
    warnings.extend(block_plan["warnings"])
    if not all(point["supported"] for point in empirical.values()):
        warnings.append(
            "A Gaussian UI endpoint lacks observed histogram support; the model value is not an interpolation of measured supported bins."
        )
    if options["bootstrap"] and len(barriers) < 2:
        warnings.append(
            "Fewer than two valid UI bootstrap draws; barrier uncertainty is unavailable."
        )
    if failures:
        warnings.append(
            "Some UI bootstrap draws failed; primary PMF uncertainty is unavailable. "
            "Supported-draw errors are conditional on a valid estimator fit; "
            "inspect the recorded reasons and valid fraction."
        )
    if not connected:
        warnings.append(
            "Disconnected overlap does not identify the PMF barrier; no recrossing trajectories are launched."
        )
    return {
        "ready": connected,
        "status": "ready" if connected else "disconnected_overlap",
        "xi_star": xi,
        "reference": 0.0,
        "free_energy_barrier_hartree": barrier if connected else None,
        "barrier_standard_error_hartree": stats["standard_error_hartree"],
        "barrier_percentile_95_ci_hartree": stats["percentile_95_ci_hartree"],
        "barrier_uncertainty": uncertainty,
        **({
            "supported_draw_barrier_standard_error_hartree": supported_stats["standard_error_hartree"],
            "supported_draw_barrier_percentile_95_ci_hartree": supported_stats["percentile_95_ci_hartree"],
            "supported_draw_standard_error": supported_standard_error,
        } if failures else {}),
        "pmf_bootstrap_barriers": barriers,
        "xi": grid,
        "free_energy": curve,
        "standard_error": standard_error,
        "derivative": derivative,
        "mask": np.full(len(grid), not connected),
        "overlap_connected": connected,
        "support": {
            "model": "Gaussian-window integration within the umbrella-center range",
            "reference": {"coordinate": 0.0, "model_supported": connected},
            "dividing_surface": {"coordinate": xi, "model_supported": connected},
            "observed_histogram": empirical,
            "search_range": interval,
            "selection": (
                "fixed"
                if options["xi_star"] is not None
                else "maximum_gaussian_ui_grid_point"
            ),
            "failure_reason": None if connected else "disconnected_overlap",
        },
        "bootstrap": {
            "requested": options["bootstrap"],
            "attempted": len(records),
            "valid": len(barriers),
            "valid_fraction": (
                len(barriers) / options["bootstrap"] if options["bootstrap"] else None
            ),
            "failures": failures,
            "failure_counts": {"ui_fit_failed": len(failures)} if failures else {},
            "draws": records,
            "surface_held_fixed": True,
            "skipped_reason": None if connected else "disconnected_overlap",
            "block_lengths": block_plan["block_lengths"],
            "block_plan": block_plan,
            "endpoint_values_hartree": np.asarray(endpoints).reshape(-1, 2),
            "endpoint_covariance_hartree_squared": (
                np.cov(np.asarray(endpoints).T, ddof=1) if len(endpoints) >= 2 else None
            ),
            "curve_valid_draws": np.full(len(grid), len(curves)),
        },
        "effective_samples_per_stream": sampled["data"]["effective_samples_per_stream"],
        "weight_ess": sampled["data"]["weight_ess"],
        "warnings": warnings,
    }


def _save_pmf_arrays(output, estimate, method):
    grid = np.asarray(estimate["xi"])
    mask = np.asarray(estimate["mask"], bool)
    curve = np.where(mask, np.nan, estimate["free_energy"])
    error = np.where(mask, np.nan, estimate["standard_error"])
    arrays = {
        "xi": grid,
        "free_energy": curve,
        "standard_error": error,
        "mask": mask,
        "standard_error_mask": mask | ~np.isfinite(error),
        "bootstrap_barriers": np.asarray(estimate["pmf_bootstrap_barriers"]),
        "method": method,
        "sampling_measure": "rate_flux",
        "xi_star": np.asarray(
            estimate["xi_star"] if estimate["xi_star"] is not None else np.nan
        ),
        "reference": 0.0,
        "ready": bool(estimate["ready"]),
        "diagnostic_only": not bool(estimate["ready"]),
    }
    for key in (
        "derivative",
        "log_density",
        "density",
        "raw_count",
        "edges",
        "overlap",
        "components",
        "window_free_energies",
    ):
        if key in estimate:
            arrays[key] = estimate[key]
    if "supported_draw_standard_error" in estimate:
        arrays["supported_draw_standard_error"] = np.where(
            mask, np.nan, estimate["supported_draw_standard_error"]
        )
    _atomic_npz(Path(output) / "rate_pmf.npz", **arrays)
    columns = [grid, curve]
    names = ["xi", "free_energy_hartree"]
    if "derivative" in estimate:
        columns.append(np.where(mask, np.nan, estimate["derivative"]))
        names.append("derivative_hartree")
    columns.extend([error, mask.astype(int), (mask | ~np.isfinite(error)).astype(int)])
    names.extend(["standard_error_hartree", "mask", "standard_error_mask"])
    np.savetxt(
        Path(output) / "rate_pmf.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(names),
        comments="",
    )


def _write_stream_anomalies(output, anomalies, branch_failures, config):
    """Persist a structured failure artifact for inconsistent umbrella streams."""
    from ..analysis.rate_support import ANOMALY_CHECKS

    write_json(Path(output) / "umbrella_stream_anomalies.json", jsonable({
        "schema_version": 1,
        "status": "umbrella_stream_anomaly",
        "temperature_beta_inverse_hartree": config["beta"],
        "beads": config.get("beads"),
        "displaced_streams": anomalies,
        "cv_branch_failures": branch_failures,
        "checks": ANOMALY_CHECKS,
        "rate_reported": False,
        "note": "These streams are excluded from any final rate; continuing them is not a convergence remedy.",
    }))


def _pmf_and_parent_tasks(sampled, config, system, output, options):
    pmf_options = rate_pmf_options(config)
    if pmf_options["source"] is not None:
        from .rate_source import source_estimate

        estimate = source_estimate(config, system)
        method = "source"
        centers = np.array([estimate["xi_star"] or 0.0])
        streams = [{"last_q": np.repeat(system.geometry[None], config["beads"], axis=0)}]
        sampled = {"summary": {"warnings": []}}
    elif pmf_options.get("sampler", "umbrella") in ("metadynamics", "opes"):
        from .rate_curve import estimate_fes_curve

        surface = sampled["rate_surface"]
        sampler = pmf_options["sampler"]
        if (surface.method != sampler or surface.metadata.get("reweighted") is not True
                or surface.metadata.get("estimator") != "reweighted_histogram"):
            raise ValueError("Built-in enhanced rate PMFs require an authenticated frozen-bias reweighted surface")
        estimate = estimate_fes_curve(
            surface, pmf_options,
            uncertainty_reason="frozen_bias_pmf_correlated_barrier_uncertainty_unavailable",
            strict_search=True,
            warnings=[
                "The rate PMF uses only frozen-bias production samples reweighted to the rate-flux measure.",
                "Correlated paired-endpoint uncertainty is not implemented for frozen-bias PMFs; barrier and total-rate uncertainty are unavailable. Independent-history spread is not a standard error.",
            ],
        )
        method = "reweighted_histogram"
        streams = [{"last_q": s["state"].q} for s in sampled["streams"]]
        coordinate = rate_coordinate(config, system)
        centers = np.array([float(np.asarray(coordinate.value(s["last_q"])).reshape(-1)[0]) for s in streams])
        if not streams:
            raise ValueError("Enhanced rate PMF did not return any completed sampling streams")
    else:
        # Sampling deliberately returns no production FES. Materialize an
        # external executor's one-shot iterator exactly once on the coordinator.
        streams = sampled["streams"] = list(sampled["streams"])
        if not streams:
            raise ValueError("Rate PMF analysis requires at least one umbrella stream")
        centers = np.array([s["center"][0] for s in streams])
        samples = [s["samples"][:, 0] for s in streams]
        stiffness = np.array([s["stiffness"][0, 0] for s in streams])
        _check_pmf_domain(centers, pmf_options)
        method = pmf_options["method"]
        if method == "mbar":
            from ..analysis.fes import mbar_rate_pmf

            estimate = mbar_rate_pmf(
                samples,
                centers,
                stiffness,
                config["beta"],
                xi_star=pmf_options["xi_star"],
                reference=0.0,
                bins=pmf_options["bins"],
                ranges=pmf_options["ranges"],
                search_range=pmf_options["search_range"],
                bootstrap=pmf_options["bootstrap"],
                seed=np.random.SeedSequence([pmf_options["analysis_seed"], 401]),
                min_count=pmf_options["min_count"],
                overlap_threshold=pmf_options["overlap_threshold"],
                compare_ui=pmf_options["compare_ui"],
                ui_points=pmf_options["points"],
                measure="rate_flux",
                min_weight_ess=pmf_options.get("min_weight_ess"),
                min_endpoint_ess=pmf_options.get("min_endpoint_ess"),
                max_center_displacement_sigma=pmf_options.get("max_center_displacement_sigma"),
            )
        else:
            if "sampling_config" in sampled or not sampled.get("data"):
                from .rate_umbrella_analysis import ui_umbrella_diagnostics

                sampled["data"] = ui_umbrella_diagnostics(streams, config, pmf_options)
            estimate = _ui_rate_pmf(
                sampled, config, centers, samples, stiffness, pmf_options
            )
            from ..analysis.rate_support import window_anomalies

            anomalies = window_anomalies(
                samples, centers, stiffness, config["beta"],
                pmf_options.get("max_center_displacement_sigma"))
            estimate["window_anomalies"] = anomalies
            estimate["gates"] = [{
                "name": "umbrella_stream_consistency", "passed": not anomalies,
                "threshold_sigma": pmf_options.get("max_center_displacement_sigma"),
                "anomalous_streams": [a["stream"] for a in anomalies],
            }]
        if "sampling_config" in sampled:
            from .rate_umbrella_analysis import (
                mbar_umbrella_diagnostics, write_rate_umbrella_diagnostics,
            )

            data = (
                mbar_umbrella_diagnostics(estimate, streams, config["beta"], pmf_options["min_count"])
                if method == "mbar" else sampled["data"]
            )
            write_rate_umbrella_diagnostics(
                sampled, data, system, method, solver=estimate.get("solver"),
            )
        else:
            # Historical Python callers may supply an already analyzed bundle.
            sampled.setdefault("summary", {"warnings": []})
        if config.get("output", {}).get("centroid_path"):
            from ..analysis.centroid_path import write_centroid_path

            write_centroid_path(
                output, streams, masses=system.masses, reference=system.geometry,
                permutations=rate_coordinate(config, system).permutations,
                beta=config["beta"], nbeads=config["beads"],
            )
    # Hard scientific gates that are independent of the estimator's own
    # support. A failed gate keeps any curve strictly diagnostic.
    gates = list(estimate.get("gates", []))
    gate_reason = None
    if method not in ("source", "reweighted_histogram"):
        state_checks = _stream_state_checks(streams, rate_coordinate(config, system))
        estimate["stream_state_checks"] = state_checks
        failed = [c for c in state_checks if not c["passed"]]
        gates.append({"name": "umbrella_cv_branch", "passed": not failed,
                      "failed_streams": [c["stream"] for c in failed]})
        if estimate.get("window_anomalies") or failed:
            gate_reason = "umbrella_stream_anomaly"
            _write_stream_anomalies(output, estimate.get("window_anomalies", []),
                                    failed, config)
        adaptive_gate = _adaptive_production_gate(sampled["summary"])
        if adaptive_gate is not None:
            gates.append(adaptive_gate)
            if not adaptive_gate["passed"]:
                gate_reason = gate_reason or "adaptive_production_targets_not_met"
    if gate_reason and estimate["ready"]:
        estimate["diagnostic_free_energy_barrier_hartree"] = estimate["free_energy_barrier_hartree"]
        estimate["free_energy_barrier_hartree"] = None
        estimate["ready"] = False
        estimate["status"] = gate_reason
        estimate["warnings"] = list(estimate["warnings"]) + [
            "Rate PMF unavailable: " + gate_reason + "; the curve is diagnostic only and no recrossing is started."]
    elif gate_reason and estimate["status"] in ("ready", None):
        estimate["status"] = gate_reason
    estimate["gates"] = gates
    _save_pmf_arrays(output, estimate, method)
    xi = estimate["xi_star"]
    barrier = estimate["free_energy_barrier_hartree"]
    fragments = config["reaction"]["fragments"]
    ma, mb = [system.masses[list(f)].sum() for f in fragments]
    reduced = ma * mb / (ma + mb)
    rinf = config["reaction"]["rinf"]
    prefactor = 4 * np.pi * rinf * rinf / np.sqrt(2 * np.pi * config["beta"] * reduced)
    electronic = float(config["reaction"].get("electronic_weight", 1.0))
    symmetry = float(config["reaction"].get("symmetry_factor", 1.0))
    if not np.isfinite(symmetry) or symmetry <= 0:
        raise ValueError("reaction.symmetry_factor must be positive")
    qtst = (
        float(prefactor * np.exp(-config["beta"] * barrier) * electronic * symmetry)
        if estimate["ready"]
        else None
    )
    if qtst is not None and not np.isfinite(qtst):
        raise FloatingPointError("nonfinite QTST estimate; inspect PMF sampling")
    source_status = estimate["status"]
    status = (
        "ready"
        if estimate["ready"]
        else (
            "insufficient_overlap"
            if source_status == "disconnected_overlap"
            else (
                "pmf_solver_failed"
                if source_status == "mbar_not_converged"
                else "unsupported_pmf"
            )
        )
    )
    diagnostics = {
        "schema_version": 2,
        "method": method,
        "sampler": pmf_options.get("sampler", "umbrella"),
        "estimator": method,
        "ready": bool(estimate["ready"]),
        "status": status,
        "failure_reason": None if estimate["ready"] else source_status,
        "curve_role": "rate" if estimate["ready"] else "diagnostic_only",
        "xi_star": xi,
        "reference": 0.0,
        "free_energy_barrier_hartree": barrier,
        "barrier_standard_error_hartree": estimate["barrier_standard_error_hartree"],
        "barrier_percentile_95_ci_hartree": estimate[
            "barrier_percentile_95_ci_hartree"
        ],
        "k_qtst_au": qtst,
        "k_qtst_cm3_s": qtst * AU_RATE_TO_CM3_S if qtst is not None else None,
        "reactant_prefactor_au": prefactor,
        "electronic_weight": electronic,
        "symmetry_factor": symmetry,
        "rinf_bohr": rinf,
        "reduced_mass_au": float(reduced),
        "pmf_method": (
            "Saved, validated rate-flux FESResult"
            if method == "source" else (
                "Frozen-bias reweighted histogram with the rate-flux metric"
                if method == "reweighted_histogram" else
                "Gaussian umbrella integration; MBAR overlap checked"
                if method == "ui" else "Histogram MBAR with paired endpoint bootstrap"
            )
        ),
        "measure": FLUX_MEASURE,
        "sampling_measure": "rate_flux",
        "overlap_connected": bool(estimate["overlap_connected"]),
        "pmf_bootstrap_barriers": estimate["pmf_bootstrap_barriers"],
        "warnings": list(
            dict.fromkeys(sampled["summary"]["warnings"] + estimate["warnings"])
        ),
        "intermediate_fes_bootstrap": 0,
        "analysis_owner": "rate",
        "intermediate_fes_purpose": (
            "Validated external FESResult; no intermediate umbrella sampling."
            if method == "source" else
            "Bias training and relaxation are discarded; only fixed-bias production enters the rate PMF."
            if method == "reweighted_histogram" else
            "Rate-owned umbrella diagnostics and sample export; sampling does not reconstruct a FES. "
            "Only the selected rate PMF estimator performs barrier bootstrap."
        ),
        "propagation": {
            **combine_propagation_stats(
                [dict(s.get("propagation", {})) for s in streams
                 if isinstance(s, dict) and s.get("propagation") is not None]),
            "streams": [dict(s.get("propagation", {})) for s in streams
                        if isinstance(s, dict) and s.get("propagation") is not None],
        },
    }
    if "pmf" in config.get("rate", {}):
        from .rate_plan import rate_pmf_plan

        diagnostics["plan"] = rate_pmf_plan(config)
    for key in (
        "support",
        "solver",
        "bootstrap",
        "window_moments",
        "ui_comparison",
        "effective_samples_per_stream",
        "weight_ess",
        "source",
        "source_method",
        "source_reweighted",
        "source_sha256",
        "barrier_uncertainty",
        "supported_draw_barrier_standard_error_hartree",
        "supported_draw_barrier_percentile_95_ci_hartree",
        "diagnostic_free_energy_barrier_hartree",
        "weight_diagnostics",
        "window_anomalies",
        "stream_state_checks",
        "gates",
    ):
        if key in estimate:
            diagnostics[key] = estimate[key]
    if "adaptive_umbrella" in sampled["summary"]:
        diagnostics["adaptive_umbrella"] = copy.deepcopy(sampled["summary"]["adaptive_umbrella"])
    diagnostics["uncertainty"] = dict(estimate["barrier_uncertainty"])
    # Retain the historical singular reason while exposing all support limits.
    diagnostics["uncertainty"]["reason"] = diagnostics["uncertainty"].get("reason") or (
        None if diagnostics["uncertainty"]["available"] else (
            "pmf_bootstrap_contains_unsupported_draws"
            if diagnostics["uncertainty"]["bootstrap_rejected"] else (
                "bootstrap_disabled" if pmf_options["bootstrap"] == 0 else (
                    "insufficient_correlated_blocks"
                    if diagnostics["uncertainty"]["block_status"] == "unavailable"
                    else "fewer_than_two_valid_bootstrap_draws"
                )
            )
        )
    )
    diagnostics = jsonable(diagnostics)
    write_json(Path(output) / "rate_pmf.json", diagnostics)
    if not estimate["ready"]:
        return diagnostics, []
    nearest = np.argsort(abs(centers - xi))
    permutations = rate_coordinate(config, system).permutations
    tasks = []
    for chain in range(options["parent_replicas"]):
        source = streams[nearest[(chain // len(permutations)) % len(nearest)]]["last_q"]
        initial_q = np.empty_like(source)
        initial_q[:, list(permutations[chain % len(permutations)]), :] = source
        tasks.append(
            {
                "config": config,
                "system": system,
                "options": options,
                "chain": chain,
                "xi_star": xi,
                "initial_q": initial_q,
                "output": str(output),
            }
        )
    return diagnostics, tasks
