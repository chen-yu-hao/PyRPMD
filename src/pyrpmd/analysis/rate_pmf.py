"""Histogram-MBAR rate barriers and paired uncertainty diagnostics.

The estimator consumes samples already drawn in the rate-flux measure.
It never changes the sampling measure or infers physical convergence.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import os

import numpy as np
from threadpoolctl import threadpool_limits

from .fes_statistics import (
    CV_CORRELATION_OBSERVABLES,
    _block_resample,
    _bootstrap_block_plan,
    cv_statistical_inefficiency,
)
from .mbar import _identical_rows, solve_mbar
from .rate_support import (
    DEFAULT_MAX_CENTER_DISPLACEMENT_SIGMA,
    DEFAULT_MIN_ENDPOINT_ESS,
    DEFAULT_MIN_WEIGHT_ESS,
    classify_point,
    fit_log_weights,
    histogram_report,
    kish_ess,
    log_histogram,
    masked_runs,
    weight_diagnostics,
    window_anomalies,
)
from .umbrella import umbrella_integration, umbrella_reduced_potentials


def _ordered_thread_map(function, tasks, max_workers):
    """Run independent analysis tasks with bounded, input-ordered results.

    ``Executor.map`` eagerly consumes its iterable. Bootstrap tasks can carry
    large index arrays, so eager submission would retain every draw in memory.
    Keep at most one task per worker in flight while still returning results in
    draw order. This also lets the coordinator start consuming results before
    all requested draws have been generated.
    """
    iterator = iter(tasks)
    results = []
    pending = {}
    next_index = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while len(pending) < max_workers:
            try:
                task = next(iterator)
            except StopIteration:
                break
            pending[pool.submit(function, task)] = next_index
            next_index += 1
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            completed = sorted(done, key=pending.__getitem__)
            for future in completed:
                index = pending.pop(future)
                while len(results) <= index:
                    results.append(None)
                results[index] = future.result()
                try:
                    task = next(iterator)
                except StopIteration:
                    continue
                pending[pool.submit(function, task)] = next_index
                next_index += 1
    return results


def _mbar_bootstrap_draw(task):
    """Evaluate one rate-PMF bootstrap draw.

    The task is intentionally a module-level function so the hot loop has a
    single implementation for serial and threaded execution. All large arrays
    live in the shared context and are never copied between worker threads.
    """
    draw, indices, context = task
    u = context["u"]
    x = context["x"]
    counts = context["fit_counts"]
    record = {"draw": draw, "valid": False, "reason": None}
    endpoints = barrier = curve = ui_barrier = None
    ui_failure = None
    try:
        mb = context["solve_mbar"](
            u[:, indices],
            counts,
            overlap_threshold=context["overlap_threshold"],
            initial_free_energies=context["initial_free_energies"],
            return_state_weights=False,
            **context["solver_options"],
        )
        record["solver_residual"] = float(mb.residual)
        if not mb.converged:
            record["reason"] = "mbar_not_converged"
        elif len(np.unique(mb.components)) != 1:
            record["reason"] = "disconnected_overlap"
        else:
            xb = x[indices]
            absolute, _, _, _, maskb = _log_free_energy(
                xb,
                mb,
                context["edges"],
                context["beta"],
                context["min_count"],
            )
            rb, sb, record["reason"] = _rate_support(
                context["grid"], maskb, context["reference"], context["xi_star"]
            )
            if record["reason"] is None:
                rv, sv = _rate_value(absolute, rb), _rate_value(absolute, sb)
                barrier = sv - rv
                endpoints = [rv, sv]
                curve = absolute - rv
                record.update(
                    valid=True,
                    reference_free_energy_hartree=rv,
                    surface_free_energy_hartree=sv,
                    barrier_hartree=barrier,
                )
                if context["ui_ready"]:
                    try:
                        ui_curve, _ = context["integrate_ui"](
                            [
                                xb[start:stop]
                                for start, stop in zip(
                                    context["offsets"][:-1], context["offsets"][1:]
                                )
                            ],
                            context["centers"],
                            context["stiffness"],
                            context["beta"],
                            context["ui_grid"],
                        )
                        ui_barrier = float(
                            np.interp(context["xi_star"], context["ui_grid"], ui_curve)
                            - np.interp(context["reference"], context["ui_grid"], ui_curve)
                        )
                        record["ui_barrier_hartree"] = ui_barrier
                    except (ValueError, FloatingPointError) as exc:
                        ui_failure = {
                            "draw": draw,
                            "reason": "ui_fit_failed",
                            "message": str(exc),
                        }
    except (FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
        record["reason"] = "mbar_solver_error"
        record["message"] = str(exc)
    return {
        "record": record,
        "endpoints": endpoints,
        "barrier": None if not record["valid"] else record["barrier_hartree"],
        "curve": curve,
        "ui_barrier": ui_barrier,
        "ui_difference": (
            None
            if ui_barrier is None or not record["valid"]
            else record["barrier_hartree"] - ui_barrier
        ),
        "ui_failure": ui_failure,
    }


def _resolve_bootstrap_workers(requested, bootstrap):
    """Resolve the bounded worker count for independent bootstrap draws."""
    if requested is not None:
        if (
            isinstance(requested, (bool, np.bool_))
            or not isinstance(requested, (int, np.integer))
            or requested < 1
        ):
            raise ValueError("bootstrap_workers must be a positive integer or None")
        return min(int(requested), int(bootstrap)) if bootstrap else 1
    # Small jobs are faster and easier to debug in-process. The default still
    # parallelizes the long production/post-processing runs that motivated this
    # path (typically 20-200 MBAR draws) without creating one worker per draw.
    if bootstrap < 16:
        return 1
    return min(int(bootstrap), max(1, min(4, os.cpu_count() or 1)))


def _log_free_energy(x, fit, edges, beta, min_count):
    """Absolute histogram free energy from log-domain weighted bin mass."""
    _, log_density, count, mask = log_histogram(x, fit_log_weights(fit), edges, min_count)
    absolute = np.full(len(count), np.nan)
    absolute[~mask] = -log_density[~mask] / beta
    with np.errstate(over="ignore", under="ignore"):
        density = np.exp(log_density)
    return absolute, density, log_density, count, mask


def _rate_bracket(grid, mask, point, name):
    exact = np.flatnonzero(np.isclose(grid, point, rtol=0, atol=1e-14))
    if len(exact):
        i = int(exact[0])
        return ((i, i, 0.0), None) if not mask[i] else (None, name + "_masked")
    right = int(np.searchsorted(grid, point))
    if right == 0 or right == len(grid):
        return None, name + "_outside_histogram_centers"
    left = right - 1
    if mask[left] or mask[right]:
        return None, name + "_masked"
    return (left, right, float((point - grid[left]) / (grid[right] - grid[left]))), None


def _rate_value(values, bracket):
    left, right, fraction = bracket
    return float((1 - fraction) * values[left] + fraction * values[right])


def _rate_support(grid, mask, reference, surface):
    ref, reason = _rate_bracket(grid, mask, reference, "reference")
    if reason:
        return ref, None, reason
    target, reason = _rate_bracket(grid, mask, surface, "dividing_surface")
    if reason:
        return ref, target, reason
    lo, hi = min(ref[0], target[0]), max(ref[1], target[1])
    if mask[lo : hi + 1].any():
        return ref, target, "masked_interval_between_endpoints"
    return ref, target, None


def _rate_draw_statistics(values):
    values = np.asarray(values, float)
    return {
        "valid": len(values),
        "standard_error_hartree": (
            float(values.std(ddof=1)) if len(values) >= 2 else None
        ),
        "percentile_95_ci_hartree": (
            np.quantile(values, [0.025, 0.975]).tolist() if len(values) >= 2 else None
        ),
    }


def _rate_bootstrap_statistics(values, *, requested, attempted, block_status):
    """Separate complete-draw uncertainty from a selected subset of draws.

    A failed PMF fit or a missing histogram bin is part of the bootstrap
    experiment. Discarding that draw cannot establish an unconditional error
    bar. Keep the surviving distribution for explicitly conditional diagnostics.
    """
    supported = _rate_draw_statistics(values)
    rejected = max(0, attempted - supported["valid"])
    reasons = []
    if not requested:
        reasons.append("bootstrap_disabled")
    elif supported["valid"] < 2:
        reasons.append("fewer_than_two_valid_bootstrap_draws")
    if block_status in ("limited_blocks", "unavailable"):
        reasons.append("pmf_bootstrap_" + block_status)
    if block_status == "unavailable":
        supported["standard_error_hartree"] = None
        supported["percentile_95_ci_hartree"] = None
    primary = dict(supported)
    if rejected:
        reasons.append("pmf_bootstrap_contains_unsupported_draws")
        primary["standard_error_hartree"] = None
        primary["percentile_95_ci_hartree"] = None
    diagnostics = {
        "available": primary["standard_error_hartree"] is not None,
        "block_status": block_status,
        "reasons": reasons,
        "bootstrap_requested": int(requested),
        "bootstrap_attempted": int(attempted),
        "bootstrap_valid": supported["valid"],
        "bootstrap_rejected": rejected,
        "surface_held_fixed": True,
        "selection_uncertainty_included": False,
    }
    if rejected:
        diagnostics["supported_draw_condition"] = (
            "conditional on the bootstrap draw having complete PMF support "
            "and a valid estimator fit"
        )
    return primary, supported, diagnostics


def mbar_rate_pmf(
    samples,
    centers,
    stiffness,
    beta,
    *,
    xi_star=None,
    reference=0.0,
    bins=100,
    ranges=None,
    search_range=None,
    bootstrap=40,
    seed=0,
    min_count=5,
    overlap_threshold=1e-6,
    measure="rate_flux",
    compare_ui=False,
    ui_points=401,
    solver_options=None,
    bootstrap_workers=None,
    min_weight_ess=DEFAULT_MIN_WEIGHT_ESS,
    min_endpoint_ess=DEFAULT_MIN_ENDPOINT_ESS,
    max_center_displacement_sigma=DEFAULT_MAX_CENTER_DISPLACEMENT_SIGMA,
    _solve_mbar=solve_mbar,
    _resample=_block_resample,
    _integrate_ui=umbrella_integration,
):
    """Histogram-MBAR rate PMF in the *already sampled* rate-flux measure.

    Samples are independent time streams, each [N] or [N,1], in the common
    ensemble exp(-beta*H_P/P)*f_xi*exp(-beta*W_i). Only beta*W_i is removed:
    no additional P, metric, radial Jacobian, or physical-energy factor occurs.
    Identical umbrella Hamiltonians are merged exactly for the MBAR equations;
    circular block resampling remains independent within every original stream.

    Endpoints are linearly interpolated in log histogram density between
    neighboring supported bin centers. A masked gap anywhere between reference
    and surface is rejected. Bin masses are accumulated from log weights, so
    tiny but nonzero weights are never lost to underflow. The search range and
    endpoints must lie within the histogram *centers*; they are never clipped
    to the last center. With xi_star=None, select the unsmoothed maximum
    bin center in search_range (default [reference,max(centers)]), requiring the
    entire search interval to be supported. Each bootstrap keeps that surface
    fixed; its uncertainty is conditional on this selection, not peak-location
    uncertainty. Unsupported point estimates return ready=False; invalid input
    raises ValueError. Missing uncertainties remain None/NaN, never zero. If
    any bootstrap draw fails, primary errors are unavailable; surviving-draw
    errors are retained only under explicit supported_draw labels.

    Hard rate gates are separate from support: a Kish weight ESS below
    ``min_weight_ess`` (global) or ``min_endpoint_ess`` (reference and
    dividing-surface bins), or a stream whose mean lies more than
    ``max_center_displacement_sigma`` bias widths from its umbrella center,
    returns ready=False. The supported curve and its would-be barrier remain
    available as explicitly diagnostic values. Pass None to disable a gate.

    compare_ui adds a cheap Gaussian-UI calculation on the *same* bootstrap
    draws, exposing paired estimator differences without a second MBAR solve.
    ``bootstrap_workers`` bounds a shared-memory thread pool used for the
    independent MBAR draws. ``None`` selects a conservative automatic count
    for larger jobs; ``1`` keeps the serial implementation. The coordinator
    still generates circular-block indices in the original seeded order, so
    serial and parallel runs consume identical resamples. Numerical convergence
    is not physical or sampling convergence.
    """
    if measure != "rate_flux":
        raise ValueError("mbar_rate_pmf requires samples in the rate_flux measure")
    arrays = []
    for sample in samples:
        a = np.asarray(sample, float)
        if a.ndim == 2 and a.shape[1] == 1:
            a = a[:, 0]
        if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
            raise ValueError(
                "rate PMF samples must be finite scalar streams with >=2 points"
            )
        arrays.append(a)
    if not arrays:
        raise ValueError("at least one sampling stream is required")
    centers = np.asarray(centers, float)
    if centers.shape not in ((len(arrays),), (len(arrays), 1)):
        raise ValueError("one scalar center per time stream is required")
    centers = centers.reshape(-1)
    stiffness = np.asarray(stiffness, float)
    if stiffness.shape not in (
        (),
        (1,),
        (1, 1),
        (len(arrays),),
        (len(arrays), 1),
        (len(arrays), 1, 1),
    ):
        raise ValueError("stiffness must be scalar or one scalar matrix per stream")
    if stiffness.size == 1:
        stiffness = np.full(len(arrays), float(stiffness.reshape(-1)[0]))
    else:
        stiffness = stiffness.reshape(-1)
    if (
        centers.shape != (len(arrays),)
        or stiffness.shape != centers.shape
        or not np.isfinite(centers).all()
        or not np.isfinite(stiffness).all()
        or np.any(stiffness <= 0)
        or not np.isfinite(beta)
        or beta <= 0
        or not np.isfinite(reference)
    ):
        raise ValueError(
            "invalid scalar umbrella centers, stiffness, beta, or reference"
        )
    for value, minimum, name in [
        (bootstrap, 0, "bootstrap"),
        (min_count, 1, "min_count"),
        (ui_points, 2, "ui_points"),
    ]:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(name + " must be an integer >= " + str(minimum))
    if bootstrap == 1:
        raise ValueError("bootstrap must be zero or >=2")
    bootstrap_workers = _resolve_bootstrap_workers(bootstrap_workers, bootstrap)
    if xi_star is not None and not np.isfinite(xi_star):
        raise ValueError("xi_star must be finite")
    for value, name in ((min_weight_ess, "min_weight_ess"),
                        (min_endpoint_ess, "min_endpoint_ess"),
                        (max_center_displacement_sigma, "max_center_displacement_sigma")):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not np.isfinite(value) or value <= 0):
            raise ValueError(name + " must be positive and finite, or None")
    options = dict(solver_options or {})
    if set(options) - {"tolerance", "max_iterations", "method"}:
        raise ValueError(
            "solver_options supports tolerance, max_iterations, and method"
        )
    counts = np.asarray([len(a) for a in arrays])
    offsets = np.concatenate(([0], np.cumsum(counts)))
    x = np.concatenate(arrays)
    raw_bins = np.asarray(bins)
    if raw_bins.ndim == 2 and raw_bins.shape[0] == 1:
        bins = raw_bins[0]
    elif raw_bins.ndim == 1 and raw_bins.size == 1:
        bins = raw_bins[0]
    if np.isscalar(bins):
        if (
            isinstance(bins, (bool, np.bool_))
            or not isinstance(bins, (int, np.integer))
            or bins < 3
        ):
            raise ValueError("bins must be an integer >=3 or explicit edges")
        limits = np.asarray(
            [x.min(), x.max()] if ranges is None else ranges, float
        ).reshape(-1)
        if (
            limits.shape != (2,)
            or not np.isfinite(limits).all()
            or limits[1] <= limits[0]
        ):
            raise ValueError("ranges must contain one finite increasing interval")
        edges = np.linspace(*limits, int(bins) + 1)
    else:
        edges = np.asarray(bins, float)
        if (
            edges.ndim != 1
            or len(edges) < 4
            or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)
        ):
            raise ValueError("explicit bins must contain >=4 increasing finite edges")
    grid = 0.5 * (edges[1:] + edges[:-1])
    representatives, state_map = _identical_rows(np.column_stack((centers, stiffness)))
    fit_counts = np.bincount(state_map, weights=counts).astype(int)
    u = umbrella_reduced_potentials(
        x[:, None],
        centers[representatives, None],
        stiffness[representatives, None, None],
        beta,
    )
    fit = _solve_mbar(u, fit_counts, overlap_threshold=overlap_threshold,
                     return_state_weights=False, **options)
    connected = len(np.unique(fit.components)) == 1
    absolute, density, log_density, count, mask = _log_free_energy(
        x, fit, edges, beta, min_count
    )
    free = absolute.copy()
    warnings = []
    reason = None
    search_endpoints = None
    if not fit.converged:
        reason = "mbar_not_converged"
    elif not connected:
        reason = "disconnected_overlap"
    selection = "fixed" if xi_star is not None else "maximum_supported_bin_center"
    auto_range = None
    if xi_star is None:
        auto_range = np.asarray(
            [reference, centers.max()] if search_range is None else search_range, float
        )
        if (
            auto_range.shape != (2,)
            or not np.isfinite(auto_range).all()
            or auto_range[1] <= auto_range[0]
        ):
            raise ValueError("search_range must be one finite increasing interval")
        low, low_reason = _rate_bracket(grid, mask, auto_range[0], "search_lower")
        high, high_reason = _rate_bracket(grid, mask, auto_range[1], "search_upper")
        search_endpoints = {
            "lower": classify_point(grid, mask, auto_range[0]),
            "upper": classify_point(grid, mask, auto_range[1]),
            "masked_intervals": masked_runs(grid, mask, *auto_range),
        }
        search_reason = low_reason or high_reason
        if search_reason is None and mask[low[0] : high[1] + 1].any():
            search_reason = "masked_search_interval"
        eligible = np.flatnonzero(
            (grid >= auto_range[0]) & (grid <= auto_range[1]) & ~mask
        )
        if not len(eligible):
            search_reason = search_reason or "no_supported_search_centers"
        reason = reason or search_reason
        if reason is None:
            selected = int(eligible[np.argmax(absolute[eligible])])
            xi_star = float(grid[selected])
            if selected in (eligible[0], eligible[-1]):
                warnings.append(
                    "The unsmoothed maximum is at an edge of the search centers; inspect or extend the search range."
                )
    reference_bracket = surface_bracket = None
    if xi_star is not None:
        reference_bracket, surface_bracket, endpoint_reason = _rate_support(
            grid, mask, reference, xi_star
        )
        reason = reason or endpoint_reason
    supported = reason is None and xi_star is not None
    log_weights = fit_log_weights(fit)
    weights_report = weight_diagnostics(
        log_weights, x, offsets, edges,
        {"reference": None if reference_bracket is None else reference_bracket[:2],
         "dividing_surface": None if surface_bracket is None else surface_bracket[:2]},
    )
    anomalies = window_anomalies(arrays, centers, stiffness, beta,
                                 max_center_displacement_sigma)
    endpoint_ess = [item["minimum_kish_ess"]
                    for item in weights_report["endpoints"].values() if item is not None]
    gates = [
        {"name": "umbrella_stream_consistency",
         "passed": not anomalies,
         "threshold_sigma": max_center_displacement_sigma,
         "anomalous_streams": [a["stream"] for a in anomalies]},
        {"name": "global_weight_ess",
         "passed": bool(min_weight_ess is None
                        or weights_report["global_kish_ess"] >= min_weight_ess),
         "value": weights_report["global_kish_ess"], "minimum": min_weight_ess},
        {"name": "endpoint_weight_ess",
         "passed": bool(min_endpoint_ess is None or not supported
                        or min(endpoint_ess, default=0.0) >= min_endpoint_ess),
         "value": min(endpoint_ess) if endpoint_ess else None,
         "minimum": min_endpoint_ess, "evaluated": supported},
    ]
    gate_reason = None
    if anomalies:
        gate_reason = "umbrella_stream_anomaly"
    elif not gates[1]["passed"]:
        gate_reason = "weight_ess_collapse"
    elif not gates[2]["passed"]:
        gate_reason = "endpoint_weight_ess_collapse"
    if anomalies or (gate_reason and fit.converged and connected):
        # A displaced stream is a state failure and the likely root cause of
        # any overlap, weight or support failure; weight gates are reported
        # ahead of the support failures they usually produce.
        reason = gate_reason
    ready = supported and reason is None
    barrier = diagnostic_barrier = None
    if reference_bracket is not None:
        free = absolute - _rate_value(absolute, reference_bracket)
    if supported:
        diagnostic_barrier = _rate_value(absolute, surface_bracket) - _rate_value(
            absolute, reference_bracket
        )
    if ready:
        barrier = diagnostic_barrier
    if not fit.converged or not connected:
        free[:] = np.nan
        mask[:] = True
    gs = np.asarray([cv_statistical_inefficiency(a) for a in arrays])
    block_plan = _bootstrap_block_plan(counts, gs)
    block_plan["observables"] = CV_CORRELATION_OBSERVABLES
    blocks = block_plan["block_lengths"]
    warnings.extend(block_plan["warnings"])
    moments = []
    for i, a in enumerate(arrays):
        mean, std = float(a.mean()), float(a.std())
        z = (a - mean) / std if std else None
        moments.append(
            {
                "stream": i,
                "center": float(centers[i]),
                "samples": len(a),
                "mean": mean,
                "standard_deviation": std,
                "skewness": float(np.mean(z**3)) if z is not None else None,
                "excess_kurtosis": float(np.mean(z**4) - 3) if z is not None else None,
                "statistical_inefficiency": float(gs[i]),
                "block_length": int(blocks[i]),
                "estimated_effective_samples": float(counts[i] / gs[i]),
            }
        )

    def endpoint_details(bracket):
        if bracket is None:
            return None
        items = []
        for index in sorted(set(bracket[:2])):
            selected = (x >= edges[index]) & (x < edges[index + 1])
            lw = log_weights[selected]
            items.append(
                {
                    "bin": index,
                    "center": float(grid[index]),
                    "raw_count": int(count[index]),
                    "density": float(density[index]),
                    "weighted_effective_samples_without_time_correction": kish_ess(lw),
                    "contributing_streams": sum(
                        bool(selected[start:stop].any())
                        for start, stop in zip(offsets[:-1], offsets[1:])
                    ),
                }
            )
        return {"bracket": list(bracket), "bins": items}

    ui = {"requested": bool(compare_ui), "ready": False, "reason": None}
    ui_grid = None
    if compare_ui and ready:
        lower, upper = min(reference, xi_star), max(reference, xi_star, centers.max())
        if lower == upper:
            lower, upper = grid[0], grid[-1]
        ui_grid = np.linspace(lower, upper, ui_points)
        try:
            ui_curve, _ = _integrate_ui(
                arrays, centers, stiffness, beta, ui_grid
            )
            ui_barrier = float(
                np.interp(xi_star, ui_grid, ui_curve)
                - np.interp(reference, ui_grid, ui_curve)
            )
            ui.update(
                ready=True,
                xi_star=xi_star,
                reference=reference,
                free_energy_barrier_hartree=ui_barrier,
                mbar_minus_ui_hartree=barrier - ui_barrier,
                xi=ui_grid,
                free_energy=ui_curve - np.interp(reference, ui_grid, ui_curve),
            )
        except (ValueError, FloatingPointError) as exc:
            ui["reason"] = "ui_fit_failed: " + str(exc)
    draws, endpoints, curves, records, failures = [], [], [], [], []
    ui_draws, paired_mbar, differences, ui_failures = [], [], [], []
    rng = np.random.default_rng(seed)
    if ready:
        base_indices = [np.arange(n) for n in counts]
        context = {
            "u": u,
            "x": x,
            "fit_counts": fit_counts,
            "overlap_threshold": overlap_threshold,
            "initial_free_energies": fit.free_energies,
            "solver_options": options,
            "edges": edges,
            "beta": beta,
            "min_count": min_count,
            "grid": grid,
            "reference": reference,
            "xi_star": xi_star,
            "offsets": offsets,
            "centers": centers,
            "stiffness": stiffness,
            "ui_ready": ui["ready"],
            "ui_grid": ui_grid,
            "solve_mbar": _solve_mbar,
            "integrate_ui": _integrate_ui,
        }

        def tasks():
            # Keep the existing single coordinator RNG stream and draw order.
            # Only the expensive solve/histogram/UI work leaves the coordinator;
            # at most one large index vector per worker is retained in memory.
            for draw in range(int(bootstrap)):
                indices = np.concatenate(
                    [
                        _resample(base, block, rng) + start
                        for base, block, start in zip(base_indices, blocks, offsets[:-1])
                    ]
                )
                yield draw, indices, context

        if bootstrap_workers == 1:
            results = [_mbar_bootstrap_draw(task) for task in tasks()]
        else:
            with threadpool_limits(limits=1, user_api="blas"):
                results = _ordered_thread_map(
                    _mbar_bootstrap_draw, tasks(), bootstrap_workers
                )
        for result in results:
            record = result["record"]
            records.append(record)
            if record["valid"]:
                endpoints.append(result["endpoints"])
                draws.append(result["barrier"])
                curves.append(result["curve"])
                if result["ui_barrier"] is not None:
                    ui_draws.append(result["ui_barrier"])
                    paired_mbar.append(result["barrier"])
                    differences.append(result["ui_difference"])
            if result["ui_failure"] is not None:
                ui_failures.append(result["ui_failure"])
            if not record["valid"]:
                failures.append(
                    {key: value for key, value in record.items() if key != "valid"}
                )
    curve_se = np.full_like(free, np.nan)
    curve_valid = np.zeros(len(grid), int)
    if curves:
        bcurves = np.asarray(curves)
        curve_valid = np.isfinite(bcurves).sum(axis=0)
    if len(curves) >= 2:
        mean = np.divide(
            np.nansum(bcurves, axis=0),
            curve_valid,
            out=np.zeros_like(free),
            where=curve_valid > 0,
        )
        square = np.nansum((bcurves - mean) ** 2, axis=0)
        variance = np.divide(
            square,
            curve_valid - 1,
            out=np.full_like(free, np.nan),
            where=curve_valid > 1,
        )
        curve_se = np.sqrt(variance)
    # A bootstrap can promote an originally masked bin above min_count. It
    # must not turn that unsupported point estimate into a reported error bar.
    curve_se[mask] = np.nan
    curve_valid[mask] = 0
    if block_plan["status"] == "unavailable":
        curve_se[:] = np.nan
    supported_curve_se = curve_se.copy()
    stats, supported_stats, uncertainty = _rate_bootstrap_statistics(
        draws, requested=bootstrap, attempted=len(records),
        block_status=block_plan["status"],
    )
    incomplete_curve_support = (~mask) & (curve_valid < bootstrap)
    curve_se[incomplete_curve_support] = np.nan
    failure_counts = {}
    for failure in failures:
        failure_counts[failure["reason"]] = failure_counts.get(failure["reason"], 0) + 1
    if reason:
        warnings.append("Rate PMF unavailable: " + reason)
    if anomalies:
        warnings.append(
            "Umbrella stream(s) " + ", ".join(str(a["stream"]) for a in anomalies)
            + " have means far outside their umbrella centers; this is a CV/state/propagation "
            "failure, not thermal fluctuation. No rate is reported."
        )
    if not gates[1]["passed"]:
        warnings.append(
            f"Reweighting collapsed: global Kish weight ESS "
            f"{weights_report['global_kish_ess']:.3g} < {min_weight_ess:g}; "
            "the log-domain curve is diagnostic only."
        )
    if bootstrap and len(draws) < 2:
        warnings.append(
            "Fewer than two valid bootstrap draws; the barrier uncertainty is unavailable."
        )
    if failures:
        warnings.append(
            "Some bootstrap draws failed; primary PMF uncertainty is unavailable. "
            "Supported-draw errors are conditional on complete PMF support and a "
            "valid estimator fit; inspect each recorded reason and the valid fraction."
        )
    elif np.any(incomplete_curve_support):
        warnings.append(
            "Some PMF curve bins lack support in every requested bootstrap draw; "
            "their primary errors are unavailable. Supported-draw errors are "
            "conditional on bin support."
        )
    if np.any(counts / gs < 20):
        warnings.append(
            "Some streams have fewer than 20 estimated independent samples; overlap alone does not establish mixing."
        )
    warnings.append(
        "Histogram PMF retains bin-width bias; uncertainty holds the selected surface fixed and is not proof of physical convergence."
    )
    ui.update(
        paired_ui_bootstrap_barriers=ui_draws,
        paired_mbar_bootstrap_barriers=paired_mbar,
        paired_difference_bootstrap_hartree=differences,
        paired_difference_statistics=_rate_draw_statistics(differences),
        ui_barrier_statistics=_rate_draw_statistics(ui_draws),
        failures=ui_failures,
    )
    if ui["ready"]:
        for key, values in (("paired_difference_statistics", differences),
                            ("ui_barrier_statistics", ui_draws)):
            primary, supported, paired_uncertainty = _rate_bootstrap_statistics(
                values, requested=bootstrap, attempted=len(records),
                block_status=block_plan["status"],
            )
            ui[key] = primary
            if paired_uncertainty["bootstrap_rejected"]:
                ui["supported_draw_" + key] = supported
        ui["uncertainty"] = paired_uncertainty
    return {
        "ready": ready,
        "status": "ready" if ready else reason,
        "xi_star": xi_star,
        "reference": float(reference),
        "free_energy_barrier_hartree": barrier,
        "diagnostic_free_energy_barrier_hartree": None if ready else diagnostic_barrier,
        "barrier_standard_error_hartree": stats["standard_error_hartree"],
        "barrier_percentile_95_ci_hartree": stats["percentile_95_ci_hartree"],
        "barrier_uncertainty": uncertainty,
        **({
            "supported_draw_barrier_standard_error_hartree": supported_stats["standard_error_hartree"],
            "supported_draw_barrier_percentile_95_ci_hartree": supported_stats["percentile_95_ci_hartree"],
        } if failures else {}),
        **({"supported_draw_standard_error": supported_curve_se}
           if np.any(incomplete_curve_support) else {}),
        "pmf_bootstrap_barriers": draws,
        "xi": grid,
        "free_energy": free,
        "standard_error": curve_se,
        "mask": mask,
        "edges": edges,
        "density": density,
        "log_density": log_density,
        "raw_count": count,
        "overlap_connected": connected,
        "overlap": fit.overlap,
        "components": fit.components,
        "window_free_energies": fit.free_energies[state_map],
        "weights": fit.weights,
        "statistical_inefficiency": gs,
        "effective_samples_per_stream": counts / gs,
        "weight_ess": weights_report["global_kish_ess"],
        "weight_diagnostics": weights_report,
        "window_moments": moments,
        "window_anomalies": anomalies,
        "gates": gates,
        "support": {
            "reference": endpoint_details(reference_bracket),
            "dividing_surface": endpoint_details(surface_bracket),
            "reference_endpoint": classify_point(grid, mask, reference),
            "dividing_surface_endpoint": (
                classify_point(grid, mask, xi_star) if xi_star is not None else None
            ),
            "search_range": auto_range.tolist() if auto_range is not None else None,
            "search_endpoints": search_endpoints,
            "histogram": histogram_report(edges),
            "histogram_accumulation": "log_sum_exp",
            "continuous_interval": supported,
            "selection": selection,
            "smoothing": "none",
            "failure_reason": reason,
        },
        "solver": {
            "converged": fit.converged,
            "residual": float(fit.residual),
            "unique_umbrella_hamiltonians": len(representatives),
            "stream_to_state": state_map,
            "overlap_indexing": "unique umbrella Hamiltonians",
            "details": fit.diagnostics,
        },
        "bootstrap": {
            "requested": int(bootstrap),
            "workers": int(bootstrap_workers),
            "backend": "thread" if bootstrap_workers > 1 else "serial",
            "attempted": len(records),
            "valid": len(draws),
            "valid_fraction": len(draws) / bootstrap if bootstrap else None,
            "failures": failures,
            "failure_counts": failure_counts,
            "draws": records,
            "skipped_reason": reason if not ready else None,
            "endpoint_values_hartree": np.asarray(endpoints).reshape(-1, 2),
            "endpoint_covariance_hartree_squared": (
                np.cov(np.asarray(endpoints).T, ddof=1) if len(endpoints) >= 2 else None
            ),
            "curve_valid_draws": curve_valid,
            "surface_held_fixed": True,
            "block_lengths": blocks,
            "block_plan": block_plan,
        },
        "ui_comparison": ui,
        "warnings": warnings,
        "measure": "rate_flux",
        "energy_unit": "hartree",
        "estimator": "histogram_mbar",
        "converged": False,
    }
