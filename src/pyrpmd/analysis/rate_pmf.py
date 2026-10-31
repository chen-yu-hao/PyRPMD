"""Histogram-MBAR rate barriers and paired uncertainty diagnostics.

The estimator consumes samples already drawn in the rate-flux measure.
It never changes the sampling measure or infers physical convergence.
"""

from __future__ import annotations

import numpy as np

from .fes_statistics import (
    CV_CORRELATION_OBSERVABLES,
    _block_resample,
    _bootstrap_block_plan,
    cv_statistical_inefficiency,
)
from .mbar import _identical_rows, solve_mbar
from .umbrella import _histogram, umbrella_integration, umbrella_reduced_potentials


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
    and surface is rejected. With xi_star=None, select the unsmoothed maximum
    bin center in search_range (default [reference,max(centers)]), requiring the
    entire search interval to be supported. Each bootstrap keeps that surface
    fixed; its uncertainty is conditional on this selection, not peak-location
    uncertainty. Unsupported point estimates return ready=False; invalid input
    raises ValueError. Missing uncertainties remain None/NaN, never zero.

    compare_ui adds a cheap Gaussian-UI calculation on the *same* bootstrap
    draws, exposing paired estimator differences without a second MBAR solve.
    Numerical convergence is not physical or sampling convergence.
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
    if xi_star is not None and not np.isfinite(xi_star):
        raise ValueError("xi_star must be finite")
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
    mass, density, count, free, mask = _histogram(
        x[:, None], fit.weights, [edges], beta, min_count
    )
    absolute = np.full_like(free, np.nan)
    absolute[~mask] = -np.log(density[~mask]) / beta
    warnings = []
    reason = None
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
    ready = reason is None and xi_star is not None
    barrier = None
    if reference_bracket is not None:
        free = absolute - _rate_value(absolute, reference_bracket)
    if ready:
        barrier = _rate_value(absolute, surface_bracket) - _rate_value(
            absolute, reference_bracket
        )
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
            w = fit.weights[selected]
            items.append(
                {
                    "bin": index,
                    "center": float(grid[index]),
                    "raw_count": int(count[index]),
                    "density": float(density[index]),
                    "weighted_effective_samples_without_time_correction": (
                        float(w.sum() ** 2 / (w @ w)) if len(w) else 0.0
                    ),
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
        for draw in range(int(bootstrap)):
            indices = np.concatenate(
                [
                    _resample(np.arange(n), block, rng) + start
                    for n, block, start in zip(counts, blocks, offsets[:-1])
                ]
            )
            xb = x[indices]
            record = {"draw": draw, "valid": False, "reason": None}
            try:
                mb = _solve_mbar(
                    u[:, indices],
                    fit_counts,
                    overlap_threshold=overlap_threshold,
                    initial_free_energies=fit.free_energies,
                    return_state_weights=False,
                    **options,
                )
                record["solver_residual"] = float(mb.residual)
                if not mb.converged:
                    record["reason"] = "mbar_not_converged"
                elif len(np.unique(mb.components)) != 1:
                    record["reason"] = "disconnected_overlap"
                else:
                    _, db, _, _, maskb = _histogram(
                        xb[:, None], mb.weights, [edges], beta, min_count
                    )
                    rb, sb, record["reason"] = _rate_support(
                        grid, maskb, reference, xi_star
                    )
                    if record["reason"] is None:
                        ab = np.full_like(db, np.nan)
                        ab[~maskb] = -np.log(db[~maskb]) / beta
                        rv, sv = _rate_value(ab, rb), _rate_value(ab, sb)
                        b = sv - rv
                        record.update(
                            valid=True,
                            reference_free_energy_hartree=rv,
                            surface_free_energy_hartree=sv,
                            barrier_hartree=b,
                        )
                        endpoints.append([rv, sv])
                        draws.append(b)
                        curves.append(ab - rv)
                        if ui["ready"]:
                            try:
                                ub, _ = _integrate_ui(
                                    [
                                        xb[start:stop]
                                        for start, stop in zip(
                                            offsets[:-1], offsets[1:]
                                        )
                                    ],
                                    centers,
                                    stiffness,
                                    beta,
                                    ui_grid,
                                )
                                uv = float(
                                    np.interp(xi_star, ui_grid, ub)
                                    - np.interp(reference, ui_grid, ub)
                                )
                                ui_draws.append(uv)
                                paired_mbar.append(b)
                                differences.append(b - uv)
                                record["ui_barrier_hartree"] = uv
                            except (ValueError, FloatingPointError) as exc:
                                ui_failures.append(
                                    {
                                        "draw": draw,
                                        "reason": "ui_fit_failed",
                                        "message": str(exc),
                                    }
                                )
            except (FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
                record["reason"] = "mbar_solver_error"
                record["message"] = str(exc)
            records.append(record)
            if not record["valid"]:
                failures.append(
                    {key: value for key, value in record.items() if key != "valid"}
                )
    curve_se = np.full_like(free, np.nan)
    curve_valid = np.zeros(len(grid), int)
    if len(curves) >= 2:
        bcurves = np.asarray(curves)
        curve_valid = np.isfinite(bcurves).sum(axis=0)
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
    stats = _rate_draw_statistics(draws)
    if block_plan["status"] == "unavailable":
        stats["standard_error_hartree"] = None
        stats["percentile_95_ci_hartree"] = None
        curve_se[:] = np.nan
    failure_counts = {}
    for failure in failures:
        failure_counts[failure["reason"]] = failure_counts.get(failure["reason"], 0) + 1
    if reason:
        warnings.append("Rate PMF unavailable: " + reason)
    if bootstrap and len(draws) < 2:
        warnings.append(
            "Fewer than two valid bootstrap draws; the barrier uncertainty is unavailable."
        )
    if failures:
        warnings.append(
            "Some bootstrap draws failed; inspect each recorded reason and the valid fraction."
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
    if block_plan["status"] == "unavailable":
        for key in ("paired_difference_statistics", "ui_barrier_statistics"):
            ui[key]["standard_error_hartree"] = None
            ui[key]["percentile_95_ci_hartree"] = None
    return {
        "ready": ready,
        "status": "ready" if ready else reason,
        "xi_star": xi_star,
        "reference": float(reference),
        "free_energy_barrier_hartree": barrier,
        "barrier_standard_error_hartree": stats["standard_error_hartree"],
        "barrier_percentile_95_ci_hartree": stats["percentile_95_ci_hartree"],
        "pmf_bootstrap_barriers": draws,
        "xi": grid,
        "free_energy": free,
        "standard_error": curve_se,
        "mask": mask,
        "edges": edges,
        "density": density,
        "raw_count": count,
        "overlap_connected": connected,
        "overlap": fit.overlap,
        "components": fit.components,
        "window_free_energies": fit.free_energies[state_map],
        "weights": fit.weights,
        "statistical_inefficiency": gs,
        "effective_samples_per_stream": counts / gs,
        "weight_ess": float(1 / (fit.weights @ fit.weights)),
        "window_moments": moments,
        "support": {
            "reference": endpoint_details(reference_bracket),
            "dividing_surface": endpoint_details(surface_bracket),
            "search_range": auto_range.tolist() if auto_range is not None else None,
            "continuous_interval": ready,
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
