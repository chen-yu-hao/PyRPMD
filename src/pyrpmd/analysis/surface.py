"""First-order surface QTST from a position-density PMF.

This is the single-dividing-surface, reactant-basin normalization of Li et al.,
J. Chem. Phys. 163, 024105 (2025), doi:10.1063/5.0273941.  It is restricted to
a linear centroid coordinate with constant mobile-atom mass metric.  The
result is a rate per adsorbed reactant, not a gas-phase bimolecular rate or a
two-dimensional concentration-based bimolecular rate coefficient.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp

from ..constants import ATOMIC_TIME_S
from .fes import (
    UmbrellaReweighting,
    _block_resample,
    _bootstrap_block_plan,
    _histogram,
    solve_mbar,
    statistical_inefficiency,
    umbrella_reduced_potentials,
)


class _UnsupportedPMF(ValueError):
    """The requested basin/surface is not continuously supported by the PMF."""


def _parameters(beta, mass_metric, basin, dividing_surface):
    if (
        not np.isscalar(beta)
        or not np.isscalar(mass_metric)
        or isinstance(beta, (bool, np.bool_))
        or isinstance(mass_metric, (bool, np.bool_))
        or not np.isfinite(beta)
        or beta <= 0
        or not np.isfinite(mass_metric)
        or mass_metric <= 0
    ):
        raise ValueError("beta and mass_metric must be finite and positive")
    limits = np.asarray(basin, float)
    if (
        limits.shape != (2,)
        or not np.isfinite(limits).all()
        or limits[1] <= limits[0]
    ):
        raise ValueError("basin must be one finite increasing interval")
    if (
        not np.isscalar(dividing_surface)
        or isinstance(dividing_surface, (bool, np.bool_))
        or not np.isfinite(dividing_surface)
    ):
        raise ValueError("dividing_surface must be finite")
    if limits[0] < dividing_surface < limits[1]:
        raise ValueError("basin must lie entirely on one side of dividing_surface")
    return limits


def _bracket(grid, point, mask, name):
    # Tolerate roundoff at a knot, without a fixed tolerance that would change
    # the support decision under a rescaling of the reaction coordinate.
    tolerance = 4 * np.finfo(float).eps * max(
        abs(point), abs(grid[0]), abs(grid[-1]), np.finfo(float).tiny
    )
    close = np.flatnonzero(np.abs(grid - point) <= tolerance)
    if len(close):
        index = int(close[np.argmin(np.abs(grid[close] - point))])
        if mask[index]:
            raise _UnsupportedPMF(name + "_masked")
        return index, index
    right = int(np.searchsorted(grid, point))
    if right == 0 or right == len(grid):
        raise _UnsupportedPMF(name + "_outside_histogram_centers")
    if mask[right - 1] or mask[right]:
        raise _UnsupportedPMF(name + "_masked")
    return right - 1, right


def _basin_statistics(grid, free_energy, beta, mass_metric, basin, dividing_surface, mask):
    limits = _parameters(beta, mass_metric, basin, dividing_surface)
    grid, free = np.asarray(grid, float), np.asarray(free_energy, float)
    if (
        grid.ndim != 1
        or len(grid) < 2
        or free.shape != grid.shape
        or not np.isfinite(grid).all()
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("grid must be finite, increasing and match free_energy")
    invalid = ~np.isfinite(free)
    if mask is not None:
        supplied = np.asarray(mask, bool)
        if supplied.shape != grid.shape:
            raise ValueError("mask must have the same shape as grid")
        invalid |= supplied
    brackets = [
        _bracket(grid, point, invalid, name)
        for point, name in zip(
            [*limits, dividing_surface],
            ["basin_lower", "basin_upper", "dividing_surface"],
        )
    ]
    start = min(bracket[0] for bracket in brackets)
    stop = max(bracket[1] for bracket in brackets)
    if invalid[start : stop + 1].any():
        raise _UnsupportedPMF("masked_interval_between_basin_and_surface")

    # Work in a local gauge before multiplying by beta. No exp(-beta*F)
    # is formed, so an arbitrary PMF offset or a high barrier cannot overflow.
    anchor = float(np.min(free[start : stop + 1]))
    nodes = grid[start : stop + 1]
    with np.errstate(over="ignore", invalid="ignore"):
        logs = -beta * (free[start : stop + 1] - anchor)
    if not np.isfinite(logs).all():
        raise ValueError("reduced free-energy differences exceed floating-point range")
    surface_log = float(np.interp(dividing_surface, nodes, logs))
    integration_nodes = np.r_[
        limits[0], nodes[(nodes > limits[0]) & (nodes < limits[1])], limits[1]
    ]
    integration_logs = np.interp(integration_nodes, nodes, logs)
    widths = np.diff(integration_nodes)
    differences = np.abs(np.diff(integration_logs))
    # Exact integral of an exponential of a linear interpolant. The expression
    # exp(max(y0,y1)) * (1-exp(-abs(y1-y0)))/abs(y1-y0) is stable at both zero
    # slope and large slopes when evaluated in logarithms with expm1.
    corrections = np.zeros_like(differences)
    nonzero = differences > 0
    corrections[nonzero] = (
        np.log(-np.expm1(-differences[nonzero])) - np.log(differences[nonzero])
    )
    log_segments = (
        np.log(widths)
        + np.maximum(integration_logs[:-1], integration_logs[1:])
        + corrections
    )
    integral_log = float(logsumexp(log_segments))
    flux_log = 0.5 * (math.log(mass_metric) - math.log(2 * math.pi) - math.log(beta))
    log_rate = flux_log + surface_log - integral_log
    log_rate_si = log_rate - math.log(ATOMIC_TIME_S)
    if max(log_rate, log_rate_si) > math.log(np.finfo(float).max):
        raise ValueError("QTST rate exceeds floating-point range")
    rate = float(np.exp(log_rate))
    return {
        "k_qtst_au": rate,
        "k_qtst_s_inv": float(np.exp(log_rate_si)),
        "log_k_qtst_au": float(log_rate),
        "log_k_qtst_s_inv": float(log_rate_si),
        "kind": "reactant_basin",
        "reaction_order": 1,
        "per": "adsorbed_reactant",
        "basin": limits.tolist(),
        "dividing_surface": float(dividing_surface),
        "mass_metric": float(mass_metric),
        "thermal_positive_velocity_au": float(np.exp(flux_log)),
        "free_energy_gauge_hartree": anchor,
        "log_surface_density_shifted": surface_log,
        "log_basin_integral_shifted": integral_log,
        "log_density_to_basin_ratio": surface_log - integral_log,
        "integration": "exact_exponential_of_piecewise_linear_log_density",
        "integration_segments": len(widths),
        "required_bin_indices": [start, stop],
        "rate_unit": "atomic_time^-1",
    }


def basin_qtst(
    grid, free_energy, beta, mass_metric, basin, dividing_surface, mask=None
):
    """Return a first-order QTST rate in inverse atomic time.

    ``free_energy`` is -log(position density)/beta at the increasing ``grid``
    points, in hartree with any common offset. For a linear centroid CV with
    gradient ``a``, ``mass_metric`` is sum(|a_i|**2 / m_i) over *mobile* atoms.
    With the physical inverse temperature beta, no bead-count factor occurs.

    The formula is sqrt(g/(2*pi*beta))*exp(-beta*F(surface)) divided by the
    integral of exp(-beta*F) over the explicitly supplied reactant ``basin``.
    The logarithmic density is linearly interpolated; its exponential is
    integrated exactly on each segment. Missing support anywhere in the basin
    or between the basin and surface raises ValueError. Extrapolation across
    histogram edges and interpolation across masked bins are prohibited.

    The caller must establish that the finite basin is the intended reactant
    state, that g is constant, and that the CV/surface defines the intended
    elementary exit. This function supplies neither a transmission coefficient
    nor a surface-area, coverage, symmetry or gas-phase normalization factor.
    """
    return _basin_statistics(
        grid, free_energy, beta, mass_metric, basin, dividing_surface, mask
    )["k_qtst_au"]


def surface_rate_pmf(
    samples,
    centers,
    stiffness,
    beta,
    *,
    basin,
    dividing_surface,
    mass_metric,
    bins=100,
    ranges=None,
    bootstrap=40,
    seed=0,
    min_count=5,
    overlap_threshold=1e-6,
    reweighting=None,
):
    """Remove harmonic umbrellas from 1D position-density sampling streams.

    ``samples`` contains finite [N,1] arrays (one per time stream), ``centers``
    has shape [K,1], and stiffness is a positive scalar or [K,1,1] matrices.
    All streams must share temperature, bead count, PES and Hamiltonian and
    sample the ordinary ``cv_density`` measure, without a flux-metric bias.

    Unsupported estimates return ready=False; invalid inputs raise ValueError.
    ``pmf_bootstrap_qtst`` and rate errors are in s^-1. The fixed basin integral
    and dividing-surface density are recomputed in the *same* circular-block
    bootstrap draw, retaining their covariance. Block resampling is separate
    for every original time stream, including duplicated umbrella centers.
    The shared block policy reports limited support and withholds QTST error
    bars when a stream is too short or has no resolved independent variation.
    Point estimates and paired-draw diagnostics remain available in that case.
    An optional ``UmbrellaReweighting`` from ``reconstruct_fes`` avoids the
    initial MBAR solve after checking the exact bias problem. Histogram and
    basin choices remain independent of the original FES binning.
    """
    limits = _parameters(beta, mass_metric, basin, dividing_surface)
    arrays = [np.asarray(sample, float) for sample in samples]
    if not arrays or any(
        sample.ndim != 2
        or sample.shape[1] != 1
        or len(sample) < 2
        or not np.isfinite(sample).all()
        for sample in arrays
    ):
        raise ValueError("samples must contain finite [N,1] streams with N >= 2")
    centers = np.asarray(centers, float)
    if centers.shape != (len(arrays), 1) or not np.isfinite(centers).all():
        raise ValueError("centers must contain one finite [1] center per stream")
    stiffness = np.asarray(stiffness, float)
    if stiffness.size == 1:
        stiffness = np.full((len(arrays), 1, 1), float(stiffness.reshape(-1)[0]))
    if stiffness.shape != (len(arrays), 1, 1) or (
        not np.isfinite(stiffness).all() or np.any(stiffness <= 0)
    ):
        raise ValueError("stiffness must be positive scalar or [K,1,1] matrices")
    for value, minimum, name in (
        (bootstrap, 0, "bootstrap"),
        (min_count, 1, "min_count"),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(name + " must be an integer >= " + str(minimum))
    if bootstrap == 1:
        raise ValueError("bootstrap must be zero or >=2")
    if not np.isfinite(overlap_threshold) or overlap_threshold < 0:
        raise ValueError("overlap_threshold must be finite and nonnegative")
    counts = np.asarray([len(sample) for sample in arrays])
    x = np.concatenate(arrays)
    raw_bins = np.asarray(bins)
    if raw_bins.ndim == 1 and raw_bins.size == 1:
        bins = raw_bins[0]
    elif raw_bins.ndim == 2 and raw_bins.shape[0] == 1:
        bins = raw_bins[0]
    if np.isscalar(bins):
        if (
            isinstance(bins, (bool, np.bool_))
            or not isinstance(bins, (int, np.integer))
            or bins < 2
        ):
            raise ValueError("bins must be an integer >=2 or explicit edges")
        extent = np.asarray(
            [x.min(), x.max()] if ranges is None else ranges, float
        ).reshape(-1)
        if (
            extent.shape != (2,)
            or not np.isfinite(extent).all()
            or extent[1] <= extent[0]
        ):
            raise ValueError("ranges must contain one finite increasing interval")
        edges = np.linspace(*extent, int(bins) + 1)
    else:
        edges = np.asarray(bins, float)
        if (
            edges.ndim != 1
            or len(edges) < 3
            or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)
        ):
            raise ValueError("bins must contain >=3 increasing finite edges")
    grid = 0.5 * (edges[:-1] + edges[1:])
    u = umbrella_reduced_potentials(x, centers, stiffness, beta)
    if reweighting is None:
        fit = solve_mbar(u, counts, overlap_threshold=overlap_threshold,
                         return_state_weights=False)
    else:
        if not isinstance(reweighting, UmbrellaReweighting):
            raise ValueError("reweighting must be a validated UmbrellaReweighting result")
        reweighting.validate_problem(u, counts, overlap_threshold)
        fit = reweighting
    connected = len(np.unique(fit.components)) == 1
    mass, density, count, free, mask = _histogram(
        x, fit.weights, [edges], beta, min_count
    )
    reason = None
    normalization = None
    if not fit.converged:
        reason = "mbar_not_converged"
    elif not connected:
        reason = "disconnected_overlap"
    else:
        try:
            normalization = _basin_statistics(
                grid, free, beta, mass_metric, limits, dividing_surface, mask
            )
        except _UnsupportedPMF as exc:
            reason = str(exc)
    if not fit.converged or not connected:
        free[:] = np.nan
        mask[:] = True
    ready = reason is None
    gs = np.asarray([statistical_inefficiency(sample) for sample in arrays])
    block_plan = _bootstrap_block_plan(counts, gs)
    blocks = block_plan["block_lengths"]
    offsets = np.r_[0, np.cumsum(counts)]
    rates, log_pairs, records, failures = [], [], [], []
    rng = np.random.default_rng(seed)
    if ready:
        for draw in range(int(bootstrap)):
            # Resample indices so the already-built bias matrix can be reused.
            indices = np.concatenate(
                [
                    _block_resample(np.arange(n), block, rng) + start
                    for n, block, start in zip(counts, blocks, offsets[:-1])
                ]
            )
            record = {"draw": draw, "valid": False, "reason": None}
            try:
                mb = solve_mbar(
                    u[:, indices],
                    counts,
                    overlap_threshold=overlap_threshold,
                    initial_free_energies=fit.free_energies,
                    return_state_weights=False,
                )
                if not mb.converged:
                    record["reason"] = "mbar_not_converged"
                elif len(np.unique(mb.components)) != 1:
                    record["reason"] = "disconnected_overlap"
                else:
                    _, _, _, fb, maskb = _histogram(
                        x[indices], mb.weights, [edges], beta, min_count
                    )
                    nb = _basin_statistics(
                        grid, fb, beta, mass_metric, limits, dividing_surface, maskb
                    )
                    value = nb["k_qtst_s_inv"]
                    pair = [
                        nb["log_surface_density_shifted"],
                        nb["log_basin_integral_shifted"],
                    ]
                    record.update(
                        valid=True,
                        k_qtst_s_inv=value,
                        log_surface_density_shifted=pair[0],
                        log_basin_integral_shifted=pair[1],
                        log_k_qtst_au=nb["log_k_qtst_au"],
                    )
                    rates.append(value)
                    log_pairs.append(pair)
            except _UnsupportedPMF as exc:
                record["reason"] = str(exc)
            except (FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
                record.update(reason="mbar_solver_error", message=str(exc))
            records.append(record)
            if not record["valid"]:
                failures.append({key: value for key, value in record.items() if key != "valid"})
    failure_counts = {}
    for failure in failures:
        failure_counts[failure["reason"]] = failure_counts.get(failure["reason"], 0) + 1
    warnings = [
        "Histogram bin-width and basin-boundary convergence require separate checks.",
        "QTST requires a transmission coefficient before it is a dynamical rate.",
    ]
    warnings.extend(block_plan["warnings"])
    if reason:
        warnings.append("Surface QTST unavailable: " + reason)
    if bootstrap and len(rates) < 2:
        warnings.append("Fewer than two valid bootstrap draws; rate uncertainty is unavailable.")
    if failures:
        warnings.append("Inspect the failed bootstrap draws and their valid fraction.")
    point_rate = normalization["k_qtst_au"] if ready else None
    supported_error_available = len(rates) >= 2 and block_plan["status"] != "unavailable"
    uncertainty_available = supported_error_available and not failures
    uncertainty_reasons = []
    if len(rates) < 2:
        uncertainty_reasons.append("fewer_than_two_valid_pmf_bootstrap_draws")
    if block_plan["status"] != "finite_block_estimate":
        uncertainty_reasons.append("pmf_bootstrap_" + block_plan["status"])
    if failures:
        uncertainty_reasons.append("pmf_bootstrap_contains_unsupported_draws")
    return {
        "ready": ready,
        "status": "ready" if ready else reason,
        "method": "mbar_surface_density",
        "measure": "cv_density",
        "xi_star": float(dividing_surface),
        "xi": grid,
        "free_energy": free,
        "mask": mask,
        "edges": edges,
        "probability_mass": mass,
        "density": density,
        "raw_count": count,
        "k_qtst_au": point_rate,
        "k_qtst_s_inv": normalization["k_qtst_s_inv"] if ready else None,
        "log_k_qtst_au": normalization["log_k_qtst_au"] if ready else None,
        "log_k_qtst_s_inv": normalization["log_k_qtst_s_inv"] if ready else None,
        "pmf_bootstrap_qtst": rates,
        "qtst_standard_error_s_inv": float(np.std(rates, ddof=1)) if uncertainty_available else None,
        "qtst_percentile_95_ci_s_inv": np.quantile(rates, [0.025, 0.975]).tolist() if uncertainty_available else None,
        "supported_draw_qtst_standard_error_s_inv": (
            float(np.std(rates, ddof=1)) if failures and supported_error_available else None
        ),
        "supported_draw_qtst_percentile_95_ci_s_inv": (
            np.quantile(rates, [0.025, 0.975]).tolist() if failures and supported_error_available else None
        ),
        "qtst_uncertainty": {
            "available": uncertainty_available,
            "block_support_resolved": uncertainty_available and block_plan["status"] == "finite_block_estimate",
            "reasons": uncertainty_reasons,
            "supported_draw_condition": (
                "conditional on the bootstrap draw having complete basin-to-surface support"
                if failures else None
            ),
        },
        "normalization": normalization,
        "support": {
            "basin": limits.tolist(),
            "dividing_surface": float(dividing_surface),
            "continuous_interval": ready,
            "failure_reason": reason,
            "interpolation": "linear_log_density_between_supported_bin_centers",
            "extrapolation": False,
            "minimum_raw_count": int(min_count),
        },
        "bootstrap": {
            "requested": int(bootstrap),
            "attempted": len(records),
            "valid": len(rates),
            "valid_fraction": len(rates) / bootstrap if bootstrap else None,
            "failures": failures,
            "failure_counts": failure_counts,
            "draws": records,
            "skipped_reason": reason if not ready else None,
            "block_lengths": blocks,
            "block_plan": block_plan,
            "surface_held_fixed": True,
            "basin_held_fixed": True,
            "log_density_integral_pairs": np.asarray(log_pairs).reshape(-1, 2),
            "log_density_integral_covariance": np.cov(np.asarray(log_pairs).T, ddof=1) if len(log_pairs) >= 2 else None,
        },
        "overlap_connected": connected,
        "overlap": fit.overlap,
        "components": fit.components,
        "window_free_energies": fit.free_energies,
        "weights": fit.weights,
        "statistical_inefficiency": gs,
        "effective_samples_per_stream": counts / gs,
        "weight_ess": float(1 / (fit.weights @ fit.weights)),
        "solver": {"converged": fit.converged, "residual": float(fit.residual), "details": fit.diagnostics},
        "warnings": warnings,
        "energy_unit": "hartree",
        "rate_unit": "s^-1",
        "converged": False,
    }


def surface_rate_error_target(value=0.2):
    """Dimensionless reporting target; it does not change the Hamiltonian."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("surface.max_relative_standard_error must be positive and finite")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("surface.max_relative_standard_error must be positive and finite")
    return value


def surface_rate_quality(pmf, statistics, rate, error, energy_error_kbt, *, target=0.2):
    """Reporting prerequisites, separate from physical convergence.

    Point estimates and supported finite-sample errors remain available when
    these prerequisites fail. Unknown correlation-block support is not a pass.
    """
    target = surface_rate_error_target(target)
    reasons = []
    if not pmf.get("ready", False):
        reasons.append("pmf_not_ready")
    if not statistics.get("plateau_detected", False):
        reasons.append("recrossing_plateau_not_established")
    if energy_error_kbt is None or not np.isfinite(energy_error_kbt):
        reasons.append("child_energy_diagnostics_unavailable")
    elif energy_error_kbt > .1:
        reasons.append("child_endpoint_energy_error_exceeds_tolerance")
    block_status = pmf.get("bootstrap", {}).get("block_plan", {}).get("status")
    if block_status != "finite_block_estimate":
        reasons.append("pmf_correlation_blocks_not_resolved")
    support = statistics.get("uncertainty", {})
    if support.get("independent_parent_chains", 0) < 2:
        reasons.append("fewer_than_two_independent_parent_chains")
    if support.get("nonzero_flux_parent_configurations", 0) < 20:
        reasons.append("fewer_than_twenty_nonzero_flux_parent_configurations")
    relative = None
    if rate is None or not np.isfinite(rate) or rate <= 0:
        reasons.append("nonpositive_or_unavailable_rate_estimate")
    if error is None or not np.isfinite(error) or error < 0:
        reasons.append("joint_rate_uncertainty_unavailable")
    elif rate is not None and np.isfinite(rate) and rate > 0:
        relative = float(error / rate)
        if relative > target:
            reasons.append("relative_standard_error_exceeds_target")
    return {
        "passed": not reasons, "reasons": reasons,
        "relative_standard_error": relative, "max_relative_standard_error": target,
        "pmf_block_status": block_status,
        "minimum_nonzero_flux_parent_configurations": 20,
        "minimum_independent_parent_chains": 2,
        "physical_convergence_established": False,
    }
