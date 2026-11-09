"""Supported one-dimensional FES endpoints for gas-phase rate estimates."""

from __future__ import annotations

import numpy as np

from ..analysis.rate_support import classify_point, masked_runs


def estimate_fes_curve(result, options, *, uncertainty_reason, warnings=(),
                       unsupported_status="unsupported_pmf", strict_search=False):
    """Reference a rate-flux surface without inventing endpoint covariance.

    A pointwise uncertainty or independent-history spread is not the paired
    uncertainty of F(xi*) - F(0).  The caller supplies provenance and the
    precise reason why that covariance is unavailable.
    """
    if (result.measure != "rate_flux" or result.free_energy.ndim != 1
            or result.cv_units != ("dimensionless",)
            or result.energy_unit != "hartree"):
        raise ValueError("Rate PMF requires a one-dimensional, dimensionless rate_flux FES in hartree")
    grid = np.asarray(result.grid[..., 0], float)
    curve = np.asarray(result.free_energy, float)
    mask = np.asarray(result.mask, bool) | ~np.isfinite(curve)
    search = options["search_range"] or [0.0, float(grid[-1])]
    if options["search_range"] is not None and (search[0] < grid[0] - 1e-12
                                                or search[1] > grid[-1] + 1e-12):
        raise ValueError(
            f"rate.pmf_search_range [{search[0]:.10g}, {search[1]:.10g}] extends beyond the "
            f"PMF grid points [{grid[0]:.10g}, {grid[-1]:.10g}]; endpoints are never clipped. "
            "Extend the FES analysis ranges or narrow the search range.")
    searched = (grid >= search[0]) & (grid <= search[1])
    eligible = searched & ~mask
    xi = options["xi_star"]
    automatic = xi is None
    if automatic and np.any(eligible):
        indices = np.flatnonzero(eligible)
        xi = float(grid[indices[np.argmax(curve[indices])]])

    def endpoint(value):
        if value is None or value < grid[0] or value > grid[-1]:
            return None
        i = int(np.searchsorted(grid, value))
        bracket = [i] if i < len(grid) and np.isclose(grid[i], value, rtol=0, atol=1e-14) else [i - 1, i]
        if any(index < 0 or index >= len(grid) or mask[index] for index in bracket):
            return None
        return float(np.interp(value, grid, curve))

    reference, surface = endpoint(0.0), endpoint(xi)
    search_supported = bool(np.any(searched) and not np.any(mask[searched])
                            and endpoint(search[0]) is not None and endpoint(search[1]) is not None)
    ready = reference is not None and surface is not None
    if ready:
        ready = not bool(np.any(mask[(grid >= min(0.0, xi)) & (grid <= max(0.0, xi))]))
    if automatic and strict_search and not search_supported:
        ready = False
    if result.metadata.get("overlap_connected") is False or result.metadata.get("connected") is False:
        ready = False
    messages = list(warnings)
    if automatic and strict_search and not search_supported:
        messages.append("Automatic dividing-surface selection requires continuous support over the complete PMF search range.")
    if not ready:
        messages.append("The PMF does not continuously support the reference, dividing surface and required search interval; recrossing is not started.")
    uncertainty = {
        "available": False, "block_status": "unavailable",
        "reason": uncertainty_reason, "reasons": [uncertainty_reason],
        "bootstrap_requested": 0, "bootstrap_attempted": 0,
        "bootstrap_valid": 0, "bootstrap_rejected": 0,
        "surface_held_fixed": True, "selection_uncertainty_included": False,
    }
    return {
        "ready": ready, "status": "ready" if ready else unsupported_status,
        "xi_star": xi, "reference": 0.0,
        "free_energy_barrier_hartree": surface - reference if ready else None,
        "barrier_standard_error_hartree": None,
        "barrier_percentile_95_ci_hartree": None,
        "barrier_uncertainty": uncertainty, "pmf_bootstrap_barriers": [],
        "xi": grid, "free_energy": curve - reference if reference is not None else curve,
        "standard_error": np.full_like(curve, np.nan), "mask": mask,
        "overlap_connected": ready,
        "bootstrap": {
            "requested": 0, "attempted": 0, "valid": 0, "failures": [],
            "block_plan": {"status": "unavailable"}, "skipped_reason": uncertainty_reason,
        },
        "warnings": messages,
        "support": {"search_range": search, "reference": reference is not None,
                    "dividing_surface": surface is not None,
                    "complete_search_range": search_supported,
                    "reference_endpoint": classify_point(grid, mask, 0.0),
                    "dividing_surface_endpoint": (classify_point(grid, mask, xi)
                                                  if xi is not None else None),
                    "search_endpoints": {
                        "lower": classify_point(grid, mask, search[0]),
                        "upper": classify_point(grid, mask, search[1]),
                        "masked_intervals": masked_runs(grid, mask, *search),
                    },
                    "grid": {"points": int(len(grid)), "first": float(grid[0]),
                             "last": float(grid[-1])}},
    }
