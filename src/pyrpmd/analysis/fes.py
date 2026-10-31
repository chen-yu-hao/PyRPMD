"""Multidimensional FES reconstruction and compatible analysis entry points.

Numerical primitives live in :mod:`.mbar`, :mod:`.umbrella`, and
:mod:`.fes_statistics`; rate-flux PMFs live in :mod:`.rate_pmf`.
The established imports from ``pyrpmd.analysis.fes`` remain available.
"""

from __future__ import annotations

import numpy as np

from .fes_statistics import (
    CV_CORRELATION_OBSERVABLES,
    _block_resample,
    _bootstrap_block_plan,
    _bootstrap_groups,
    _grouped_resample_indices as _resample_groups,
    cv_statistical_inefficiency,
    statistical_inefficiency,
)
from .mbar import (
    MBARResult,
    UmbrellaReweighting,
    _identical_rows,
    _reweighting_signature,
    solve_mbar,
)
from .rate_pmf import (
    _rate_bootstrap_statistics,
    _rate_bracket,
    _rate_draw_statistics,
    _rate_support,
    _rate_value,
    mbar_rate_pmf as _mbar_rate_pmf,
)
from .umbrella import _histogram, umbrella_integration, umbrella_reduced_potentials


def _grouped_resample_indices(counts, block_lengths, groups, rng):
    """Use the public module's resampler, including instrumentation hooks."""
    return _resample_groups(counts, block_lengths, groups, rng, resample=_block_resample)


def reconstruct_fes(
    samples,
    centers,
    stiffness,
    beta,
    bins=40,
    ranges=None,
    periods=None,
    bootstrap=20,
    seed=0,
    min_count=1,
    overlap_threshold=1e-6,
    return_reweighting=False,
    correlation_observables=None,
    *,
    bootstrap_groups=None,
):
    """Reconstruct 1D/2D density w.r.t. ds1...dsD, not bin probability mass.

    Bootstrap resamples contiguous circular time blocks independently inside
    each window/replica by default. With bootstrap_groups, equally spaced and
    synchronized streams in a group use the same clock indices and the largest
    estimated correlation block in that group (for replica-exchange ladders).
    Different groups remain independent. No unsupported region is interpolated.
    Disconnected overlap components are reported; a single global FES is then
    masked because its relative offset is not identifiable.

    Periodic samples are wrapped into the histogram's coordinate convention;
    adding an integer number of periods cannot change the surface. A periodic
    axis defaults to [-period/2, period/2] and may cover at most one period.
    The error is relative to the original minimum bin, which is held fixed in
    every bootstrap draw. A primary bin error requires support in every
    requested draw, including a converged, connected MBAR fit and the fixed
    reference. ``supported_draw_standard_error`` retains an explicitly
    conditional diagnostic from surviving draws; per-bin valid counts expose
    the selection. Neither error is available for unresolved correlation blocks.

    ``correlation_observables`` optionally supplies one finite real [Ns,M]
    array per stream, with the same M >= 1 across streams. Its observations
    must be aligned to the CV samples. The largest statistical inefficiency
    among CVs and these additional observables controls diagnostics and block
    lengths only; MBAR samples, raw counts and point weights are unchanged.
    This allows a hidden coordinate to enlarge correlation blocks without
    treating it as an additional biased CV or changing the target measure.
    """
    arrays = [np.asarray(a, float) for a in samples]
    if not arrays or any(a.ndim != 2 or not len(a) for a in arrays):
        raise ValueError("samples must contain nonempty [Ns,D] arrays")
    d = arrays[0].shape[1]
    if d not in (1, 2) or any(
        a.shape[1] != d or not np.isfinite(a).all() for a in arrays
    ):
        raise ValueError("FES supports finite one- or two-dimensional samples")
    extra = None
    if correlation_observables is not None:
        try:
            supplied = list(correlation_observables)
        except TypeError as exc:
            raise ValueError("correlation_observables requires one [Ns,M] array per stream") from exc
        if len(supplied) != len(arrays):
            raise ValueError("correlation_observables stream count must match samples")
        extra = []
        for index, (value, sample) in enumerate(zip(supplied, arrays)):
            value = np.asarray(value)
            if (value.dtype.kind not in "fiu" or value.ndim != 2
                    or value.shape[0] != len(sample) or value.shape[1] < 1
                    or not np.isfinite(value).all()):
                raise ValueError(f"correlation_observables stream {index} must be finite real [Ns,M] aligned with samples")
            if extra and value.shape[1] != extra[0].shape[1]:
                raise ValueError("correlation_observables must have the same M in every stream")
            with np.errstate(over="ignore", invalid="ignore"):
                value = np.asarray(value, dtype=float)
            if not np.isfinite(value).all():
                raise ValueError(f"correlation_observables stream {index} must be finite float64 observations")
            extra.append(value)
    centers = np.asarray(centers, float)
    if centers.shape != (len(arrays), d) or not np.isfinite(centers).all():
        raise ValueError("one finite center per sampling stream required")
    k = np.asarray(stiffness, float)
    if k.ndim == 2:
        k = np.broadcast_to(k, (len(arrays), d, d))
    if k.shape != (len(arrays), d, d) or not np.isfinite(k).all():
        raise ValueError("stiffness must be finite with one [D,D] matrix per stream")
    if any(not np.allclose(a, a.T) or np.linalg.eigvalsh(a).min() <= 0 for a in k):
        raise ValueError("all umbrella matrices must be symmetric positive definite")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive")
    if (
        isinstance(min_count, (bool, np.bool_))
        or not isinstance(min_count, (int, np.integer))
        or min_count < 1
    ):
        raise ValueError("min_count must be a positive integer")
    if (
        isinstance(bootstrap, (bool, np.bool_))
        or not isinstance(bootstrap, (int, np.integer))
        or bootstrap < 0
        or bootstrap == 1
    ):
        raise ValueError("bootstrap must be integer zero or >=2")
    periods = (None,) * d if periods is None else tuple(periods)
    if len(periods) != d or any(
        p is not None and (not np.isscalar(p) or not np.isfinite(p) or p <= 0)
        for p in periods
    ):
        raise ValueError("periods must contain one positive finite number or None per CV")
    x = np.concatenate(arrays)
    counts = np.array([len(a) for a in arrays])
    groups = _bootstrap_groups(bootstrap_groups, counts)
    if np.isscalar(bins):
        bins = [bins] * d
    if len(bins) != d:
        raise ValueError("bins must have one entry per CV")
    if ranges is None:
        ranges = [
            (-p / 2, p / 2) if p is not None
            else (float(x[:, j].min()), float(x[:, j].max()))
            for j, p in enumerate(periods)
        ]
    bounds = np.asarray(ranges, float)
    if bounds.shape != (d, 2) or not np.isfinite(bounds).all():
        raise ValueError("ranges must contain finite [lower,upper] limits per CV")
    edges = []
    for j, b in enumerate(bins):
        if np.isscalar(b):
            lo, hi = bounds[j]
            if (
                isinstance(b, (bool, np.bool_))
                or not isinstance(b, (int, np.integer))
                or b < 2
                or not hi > lo
            ):
                raise ValueError("histogram needs integer >=2 bins and nonzero range")
            edges.append(np.linspace(lo, hi, int(b) + 1))
        else:
            edge = np.asarray(b, float)
            if (edge.ndim != 1 or len(edge) < 3 or not np.isfinite(edge).all()
                    or np.any(np.diff(edge) <= 0)):
                raise ValueError("invalid bin edges")
            edges.append(edge)
    if any(p is not None for p in periods):
        arrays = [a.copy() for a in arrays]
        for j, p in enumerate(periods):
            if p is None:
                continue
            lo, hi = edges[j][0], edges[j][-1]
            if hi - lo > p and not np.isclose(hi - lo, p, rtol=1e-10, atol=1e-12):
                raise ValueError("a periodic FES axis may span at most one period")
            for a in arrays:
                a[:, j] = (a[:, j] - lo) % p + lo
        x = np.concatenate(arrays)
    u = umbrella_reduced_potentials(x, centers, k, beta, periods)
    mbar = solve_mbar(u, counts, overlap_threshold=overlap_threshold,
                      return_state_weights=False)
    if not mbar.converged:
        raise RuntimeError(
            f"MBAR did not converge; equation residual={mbar.residual:g}"
        )
    mass, density, count, f, mask = _histogram(x, mbar.weights, edges, beta, min_count)
    cv_gs = np.array([cv_statistical_inefficiency(a, periods) for a in arrays])
    gs = cv_gs
    if extra is not None:
        extra_gs = np.array([statistical_inefficiency(a) for a in extra])
        gs = np.maximum(cv_gs, extra_gs)
    block_gs = gs.copy()
    for group in groups:
        block_gs[group] = np.max(gs[group])
    block_plan = _bootstrap_block_plan(counts, block_gs)
    block_plan["observables"] = CV_CORRELATION_OBSERVABLES
    if bootstrap_groups is not None:
        block_plan["resampling"] = "synchronous_within_group"
        block_plan["group_ids"] = np.asarray(bootstrap_groups, dtype=int)
        block_plan["group_max_statistical_inefficiency"] = block_gs
        block_plan["warnings"].append(
            "Exchange windows are jointly block-resampled within each independent ladder; "
            "CV-based block lengths do not establish hidden-coordinate or walker mixing.")
    se = np.full_like(f, np.nan)
    connected = len(set(mbar.components)) == 1
    boots = []
    records = []
    valid = np.zeros(f.shape, dtype=int)
    ref = None
    if connected and bootstrap and np.any(~mask):
        rng = np.random.default_rng(seed)
        ref = np.unravel_index(np.nanargmin(f), f.shape)
        for draw in range(int(bootstrap)):
            indices = _grouped_resample_indices(
                counts, block_plan["block_lengths"], groups, rng)
            xb = x[indices]
            # Every umbrella is already evaluated at each original (wrapped)
            # frame. Resampling columns retains the identical bias problem.
            mb = solve_mbar(
                u[:, indices],
                counts,
                overlap_threshold=overlap_threshold,
                initial_free_energies=mbar.free_energies,
                return_state_weights=False,
            )
            record = {"draw": draw, "valid": False, "reason": None,
                      "solver_residual": mb.residual}
            records.append(record)
            if not mb.converged:
                record["reason"] = "mbar_not_converged"
                continue
            if len(set(mb.components)) > 1:
                record["reason"] = "disconnected_overlap"
                continue
            _, _, _, fb, _ = _histogram(xb, mb.weights, edges, beta, min_count)
            if np.isfinite(fb[ref]):
                boots.append(fb - fb[ref])
                record["valid"] = True
            else:
                record["reason"] = "reference_bin_masked"
        if boots:
            b = np.asarray(boots)
            valid = np.isfinite(b).sum(axis=0)
        if len(boots) >= 2:
            means = np.divide(
                np.nansum(b, axis=0), valid, out=np.zeros_like(f), where=valid > 0
            )
            square = np.nansum((b - means) ** 2, axis=0)
            np.sqrt(
                np.divide(square, valid - 1, out=se, where=valid > 1),
                out=se,
                where=valid > 1,
            )
    # Resampling can promote an originally undersampled bin past min_count.
    # Such a conditional error is not evidence supporting the masked estimate.
    se[mask] = np.nan
    valid[mask] = 0
    if block_plan["status"] == "unavailable":
        se[:] = np.nan
    if not connected:
        # Conditional component surfaces may be reconstructed separately, but
        # displaying arbitrary offsets as one FES would be scientifically wrong.
        f[:] = np.nan
        mask[:] = True
        se[:] = np.nan
    supported_se = se.copy()
    # Missing fits, reference bins or local histogram support select a subset
    # of draws. Its spread is useful only with an explicit conditional label.
    # Point estimates and their original support mask remain unchanged.
    se[valid < bootstrap] = np.nan
    ess = 1 / float(np.sum(mbar.weights**2))
    result = {
        "free_energy": f,
        "probability_mass": mass,
        "density": density,
        "standard_error": se,
        "supported_draw_standard_error": supported_se,
        "standard_error_scope": "all requested draws support the original bin and fixed reference",
        "supported_draw_condition": (
            "conditional on a converged, connected MBAR fit and histogram "
            "support for this bin and the fixed reference in the bootstrap draw"
        ),
        "mask": mask,
        "raw_count": count,
        "edges": edges,
        "overlap": mbar.overlap,
        "components": mbar.components,
        "connected": connected,
        "window_free_energies": mbar.free_energies,
        "statistical_inefficiency": gs,
        "effective_samples_per_stream": counts / gs,
        "weight_ess": ess,
        "bootstrap_replicates": len(boots),
        "bootstrap_requested": int(bootstrap),
        "bootstrap_attempted": len(records),
        "bootstrap_valid_per_bin": valid,
        "bootstrap_reference_bin": None if ref is None else np.asarray(ref),
        "bootstrap_diagnostics": records,
        "bootstrap_block_plan": block_plan,
        "weights": mbar.weights,
        "measure": "cv_density",
        "energy_unit": "hartree",
    }
    if return_reweighting:
        result["_reweighting"] = UmbrellaReweighting.from_result(mbar, u, counts, overlap_threshold)
    if extra is not None:
        result["cv_statistical_inefficiency"] = cv_gs
        result["additional_statistical_inefficiency"] = extra_gs
    return result


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
):
    # Resolve these hooks at call time: downstream diagnostics instrument the
    # historical analysis.fes entry points. No shared module state is changed.
    return _mbar_rate_pmf(
        samples, centers, stiffness, beta,
        xi_star=xi_star, reference=reference, bins=bins, ranges=ranges,
        search_range=search_range, bootstrap=bootstrap, seed=seed,
        min_count=min_count, overlap_threshold=overlap_threshold,
        measure=measure, compare_ui=compare_ui, ui_points=ui_points,
        solver_options=solver_options, _solve_mbar=solve_mbar,
        _resample=_block_resample, _integrate_ui=umbrella_integration,
    )


mbar_rate_pmf.__doc__ = _mbar_rate_pmf.__doc__
