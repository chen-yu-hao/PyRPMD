"""Histogram support, log-domain weights and window-health gates for rate PMFs.

These helpers never change the sampled measure. They make the support of a
histogram PMF explicit (bin centers, not edges, are the only places where a
histogram density is defined), keep tiny normalized weights in the log domain,
and report weight degeneracy and umbrella streams that are inconsistent with
their own bias. Every gate here rejects a *final rate*; diagnostic curves may
still be recovered for inspection.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp

HARTREE_TO_KCAL_MOL = 627.509474

# Defaults for the hard rate gates. A Kish effective sample size near one
# means that a single frame owns the reweighted ensemble; no binning or
# log-domain arithmetic can turn such a sample into a rate PMF.
DEFAULT_MIN_WEIGHT_ESS = 10.0
DEFAULT_MIN_ENDPOINT_ESS = 2.0
# A harmonic window of reduced stiffness beta*K has a bias-only width
# sigma=1/sqrt(beta*K). A mean displaced by 20 sigma carries a reduced bias of
# 200 kT at its own mean; ordinary PMF slopes shift the mean by F'(xi)/K.
DEFAULT_MAX_CENTER_DISPLACEMENT_SIGMA = 20.0


def histogram_centers(edges):
    edges = np.asarray(edges, float)
    return 0.5 * (edges[1:] + edges[:-1])


def histogram_report(edges):
    """Resolved one-dimensional binning, persisted with every rate PMF."""
    edges = np.asarray(edges, float)
    centers = histogram_centers(edges)
    widths = np.diff(edges)
    uniform = bool(np.allclose(widths, widths[0], rtol=1e-10, atol=0))
    return {
        "bins": int(len(centers)),
        "first_edge": float(edges[0]),
        "last_edge": float(edges[-1]),
        "first_center": float(centers[0]),
        "last_center": float(centers[-1]),
        "uniform_width": float(widths[0]) if uniform else None,
        "edges": edges.tolist(),
        "centers": centers.tolist(),
        "support_rule": (
            "A histogram PMF is defined only at bin centers. Endpoints between "
            "two supported centers are interpolated; endpoints outside "
            "[first_center, last_center] are rejected, never clipped."
        ),
    }


def uniform_edges(bins, ranges):
    lower, upper = (float(v) for v in ranges)
    return np.linspace(lower, upper, int(bins) + 1)


def classify_point(grid, mask, point, *, atol=1e-12):
    """Locate a coordinate relative to histogram centers and their mask.

    ``status`` is one of ``on_center``, ``between_centers``,
    ``outside_histogram_centers`` or ``masked``. The bracket is returned for
    every point inside the center range, so a masked endpoint names its bins.
    """
    grid = np.asarray(grid, float)
    mask = np.asarray(mask, bool)
    point = float(point)
    record = {"coordinate": point, "first_center": float(grid[0]),
              "last_center": float(grid[-1])}
    exact = np.flatnonzero(np.isclose(grid, point, rtol=0, atol=atol))
    if len(exact):
        bracket = [int(exact[0])]
        status = "on_center"
    else:
        right = int(np.searchsorted(grid, point))
        if right == 0 or right == len(grid):
            side = "below" if right == 0 else "above"
            nearest = 0 if right == 0 else len(grid) - 1
            record.update(
                status="outside_histogram_centers", side=side, bracket=None,
                nearest_center=float(grid[nearest]),
                distance_to_nearest_center=float(abs(point - grid[nearest])),
            )
            return record
        bracket = [right - 1, right]
        status = "between_centers"
    masked = [int(i) for i in bracket if mask[i]]
    record.update(
        status="masked" if masked else status,
        bracket=bracket,
        bracket_centers=[float(grid[i]) for i in bracket],
        masked_bins=masked,
    )
    return record


def masked_runs(grid, mask, lower, upper):
    """Contiguous masked center intervals intersecting [lower, upper]."""
    grid = np.asarray(grid, float)
    mask = np.asarray(mask, bool)
    inside = np.flatnonzero((grid >= lower) & (grid <= upper))
    runs, start = [], None
    for index in inside:
        if mask[index] and start is None:
            start = index
        if start is not None and (not mask[index] or index == inside[-1]):
            stop = index if mask[index] else index - 1
            runs.append({"first_bin": int(start), "last_bin": int(stop),
                         "first_center": float(grid[start]),
                         "last_center": float(grid[stop])})
            start = None
    return runs


def preflight_message(name, point, edges, *, option):
    """Return an actionable error when ``point`` lies outside bin centers."""
    centers = histogram_centers(edges)
    if centers[0] - 1e-12 <= point <= centers[-1] + 1e-12:
        return None
    width = float(edges[1] - edges[0])
    side = "below the first" if point < centers[0] else "above the last"
    if point > centers[-1]:
        needed = point + 0.5 * width
        fix = (f"extend the upper histogram edge to at least {needed:.10g} "
               f"(keeping the bin width {width:.6g} requires "
               f"{math.ceil((needed - edges[0]) / width - 1e-9)} bins) or lower the endpoint "
               f"to the last center {centers[-1]:.10g}")
    else:
        needed = point - 0.5 * width
        fix = (f"extend the lower histogram edge to at most {needed:.10g} "
               f"(keeping the bin width {width:.6g} requires "
               f"{math.ceil((edges[-1] - needed) / width - 1e-9)} bins) or raise the endpoint "
               f"to the first center {centers[0]:.10g}")
    return (
        f"{name}={point:.10g} lies {side} histogram bin center: {option} gives "
        f"{len(centers)} bins on [{edges[0]:.10g}, {edges[-1]:.10g}] whose centers span "
        f"[{centers[0]:.10g}, {centers[-1]:.10g}]. A histogram PMF is defined only at "
        f"bin centers and is never clipped or extrapolated; {fix}."
    )


def log_histogram(x, log_weights, edges, min_count=1):
    """Weighted 1D histogram accumulated with log-sum-exp in every bin.

    Returns log mass, log density, raw counts and the unsupported mask. A bin
    is masked only for too few raw frames or genuinely zero weight; ordinary
    exponentiation underflow can no longer hide a populated bin.
    """
    x = np.asarray(x, float).reshape(-1)
    lw = np.asarray(log_weights, float).reshape(-1)
    edges = np.asarray(edges, float)
    if x.shape != lw.shape:
        raise ValueError("one log weight per sample is required")
    nbins = len(edges) - 1
    # np.histogram convention: half-open bins, the last edge is inclusive.
    index = np.searchsorted(edges, x, side="right") - 1
    index[x == edges[-1]] = nbins - 1
    inside = (index >= 0) & (index < nbins)
    index, lw = index[inside], lw[inside]
    count = np.bincount(index, minlength=nbins).astype(float)
    finite = np.isfinite(lw)
    peak = np.full(nbins, -np.inf)
    np.maximum.at(peak, index[finite], lw[finite])
    total = np.zeros(nbins)
    shifted = lw[finite] - peak[index[finite]]
    np.add.at(total, index[finite], np.exp(shifted))
    with np.errstate(divide="ignore"):
        log_mass = np.where(total > 0, peak + np.log(total), -np.inf)
    log_density = log_mass - np.log(np.diff(edges))
    mask = (count < min_count) | ~np.isfinite(log_mass)
    return log_mass, log_density, count, mask


def kish_ess(log_weights):
    """Kish effective sample size, evaluated stably from log weights."""
    lw = np.asarray(log_weights, float).reshape(-1)
    lw = lw[np.isfinite(lw)]
    if not len(lw):
        return 0.0
    return float(np.exp(2 * logsumexp(lw) - logsumexp(2 * lw)))


def fit_log_weights(fit):
    """Normalized log weights of an MBAR fit, also for legacy result objects."""
    lw = getattr(fit, "log_weights", None)
    if lw is None:
        with np.errstate(divide="ignore"):
            lw = np.log(np.asarray(fit.weights, float))
    lw = np.asarray(lw, float)
    finite = np.isfinite(lw)
    if np.any(finite):
        lw = lw - logsumexp(lw[finite])
    return lw


def weight_diagnostics(log_weights, x, offsets, edges, brackets=None):
    """Global, per-stream and endpoint degeneracy of normalized weights."""
    lw = np.asarray(log_weights, float)
    finite = lw[np.isfinite(lw)]
    streams = []
    for stream, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        part = lw[start:stop]
        part_finite = part[np.isfinite(part)]
        streams.append({
            "stream": stream,
            "frames": int(stop - start),
            "kish_ess": kish_ess(part),
            "weight_fraction": (float(np.exp(logsumexp(part_finite)))
                                if len(part_finite) else 0.0),
            "max_normalized_weight": (float(np.exp(part_finite.max()))
                                      if len(part_finite) else 0.0),
        })
    endpoints = {}
    for name, bracket in (brackets or {}).items():
        if bracket is None:
            endpoints[name] = None
            continue
        bins = []
        for index in sorted(set(bracket)):
            selected = (x >= edges[index]) & (
                (x < edges[index + 1]) | ((index == len(edges) - 2) & (x == edges[-1])))
            bins.append({"bin": int(index), "raw_count": int(selected.sum()),
                         "kish_ess": kish_ess(lw[selected])})
        endpoints[name] = {"bins": bins,
                           "minimum_kish_ess": min((b["kish_ess"] for b in bins), default=0.0)}
    return {
        "global_kish_ess": kish_ess(lw),
        "frames": int(len(lw)),
        "zero_weight_frames": int((~np.isfinite(lw)).sum()),
        "max_normalized_weight": float(np.exp(finite.max())) if len(finite) else 0.0,
        "log_weight_span": float(finite.max() - finite.min()) if len(finite) else None,
        "per_stream": streams,
        "minimum_stream_kish_ess": min((s["kish_ess"] for s in streams), default=0.0),
        "endpoints": endpoints,
        "definition": "Kish ESS (sum w)^2/sum w^2 of normalized MBAR target weights, "
                      "without time-correlation correction; computed in the log domain.",
    }


def window_anomalies(arrays, centers, stiffness, beta, threshold=DEFAULT_MAX_CENTER_DISPLACEMENT_SIGMA):
    """Flag umbrella streams whose mean is inconsistent with their own bias.

    A harmonic umbrella K(xi-c)^2/2 shifts a stream mean by roughly
    F'(xi)/K. Measured in the bias-only width sigma=1/sqrt(beta*K), a
    displacement of ``threshold`` sigma means a reduced bias energy of
    threshold^2/2 kT at the stream's own mean. This is a state, coordinate or
    propagation failure (wrong CV branch, sign, center, atom order, stale
    checkpoint), not thermal fluctuation, and continuing the same stream does
    not repair it.
    """
    records = []
    if threshold is None:
        return records
    for i, a in enumerate(arrays):
        a = np.asarray(a, float).reshape(-1)
        k = float(stiffness[i])
        sigma = 1.0 / math.sqrt(beta * k)
        mean = float(a.mean())
        displacement = mean - float(centers[i])
        ratio = abs(displacement) / sigma
        if ratio <= threshold:
            continue
        bias = 0.5 * k * displacement * displacement
        records.append({
            "stream": i,
            "reason": "stream_mean_far_from_umbrella_center",
            "center": float(centers[i]),
            "stiffness_hartree": k,
            "mean": mean,
            "standard_deviation": float(a.std()),
            "minimum": float(a.min()),
            "maximum": float(a.max()),
            "samples": int(len(a)),
            "bias_width_sigma": sigma,
            "displacement": displacement,
            "displacement_in_sigma": ratio,
            "threshold_sigma": float(threshold),
            "reduced_bias_at_mean_kT": float(beta * bias),
            "bias_at_mean_hartree": bias,
            "bias_at_mean_kcal_mol": bias * HARTREE_TO_KCAL_MOL,
            "fraction_beyond_threshold": float(np.mean(np.abs(a - centers[i]) > threshold * sigma)),
        })
    return records


ANOMALY_CHECKS = [
    "Recompute the centroid coordinate from the stream checkpoint and compare it with the recorded samples.",
    "For the RPMDrate xi=s0/(s0-s1), check the branch: s0-s1 must stay positive. Past the singular surface s0=s1 xi returns from -inf toward ~1/3 and the umbrella force no longer restrains the product channel.",
    "Check the window center, stiffness unit and sign, CV identity, atom order, fragment partition and equivalent paths.",
    "Compare native and Python forces/CV gradients at the saved ring; inspect checkpoint rollback and resume provenance.",
    "Do not continue the anomalous stream as a convergence remedy; restart the window from an in-branch geometry in a fresh output directory.",
]
