"""Equilibrium histograms from independent, frozen-bias production streams.

Only configurations strictly after bias adaptation and fixed-bias relaxation
are eligible.  The physical bias is removed with exp(beta*V), with no bead
factor.  If the sampler includes the RPMDrate flux metric, this removes only
the history bias and preserves that target measure.

Replica spread is descriptive.  This module does not manufacture a standard
error from weighted counts, nor bootstrap an adapting bias history.
"""

from __future__ import annotations

import numpy as np

from ..opes import normalized_weights
from .fes_result import estimate_fes_uncertainty


def frozen_samples(stream, beta, *, equilibration_steps, freeze_after_steps,
                   freeze_equilibration_steps=0, final_bias=None):
    """Select the stationary stage, optionally auditing its recorded bias.

    A sample observed at the final deposition step still precedes that update
    and is deliberately excluded, even when no additional relaxation is used.
    """
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("frozen-bias beta must be positive and finite")
    for value, minimum, name in ((equilibration_steps, 0, "equilibration_steps"),
                                 (freeze_after_steps, 1, "freeze_after_steps"),
                                 (freeze_equilibration_steps, 0, "freeze_equilibration_steps")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"frozen-bias {name} must be an integer >= {minimum}")
    values = np.asarray(stream["samples"], float)
    clocks = np.asarray(stream["sample_steps"])
    bias = np.asarray(stream["sample_bias"], float)
    if (values.ndim != 2 or clocks.shape != (len(values),)
            or bias.shape != (len(values),) or clocks.dtype.kind not in "iu"
            or np.any(np.diff(clocks) <= 0)
            or not np.isfinite(values).all() or not np.isfinite(bias).all()):
        raise ValueError("frozen-bias samples, clocks and bias observations disagree")
    cutoff = equilibration_steps + freeze_after_steps + freeze_equilibration_steps
    selected = clocks > cutoff
    if not selected.any():
        raise ValueError("reweighted_histogram has no samples after bias freezing and relaxation")
    retained = values[selected].copy()
    observed = bias[selected]
    if final_bias is not None:
        # Direct MetaD evaluation allocates sample x hill temporaries. Audit
        # bounded chunks so a long frozen production does not multiply its
        # full sample count by the full adaptation history in memory.
        history = getattr(final_bias, "hill_count", None)
        if history is None:
            history = len(final_bias) if hasattr(final_bias, "__len__") else 1
        chunk = max(1, 65536 // max(1, history * retained.shape[1]))
        expected = np.empty(len(retained))
        for start in range(0, len(retained), chunk):
            part = np.asarray(final_bias.physical_energy(retained[start:start+chunk]), float)
            if part.shape != expected[start:start+chunk].shape:
                raise ValueError("final fixed bias returned invalid observation dimensions")
            expected[start:start+chunk] = part
        # A constant gauge shift in a complete independent history is
        # physically irrelevant and cancels on weight normalization.
        if (expected.shape != observed.shape or not np.isfinite(expected).all()
                or not np.allclose(expected - expected[0], observed - observed[0], rtol=1e-10, atol=1e-12)):
            raise ValueError("frozen production observations disagree with the final fixed bias")
    return retained, beta * observed


def weighted_histogram(values, log_weights, edges, beta):
    """Return a bin-volume-corrected PMF and weight-only effective count."""
    weights = normalized_weights(log_weights)
    values = np.asarray(values, float)
    if values.ndim != 2 or len(weights) != len(values) or len(edges) != values.shape[1]:
        raise ValueError("weighted histogram dimensions disagree")
    for edge in edges:
        edge = np.asarray(edge, float)
        if edge.ndim != 1 or len(edge) < 2 or not np.isfinite(edge).all() or np.any(np.diff(edge) <= 0):
            raise ValueError("weighted histogram edges must be finite and increasing")
    if not np.isfinite(beta) or beta <= 0 or not np.isfinite(values).all():
        raise ValueError("weighted histogram beta and samples must be finite")
    mass = np.histogramdd(values, bins=edges, weights=weights)[0]
    volume = np.prod(np.meshgrid(*[np.diff(edge) for edge in edges], indexing="ij"), axis=0)
    with np.errstate(divide="ignore"):
        free = -np.log(mass / volume) / beta
    return free, float(1.0 / np.sum(weights**2))


def frozen_bias_fes(streams, beta, *, bins, ranges, periods, widths,
                    equilibration_steps, freeze_after_steps,
                    freeze_equilibration_steps=0, min_count=1, final_biases=None):
    """Reconstruct separate fixed-bias histories on their common support.

    Normalizing each history independently makes its arbitrary bias offset
    irrelevant.  Histories are never pooled as though they shared one bias.
    The numerical aggregation matches the historical OPES frozen estimator.
    """
    streams = list(streams)
    if not streams or (final_biases is not None and len(final_biases) != len(streams)):
        raise ValueError("one final frozen bias per independent history is required")
    dimension = len(periods)
    if dimension not in (1, 2, 3) or len(bins) != dimension or len(widths) != dimension:
        raise ValueError("frozen-bias grid dimensions disagree")
    if isinstance(min_count, (bool, np.bool_)) or not isinstance(min_count, (int, np.integer)) or min_count < 1:
        raise ValueError("frozen-bias min_count must be a positive integer")
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2 for n in bins):
        raise ValueError("frozen-bias bins must contain integer counts >= 2")
    arrays, log_weights = [], []
    for index, stream in enumerate(streams):
        values, logs = frozen_samples(
            stream, beta, equilibration_steps=equilibration_steps,
            freeze_after_steps=freeze_after_steps,
            freeze_equilibration_steps=freeze_equilibration_steps,
            final_bias=None if final_biases is None else final_biases[index],
        )
        if values.shape[1] != dimension:
            raise ValueError("frozen-bias samples do not match the selected CVs")
        arrays.append(values)
        log_weights.append(logs)
    bounds = ranges
    if bounds is None:
        combined = np.concatenate(arrays)
        bounds = []
        for axis, period in enumerate(periods):
            lo, hi = combined[:, axis].min(), combined[:, axis].max()
            width = max(float(hi - lo), widths[axis])
            bounds.append([-period / 2, period / 2] if period is not None
                          else [lo - 0.05 * width, hi + 0.05 * width])
    if len(bounds) != dimension:
        raise ValueError("frozen-bias ranges do not match the selected CVs")
    edges = [np.linspace(lo, hi, n + 1) for (lo, hi), n in zip(bounds, bins)]
    for values in arrays:
        for axis, period in enumerate(periods):
            if period is not None:
                values[:, axis] = (values[:, axis] - bounds[axis][0]) % period + bounds[axis][0]
    counts = np.asarray([np.histogramdd(values, bins=edges)[0].astype(int) for values in arrays])
    raw, ess = [], []
    for values, logs in zip(arrays, log_weights):
        free, effective = weighted_histogram(values, logs, edges, beta)
        raw.append(free)
        ess.append(effective)
    raw = np.asarray(raw)
    mask = (counts < min_count) | ~np.isfinite(raw)
    common = ~np.any(mask, axis=0)
    free = np.full(raw.shape[1:], np.nan)
    spread = np.full_like(free, np.nan)
    reference = None
    if common.any():
        reference = np.unravel_index(np.argmin(np.where(common, raw.mean(axis=0), np.inf)), common.shape)
        shifted = raw - raw[(slice(None),) + reference].reshape((-1,) + (1,) * dimension)
        free[common] = shifted[:, common].mean(axis=0)
        spread = estimate_fes_uncertainty(shifted, kind="replica_spread", mask=~common)
    else:
        shifted = raw.copy()
        for index in range(len(streams)):
            if (~mask[index]).any():
                shifted[index] -= np.min(raw[index][~mask[index]])
    shifted[mask] = np.nan
    return {
        "free_energy": free, "free_energy_per_replica": shifted,
        "replica_spread": spread, "standard_error": np.full_like(free, np.nan),
        "mask": ~common, "mask_per_replica": mask, "raw_count": counts.sum(axis=0),
        "raw_count_per_replica": counts, "edges": edges,
        "weight_effective_sample_size": ess,
        "retained_samples_per_replica": [len(values) for values in arrays],
        "common_support": bool(common.any()),
        "reference_bin": None if reference is None else list(reference),
    }
