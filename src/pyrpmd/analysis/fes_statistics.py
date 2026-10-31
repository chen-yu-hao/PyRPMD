"""Correlation diagnostics and time-block resampling for umbrella streams."""

from __future__ import annotations

import numpy as np

CV_CORRELATION_OBSERVABLES = (
    "CV and nonconstant centered CV squared; periodic CVs use nonconstant first and second circular harmonics"
)


def statistical_inefficiency(series):
    """Initial-positive-sequence estimate; a diagnostic, not proof of mixing."""
    x = np.asarray(series, float)
    if x.ndim == 2:
        return max(statistical_inefficiency(x[:, i]) for i in range(x.shape[1]))
    n = len(x)
    if n < 4:
        return 1.0
    x = x - x.mean()
    var = float(x @ x / n)
    if var <= np.finfo(float).tiny:
        return float(n)
    nfft = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(spec * spec.conj(), nfft)[:n]
    ac = ac / (np.arange(n, 0, -1) * var)
    total = 0.0
    for lag in range(1, n - 1, 2):
        pair = ac[lag] + ac[lag + 1]
        if not np.isfinite(pair) or pair <= 0:
            break
        total += pair
    return float(np.clip(1 + 2 * total, 1, n))


def cv_statistical_inefficiency(values, periods=None):
    """Estimate the slowest resolved linear/amplitude CV correlation.

    Fast sign changes can hide a slowly relaxing oscillator amplitude from
    the initial-positive-sequence estimate of the coordinate alone. Include
    its centered square, or first/second circular harmonics for periodic CVs.
    Constant derived traces contain no additional fluctuations to diagnose;
    a truly constant CV retains the existing unresolved-variation convention
    (g=N for N>=4). This still cannot establish equilibration or detect every
    slow variable relevant to a histogram or rate.
    """
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError("CV correlation requires finite nonempty [N,D] values")
    periods = (None,) * values.shape[1] if periods is None else tuple(periods)
    if len(periods) != values.shape[1] or any(
        p is not None and (not np.isscalar(p) or not np.isfinite(p) or p <= 0)
        for p in periods
    ):
        raise ValueError("periods must contain one positive finite number or None per CV")
    gs = []
    for j, period in enumerate(periods):
        if period is None:
            coordinate = values[:, j]
            gs.append(statistical_inefficiency(coordinate))
            square = (coordinate - coordinate.mean())**2
            if (np.ptp(square) > 16 * np.finfo(float).eps * np.max(square)
                    and np.var(square) > np.finfo(float).tiny):
                gs.append(statistical_inefficiency(square))
        else:
            angle = 2 * np.pi * (values[:, j] % period) / period
            traces = (np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle))
            # Discard roundoff-sized variation, e.g. sin(pi) versus sin(0).
            # A constant circular CV may have different raw branch labels.
            varying = [trace for trace in traces if np.ptp(trace) > 16 * np.finfo(float).eps]
            gs.extend(statistical_inefficiency(trace) for trace in varying)
            if not varying:
                gs.append(statistical_inefficiency(np.zeros(len(values))))
    return float(max(gs))


def _block_resample(x, block, rng):
    n = len(x)
    block = max(1, min(int(block), n))
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    # Row-major flattening preserves the original block/start order and RNG
    # consumption while avoiding one Python loop and small array per block.
    ix = ((starts[:, None] + np.arange(block)) % n).reshape(-1)[:n]
    return x[ix]


def _bootstrap_block_plan(counts, inefficiencies):
    """Choose correlation blocks and expose when their errors are unreliable.

    A block of just g points truncates a substantial part of the covariance
    sum (about 40% for a long AR(1) stream). Five g is a conservative default,
    not a proof of mixing. The n//2 cap avoids resampling a single circular
    block that only rotates the original data and falsely gives zero error.
    Short or capped streams remain explicitly qualified in the diagnostics.
    """
    raw_counts = np.asarray(counts)
    gs = np.asarray(inefficiencies, float)
    if (
        raw_counts.ndim != 1
        or not len(raw_counts)
        or raw_counts.dtype.kind not in "iu"
        or np.any(raw_counts < 1)
        or gs.shape != raw_counts.shape
        or not np.isfinite(gs).all()
        or np.any(gs < 1)
    ):
        raise ValueError("bootstrap counts and inefficiencies must be finite positive vectors")
    counts = raw_counts.astype(int)
    requested = np.ceil(5.0 * gs).astype(int)
    used = np.minimum(requested, np.maximum(1, counts // 2))
    capped = used < requested
    blocks = counts / used
    unavailable = bool(np.any(counts < 4) or np.any(gs >= counts))
    limited = bool(np.any(capped) or np.any(blocks < 20))
    status = "unavailable" if unavailable else (
        "limited_blocks" if limited else "finite_block_estimate"
    )
    warnings = []
    if unavailable:
        warnings.append(
            "Bootstrap uncertainty is unavailable: a stream has fewer than four points "
            "or no resolved independent variation. More sampling is required."
        )
    elif limited:
        warnings.append(
            "Bootstrap uncertainty has limited block support: some streams have fewer "
            "than 20 blocks or the requested 5*g length was capped at half the stream. "
            "The reported error is a finite-sample diagnostic, not a reliable interval."
        )
    return {
        "policy": "ceil(5*g), capped at max(1, floor(N/2))",
        "requested_block_lengths": requested,
        "block_lengths": used,
        "block_length_capped": capped,
        "blocks_per_stream": blocks,
        "status": status,
        "warnings": warnings,
        "mixing_established": False,
    }


def _bootstrap_groups(groups, counts):
    """A partition of synchronized streams; independent streams are singleton groups."""
    if groups is None:
        return [np.array([i], dtype=int) for i in range(len(counts))]
    labels = np.asarray(groups)
    if (labels.shape != counts.shape or labels.dtype.kind not in "iu"
            or np.any(labels < 0)):
        raise ValueError("bootstrap_groups needs one nonnegative integer per stream")
    result = [np.flatnonzero(labels == label) for label in dict.fromkeys(labels.tolist())]
    if any(np.any(counts[group] != counts[group[0]]) for group in result):
        raise ValueError("streams in a bootstrap group must have equal synchronized lengths")
    return result


def _grouped_resample_indices(counts, block_lengths, groups, rng, *, resample=None):
    """Preserve cross-window correlation by using one clock draw per exchange ladder."""
    if resample is None:
        resample = _block_resample
    offsets = np.r_[0, np.cumsum(counts)]
    indices = [None] * len(counts)
    for group in groups:
        first = group[0]
        clock = resample(np.arange(counts[first]), block_lengths[first], rng)
        for i in group:
            indices[i] = offsets[i] + clock
    return np.concatenate(indices)
