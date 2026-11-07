"""Flux--side reaction response estimators for quantum-correction workflows.

The estimators here operate on *already sampled* equilibrium flux and side
observables.  They keep the reactant normalization explicit and resample whole
replicas, never time frames.  This makes the statistical object suitable for
combining with an independently converged RPMD or finite-dimensional quantum
reference.

A flux--side curve is a rate candidate, not automatically an exact rate: the
surface delta representation, reactant ensemble, propagation, time step,
plateau, and independent-chain uncertainty all require separate convergence
checks.  The hybrid correction is deliberately labelled a transferable-defect
approximation.  No generally exact flux--side correction exists for an
arbitrary coupled system merely from a lower-dimensional quantum calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_EPS = np.finfo(float).eps


def _time(value):
    t = np.asarray(value, dtype=float)
    if t.ndim != 1 or t.size < 2 or not np.isfinite(t).all() or t[0] < 0 or np.any(np.diff(t) <= 0):
        raise ValueError("time must be finite, increasing and contain at least two points")
    if t[0] != 0.0:
        raise ValueError("flux-side time must start at zero")
    return t.copy()


def _replica_series(value, name, *, side=False):
    """Normalize scalar or matrix-valued replica curves.

    Scalar inputs use ``[replica,time]`` (or ``[time]`` for one replica).
    Matrix-valued flux-side observables use ``[replica,time,n,n]`` (or
    ``[time,n,n]`` for one replica).  Keeping the replica axis explicit here
    ensures that bootstrap resampling never mixes independent time frames or
    matrix elements.
    """
    x = np.asarray(value, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim == 3:
        # A single matrix-valued replica: [time,n,n].
        if x.shape[1] != x.shape[2]:
            raise ValueError(f"{name} matrix curves must be square")
        x = x[None, ...]
    if x.ndim not in (2, 4) or x.shape[0] < 1 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError(
            f"{name} must be finite [replica,time], [time], "
            "[replica,time,n,n] or [time,n,n]"
        )
    if x.ndim == 4 and x.shape[2] != x.shape[3]:
        raise ValueError(f"{name} matrix curves must be square")
    if side and np.any((x < 0) | (x > 1)):
        raise ValueError("side values must lie in [0,1]")
    return x.copy()


def _positive_population(value, replicas):
    pop = np.asarray(value, dtype=float)
    if pop.ndim == 0:
        pop = np.full(replicas, float(pop))
    if pop.shape != (replicas,) or not np.isfinite(pop).all() or np.any(pop <= 0):
        raise ValueError("reactant_population must be positive scalar or [replica]")
    return pop


def _weights(value, replicas):
    if value is None:
        return np.ones(replicas)
    weight = np.asarray(value, dtype=float)
    if weight.shape != (replicas,) or not np.isfinite(weight).all() or np.any(weight < 0) or not np.any(weight > 0):
        raise ValueError("sample_weight must be nonnegative [replica] with positive support")
    return weight


def _weighted_ratio(numerator, denominator, weights):
    scale = float(weights.sum())
    return float(np.sum(weights * numerator) / max(scale, _EPS) / (np.sum(weights * denominator) / max(scale, _EPS)))


def _bootstrap_indices(replicas, count, rng, support=None):
    if count <= 0 or replicas < 2:
        return np.empty((0, replicas), dtype=int)
    if support is None:
        return rng.integers(0, replicas, size=(count, replicas))
    support = np.asarray(support, dtype=int)
    if support.size == replicas:
        return rng.integers(0, replicas, size=(count, replicas))
    # Zero-weight replicas carry no contribution to the weighted estimator.
    # Restricting the resampling pool to positive support avoids a zero-weight
    # draw with an undefined weighted denominator while retaining complete
    # replica curves.
    positions = rng.integers(0, support.size, size=(count, replicas))
    return support[positions]


@dataclass(frozen=True)
class FluxSideResult:
    """Finite-time flux--side response and uncertainty diagnostics.

    ``correlation`` is ``<F(0) h(t)>`` in the supplied flux convention, and
    ``rate`` is this curve divided by the supplied reactant population.  The
    object intentionally carries ``physical_converged=False`` until callers
    verify the finite-surface, dynamics, plateau and source-chain limits.
    """

    time: np.ndarray
    correlation: np.ndarray
    rate: np.ndarray
    standard_error: np.ndarray | None
    bootstrap_rate: np.ndarray
    diagnostics: dict
    confidence_interval: np.ndarray | None = None

    def __post_init__(self):
        arrays = {}
        time = np.array(self.time, dtype=float, copy=True)
        if time.ndim != 1 or not np.isfinite(time).all():
            raise ValueError("time must be finite one-dimensional")
        for name in ("correlation", "rate"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.ndim not in (1, 3) or value.shape[0] != len(time) or not np.isfinite(value).all():
                raise ValueError(f"{name} must have shape [time] or [time,n,n]")
            if value.ndim == 3 and value.shape[1] != value.shape[2]:
                raise ValueError(f"{name} matrix samples must be square")
            value.flags.writeable = False
            arrays[name] = value
        time.flags.writeable = False
        arrays["time"] = time
        bootstrap = np.array(self.bootstrap_rate, dtype=float, copy=True)
        if bootstrap.ndim not in (2, 4) or bootstrap.shape[1] != len(time):
            raise ValueError("bootstrap_rate must have shape [draw,time] or [draw,time,n,n]")
        if bootstrap.ndim == 4 and bootstrap.shape[2] != bootstrap.shape[3]:
            raise ValueError("bootstrap_rate matrix samples must be square")
        if not np.isfinite(bootstrap).all():
            raise ValueError("bootstrap_rate must be finite")
        bootstrap.flags.writeable = False
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "bootstrap_rate", bootstrap)
        if self.standard_error is not None:
            error = np.array(self.standard_error, dtype=float, copy=True)
            if error.shape != arrays["rate"].shape or not np.isfinite(error).all():
                raise ValueError("standard_error must match the rate shape")
            error.flags.writeable = False
            object.__setattr__(self, "standard_error", error)
        if self.confidence_interval is not None:
            interval = np.array(self.confidence_interval, dtype=float, copy=True)
            expected = (2,) + arrays["rate"].shape
            if interval.shape != expected or not np.isfinite(interval).all():
                raise ValueError("confidence_interval must have shape [2] + rate.shape")
            if np.any(interval[0] > interval[1]):
                raise ValueError("confidence_interval lower bound exceeds upper bound")
            interval.flags.writeable = False
            object.__setattr__(self, "confidence_interval", interval)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def physical_converged(self):
        return False


def flux_side_rate(
    time,
    flux,
    side,
    reactant_population,
    *,
    sample_weight=None,
    bootstrap=0,
    rng=None,
    plateau_start=None,
    plateau_stop=None,
    relative_tolerance=0.1,
    confidence_level=0.95,
):
    """Estimate ``k_FS(t)=<F(0)h(t)>/<h_R>`` from complete replicas.

    ``flux`` must include the chosen delta-surface measure and signed surface
    velocity. ``side`` is a product-side indicator or a smoothed indicator in
    ``[0,1]``.  Scalar curves use ``[replica,time]`` and matrix-valued curves
    use ``[replica,time,n,n]``; a leading replica axis may be omitted for one
    curve.  A scalar reactant population applies to every replica; passing a
    vector preserves covariance when each replica has its own normalization.
    Bootstrap draws resample whole replicas and therefore require independent
    replicas for an uncertainty interpretation.  When requested, the result
    carries a percentile confidence interval and weighted effective sample
    size diagnostics.
    """
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be finite and positive")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and between zero and one")
    t = _time(time)
    f = _replica_series(flux, "flux")
    h = _replica_series(side, "side", side=True)
    if f.ndim == 4 and h.ndim == 2 and f.shape[:2] == h.shape:
        # A scalar side indicator naturally gates every matrix element of a
        # matrix-valued flux curve.
        h = np.broadcast_to(h[..., None, None], f.shape).copy()
    elif h.ndim == 4 and f.ndim == 2 and h.shape[:2] == f.shape:
        # Retain the symmetric broadcast for callers supplying a scalar flux
        # with a matrix-valued smooth side observable.
        f = np.broadcast_to(f[..., None, None], h.shape).copy()
    if f.shape != h.shape or f.shape[1] != len(t):
        raise ValueError("flux, side and time shapes must agree")
    replicas = f.shape[0]
    population = _positive_population(reactant_population, replicas)
    weights = _weights(sample_weight, replicas)
    per_replica_corr = f * h
    correlation = np.average(per_replica_corr, axis=0, weights=weights)
    denominator = float(np.average(population, weights=weights))
    rate = correlation / denominator
    if isinstance(bootstrap, (bool, np.bool_)) or int(bootstrap) != bootstrap or int(bootstrap) < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    bootstrap = int(bootstrap)
    if rng is None:
        rng = np.random.default_rng()
    if not hasattr(rng, "integers"):
        raise ValueError("rng must provide an integers method")
    draws = []
    support = np.flatnonzero(weights > 0)
    indices = _bootstrap_indices(replicas, bootstrap, rng, support=support)
    for ix in indices:
        wb = weights[ix]
        cb = per_replica_corr[ix]
        pb = population[ix]
        draws.append(np.average(cb, axis=0, weights=wb) / np.average(pb, weights=wb))
    if draws:
        draws = np.stack(draws, axis=0)
    else:
        draws = np.empty((0,) + rate.shape, dtype=float)
    error = draws.std(axis=0, ddof=1) if len(draws) >= 2 else None
    interval = (
        np.quantile(draws, [(1.0 - confidence_level) / 2.0,
                            1.0 - (1.0 - confidence_level) / 2.0], axis=0)
        if len(draws) >= 2 else None
    )
    effective_sample_size = float(weights.sum() ** 2 / np.sum(weights ** 2))
    bootstrap_effective_sample_size = (
        np.asarray(
            [wb.sum() ** 2 / np.sum(wb ** 2)
             for wb in (weights[ix] for ix in indices)],
            dtype=float,
        )
        if len(indices) else np.empty(0, dtype=float)
    )
    diagnostics = {
        "estimator": "equilibrium_flux_side",
        "normalization": "weighted_mean_F0h_t / weighted_mean_reactant_population",
        "replicas": int(replicas),
        "bootstrap_requested": int(bootstrap),
        "bootstrap_valid": int(len(draws)),
        "effective_sample_size": effective_sample_size,
        "bootstrap_effective_sample_size": bootstrap_effective_sample_size,
        "bootstrap_support_size": int(len(support)),
        "confidence_level": float(confidence_level),
        "confidence_interval_available": bool(interval is not None),
        "uncertainty_scope": "whole-replica bootstrap; independent replicas required",
        "physical_converged": False,
        "surface_and_side_convention": "caller-supplied flux and h in [0,1]",
        "plateau_detected": False,
        "candidate_rate": None,
        "plateau_diagnostics": None,
    }
    if (plateau_start is None) != (plateau_stop is None):
        raise ValueError("plateau_start and plateau_stop must be supplied together")
    if plateau_start is not None:
        start, stop = float(plateau_start), float(plateau_stop)
        if not np.isfinite(start + stop) or start < t[0] or stop > t[-1] or stop <= start:
            raise ValueError("invalid plateau interval")
        ix = np.flatnonzero((t >= start) & (t <= stop))
        if len(ix) < 2:
            raise ValueError("plateau interval needs at least two points")
        candidate = np.mean(rate[ix], axis=0)
        scale = float(np.max(np.abs(candidate)))
        excursion = float(np.max(np.abs(rate[ix] - candidate)))
        stable = bool(scale > 0 and excursion <= relative_tolerance * scale)
        diagnostics["candidate_rate"] = (
            float(candidate) if np.ndim(candidate) == 0 else np.array(candidate, copy=True)
        )
        diagnostics["plateau_detected"] = bool(stable and len(draws) >= 2)
        diagnostics["plateau_diagnostics"] = {
            "interval": [start, stop], "samples": int(len(ix)),
            "relative_tolerance": float(relative_tolerance),
            "maximum_excursion": excursion,
            "maximum_relative_excursion": excursion / max(scale, _EPS),
            "shape_stable": stable,
        }
    return FluxSideResult(t, correlation, rate, error, draws, diagnostics, interval)


def _curve(value, name):
    if isinstance(value, FluxSideResult):
        return np.asarray(value.rate, dtype=float)
    curve = np.asarray(value, dtype=float)
    if curve.ndim not in (1, 3) or not np.isfinite(curve).all():
        raise ValueError(f"{name} must be a finite [time] or [time,n,n] rate curve or FluxSideResult")
    if curve.ndim == 3 and curve.shape[1] != curve.shape[2]:
        raise ValueError(f"{name} matrix rate curve must be square")
    return curve


def hybrid_flux_side_correction(
    time,
    rpmd_full_rate,
    rpmd_sub_rate,
    quantum_sub_rate,
    *,
    mode="additive_defect",
    ridge=0.0,
):
    """Transfer a low-dimensional flux--side defect to a full RPMD curve.

    ``mode='additive_defect'`` uses ``k_RP,full + k_Q,sub - k_RP,sub`` and
    preserves the zero-defect and full-subspace algebraic limits.  The optional
    ``mode='inverse_defect'`` applies the corresponding scalar inverse formula
    with a positive ridge.  Both are controlled approximations: a low-dimensional
    quantum rate does not determine the exact rate of a coupled full system.
    """
    t = _time(time)
    full, sub, quantum = (_curve(v, n) for v, n in (
        (rpmd_full_rate, "rpmd_full_rate"),
        (rpmd_sub_rate, "rpmd_sub_rate"),
        (quantum_sub_rate, "quantum_sub_rate"),
    ))
    if not (full.shape == sub.shape == quantum.shape) or full.shape[0] != len(t):
        raise ValueError("all rate curves must match the time grid")
    ridge = float(ridge)
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    if mode == "additive_defect":
        corrected = full + quantum - sub
    elif mode == "inverse_defect":
        if ridge <= 0:
            raise ValueError("inverse_defect requires a positive ridge")
        if full.ndim == 1:
            corrected = np.divide(1.0, np.divide(1.0, full + ridge)
                                  + np.divide(1.0, quantum + ridge)
                                  - np.divide(1.0, sub + ridge)) - ridge
        else:
            corrected = np.empty_like(full)
            eye = np.eye(full.shape[-1])
            for i in range(len(t)):
                fi = np.linalg.pinv(full[i] + ridge * eye)
                qi = np.linalg.pinv(quantum[i] + ridge * eye)
                si = np.linalg.pinv(sub[i] + ridge * eye)
                corrected[i] = np.linalg.pinv(fi + qi - si) - ridge * eye
    else:
        raise ValueError("mode must be additive_defect or inverse_defect")
    diagnostics = {
        "mode": mode,
        "ridge": ridge,
        "inverse_defect_rms": float(np.sqrt(np.mean(np.abs(quantum - sub) ** 2))),
        "physical_converged": False,
        "approximation": "matched-subspace-flux-side-defect-transfer",
        "exact_general_rate_claim": False,
        "assumption": "the selected subspace carries the full-system quantum flux-side defect",
    }
    corrected = np.asarray(corrected, dtype=float)
    if not np.isfinite(corrected).all():
        raise FloatingPointError("flux-side correction produced non-finite values")
    return {"time": t, "rate": corrected, "diagnostics": diagnostics}


# Descriptive aliases used by workflow code and notebooks.
estimate_flux_side_rate = flux_side_rate
quantum_corrected_flux_side = hybrid_flux_side_correction

__all__ = [
    "FluxSideResult", "flux_side_rate", "estimate_flux_side_rate",
    "hybrid_flux_side_correction", "quantum_corrected_flux_side",
]
