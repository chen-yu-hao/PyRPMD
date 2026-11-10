"""Reduce explicitly constant metric grids to the shared constant-mass model."""

from dataclasses import replace

import numpy as np


def constant_metric_model(model):
    """Use m_sigma/g for a constant inverse metric; reject variable metrics.

    A callable is not a declaration of a constant metric. Sampling it at DVR
    nodes cannot prove it is constant along an RPMD trajectory, so callers
    must supply an explicitly constant grid for this exact reduction.
    """
    metric = getattr(model, "metric", None)
    if metric is None:
        return model
    if callable(metric):
        raise NotImplementedError(
            "Position-dependent/callable subspace metrics have no validated shared "
            "quantum/RPMD rate implementation; use a Cartesian SubspaceModel or "
            "an explicitly constant metric grid."
        )
    metric = np.asarray(metric, dtype=float)
    if not np.isfinite(metric).all() or np.any(metric <= 0):
        raise ValueError("metric must be positive and finite")
    value = float(metric.flat[0])
    if not np.all(metric == value):
        raise NotImplementedError(
            "Position-dependent subspace metrics require a consistent quantum "
            "measure, reversible sampler, constrained mean force and flux; "
            "use Cartesian coordinates instead."
        )
    masses = np.array(model.masses, copy=True)
    masses[0] /= value
    return replace(model, masses=masses, metric=None,
                   provenance={**model.provenance, "constant_inverse_metric": value})
