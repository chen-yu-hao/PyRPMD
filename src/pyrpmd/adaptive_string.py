"""Adaptive finite-temperature string geometry from a supplied mean-force oracle.

The caller must provide a CONSISTENT gradient of free energy in collective
variables (including any metric/Jacobian correction). Feeding an ordinary PES
gradient only gives a minimum-energy string, not a free-energy path.

This independent analysis layer does not change the NRPMD dynamics or claim
that a grand-canonical electronic ensemble is equivalent to a mapping ensemble.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StringResult:
    images: np.ndarray
    maximum_perpendicular_force: float
    converged: bool
    iterations: int
    history: tuple


def _validated(points):
    points = np.asarray(points, dtype=np.float64)
    if (points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2
            or not np.isfinite(points).all()):
        raise ValueError("images must be finite [n_images>=3, n_cv>=2]")
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 0):
        raise ValueError("adjacent string images must be distinct")
    return points.copy()


def _reparameterize(points, n):
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(distances)))
    if arc[-1] <= 0:
        raise ValueError("degenerate path")
    positions = np.linspace(0.0, arc[-1], n)
    out = np.column_stack([np.interp(positions, arc, points[:, j])
                           for j in range(points.shape[1])])
    out[0], out[-1] = points[0], points[-1]
    return out


def adaptive_mean_force_string(images, mean_force_gradient, *, step=0.01,
                               tolerance=1e-3, max_iterations=2000,
                               max_images=80, insertion_threshold=0.25,
                               adapt_every=25):
    """Relax interior images normal to path and adapt by turning angle.

    Parameters are in the caller's CV units. Gradient must be of the
    dimensionless or appropriately scaled *free energy*, not a biased force;
    sampling uncertainties should be converged before each update.
    No statistical errors or kinetics are estimated by this routine.
    """
    path = _validated(images)
    if not callable(mean_force_gradient):
        raise TypeError("mean_force_gradient must be callable")
    if not (np.isfinite(step) and step > 0 and np.isfinite(tolerance)
            and tolerance > 0 and np.isfinite(insertion_threshold)
            and 0 < insertion_threshold < np.pi):
        raise ValueError("invalid step, tolerance or insertion threshold")
    if (not isinstance(max_iterations, int) or max_iterations < 1
            or not isinstance(max_images, int) or max_images < len(path)
            or not isinstance(adapt_every, int) or adapt_every < 1):
        raise ValueError("invalid iteration/image limits")
    records = []
    for iteration in range(1, max_iterations + 1):
        tangents = np.empty_like(path)
        tangents[1:-1] = path[2:] - path[:-2]
        tangents[0] = path[1] - path[0]
        tangents[-1] = path[-1] - path[-2]
        norm = np.linalg.norm(tangents, axis=1)
        if np.any(norm <= 0):
            raise ValueError("undefined string tangent")
        tangents /= norm[:, None]
        gradient = np.asarray(mean_force_gradient(path.copy()), dtype=float)
        if gradient.shape != path.shape or not np.isfinite(gradient).all():
            raise ValueError("mean force gradient must be finite [n_images,n_cv]")
        perpendicular = gradient - np.sum(gradient * tangents, axis=1)[:, None] * tangents
        residual = float(np.max(np.linalg.norm(perpendicular[1:-1], axis=1)))
        records.append(residual)
        if residual <= tolerance:
            return StringResult(path, residual, True, iteration, tuple(records))
        moved = path.copy()
        moved[1:-1] -= step * perpendicular[1:-1]
        path = _reparameterize(moved, len(moved))
        if iteration % adapt_every == 0 and len(path) < max_images:
            segment = np.diff(path, axis=0)
            unit = segment / np.linalg.norm(segment, axis=1)[:, None]
            turns = np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1, 1))
            candidates = np.where(turns > insertion_threshold)[0] + 1
            if len(candidates):
                # Insert at most one image per adaptation, at the sharpest bend.
                pivot = int(candidates[np.argmax(turns[candidates - 1])])
                midpoint = (path[pivot] + path[pivot + 1]) * 0.5
                path = np.insert(path, pivot + 1, midpoint, axis=0)
                path = _reparameterize(path, len(path))
    return StringResult(path, records[-1], False, max_iterations, tuple(records))
