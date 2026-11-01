"""Euclidean mean-force string geometry, not a free-energy sampler.

The oracle must provide a consistent free-energy gradient in scaled Cartesian
or constant-Euclidean-metric CVs. General nonlinear CVs need their metric and
Jacobian treatment outside this API; a PES gradient yields only an MEP.
No sampling error, kinetic rate or physical convergence is inferred here.
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


def _real_array(value, name):
    value = np.asarray(value)
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite real numerical values")
    return np.array(value, dtype=np.float64, copy=True)


def _positive_real(value, name):
    array = _real_array(value, name)
    if array.ndim != 0 or array <= 0:
        raise ValueError(f"{name} must be a positive finite real scalar")
    return float(array)


def _positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validated(points):
    points = _real_array(points, "images")
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        raise ValueError("images must be finite [n_images>=3, n_cv>=2]")
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if not np.isfinite(distances).all() or np.any(distances <= 0):
        raise ValueError("adjacent string images must be distinct with finite distances")
    return points


def _reparameterize(points, n):
    points = _validated(points)
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if not np.isfinite(arc).all() or np.any(np.diff(arc) <= 0):
        raise ValueError("degenerate or numerically unresolved path arc length")
    positions = np.linspace(0.0, arc[-1], n)
    out = np.column_stack([np.interp(positions, arc, points[:, j])
                           for j in range(points.shape[1])])
    out[0], out[-1] = points[0], points[-1]
    return _validated(out)


def _perpendicular(path, oracle):
    tangent = path[2:] - path[:-2]
    norm = np.linalg.norm(tangent, axis=1)
    if not np.isfinite(norm).all() or np.any(norm <= 0):
        raise ValueError("undefined string tangent")
    tangent /= norm[:, None]
    gradient = _real_array(oracle(path.copy()), "mean force gradient")
    if gradient.shape != path.shape:
        raise ValueError("mean force gradient must have shape [n_images,n_cv]")
    interior = gradient[1:-1]
    force = interior - np.sum(interior * tangent, axis=1)[:, None] * tangent
    residual = float(np.max(np.linalg.norm(force, axis=1)))
    if not np.isfinite(residual):
        raise ValueError("nonfinite perpendicular force")
    return force, residual


def adaptive_mean_force_string(images, mean_force_gradient, *, step=0.01,
                               tolerance=1e-3, max_iterations=2000,
                               max_images=80, insertion_threshold=0.25,
                               adapt_every=25):
    """Relax a fixed-endpoint string and insert images at sharp bends.

    ``iterations`` counts completed updates. ``history`` includes the initial
    residual and one value per update; its last entry ALWAYS describes the
    returned images, including when the update budget is exhausted. Convergence
    is a deterministic perpendicular-gradient criterion only. A noisy mean-force
    oracle needs separate uncertainty and sampling-convergence checks.
    """
    path = _validated(images)
    if not callable(mean_force_gradient):
        raise TypeError("mean_force_gradient must be callable")
    step = _positive_real(step, "step")
    tolerance = _positive_real(tolerance, "tolerance")
    threshold = _positive_real(insertion_threshold, "insertion_threshold")
    if threshold >= np.pi:
        raise ValueError("insertion_threshold must be less than pi")
    max_iterations = _positive_integer(max_iterations, "max_iterations")
    max_images = _positive_integer(max_images, "max_images")
    adapt_every = _positive_integer(adapt_every, "adapt_every")
    if max_images < len(path):
        raise ValueError("max_images cannot be smaller than the initial string")
    force, residual = _perpendicular(path, mean_force_gradient)
    records = [residual]
    iteration = 0
    while residual > tolerance and iteration < max_iterations:
        moved = path.copy()
        moved[1:-1] -= step * force
        path = _reparameterize(moved, len(moved))
        iteration += 1
        if iteration % adapt_every == 0 and len(path) < max_images:
            segment = np.diff(path, axis=0)
            unit = segment / np.linalg.norm(segment, axis=1)[:, None]
            turns = np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1, 1))
            candidates = np.where(turns > threshold)[0] + 1
            if len(candidates):
                pivot = int(candidates[np.argmax(turns[candidates - 1])])
                midpoint = (path[pivot] + path[pivot + 1]) * 0.5
                path = _reparameterize(np.insert(path, pivot + 1, midpoint, axis=0), len(path) + 1)
        force, residual = _perpendicular(path, mean_force_gradient)
        records.append(residual)
    return StringResult(path.copy(), residual, residual <= tolerance,
                        iteration, tuple(records))
