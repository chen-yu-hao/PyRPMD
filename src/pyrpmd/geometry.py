"""Fixed-cell periodic geometry, using row lattice vectors in bohr.

Dynamics keeps continuous, unwrapped Cartesian coordinates. Minimum images
are an observable/PES operation; they must never be applied independently
to ring beads before a centroid or spring force is evaluated.
"""

from __future__ import annotations

import math
import numpy as np


def validate_cell(cell, pbc=(False, False, False)):
    """Return independent validated cell and strict boolean periodic flags."""
    flags = np.asarray(pbc)
    if flags.shape != (3,) or flags.dtype.kind != "b":
        raise ValueError("pbc must contain exactly three booleans")
    flags = tuple(bool(x) for x in flags)
    if cell is None:
        if any(flags):
            raise ValueError("periodic boundaries require a cell")
        return None, flags
    raw = np.asarray(cell)
    if raw.shape != (3, 3) or raw.dtype.kind not in "iuf":
        raise ValueError("cell must be a real numerical 3x3 matrix of row vectors")
    value = np.array(raw, dtype=float, order="C", copy=True)
    if not np.isfinite(value).all():
        raise ValueError("cell must contain finite values")
    singular = np.linalg.svd(value, compute_uv=False)
    if singular[-1] <= 1e-12 * max(singular[0], np.finfo(float).tiny):
        raise ValueError("cell must be nonsingular and well conditioned")
    return value, flags


def minimum_image(vectors, cell, pbc=(True, True, True)):
    """Return exact nearest-image displacement(s) for general fixed cells.

    ``vectors`` has shape ``(...,3)``. A QR closest-lattice-point search is
    used in the periodic subspace (dimension at most three). Rounding each
    fractional component is insufficient for skew cells and is not used.
    At a Voronoi boundary the nearest image and its derivative are not unique.
    """
    cell, flags = validate_cell(cell, pbc)
    raw = np.asarray(vectors)
    if raw.ndim < 1 or raw.shape[-1] != 3 or raw.dtype.kind not in "iuf":
        raise ValueError("displacements must be real arrays with final dimension 3")
    value = np.asarray(raw, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError("displacements must be finite")
    if not any(flags):
        return value.copy()
    basis = cell[np.asarray(flags)]
    orthogonal, triangular = np.linalg.qr(basis.T, mode="reduced")
    dim = len(basis)
    result = np.empty_like(value)
    for vector, target in zip(value.reshape(-1, 3), result.reshape(-1, 3)):
        projected = orthogonal.T @ vector
        # A finite Babai candidate bounds the exact sphere enumeration.
        candidate = np.zeros(dim, dtype=np.int64)
        for axis in range(dim - 1, -1, -1):
            center = (
                projected[axis] - triangular[axis, axis + 1:] @ candidate[axis + 1:]
            ) / triangular[axis, axis]
            if abs(center) > np.iinfo(np.int64).max / 4:
                raise ValueError("displacement is too large for a reliable periodic image")
            candidate[axis] = int(np.rint(center))
        best = candidate.copy()
        residual = projected - triangular @ best
        best_squared = float(residual @ residual)
        work = candidate.copy()

        def search(axis, partial_squared):
            nonlocal best, best_squared
            if axis < 0:
                if partial_squared < best_squared:
                    best, best_squared = work.copy(), partial_squared
                return
            remaining = max(0.0, best_squared - partial_squared)
            center = (
                projected[axis] - triangular[axis, axis + 1:] @ work[axis + 1:]
            ) / triangular[axis, axis]
            radius = math.sqrt(remaining) / abs(triangular[axis, axis])
            # The tolerance only expands the search; it never admits a worse image.
            tol = 8 * np.finfo(float).eps * max(1.0, abs(center), radius)
            lo, hi = math.ceil(center - radius - tol), math.floor(center + radius + tol)
            nearest = int(np.rint(center))
            for offset in range(max(nearest - lo, hi - nearest) + 1):
                choices = (nearest,) if offset == 0 else (nearest - offset, nearest + offset)
                for number in choices:
                    if number < lo or number > hi:
                        continue
                    delta = triangular[axis, axis] * (center - number)
                    cost = partial_squared + delta * delta
                    if cost <= best_squared + tol * max(1.0, best_squared):
                        work[axis] = number
                        search(axis - 1, cost)

        search(dim - 1, 0.0)
        target[:] = vector - best @ basis
    return result
