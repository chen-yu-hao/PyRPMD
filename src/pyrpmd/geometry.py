"""Fixed-cell geometry in atomic units, with lattice vectors stored as rows.

Propagation coordinates remain unwrapped. These operations are for PES/CV
geometry or explicit output; independently wrapping ring-polymer beads before
forming a centroid is not a valid trajectory-unwrapping procedure.
"""

from __future__ import annotations

import math
import numpy as np


def validate_cell_pbc(cell=None, pbc=None):
    """Copy and validate a full-rank cell and three explicit boolean flags.

    A supplied cell defaults to full periodicity. A molecule without a cell
    defaults to no periodicity. Partial periodicity still uses a nonsingular
    3x3 cell, including a finite nonperiodic vector for slabs/wires.
    """
    if pbc is None:
        flags = np.full(3, cell is not None, dtype=bool)
    else:
        value = np.asarray(pbc)
        if value.shape != (3,) or value.dtype.kind != "b":
            raise ValueError("pbc must contain exactly three booleans")
        flags = value.copy()
    if cell is None:
        if flags.any():
            raise ValueError("Periodic boundary conditions require a cell")
        return None, flags
    value = np.asarray(cell)
    if value.shape != (3, 3) or value.dtype.kind not in "iuf":
        raise ValueError("cell must be a real 3x3 matrix of row lattice vectors")
    value = np.array(value, dtype=float, order="C", copy=True)
    if not np.isfinite(value).all():
        raise ValueError("cell must contain only finite values")
    singular = np.linalg.svd(value, compute_uv=False)
    if singular[0] <= 0 or singular[-1] <= 1e-12 * singular[0]:
        raise ValueError("cell must be nonsingular and numerically well conditioned")
    return value, flags


def _vectors(value):
    a = np.asarray(value)
    if a.ndim < 1 or a.shape[-1] != 3 or a.dtype.kind not in "iuf":
        raise ValueError("Coordinates/displacements must be real arrays ending in 3")
    a = np.array(a, dtype=float, copy=True)
    if not np.isfinite(a).all():
        raise ValueError("Coordinates/displacements must be finite")
    return a


def wrap_positions(positions, cell, pbc=None):
    """Return images in the [0,1) fractional cell along periodic directions."""
    q = _vectors(positions)
    cell, flags = validate_cell_pbc(cell, pbc)
    if not flags.any():
        return q
    fractional = q @ np.linalg.inv(cell)
    images = np.zeros_like(fractional)
    images[..., flags] = np.floor(fractional[..., flags])
    return q - images @ cell


def minimum_image(displacements, cell, pbc=None):
    """Return shortest Euclidean images for general triclinic/partial PBC.

    Fractional-coordinate rounding alone is incorrect in a skew cell. A QR
    closest-lattice-vector search enumerates the bounded sphere in at most
    three dimensions. The orthogonal-cell path is vectorized. Pathological
    searches fail explicitly instead of returning an unverified image.
    """
    d = _vectors(displacements)
    cell, flags = validate_cell_pbc(cell, pbc)
    if not flags.any():
        return d
    basis = cell[flags]
    gram = basis @ basis.T
    normalized_gram = gram / np.sqrt(np.outer(np.diag(gram), np.diag(gram)))
    if np.allclose(normalized_gram, np.eye(len(basis)), rtol=0, atol=1e-14):
        coefficients = (d @ basis.T) / np.diag(gram)
        return d - np.rint(coefficients) @ basis
    orthogonal, triangular = np.linalg.qr(basis.T, mode="reduced")
    k = len(basis)
    flat = d.reshape(-1, 3)
    out = np.empty_like(flat)
    for index, vector in enumerate(flat):
        target = orthogonal.T @ vector
        nearest = np.zeros(k, dtype=np.int64)
        for level in range(k - 1, -1, -1):
            nearest[level] = int(np.rint(
                (target[level] - triangular[level, level + 1:] @ nearest[level + 1:])
                / triangular[level, level]
            ))
        # Remove the possibly very large global image before the sphere search.
        reduced = vector - nearest @ basis
        target = orthogonal.T @ reduced
        best = float(target @ target)
        winner = np.zeros(k, dtype=np.int64)
        trial = np.zeros(k, dtype=np.int64)
        visits = 0

        def search(level, partial):
            nonlocal best, winner, visits
            visits += 1
            if visits > 100000:
                raise ValueError("cell is too skew for a bounded minimum-image search")
            if level < 0:
                if partial < best:
                    best = partial
                    winner = trial.copy()
                return
            center = (target[level] - triangular[level, level + 1:] @ trial[level + 1:]) / triangular[level, level]
            tolerance = 32 * np.finfo(float).eps * max(1.0, best)
            radius = math.sqrt(max(0.0, best - partial + tolerance)) / abs(triangular[level, level])
            lo, hi = math.ceil(center - radius), math.floor(center + radius)
            if hi - lo > 100000:
                raise ValueError("cell is too skew for a bounded minimum-image search")
            # Near-centre candidates tighten the sphere early.
            for value in sorted(range(lo, hi + 1), key=lambda n: abs(n - center)):
                term = (triangular[level, level] * (value - center)) ** 2
                if partial + term <= best + tolerance:
                    trial[level] = value
                    search(level - 1, partial + term)

        search(k - 1, 0.0)
        out[index] = reduced - winner @ basis
    return out.reshape(d.shape)
