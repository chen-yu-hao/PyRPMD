"""Fixed-cell geometry in atomic units, with lattice vectors stored as rows.

Propagation coordinates remain unwrapped. These operations are for PES/CV
geometry or explicit output; independently wrapping ring-polymer beads before
forming a centroid is not a valid trajectory-unwrapping procedure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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


def _readonly(value):
    """Own a small immutable array whose write flag cannot be re-enabled."""
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _reduce_periodic_basis(basis):
    """Shorten a lattice basis using integer row operations only.

    Size reduction and the Lovasz condition (delta=3/4) keep the exact
    closest-vector search from depending on an arbitrary, very sheared
    representation of the same lattice. This is a search preconditioner,
    not a minimum-image approximation: the QR sphere search below still
    certifies the nearest image. Only periodic vectors enter this routine.
    """
    reduced = np.array(basis, dtype=float, copy=True)
    level = 1
    for _ in range(256):
        if level >= len(reduced):
            break
        for previous in range(level - 1, -1, -1):
            triangular = np.linalg.qr(reduced.T, mode="reduced")[1]
            coefficient = triangular[previous, level] / triangular[previous, previous]
            if not np.isfinite(coefficient) or abs(coefficient) > 2 ** 52:
                # Do not introduce an unreliable integer operation. The
                # original bounded search retains its explicit failure mode.
                return np.array(basis, dtype=float, copy=True)
            multiple = int(np.rint(coefficient))
            if multiple:
                reduced[level] -= multiple * reduced[previous]
                if not np.isfinite(reduced).all():
                    return np.array(basis, dtype=float, copy=True)
        triangular = np.linalg.qr(reduced.T, mode="reduced")[1]
        coefficient = triangular[level - 1, level] / triangular[level - 1, level - 1]
        ratio = abs(triangular[level, level] / triangular[level - 1, level - 1])
        # Compare a dimensionless ratio instead of squaring cell lengths.
        if ratio >= 1 or ratio ** 2 >= 0.75 - coefficient ** 2:
            level += 1
        else:
            reduced[[level - 1, level]] = reduced[[level, level - 1]]
            level = max(1, level - 1)
    # Even if a floating-point boundary prevents termination, every completed
    # row operation preserves the lattice; the bounded exact search remains
    # the authority. The cap bounds construction work, not image accuracy.
    return reduced


@dataclass(frozen=True, eq=False, init=False)
class FixedCellGeometry:
    """Validated, immutable nearest-image geometry for a fixed cell.

    Prepare this once for repeated CV/PES geometry calls. The reduced lattice
    basis, orthogonality decision and QR decomposition do not depend on positions.
    Arrays own immutable snapshots, so mutating a caller's cell or flags
    cannot invalidate the cached decomposition. The free ``minimum_image``
    function retains its one-shot array API.
    """

    cell: np.ndarray | None
    pbc: np.ndarray
    basis: np.ndarray
    squared_lengths: np.ndarray
    orthogonal: np.ndarray | None
    triangular: np.ndarray | None

    def __init__(self, cell=None, pbc=None):
        cell, flags = validate_cell_pbc(cell, pbc)
        basis = np.empty((0, 3)) if cell is None else cell[flags]
        gram = basis @ basis.T
        diagonal = np.diag(gram)
        orthogonal = triangular = None
        if len(basis):
            normalized_gram = gram / np.sqrt(np.outer(diagonal, diagonal))
            if not np.allclose(normalized_gram, np.eye(len(basis)), rtol=0, atol=1e-14):
                basis = _reduce_periodic_basis(basis)
                gram = basis @ basis.T
                diagonal = np.diag(gram)
                normalized_gram = gram / np.sqrt(np.outer(diagonal, diagonal))
                if not np.allclose(normalized_gram, np.eye(len(basis)), rtol=0, atol=1e-14):
                    orthogonal, triangular = np.linalg.qr(basis.T, mode="reduced")
        for name, value in (
            ("cell", cell), ("pbc", flags), ("basis", basis),
            ("squared_lengths", diagonal), ("orthogonal", orthogonal),
            ("triangular", triangular),
        ):
            object.__setattr__(self, name, None if value is None else _readonly(value))

    def minimum_image(self, displacements):
        """Return exact shortest images without rebuilding fixed-cell data."""
        return _minimum_image(_vectors(displacements), self)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        # NumPy's ordinary pickle roundtrip makes arrays writable. Rebuild
        # through the constructor when a CV is sent to another process.
        return type(self), (self.cell, self.pbc)


def minimum_image(displacements, cell, pbc=None):
    """Return shortest Euclidean images for general triclinic/partial PBC.

    Fractional-coordinate rounding alone is incorrect in a skew cell. A QR
    closest-lattice-vector search enumerates the bounded sphere in at most
    three dimensions after integer reduction of the periodic lattice basis.
    The orthogonal-cell path is vectorized. Pathological searches fail explicitly
    instead of returning an unverified image.
    Repeated evaluations can prepare a ``FixedCellGeometry`` once instead.
    """
    d = _vectors(displacements)
    return _minimum_image(d, FixedCellGeometry(cell, pbc))


def _minimum_image(d, geometry):
    basis = geometry.basis
    if not len(basis):
        return d
    if geometry.triangular is None:
        coefficients = (d @ basis.T) / geometry.squared_lengths
        return d - np.rint(coefficients) @ basis
    orthogonal, triangular = geometry.orthogonal, geometry.triangular
    k = len(basis)
    flat = d.reshape(-1, 3)
    out = np.empty_like(flat)
    for index, vector in enumerate(flat):
        target = orthogonal.T @ vector
        nearest = np.zeros(k, dtype=np.int64)
        for level in range(k - 1, -1, -1):
            center = (target[level] - triangular[level, level + 1:] @ nearest[level + 1:]) / triangular[level, level]
            if abs(center) > np.iinfo(np.int64).max / 4:
                raise ValueError("displacement is too large for a reliable periodic image")
            nearest[level] = int(np.rint(center))
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


def validate_cell(cell, pbc=(False, False, False)):
    """Validate explicit boundary flags, returning the surface API tuple form."""
    value, flags = validate_cell_pbc(cell, pbc)
    return value, tuple(bool(flag) for flag in flags)
