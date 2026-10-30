"""Fixed-cell geometric CVs and continuous linear migration coordinates.

Distances/angles use shortest lattice images. Projections remain unwrapped:
one local migration barrier must not jump to a different surface when a bead
or an entire centroid crosses the simulation-cell boundary.
"""

from __future__ import annotations

import numpy as np

from .cv import (
    PeriodicDistance, BondDifference, Angle, _indices, _distance,
    _AffineCoordinate, _PeriodicCellCV,
)
from .geometry import validate_cell_pbc


# Keep the historical import path as an alias to the public factory's class.
# Value, gradient and optional Hessian evaluation now have one implementation.


class PeriodicBondDifference(_PeriodicCellCV, BondDifference):
    def __init__(self, cell, pbc, **kwargs):
        super().__init__(**kwargs)
        self._bind_geometry(cell, pbc)

    def centroid_value_gradient(self, q):
        a, ga = _distance(q, *self.bonds[0], geometry=self._geometry)
        b, gb = _distance(q, *self.bonds[1], geometry=self._geometry)
        return a - b, ga - gb


class PeriodicAngle(_PeriodicCellCV, Angle):
    def __init__(self, atoms, cell, pbc, **kwargs):
        super().__init__(atoms, **kwargs)
        self._bind_geometry(cell, pbc)

    def centroid_value_gradient(self, q):
        a, b, c = self.atoms
        if min(a, b, c) < 0 or max(a, b, c) >= len(q):
            raise ValueError("angle atom index out of range")
        # Within one image branch this is a constant translation, so the
        # ordinary analytic angle gradient is exactly the desired derivative.
        image = np.array(q, dtype=float, copy=True)
        image[a] = q[b] + self._geometry.minimum_image(q[a] - q[b])
        image[c] = q[b] + self._geometry.minimum_image(q[c] - q[b])
        return super().centroid_value_gradient(image)


class LinearProjection(_AffineCoordinate):
    """Unwrapped atom displacement along a unit direction, optionally relative
    to a mass-weighted reference group. Its mass metric is constant.
    """

    units = ("bohr",)

    def __init__(self, atom, direction, masses, reference_atoms=None, origin=0.0, **kwargs):
        super().__init__(**kwargs)
        self.atom = _indices([atom])[0]
        m = np.asarray(masses, dtype=float)
        if m.ndim != 1 or not np.isfinite(m).all() or np.any(m <= 0):
            raise ValueError("Projection requires positive finite atomic masses")
        if self.atom < 0 or self.atom >= len(m):
            raise ValueError("Projection atom index out of range")
        vector = np.asarray(direction)
        if vector.shape != (3,) or vector.dtype.kind not in "iuf":
            raise ValueError("Projection direction must be a real 3-vector")
        vector = vector.astype(float)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-15:
            raise ValueError("Projection direction must be finite and nonzero")
        self.direction = vector / norm
        self.weights = np.zeros(len(m))
        self.weights[self.atom] = 1.0
        if reference_atoms is not None:
            refs = _indices(reference_atoms)
            if not refs or len(set(refs)) != len(refs) or min(refs) < 0 or max(refs) >= len(m) or self.atom in refs:
                raise ValueError("Projection reference atoms must be distinct, in range and exclude the migrating atom")
            self.weights[list(refs)] = -m[list(refs)] / np.sum(m[list(refs)])
        self.origin = float(origin)
        if not np.isfinite(self.origin):
            raise ValueError("Projection origin must be finite")

    def _affine_gradient(self):
        # Retain the public direction/weight semantics, including the
        # unnormalized reciprocal vector installed by FractionalCoordinate.
        return self.weights[:, None] * self.direction[None]


class FractionalCoordinate(LinearProjection):
    """Continuous fractional coordinate along one reciprocal cell vector.

No modulo operation is applied. Umbrella windows and local rate surfaces
therefore refer to explicit periodic images. Units are dimensionless.
    """

    units = ("dimensionless",)

    def __init__(self, atom, axis, cell, pbc, masses, **kwargs):
        self.cell, self.pbc = validate_cell_pbc(cell, pbc)
        if self.cell is None:
            raise ValueError("Fractional coordinates require a cell")
        axis = {"x": 0, "y": 1, "z": 2}.get(axis, axis)
        if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)) or axis not in (0, 1, 2):
            raise ValueError("Fractional coordinate axis must be 0, 1 or 2")
        self.axis = int(axis)
        vector = np.linalg.inv(self.cell)[:, self.axis]
        super().__init__(atom, vector, masses, **kwargs)
        self.direction = vector
