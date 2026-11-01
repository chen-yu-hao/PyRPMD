"""Per-bead, mass-weighted SHAKE/RATTLE for fixed pair distances.

Distances and coordinates are in bohr, masses in electron masses, and momenta
in atomic units. These constraints define a constrained Cartesian ring
Hamiltonian; they do not supply a rigid-rotor path integral or a Fixman term.
"""

from __future__ import annotations

from collections.abc import Mapping
import numbers

import numpy as np

from .geometry import FixedCellGeometry
from .units import parse_quantity


class ConstraintError(ValueError):
    """A constrained state or a SHAKE/RATTLE solve is invalid."""


def normalize_bond_constraints(value, natoms):
    """Return sorted ``(i, j, length_bohr)`` triples with unique atom pairs.

    Input mappings contain exactly ``atoms=[i,j]`` and ``length``. Numeric
    lengths are atomic units; explicit strings such as ``'0.96 angstrom'``
    are accepted. The Python API also accepts triples in atomic units.
    """
    if isinstance(value, (str, bytes, Mapping)) or value is None:
        raise ValueError("bond_constraints must be a sequence of pair-distance constraints")
    result, seen = [], set()
    for item in value:
        if isinstance(item, Mapping):
            if set(item) != {"atoms", "length"}:
                raise ValueError("a bond constraint requires exactly atoms and length")
            atoms, length = item["atoms"], item["length"]
        else:
            if not isinstance(item, (tuple, list, np.ndarray)) or len(item) != 3:
                raise ValueError("a bond constraint must be an atoms/length mapping or (i,j,length)")
            atoms, length = item[:2], item[2]
        if (not isinstance(atoms, (tuple, list, np.ndarray)) or len(atoms) != 2
                or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, numbers.Integral)
                       or not 0 <= i < natoms for i in atoms) or atoms[0] == atoms[1]):
            raise ValueError("bond constraint atoms must be two distinct in-range integer indices")
        pair = tuple(sorted(int(i) for i in atoms))
        if pair in seen:
            raise ValueError("duplicate bond constraint atom pair")
        if isinstance(length, (bool, np.bool_)):
            raise ValueError("bond constraint length must be positive and finite")
        length = parse_quantity(length, "length")
        if not np.isfinite(length) or length <= 0:
            raise ValueError("bond constraint length must be positive and finite")
        seen.add(pair)
        result.append((*pair, float(length)))
    return tuple(sorted(result))


class BondConstraints:
    """SHAKE position and RATTLE momentum projections on every bead.

    ``tolerance`` bounds relative bond-length error and the fractional bond
    extension rate in inverse atomic time. Every solve is bounded. Callers
    use working copies and commit only after all constraints and PES calls
    succeed. Periodic bonds use one fixed nearest image during each drift.
    """

    def __init__(self, masses, bonds, *, cell=None, pbc=None, tolerance=1e-12,
                 max_iterations=1000):
        raw = np.asarray(masses)
        if raw.ndim != 1 or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all() or np.any(raw <= 0):
            raise ValueError("constraint masses must be finite and positive")
        self.masses = np.array(raw, dtype=float, copy=True)
        self.masses.flags.writeable = False
        self.bonds = normalize_bond_constraints(bonds, len(raw))
        self.geometry = FixedCellGeometry(cell, pbc)
        if (isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, numbers.Real)
                or not np.isfinite(tolerance) or not 0 < tolerance < 1e-2):
            raise ValueError("constraint tolerance must be finite and in (0, 0.01)")
        if (isinstance(max_iterations, (bool, np.bool_))
                or not isinstance(max_iterations, numbers.Integral) or max_iterations < 1):
            raise ValueError("constraint max_iterations must be a positive integer")
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        if self.geometry.pbc.any():
            heights = 1.0 / np.linalg.norm(np.linalg.inv(self.geometry.cell), axis=0)
            limit = 0.5 * heights[self.geometry.pbc].min()
            if any(length >= limit for _, _, length in self.bonds):
                raise ValueError("bond constraints must be shorter than half the periodic cell height")

    def _check_arrays(self, q, p=None):
        arrays = (q,) if p is None else (q, p)
        for array in arrays:
            if (not isinstance(array, np.ndarray) or array.dtype != np.float64
                    or array.ndim != 3 or array.shape[1:] != (len(self.masses), 3)
                    or array.shape[0] < 1 or not np.isfinite(array).all()):
                raise ConstraintError("constraints require finite float64 arrays [P,N,3]")
        if p is not None and (p.shape != q.shape or np.shares_memory(q, p)):
            raise ConstraintError("constraint positions and momenta must be separate equal-shaped arrays")

    def residuals(self, q, p=None):
        self._check_arrays(q, p)
        position, velocity = 0.0, 0.0
        for i, j, length in self.bonds:
            r = self.geometry.minimum_image(q[:, i] - q[:, j])
            position = max(position, float(np.max(np.abs(np.linalg.norm(r, axis=-1) / length - 1))))
            if p is not None:
                v = p[:, i] / self.masses[i] - p[:, j] / self.masses[j]
                velocity = max(velocity, float(np.max(np.abs(np.sum(r * v, axis=-1))) / length**2))
        return position, velocity

    def check(self, q, p=None):
        position, velocity = self.residuals(q, p)
        # A small roundoff allowance avoids rejecting a just-converged solve.
        if position > self.tolerance * 1.05 or velocity > self.tolerance * 1.05:
            raise ConstraintError("state violates bond constraints; initialize positions/momenta explicitly")

    def project_positions(self, q, *, previous=None, momenta=None, dt=None):
        """Solve distances; old gradients give the SHAKE impulse during drift.

        With ``previous=None`` this explicitly prepares an initial geometry.
        During integration the position impulse also corrects momenta by
        ``mass * displacement / dt``; omitting that impulse is not RATTLE.
        """
        self._check_arrays(q, momenta)
        if previous is not None:
            self._check_arrays(previous)
            if previous.shape != q.shape or momenta is None or not np.isfinite(dt) or dt == 0:
                raise ConstraintError("SHAKE drift needs previous positions, momenta and nonzero finite dt")
        elif momenta is not None:
            raise ConstraintError("initial position projection does not accept a drift momentum")
        old = q.copy() if previous is None else previous
        directions, images = [], []
        for i, j, length in self.bonds:
            r = self.geometry.minimum_image(old[:, i] - old[:, j])
            if np.any(np.linalg.norm(r, axis=-1) <= 1e-14 * length):
                raise ConstraintError("coincident constrained atoms have no defined bond direction")
            directions.append(r)
            images.append(old[:, i] - old[:, j] - r)
        for _ in range(self.max_iterations):
            for (i, j, length), reference, image in zip(self.bonds, directions, images):
                r = q[:, i] - q[:, j] - image
                square = np.sum(r * r, axis=-1)
                active = np.abs(np.sqrt(square) / length - 1) > self.tolerance
                if not active.any():
                    continue
                direction = r if previous is None else reference
                dot = np.sum(r * direction, axis=-1)
                if np.any(dot[active] <= 1e-14 * length**2):
                    raise ConstraintError("SHAKE bond turned through 90 degrees; reduce the timestep")
                impulse = np.zeros_like(r)
                scale = -(square[active] - length**2) / (
                    2 * (1 / self.masses[i] + 1 / self.masses[j]) * dot[active])
                impulse[active] = scale[:, None] * direction[active]
                q[:, i] += impulse / self.masses[i]
                q[:, j] -= impulse / self.masses[j]
                if momenta is not None:
                    momenta[:, i] += impulse / dt
                    momenta[:, j] -= impulse / dt
            if self.residuals(q)[0] <= self.tolerance:
                return
        raise ConstraintError("SHAKE bond positions did not converge within max_iterations")

    def project_momenta(self, q, p):
        """Condition a Maxwell draw or final kick onto the mass-metric tangent."""
        self._check_arrays(q, p)
        vectors = [self.geometry.minimum_image(q[:, i] - q[:, j]) for i, j, _ in self.bonds]
        for _ in range(self.max_iterations):
            for (i, j, length), r in zip(self.bonds, vectors):
                v = p[:, i] / self.masses[i] - p[:, j] / self.masses[j]
                dot = np.sum(r * v, axis=-1)
                active = np.abs(dot) / length**2 > self.tolerance
                if not active.any():
                    continue
                square = np.sum(r[active] ** 2, axis=-1)
                if np.any(square <= 1e-28 * length**2):
                    raise ConstraintError("coincident constrained atoms have no defined tangent")
                impulse = (dot[active] / ((1 / self.masses[i] + 1 / self.masses[j]) * square))[:, None] * r[active]
                p[active, i] -= impulse
                p[active, j] += impulse
            if self.residuals(q, p)[1] <= self.tolerance:
                return
        raise ConstraintError("RATTLE bond momenta did not converge within max_iterations")
