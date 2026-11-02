"""Nonperiodic bent-triatomic quasi-centroids and small constraint blocks.

This module owns molecular geometry only.  A quasi-centroid is obtained from
bead averages of two bond lengths and their included angle; it is not the
geometry of the Cartesian centroid.  All lengths and masses are atomic units.
"""

from __future__ import annotations

import numpy as np

from .cmd import _integer, _real, _scalar


class TriatomicQuasiCentroid:
    """Explicit, disjoint bent triatomics ordered ``(center, end1, end2)``.

    Every atom must belong to exactly one molecule.  Coordinates are unwrapped,
    nonperiodic Cartesian positions.  No minimum-image convention, molecular
    discovery, or permutation of terminal atoms is inferred.  Angles are in
    radians.  ``singularity_tolerance`` rejects zero bonds, nearly linear
    molecules and Cartesian centroids without an identifiable molecular plane.
    """

    def __setattr__(self, name, value):
        if self.__dict__.get("_frozen", False):
            raise AttributeError("quasi-centroid geometry is immutable; create a new geometry")
        object.__setattr__(self, name, value)

    def __init__(self, molecules, masses, *, singularity_tolerance=1e-10):
        mass = _real(masses, "masses")
        if mass.ndim != 1 or not len(mass) or np.any(mass <= 0):
            raise ValueError("masses must be a nonempty positive vector")
        elements = np.asarray(molecules, dtype=object)
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
               for value in elements.flat):
            raise ValueError("molecular atom indices must be integers, not booleans")
        raw = np.asarray(molecules)
        if (raw.dtype.kind not in "iu" or raw.ndim != 2 or raw.shape[1:] != (3,)
                or not len(raw) or np.any(raw < 0) or np.any(raw >= len(mass))):
            raise ValueError("molecules must be nonempty integer (center,end1,end2) triples")
        indices = np.array(raw, dtype=np.int64, copy=True)
        if not np.array_equal(np.sort(indices.ravel()), np.arange(len(mass))):
            raise ValueError("molecules must be disjoint triples covering every atom exactly once")
        tolerance = _scalar(singularity_tolerance, "singularity_tolerance")
        if tolerance >= 1e-2:
            raise ValueError("singularity_tolerance must be smaller than 0.01")
        self.masses, self.indices = mass, indices
        self.singularity_tolerance = tolerance
        self.natoms, self.nmolecules = len(mass), len(indices)
        self.local_masses = mass[indices]
        if not np.isfinite(self.local_masses.sum(axis=1)).all() or not np.isfinite(1 / mass).all():
            raise ValueError("molecular mass scale overflows")
        for value in (self.masses, self.indices, self.local_masses):
            value.flags.writeable = False
        self._frozen = True

    @property
    def molecules(self):
        """Read-only atom-index triples in their explicitly supplied order."""
        return self.indices

    def _positions(self, positions):
        q = _real(positions, "positions")
        if q.ndim < 2 or q.shape[-2:] != (self.natoms, 3) or min(q.shape) < 1:
            raise ValueError(f"positions must have trailing shape ({self.natoms},3)")
        return q

    def _vectors(self, positions):
        q = self._positions(positions)
        local = q[..., self.indices, :]
        a, b = local[..., 1, :] - local[..., 0, :], local[..., 2, :] - local[..., 0, :]
        r1, r2 = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
        if not np.isfinite(r1).all() or not np.isfinite(r2).all():
            raise ValueError("quasi-centroid bond geometry overflows")
        if np.any(r1 <= self.singularity_tolerance) or np.any(r2 <= self.singularity_tolerance):
            raise ValueError("quasi-centroid geometry contains a zero or singular bond")
        u, v = a / r1[..., None], b / r2[..., None]
        cosine = np.clip(np.sum(u * v, axis=-1), -1., 1.)
        sine = np.linalg.norm(np.cross(u, v), axis=-1)
        if np.any(sine <= self.singularity_tolerance):
            raise ValueError("quasi-centroid geometry requires bent, non-collinear triatomics")
        return q, r1, r2, u, v, cosine, sine

    def internal_coordinates(self, positions):
        """Return ``[...,molecule,3]`` values ``(r1,r2,theta)``.

        Input has shape ``[...,atom,3]``.  This function does not average beads.
        """
        _, r1, r2, _, _, cosine, sine = self._vectors(positions)
        return np.stack((r1, r2, np.arctan2(sine, cosine)), axis=-1)

    def internal_gradients(self, positions):
        """Local Cartesian derivatives ``[...,molecule,coordinate,atom,xyz]``.

        The local atom axis follows the explicit molecular triple.  Storing
        these three-atom blocks avoids a dense atom-by-atom derivative array.
        """
        _, r1, r2, u, v, cosine, sine = self._vectors(positions)
        result = np.zeros((*r1.shape, 3, 3, 3))
        result[..., 0, 0, :] = -u
        result[..., 0, 1, :] = u
        result[..., 1, 0, :] = -v
        result[..., 1, 2, :] = v
        first = (cosine[..., None] * u - v) / (r1 * sine)[..., None]
        second = (cosine[..., None] * v - u) / (r2 * sine)[..., None]
        result[..., 2, 0, :] = -first - second
        result[..., 2, 1, :] = first
        result[..., 2, 2, :] = second
        return result

    def mean_coordinates(self, positions):
        """Average internal coordinates over beads, for ``[P,N,3]`` or ``[F,P,N,3]``."""
        q = self._positions(positions)
        if q.ndim not in (3, 4):
            raise ValueError("ring positions must have shape [bead,atom,3] or [frame,bead,atom,3]")
        return self.internal_coordinates(q).mean(axis=-3)

    def quasi_centroid(self, positions):
        """Reconstruct Cartesian QC using mass-weighted Eckart alignment.

        The returned shape is ``[N,3]`` or ``[F,N,3]``.  Each molecular center
        of mass equals the Cartesian-centroid center of mass.  A proper
        mass-weighted Kabsch rotation aligns the internally reconstructed
        molecule to its Cartesian centroid.  Degenerate orientation is an
        explicit error, including rings spread uniformly over orientations.
        """
        q = self._positions(positions)
        if q.ndim not in (3, 4):
            raise ValueError("ring positions must have shape [bead,atom,3] or [frame,bead,atom,3]")
        values = self.mean_coordinates(q)
        center = q.mean(axis=-3)[..., self.indices, :]
        masses = self.local_masses[..., None]
        com = np.sum(center * masses, axis=-2, keepdims=True) / np.sum(masses, axis=-2, keepdims=True)
        reference = np.zeros((*values.shape[:-1], 3, 3))
        half_angle = values[..., 2] / 2
        reference[..., 1, 0] = values[..., 0] * np.cos(half_angle)
        reference[..., 1, 1] = values[..., 0] * np.sin(half_angle)
        reference[..., 2, 0] = values[..., 1] * np.cos(half_angle)
        reference[..., 2, 1] = -values[..., 1] * np.sin(half_angle)
        reference -= np.sum(reference * masses, axis=-2, keepdims=True) / np.sum(masses, axis=-2, keepdims=True)
        covariance = np.einsum("...mai,ma,...maj->...mij", reference, self.local_masses, center - com)
        left, singular, right = np.linalg.svd(covariance)
        if (np.any(singular[..., 0] <= self.singularity_tolerance)
                or np.any(singular[..., 1] <= self.singularity_tolerance * singular[..., 0])):
            raise ValueError("Cartesian centroid has degenerate molecular orientation; Eckart alignment is undefined")
        determinant = np.linalg.det(left @ right)
        left[..., :, -1] *= np.where(determinant < 0, -1., 1.)[..., None]
        rotation = left @ right
        fitted = np.einsum("...mai,...mij->...maj", reference, rotation) + com
        result = np.empty((*q.shape[:-3], self.natoms, 3))
        result[..., self.indices, :] = fitted
        return result

    def metadata(self):
        return {"geometry": "disjoint_bent_triatomics", "molecules": self.indices.tolist(),
                "masses_au": self.masses.tolist(), "singularity_tolerance": self.singularity_tolerance,
                "coordinates": ["bond1_bohr", "bond2_bohr", "angle_radian"], "periodic": False}


class QuasiCentroidConstraint:
    """Fix bead-mean internal coordinates with molecule-local 3-by-3 solves.

    Constraint values are ``mean_beads(r1,r2,theta)-target``.  The dimensional
    tolerance is applied independently in bohr and radians.  Position
    projection is a mass-weighted Newton projection; RATTLE additionally uses
    the initial Jacobian as its fixed impulse direction.
    """

    def __init__(self, geometry, target, *, tolerance=1e-10, max_iterations=50):
        if not isinstance(geometry, TriatomicQuasiCentroid):
            raise TypeError("geometry must be a TriatomicQuasiCentroid")
        values = _real(target, "target")
        if values.shape != (geometry.nmolecules, 3):
            raise ValueError("target must have shape [molecule,3]")
        if (np.any(values[:, :2] <= geometry.singularity_tolerance)
                or np.any(values[:, 2] <= geometry.singularity_tolerance)
                or np.any(values[:, 2] >= np.pi - geometry.singularity_tolerance)):
            raise ValueError("target requires positive bonds and bent angles in (0,pi)")
        self.geometry, self.target = geometry, values
        self.tolerance = _scalar(tolerance, "tolerance")
        self.max_iterations = _integer(max_iterations, "max_iterations")
        self.target.flags.writeable = False

    def _ring(self, q):
        value = self.geometry._positions(q)
        if value.ndim != 3:
            raise ValueError("constraint requires [bead,atom,3] positions")
        return value

    def values(self, q):
        return self.geometry.mean_coordinates(self._ring(q)) - self.target

    def local_jacobian(self, q):
        """Return ``[molecule,coordinate,bead,local_atom,xyz]`` blocks."""
        q = self._ring(q)
        return np.transpose(self.geometry.internal_gradients(q), (1, 2, 0, 3, 4)) / len(q)

    def metric(self, q):
        """Mass metric blocks ``G=C M^-1 C.T``; no global dense Hessian."""
        jacobian = self.local_jacobian(q)
        return np.einsum("mibac,ma,mjbac->mij", jacobian, 1 / self.geometry.local_masses, jacobian)

    def logdet_metric(self, q):
        sign, logdet = np.linalg.slogdet(self.metric(q))
        if np.any(sign <= 0) or not np.isfinite(logdet).all():
            raise ValueError("quasi-centroid constraint has a singular mass metric")
        return float(logdet.sum())

    def _scatter(self, local):
        result = np.empty((local.shape[1], self.geometry.natoms, 3))
        result[:, self.geometry.indices, :] = np.transpose(local, (1, 0, 2, 3))
        return result

    def project_positions(self, q, *, reference=None):
        """Return projected positions, leaving the input unchanged.

        With ``reference`` the impulse directions are held at that position,
        as required by RATTLE.  Failure raises instead of returning a partially
        constrained configuration.
        """
        result = self._ring(q)
        fixed = None if reference is None else self.local_jacobian(reference)
        inverse_mass = 1 / self.geometry.local_masses
        for _ in range(self.max_iterations):
            residual = self.values(result)
            if np.max(np.abs(residual)) <= self.tolerance:
                return result
            current = self.local_jacobian(result)
            direction = current if fixed is None else fixed
            metric = np.einsum("mibac,ma,mjbac->mij", current, inverse_mass, direction)
            try:
                multiplier = np.linalg.solve(metric, residual[..., None])[..., 0]
            except np.linalg.LinAlgError as error:
                raise ValueError("singular quasi-centroid position projection") from error
            local = np.einsum("mibac,mi,ma->mbac", direction, multiplier, inverse_mass)
            result -= self._scatter(local)
        raise ValueError("quasi-centroid position projection did not converge")

    def project_momenta(self, q, p):
        """Project canonical momenta onto ``C M^-1 p=0`` without changing inputs."""
        q = self._ring(q)
        p = _real(p, "momenta")
        if p.shape != q.shape:
            raise ValueError("momenta must match [bead,atom,3] positions")
        jacobian = self.local_jacobian(q)
        local_p = np.transpose(p[:, self.geometry.indices, :], (1, 0, 2, 3))
        inverse_mass = 1 / self.geometry.local_masses
        velocity = np.einsum("mibac,mbac,ma->mi", jacobian, local_p, inverse_mass)
        metric = np.einsum("mibac,ma,mjbac->mij", jacobian, inverse_mass, jacobian)
        try:
            multiplier = np.linalg.solve(metric, velocity[..., None])[..., 0]
        except np.linalg.LinAlgError as error:
            raise ValueError("singular quasi-centroid momentum projection") from error
        local = np.einsum("mibac,mi->mbac", jacobian, multiplier)
        return p - self._scatter(local)

    def velocity_residual(self, q, p):
        q = self._ring(q)
        p = _real(p, "momenta")
        if p.shape != q.shape:
            raise ValueError("momenta must match positions")
        local = np.transpose(p[:, self.geometry.indices, :], (1, 0, 2, 3))
        return np.einsum("mibac,mbac,ma->mi", self.local_jacobian(q), local,
                         1 / self.geometry.local_masses)

    def metadata(self):
        return {"geometry": self.geometry.metadata(), "target": self.target.tolist(),
                "tolerance": self.tolerance, "max_iterations": self.max_iterations}
