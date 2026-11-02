"""Nonperiodic bent-triatomic coordinates for quasi-centroid methods.

Lengths are in bohr and angles in radians. The internal quasi-centroid is
the arithmetic bead mean of each bond length and of the bond angle, not the
internal coordinates of the Cartesian centroid. Cartesian reconstruction
uses a mass-weighted Eckart/Kabsch fit to that centroid. This module does
not infer molecular connectivity or unwrap periodic molecules.
"""

from __future__ import annotations

import numpy as np

from .cmd import _integer, _real
from .potentials.base import coordinates


class TriatomicCoordinates:
    """Disjoint bent molecules, indexed ``(central, terminal1, terminal2)``.

    ``values(q)`` accepts ``[batch,atom,3]`` and returns
    ``[batch,molecule,3]`` in the order ``r1, r2, theta``. ``forces`` pulls
    derivatives with respect to those three coordinates back to Cartesian
    **negative** gradients. Molecules may cover only a subset of the atoms;
    unselected atoms receive zero correction force.

    Periodic cells, shared atoms, zero bonds and linear angles are explicitly
    unsupported. Molecular identity and atom order are supplied by the caller.
    """

    def __init__(self, molecules, natoms, *, cell=None, pbc=False):
        self.natoms = _integer(natoms, "natoms")
        raw = np.asarray(molecules)
        if (raw.dtype.kind not in "iu" or raw.ndim != 2 or raw.shape[1] != 3
                or len(raw) == 0 or np.any(raw < 0) or np.any(raw >= self.natoms)):
            raise ValueError("molecules must be a nonempty integer [molecule,3] array of valid atom indices")
        if len(np.unique(raw)) != raw.size:
            raise ValueError("triatomic molecules must have distinct, nonoverlapping atom indices")
        flags = np.asarray(pbc)
        if flags.dtype.kind != "b" or flags.shape not in ((), (3,)):
            raise ValueError("pbc must be a boolean or three booleans")
        if cell is not None or np.any(flags):
            raise ValueError("quasi-centroid triatomics currently support nonperiodic systems without a cell")
        self.molecules = np.array(raw, dtype=np.int64, copy=True)
        self.molecules.flags.writeable = False

    def _geometry(self, q):
        q = coordinates(q)
        if q.shape[1] != self.natoms:
            raise ValueError(f"coordinates must contain {self.natoms} atoms")
        atoms = q[:, self.molecules, :]
        u, v = atoms[:, :, 1] - atoms[:, :, 0], atoms[:, :, 2] - atoms[:, :, 0]
        r1, r2 = np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)
        if np.any(r1 <= 1e-12) or np.any(r2 <= 1e-12):
            raise ValueError("quasi-centroid coordinates require nonzero molecular bonds")
        a, b = u / r1[..., None], v / r2[..., None]
        cosine = np.clip(np.sum(a * b, axis=-1), -1., 1.)
        sine = np.linalg.norm(np.cross(a, b), axis=-1)
        if np.any(sine <= 1e-10):
            raise ValueError("quasi-centroid coordinates require bent, noncollinear triatomics")
        angle = np.arctan2(sine, cosine)
        return q, r1, r2, a, b, cosine, sine, angle

    def values(self, q):
        """Internal coordinates ``[batch,molecule,(r1,r2,theta)]``."""
        _, r1, r2, _, _, _, _, angle = self._geometry(q)
        return np.stack((r1, r2, angle), axis=-1)

    def forces(self, q, derivatives):
        """Return ``-sum_k (dV/ds_k) grad_q(s_k)`` without bead scaling."""
        q, r1, r2, a, b, cosine, sine, _ = self._geometry(q)
        d = _real(derivatives, "internal-coordinate derivatives")
        if d.shape != (len(q), len(self.molecules), 3):
            raise ValueError("derivatives must have shape [batch,molecule,3]")
        angle_a = (cosine[..., None] * a - b) / (r1 * sine)[..., None]
        angle_b = (cosine[..., None] * b - a) / (r2 * sine)[..., None]
        force_a = -d[..., 0, None] * a - d[..., 2, None] * angle_a
        force_b = -d[..., 1, None] * b - d[..., 2, None] * angle_b
        result = np.zeros_like(q)
        result[:, self.molecules[:, 1], :] = force_a
        result[:, self.molecules[:, 2], :] = force_b
        result[:, self.molecules[:, 0], :] = -force_a - force_b
        return result

    def _frames(self, q):
        raw = _real(q, "ring-polymer positions")
        single = raw.ndim == 3
        if single:
            raw = raw[None]
        if (raw.ndim != 4 or raw.shape[2:] != (self.natoms, 3)
                or raw.shape[0] == 0 or raw.shape[1] == 0):
            raise ValueError("positions must be [bead,atom,3] or [frame,bead,atom,3]")
        return raw, single

    def mean_values(self, q):
        """Bead-mean internal coordinates, optionally for several frames."""
        frames, single = self._frames(q)
        values = self.values(frames.reshape(-1, self.natoms, 3))
        mean = values.reshape(*frames.shape[:2], len(self.molecules), 3).mean(axis=1)
        return mean[0] if single else mean

    def quasicentroids(self, q, masses):
        """Reconstruct geometry using molecular COM and a weighted Eckart fit.

        An input ring ``[P,N,3]`` yields ``[N,3]``; a batch of rings yields
        ``[frame,N,3]``. Each molecule has the bead-mean bond lengths/angle
        and the molecular Cartesian centroid's COM. The fitted orientation
        minimizes ``sum_a m_a |Q_qc[a] - Q_centroid[a]|**2`` over proper
        rotations. Degenerate centroid orientations raise an error rather
        than selecting an arbitrary laboratory axis. Unselected atoms retain
        their ordinary Cartesian centroids.
        """
        frames, single = self._frames(q)
        mass = _real(masses, "masses")
        if mass.shape != (self.natoms,) or np.any(mass <= 0):
            raise ValueError("masses must be a positive vector with one entry per atom")
        mean = self.mean_values(frames)
        result = frames.mean(axis=1)
        for mi, indices in enumerate(self.molecules):
            weights = mass[indices]
            total = float(weights.sum())
            target = result[:, indices].copy()
            center = np.einsum("a,fac->fc", weights, target) / total
            target -= center[:, None, :]
            values = mean[:, mi]
            body = np.zeros_like(target)
            body[:, 1, 0] = values[:, 0]
            body[:, 2, 0] = values[:, 1] * np.cos(values[:, 2])
            body[:, 2, 1] = values[:, 1] * np.sin(values[:, 2])
            body -= (np.einsum("a,fac->fc", weights, body) / total)[:, None, :]
            covariance = np.einsum("fai,a,faj->fij", body, weights, target)
            left, singular, right = np.linalg.svd(covariance)
            if np.any(singular[:, 0] <= 1e-24) or np.any(singular[:, 1] <= 1e-12 * singular[:, 0]):
                raise ValueError("Cartesian centroid has a degenerate molecular orientation for an Eckart fit")
            handedness = np.linalg.det(left @ right)
            left[:, :, -1] *= np.where(handedness < 0, -1., 1.)[:, None]
            rotation = left @ right
            result[:, indices] = body @ rotation + center[:, None, :]
        return result[0] if single else result

    def to_dict(self):
        """Portable topology and geometric conventions for scientific identity."""
        return {"molecules": self.molecules.tolist(), "natoms": self.natoms,
                "coordinate_order": ["r1", "r2", "theta"],
                "units": ["bohr", "bohr", "radian"], "periodic": False,
                "quasicentroid": "arithmetic_bead_mean_internal_coordinates",
                "orientation": "mass_weighted_eckart_kabsch_to_cartesian_centroid"}
