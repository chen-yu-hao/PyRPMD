"""Ring-polymer centroid moments along the RPMD rate coordinate.

During the umbrella stage of a gas-phase rate run every recorded ring
polymer is aligned (mass-weighted Kabsch) onto the reference transition-state
geometry, after assigning it to the nearest equivalent reaction path.  Each
window accumulates additive raw moments, so replicas and checkpoint resumes
combine exactly:

* the mean aligned centroid ``xbar(xi)`` -- the ring-polymer mean path;
* the centroid covariance (mass weighted) -- orientational and thermal spread;
* the bead-about-centroid covariance ``Sigma(xi)`` (mass weighted) -- the
  quantum width of the ring polymer in every direction.

:func:`mean_path_from_rate_run` reads the result.  :class:`RingPolymerMeanPath`
keeps only the orientation-locked windows for a path (near the reactant
asymptote the fragments rotate freely and a mean geometry has no meaning),
computes the path curvature ``kappa`` and the width ``rho = sqrt(2 n^T Sigma
n)`` along the curvature normal ``n``, and proposes the transverse direction
for a two-dimensional subspace when ``kappa*rho`` is large.  Coordinates are
bohr; mass-weighted quantities use atomic mass units of electron mass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PREFIX = "centroid_path_"
FILENAME = "centroid_path.npz"
SCHEMA = "pyrpmd.centroid_path"
MAX_COVARIANCE_DIMENSION = 900
_FIELDS = ("count", "sum_xi", "sum_xi2", "sum_centroid", "sum_centroid_outer",
           "sum_bead_covariance", "permutation_counts", "sum_distance")


def kabsch_rotation(mobile: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Proper rotation ``R`` minimizing ``sum_i w_i |mobile_i R - target_i|^2``.

    Both point sets must already be centred at their weighted centre.
    """
    h = (mobile * weights[:, None]).T @ target
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    return u @ np.diag([1.0, 1.0, d]) @ vt


def rigid_body_modes(geometry: Any, masses: Any, *, tolerance: float = 1e-8) -> np.ndarray:
    """Orthonormal mass-weighted translations and rotations, ``[3N, 5 or 6]``."""
    x = np.asarray(geometry, dtype=float).reshape(-1, 3)
    m = np.asarray(masses, dtype=float)
    sqrt_m = np.sqrt(m)[:, None]
    centre = np.sum(x * m[:, None], axis=0) / m.sum()
    r = x - centre
    vectors = []
    for axis in range(3):
        unit = np.zeros(3)
        unit[axis] = 1.0
        vectors.append((sqrt_m * unit[None, :]).ravel())
        vectors.append((sqrt_m * np.cross(unit[None, :], r)).ravel())
    q, s, _ = np.linalg.svd(np.column_stack(vectors), full_matrices=False)
    return q[:, s > tolerance * s.max()]


class CentroidAccumulator:
    """Additive aligned-centroid and bead-covariance moments for one window."""

    def __init__(self, masses: Any, reference: Any, permutations: Sequence[Sequence[int]] | None = None):
        self.masses = np.asarray(masses, dtype=float)
        n = len(self.masses)
        if 3 * n > MAX_COVARIANCE_DIMENSION:
            raise ValueError(f"centroid path covariances are limited to {MAX_COVARIANCE_DIMENSION // 3} atoms")
        reference = np.asarray(reference, dtype=float).reshape(n, 3)
        self.reference = reference - np.sum(reference * self.masses[:, None], axis=0) / self.masses.sum()
        perms = [tuple(range(n))] if not permutations else [tuple(int(i) for i in p) for p in permutations]
        if any(sorted(p) != list(range(n)) for p in perms):
            raise ValueError("permutations must be permutations of the atom indices")
        self.permutations = perms
        self.sqrt_m = np.sqrt(self.masses)
        self.count = 0
        self.sum_xi = 0.0
        self.sum_xi2 = 0.0
        self.sum_centroid = np.zeros((n, 3))
        self.sum_centroid_outer = np.zeros((3 * n, 3 * n))
        self.sum_bead_covariance = np.zeros((3 * n, 3 * n))
        self.permutation_counts = np.zeros(len(perms), dtype=np.int64)
        self.pairs = np.triu_indices(n, 1)
        self.sum_distance = np.zeros(len(self.pairs[0]))
        self.restored_incomplete = False

    def add(self, q: Any, xi: float) -> None:
        q = np.asarray(q, dtype=float)
        centroid = q.mean(axis=0)
        best = None
        for index, perm in enumerate(self.permutations):
            relabelled = centroid[list(perm)]
            centre = np.sum(relabelled * self.masses[:, None], axis=0) / self.masses.sum()
            mobile = relabelled - centre
            rotation = kabsch_rotation(mobile, self.reference, self.masses)
            deviation = float(np.sum(self.masses[:, None] * (mobile @ rotation - self.reference) ** 2))
            if best is None or deviation < best[0]:
                best = (deviation, index, perm, centre, rotation)
        _, index, perm, centre, rotation = best
        beads = (q[:, list(perm)] - centre) @ rotation
        aligned = beads.mean(axis=0)
        weighted = (aligned * self.sqrt_m[:, None]).ravel()
        spread = ((beads - aligned) * self.sqrt_m[None, :, None]).reshape(len(q), -1)
        self.count += 1
        self.sum_xi += float(xi)
        self.sum_xi2 += float(xi) ** 2
        self.sum_centroid += aligned
        self.sum_centroid_outer += np.outer(weighted, weighted)
        self.sum_bead_covariance += spread.T @ spread / len(q)
        self.permutation_counts[index] += 1
        self.sum_distance += np.linalg.norm(aligned[self.pairs[0]] - aligned[self.pairs[1]], axis=1)

    def to_arrays(self) -> dict:
        return {PREFIX + name: np.asarray(getattr(self, name)) for name in _FIELDS}

    def restore(self, arrays: Mapping[str, np.ndarray]) -> None:
        if PREFIX + "sum_distance" not in arrays:
            # Checkpoints written before distance accumulation: keep their
            # moments, mark the distances (and so the window) incomplete.
            arrays = dict(arrays)
            arrays[PREFIX + "sum_distance"] = np.zeros_like(self.sum_distance)
            self.restored_incomplete = True
        for name in _FIELDS:
            value = np.asarray(arrays[PREFIX + name])
            current = np.asarray(getattr(self, name))
            if value.shape != current.shape:
                raise ValueError(f"centroid path checkpoint field {name} has the wrong shape")
            setattr(self, name, value.item() if value.ndim == 0 and name != "permutation_counts" else value.copy())
        self.count = int(self.count)


def combine_moments(moments: Sequence[Mapping[str, np.ndarray]]) -> dict:
    """Sum raw moments of replicas of one window."""
    total = {key: np.array(moments[0][key], copy=True) for key in moments[0]}
    for item in moments[1:]:
        for key in total:
            total[key] = total[key] + np.asarray(item[key])
    return total


def write_centroid_path(output, streams, *, masses, reference, permutations, beta, nbeads) -> Path | None:
    """Write ``centroid_path.npz`` from umbrella streams that carry moments."""
    carrying = [s for s in streams if s.get("centroid_moments") is not None]
    if not carrying:
        return None
    by_center: dict[float, list] = {}
    expected: dict[float, int] = {}
    for stream in carrying:
        center = float(np.asarray(stream["center"]).reshape(-1)[0])
        by_center.setdefault(center, []).append(stream["centroid_moments"])
        expected[center] = expected.get(center, 0) + int(len(stream["samples"]))
    centers = np.array(sorted(by_center))
    n = len(masses)
    count = np.zeros(len(centers), dtype=np.int64)
    xi_mean = np.full(len(centers), np.nan)
    xi_var = np.full(len(centers), np.nan)
    centroids = np.full((len(centers), n, 3), np.nan)
    centroid_cov = np.full((len(centers), 3 * n, 3 * n), np.nan)
    bead_cov = np.full((len(centers), 3 * n, 3 * n), np.nan)
    perm_counts = np.zeros((len(centers), len(permutations or [0])), dtype=np.int64)
    complete = np.zeros(len(centers), dtype=bool)
    distances = np.full((len(centers), n * (n - 1) // 2), np.nan)
    for i, center in enumerate(centers):
        total = combine_moments(by_center[center])
        c = int(total[PREFIX + "count"])
        count[i] = c
        perm_counts[i] = total[PREFIX + "permutation_counts"]
        complete[i] = c == expected[center]
        if c == 0:
            continue
        xi_mean[i] = float(total[PREFIX + "sum_xi"]) / c
        xi_var[i] = float(total[PREFIX + "sum_xi2"]) / c - xi_mean[i] ** 2
        centroids[i] = total[PREFIX + "sum_centroid"] / c
        weighted = (centroids[i] * np.sqrt(np.asarray(masses))[:, None]).ravel()
        centroid_cov[i] = total[PREFIX + "sum_centroid_outer"] / c - np.outer(weighted, weighted)
        bead_cov[i] = total[PREFIX + "sum_bead_covariance"] / c
        distances[i] = total[PREFIX + "sum_distance"] / c
    path = Path(output) / FILENAME
    metadata = {
        "schema": SCHEMA, "schema_version": 1, "beta": float(beta), "nbeads": int(nbeads),
        "alignment": "mass-weighted Kabsch onto the reference geometry after equivalent-path assignment",
        "coordinates": "bohr; covariances mass weighted (sqrt(m) x)",
        "complete_windows": int(complete.sum()), "windows": int(len(centers)),
    }
    from ..checkpoint import atomic_savez

    atomic_savez(path, window_center=centers, count=count, xi_mean=xi_mean, xi_variance=xi_var,
                 centroid=centroids, centroid_covariance=centroid_cov, bead_covariance=bead_cov,
                 permutation_counts=perm_counts, complete=complete, mean_distance=distances,
                 masses=np.asarray(masses, dtype=float), reference=np.asarray(reference, dtype=float),
                 metadata=json.dumps(metadata))
    return path


def _local_quadratic(parameter, values, bandwidth):
    """Value, first and second derivative of a Gaussian-weighted local quadratic fit."""
    smooth, first, second = (np.empty_like(values) for _ in range(3))
    for i, centre in enumerate(parameter):
        delta = parameter - centre
        weight = np.sqrt(np.exp(-0.5 * (delta / bandwidth) ** 2))
        design = np.column_stack([np.ones_like(delta), delta, delta * delta]) * weight[:, None]
        coefficients = np.linalg.lstsq(design, values * weight[:, None], rcond=None)[0]
        smooth[i], first[i], second[i] = coefficients[0], coefficients[1], 2.0 * coefficients[2]
    return smooth, first, second


@dataclass(frozen=True)
class RingPolymerMeanPath:
    """Per-window ring-polymer centroid statistics from a rate run."""

    xi: np.ndarray
    window_center: np.ndarray
    count: np.ndarray
    centroid: np.ndarray
    centroid_covariance: np.ndarray
    bead_covariance: np.ndarray
    complete: np.ndarray
    masses: np.ndarray
    reference: np.ndarray
    beta: float
    nbeads: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    mean_distance: np.ndarray | None = None

    @property
    def linear_axis(self) -> np.ndarray | None:
        """Unit axis of a linear reference geometry, else ``None``."""
        ref = np.asarray(self.reference, dtype=float)
        ref = ref - np.sum(ref * self.masses[:, None], axis=0) / self.masses.sum()
        values, vectors = np.linalg.eigh((ref * self.masses[:, None]).T @ ref)
        if values[-1] <= 0 or values[-2] > 1e-8 * values[-1]:
            return None
        return vectors[:, -1]

    @property
    def mean_centroid(self) -> np.ndarray:
        """Window-mean centroids; azimuthally averaged for a linear reference.

        Kabsch alignment onto a linear reference leaves the rotation about its
        axis undetermined, so aligned bent configurations do not average over
        the azimuth and the raw mean looks bent.  The exact azimuthal average
        of an axially symmetric ensemble lies on the axis, so the mean is
        projected onto it.  Covariances are kept as sampled.
        """
        axis = self.linear_axis
        if axis is None:
            return self.centroid
        if self.mean_distance is not None and np.isfinite(self.mean_distance).all():
            # Mean bond lengths along the axis: librating light atoms make
            # axial projections <r cos(theta)> shorter than <r>.
            n = len(self.masses)
            ref = np.asarray(self.reference, dtype=float)
            order = np.argsort(ref @ axis)
            lookup = {pair: k for k, pair in enumerate(zip(*np.triu_indices(n, 1)))}
            out = np.zeros_like(self.centroid)
            for w, row in enumerate(self.mean_distance):
                position = np.zeros(n)
                for a, b in zip(order[:-1], order[1:]):
                    position[b] = position[a] + row[lookup[(min(a, b), max(a, b))]]
                position -= np.sum(position * self.masses) / self.masses.sum()
                out[w] = position[:, None] * axis[None, :]
            return out
        return np.einsum("wni,i,j->wnj", self.centroid, axis, axis)

    @property
    def orientation_spread(self) -> np.ndarray:
        """RMS per-atom spread (bohr) of aligned centroids about their mean."""
        n = len(self.masses)
        out = np.full(len(self.xi), np.nan)
        for i, cov in enumerate(self.centroid_covariance):
            if not np.isfinite(cov).all():
                continue
            blocks = np.diagonal(cov).reshape(n, 3).sum(axis=1) / self.masses
            out[i] = float(np.sqrt(np.mean(np.maximum(blocks, 0.0))))
        return out

    def locked(self, *, spread: float | None = None, factor: float = 2.0) -> np.ndarray:
        """Contiguous windows around the highest-xi window whose spread is small.

        Towards the reactant asymptote the fragments rotate freely and the
        aligned centroids spread over several bohr.  A window is kept while
        its spread stays below ``spread`` bohr, by default ``factor`` times
        the median spread of the three highest-xi windows (thermal and
        zero-point motion near the transition state set the scale).
        """
        values = self.orientation_spread
        if spread is None:
            order = np.argsort(self.xi)
            top = values[order[-3:]]
            spread = factor * float(np.nanmedian(top))
        ok = (self.count > 0) & (values <= spread)
        order = np.argsort(self.xi)
        mask = np.zeros(len(self.xi), dtype=bool)
        anchor = order[-1]
        if not ok[anchor]:
            candidates = order[ok[order]]
            if not len(candidates):
                return mask
            anchor = candidates[-1]
        position = int(np.flatnonzero(order == anchor)[0])
        for step in (-1, 1):
            j = position
            while 0 <= j < len(order) and ok[order[j]]:
                mask[order[j]] = True
                j += step
        return mask

    def path(self, mask: np.ndarray | None = None) -> np.ndarray:
        """Mean centroid geometries ``[windows, atoms, 3]`` ordered by xi."""
        mask = self.locked() if mask is None else np.asarray(mask, dtype=bool)
        order = np.argsort(self.xi)
        order = order[mask[order]]
        return self.mean_centroid[order]

    def curvature_width(self, mask: np.ndarray | None = None, *, bandwidth: float | None = None,
                        threshold: float = 0.5) -> dict:
        """Curvature x width along the locked path and a 2D candidate direction.

        Works in mass-weighted coordinates.  Rigid-body directions are
        removed from tangents, normals and widths.  ``rho = sqrt(2 n^T Sigma
        n)`` uses the bead-about-centroid covariance along the curvature
        normal; the candidate direction is that normal at the largest
        ``kappa*rho``.  The widest transverse bead mode is reported as well.
        """
        mask = self.locked() if mask is None else np.asarray(mask, dtype=bool)
        order = np.argsort(self.xi)
        order = order[mask[order]]
        if len(order) < 5:
            raise ValueError("at least five locked windows are needed for curvature")
        sqrt_m = np.sqrt(self.masses)
        points = (self.mean_centroid[order] * sqrt_m[None, :, None]).reshape(len(order), -1)
        chord = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if bandwidth is None:
            bandwidth = 2.0 * float(np.median(np.diff(chord)))
        smooth, first, second = _local_quadratic(chord, points, bandwidth)
        kappa = np.zeros(len(order))
        rho = np.zeros(len(order))
        normals = np.zeros_like(points)
        widest = np.zeros_like(points)
        widest_width = np.zeros(len(order))
        for k, index in enumerate(order):
            rigid = rigid_body_modes(smooth[k].reshape(-1, 3) / sqrt_m[:, None], self.masses)
            project = np.eye(points.shape[1]) - rigid @ rigid.T
            tangent = project @ first[k]
            speed = np.linalg.norm(tangent)
            tangent = tangent / speed
            accel = project @ second[k]
            curvature = (accel - (accel @ tangent) * tangent) / speed**2
            kappa[k] = float(np.linalg.norm(curvature))
            normal = curvature / kappa[k] if kappa[k] > 0 else np.zeros_like(curvature)
            normals[k] = normal
            sigma = self.bead_covariance[index]
            rho[k] = float(np.sqrt(2.0 * max(normal @ sigma @ normal, 0.0)))
            transverse = project - np.outer(tangent, tangent)
            values, vectors = np.linalg.eigh(transverse @ sigma @ transverse)
            widest[k] = vectors[:, -1]
            widest_width[k] = float(np.sqrt(2.0 * max(values[-1], 0.0)))
        product = kappa * rho
        peak = int(np.argmax(product))
        candidate = normals[peak]
        overlap = float(abs(candidate @ widest[peak]))
        return {
            "xi": self.xi[order].tolist(),
            "arc_length": chord.tolist(),
            "curvature": kappa.tolist(),
            "width": rho.tolist(),
            "curvature_width": product.tolist(),
            "maximum": float(product[peak]),
            "maximum_xi": float(self.xi[order][peak]),
            "threshold": float(threshold),
            "recommended_dimension": 2 if product[peak] > threshold else 1,
            "candidate_direction": candidate.tolist(),
            "candidate_geometry": smooth[peak].tolist(),
            "widest_transverse_mode": widest[peak].tolist(),
            "widest_transverse_width": float(widest_width[peak]),
            "candidate_widest_overlap": overlap,
            "bandwidth": float(bandwidth),
            "provisional_threshold": True,
        }


def mean_path_from_rate_run(directory) -> RingPolymerMeanPath:
    """Read ``centroid_path.npz`` written by a rate run (or its ``run/`` subdirectory)."""
    root = Path(directory).resolve()
    for candidate in (root / FILENAME, root / "run" / FILENAME):
        if candidate.is_file():
            path = candidate
            break
    else:
        raise ValueError(f"No {FILENAME} in {root}; run the rate with output.centroid_path = true")
    from ..checkpoint import load_checkpoint

    with load_checkpoint(path) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata.get("schema") != SCHEMA:
            raise ValueError(f"{path} is not a {SCHEMA} file")
        return RingPolymerMeanPath(
            np.asarray(data["xi_mean"]), np.asarray(data["window_center"]), np.asarray(data["count"]),
            np.asarray(data["centroid"]), np.asarray(data["centroid_covariance"]),
            np.asarray(data["bead_covariance"]), np.asarray(data["complete"]),
            np.asarray(data["masses"]), np.asarray(data["reference"]),
            float(metadata["beta"]), int(metadata["nbeads"]), metadata,
            np.asarray(data["mean_distance"]) if "mean_distance" in data.files else None,
        )


__all__ = [
    "CentroidAccumulator", "RingPolymerMeanPath", "kabsch_rotation", "mean_path_from_rate_run",
    "rigid_body_modes", "write_centroid_path",
]
