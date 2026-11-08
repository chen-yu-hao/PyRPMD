"""Ring-polymer centroid moments, mean paths and curvature x width."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyrpmd.analysis.centroid_path import (
    CentroidAccumulator,
    combine_moments,
    kabsch_rotation,
    mean_path_from_rate_run,
    rigid_body_modes,
    write_centroid_path,
)

MASSES = np.array([12.0, 1.0, 16.0]) * 1822.888


def bent():
    return np.array([[0.0, 0.0, 0.0], [1.1, 1.6, 0.0], [2.4, 0.0, 0.0]])


def centred(x, m=MASSES):
    return x - np.sum(x * m[:, None], axis=0) / m.sum()


def test_kabsch_recovers_rotation_and_rigid_modes_count():
    x = centred(bent())
    rotation = Rotation.from_euler("xyz", [0.3, -1.1, 2.0]).as_matrix()
    r = kabsch_rotation(x @ rotation, x, MASSES)
    assert np.allclose(x @ rotation @ r, x, atol=1e-12)
    assert rigid_body_modes(bent(), MASSES).shape == (9, 6)
    linear = np.array([[-2.0, 0, 0], [0.0, 0, 0], [2.5, 0, 0]])
    modes = rigid_body_modes(linear, MASSES)
    assert modes.shape == (9, 5)
    assert np.allclose(modes.T @ modes, np.eye(5), atol=1e-12)


def _ring(rng, geometry, beads=4, noise=0.05, perm=None):
    q = np.repeat(geometry[None], beads, axis=0) + noise * rng.normal(size=(beads, 3, 3))
    q -= q.mean(axis=0) - geometry  # centroid exactly at the geometry
    rotation = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
    q = q @ rotation + rng.normal(size=3)
    if perm is not None:
        q = q[:, perm]
    return q


def test_accumulator_aligns_and_assigns_equivalent_paths():
    rng = np.random.default_rng(1)
    masses = np.array([1.0, 12.0, 1.0]) * 1822.888
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 1.4, 0.0], [2.6, 0.2, 0.0]])
    swap = (2, 1, 0)
    acc = CentroidAccumulator(masses, reference, [(0, 1, 2), swap])
    for k in range(200):
        acc.add(_ring(rng, reference, perm=list(swap) if k % 2 else None), xi=0.5 + 0.01 * k)
    mean = acc.sum_centroid / acc.count
    assert np.allclose(mean, centred(reference, masses), atol=0.02)
    assert acc.permutation_counts.tolist() == [100, 100]
    distance = acc.sum_distance / acc.count
    assert np.allclose(distance, [np.linalg.norm(reference[i] - reference[j]) for i, j in ((0, 1), (0, 2), (1, 2))],
                       atol=0.02)


def test_accumulator_checkpoint_round_trip_and_combination():
    rng = np.random.default_rng(2)
    acc = CentroidAccumulator(MASSES, bent())
    for _ in range(5):
        acc.add(_ring(rng, bent()), 0.9)
    copy = CentroidAccumulator(MASSES, bent())
    copy.restore(acc.to_arrays())
    for name, value in acc.to_arrays().items():
        assert np.array_equal(copy.to_arrays()[name], value)
    total = combine_moments([acc.to_arrays(), acc.to_arrays()])
    assert int(total["centroid_path_count"]) == 10
    legacy = {k: v for k, v in acc.to_arrays().items() if k != "centroid_path_sum_distance"}
    old = CentroidAccumulator(MASSES, bent())
    old.restore(legacy)
    assert old.restored_incomplete


def _streams(rng, xis, geometry_of):
    streams = []
    for xi in xis:
        acc = CentroidAccumulator(MASSES, geometry_of(1.0))
        geometry = geometry_of(xi)
        for _ in range(30):
            acc.add(_ring(rng, geometry, noise=0.02), xi)
        streams.append({"center": np.array([xi]), "samples": np.zeros((30, 1)), "centroid_moments": acc.to_arrays()})
    return streams


def test_written_path_round_trip_and_linear_reference_uses_mean_distances(tmp_path):
    rng = np.random.default_rng(3)

    def linear(xi):
        return np.array([[-2.0 - 0.5 * (1 - xi), 0.0, 0.0], [0.0, 0.0, 0.0], [2.5 + 2.0 * (1 - xi), 0.0, 0.0]])

    xis = np.linspace(0.6, 1.2, 7)
    streams = _streams(rng, xis, linear)
    streams[3]["samples"] = np.zeros((31, 1))  # one window has a missing sample
    path = write_centroid_path(tmp_path, streams, masses=MASSES, reference=linear(1.0),
                               permutations=[(0, 1, 2)], beta=1000.0, nbeads=4)
    assert path.name == "centroid_path.npz"
    mean = mean_path_from_rate_run(tmp_path)
    assert np.allclose(mean.xi, xis)
    assert mean.complete.tolist() == [True, True, True, False, True, True, True]
    assert mean.linear_axis is not None
    geometry = mean.path(np.ones(7, dtype=bool))
    for g, xi in zip(geometry, xis):
        assert np.allclose(np.abs(g[:, 1:]), 0.0, atol=1e-12)
        target = linear(xi)
        assert np.linalg.norm(g[1] - g[0]) == pytest.approx(np.linalg.norm(target[1] - target[0]), abs=0.01)
    assert mean.metadata["schema"] == "pyrpmd.centroid_path"


def test_locked_windows_exclude_freely_rotating_reactants():
    rng = np.random.default_rng(4)
    reference = bent()
    acc_far = CentroidAccumulator(MASSES, reference)
    for _ in range(100):
        g = reference.copy()
        direction = Rotation.random(random_state=int(rng.integers(1 << 30))).apply([6.0, 0.0, 0.0])
        g[2] = g[1] + direction  # third atom rotates freely around the others
        acc_far.add(_ring(rng, g, noise=0.01), 0.1)
    acc_near = CentroidAccumulator(MASSES, reference)
    for _ in range(100):
        acc_near.add(_ring(rng, reference, noise=0.05), 1.0)
    streams = [{"center": np.array([x]), "samples": np.zeros((100, 1)), "centroid_moments": a.to_arrays()}
               for x, a in ((0.1, acc_far), (0.95, acc_near), (1.0, acc_near))]
    tmp = __import__("tempfile").mkdtemp()
    write_centroid_path(tmp, streams, masses=MASSES, reference=reference, permutations=None, beta=1.0, nbeads=4)
    mean = mean_path_from_rate_run(tmp)
    assert mean.locked().tolist() == [False, True, True]


def test_curvature_width_on_a_circular_path():
    from pyrpmd.analysis.centroid_path import RingPolymerMeanPath

    masses = np.full(3, 1822.888)
    radius_mw = 400.0
    sigma2 = 4.0
    angles = np.linspace(0.0, 1.2, 25)
    centroids = np.zeros((len(angles), 3, 3))
    # Atom 0 moves on a circle (mass-weighted radius 400); atoms 1, 2 balance it.
    centroids[:, 0, 0] = radius_mw * np.cos(angles) / np.sqrt(masses[0])
    centroids[:, 0, 1] = radius_mw * np.sin(angles) / np.sqrt(masses[0])
    centroids[:, 1, 2] = 5.0
    centroids[:, 2, 2] = -5.0
    centroids -= np.sum(centroids * masses[None, :, None], axis=1, keepdims=True) / masses.sum()
    bead = np.repeat((sigma2 * np.eye(9))[None], len(angles), axis=0)
    path = RingPolymerMeanPath(angles, angles, np.full(len(angles), 10), centroids, np.zeros_like(bead), bead,
                               np.ones(len(angles), dtype=bool), masses, centroids[-1], 1.0, 4, {})
    result = path.curvature_width(np.ones(len(angles), dtype=bool))
    middle = len(angles) // 2
    # The rigid-body projection removes the overall rotation component, so the
    # internal curvature differs from 1/R; it must still be finite and
    # rho = sqrt(2 sigma^2) for an isotropic bead covariance.
    assert result["width"][middle] == pytest.approx(np.sqrt(2 * sigma2), rel=1e-6)
    assert result["curvature"][middle] > 0
    assert result["recommended_dimension"] in (1, 2)
    assert len(result["candidate_direction"]) == 9
