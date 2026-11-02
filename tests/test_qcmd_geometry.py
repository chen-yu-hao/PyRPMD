"""Independent geometry and finite-difference references for quasi-centroids."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyrpmd.qcmd_geometry import TriatomicCoordinates


def ring(beads=9):
    x = np.zeros((beads, 3, 3))
    angle = np.linspace(1.7, 1.9, beads)
    r1 = np.linspace(1.6, 2.0, beads)
    r2 = np.linspace(1.9, 1.7, beads)
    x[:, 1, 0] = r1
    x[:, 2, 0] = r2 * np.cos(angle)
    x[:, 2, 1] = r2 * np.sin(angle)
    rotations = Rotation.from_rotvec(np.linspace(-.35, .35, beads)[:, None] * [1., .5, -.8]).as_matrix()
    return x @ rotations + np.linspace(-.2, .2, beads)[:, None, None] * [1., .2, -.3]


def test_internal_values_are_exact_and_invariant():
    geometry = TriatomicCoordinates([(0, 1, 2)], 3)
    q = ring()
    result = geometry.values(q)
    np.testing.assert_allclose(result[:, 0, 0], np.linspace(1.6, 2., 9), atol=4e-16)
    np.testing.assert_allclose(result[:, 0, 1], np.linspace(1.9, 1.7, 9), atol=4e-16)
    np.testing.assert_allclose(result[:, 0, 2], np.linspace(1.7, 1.9, 9), atol=4e-16)
    rotation = Rotation.from_rotvec([.4, -1., .7]).as_matrix()
    moved = q @ rotation + [4., -2., .3]
    np.testing.assert_allclose(geometry.values(moved), result, atol=9e-16)
    np.testing.assert_allclose(geometry.mean_values(q), [[1.8, 1.8, 1.8]], atol=4e-16)


def test_force_pullback_matches_finite_difference_and_has_no_net_force_or_torque():
    geometry = TriatomicCoordinates([(0, 1, 2)], 3)
    q = ring(4)
    rng = np.random.default_rng(9)
    linear = rng.normal(size=(4, 1, 3))
    curvature = np.array([.7, 1.2, .4])
    internal = geometry.values(q)
    derivative = linear + curvature * internal
    forces = geometry.forces(q, derivative)

    def energy(x):
        s = geometry.values(x)
        return np.sum(linear * s + .5 * curvature * s**2, axis=(1, 2))

    numeric = np.empty_like(q)
    eps = 2e-6
    for atom in range(3):
        for axis in range(3):
            plus, minus = q.copy(), q.copy()
            plus[:, atom, axis] += eps
            minus[:, atom, axis] -= eps
            numeric[:, atom, axis] = -(energy(plus) - energy(minus)) / (2 * eps)
    np.testing.assert_allclose(forces, numeric, atol=7e-10)
    np.testing.assert_allclose(forces.sum(axis=1), 0., atol=4e-16)
    np.testing.assert_allclose(np.cross(q, forces).sum(axis=1), 0., atol=2e-15)
    rotation = Rotation.from_rotvec([.6, -.3, .8]).as_matrix()
    np.testing.assert_allclose(geometry.forces(q @ rotation, derivative), forces @ rotation, atol=2e-15)


def test_reconstruction_obeys_mean_constraints_com_eckart_and_equivariance():
    geometry = TriatomicCoordinates([(0, 1, 2)], 3)
    q, masses = ring(12), np.array([15.9, 1., 2.])
    qc = geometry.quasicentroids(q, masses)
    np.testing.assert_allclose(geometry.values(qc[None])[0], geometry.mean_values(q), atol=9e-16)
    centroid = q.mean(axis=0)
    center = np.sum(masses[:, None] * centroid, axis=0) / masses.sum()
    np.testing.assert_allclose(np.sum(masses[:, None] * qc, axis=0) / masses.sum(), center, atol=2e-16)
    # Stationarity under infinitesimal rotations is the mass-weighted Eckart condition.
    np.testing.assert_allclose(np.sum(masses[:, None] * np.cross(qc - center, centroid - center), axis=0),
                               0., atol=1e-15)
    rotation = Rotation.from_rotvec([1.1, -.7, .3]).as_matrix()
    shift = np.array([3., 8., -1.])
    actual = geometry.quasicentroids(q @ rotation + shift, masses)
    np.testing.assert_allclose(actual, qc @ rotation + shift, atol=3e-15)
    # The quasi-centroid removes the artificial bond shortening from bead rotation.
    assert np.all(geometry.values(centroid[None])[0, 0, :2] < geometry.mean_values(q)[0, :2])


def test_reconstruction_is_the_centroid_in_the_one_bead_limit_and_batches_molecules():
    q = ring(1)
    frames = np.stack([q, q + [.1, -.2, .3]])
    doubled = np.concatenate([frames, frames + [3., 2., 1.], np.ones((2, 1, 1, 3))], axis=2)
    geometry = TriatomicCoordinates([(0, 1, 2), (3, 4, 5)], 7)
    masses = [16., 1., 1., 12., 2., 3., 4.]
    result = geometry.quasicentroids(doubled, masses)
    np.testing.assert_allclose(result, doubled[:, 0], atol=2e-15)
    assert geometry.mean_values(doubled).shape == (2, 2, 3)
    force = geometry.forces(doubled[:, 0], np.ones((2, 2, 3)))
    np.testing.assert_array_equal(force[:, 6], 0.)


@pytest.mark.parametrize("topology,natoms", [([], 3), ([[0., 1., 2.]], 3),
    ([[0, 1, 1]], 3), ([[0, 1, 3]], 3), ([[0, 1, 2], [2, 3, 4]], 5),
    ([[0, 1, 2]], True), ([[False, True, True]], 3)])
def test_invalid_topologies_fail(topology, natoms):
    with pytest.raises(ValueError):
        TriatomicCoordinates(topology, natoms)


def test_singular_geometries_and_periodic_cells_fail_explicitly():
    geometry = TriatomicCoordinates([(0, 1, 2)], 3)
    with pytest.raises(ValueError, match="nonzero"):
        geometry.values(np.zeros((1, 3, 3)))
    with pytest.raises(ValueError, match="noncollinear"):
        geometry.values(np.array([[[0., 0, 0], [1., 0, 0], [2., 0, 0]]]))
    with pytest.raises(ValueError, match="nonperiodic"):
        TriatomicCoordinates([(0, 1, 2)], 3, pbc=True)
    with pytest.raises(ValueError, match="nonperiodic"):
        TriatomicCoordinates([(0, 1, 2)], 3, cell=np.eye(3))
    q = ring(1)
    q = np.concatenate([q, -q], axis=0)
    with pytest.raises(ValueError, match="degenerate"):
        geometry.quasicentroids(q, [16., 1., 1.])


@pytest.mark.parametrize("method,args", [
    ("values", (np.zeros((2, 4, 3)),)),
    ("mean_values", (np.zeros((0, 3, 3)),)),
    ("quasicentroids", (ring(), [0., 1., 1.])),
    ("forces", (ring(), np.zeros((9, 3)))),
])
def test_invalid_geometry_shapes_and_masses_fail(method, args):
    with pytest.raises(ValueError):
        getattr(TriatomicCoordinates([(0, 1, 2)], 3), method)(*args)
