"""Independent geometry and derivative checks for the quasi-centroid layer."""

import numpy as np
import pytest

from pyrpmd._qcmd_geometry import QuasiCentroidConstraint, TriatomicQuasiCentroid


def molecule(r1=1.8, r2=1.9, angle=1.85):
    return np.array([[0., 0., 0.], [r1, 0., 0.],
                     [r2 * np.cos(angle), r2 * np.sin(angle), 0.]])


def ring(beads=7):
    rng = np.random.default_rng(107)
    return molecule()[None] + rng.normal(scale=.08, size=(beads, 3, 3))


def test_quasi_centroid_matches_mean_internals_and_com_not_cartesian_centroid():
    geometry = TriatomicQuasiCentroid([[0, 1, 2]], [16., 1., 2.])
    q = np.stack((ring(), ring() + .2))
    qc = geometry.quasi_centroid(q)
    expected = []
    for frame in q:
        a, b = frame[:, 1] - frame[:, 0], frame[:, 2] - frame[:, 0]
        theta = np.arccos(np.sum(a * b, axis=1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1))
        expected.append([np.linalg.norm(a, axis=1).mean(), np.linalg.norm(b, axis=1).mean(), theta.mean()])
    np.testing.assert_allclose(geometry.internal_coordinates(qc)[:, 0], expected, atol=2e-15)
    np.testing.assert_allclose((qc * geometry.masses[None, :, None]).sum(axis=1),
                               (q.mean(axis=1) * geometry.masses[None, :, None]).sum(axis=1), atol=2e-14)
    assert np.max(np.abs(qc - q.mean(axis=1))) > 1e-4
    # Eckart stationarity of the mass-weighted best-fit orientation.
    center = (qc * geometry.masses[None, :, None]).sum(axis=1) / geometry.masses.sum()
    torque = np.sum(geometry.masses[None, :, None] *
                    np.cross(qc - center[:, None], q.mean(axis=1) - center[:, None]), axis=1)
    np.testing.assert_allclose(torque, 0, atol=3e-14)


def test_quasi_centroid_is_rotation_translation_covariant_and_p1_exact():
    geom = TriatomicQuasiCentroid([[0, 1, 2]], [16., 1., 1.])
    q = ring()
    rotation, _ = np.linalg.qr(np.random.default_rng(139).normal(size=(3, 3)))
    rotation[:, -1] *= np.linalg.det(rotation)
    shift = np.array([4., -2., .7])
    np.testing.assert_allclose(geom.quasi_centroid(q @ rotation + shift),
                               geom.quasi_centroid(q) @ rotation + shift, atol=1e-14)
    np.testing.assert_allclose(geom.mean_coordinates(q @ rotation + shift), geom.mean_coordinates(q), atol=2e-15)
    np.testing.assert_allclose(geom.quasi_centroid(q[:1]), q[0], atol=3e-15)


def test_multi_molecule_nontrivial_atom_order_and_independent_transforms():
    geometry = TriatomicQuasiCentroid([[3, 1, 5], [2, 4, 0]], [1., 1., 16., 16., 1., 1.])
    q = np.empty((7, 6, 3))
    q[:, [3, 1, 5]] = ring()
    q[:, [2, 4, 0]] = ring() * 1.2 + [4., 1., -.2]
    qc = geometry.quasi_centroid(q)
    np.testing.assert_allclose(geometry.internal_coordinates(qc), geometry.mean_coordinates(q), atol=3e-15)


def test_internal_gradients_match_independent_cartesian_finite_difference():
    geom = TriatomicQuasiCentroid([[2, 0, 1]], [1., 1., 16.])
    q = ring()[:, [1, 2, 0]]
    derivative = geom.internal_gradients(q)
    epsilon = 1e-6
    for local, atom in enumerate(geom.indices[0]):
        for axis in range(3):
            plus, minus = q.copy(), q.copy()
            plus[:, atom, axis] += epsilon
            minus[:, atom, axis] -= epsilon
            expected = (geom.internal_coordinates(plus) - geom.internal_coordinates(minus)) / (2 * epsilon)
            np.testing.assert_allclose(derivative[..., local, axis], expected, atol=4e-10)
    np.testing.assert_allclose(derivative.sum(axis=-2), 0, atol=1e-16)


def test_constraint_projection_and_momentum_orthogonality():
    geom = TriatomicQuasiCentroid([[0, 1, 2]], [16., 1., 2.])
    target = np.array([[1.82, 1.91, 1.86]])
    constraint = QuasiCentroidConstraint(geom, target, tolerance=1e-12)
    q = ring()
    before = q.copy()
    projected = constraint.project_positions(q)
    np.testing.assert_array_equal(q, before)
    np.testing.assert_allclose(geom.mean_coordinates(projected), target, atol=1e-12)
    np.testing.assert_allclose((projected * geom.masses[None, :, None]).sum(axis=1),
                               (q * geom.masses[None, :, None]).sum(axis=1), atol=2e-15)
    p = np.random.default_rng(287).normal(size=q.shape)
    tangent = constraint.project_momenta(projected, p)
    np.testing.assert_allclose(constraint.velocity_residual(projected, tangent), 0, atol=1e-15)
    np.testing.assert_allclose(tangent.sum(axis=1), p.sum(axis=1), atol=1e-15)
    np.testing.assert_allclose(constraint.project_momenta(projected, tangent), tangent, atol=5e-16)
    jacobian = constraint.local_jacobian(projected)[0].reshape(3, -1)
    dense_inverse_mass = np.tile(np.repeat(1 / geom.masses, 3), len(q))
    independent = (jacobian * dense_inverse_mass) @ jacobian.T
    np.testing.assert_allclose(constraint.metric(projected)[0], independent, atol=5e-17)


@pytest.mark.parametrize("triples,masses", [
    ([[0, 1, 1]], [16., 1., 1.]), ([[0, 1, 2]], [16., 1., 1., 4.]),
    ([[0., 1., 2.]], [16., 1., 1.]), ([[0, 1, 3]], [16., 1., 1.]),
    ([[0, 1, 2]], [16., 0., 1.]), ([[True, False, True]], [16., 1., 1.]),
])
def test_geometry_rejects_invalid_topologies(triples, masses):
    with pytest.raises(ValueError):
        TriatomicQuasiCentroid(triples, masses)


def test_geometry_rejects_singular_molecules_and_undefined_centroid_orientation():
    geom = TriatomicQuasiCentroid([[0, 1, 2]], [16., 1., 1.])
    for q in (np.zeros((1, 3, 3)), np.arange(9).reshape(1, 3, 3)):
        with pytest.raises(ValueError, match="singular|non-collinear"):
            geom.internal_coordinates(q)
    q = molecule()
    with pytest.raises(ValueError, match="degenerate molecular orientation"):
        geom.quasi_centroid(np.stack((q, -q)))
    with pytest.raises(ValueError, match="did not converge"):
        QuasiCentroidConstraint(geom, [[2., 2.2, 1.3]], max_iterations=1).project_positions(ring())


def test_geometry_topology_and_singularity_rules_are_immutable():
    geom = TriatomicQuasiCentroid([[0, 1, 2]], [16., 1., 1.])
    np.testing.assert_array_equal(geom.molecules, [[0, 1, 2]])
    for field, value in (("molecules", [[1, 0, 2]]), ("singularity_tolerance", 1e-5),
                         ("masses", np.ones(3))):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(geom, field, value)
    with pytest.raises(ValueError, match="read-only"):
        geom.molecules[0, 0] = 1
