"""Nearest-image geometry must be independent of an integer lattice basis."""

import itertools

import numpy as np
import pytest

from pyrpmd.cv import PeriodicDistance
from pyrpmd.geometry import FixedCellGeometry, minimum_image


def equivalent_cell(cell, flags):
    """Change only the periodic basis by a unimodular integer matrix."""
    indices = np.flatnonzero(flags)
    transformed = cell.copy()
    if len(indices) == 2:
        transform = np.array([[1000, 1], [1, 0]])
    elif len(indices) == 3:
        transform = np.array([[1000, 1, 0], [1, 0, 0], [500, -300, 1]])
    else:
        return transformed
    transformed[indices] = transform @ cell[indices]
    assert abs(round(np.linalg.det(transform))) == 1
    return transformed


@pytest.mark.parametrize("flags", list(itertools.product((False, True), repeat=3)))
def test_integer_shear_preserves_images_for_every_periodic_subspace(flags):
    # These cells represent exactly the same physical lattice. In the original
    # QR search, a long first vector exhausted the search even for this simple
    # orthorhombic lattice; swapping rows could change failure into success.
    original = np.diag([3., 4., 5.])
    transformed = equivalent_cell(original, flags)
    vectors = np.random.default_rng(946).uniform(-12, 12, (41, 3))
    expected = vectors.copy()
    expected[:, flags] -= (
        np.rint(vectors[:, flags] / np.diag(original)[list(flags)])
        * np.diag(original)[list(flags)]
    )
    actual = minimum_image(vectors, transformed, flags)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
    # Reduction must not add artificial periodicity in the slab/wire vacuum.
    np.testing.assert_array_equal(actual[:, np.logical_not(flags)], vectors[:, np.logical_not(flags)])
    geometry = FixedCellGeometry(transformed, flags)
    np.testing.assert_array_equal(geometry.cell, transformed)
    np.testing.assert_array_equal(geometry.pbc, flags)


@pytest.mark.parametrize("flags", [(True, True, False), (True, False, True), (False, True, True), (True, True, True)])
def test_reduced_triclinic_images_match_independent_exhaustive_lattice(flags):
    original = np.array([[4., 0., 0.], [1.2, 3.1, 0.], [.3, -.8, 5.]])
    transformed = equivalent_cell(original, flags)
    vectors = np.random.default_rng(1003).uniform(-4, 4, (29, 3))
    integers = np.array(list(itertools.product(*[
        range(-4, 5) if flag else (0,) for flag in flags
    ])))
    candidates = vectors[:, None] - (integers @ original)[None]
    squared = np.sum(candidates ** 2, axis=-1)
    expected = candidates[np.arange(len(vectors)), squared.argmin(axis=1)]
    actual = minimum_image(vectors, transformed, flags)
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=0)


def test_distance_and_force_direction_do_not_depend_on_surface_supercell_basis():
    original = np.array([[4., 0., 0.], [1.5, 3., 0.], [0., 0., 15.]])
    flags = (True, True, False)
    transformed = equivalent_cell(original, flags)
    q = np.array([[[.2, .3, .4], [3.9, .8, 1.2]],
                  [[.3, .2, .5], [4.0, .7, 1.3]],
                  [[.4, .4, .6], [4.1, .9, 1.4]]])
    for estimator in ("centroid", "bead_average"):
        reference = PeriodicDistance([0, 1], original, flags, estimator=estimator)
        changed = PeriodicDistance([0, 1], transformed, flags, estimator=estimator)
        expected = reference.value_and_jacobian(q)
        actual = changed.value_and_jacobian(q)
        np.testing.assert_allclose(actual.value, expected.value, atol=2e-12, rtol=0)
        np.testing.assert_allclose(actual.jacobian, expected.jacobian, atol=2e-12, rtol=0)
        np.testing.assert_allclose(changed.value(q + original[0]), actual.value, atol=2e-12, rtol=0)


def test_triclinic_partial_images_agree_with_ase_reference():
    ase_geometry = pytest.importorskip("ase.geometry")
    rng = np.random.default_rng(512)
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    original = np.array([[4., 0., 0.], [1.2, 3.1, 0.], [.3, -.8, 5.]]) @ rotation
    vectors = rng.uniform(-5, 5, (53, 3))
    for flags in itertools.product((False, True), repeat=3):
        # ASE's independent Minkowski-reduction implementation is an oracle;
        # no ASE code or runtime dependency is introduced into PyRPMD.
        expected, _ = ase_geometry.find_mic(vectors, original, pbc=flags)
        transformed = equivalent_cell(original, flags)
        actual = minimum_image(vectors, transformed, flags)
        np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=0)
