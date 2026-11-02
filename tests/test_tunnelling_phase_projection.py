"""Independent spectral checks for the splitting phase-coordinate removal."""

import numpy as np
import pytest

from pyrpmd.tunnelling import _phase_projected_spectrum


@pytest.mark.parametrize("size", [2, 3, 8, 31, 128])
@pytest.mark.parametrize("layout", ["C", "F", "view"])
def test_phase_projection_matches_explicit_qr_without_mutating_inputs(size, layout):
    rng = np.random.default_rng(1803 + size)
    matrix = rng.normal(size=(size, size))
    matrix += matrix.T
    if layout == "F":
        matrix = np.asfortranarray(matrix)
    elif layout == "view":
        storage = np.zeros((2 * size, 2 * size))
        storage[::2, ::2] = matrix
        matrix = storage[::2, ::2]
    tangent = rng.normal(size=size)
    original_matrix, original_tangent = matrix.copy(), tangent.copy()
    matrix.flags.writeable = False
    tangent.flags.writeable = False

    complement = np.linalg.qr(tangent[:, None], mode="complete")[0][:, 1:]
    reference = np.linalg.eigvalsh(complement.T @ matrix @ complement)
    actual = _phase_projected_spectrum(matrix, tangent)

    np.testing.assert_allclose(actual, reference, rtol=3e-13, atol=3e-13)
    np.testing.assert_array_equal(matrix, original_matrix)
    np.testing.assert_array_equal(tangent, original_tangent)


@pytest.mark.parametrize("direction", [1., -1.])
@pytest.mark.parametrize("offset", [0., 1e-16, 1e-8])
def test_axis_aligned_phase_is_removed_even_if_it_is_not_a_zero_eigenmode(direction, offset):
    # A small negative mode is not a valid proxy for the geometric tangent.
    # The exact tangent selects the large positive eigenvalue in the axis case.
    matrix = np.diag([30., -1e-12, -2., 5., 8.])
    tangent = np.array([direction, offset, -offset, offset, 0.])
    complement = np.linalg.qr(tangent[:, None], mode="complete")[0][:, 1:]
    reference = np.linalg.eigvalsh(complement.T @ matrix @ complement)
    actual = _phase_projected_spectrum(matrix, tangent)

    np.testing.assert_allclose(actual, reference, rtol=3e-14, atol=3e-14)
    if offset == 0:
        np.testing.assert_array_equal(actual, np.array([-2., -1e-12, 5., 8.]))


def test_phase_projection_is_invariant_to_tangent_orientation_and_scale():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(24, 24))
    matrix += matrix.T
    tangent = rng.normal(size=24)
    reference = _phase_projected_spectrum(matrix, tangent)
    for factor in (-1., 1e-100, -1e100):
        np.testing.assert_allclose(_phase_projected_spectrum(matrix, factor * tangent),
                                   reference, rtol=1e-13, atol=1e-13)


def test_phase_projection_keeps_degenerate_modes():
    # For H = a*I + b*t*t.T the complementary eigenvalues are exactly a,
    # independently of the tangent orientation and removed eigenvalue.
    tangent = np.linspace(-2., 3., 19)
    tangent /= np.linalg.norm(tangent)
    matrix = 2.75 * np.eye(len(tangent)) - 10. * np.outer(tangent, tangent)
    np.testing.assert_allclose(_phase_projected_spectrum(matrix, tangent),
                               np.full(len(tangent) - 1, 2.75), rtol=2e-15)
