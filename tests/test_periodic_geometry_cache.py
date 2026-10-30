"""Fixed-cell cache, shared CV derivatives and allocation regressions."""

import copy
from dataclasses import FrozenInstanceError
import itertools
import pickle

import numpy as np
import pytest

from pyrpmd.cv import PeriodicDistance, SurfaceHeight, SurfaceLateral, make_cv
from pyrpmd.cv_periodic import (
    PeriodicDistance as LegacyPeriodicDistance,
    PeriodicAngle, PeriodicBondDifference, LinearProjection, FractionalCoordinate,
)
from pyrpmd.geometry import FixedCellGeometry, minimum_image
from pyrpmd.system import System


def skew_cell():
    return np.array([[4.0, 0.0, 0.0], [3.2, 1.1, 0.0], [0.4, -0.3, 5.0]])


@pytest.mark.parametrize("flags", list(itertools.product((False, True), repeat=3)))
def test_prepared_images_match_exhaustive_partial_pbc_search(flags):
    cell = skew_cell()
    vectors = np.random.default_rng(621).uniform(-2, 2, (13, 3))
    prepared = FixedCellGeometry(cell, flags)
    integers = np.array(list(itertools.product(*[
        range(-8, 9) if flag else (0,) for flag in flags
    ])))
    candidates = vectors[:, None] - (integers @ cell)[None]
    distances = np.sum(candidates * candidates, axis=-1)
    expected = candidates[np.arange(len(vectors)), distances.argmin(axis=1)]
    actual = prepared.minimum_image(vectors)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_array_equal(actual, minimum_image(vectors, cell, flags))
    translated = vectors + (np.array([301, -207, 49]) * flags) @ cell
    np.testing.assert_allclose(prepared.minimum_image(translated), actual, atol=3e-13)


@pytest.mark.parametrize("cell", [None, np.diag([4.0, 5.0, 6.0]), skew_cell()])
def test_geometry_snapshots_and_process_roundtrips_are_immutable(cell):
    flags = np.array([False, False, False] if cell is None else [True, True, False])
    source = None if cell is None else cell.copy()
    prepared = FixedCellGeometry(source, flags)
    vectors = np.array([[6.3, -4.0, 12.0], [-2.7, 1.6, 4.0]])
    expected = prepared.minimum_image(vectors)
    flags[:] = ~flags
    if source is not None:
        source[:] *= 7
    for geometry in (prepared, copy.deepcopy(prepared), pickle.loads(pickle.dumps(prepared))):
        np.testing.assert_array_equal(geometry.minimum_image(vectors), expected)
        for value in (
            geometry.cell, geometry.pbc, geometry.basis, geometry.squared_lengths,
            geometry.orthogonal, geometry.triangular,
        ):
            if value is not None:
                assert not value.flags.writeable
                with pytest.raises(ValueError):
                    value.setflags(write=True)
        with pytest.raises(FrozenInstanceError):
            geometry.cell = np.eye(3)


def test_periodic_cv_reuses_decomposition_and_rebinds_without_stale_cache(monkeypatch):
    cell = skew_cell()
    flags = [True, True, False]
    q = np.array([[0.2, 0.3, 0.4], [3.7, 0.2, 0.1], [0.5, 1.3, 0.6]])
    cvs = (
        PeriodicDistance([0, 1], cell, flags),
        PeriodicBondDifference(atoms=[1, 0, 2], cell=cell, pbc=flags),
        PeriodicAngle([1, 0, 2], cell, flags),
    )
    expected = [cv.centroid_value_gradient(q) for cv in cvs]

    def unexpected(*args, **kwargs):
        raise AssertionError("fixed-cell decomposition was rebuilt during CV evaluation")

    with monkeypatch.context() as guard:
        guard.setattr(np.linalg, "svd", unexpected)
        guard.setattr(np.linalg, "qr", unexpected)
        for cv, (value, gradient) in zip(cvs, expected):
            for _ in range(3):
                actual, derivative = cv.centroid_value_gradient(q)
                assert actual == value
                np.testing.assert_array_equal(derivative, gradient)

    source = np.eye(3) * 4
    cv = PeriodicDistance([0, 1], source)
    pair = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
    assert cv.centroid_value_gradient(pair)[0] == pytest.approx(0.2)
    source[:] *= 2
    assert cv.centroid_value_gradient(pair)[0] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        cv.cell[0, 0] = 8
    cv.cell = source
    assert cv.centroid_value_gradient(pair)[0] == pytest.approx(3.8)
    cv.cell = np.eye(3) * 4
    cv.pbc = (False, True, True)
    assert cv.centroid_value_gradient(pair)[0] == pytest.approx(3.8)
    with pytest.raises(ValueError):
        cv.pbc = (1, 1, 1)
    assert cv.pbc == (False, True, True)
    assert cv.centroid_value_gradient(pair)[0] == pytest.approx(3.8)


@pytest.mark.parametrize("kind", ["periodic_distance", "periodic_bond_difference", "periodic_angle"])
def test_public_periodic_cvs_keep_readonly_caches_after_copy_and_pickle(kind):
    q = np.array([[0.2, 0.3, 0.4], [3.7, 0.2, 0.1], [0.5, 1.3, 0.6]])
    system = System(("H",) * 3, np.ones(3), q, cell=skew_cell(), pbc=(True, True, False))
    atoms = [0, 1] if kind == "periodic_distance" else [1, 0, 2]
    original = make_cv({"kind": kind, "atoms": atoms}, system)
    expected = original.value_and_jacobian(q[None])
    for restored in (copy.deepcopy(original), pickle.loads(pickle.dumps(original))):
        assert type(restored) is type(original)
        assert isinstance(restored.pbc, tuple if kind == "periodic_distance" else np.ndarray)
        result = restored.value_and_jacobian(q[None])
        np.testing.assert_array_equal(result.value, expected.value)
        np.testing.assert_array_equal(result.jacobian, expected.jacobian)
        with pytest.raises(ValueError):
            restored.cell.setflags(write=True)
        with pytest.raises(ValueError):
            restored._geometry.pbc.setflags(write=True)
        restored.cell = np.eye(3) * 20
        np.testing.assert_array_equal(original.cell, skew_cell())
        np.testing.assert_array_equal(original.value(q[None]), expected.value)


@pytest.mark.parametrize("estimator", ["centroid", "bead_average"])
def test_periodic_distance_factory_and_legacy_alias_do_not_allocate_hessian(monkeypatch, estimator):
    assert LegacyPeriodicDistance is PeriodicDistance
    natoms = 1000
    q = np.zeros((3, natoms, 3))
    q[:, 1, 0] = [3.7, 3.8, 3.9]
    system = System(("H",) * natoms, np.ones(natoms), q[0], cell=np.eye(3) * 4)
    cv = make_cv({"kind": "periodic_distance", "atoms": [0, 1], "estimator": estimator}, system)
    assert type(cv) is PeriodicDistance
    zeros = np.zeros

    def reject_dense_hessian(shape, *args, **kwargs):
        if np.ndim(shape) == 1 and len(shape) == 4:
            raise AssertionError("ordinary periodic-distance evaluation allocated a Hessian")
        return zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", reject_dense_hessian)
    result = cv.value_and_jacobian(q)
    np.testing.assert_allclose(result.value, [0.2], atol=3e-16)
    expected = np.zeros_like(result.jacobian)
    expected[0, :, 0, 0] = 1 / 3
    expected[0, :, 1, 0] = -1 / 3
    np.testing.assert_array_equal(result.jacobian, expected)


def test_explicit_periodic_distance_hessian_matches_gradient_derivative():
    cv = PeriodicDistance([0, 2], skew_cell(), (True, True, False))
    q = np.array([[0.2, 0.3, 0.4], [3.7, 0.2, 0.1], [0.5, 1.3, 0.6]])
    value, gradient, hessian = cv.centroid_value_gradient_hessian(q)
    ordinary, ordinary_gradient = cv.centroid_value_gradient(q)
    assert ordinary == value
    np.testing.assert_array_equal(ordinary_gradient, gradient)
    numerical = np.empty_like(hessian)
    for atom, axis in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[atom, axis] += 1e-6
        minus[atom, axis] -= 1e-6
        numerical[:, :, atom, axis] = (
            cv.centroid_value_gradient(plus)[1] - cv.centroid_value_gradient(minus)[1]
        ) / 2e-6
    np.testing.assert_allclose(hessian, numerical, rtol=2e-8, atol=2e-9)


def test_periodic_distance_preserves_hessian_override_extension_point():
    class ModifiedDistance(PeriodicDistance):
        def centroid_value_gradient_hessian(self, q):
            value, gradient, hessian = super().centroid_value_gradient_hessian(q)
            return 2 * value, 2 * gradient, 2 * hessian

    q = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
    modified = ModifiedDistance([0, 1], np.eye(3) * 4)
    baseline = PeriodicDistance([0, 1], np.eye(3) * 4)
    expected, derivative = baseline.centroid_value_gradient(q)
    assert modified.centroid_value_gradient(q)[0] == pytest.approx(2 * expected)
    np.testing.assert_array_equal(modified.centroid_value_gradient(q)[1], 2 * derivative)
    baseline.centroid_value_gradient_hessian = modified.centroid_value_gradient_hessian
    assert baseline.centroid_value_gradient(q)[0] == pytest.approx(2 * expected)


@pytest.mark.parametrize("estimator", ["centroid", "bead_average"])
def test_shared_affine_evaluation_preserves_projection_units_and_derivatives(estimator):
    masses = np.array([2.0, 3.0, 5.0])
    direction = np.array([1.0, -0.4, 0.2])
    options = dict(origin=0.7, reference_atoms=[1, 2], estimator=estimator)
    q = np.random.default_rng(71).normal(size=(4, 3, 3))
    cvs = (
        SurfaceHeight([0], masses, normal=direction, **options),
        SurfaceLateral([0], masses, direction=direction, **options),
        LinearProjection(0, direction, masses, **options),
    )
    weights = np.array([1.0, -3 / 8, -5 / 8])
    gradient = weights[:, None] * (direction / np.linalg.norm(direction))[None]
    expected = float(np.sum(gradient * q.mean(axis=0)) - options["origin"])
    jacobian = np.broadcast_to(gradient[None, None] / len(q), (1,) + q.shape)
    for cv in cvs:
        result = cv.value_and_jacobian(q)
        np.testing.assert_allclose(result.value, [expected], atol=2e-16)
        np.testing.assert_array_equal(result.jacobian, jacobian)
        value, derivative, hessian = cv.centroid_value_gradient_hessian(q[0])
        assert value == float(np.sum(gradient * q[0]) - options["origin"])
        np.testing.assert_array_equal(hessian, 0)
        derivative[:] = 99
        np.testing.assert_array_equal(cv.centroid_value_gradient(q[0])[1], gradient)

    fractional = FractionalCoordinate(0, 1, skew_cell(), [True, True, False], masses, **options)
    reciprocal = np.linalg.inv(skew_cell())[:, 1]
    coefficients = weights[:, None] * reciprocal[None]
    result = fractional.value_and_jacobian(q)
    np.testing.assert_allclose(result.value, [np.sum(coefficients * q.mean(axis=0)) - options["origin"]], atol=2e-16)
    np.testing.assert_array_equal(result.jacobian, np.broadcast_to(coefficients[None, None] / len(q), (1,) + q.shape))
    assert fractional.units == ("dimensionless",)
