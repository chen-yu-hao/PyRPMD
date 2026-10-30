"""Derivative demand must not change CV mathematics or extension semantics."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from pyrpmd.cv import (
    BondDifference, Cartesian, COMDistance, CVResult, CVStack, Distance,
    PeriodicDistance, RPMDrateCoordinate,
)


REFERENCE = np.array([
    [-2.0, 0.1, 0.0], [0.0, 0.0, 0.0],
    [1.8, 0.2, 0.0], [3.1, 0.4, 0.1],
])
RING = REFERENCE[None] + np.random.default_rng(794).normal(0, 0.2, (3, 4, 3))


def coordinate():
    return RPMDrateCoordinate(
        [[0], [1, 2, 3]], np.ones(4), REFERENCE,
        [[0, 1], [1, 2]], [[1, 2], [2, 3]], 12,
        equivalent_paths=[[0, 3, 2, 1]],
    )


def make_cv(kind, estimator="centroid"):
    if kind == "distance":
        return Distance([0, 1], estimator=estimator)
    if kind == "periodic_distance":
        return PeriodicDistance([0, 1], np.eye(3) * 8, estimator=estimator)
    if kind == "bond_difference":
        return BondDifference(atoms=[0, 1, 2], estimator=estimator)
    if kind == "cartesian":
        return Cartesian(2, "y", estimator=estimator)
    if kind == "com":
        return COMDistance([[0], [1, 2, 3]], [1., 2., 3., 4.], estimator=estimator)
    if kind == "rate":
        return coordinate()
    if kind == "surface":
        return coordinate().dividing_surface(0.37)
    raise AssertionError(kind)


ORDINARY = ["distance", "periodic_distance", "bond_difference", "cartesian", "com"]
CASES = [(kind, estimator) for kind in ORDINARY
         for estimator in ("centroid", "bead_average")] + [
             ("rate", "centroid"), ("surface", "centroid"),
         ]


def reject_hessians(monkeypatch):
    zeros, einsum = np.zeros, np.einsum

    def checked_zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 4:
            raise AssertionError("ordinary CV evaluation allocated a dense Hessian")
        return zeros(shape, *args, **kwargs)

    def checked_einsum(expression, *args, **kwargs):
        if isinstance(expression, str) and expression.endswith("->acbd"):
            raise AssertionError("ordinary CV evaluation contracted a dense Hessian")
        return einsum(expression, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", checked_zeros)
    monkeypatch.setattr(np, "einsum", checked_einsum)


@pytest.mark.parametrize("kind,estimator", CASES)
def test_values_match_full_ring_estimator_without_gradient_or_jacobian_allocations(
    monkeypatch, kind, estimator,
):
    cv = make_cv(kind, estimator)
    expected = cv.value_and_jacobian(RING).value
    reject_hessians(monkeypatch)

    def no_derivative_array(*args, **kwargs):
        raise AssertionError("value-only CV evaluation allocated a derivative array")

    monkeypatch.setattr(np, "zeros_like", no_derivative_array)
    monkeypatch.setattr(np, "broadcast_to", no_derivative_array)
    np.testing.assert_array_equal(cv.value(RING), expected)
    np.testing.assert_array_equal(cv.value(SimpleNamespace(q=RING)), expected)
    np.testing.assert_array_equal(CVStack([cv, cv]).value(RING), np.tile(expected, 2))


@pytest.mark.parametrize("kind", ["com", "rate", "surface"])
def test_gradient_equals_explicit_hessian_result_without_allocating_hessian(monkeypatch, kind):
    cv = make_cv(kind)
    q = RING.mean(axis=0)
    expected, gradient, _ = cv.centroid_value_gradient_hessian(q)
    reject_hessians(monkeypatch)
    value, actual = cv.centroid_value_gradient(q)
    assert value == expected
    np.testing.assert_array_equal(actual, gradient)
    result = cv.value_and_jacobian(RING)
    np.testing.assert_array_equal(result.value, [expected])
    np.testing.assert_array_equal(result.jacobian[0], np.broadcast_to(gradient / len(RING), RING.shape))


def test_unmodified_periodic_distance_subclass_keeps_existing_hessian_free_gradient(monkeypatch):
    class NamedPeriodicDistance(PeriodicDistance):
        pass

    cv = NamedPeriodicDistance([0, 1], np.eye(3) * 8)
    expected = cv.value_and_jacobian(RING)
    reject_hessians(monkeypatch)
    actual = cv.value_and_jacobian(RING)
    np.testing.assert_array_equal(actual.value, expected.value)
    np.testing.assert_array_equal(actual.jacobian, expected.jacobian)


def test_hessian_request_only_allocates_selected_bond_hessians(monkeypatch):
    cv = coordinate()
    zeros = np.zeros
    shapes = []

    def record_zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 4:
            shapes.append(shape)
        return zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", record_zeros)
    cv.surfaces(RING.mean(axis=0))
    # Four possible path/bond pairs require only the selected pair's two
    # distance Hessians. The COM Hessian is a separate einsum contraction.
    assert shapes == [(4, 3, 4, 3)] * 2


def error_of(function):
    try:
        function()
    except Exception as error:
        return type(error), str(error)
    raise AssertionError("expected invalid coordinates to be rejected")


@pytest.mark.parametrize("kind,estimator", CASES)
@pytest.mark.parametrize("defect", ["rank", "axes", "nan", "infinity"])
def test_value_preserves_coordinate_validation(kind, estimator, defect):
    cv = make_cv(kind, estimator)
    q = RING.copy()
    if defect == "rank":
        q = q[0]
    elif defect == "axes":
        q = q[:, :, :2]
    elif defect == "nan":
        q[0, 0, 0] = np.nan
    else:
        q[0, 0, 0] = np.inf
    assert error_of(lambda: cv.value(q)) == error_of(lambda: cv.value_and_jacobian(q))


@pytest.mark.parametrize("kind", ["distance", "periodic_distance", "com", "rate", "surface"])
def test_value_keeps_zero_distance_error(kind):
    cv = make_cv(kind)
    q = np.zeros_like(RING)
    assert error_of(lambda: cv.value(q)) == error_of(lambda: cv.value_and_jacobian(q))


@pytest.mark.parametrize("cv", [Distance([0, 0]), Distance([-1, 1]), Distance([0, 9]), Cartesian(9)])
def test_value_keeps_atom_index_errors(cv):
    assert error_of(lambda: cv.value(RING)) == error_of(lambda: cv.value_and_jacobian(RING))


def test_periodic_value_keeps_coincident_image_error():
    cv = PeriodicDistance([0, 1], np.eye(3) * 8)
    q = np.zeros((1, 2, 3))
    q[0, 1, 0] = 8
    assert error_of(lambda: cv.value(q)) == (
        ValueError, "periodic distance derivative is undefined for coincident images",
    )


@pytest.mark.parametrize("kind", ["com", "rate", "surface"])
@pytest.mark.parametrize("shape", [(3, 3), (4, 2), (4, 4)])
def test_gradient_preserves_malformed_centroid_errors(kind, shape):
    cv = make_cv(kind)
    q = np.arange(np.prod(shape), dtype=float).reshape(shape)
    assert error_of(lambda: cv.centroid_value_gradient(q)) == error_of(
        lambda: cv.centroid_value_gradient_hessian(q)
    )


def test_rate_singularity_is_still_rejected_but_linear_surface_is_defined():
    cv = coordinate()
    q = REFERENCE.copy()
    cv.rinf = cv.com.centroid_value_gradient(q)[0]
    expected = (ValueError, "RPMDrate xi is singular where s0=s1")
    assert error_of(lambda: cv.value(q[None])) == expected
    assert error_of(lambda: cv.centroid_value_gradient(q)) == expected
    assert error_of(lambda: cv.centroid_value_gradient_hessian(q)) == expected
    np.testing.assert_array_equal(cv.dividing_surface(0.5).value(q[None]), [0.0])


def test_minmax_branch_ties_keep_first_bond_then_first_path():
    q = np.array([[0., 2., 0.], [-1., 0., 0.], [1., 0., 0.]])
    cv = RPMDrateCoordinate(
        [[0], [1, 2]], np.ones(3), q,
        [[0, 1], [0, 2]], [[1, 2], [1, 2]], 10,
        equivalent_paths=[[0, 2, 1]],
    )
    _, gf, hf = Distance([0, 1]).centroid_value_gradient_hessian(q)
    _, gb, hb = Distance([1, 2]).centroid_value_gradient_hessian(q)
    _, _, _, s1, g1, h1 = cv.surfaces(q)
    assert s1 == 0.0
    np.testing.assert_array_equal(g1, -gf + gb)
    np.testing.assert_array_equal(h1, -hf + hb)
    value, gradient, _ = cv.centroid_value_gradient_hessian(q)
    np.testing.assert_array_equal(cv.value(q[None]), [value])
    np.testing.assert_array_equal(cv.centroid_value_gradient(q)[1], gradient)


def test_unselected_degenerate_bond_is_not_silently_ignored():
    cv = coordinate()
    q = REFERENCE.copy()
    # The second equivalent path contains this forming bond; the original
    # implementation validates it even if another path would be selected.
    q[3] = q[0]
    expected = (ValueError, "CV derivative is undefined for coincident atoms")
    assert error_of(lambda: cv.value(q[None])) == expected
    assert error_of(lambda: cv.centroid_value_gradient(q)) == expected
    assert error_of(lambda: cv.surfaces(q)) == expected


def install_override(monkeypatch, cv, method, replacement, mode):
    if mode == "instance":
        monkeypatch.setattr(cv, method, MethodType(replacement, cv))
    elif mode == "class":
        monkeypatch.setattr(type(cv), method, replacement)
    else:
        cv.__class__ = type("CustomCV", (type(cv),), {method: replacement})


@pytest.mark.parametrize("kind", ["com", "rate", "surface", "periodic_distance"])
@pytest.mark.parametrize("mode", ["instance", "class", "subclass"])
def test_hessian_extensions_remain_visible_to_gradient_and_value(monkeypatch, kind, mode):
    cv = make_cv(kind)
    original = type(cv).centroid_value_gradient_hessian

    def changed(self, q):
        value, gradient, hessian = original(self, q)
        return 2 * value + 0.7, 2 * gradient, 2 * hessian

    install_override(monkeypatch, cv, "centroid_value_gradient_hessian", changed, mode)
    q = RING.mean(axis=0)
    expected, gradient, _ = changed(cv, q)
    actual, derivative = cv.centroid_value_gradient(q)
    assert actual == expected
    np.testing.assert_array_equal(derivative, gradient)
    np.testing.assert_array_equal(cv.value(RING), [expected])
    np.testing.assert_array_equal(CVStack([cv]).value(RING), [expected])


@pytest.mark.parametrize("method", ["value_and_jacobian", "centroid_value_gradient"])
@pytest.mark.parametrize("mode", ["instance", "class", "subclass"])
def test_value_only_dispatch_preserves_first_derivative_overrides(monkeypatch, method, mode):
    cv = make_cv("distance")
    original = getattr(type(cv), method)

    def changed(self, q):
        result = original(self, q)
        if method == "value_and_jacobian":
            return CVResult(3 * result.value + 0.2, 3 * result.jacobian)
        value, derivative = result
        return 3 * value + 0.2, 3 * derivative

    install_override(monkeypatch, cv, method, changed, mode)
    expected = cv.value_and_jacobian(RING).value
    np.testing.assert_array_equal(cv.value(RING), expected)
    np.testing.assert_array_equal(CVStack([cv]).value(RING), expected)


@pytest.mark.parametrize("kind", ["rate", "surface"])
@pytest.mark.parametrize("nested", ["surfaces", "com"])
@pytest.mark.parametrize("mode", ["instance", "class", "subclass"])
def test_rate_fast_paths_keep_nested_extension_points(monkeypatch, kind, nested, mode):
    cv = make_cv(kind)
    rate = cv.coordinate if kind == "surface" else cv
    target = rate.com if nested == "com" else rate
    method = "centroid_value_gradient_hessian" if nested == "com" else "surfaces"
    original = getattr(type(target), method)

    def changed(self, q):
        result = original(self, q)
        return tuple(1.3 * part for part in result[:3]) + result[3:]

    install_override(monkeypatch, target, method, changed, mode)
    expected, derivative, _ = cv.centroid_value_gradient_hessian(RING.mean(axis=0))
    value, gradient = cv.centroid_value_gradient(RING.mean(axis=0))
    assert value == expected
    np.testing.assert_array_equal(gradient, derivative)
    np.testing.assert_array_equal(cv.value(RING), [expected])
    np.testing.assert_array_equal(CVStack([cv]).value(RING), [expected])


@pytest.mark.parametrize("mode", ["instance", "class", "subclass"])
def test_cvstack_honors_its_own_jacobian_override(monkeypatch, mode):
    stack = CVStack([make_cv("distance")])
    original = CVStack.value_and_jacobian

    def changed(self, q):
        result = original(self, q)
        return CVResult(result.value + 4, result.jacobian)

    install_override(monkeypatch, stack, "value_and_jacobian", changed, mode)
    np.testing.assert_array_equal(stack.value(RING), stack.value_and_jacobian(RING).value)


def test_cvstack_keeps_custom_child_jacobian_protocol_instead_of_independent_value():
    class Plugin:
        dimension = 1
        units = ("dimensionless",)
        periods = (None,)

        def value(self, q):
            raise AssertionError("CVStack historically uses the plugin Jacobian protocol")

        def value_and_jacobian(self, q):
            return CVResult(np.array([7.]), np.zeros((1,) + q.shape))

    cv = make_cv("com")
    expected = cv.value_and_jacobian(RING).value
    cv.value = lambda q: np.array([99.])
    np.testing.assert_array_equal(CVStack([cv, Plugin()]).value(RING), [expected[0], 7.])


def test_cvstack_does_not_suppress_custom_jacobian_shape_errors():
    cv = make_cv("distance")
    cv.value_and_jacobian = lambda q: CVResult(np.array([7.]), np.array(0.))
    stack = CVStack([cv])
    assert error_of(lambda: stack.value(RING)) == error_of(lambda: stack.value_and_jacobian(RING))
