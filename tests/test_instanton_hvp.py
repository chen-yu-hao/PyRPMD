"""Independent quadratic reference for matrix-free action Hessian products."""

import numpy as np
import pytest

from pyrpmd.instanton import RingPolymerAction
from pyrpmd.potentials.base import PESResult


class QuadraticPotential:
    def __init__(self, force_constants):
        self.curvature = np.asarray(force_constants, dtype=np.float64)

    def evaluate(self, q):
        return PESResult(
            0.5 * np.einsum("bni,nij,bnj->b", q, self.curvature, q),
            -np.einsum("nij,bnj->bni", self.curvature, q),
        )

    def hessian(self, q):
        n = len(self.curvature)
        h = np.zeros((3*n, 3*n), dtype=np.float64)
        for i in range(n):
            h[3*i:3*i+3, 3*i:3*i+3] = self.curvature[i]
        return np.broadcast_to(h, (len(q), 3*n, 3*n)).copy()

    def hessian_vector_product(self, q, v):
        return np.einsum("nij,bnj->bni", self.curvature, v)


def test_action_hvp_exact_against_dense_and_gradient_differences():
    rng = np.random.default_rng(421)
    a = rng.normal(size=(2, 3, 3))
    curvature = np.einsum("nij,nkj->nik", a, a) + np.eye(3)[None]
    model = QuadraticPotential(curvature)
    action = RingPolymerAction(model, [1837., 29000.], 300)
    q = rng.normal(size=(8, 2, 3)) * .05
    v = rng.normal(size=q.shape)
    full = action.hessian(q)
    hv = action.hessian_vector_product(q, v)
    assert hv.shape == q.shape
    np.testing.assert_allclose(hv.ravel(), full @ v.ravel(), rtol=1e-11, atol=1e-9)
    epsilon = 1e-5
    finite_difference = (
        action.evaluate(q + epsilon * v)["gradient"]
        - action.evaluate(q - epsilon * v)["gradient"]
    ) / (2 * epsilon)
    np.testing.assert_allclose(hv, finite_difference, rtol=1e-9, atol=1e-7)


def test_action_hvp_active_mask_shape_and_strict_missing_provider():
    class NoHVP:
        pass

    rng = np.random.default_rng(42)
    curvature = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    model = QuadraticPotential(curvature)
    mask = np.array([[True, False, True], [True, True, False]])
    action = RingPolymerAction(model, [1837., 29000.], 300, active_mask=mask)
    q = rng.normal(size=(8, 2, 3)) * .01
    q[:, ~mask] = q[0, ~mask]
    v = np.zeros_like(q)
    v[:, mask] = rng.normal(size=(8, mask.sum()))
    expected = action.hessian(q) @ v[:, mask].ravel()
    product = action.hessian_vector_product(q, v)
    assert product.shape == q.shape
    np.testing.assert_allclose(product[:, mask].ravel(), expected,
                               rtol=1e-11, atol=1e-9)
    np.testing.assert_array_equal(product[:, ~mask], 0.0)
    with pytest.raises(ValueError, match="inactive"):
        action.hessian_vector_product(q, np.ones_like(q))
    with pytest.raises(ValueError, match="float64"):
        action.hessian_vector_product(q, v.astype(np.float32))
    action.potential = NoHVP()
    with pytest.raises(ValueError, match="hessian_vector_product"):
        action.hessian_vector_product(q, v)
