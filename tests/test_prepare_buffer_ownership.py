"""Transition-state preparation must retain forces across PES evaluations."""

import numpy as np

from pyrpmd.potentials import PESResult
from pyrpmd.system import System
from pyrpmd.workflows.prepare import numerical_hessian, prepare_rate


class ReusingQuadraticPotential:
    """Analytical quadratic PES with one result buffer, like a compiled PES."""

    def __init__(self, hessian, center):
        self.hessian = np.asarray(hessian, dtype=float)
        self.center = np.asarray(center, dtype=float)
        self.result = PESResult(np.empty(1), np.empty((1, *self.center.shape)))

    def evaluate(self, q):
        delta = (q - self.center).reshape(1, -1)
        gradient = delta @ self.hessian
        self.result.energy[:] = 0.5 * np.sum(delta * gradient, axis=1)
        self.result.forces[:] = -gradient.reshape(q.shape)
        return self.result


def test_numerical_hessian_retains_reused_force_buffers():
    center = np.array([[0.2, -0.4, 0.7], [1.3, 0.1, -0.5]])
    # Cross derivatives distinguish the entire tensor from a diagonal-only
    # check; the known Hessian does not depend on the displacement or center.
    hessian = np.diag([0.4, 0.8, 1.2, 1.6, 2.0, 2.4])
    hessian[0, 4] = hessian[4, 0] = 0.13
    hessian[1, 5] = hessian[5, 1] = -0.19
    geometry = center + np.arange(6).reshape(2, 3) * 0.03
    provider = ReusingQuadraticPotential(hessian, center)

    actual = numerical_hessian(provider, geometry)

    np.testing.assert_allclose(actual, hessian, rtol=1e-10, atol=1e-11)


def test_prepare_rate_retains_saddle_mode_with_reused_force_buffers():
    geometry = np.array([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    eigenvalues = np.array([-0.4, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    rotation = np.eye(9)
    angle = 0.37
    rotation[np.ix_([0, 8], [0, 8])] = [
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ]
    hessian = rotation @ np.diag(eigenvalues) @ rotation.T
    provider = ReusingQuadraticPotential(hessian, geometry)
    system = System(("H", "H", "H"), np.ones(3), geometry)

    result = prepare_rate(
        {"reaction": {"preparation": {"trace_steps": 2}}},
        system,
        provider,
        numerical=True,
    )

    np.testing.assert_allclose(result["hessian_eigenvalues"], eigenvalues, rtol=1e-10, atol=1e-11)
    assert "negative mass-weighted Hessian eigenvalues = 1" in result["diagnostics"]
    assert len(result["trace_endpoints"]) == 2
