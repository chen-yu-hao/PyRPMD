"""Independent finite-P trace and force controls for the isomorphic methods."""

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.linalg import expm

from pyrpmd.isomorphic import (IsomorphicEhrenfestDynamics,
                               IsomorphicEhrenfestState, IsomorphicPotential)
from pyrpmd.nonadiabatic import DiabaticResult
from pyrpmd.potentials.base import PotentialError


class TwoStateModel:
    def __init__(self, coupling=0.2, offset=0.0):
        self.coupling, self.offset = coupling, offset

    def evaluate(self, q):
        direction = np.zeros_like(q)
        direction[:, 0, 0] = 1
        direction[:, -1, 2] = 0.3
        x = np.sum(q * direction, axis=(1, 2))
        common = 0.2 * np.sum(q*q, axis=(1, 2)) + self.offset
        value = np.zeros((len(q), 2, 2))
        grad = np.zeros((*value.shape, *q.shape[1:]))
        value[:, 0, 0], value[:, 1, 1] = common + 0.3*x, common - 0.3*x + 0.1
        value[:, 0, 1] = value[:, 1, 0] = self.coupling * np.exp(0.1*x)
        grad[:, 0, 0], grad[:, 1, 1] = 0.4*q + 0.3*direction, 0.4*q - 0.3*direction
        grad[:, 0, 1] = grad[:, 1, 0] = 0.1 * value[:, 0, 1, None, None] * direction
        return DiabaticResult(value, grad)


class ConstantMatrix:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)

    def evaluate(self, q):
        return DiabaticResult(np.repeat(self.matrix[None], len(q), axis=0),
                              np.zeros((len(q), *self.matrix.shape, *q.shape[1:])))


@pytest.mark.parametrize("nbeads", [1, 2, 5])
@pytest.mark.parametrize("coupling", [0.0, 0.2, -0.2])
def test_isomorphic_finite_p_boltzmann_trace_is_exact(nbeads, coupling):
    q = np.random.default_rng(11).normal(scale=0.5, size=(nbeads, 2, 3))
    model = TwoStateModel(coupling)
    beta = 3.4
    result = IsomorphicPotential(model, beta, nbeads).evaluate(q)
    electronic = np.eye(2)
    for matrix in model.evaluate(q).matrix:
        electronic = electronic @ expm(-beta/nbeads * matrix)
    expected = np.trace(electronic)
    assert np.trace(expm(-beta * result.matrix)) == pytest.approx(expected, rel=3e-13)
    assert result.log_electronic_weight == pytest.approx(np.log(expected), abs=3e-13)


@pytest.mark.parametrize("coupling", [0.0, 0.2, -0.2])
def test_isomorphic_whole_ring_gradients(coupling):
    q = np.random.default_rng(8).normal(scale=0.3, size=(4, 2, 3))
    potential = IsomorphicPotential(TwoStateModel(coupling), 4.1, 4)
    result = potential.evaluate(q)
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += 2e-6
        minus[index] -= 2e-6
        numerical = (potential.evaluate(plus).matrix - potential.evaluate(minus).matrix) / 4e-6
        np.testing.assert_allclose(result.gradients[(slice(None), slice(None), *index)],
                                   numerical, atol=4e-9, rtol=2e-7)


def test_collapsed_ring_and_single_bead_return_physical_matrix():
    geometry = np.array([[[0.4, -0.1, 0.2]]])
    model = TwoStateModel()
    for count in (1, 3, 8):
        q = np.repeat(geometry, count, axis=0)
        result = IsomorphicPotential(model, 2, count).evaluate(q)
        np.testing.assert_allclose(result.matrix, model.evaluate(geometry).matrix[0], atol=1e-14)


@pytest.mark.parametrize("nbeads", [1, 2, 4, 8])
@pytest.mark.parametrize("coupling", [1e-8, 1e-12, 1e-30])
def test_constant_weak_coupling_is_not_replaced_by_trace_roundoff(nbeads, coupling):
    matrix = np.array([[0.2, coupling], [coupling, -0.2]])
    result = IsomorphicPotential(ConstantMatrix(matrix), 3, nbeads).evaluate(np.zeros((nbeads, 1, 3)))
    np.testing.assert_array_equal(result.matrix, matrix)
    np.testing.assert_array_equal(result.gradients, 0)


def test_noncollapsed_unresolved_weak_coupling_is_rejected():
    q = np.zeros((4, 1, 3))
    q[:, 0, 0] = [-0.5, -0.2, 0.1, 0.4]
    with pytest.raises(PotentialError, match="numerically unresolved"):
        IsomorphicPotential(TwoStateModel(coupling=1e-12), 3, 4).evaluate(q)


def test_cyclic_symmetry_and_large_scalar_shift():
    q = np.random.default_rng(3).normal(scale=0.2, size=(4, 1, 3))
    result = IsomorphicPotential(TwoStateModel(), 3, 4).evaluate(q)
    rotated = IsomorphicPotential(TwoStateModel(), 3, 4).evaluate(np.roll(q, 1, axis=0))
    np.testing.assert_allclose(rotated.matrix, result.matrix, atol=1e-14)
    np.testing.assert_allclose(rotated.gradients, np.roll(result.gradients, 1, axis=2), atol=1e-14)
    shifted = IsomorphicPotential(TwoStateModel(offset=1e5), 3, 4).evaluate(q)
    np.testing.assert_allclose(shifted.matrix - 1e5*np.eye(2), result.matrix, atol=2e-11)
    np.testing.assert_allclose(shifted.gradients, result.gradients, atol=3e-11)


def test_isomorphic_frozen_electronic_rotation():
    matrix = np.array([[0.3, 0.17], [0.17, -0.2]])
    q = np.zeros((4, 1, 3))
    state = IsomorphicEhrenfestState(q, q, amplitudes=[0.8, 0.6j])
    original = state.amplitudes.copy()
    dynamics = IsomorphicEhrenfestDynamics(ConstantMatrix(matrix), [1], 3, 4, 0.02, native=False)
    for _ in range(10):
        dynamics.step(state)
    expected = expm(-0.2j * matrix) @ original
    np.testing.assert_allclose(state.amplitudes, expected, atol=3e-14)
    assert sum(state.populations()) == pytest.approx(1, abs=3e-14)


def test_isomorphic_exact_electronic_force_impulse():
    q = np.array([[[0.4, -0.1, 0.2]], [[-0.3, 0.1, -0.4]]])
    state = IsomorphicEhrenfestState(q, np.zeros_like(q), amplitudes=[0.8, 0.6j])
    dynamics = IsomorphicEhrenfestDynamics(TwoStateModel(), [1], 3, 2, 0.01, native=False)
    result = dynamics.potential.evaluate(q)
    initial = state.amplitudes.copy()
    duration = 0.7

    def force(time):
        amplitude = expm(-1j*time*result.matrix) @ initial
        return -2*np.einsum("i,ijbac,j->bac", amplitude.conj(), result.gradients, amplitude).real

    expected = quad_vec(force, 0, duration, epsabs=1e-12)[0]
    dynamics._potential_step(state, duration)
    np.testing.assert_allclose(state.p, expected, atol=3e-14)


def test_isomorphic_reversibility_and_energy_convergence():
    q = np.random.default_rng(7).normal(scale=0.15, size=(3, 1, 3))
    start = IsomorphicEhrenfestState(q, np.full_like(q, 0.07), amplitudes=[0.8, 0.6j])
    errors = []
    for dt in (0.02, 0.01):
        dynamics = IsomorphicEhrenfestDynamics(TwoStateModel(), [2], 3, 3, dt, native=False)
        state = start.copy()
        energy = dynamics.energy(state)
        count = round(0.2/dt)
        for _ in range(count):
            dynamics.step(state)
        errors.append(abs(dynamics.energy(state) - energy))
        backward = IsomorphicEhrenfestDynamics(TwoStateModel(), [2], 3, 3, -dt, native=False)
        for _ in range(count):
            backward.step(state)
        for key in ("q", "p", "amplitudes"):
            np.testing.assert_allclose(getattr(state, key), getattr(start, key), atol=5e-13)
    assert errors[1] < errors[0]/3.8


def test_isomorphic_rejects_sign_changing_and_complex_coupling():
    class Changing(TwoStateModel):
        def evaluate(self, q):
            result = super().evaluate(q)
            result.matrix[:, 0, 1] = result.matrix[:, 1, 0] = q[:, 0, 0]
            result.gradients[:, 0, 1] = result.gradients[:, 1, 0] = 0
            result.gradients[:, 0, 1, 0, 0] = result.gradients[:, 1, 0, 0, 0] = 1
            return result
    q = np.zeros((3, 1, 3))
    q[:, 0, 0] = [-1, 0.3, 1]
    with pytest.raises(PotentialError, match="nonzero sign"):
        IsomorphicPotential(Changing(), 2, 3).evaluate(q)
    with pytest.raises(PotentialError, match="real|complex"):
        IsomorphicPotential(ConstantMatrix([[0, 0.1j], [-0.1j, 0.1]]), 2, 3).evaluate(q)


def test_isomorphic_failed_step_is_transactional():
    class Failed(TwoStateModel):
        calls = 0
        def evaluate(self, q):
            self.calls += 1
            if self.calls == 3:
                raise PotentialError("injected failure")
            return super().evaluate(q)
    q = np.zeros((3, 1, 3))
    state = IsomorphicEhrenfestState(q, np.full_like(q, 0.1), amplitudes=[1, 0])
    before = state.copy()
    dynamics = IsomorphicEhrenfestDynamics(Failed(), [1], 3, 3, 0.01, native=False)
    with pytest.raises(PotentialError, match="injected"):
        dynamics.step(state)
    for key in ("q", "p", "amplitudes"):
        np.testing.assert_array_equal(getattr(state, key), getattr(before, key))
    assert state.step == before.step and state.time == before.time

