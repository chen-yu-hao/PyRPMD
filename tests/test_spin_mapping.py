"""References use matrix exponentials/ODEs, independent of Bloch rotations."""
import numpy as np
import pytest
from scipy.integrate import quad_vec, solve_ivp
from scipy.linalg import expm

from pyrpmd.dynamics import State
from pyrpmd.nonadiabatic import DiabaticResult
from pyrpmd.potentials.base import PotentialError
from pyrpmd.spin_mapping import (SMNRPMDDynamics, SMNRPMDState, W_RADIUS, _spin_mint,
                                 _spin_to_mapping, _mapping_to_spin)


PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


class MatrixPotential:
    def __init__(self, matrix, slope=None, curvature=0.0):
        self.matrix = np.array(matrix, dtype=complex)
        self.slope = np.zeros((2, 2), complex) if slope is None else np.array(slope, complex)
        self.curvature = curvature

    def evaluate(self, q):
        b, n, _ = q.shape
        matrix = self.matrix[None] + q[:, 0, 0, None, None] * self.slope
        matrix += 0.5 * self.curvature * np.sum(q * q, axis=(1, 2))[:, None, None] * np.eye(2)
        gradients = self.curvature * q[:, None, None] * np.eye(2)[None, :, :, None, None]
        gradients = np.broadcast_to(gradients, (b, 2, 2, n, 3)).astype(complex).copy()
        gradients[:, :, :, 0, 0] += self.slope
        return DiabaticResult(matrix, gradients)


def initial(beads=3):
    rng = np.random.default_rng(781)
    nuclear = State(rng.normal(size=(beads, 1, 3)), rng.normal(size=(beads, 1, 3)))
    return SMNRPMDState.from_nuclear(nuclear, 0, rng)


def density(spin):
    return np.eye(2) / 2 + W_RADIUS * np.einsum("i,ijk->jk", spin, PAULI)


@pytest.mark.parametrize("initial_state", [0, 1])
def test_focused_initialization_and_ownership(initial_state):
    nuclear = State(np.zeros((1024, 1, 3)), np.zeros((1024, 1, 3)))
    state = SMNRPMDState.from_nuclear(nuclear, initial_state, 12)
    np.testing.assert_allclose(state.populations(), np.eye(2)[initial_state], atol=2e-14)
    np.testing.assert_allclose(np.linalg.norm(state.spin, axis=1), 1.0, atol=3e-16)
    assert len(np.unique(state.spin[:, 0])) == 1024
    copy = state.copy()
    copy.q[:] = 5
    copy.spin[:] *= -1
    assert np.all(nuclear.q == 0) and np.all(state.q == 0)
    np.testing.assert_array_equal(copy.spin, -state.spin)
    assert abs(state.density_matrix()[0, 1]) < 6 * np.sqrt(0.5 / 1024)


def test_pure_electronic_complex_rotation_and_coherence_sign():
    matrix = np.array([[0.3, 0.2 + 0.7j], [0.2 - 0.7j, -0.4]])
    state = initial()
    before = [density(s) for s in state.spin]
    driver = SMNRPMDDynamics(MatrixPotential(matrix), [1.4], 3, 3, 0.07, native=False)
    for _ in range(13):
        driver.step(state)
    unitary = expm(-1j * matrix * 0.91)
    for spin, rho in zip(state.spin, before):
        np.testing.assert_allclose(density(spin), unitary @ rho @ unitary.conj().T, atol=3e-15)
    np.testing.assert_allclose(driver.density_matrix(state), np.mean([density(s) for s in state.spin], axis=0), atol=2e-16)


@pytest.mark.parametrize("gap", [0.0, 1e-15, 0.8])
def test_exact_fixed_q_impulse_against_independent_matrix_integral(gap):
    pot = MatrixPotential([[0.2, 0], [0, 0.2 + gap]], [[0.6, 0.2 + 0.5j], [0.2 - 0.5j, -0.4]])
    state = initial(1)
    state.q[:] = 0
    p0, rho0 = state.p.copy(), density(state.spin[0])
    result = pot.evaluate(state.q)
    duration = 0.83
    def force(t):
        u = expm(-1j * pot.matrix * t)
        rho = u @ rho0 @ u.conj().T
        return -np.einsum("ij,jiac->ac", rho, result.gradients[0]).real
    impulse, _ = quad_vec(force, 0, duration, epsabs=1e-12)
    _spin_mint(state, result, duration)
    np.testing.assert_allclose(state.p, p0 + impulse, atol=2e-14)
    u = expm(-1j * pot.matrix * duration)
    np.testing.assert_allclose(density(state.spin[0]), u @ rho0 @ u.conj().T, atol=2e-15)


def test_p1_full_equations_against_density_matrix_ode():
    pot = MatrixPotential([[0.2, 0.3j], [-0.3j, -0.1]], [[0.4, 0.1], [0.1, -0.4]], 0.5)
    state = initial(1)
    mass = 1.8
    y0 = np.concatenate((state.q.ravel(), state.p.ravel(), density(state.spin[0]).ravel()))
    def rhs(t, y):
        q, p, rho = y[:3].real.reshape(1, 1, 3), y[3:6].real, y[6:].reshape(2, 2)
        value = pot.evaluate(q)
        force = -np.einsum("ij,jiac->ac", rho, value.gradients[0]).real.ravel()
        drho = -1j * (value.matrix[0] @ rho - rho @ value.matrix[0])
        return np.concatenate((p / mass, force, drho.ravel()))
    reference = solve_ivp(rhs, (0, 0.2), y0, rtol=2e-12, atol=2e-13).y[:, -1]
    errors = []
    for dt in (0.01, 0.005):
        trial = state.copy()
        driver = SMNRPMDDynamics(pot, [mass], 2, 1, dt, native=False)
        for _ in range(round(0.2 / dt)):
            driver.step(trial)
        observed = np.concatenate((trial.q.ravel(), trial.p.ravel(), density(trial.spin[0]).ravel()))
        errors.append(np.linalg.norm(observed - reference))
    assert 3.9 < errors[0] / errors[1] < 4.1
    assert errors[1] < 1e-5


def test_energy_convergence_time_reversal_and_signed_kernel():
    pot = MatrixPotential([[0.2, 0.15], [0.15, -0.2]], [[0.3, 0.1j], [-0.1j, -0.3]], 0.6)
    state = initial(4)
    errors = []
    for dt in (0.01, 0.005):
        trial = state.copy()
        driver = SMNRPMDDynamics(pot, [1.7], 4, 4, dt, native=False, integrator="normal-mode-bab")
        e0 = driver.energy(trial)
        for _ in range(round(0.4 / dt)):
            driver.step(trial)
        errors.append(abs(driver.energy(trial) - e0))
        reverse = SMNRPMDDynamics(pot, [1.7], 4, 4, -dt, native=False, integrator="normal-mode-bab")
        for _ in range(round(0.4 / dt)):
            reverse.step(trial)
        for name in ("q", "p", "spin"):
            np.testing.assert_allclose(getattr(trial, name), getattr(state, name), atol=1e-13)
    assert 3.8 < errors[0] / errors[1] < 4.2
    polar = SMNRPMDState(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), spin=[[0, 0, 1]])
    assert polar.populations()[1] < 0
    np.testing.assert_allclose(np.trace(polar.density_matrix()), 1)


def test_constant_unitary_gauge_covariance_and_scalar_energy_shift():
    pot = MatrixPotential([[0.2, 0.3j], [-0.3j, -0.1]], [[0.4, 0.1], [0.1, -0.4]], 0.5)
    unitary = expm(-0.37j * (PAULI[0] + 0.6 * PAULI[1]))
    other = MatrixPotential(unitary @ pot.matrix @ unitary.conj().T + 3.2 * np.eye(2),
                            unitary @ pot.slope @ unitary.conj().T, pot.curvature)
    state = initial()
    transformed = state.copy()
    for b, spin in enumerate(state.spin):
        rho = unitary @ density(spin) @ unitary.conj().T
        transformed.spin[b] = np.einsum("ij,kji->k", rho, PAULI).real / (2 * W_RADIUS)
    d1 = SMNRPMDDynamics(pot, [2.1], 4, 3, 0.02, native=False)
    d2 = SMNRPMDDynamics(other, [2.1], 4, 3, 0.02, native=False)
    np.testing.assert_allclose(d2.energy(transformed) - d1.energy(state), 3 * 3.2, atol=1e-14)
    for _ in range(20):
        d1.step(state)
        d2.step(transformed)
    np.testing.assert_allclose(state.q, transformed.q, atol=2e-14)
    np.testing.assert_allclose(state.p, transformed.p, atol=2e-14)
    np.testing.assert_allclose(d2.density_matrix(transformed), unitary @ d1.density_matrix(state) @ unitary.conj().T, atol=2e-14)


def test_step_failure_is_transactional():
    class Failing(MatrixPotential):
        def evaluate(self, q):
            self.calls = getattr(self, "calls", 0) + 1
            if self.calls == 2:
                raise PotentialError("PES failed on second evaluation")
            return super().evaluate(q)
    state = initial()
    before = state.copy()
    driver = SMNRPMDDynamics(Failing(np.eye(2)), [1], 2, 3, 0.03, native=False)
    with pytest.raises(PotentialError, match="second"):
        driver.step(state)
    for key in ("q", "p", "spin"):
        np.testing.assert_array_equal(getattr(state, key), getattr(before, key))
    assert (state.step, state.time) == (before.step, before.time)


@pytest.mark.parametrize("spin", [[[0, 0, 0]], [[0, 0, 2]], [[0, 0, np.nan]], [[0j, 0j, 1j]],
                                 [[True, False, False]], [[2**32, 0, 1]]])
def test_invalid_spin_rejected(spin):
    with pytest.raises(ValueError):
        SMNRPMDState(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), spin=spin)


def test_two_state_only_and_mutation_validation():
    with pytest.raises(ValueError, match="two"):
        SMNRPMDDynamics(MatrixPotential(np.eye(2)), [1], 2, 1, 0.01, 3, native=False)
    state = initial(1)
    driver = SMNRPMDDynamics(MatrixPotential(np.eye(2)), [1], 2, 1, 0.01, native=False)
    state.spin *= 2
    with pytest.raises(ValueError, match="spin"):
        driver.step(state)


def test_spinor_charts_at_poles_and_global_phase_invariance():
    spins = np.array([[0., 0, 1], [0., 0, -1], [1e-12, 0, 1], [0., 1e-12, -1], [.6, -.8, 0.]])
    z = _spin_to_mapping(spins)
    np.testing.assert_allclose(np.sum(abs(z)**2, axis=1), 2*np.sqrt(3), atol=1e-15)
    np.testing.assert_allclose(_mapping_to_spin(z), spins, atol=3e-16)
    np.testing.assert_allclose(_mapping_to_spin(z*np.exp(1j*np.arange(len(spins))[:, None])), spins, atol=3e-16)
    for spin, mapping in zip(spins, z):
        rho = .5*np.outer(mapping, mapping.conj()) - .5*(np.sqrt(3)-1)*np.eye(2)
        np.testing.assert_allclose(rho, density(spin), atol=5e-16)


@pytest.mark.parametrize("backend", ["native", "torch"])
@pytest.mark.parametrize("gap", [0., .6])
def test_accelerated_backend_matches_independent_bloch_flow(backend, gap):
    if backend == "torch":
        pytest.importorskip("torch")
    else:
        pytest.importorskip("pyrpmd._electronic")
    pot = MatrixPotential([[.3, gap*(.2+.1j)], [gap*(.2-.1j), .3-gap]], [[.6, .2+.5j], [.2-.5j, -.4]])
    state = initial(5)
    state.q[:] = 0
    state.spin[:] = [[0, 0, 1], [0, 0, -1], [.6, -.8, 0], [1e-12, 0, 1], [0, 1e-12, -1]]
    expected = state.copy()
    ref = SMNRPMDDynamics(pot, [1.3], 4, 5, .02, native=False)
    accelerated = SMNRPMDDynamics(pot, [1.3], 4, 5, .02, native=False, electronic_backend=backend)
    for _ in range(15):
        ref.step(expected)
        accelerated.step(state)
    np.testing.assert_allclose(state.q, expected.q, atol=3e-14)
    np.testing.assert_allclose(state.p, expected.p, atol=3e-14)
    np.testing.assert_allclose(state.spin, expected.spin, atol=3e-14)


@pytest.mark.parametrize("kwargs", [{"electronic_backend": "cuda"}, {"electronic_device": "cuda"}])
def test_electronic_backend_validation_is_explicit(kwargs):
    with pytest.raises(ValueError):
        SMNRPMDDynamics(MatrixPotential(np.eye(2)), [1], 2, 1, .01, native=False, **kwargs)
