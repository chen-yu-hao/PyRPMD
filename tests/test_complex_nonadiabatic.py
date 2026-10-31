"""Complex SOC checks against independent exponentials and Hamilton equations."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from pyrpmd.dynamics import State
from pyrpmd.nonadiabatic import (
    DiabaticResult, DiagonalPotential, NRPMDDynamics, NRPMDState,
    validate_diabatic_result, _mint,
)
from pyrpmd.potentials.base import PotentialError
from pyrpmd.potentials.diabatic import DiabaticPythonPotential, check_diabatic_potential


class ComplexMatrix:
    def __init__(self, base, gradient=None, curvature=0.0):
        self.base = np.asarray(base, dtype=complex)
        self.gradient = np.zeros_like(self.base) if gradient is None else np.asarray(gradient, dtype=complex)
        self.curvature = curvature

    def evaluate(self, q):
        s = len(self.base)
        eye = np.eye(s)
        matrix = self.base[None] + q[:, 0, 0, None, None]*self.gradient
        matrix += .5*self.curvature*np.sum(q*q, axis=(1, 2))[:, None, None]*eye
        gradients = self.curvature*q[:, None, None]*eye[None, :, :, None, None]
        gradients = np.broadcast_to(gradients, (*matrix.shape, *q.shape[1:])).astype(complex).copy()
        gradients[..., 0, 0] += self.gradient
        return DiabaticResult(matrix, gradients)


def make_state(beads=2, states=3):
    rng = np.random.default_rng(921)
    return NRPMDState.from_nuclear(
        State(rng.normal(size=(beads, 2, 3)), rng.normal(size=(beads, 2, 3))),
        states, 0, rng,
    )


def hermitian(rng, states):
    a = rng.normal(size=(states, states)) + 1j*rng.normal(size=(states, states))
    return .5*(a+a.conj().T)


@pytest.mark.parametrize("states", [1, 2, 4, 8])
@pytest.mark.parametrize("duration", [0., .43, -.43])
def test_soc_mint_matches_matrix_exponential_force_quadrature(states, duration):
    rng = np.random.default_rng(5)
    s = make_state(states=states)
    pot = ComplexMatrix(hermitian(rng, states), hermitian(rng, states), .13)
    r = validate_diabatic_result(pot.evaluate(s.q), s.q.shape)
    z0 = s.mapping_q + 1j*s.mapping_p
    expected_p = s.p.copy()
    nodes, weights = np.polynomial.legendre.leggauss(40)
    expected_z = []
    for b in range(len(s.q)):
        expected_z.append(expm(-1j*duration*r.matrix[b]) @ z0[b])
        for node, weight in zip(nodes, weights):
            time = .5*duration*(node+1)
            z = expm(-1j*time*r.matrix[b]) @ z0[b]
            gradient = np.einsum("i,ijac,j->ac", z.conj(), r.gradients[b], z).real
            trace = np.trace(r.gradients[b], axis1=0, axis2=1).real
            expected_p[b] -= .25*duration*weight*(gradient-trace)
    _mint(s, r, duration)
    np.testing.assert_allclose(s.p, expected_p, atol=3e-13)
    np.testing.assert_allclose(s.mapping_q+1j*s.mapping_p, expected_z, atol=3e-13)


def test_complex_degenerate_eigenbasis_is_invariant():
    rng = np.random.default_rng(92)
    u, _ = np.linalg.qr(rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3)))
    energies = np.array([.3, .3, -.7])
    pot = ComplexMatrix((u*energies)@u.conj().T, hermitian(rng, 3))
    s = make_state()
    s.q[:] = 0
    r = pot.evaluate(s.q)
    before = s.copy()
    _mint(s, r, .41)
    _mint(s, r, -.41)
    for key in ("q", "p", "mapping_q", "mapping_p"):
        np.testing.assert_allclose(getattr(s, key), getattr(before, key), atol=2e-14)


def test_complex_constant_unitary_basis_covariance_of_force_energy_and_flow():
    rng = np.random.default_rng(72)
    u, _ = np.linalg.qr(rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3)))
    a, g = hermitian(rng, 3), hermitian(rng, 3)
    d1 = NRPMDDynamics(ComplexMatrix(a, g, .3), [1.7, 2.4], 4., 2, .008, 3, native=False)
    d2 = NRPMDDynamics(ComplexMatrix(u.conj().T@a@u, u.conj().T@g@u, .3),
                      [1.7, 2.4], 4., 2, .008, 3, native=False)
    s1, s2 = make_state(), make_state()
    z = (s2.mapping_q+1j*s2.mapping_p)@u.conj()
    s2.mapping_q[:], s2.mapping_p[:] = z.real, z.imag
    assert d1.energy(s1) == pytest.approx(d2.energy(s2), abs=1e-14)
    for _ in range(20):
        d1.step(s1)
        d2.step(s2)
    np.testing.assert_allclose(s1.q, s2.q, atol=2e-14)
    np.testing.assert_allclose(s1.p, s2.p, atol=1e-13)
    np.testing.assert_allclose((s1.mapping_q+1j*s1.mapping_p)@u.conj(),
                               s2.mapping_q+1j*s2.mapping_p, atol=1e-13)


def test_soc_hamiltonian_contains_imaginary_cross_term_and_force():
    # A purely imaginary Hermitian off-diagonal has a nonzero force for x=(1,0), y=(0,1).
    pot = ComplexMatrix([[0, 2j], [-2j, 0]], [[0, .7j], [-.7j, 0]])
    s = NRPMDState(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)),
                   mapping_q=[[1., 0.]], mapping_p=[[0., 1.]])
    d = NRPMDDynamics(pot, [2.], 4., 1, .01, 2, native=False)
    assert d.energy(s) == pytest.approx(-2.)
    r = pot.evaluate(s.q)
    _mint(s, r, .05)
    # The state is an eigenstate of V and dV, so its force is constant.
    assert s.p[0, 0, 0] == pytest.approx(.05*.7)


def test_full_soc_dynamics_has_second_order_error_against_hamiltonian_ode():
    pot = ComplexMatrix([[.1, .07+.13j], [.07-.13j, -.1]],
                        [[.2, .03-.06j], [.03+.06j, -.2]], .3)
    s = NRPMDState.from_nuclear(State([[[.4, -.1, .2]]], [[[.1, -.3, .5]]]), 2, 0, 56)
    def pack(t):
        return np.concatenate([t.q.ravel(), t.p.ravel(), t.mapping_q.ravel(), t.mapping_p.ravel()])
    def rhs(t, y):
        q, p, z = y[:3], y[3:6], y[6:8]+1j*y[8:]
        r = pot.evaluate(q.reshape(1, 1, 3))
        force = -.5*(np.einsum("i,ijac,j->ac", z.conj(), r.gradients[0], z)
                      - np.trace(r.gradients[0], axis1=0, axis2=1)).real
        dz = -1j*r.matrix[0]@z
        return np.r_[p/1.7, force.ravel(), dz.real, dz.imag]
    exact = solve_ivp(rhs, (0., .8), pack(s), method="DOP853", rtol=1e-12, atol=1e-13).y[:, -1]
    errors = []
    for dt in [.08, .04, .02]:
        d = NRPMDDynamics(pot, [1.7], 4., 1, dt, 2, native=False)
        trial = s.copy()
        for _ in range(round(.8/dt)):
            d.step(trial)
        errors.append(np.linalg.norm(pack(trial)-exact))
    assert 3.9 < errors[0]/errors[1] < 4.1
    assert 3.9 < errors[1]/errors[2] < 4.1


def test_soc_forward_then_negative_time_flow_and_symplectic_jacobian():
    pot = ComplexMatrix([[.1, .07+.13j], [.07-.13j, -.1]],
                        [[.2, .03-.06j], [.03+.06j, -.2]], .3)
    s = NRPMDState.from_nuclear(State([[[.4, -.1, .2]]], [[[.1, -.3, .5]]]), 2, 0, 56)
    def pack(t):
        return np.concatenate([t.q.ravel(), t.p.ravel(), t.mapping_q.ravel(), t.mapping_p.ravel()])
    forward = NRPMDDynamics(pot, [1.7], 4., 1, .09, 2, native=False)
    backward = NRPMDDynamics(pot, [1.7], 4., 1, -.09, 2, native=False)
    original = pack(s)
    for _ in range(10): forward.step(s)
    for _ in range(10): backward.step(s)
    np.testing.assert_allclose(pack(s), original, atol=2e-14)
    def advance(y):
        t = NRPMDState(y[:3].reshape(1, 1, 3), y[3:6].reshape(1, 1, 3),
                       mapping_q=y[6:8].reshape(1, 2), mapping_p=y[8:].reshape(1, 2))
        forward.step(t)
        return pack(t)
    eps = 2e-6
    jac = np.column_stack([(advance(original+eps*e)-advance(original-eps*e))/(2*eps) for e in np.eye(10)])
    omega = np.zeros((10, 10))
    for i, j in [(0, 3), (1, 4), (2, 5), (6, 8), (7, 9)]:
        omega[i, j], omega[j, i] = 1., -1.
    np.testing.assert_allclose(jac.T@omega@jac, omega, atol=6e-10)


def test_soc_adapter_preserves_units_and_complex_finite_difference_norm():
    p = DiabaticPythonPotential(ComplexMatrix([[.1, .2j], [-.2j, -.1]],
                                              [[.3, .4j], [-.4j, -.2]], .7),
                                units={"length": "angstrom", "energy": "eV"})
    q = np.arange(6.).reshape(1, 2, 3)/10
    r = p.evaluate(q)
    assert np.iscomplexobj(r.matrix) and np.iscomplexobj(r.gradients)
    check = check_diabatic_potential(p, q, 2)
    assert 0 <= check["rms_gradient_error"] < 1e-11
    diagonal = DiagonalPotential(p, 1, 2).evaluate(q)
    assert diagonal.energy.dtype.kind == diagonal.forces.dtype.kind == "f"
    np.testing.assert_allclose(diagonal.energy, r.matrix[:, 1, 1].real)
    np.testing.assert_allclose(diagonal.forces, -r.gradients[:, 1, 1].real)


@pytest.mark.parametrize("defect", ["nonhermitian_matrix", "nonhermitian_gradient", "imaginary_diagonal", "nonfinite"])
def test_soc_contract_rejects_invalid_complex_values(defect):
    q = np.zeros((1, 1, 3))
    r = ComplexMatrix([[0, .1j], [-.1j, 0]]).evaluate(q)
    if defect == "nonhermitian_matrix": r.matrix[0, 1, 0] = .1j
    if defect == "nonhermitian_gradient": r.gradients[0, 1, 0, 0, 0] = .1j
    if defect == "imaginary_diagonal": r.matrix[0, 0, 0] = .1j
    if defect == "nonfinite": r.gradients[0, 0, 0, 0, 0] = np.nan*1j
    with pytest.raises(PotentialError):
        validate_diabatic_result(r, q.shape)


def test_nrpmd_clock_overflow_cannot_commit_a_step():
    s = NRPMDState.from_nuclear(State(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), time=1e308), 1, 0, 2)
    d = NRPMDDynamics(ComplexMatrix([[0.]]), [2.], 4., 1, 1e308, 1, native=False)
    before = s.copy()
    with pytest.raises(ValueError, match="clock"): d.step(s)
    for key in ("q", "p", "mapping_q", "mapping_p"):
        np.testing.assert_array_equal(getattr(s, key), getattr(before, key))
    assert (s.step, s.time) == (before.step, before.time)
