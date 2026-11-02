"""CA-RP-MASH event, estimator and gauge checks against independent references."""
import copy

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm
from scipy.optimize import brentq

from pyrpmd.dynamics import State
from pyrpmd.nonadiabatic import DiabaticResult
from pyrpmd.potentials.base import PotentialError
from pyrpmd.rp_mash import RPMASHDynamics, RPMASHState


PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


class RotatingSurfaces:
    """Constant eigenvalues +/-g with an angle linear in nuclear coordinates."""
    def __init__(self, gap=0.2, direction=None, basis=None):
        self.gap = gap
        self.direction = np.array([[1., 0, 0]]) if direction is None else np.array(direction, float)
        self.basis = np.eye(2) if basis is None else np.asarray(basis)

    def evaluate(self, q):
        angle = np.einsum("bac,ac->b", q, self.direction)
        matrix = self.gap * (np.cos(angle)[:, None, None] * PAULI[2] + np.sin(angle)[:, None, None] * PAULI[0])
        derivative = self.gap * (-np.sin(angle)[:, None, None] * PAULI[2] + np.cos(angle)[:, None, None] * PAULI[0])
        gradients = derivative[:, :, :, None, None] * self.direction[None, None, None]
        matrix = self.basis @ matrix @ self.basis.conj().T
        gradients = np.einsum("ij,bjkac,kl->bilac", self.basis, gradients, self.basis.conj().T)
        return DiabaticResult(matrix.real, gradients.real)


class Constant:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)

    def evaluate(self, q):
        return DiabaticResult(np.broadcast_to(self.matrix, (len(q), 2, 2)).copy(),
                              np.zeros((len(q), 2, 2, *q.shape[1:])))


def state(beads=1, p=2., z=-.2):
    q = np.zeros((beads, 1, 3))
    momentum = np.zeros_like(q)
    momentum[:, 0, 0] = p
    return RPMASHState(q, momentum, spin=[np.sqrt(1-z*z), 0, z], active_surface=int(z > 0))


def rotate(spin, omega, time):
    # expm of the cross-product generator, independent of implementation.
    x, y, z = omega
    generator = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return expm(time * generator) @ spin


@pytest.mark.parametrize("beads", [1, 2, 8])
def test_upward_event_matches_piecewise_analytic_mash(beads):
    initial = state(beads)
    before = initial.copy()
    driver = RPMASHDynamics(RotatingSurfaces(), [1], 3, beads, .2, rtol=1e-11, atol=1e-13)
    e0 = driver.energy(initial)
    event_time = brentq(lambda t: rotate(before.spin, [0, -2, .4], t)[2], 0, .2)
    at_event = rotate(before.spin, [0, -2, .4], event_time)
    p_after = np.sqrt(2.**2 - 2*.4)
    expected_ad = rotate(at_event, [0, -p_after, .4], .2-event_time)
    expected_q = 2*event_time + p_after*(.2-event_time)
    expected_fixed = rotate(expected_ad, [0, 1, 0], expected_q)
    driver.step(initial)
    assert initial.active_surface == 1 and len(driver.last_hops) == 1
    hop = driver.last_hops[0]
    assert hop.accepted and not hop.frustrated
    assert abs(hop.time-event_time) < 2e-11
    assert abs(hop.ring_energy_gap - beads*.4) < 1e-14
    np.testing.assert_allclose(initial.q[:, 0, 0], expected_q, atol=2e-11)
    np.testing.assert_allclose(initial.p[:, 0, 0], p_after, atol=2e-11)
    np.testing.assert_allclose(initial.spin, expected_fixed, atol=2e-11)
    assert abs(driver.energy(initial) - e0) < 2e-11
    np.testing.assert_array_equal(driver.populations(initial), [0, 1])


def test_frustrated_hop_reflects_and_remains_in_hemisphere():
    initial = state(p=.4, z=-.05)
    before = initial.copy()
    driver = RPMASHDynamics(RotatingSurfaces(), [1], 3, 1, .2, rtol=1e-11, atol=1e-13)
    e0 = driver.energy(initial)
    event_time = brentq(lambda t: rotate(before.spin, [0, -.4, .4], t)[2], 0, .2)
    at_event = rotate(before.spin, [0, -.4, .4], event_time)
    expected_ad = rotate(at_event, [0, .4, .4], .2-event_time)
    expected_q = .4*event_time - .4*(.2-event_time)
    driver.step(initial)
    assert initial.active_surface == 0 and len(driver.last_hops) == 1
    assert driver.last_hop.frustrated and not driver.last_hop.accepted
    np.testing.assert_allclose(initial.p[0, 0, 0], -.4, atol=2e-12)
    np.testing.assert_allclose(initial.q[0, 0, 0], expected_q, atol=2e-11)
    np.testing.assert_allclose(initial.spin, rotate(expected_ad, [0, 1, 0], expected_q), atol=2e-11)
    assert abs(driver.energy(initial)-e0) < 2e-12


@pytest.mark.parametrize("source,scale", [(0, 1.0), (1, 1.0), (0, .01)])
def test_nonuniform_mass_rescaling_energy_and_mass_weighted_reflection(source, scale):
    direction = np.array([[1., .7, 0], [.2, 2., 0]])
    masses = np.array([2., 7.])
    driver = RPMASHDynamics(RotatingSurfaces(direction=direction), masses, 2, 3, .1)
    q = np.zeros((3, 2, 3))
    p = scale*np.array([[[2., 1., .2], [1., 3., .4]], [[1.5, 1., -.2], [1., 4., .8]], [[2.5, 1., .7], [1., 2., .2]]])
    before = p.copy()
    updated, event = driver._impulse(q, p, source, .04)
    np.testing.assert_array_equal(p, before)
    difference = updated-p
    factor = difference[0, 0, 0]/direction[0, 0]
    np.testing.assert_allclose(difference, np.broadcast_to(factor*direction, p.shape), atol=2e-15)
    kinetic_delta = np.sum((updated**2-p**2)/(2*masses[None, :, None]))
    np.testing.assert_allclose(kinetic_delta + (event.ring_energy_gap if event.accepted else 0), 0, atol=2e-14)
    old_projection = np.sum(p*direction/masses[None, :, None])
    new_projection = np.sum(updated*direction/masses[None, :, None])
    if scale == .01:
        assert event.frustrated
        np.testing.assert_allclose(new_projection, -old_projection, atol=1e-14)
    else:
        assert event.accepted and new_projection*old_projection > 0
    # All beads receive the same shift, preserving every internal momentum.
    np.testing.assert_allclose(updated-updated.mean(axis=0), p-p.mean(axis=0), atol=1e-15)


def test_constant_electronic_matrix_free_ring_and_equator_no_false_events():
    potential = Constant([[.3, .2], [.2, -.1]])
    rng = np.random.default_rng(2)
    nuclear = State(rng.normal(size=(3, 1, 3)), rng.normal(size=(3, 1, 3)))
    driver = RPMASHDynamics(potential, [1.4], 4, 3, .13, rtol=1e-11, atol=1e-13)
    initial = driver.initialize(nuclear, 1)
    rho0 = initial.spin_density_matrix()
    e0 = driver.energy(initial)
    for _ in range(5):
        driver.step(initial)
        assert driver.last_hops == () and initial.active_surface == 1
    u = expm(-1j*np.array(potential.matrix)*.65)
    np.testing.assert_allclose(initial.spin_density_matrix(), u@rho0@u.conj().T, atol=2e-12)
    assert abs(driver.energy(initial)-e0) < 1e-10
    equator = RPMASHState(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), spin=[1., 0, 0], active_surface=0)
    diagonal = RPMASHDynamics(Constant([[.2, 0], [0, -.2]]), [1], 2, 1, .2)
    diagonal.step(equator)
    assert diagonal.last_hops == ()
    np.testing.assert_allclose(equator.spin, [np.cos(.08), np.sin(.08), 0], atol=2e-12)


def test_population_sampling_has_required_importance_weight_and_rng_replay():
    driver = RPMASHDynamics(Constant([[.3, 0], [0, -.3]]), [1], 2, 1, .02, seed=21)
    parent = State(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)))
    rng_before = copy.deepcopy(driver.rng.bit_generator.state)
    first = driver.initialize(parent, 1)
    driver.rng.bit_generator.state = rng_before
    replay = driver.initialize(parent, 1)
    np.testing.assert_array_equal(first.spin, replay.spin)
    samples = [driver.initialize(parent, 1) for _ in range(500)]
    sz = np.array([s.spin[2] for s in samples])
    assert abs(np.mean(sz)-2/3) < 5*np.sqrt((1/18)/500)
    assert abs(np.mean(sz**2)-.5) < 5*np.sqrt((1/12)/500)
    assert all(s.active_surface == 1 for s in samples)
    assert not np.shares_memory(first.q, parent.q)
    snapshot = copy.deepcopy(driver.rng.bit_generator.state)
    driver.step(first)
    assert snapshot == driver.rng.bit_generator.state  # Hopping is deterministic.


def test_operator_pair_weights_reproduce_exact_uncoupled_electronic_trace():
    matrix = np.array([[.2, .3], [.3, -.4]])
    driver = RPMASHDynamics(Constant(matrix), [1], 2, 1, .03)
    a = np.array([[.8, .2+.4j], [.2-.4j, -.1]])
    b = np.array([[.2, -.1+.3j], [-.1-.3j, 1.1]])
    unitary = expm(-1j*matrix*.73)
    exact = np.trace(a@unitary.conj().T@b@unitary).real
    q = np.zeros((1, 1, 3))
    frame, _ = driver._frame(q)
    nodes, weights = leggauss(6)
    integral = 0.
    for sign in (-1, 1):
        for node, weight in zip(nodes, weights):
            sz = sign*(node+1)/2
            for phi in np.arange(12)*2*np.pi/12:
                spin_ad = np.array([np.sqrt(1-sz*sz)*np.cos(phi), np.sqrt(1-sz*sz)*np.sin(phi), sz])
                initial = RPMASHState(q, q, spin=frame[0]@spin_ad, active_surface=int(sz > 0), sampling="uniform-sphere")
                rho = unitary@initial.spin_density_matrix()@unitary.conj().T
                spin = np.einsum("ij,kji->k", rho, PAULI).real
                final = RPMASHState(q, q, time=.73, spin=spin, active_surface=initial.active_surface, sampling="uniform-sphere")
                integral += weight/4/12 * driver.correlation(initial, final, a, b)
    np.testing.assert_allclose(integral, exact, atol=2e-14)
    with pytest.raises(ValueError, match="uniform-sphere"):
        driver.correlation(driver.initialize(State(q, q)), final, a, b)


@pytest.mark.parametrize("unitary", [np.diag([1., -1.]), np.array([[.8, -.6], [.6, .8]])])
def test_real_basis_gauge_invariance_including_event(unitary):
    original = state()
    transformed = original.copy()
    rho = unitary@original.spin_density_matrix()@unitary.T
    transformed.spin[:] = np.einsum("ij,kji->k", rho, PAULI).real
    one = RPMASHDynamics(RotatingSurfaces(), [1], 2, 1, .2, rtol=1e-11, atol=1e-13)
    two = RPMASHDynamics(RotatingSurfaces(basis=unitary), [1], 2, 1, .2, rtol=1e-11, atol=1e-13)
    one.step(original)
    two.step(transformed)
    assert original.active_surface == transformed.active_surface == 1
    np.testing.assert_allclose(original.q, transformed.q, atol=2e-11)
    np.testing.assert_allclose(original.p, transformed.p, atol=2e-11)
    np.testing.assert_allclose(transformed.spin_density_matrix(), unitary@original.spin_density_matrix()@unitary.T, atol=2e-11)


def test_failures_are_transactional_and_do_not_consume_rng():
    class Failing(RotatingSurfaces):
        def evaluate(self, q):
            self.calls = getattr(self, "calls", 0) + 1
            if self.calls == 10:
                raise PotentialError("provider failed during integration")
            return super().evaluate(q)
    initial = state()
    before = initial.copy()
    driver = RPMASHDynamics(Failing(), [1], 2, 1, .2)
    snapshot = copy.deepcopy(driver.rng.bit_generator.state)
    with pytest.raises(PotentialError, match="during"):
        driver.step(initial)
    for key in ("q", "p", "spin"):
        np.testing.assert_array_equal(getattr(initial, key), getattr(before, key))
    assert (initial.step, initial.time, initial.active_surface) == (before.step, before.time, before.active_surface)
    assert driver.last_hops == () and snapshot == driver.rng.bit_generator.state


def test_event_limit_fails_without_partial_state_commit():
    initial = state()
    before = initial.copy()
    driver = RPMASHDynamics(RotatingSurfaces(), [1], 2, 1, 4., max_events=1, max_step=.05)
    with pytest.raises(ValueError, match="too many"):
        driver.step(initial)
    np.testing.assert_array_equal(initial.q, before.q)
    np.testing.assert_array_equal(initial.spin, before.spin)
    assert driver.last_hops == ()


@pytest.mark.parametrize("matrix,error", [([[0, 0], [0, 0]], "nondegenerate"),
                                         ([[0, .1j], [-.1j, 0]], "real symmetric")])
def test_degenerate_and_complex_pes_rejected(matrix, error):
    driver = RPMASHDynamics(Constant(matrix), [1], 2, 1, .1)
    with pytest.raises(PotentialError, match=error):
        driver.initialize(State(np.zeros((1, 1, 3)), np.zeros((1, 1, 3))))


@pytest.mark.parametrize("kwargs", [{"nstates": 3}, {"native": True}, {"rtol": True}, {"atol": 0},
                                   {"max_events": False}, {"max_step": -1}, {"gap_tolerance": np.nan}])
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        RPMASHDynamics(RotatingSurfaces(), [1], 2, 1, .1, **kwargs)


def test_inconsistent_active_surface_rejected():
    initial = state()
    initial.active_surface = 1
    driver = RPMASHDynamics(RotatingSurfaces(), [1], 2, 1, .1)
    with pytest.raises(ValueError, match="inconsistent"):
        driver.step(initial)


def test_deterministic_hop_time_reversal():
    initial = state(2)
    before = initial.copy()
    driver = RPMASHDynamics(RotatingSurfaces(), [1], 2, 2, .2, rtol=1e-11, atol=1e-13)
    driver.step(initial)
    assert initial.active_surface == 1
    initial.p *= -1
    initial.spin[1] *= -1  # Complex conjugation for a real matrix Hamiltonian.
    driver.step(initial)
    assert initial.active_surface == before.active_surface
    np.testing.assert_allclose(initial.q, before.q, atol=3e-11)
    np.testing.assert_allclose(initial.p, -before.p, atol=3e-11)
    np.testing.assert_allclose(initial.spin, before.spin, atol=3e-11)
