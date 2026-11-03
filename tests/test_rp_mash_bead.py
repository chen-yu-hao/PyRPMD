"""Independent overlap, quantum, event and free-ring references for BA-RP-MASH."""
import copy

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import brentq

from pyrpmd.analysis.quantum_reference import SplitOperator1D
from pyrpmd.dynamics import State
from pyrpmd.nonadiabatic import DiabaticResult
from pyrpmd.potentials.base import PotentialError
from pyrpmd.rp_mash import RPMASHDynamics, RPMASHState
from pyrpmd.rp_mash_bead import BARPMASHDynamics, BARPMASHState


PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


class TwistedSurfaces:
    """Known adiabatic eigenvalues with linear or nonlinear rotating vectors."""
    def __init__(self, direction=None, curvature=0., gap_curvature=0., scalar=0., basis=None):
        self.direction = np.array([[1., 0, 0]] if direction is None else direction, float)
        self.curvature, self.gap_curvature, self.scalar = curvature, gap_curvature, scalar
        self.basis = np.eye(2) if basis is None else np.asarray(basis)
        self.batch_shapes = []

    def evaluate(self, q):
        self.batch_shapes.append(q.shape)
        u = np.einsum("bac,ac->b", q, self.direction)
        angle = u + self.curvature*u*u
        radius = .2 + self.gap_curvature*u*u
        field = np.cos(angle)[:, None, None]*PAULI[2] + np.sin(angle)[:, None, None]*PAULI[0]
        tangent = -np.sin(angle)[:, None, None]*PAULI[2] + np.cos(angle)[:, None, None]*PAULI[0]
        matrix = radius[:, None, None]*field + self.scalar*np.sum(q*q, axis=(1, 2))[:, None, None]*np.eye(2)
        derivative = (2*self.gap_curvature*u)[:, None, None]*field + (radius*(1+2*self.curvature*u))[:, None, None]*tangent
        gradients = derivative[:, :, :, None, None]*self.direction[None, None, None]
        gradients = gradients + 2*self.scalar*np.eye(2)[None, :, :, None, None]*q[:, None, None]
        matrix = self.basis @ matrix @ self.basis.conj().T
        gradients = np.einsum("ij,bjkac,kl->bilac", self.basis, gradients, self.basis.conj().T)
        return DiabaticResult(matrix.real, gradients.real)


class Constant:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)

    def evaluate(self, q):
        return DiabaticResult(np.broadcast_to(self.matrix, (len(q), 2, 2)).copy(),
                              np.zeros((len(q), 2, 2, *q.shape[1:])))


def spin_state(beads=1, momentum=2., z=-.2):
    q = np.zeros((beads, 1, 3))
    p = np.zeros_like(q)
    p[:, 0, 0] = momentum
    return BARPMASHState(q, p, spin=[np.sqrt(1-z*z), 0, z], active_surface=int(z > 0))


def rotate(spin, omega, time):
    x, y, z = omega
    return expm(time*np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])) @ spin


def overlap_derivative(potential, q, step=2e-6):
    """Finite-difference *eigenvector overlaps*, without analytic gradients.

    Order (upper, lower) and positive orientation fix the real two-state
    Pauli frame.  Align each displaced column to its central overlap sign.
    """
    _, frame = np.linalg.eigh(potential.evaluate(q).matrix)
    frame = frame[:, :, ::-1].real.copy()
    frame[:, :, 1] *= np.sign(np.linalg.det(frame))[:, None]
    derivative = np.empty_like(q)
    for bead, atom, coordinate in np.ndindex(q.shape):
        displaced = []
        for shift in (step, -step):
            geometry = q.copy()
            geometry[bead, atom, coordinate] += shift
            _, vectors = np.linalg.eigh(potential.evaluate(geometry).matrix[bead])
            vectors = vectors[:, ::-1].real
            vectors *= np.sign(np.sum(frame[bead]*vectors, axis=0))[None, :]
            displaced.append(vectors[:, 1])
        derivative[bead, atom, coordinate] = frame[bead, :, 0] @ (displaced[0]-displaced[1])/(2*step)
    return derivative


def test_bead_generator_and_forces_match_overlap_and_energy_finite_differences():
    model = TwistedSurfaces([[1., .7, .15], [.2, .5, -.1]], curvature=.3, gap_curvature=.04, scalar=.01)
    masses = np.array([2., 7.])
    rng = np.random.default_rng(51)
    q, p = rng.normal(size=(2, 3, 2, 3))
    driver = BARPMASHDynamics(model, masses, 5., 3, .01)
    state = BARPMASHState(q, p, spin=[.6, 0, -.8])
    values = driver._electronic(q)
    reference_nac = overlap_derivative(model, q)
    np.testing.assert_allclose(driver._bead_nac(values), reference_nac, rtol=2e-9, atol=2e-10)
    y = np.concatenate((q.ravel(), p.ravel(), state.spin))
    model.batch_shapes.clear()
    rhs = driver._rhs(0., y, 0)
    assert model.batch_shapes == [q.shape]  # One batch, no centroid evaluation.
    energies = np.linalg.eigvalsh(model.evaluate(q).matrix)
    mean_gap = np.mean(energies[:, 1]-energies[:, 0])
    contraction = np.sum(p*reference_nac/masses[None, :, None])/len(q)
    expected_spin = np.cross([0., 2*contraction, mean_gap], state.spin)
    np.testing.assert_allclose(rhs[-3:], expected_spin, atol=2e-10)
    np.testing.assert_array_equal(rhs[:q.size].reshape(q.shape), p/masses[None, :, None])
    centroid_gap = np.diff(np.linalg.eigvalsh(model.evaluate(q.mean(axis=0, keepdims=True)).matrix))[0, 0]
    mean_matrix_gap = np.diff(np.linalg.eigvalsh(model.evaluate(q).matrix.mean(axis=0)))[0]
    assert abs(mean_gap-centroid_gap) > .01
    assert abs(mean_gap-mean_matrix_gap) > .01

    def reference_energy(positions):
        eigenvalues = np.linalg.eigvalsh(model.evaluate(positions).matrix)
        difference = positions-np.roll(positions, -1, axis=0)
        return eigenvalues[:, 0].sum() + .5*(len(q)/5.)**2*np.sum(masses[None, :, None]*difference**2)

    forces = np.empty_like(q)
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += 2e-6
        minus[index] -= 2e-6
        forces[index] = -(reference_energy(plus)-reference_energy(minus))/4e-6
    np.testing.assert_allclose(rhs[q.size:2*q.size].reshape(q.shape), forces, rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("beads", [1, 2, 8])
@pytest.mark.parametrize("momentum,z,accepted", [(2., -.2, True), (.4, -.05, False)])
def test_resolved_event_matches_piecewise_analytic_solution(beads, momentum, z, accepted):
    state = spin_state(beads, momentum, z)
    initial_spin = state.spin.copy()
    driver = BARPMASHDynamics(TwistedSurfaces(), [1.], 3., beads, .2, rtol=1e-11, atol=1e-13)
    before = driver.energy(state)
    event_time = brentq(lambda t: rotate(initial_spin, [0., -momentum, .4], t)[2], 0., .2)
    at_event = rotate(initial_spin, [0., -momentum, .4], event_time)
    p_after = np.sqrt(momentum**2-.8) if accepted else -momentum
    expected_spin = rotate(at_event, [0., -p_after, .4], .2-event_time)
    expected_q = momentum*event_time+p_after*(.2-event_time)
    driver.step(state)
    assert len(driver.last_hops) == 1
    hop = driver.last_hop
    assert hop.accepted is accepted and hop.frustrated is not accepted
    assert state.active_surface == int(accepted)
    assert abs(hop.time-event_time) < 3e-11
    assert abs(hop.ring_energy_gap-beads*.4) < 2e-14
    assert abs(hop.energy_error) < 2e-14
    np.testing.assert_allclose(state.q[:, 0, 0], expected_q, atol=3e-11)
    np.testing.assert_allclose(state.p[:, 0, 0], p_after, atol=3e-11)
    np.testing.assert_allclose(state.spin, expected_spin, atol=3e-11)
    assert abs(driver.energy(state)-before) < 3e-11


@pytest.mark.parametrize("source,scale", [(0, 5.), (1, 5.), (0, .005)])
def test_nonidentical_bead_impulse_unequal_masses_and_total_gap(source, scale):
    model = TwistedSurfaces([[1., .7, .15], [.2, .5, -.1]], curvature=.3, gap_curvature=.04)
    masses = np.array([2., 7.])
    q = np.random.default_rng(19).normal(size=(3, 2, 3))*.3
    direction = overlap_derivative(model, q)
    p = scale*masses[None, :, None]*direction
    before = p.copy()
    driver = BARPMASHDynamics(model, masses, 5., 3, .02)
    updated, hop = driver._impulse(q, p, source, .01)
    np.testing.assert_array_equal(p, before)
    difference = updated-p
    factor = np.sum(difference*direction)/np.sum(direction**2)
    np.testing.assert_allclose(difference, factor*direction, rtol=2e-8, atol=2e-10)
    energies = np.linalg.eigvalsh(model.evaluate(q).matrix)
    gap = np.sum(energies[:, 1-source]-energies[:, source])
    np.testing.assert_allclose(hop.ring_energy_gap, gap, rtol=0., atol=1e-14)
    kinetic_change = np.sum((updated**2-p**2)/(2*masses[None, :, None]))
    np.testing.assert_allclose(kinetic_change+(gap if hop.accepted else 0.), 0., atol=4e-13)
    old_projection = np.sum(p*direction/masses[None, :, None])
    new_projection = np.sum(updated*direction/masses[None, :, None])
    if scale < .01:
        assert hop.frustrated and not hop.accepted
        np.testing.assert_allclose(new_projection, -old_projection, rtol=2e-9, atol=1e-12)
    else:
        assert hop.accepted and old_projection*new_projection > 0.
    # A nonuniform bead NAC really changes internal momenta, unlike CA.
    assert np.max(np.abs(difference-difference.mean(axis=0))) > 1e-4


@pytest.mark.parametrize("momentum,z", [(2., -.2), (.4, -.05), (-2., .2)])
def test_p1_reduces_to_ca_equations_and_events_after_basis_transform(momentum, z):
    model = TwistedSurfaces(curvature=.15, gap_curvature=.02)
    ba_state = spin_state(momentum=momentum, z=z)
    ba_state.q[0, 0, 0] = .3
    ca = RPMASHDynamics(model, [1.], 4., 1, .2, rtol=1e-11, atol=1e-13, max_step=.01)
    ba = BARPMASHDynamics(model, [1.], 4., 1, .2, rtol=1e-11, atol=1e-13, max_step=.01)
    frame = ca._frame(ba_state.q)[0][0]
    ca_state = RPMASHState(ba_state.q, ba_state.p, spin=frame@ba_state.spin,
                          active_surface=ba_state.active_surface)
    ca.step(ca_state)
    ba.step(ba_state)
    assert len(ca.last_hops) == len(ba.last_hops) == 1
    assert ca_state.active_surface == ba_state.active_surface
    assert ca.last_hop.accepted == ba.last_hop.accepted
    assert abs(ca.last_hop.time-ba.last_hop.time) < 2e-10
    np.testing.assert_allclose(ba_state.q, ca_state.q, atol=3e-10)
    np.testing.assert_allclose(ba_state.p, ca_state.p, atol=3e-10)
    final_frame = ca._frame(ca_state.q)[0][0]
    np.testing.assert_allclose(final_frame@ba_state.spin, ca_state.spin, atol=3e-10)
    assert abs(ba.energy(ba_state)-ca.energy(ca_state)) < 3e-10


def test_constant_matrix_free_ring_matches_exact_normal_mode_generator():
    beads, mass, beta, duration = 4, 1.4, 4., .65
    rng = np.random.default_rng(2)
    nuclear = State(rng.normal(size=(beads, 1, 3)), rng.normal(size=(beads, 1, 3)))
    matrix = np.array([[.3, .2], [.2, -.1]])
    driver = BARPMASHDynamics(Constant(matrix), [mass], beta, beads, duration/5,
                             rtol=1e-11, atol=1e-13)
    state = driver.initialize(nuclear, 1)
    rho0, e0 = state.spin_density_matrix(), driver.energy(state)
    for _ in range(5):
        driver.step(state)
        assert driver.last_hops == ()
    identity = np.eye(beads)
    laplacian = 2*identity-np.roll(identity, 1, axis=0)-np.roll(identity, -1, axis=0)
    generator = np.block([[np.zeros_like(identity), identity/mass],
                          [-mass*(beads/beta)**2*laplacian, np.zeros_like(identity)]])
    expected = expm(duration*generator) @ np.concatenate((nuclear.q[:, 0], nuclear.p[:, 0]))
    np.testing.assert_allclose(state.q[:, 0], expected[:beads], atol=3e-10)
    np.testing.assert_allclose(state.p[:, 0], expected[beads:], atol=3e-10)
    energies = np.linalg.eigvalsh(matrix)[::-1]
    unitary = expm(-1j*np.diag(energies)*duration)
    np.testing.assert_allclose(state.spin_density_matrix(), unitary@rho0@unitary.conj().T, atol=2e-11)
    assert abs(driver.energy(state)-e0) < 3e-9


@pytest.mark.parametrize("beads", [1, 4, 8])
def test_operator_correlation_matches_fourier_exact_electronic_transfer(beads):
    # Constant nuclear-independent H makes the electronic transfer exactly
    # separable, even for a moving finite-mass Fourier wavepacket.  Transform
    # its diabatic population operators once into the known adiabatic frame.
    matrix = np.array([[.2, .3], [.3, -.4]])
    energies, vectors = np.linalg.eigh(matrix)
    vectors = vectors[:, ::-1]
    vectors[:, 1] *= np.sign(np.linalg.det(vectors))
    initial_operator = vectors.T @ np.diag([1., 0.]) @ vectors
    final_operator = vectors.T @ np.diag([0., 1.]) @ vectors
    grid = np.linspace(-10., 10., 128, endpoint=False)
    psi = np.zeros((len(grid), 2), complex)
    psi[:, 0] = np.exp(-grid**2/4 + 1j*.3*grid)
    psi /= np.sqrt((grid[1]-grid[0])*np.sum(abs(psi)**2))
    reference = SplitOperator1D(grid, np.broadcast_to(matrix, (len(grid), 2, 2)), 3., .05)
    exact = reference.propagate(psi, 20, sample_stride=20)
    expected = exact.populations[-1, 1]
    assert expected > .05
    assert np.max(exact.edge_population) < 1e-12
    np.testing.assert_allclose(exact.norm, 1., rtol=0., atol=5e-14)

    driver = BARPMASHDynamics(Constant(matrix), [3.], 4., beads, 1., rtol=1e-11, atol=1e-13)
    q = np.zeros((beads, 1, 3))
    estimate = 0.
    # Two Gauss nodes per hemisphere and four azimuths integrate Eq. 29
    # exactly in this separable limit: no stochastic sampling tolerance.
    for z in np.ravel([-(.5+np.array([-1., 1.])/(2*np.sqrt(3))),
                       .5+np.array([-1., 1.])/(2*np.sqrt(3))]):
        for phi in np.arange(4)*np.pi/2:
            spin = [np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z]
            initial = BARPMASHState(q, q, spin=spin, active_surface=int(z > 0), sampling="uniform-sphere")
            final = initial.copy()
            driver.step(final)
            estimate += driver.correlation(initial, final, initial_operator, final_operator)/16
    np.testing.assert_allclose(estimate, expected, rtol=0., atol=2e-11)


def test_no_centroid_gap_requirement_and_no_centroid_provider_call():
    class LinearCrossing:
        def evaluate(self, q):
            assert len(q) == 2, "BA must not evaluate a centroid electronic frame"
            matrix = np.zeros((len(q), 2, 2))
            matrix[:, 0, 0], matrix[:, 1, 1] = q[:, 0, 0], -q[:, 0, 0]
            gradients = np.zeros((len(q), 2, 2, 1, 3))
            gradients[:, 0, 0, 0, 0], gradients[:, 1, 1, 0, 0] = 1., -1.
            return DiabaticResult(matrix, gradients)

    q = np.zeros((2, 1, 3))
    q[:, 0, 0] = [-1., 1.]
    driver = BARPMASHDynamics(LinearCrossing(), [1.], 20., 2, .01)
    state = driver.initialize(State(q, np.zeros_like(q)), 0)
    driver.step(state)
    assert driver.last_hops == ()
    np.testing.assert_allclose(state.q.mean(axis=0), 0., atol=1e-15)
    np.testing.assert_allclose(driver.populations(state), [1., 0.])


def test_cyclic_bead_permutation_covariance_with_nonuniform_beads():
    model = TwistedSurfaces(curvature=.2, gap_curvature=.01)
    q = np.zeros((4, 1, 3))
    q[:, 0, 0] = [-.2, .1, .05, .3]
    p = np.zeros_like(q)
    p[:, 0, 0] = [2., 2.1, 1.9, 2.2]
    first = BARPMASHState(q, p, spin=[np.sqrt(.96), 0., -.2])
    second = BARPMASHState(np.roll(q, 1, axis=0), np.roll(p, 1, axis=0), spin=first.spin)
    one = BARPMASHDynamics(model, [1.], 20., 4, .2, rtol=1e-11, atol=1e-13)
    two = BARPMASHDynamics(model, [1.], 20., 4, .2, rtol=1e-11, atol=1e-13)
    one.step(first)
    two.step(second)
    assert first.active_surface == second.active_surface == 1
    np.testing.assert_allclose(np.roll(first.q, 1, axis=0), second.q, atol=2e-11)
    np.testing.assert_allclose(np.roll(first.p, 1, axis=0), second.p, atol=2e-11)
    np.testing.assert_allclose(first.spin, second.spin, atol=2e-11)
    assert abs(one.last_hop.time-two.last_hop.time) < 2e-11


def test_equatorial_outgoing_and_stationary_spins_do_not_retrigger_events():
    driver = BARPMASHDynamics(Constant([[.2, 0.], [0., -.2]]), [1.], 2., 1, .2)
    state = BARPMASHState(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), spin=[1., 0., 0.])
    driver.step(state)
    assert driver.last_hops == ()
    np.testing.assert_allclose(state.spin, [np.cos(.08), np.sin(.08), 0.], atol=2e-12)


def test_population_initialization_and_rng_replay_are_owned_and_deterministic():
    driver = BARPMASHDynamics(Constant([[.3, 0.], [0., -.3]]), [1.], 2., 1, .02, seed=21)
    nuclear = State(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)))
    snapshot = copy.deepcopy(driver.rng.bit_generator.state)
    first = driver.initialize(nuclear, 1)
    driver.rng.bit_generator.state = snapshot
    replay = driver.initialize(nuclear, 1)
    np.testing.assert_array_equal(first.spin, replay.spin)
    assert not np.shares_memory(first.q, nuclear.q)
    samples = np.array([driver.initialize(nuclear, 1).spin[2] for _ in range(400)])
    assert abs(samples.mean()-2/3) < 5*np.sqrt((1/18)/400)
    assert abs(np.mean(samples**2)-.5) < 5*np.sqrt((1/12)/400)
    snapshot = copy.deepcopy(driver.rng.bit_generator.state)
    driver.step(first)
    assert snapshot == driver.rng.bit_generator.state


def test_provider_failure_and_event_limit_do_not_commit_partial_state():
    class Failing(TwistedSurfaces):
        def evaluate(self, q):
            if len(self.batch_shapes) == 9:
                raise PotentialError("provider failed during integration")
            return super().evaluate(q)

    for model, dt, limit, message in ((Failing(), .2, 64, "provider failed"),
                                      (TwistedSurfaces(), 4., 1, "too many BA-RP-MASH")):
        state = spin_state()
        before = state.copy()
        driver = BARPMASHDynamics(model, [1.], 2., 1, dt, max_events=limit, max_step=.05)
        snapshot = copy.deepcopy(driver.rng.bit_generator.state)
        with pytest.raises((ValueError, PotentialError), match=message):
            driver.step(state)
        for key in ("q", "p", "spin"):
            np.testing.assert_array_equal(getattr(state, key), getattr(before, key))
        assert (state.step, state.time, state.active_surface) == (before.step, before.time, before.active_surface)
        assert driver.last_hops == () and snapshot == driver.rng.bit_generator.state


@pytest.mark.parametrize("matrix,error", [([[0., 0.], [0., 0.]], "nondegenerate"),
                                         ([[0., 1e-14j], [-1e-14j, 1.]], "real symmetric")])
def test_unsupported_potential_fails_without_rng_consumption(matrix, error):
    driver = BARPMASHDynamics(Constant(matrix), [1.], 2., 1, .1)
    snapshot = copy.deepcopy(driver.rng.bit_generator.state)
    with pytest.raises(PotentialError, match=error):
        driver.initialize(State(np.zeros((1, 1, 3)), np.zeros((1, 1, 3))))
    assert snapshot == driver.rng.bit_generator.state


@pytest.mark.parametrize("kwargs", [{"nstates": 3}, {"native": True}, {"rtol": True}, {"rtol": 1e-20},
                                   {"atol": 0}, {"max_events": False}, {"max_step": -1},
                                   {"gap_tolerance": np.nan}, {"threads": 2}])
def test_constructor_rejects_unsupported_or_invalid_options(kwargs):
    with pytest.raises((ValueError, RuntimeError)):
        BARPMASHDynamics(TwistedSurfaces(), [1.], 2., 1, .1, **kwargs)


def test_state_types_and_unsupported_observable_bases_are_not_interchanged():
    driver = BARPMASHDynamics(TwistedSurfaces(), [1.], 2., 1, .1)
    state = spin_state()
    ca_state = RPMASHState(state.q, state.p, spin=state.spin)
    with pytest.raises(ValueError, match="BARPMASHState"):
        driver.step(ca_state)
    with pytest.raises(ValueError, match="RPMASHState"):
        RPMASHDynamics(TwistedSurfaces(), [1.], 2., 1, .1).step(state)
    uniform = driver.initialize(State(state.q, state.p), sampling="uniform-sphere")
    with pytest.raises(ValueError, match="only shared adiabatic"):
        driver.correlation(uniform, uniform, np.eye(2), np.eye(2), basis="diabatic")
    with pytest.raises(ValueError, match="uniform-sphere"):
        driver.correlation(state, state, np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match="Hermitian"):
        driver.correlation(uniform, uniform, [[0., 1.], [0., 0.]], np.eye(2))


def test_rejects_aliased_state_arrays_boolean_values_and_mutated_masses():
    state = spin_state()
    driver = BARPMASHDynamics(TwistedSurfaces(), [1.], 2., 1, .1)
    state.p = state.q
    with pytest.raises(ValueError, match="share memory"):
        driver.step(state)
    with pytest.raises(ValueError, match="booleans"):
        BARPMASHState(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), spin=[True, 0., 0.])
    with pytest.raises(ValueError, match="booleans"):
        BARPMASHDynamics(TwistedSurfaces(), [True, 2.], 2., 1, .1)
    driver.masses = np.array([2.])
    with pytest.raises(ValueError, match="immutable"):
        driver.step(spin_state())


@pytest.mark.parametrize("driver_class", [RPMASHDynamics, BARPMASHDynamics])
def test_impulse_overflow_cannot_silently_accept_a_zero_energy_change(driver_class):
    driver = driver_class(TwistedSurfaces(), [1e-200], 2., 1, .1)
    q = np.zeros((1, 1, 3))
    p = np.ones_like(q)
    before = p.copy()
    with pytest.raises(ValueError, match="momentum rescaling overflow"):
        driver._impulse(q, p, 0, 0.)
    np.testing.assert_array_equal(p, before)


@pytest.mark.parametrize("sampling", [np.array("population"), np.array(["population"]), ["population"]])
def test_sampling_and_basis_require_scalar_strings(sampling):
    q = np.zeros((1, 1, 3))
    driver = BARPMASHDynamics(TwistedSurfaces(), [1.], 2., 1, .1)
    with pytest.raises(ValueError, match="sampling"):
        BARPMASHState(q, q, spin=[0., 0., -1.], sampling=sampling)
    with pytest.raises(ValueError, match="sampling"):
        driver.initialize(State(q, q), sampling=sampling)
    state = spin_state()
    state.sampling = sampling
    with pytest.raises(ValueError, match="sampling"):
        driver.step(state)
    state = driver.initialize(State(q, q), sampling="uniform-sphere")
    with pytest.raises(ValueError, match="only shared adiabatic"):
        driver.correlation(state, state, np.eye(2), np.eye(2), basis=np.array(["adiabatic"]))


def test_initialization_failure_after_sampling_restores_external_rng():
    class Failing(TwistedSurfaces):
        def evaluate(self, q):
            if self.batch_shapes:
                raise PotentialError("post-sampling validation failure")
            return super().evaluate(q)

    driver = BARPMASHDynamics(Failing(), [1.], 2., 1, .1)
    rng = np.random.default_rng(42)
    snapshot = copy.deepcopy(rng.bit_generator.state)
    q = np.zeros((1, 1, 3))
    with pytest.raises(PotentialError, match="post-sampling"):
        driver.initialize(State(q, q), rng=rng)
    assert snapshot == rng.bit_generator.state


def test_time_reversal_retraces_a_nonuniform_ring_with_an_accepted_event():
    model = TwistedSurfaces(curvature=.2, gap_curvature=.01)
    q = np.zeros((4, 1, 3))
    q[:, 0, 0] = [-.2, .1, .05, .3]
    p = np.zeros_like(q)
    p[:, 0, 0] = [2., 2.1, 1.9, 2.2]
    initial = BARPMASHState(q, p, spin=[np.sqrt(.96), 0., -.2])
    state = initial.copy()
    driver = BARPMASHDynamics(model, [1.], 20., 4, .2,
                             rtol=1e-11, atol=1e-13, max_step=.01)
    driver.step(state)
    assert driver.last_hop.accepted and state.active_surface == 1
    state.p *= -1
    state.spin[1] *= -1
    driver.step(state)
    assert driver.last_hop.accepted and state.active_surface == 0
    np.testing.assert_allclose(state.q, initial.q, atol=3e-10)
    np.testing.assert_allclose(state.p, -initial.p, atol=3e-10)
    np.testing.assert_allclose(state.spin, initial.spin*[1., -1., 1.], atol=3e-10)
