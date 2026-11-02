"""Independent references for the 2023 Runeson--Manolopoulos MASH variant."""
import copy

import numpy as np
import pytest
from scipy.linalg import expm, expm_frechet
from scipy.optimize import brentq

from pyrpmd.adiabatic import AdiabaticPythonPotential, AdiabaticResult
from pyrpmd.dynamics import State
from pyrpmd.mash import (MASHDynamics, MultistateMASHDynamics, MultistateMASHState,
                         multistate_mash_parameters)
from pyrpmd.nonadiabatic import DiabaticResult
from pyrpmd.potentials.base import PotentialError


class Constant:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)

    def evaluate(self, q):
        s = len(self.matrix)
        return DiabaticResult(np.broadcast_to(self.matrix, (len(q), s, s)).copy(),
                              np.zeros((len(q), s, s, *q.shape[1:])))


class Rotating:
    """Exact U(q) E U(q)^T and its matrix-exponential Frechet derivatives."""
    def __init__(self, energies=(0., .002), generators=None, basis=None):
        self.energies = np.array(energies)
        s = len(energies)
        if generators is None:
            generators = np.zeros((1, 3, s, s))
            generators[0, 0, 0, 1], generators[0, 0, 1, 0] = -1., 1.
        self.generators = np.asarray(generators)
        self.basis = np.eye(s) if basis is None else np.asarray(basis)

    def vectors(self, q):
        return self.basis @ expm(np.einsum("ac,acij->ij", q, self.generators))

    def evaluate(self, q):
        s, natoms = len(self.energies), q.shape[1]
        matrices = np.empty((len(q), s, s))
        gradients = np.zeros((len(q), s, s, natoms, 3))
        diagonal = np.diag(self.energies)
        for b, geometry in enumerate(q):
            argument = np.einsum("ac,acij->ij", geometry, self.generators)
            u = expm(argument)
            matrices[b] = self.basis @ u @ diagonal @ u.T @ self.basis.T
            for a, c in np.ndindex(natoms, 3):
                if np.any(self.generators[a, c]):
                    du = expm_frechet(argument, self.generators[a, c], compute_expm=False)
                    derivative = du @ diagonal @ u.T + u @ diagonal @ du.T
                    gradients[b, :, :, a, c] = self.basis @ derivative @ self.basis.T
        return DiabaticResult(matrices, gradients)


class AdiabaticRotating:
    """Same two-state model as Rotating, through the old MASH interface."""
    def __init__(self, gap):
        self.gap = gap

    def evaluate(self, q, *, previous_q=None, previous_context=None):
        x = q[:, 0, 0]
        u = np.empty((len(q), 2, 2))
        u[:, 0, 0] = u[:, 1, 1] = np.cos(x)
        u[:, 0, 1], u[:, 1, 0] = -np.sin(x), np.sin(x)
        gradient = np.zeros((len(q), 2, 1, 3))
        coupling = np.zeros((len(q), 2, 2, 1, 3))
        coupling[:, 0, 1, 0, 0], coupling[:, 1, 0, 0, 0] = -1., 1.
        overlaps = None if previous_context is None else previous_context.swapaxes(1, 2) @ u
        return AdiabaticResult(np.tile([0., self.gap], (len(q), 1)), gradient,
                               coupling, overlaps, context=u)


def nuclear(momentum=.2, natoms=1):
    q = np.zeros((1, natoms, 3))
    p = np.zeros_like(q)
    p[0, 0, 0] = momentum
    return State(q, p)


def incoming(driver, populations=(.55, .45)):
    c = np.sqrt(populations).astype(complex)
    c[1] *= -1
    return driver.initialize(nuclear(), coefficients=c)


@pytest.mark.parametrize("gap,accepted", [(.002, True), (.04, False)])
def test_resolved_two_state_event_matches_piecewise_exact_electronics(gap, accepted):
    driver = MultistateMASHDynamics(Rotating((0., gap)), [1.], 2., 1, .5, 2,
                                    rtol=1e-11, atol=1e-13)
    state = incoming(driver)
    c0 = state.coefficients.copy()
    e0 = driver.energy(state)
    coupling = np.array([[0., -1.], [1., 0.]])

    def propagate(c, velocity, duration):
        return expm((-1j * np.diag([0., gap]) - velocity * coupling) * duration) @ c

    def boundary(t):
        c = propagate(c0, .2, t)
        return abs(c[0]) ** 2 - abs(c[1]) ** 2

    event_time = brentq(boundary, 0., .5, xtol=1e-14)
    at_event = propagate(c0, .2, event_time)
    momentum = np.sqrt(.2 ** 2 - 2 * gap) if accepted else -.2
    q_expected = .2 * event_time + momentum * (.5 - event_time)
    c_ad = propagate(at_event, momentum, .5 - event_time)
    c_expected = expm(coupling * q_expected) @ c_ad
    driver.step(state)
    assert len(driver.last_hops) == 1
    hop = driver.last_hop
    assert hop.accepted == accepted and hop.frustrated != accepted
    assert hop.time == pytest.approx(event_time, abs=1e-10)
    assert state.active_surface == int(accepted)
    np.testing.assert_allclose(state.q[0, 0, 0], q_expected, atol=2e-10)
    np.testing.assert_allclose(state.p[0, 0, 0], momentum, atol=2e-12)
    np.testing.assert_allclose(state.coefficients, c_expected, atol=2e-10)
    assert driver.energy(state) == pytest.approx(e0, abs=2e-12)
    assert abs(hop.energy_error) < 1e-15


@pytest.mark.parametrize("gap", [.002, .04])
def test_two_state_limit_converges_to_existing_mash_same_coefficients(gap):
    """Compare equations of motion, not the deliberately different estimators."""
    reference = MultistateMASHDynamics(Rotating((0., gap)), [1], 2, 1, .5, 2,
                                       rtol=1e-11, atol=1e-13)
    exact = incoming(reference)
    reference.step(exact)
    c_ad_exact = Rotating((0., gap)).vectors(exact.q[0]).T @ exact.coefficients
    errors = []
    for steps in (10, 80):
        old = MASHDynamics(AdiabaticPythonPotential(AdiabaticRotating(gap)), [1], 2, 1,
                           .5 / steps, native=False)
        state = old.initialize(nuclear(), adiabatic_coefficients=[np.sqrt(.55), -np.sqrt(.45)])
        for _ in range(steps):
            old.step(state)
        assert state.active_surface == exact.active_surface
        np.testing.assert_allclose(state.p, exact.p, atol=2e-12)
        assert old.energy(state) == pytest.approx(reference.energy(exact), abs=2e-12)
        errors.append(np.linalg.norm(state.q-exact.q)
                      + np.linalg.norm(np.outer(state.coefficients, state.coefficients.conj())
                                       - np.outer(c_ad_exact, c_ad_exact.conj())))
    assert errors[1] < errors[0] / 3
    assert errors[1] < .004


def test_generalized_boundary_direction_matches_population_finite_difference():
    generators = np.zeros((2, 3, 3, 3))
    generators[0, 0, 0, 2], generators[0, 0, 2, 0] = -1., 1.
    generators[1, 1, 1, 2], generators[1, 1, 2, 1] = -.8, .8
    model = Rotating((-.2, .1, .7), generators)
    driver = MultistateMASHDynamics(model, [2., 7.], 2., 1, .1, 3)
    q = np.zeros((1, 2, 3))
    c = np.sqrt([.4, .4, .2]) * np.exp(1j * np.array([0., .4, -.3]))
    direction, _ = driver._boundary_direction(q, c, 0, 1)
    numerical = np.zeros((2, 3))
    for atom, axis in np.ndindex(2, 3):
        plus, minus = q.copy(), q.copy()
        plus[0, atom, axis], minus[0, atom, axis] = 1e-5, -1e-5
        values = []
        for geometry in (plus, minus):
            adiabatic = model.vectors(geometry[0]).T @ c
            values.append(abs(adiabatic[0]) ** 2 - abs(adiabatic[1]) ** 2)
        numerical[atom, axis] = (values[0]-values[1]) / (2e-5)
    np.testing.assert_allclose(2 * direction, numerical, atol=2e-10)
    # The pair NAC d_01 is exactly zero, yet the E3 direction is nonzero.
    # Replacing E3 with pair-NAC rescaling would fail this independent check.
    assert np.linalg.norm(direction) > .1
    assert np.all(generators[:, :, 0, 1] == 0.)


@pytest.mark.parametrize("scale", [4., .01])
def test_three_state_mass_weighted_impulse_preserves_energy_and_tangent_momentum(scale):
    generators = np.zeros((2, 3, 3, 3))
    generators[0, 0, 0, 2], generators[0, 0, 2, 0] = -1., 1.
    generators[1, 1, 1, 2], generators[1, 1, 2, 1] = -.8, .8
    driver = MultistateMASHDynamics(Rotating((-.2, .1, .7), generators), [2., 7.], 2., 1, .1, 3)
    q = np.zeros((1, 2, 3))
    c = np.sqrt([.4, .4, .2]) * np.exp(1j * np.array([0., .4, -.3]))
    direction, _ = driver._boundary_direction(q, c, 0, 1)
    p = scale * np.array([[[2., .3, .1], [.2, -3., .4]]])
    before = p.copy()
    updated, hop = driver._impulse(q, p, c, 0, 1, .05)
    np.testing.assert_array_equal(p, before)
    assert hop.accepted == (scale > 1)
    assert abs(hop.energy_error) < 1e-14
    mass_root = np.sqrt(driver.masses[:, None])
    normal = direction / mass_root
    normal /= np.linalg.norm(normal)
    old_scaled, new_scaled = p[0] / mass_root, updated[0] / mass_root
    old_projection = np.sum(old_scaled * normal)
    new_projection = np.sum(new_scaled * normal)
    np.testing.assert_allclose(old_scaled - old_projection * normal,
                               new_scaled - new_projection * normal, atol=2e-15)
    if hop.frustrated:
        assert new_projection == pytest.approx(-old_projection, abs=2e-15)
    else:
        assert old_projection * new_projection > 0


@pytest.mark.parametrize("states", [2, 3, 7])
def test_initial_uniform_region_sampling_and_correct_observable_weights(states):
    driver = MultistateMASHDynamics(Constant(np.diag(np.arange(states))), [1.], 2., 1, .1,
                                    states, seed=819)
    parent = nuclear(0.)
    samples = [driver.initialize(parent, states - 1) for _ in range(1800)]
    rho = np.mean([driver.density_matrix(sample) for sample in samples], axis=0)
    expected = np.diag(np.eye(states)[-1])
    np.testing.assert_allclose(rho, expected, atol=.06)
    mean_maximum = np.mean([abs(sample.coefficients[-1]) ** 2 for sample in samples])
    harmonic = sum(1. / k for k in range(1, states + 1))
    assert mean_maximum == pytest.approx(harmonic / states, abs=.015)
    assert all(sample.active_surface == states - 1 for sample in samples)
    alpha, offset = multistate_mash_parameters(states)
    assert alpha + states * offset == pytest.approx(1.)
    custom = driver.initialize(parent, coefficients=np.eye(states)[0])
    assert np.min(driver.populations(custom)) < 0.
    assert np.trace(driver.density_matrix(custom)) == pytest.approx(1.)
    np.testing.assert_array_equal(driver.active_populations(custom), np.eye(states)[0])


def test_constant_three_state_electronic_evolution_and_density_are_exact():
    matrix = np.array([[.2, .3, -.1], [.3, -.4, .2], [-.1, .2, .7]])
    driver = MultistateMASHDynamics(Constant(matrix), [1.4], 2., 1, .23, 3,
                                    rtol=1e-11, atol=1e-13)
    state = driver.initialize(nuclear(.2), 1, basis="diabatic", rng=721)
    before = state.copy()
    rho0 = driver.density_matrix(state, basis="diabatic")
    for _ in range(4):
        driver.step(state)
        assert driver.last_hops == ()
    unitary = expm(-1j * matrix * .92)
    np.testing.assert_allclose(state.coefficients, unitary @ before.coefficients, atol=3e-12)
    np.testing.assert_allclose(driver.density_matrix(state, basis="diabatic"),
                               unitary @ rho0 @ unitary.conj().T, atol=6e-12)
    np.testing.assert_allclose(state.q, before.q + .92 * before.p / 1.4, atol=3e-13)
    assert driver.electronic_norm(state) == pytest.approx(1., abs=2e-13)


def test_real_basis_rotation_preserves_observables_and_resolved_events():
    rng = np.random.default_rng(724)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    one = MultistateMASHDynamics(Rotating((0., .002, .01)), [1.], 2., 1, .5, 3,
                                 rtol=1e-11, atol=1e-13)
    two = MultistateMASHDynamics(Rotating((0., .002, .01), basis=orthogonal), [1.], 2., 1, .5, 3,
                                 rtol=1e-11, atol=1e-13)
    first = incoming(one, (.45, .4, .15))
    second = two.initialize(nuclear(), coefficients=orthogonal @ first.coefficients, basis="diabatic")
    one.step(first)
    two.step(second)
    assert len(one.last_hops) == len(two.last_hops) == 1
    assert first.active_surface == second.active_surface == 1
    np.testing.assert_allclose(first.q, second.q, atol=1e-11)
    np.testing.assert_allclose(first.p, second.p, atol=1e-11)
    np.testing.assert_allclose(two.density_matrix(second, basis="diabatic"),
                               orthogonal @ one.density_matrix(first, basis="diabatic") @ orthogonal.T,
                               atol=1e-10)


def test_rng_replay_and_plain_state_reconstruction_resume_exactly():
    model = Rotating((0., .002, .01))
    one = MultistateMASHDynamics(model, [1.], 2., 1, .2, 3, seed=28)
    rng_state = copy.deepcopy(one.rng.bit_generator.state)
    first = one.initialize(nuclear())
    one.rng.bit_generator.state = rng_state
    replay = one.initialize(nuclear())
    np.testing.assert_array_equal(first.coefficients, replay.coefficients)
    one.step(first)
    two = MultistateMASHDynamics(model, [1.], 2., 1, .2, 3)
    restored = MultistateMASHState(first.q, first.p, first.step, first.time,
                                  first.coefficients, first.active_surface, first.sampling)
    expected_rng = copy.deepcopy(one.rng.bit_generator.state)
    for _ in range(3):
        one.step(first)
        two.step(restored)
    for name in ("q", "p", "coefficients"):
        np.testing.assert_array_equal(getattr(first, name), getattr(restored, name))
    assert first.step == restored.step and first.time == restored.time
    assert first.active_surface == restored.active_surface
    assert one.rng.bit_generator.state == expected_rng


def test_failed_step_rolls_back_state_rng_and_hop_diagnostics():
    class Failing(Rotating):
        def evaluate(self, q):
            if np.max(q) > .01:
                raise PotentialError("intentional provider failure after partial integration")
            return super().evaluate(q)
    driver = MultistateMASHDynamics(Failing((0., .002, .01)), [1.], 2., 1, .5, 3, seed=29)
    state = incoming(driver, (.45, .4, .15))
    before = state.copy()
    rng = copy.deepcopy(driver.rng.bit_generator.state)
    with pytest.raises(PotentialError, match="intentional provider failure"):
        driver.step(state)
    for name in ("q", "p", "coefficients"):
        np.testing.assert_array_equal(getattr(state, name), getattr(before, name))
    assert state.step == before.step and state.time == before.time
    assert state.active_surface == before.active_surface
    assert rng == driver.rng.bit_generator.state
    assert driver.last_hops == () and driver.last_hop is None


def test_multiple_resolved_events_and_event_limit_failure_are_atomic():
    model = Rotating((0., .002, .01))
    kwargs = dict(max_step=.05, rtol=1e-10, atol=1e-12)
    full = MultistateMASHDynamics(model, [1.], 2., 1, 2., 3, **kwargs)
    state = full.initialize(nuclear(2.), coefficients=[np.sqrt(.45), -np.sqrt(.4), np.sqrt(.15)])
    before = state.copy()
    energy = full.energy(state)
    full.step(state)
    assert len(full.last_hops) == 3
    assert [hop.target for hop in full.last_hops] == [1, 0, 1]
    assert all(hop.accepted for hop in full.last_hops)
    assert np.all(np.diff([hop.time for hop in full.last_hops]) > 0)
    assert full.energy(state) == pytest.approx(energy, abs=2e-11)
    limited = MultistateMASHDynamics(model, [1.], 2., 1, 2., 3, max_events=1, **kwargs)
    failed = before.copy()
    with pytest.raises(ValueError, match="too many multistate MASH events"):
        limited.step(failed)
    for name in ("q", "p", "coefficients"):
        np.testing.assert_array_equal(getattr(failed, name), getattr(before, name))
    assert failed.step == before.step and failed.time == before.time
    assert failed.active_surface == before.active_surface
    assert limited.last_hops == ()


def test_stationary_population_boundaries_do_not_create_false_events():
    driver = MultistateMASHDynamics(Constant(np.diag([0., .3, .8])), [1.], 2., 1, .4, 3,
                                    rtol=1e-11, atol=1e-13)
    initial = np.ones(3, dtype=complex) / np.sqrt(3.)
    state = driver.initialize(nuclear(0.), coefficients=initial)
    driver.step(state)
    assert driver.last_hops == ()
    np.testing.assert_allclose(state.coefficients, initial * np.exp(-1j * np.array([0., .3, .8]) * .4),
                               atol=2e-13)


def test_exact_frame_cache_reduces_provider_calls_without_changing_trajectory():
    class Counted(Rotating):
        calls = 0
        def evaluate(self, q):
            self.calls += 1
            return super().evaluate(q)

    class Uncached(MultistateMASHDynamics):
        def _frame(self, q):
            enabled = hasattr(self, "_frame_cache")
            saved = getattr(self, "_frame_cache", None)
            if enabled:
                del self._frame_cache
            try:
                return super()._frame(q)
            finally:
                if enabled:
                    self._frame_cache = saved

    cached_model, direct_model = Counted((0., .002, .01)), Counted((0., .002, .01))
    cached = MultistateMASHDynamics(cached_model, [1.], 2., 1, .5, 3)
    direct = Uncached(direct_model, [1.], 2., 1, .5, 3)
    one, two = incoming(cached, (.45, .4, .15)), incoming(direct, (.45, .4, .15))
    cached.step(one)
    direct.step(two)
    for name in ("q", "p", "coefficients"):
        np.testing.assert_array_equal(getattr(one, name), getattr(two, name))
    assert cached.last_hops == direct.last_hops
    assert cached_model.calls < direct_model.calls
    assert not hasattr(cached, "_frame_cache")


def test_explicitly_rejects_unsupported_options_and_inconsistent_states():
    potential = Constant(np.diag([0., 1., 2.]))
    with pytest.raises(ValueError, match="P=1"):
        MultistateMASHDynamics(potential, [1.], 2., 2, .1, 3)
    with pytest.raises(ValueError, match="native"):
        MultistateMASHDynamics(potential, [1.], 2., 1, .1, 3, native=True)
    with pytest.raises(ValueError, match="double-precision minimum"):
        MultistateMASHDynamics(potential, [1.], 2., 1, .1, 3, rtol=1e-16)
    with pytest.raises(ValueError, match="at least two"):
        multistate_mash_parameters(1)
    with pytest.raises(ValueError, match="integer"):
        multistate_mash_parameters(True)
    driver = MultistateMASHDynamics(potential, [1.], 2., 1, .1, 3)
    with pytest.raises(ValueError, match="normalized"):
        driver.initialize(nuclear(), coefficients=[1., 1., 0.])
    with pytest.raises(ValueError, match="basis"):
        driver.initialize(nuclear(), basis="other")
    state = driver.initialize(nuclear(), coefficients=[1., 0., 0.])
    state.active_surface = 1
    with pytest.raises(ValueError, match="largest adiabatic"):
        driver.step(state)
    with pytest.raises(ValueError, match="resolved population"):
        driver.attempt_hop(state, 2)
    driver.dt = .2
    with pytest.raises(ValueError, match="immutable"):
        driver.initialize(nuclear())
    with pytest.raises(PotentialError, match="nondegenerate"):
        MultistateMASHDynamics(Constant(np.diag([0., 0., 1.])), [1.], 2., 1, .1, 3).initialize(nuclear())
    soc = np.diag([0., 1., 2.]).astype(complex)
    soc[0, 1], soc[1, 0] = 1e-15j, -1e-15j
    with pytest.raises(PotentialError, match="complex SOC"):
        MultistateMASHDynamics(Constant(soc), [1.], 2., 1, .1, 3).initialize(nuclear())
