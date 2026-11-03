"""BA-RP-MASH state, sampling, free-ring and transaction contracts."""
import copy

import numpy as np
import pytest
from scipy.linalg import expm

from pyrpmd.dynamics import State
from pyrpmd.nonadiabatic import DiabaticResult
from pyrpmd.potentials.base import PotentialError
from pyrpmd.rp_mash import RPMASHDynamics, RPMASHState
from pyrpmd.rp_mash_ba import RPMASHBADynamics, RPMASHBAState
from test_rp_mash import Constant, RotatingSurfaces


def parent(beads=3):
    q = np.zeros((beads, 1, 3))
    p = np.zeros_like(q)
    p[:, 0, 0] = 2.
    return State(q, p)


def driver(beads=3, potential=None, **kwargs):
    return RPMASHBADynamics(RotatingSurfaces() if potential is None else potential,
                            [1.], 3., beads, .2, **kwargs)


def crossing(beads=3):
    nuclear = parent(beads)
    return RPMASHBAState(nuclear.q, nuclear.p, spin=[np.sqrt(.96), 0., -.2])


def test_state_copy_and_wrong_representation_are_explicit():
    nuclear = parent()
    ba = driver().initialize(nuclear, rng=1)
    cloned = ba.copy()
    assert type(cloned) is RPMASHBAState
    for key in ("q", "p", "spin"):
        np.testing.assert_array_equal(getattr(ba, key), getattr(cloned, key))
        assert not np.shares_memory(getattr(ba, key), getattr(cloned, key))
    assert not np.shares_memory(ba.q, nuclear.q)
    ca = RPMASHDynamics(RotatingSurfaces(), [1], 3, 3, .2)
    with pytest.raises(ValueError, match="RPMASHState"):
        ca.step(ba)
    ca_state = ca.initialize(nuclear, rng=1)
    with pytest.raises(ValueError, match="RPMASHBAState"):
        driver().step(ca_state)


def test_population_importance_sampling_and_rng_replay():
    dynamics = driver(potential=Constant([[.2, 0], [0, -.2]]), seed=351)
    nuclear = parent()
    snapshot = copy.deepcopy(dynamics.rng.bit_generator.state)
    first = dynamics.initialize(nuclear, 1)
    dynamics.rng.bit_generator.state = snapshot
    replay = dynamics.initialize(nuclear, 1)
    np.testing.assert_array_equal(first.spin, replay.spin)
    spins = np.array([dynamics.initialize(nuclear, 1).spin for _ in range(1000)])
    # Density on the selected hemisphere is 2*z, not uniform or a pole.
    assert abs(spins[:, 2].mean() - 2/3) < 5*np.sqrt(1/18/1000)
    assert abs(np.mean(spins[:, 2]**2) - .5) < 5*np.sqrt(1/12/1000)
    np.testing.assert_allclose(spins[:, :2].mean(axis=0), 0., atol=.06)
    assert np.all(spins[:, 2] > 0.)
    before = copy.deepcopy(dynamics.rng.bit_generator.state)
    dynamics.step(first)
    assert before == dynamics.rng.bit_generator.state
    np.testing.assert_array_equal(dynamics.populations(first), [0., 1.])


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_initialization_rolls_back_external_rng_on_failure(error):
    class FailOnValidation(RotatingSurfaces):
        calls = 0

        def evaluate(self, q):
            self.calls += 1
            if self.calls == 2:
                raise error("failed after electronic draws")
            return super().evaluate(q)

    dynamics = driver(potential=FailOnValidation())
    generator = np.random.default_rng(71)
    snapshot = copy.deepcopy(generator.bit_generator.state)
    with pytest.raises(error, match="electronic draws"):
        dynamics.initialize(parent(), rng=generator)
    assert generator.bit_generator.state == snapshot


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_failed_step_preserves_state_rng_and_previous_hop_record(error):
    dynamics = driver()
    state = crossing()
    dynamics.step(state)
    assert dynamics.last_hop.accepted
    before, records = state.copy(), dynamics.last_hops
    snapshot = copy.deepcopy(dynamics.rng.bit_generator.state)

    class FailInODE(RotatingSurfaces):
        calls = 0

        def evaluate(self, q):
            self.calls += 1
            if self.calls == 5:
                raise error("provider failed in ODE")
            return super().evaluate(q)

    dynamics.potential = FailInODE()
    with pytest.raises(error, match="in ODE"):
        dynamics.step(state)
    for key in ("q", "p", "spin"):
        np.testing.assert_array_equal(getattr(state, key), getattr(before, key))
    assert (state.step, state.time, state.active_surface) == (before.step, before.time, before.active_surface)
    assert dynamics.last_hops is records
    assert dynamics.last_hop is records[-1]
    assert dynamics.rng.bit_generator.state == snapshot


def test_event_limit_is_transactional():
    dynamics = RPMASHBADynamics(RotatingSurfaces(), [1], 3, 2, 4.,
                                max_step=.05, max_events=1)
    state = crossing(2)
    before = state.copy()
    with pytest.raises(ValueError, match="too many BA-RP-MASH"):
        dynamics.step(state)
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    np.testing.assert_array_equal(state.spin, before.spin)
    assert dynamics.last_hops == () and state.step == 0


@pytest.mark.parametrize("beads", [1, 2, 7])
def test_constant_matrix_exact_free_ring_and_adiabatic_precession(beads):
    rng = np.random.default_rng(5)
    q, p = rng.normal(size=(2, beads, 1, 3))
    mass, beta, dt = 1.7, 4., .17
    matrix = np.array([[.35, -.14], [-.14, -.15]])
    dynamics = RPMASHBADynamics(Constant(matrix), [mass], beta, beads, dt,
                                rtol=1e-11, atol=1e-13)
    state = dynamics.initialize(State(q, p), 1, rng=12)
    rho, initial_energy = state.spin_density_matrix(), dynamics.energy(state)
    # Independent Fourier normal modes of the physical spring Hamiltonian.
    frequency = 2*beads/beta*np.sin(np.pi*np.arange(beads)/beads)
    phase = frequency*dt
    cosine = np.cos(phase)[:, None, None]
    sine_over_frequency = (dt*np.sinc(phase/np.pi))[:, None, None]
    qhat, phat = np.fft.fft(q, axis=0), np.fft.fft(p, axis=0)
    expected_q = np.fft.ifft(cosine*qhat+sine_over_frequency*phat/mass, axis=0).real
    expected_p = np.fft.ifft(cosine*phat-mass*frequency[:, None, None]**2*sine_over_frequency*qhat, axis=0).real
    dynamics.step(state)
    np.testing.assert_allclose(state.q, expected_q, atol=4e-11)
    np.testing.assert_allclose(state.p, expected_p, atol=4e-11)
    energies = np.linalg.eigvalsh(matrix)
    unitary = expm(-1j*np.diag(energies[::-1])*dt)
    np.testing.assert_allclose(state.spin_density_matrix(), unitary@rho@unitary.conj().T, atol=2e-12)
    assert abs(dynamics.energy(state)-initial_energy) < 1e-9
    assert dynamics.last_hops == ()


def test_equatorial_constant_spin_does_not_produce_false_hop():
    dynamics = driver(1, Constant([[.2, 0], [0, -.2]]))
    nuclear = parent(1)
    state = RPMASHBAState(nuclear.q, nuclear.p, spin=[1., 0., 0.])
    dynamics.step(state)
    assert dynamics.last_hops == ()
    np.testing.assert_allclose(state.spin, [np.cos(.08), np.sin(.08), 0.], atol=2e-12)


def test_every_bead_gap_is_checked_even_when_centroid_gap_is_nonzero():
    class ClosingGap:
        def evaluate(self, q):
            matrix = np.zeros((len(q), 2, 2))
            matrix[:, 0, 0], matrix[:, 1, 1] = q[:, 0, 0], -q[:, 0, 0]
            gradient = np.zeros((len(q), 2, 2, 1, 3))
            gradient[:, 0, 0, 0, 0], gradient[:, 1, 1, 0, 0] = 1., -1.
            return DiabaticResult(matrix, gradient)

    nuclear = parent(2)
    nuclear.q[1, 0, 0] = 2.
    dynamics = driver(2, ClosingGap())
    before = copy.deepcopy(dynamics.rng.bit_generator.state)
    with pytest.raises(PotentialError, match="BA-RP-MASH.*nondegenerate"):
        dynamics.initialize(nuclear)
    assert before == dynamics.rng.bit_generator.state


@pytest.mark.parametrize("kwargs", [{"nstates": 3}, {"native": True}, {"native": None},
                                     {"rtol": True}, {"rtol": 1e-20}, {"atol": 0.}, {"max_step": -1.},
                                     {"max_events": False}, {"gap_tolerance": np.nan}])
def test_unsupported_constructor_options_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        driver(**kwargs)


def test_complex_provider_and_unimplemented_observables_fail_explicitly():
    dynamics = driver(potential=Constant([[.2, .1j], [-.1j, -.2]]))
    with pytest.raises(PotentialError, match="BA-RP-MASH.*real symmetric"):
        dynamics.initialize(parent())
    dynamics = driver()
    with pytest.raises(ValueError, match="only population"):
        dynamics.initialize(parent(), sampling="uniform-sphere")
    state = dynamics.initialize(parent())
    with pytest.raises(NotImplementedError, match="population preparation"):
        dynamics.correlation(state, state, np.eye(2), np.eye(2))
    assert not hasattr(dynamics, "density_matrix")


def test_derived_electronic_or_ring_energy_overflow_fails_explicitly():
    dynamics = driver(potential=Constant([[1e200, 0.], [0., -1e200]]))
    with pytest.raises(PotentialError, match="electronic.*overflow"):
        dynamics.initialize(parent())
    dynamics = driver()
    state = dynamics.initialize(parent())
    state.p[:] = 1e200
    with pytest.raises(ValueError, match="ring energy overflow"):
        dynamics.energy(state)


@pytest.mark.parametrize("mutation", ["hemisphere", "dtype", "alias", "norm", "sampling", "masses", "clock"])
def test_mutated_state_or_parameters_cannot_silently_change_calculation(mutation):
    dynamics, state = driver(1), crossing(1)
    if mutation == "hemisphere":
        state.active_surface = 1
    elif mutation == "dtype":
        state.spin = state.spin.astype(np.float32)
    elif mutation == "alias":
        state.q[0, 0] = state.spin
        state.spin = state.q[0, 0]
    elif mutation == "norm":
        state.spin *= .9
    elif mutation == "sampling":
        state.sampling = "uniform-sphere"
    elif mutation == "clock":
        state.time = np.inf
    else:
        dynamics.masses = np.array([2.])
    with pytest.raises(ValueError):
        dynamics.step(state)


def test_full_ring_time_reversal_including_hop():
    dynamics = driver(rtol=1e-11, atol=1e-13)
    state = crossing()
    state.q[:, 0, 0] = [-.03, .01, .02]
    state.p[:, 0, 0] = [2.1, 1.8, 2.1]
    before = state.copy()
    dynamics.step(state)
    assert dynamics.last_hop.accepted
    state.p *= -1
    state.spin[1] *= -1
    dynamics.step(state)
    assert state.active_surface == before.active_surface
    np.testing.assert_allclose(state.q, before.q, rtol=0., atol=5e-11)
    np.testing.assert_allclose(state.p, -before.p, rtol=0., atol=5e-11)
    np.testing.assert_allclose(state.spin, before.spin, rtol=0., atol=5e-11)


@pytest.mark.parametrize("basis", [np.diag([1., -1.]), np.array([[.8, -.6], [.6, .8]])])
def test_real_diabatic_basis_changes_preserve_physics(basis):
    first, second = crossing(), crossing()
    orientation = round(np.linalg.det(basis))
    # A reflection of the fixed basis reverses the adiabatic x/y axes,
    # whereas a proper rotation leaves the common adiabatic spin unchanged.
    second.spin[:2] *= orientation
    one = driver(rtol=1e-11, atol=1e-13)
    two = driver(potential=RotatingSurfaces(basis=basis), rtol=1e-11, atol=1e-13)
    one.step(first)
    two.step(second)
    assert first.active_surface == second.active_surface
    np.testing.assert_allclose(first.q, second.q, rtol=0., atol=3e-11)
    np.testing.assert_allclose(first.p, second.p, rtol=0., atol=3e-11)
    second.spin[:2] *= orientation
    np.testing.assert_allclose(first.spin, second.spin, rtol=0., atol=3e-11)
