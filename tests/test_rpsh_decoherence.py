"""Independent checks of the optional centroid energy-based correction."""

import copy

import numpy as np
import pytest

from pyrpmd.nonadiabatic import DiabaticResult
from pyrpmd.surface_hopping import HopResult, RPSHDynamics, RPSHState


class ConstantMatrix:
    def __init__(self, energies, rotation=None):
        self.energies = np.asarray(energies, dtype=float)
        rotation = np.eye(len(energies)) if rotation is None else np.asarray(rotation)
        self.matrix = rotation@np.diag(self.energies)@rotation.T

    def evaluate(self, q):
        count = len(q)
        matrix = np.broadcast_to(self.matrix, (count, *self.matrix.shape)).copy()
        gradients = np.zeros((*matrix.shape, *q.shape[1:]))
        return DiabaticResult(matrix, gradients)


class RotatingMatrix:
    """Flat adiabatic surfaces with nonzero, analytically known coupling."""
    energies = np.array((0., 0.2))

    @staticmethod
    def basis(x):
        c, s = np.cos(0.5*x), np.sin(0.5*x)
        return np.array([[c, -s], [s, c]])

    def evaluate(self, q):
        matrix = np.empty((len(q), 2, 2))
        gradients = np.zeros((*matrix.shape, *q.shape[1:]))
        generator = np.array([[0., -0.5], [0.5, 0.]])
        for b, x in enumerate(q[:, 0, 0]):
            u = self.basis(x)
            matrix[b] = u@np.diag(self.energies)@u.T
            gradients[b, :, :, 0, 0] = generator@matrix[b]-matrix[b]@generator
        return DiabaticResult(matrix, gradients)


def driver(potential=None, *, nbeads=1, dt=0.3, masses=(2.,), **kwargs):
    if potential is None:
        potential = ConstantMatrix((0., 0.2, 0.7))
    return RPSHDynamics(potential, masses, beta=5., nbeads=nbeads,
                        dt=dt, nstates=len(potential.energies), native=False,
                        **kwargs)


def state(nbeads=1, coefficients=None, *, active=1, momentum=2.):
    if coefficients is None:
        coefficients = np.sqrt((0.2, 0.3, 0.5))*np.exp(1j*np.array((0.4, -0.2, 1.1)))
    q = np.zeros((nbeads, 1, 3))
    p = np.zeros_like(q)
    p[:, 0, 0] = momentum
    return RPSHState(q, p, coefficients=coefficients, active_surface=active)


def expected_edc(coefficients, energies, active, dt, kinetic, constant):
    """Closed-form uncoupled TDSE plus Eq. (17), with no integrator involved."""
    expected = coefficients*np.exp(-1j*np.asarray(energies)*dt)
    if kinetic == 0.:
        return expected
    inactive = np.arange(len(energies)) != active
    tau = (1.+constant/kinetic)/np.abs(np.asarray(energies)[inactive]-energies[active])
    expected[inactive] *= np.exp(-dt/tau)
    phase = expected[active]/abs(expected[active])
    expected[active] = phase*np.sqrt(1.-np.sum(np.abs(expected[inactive])**2))
    return expected


def test_uncoupled_edc_matches_analytic_amplitudes_and_active_phase():
    dynamics = driver(decoherence="edc", decoherence_constant=0.1)
    initial = state()
    expected = expected_edc(initial.coefficients, (0., 0.2, 0.7), 1,
                            dynamics.dt, kinetic=1., constant=0.1)
    dynamics.step(initial)
    np.testing.assert_allclose(initial.coefficients, expected, rtol=0., atol=3e-14)
    assert np.vdot(initial.coefficients, initial.coefficients).real == pytest.approx(1., abs=2e-14)
    assert initial.active_surface == 1
    assert dynamics.last_hop is None


def test_repeated_edc_has_analytic_exponential_decay():
    dynamics = driver(dt=0.04, decoherence="edc", decoherence_constant=0.17)
    initial = state()
    expected = expected_edc(initial.coefficients, (0., 0.2, 0.7), 1,
                            20*dynamics.dt, kinetic=1., constant=0.17)
    for _ in range(20):
        dynamics.step(initial)
    np.testing.assert_allclose(initial.coefficients, expected, rtol=0., atol=8e-14)


def test_correction_preserves_nuclear_state_and_active_surface_energy():
    dynamics = driver(nbeads=4, decoherence="edc")
    initial = state(4)
    initial.q[:, 0, 1] = (-0.4, 0.2, 0.3, -0.1)
    initial.p[:, 0, 1] = (0.3, -0.2, 0.4, -0.5)
    before = initial.copy()
    energy = dynamics.energy(initial)
    dynamics._apply_decoherence(initial)
    np.testing.assert_array_equal(initial.q, before.q)
    np.testing.assert_array_equal(initial.p, before.p)
    assert initial.active_surface == before.active_surface
    assert initial.step == before.step and initial.time == before.time
    assert dynamics.energy(initial) == energy
    assert not np.array_equal(initial.coefficients, before.coefficients)


def test_one_bead_and_ring_internal_modes_have_identical_centroid_edc():
    single = state(1)
    ring = state(4)
    ring.p[:, 0, 0] += (3., -3., 1., -1.)
    ring.q[:, 0, 1] = (-0.2, 0.2, 0.3, -0.3)
    one_driver = driver(decoherence="edc")
    ring_driver = driver(nbeads=4, decoherence="edc")
    # Internal kinetic and spring energies differ, while the physical centroid
    # velocity and hence the chosen decoherence times are the same.
    assert ring_driver.energy(ring)/4 != pytest.approx(one_driver.energy(single))
    one_driver.step(single)
    ring_driver.step(ring)
    np.testing.assert_allclose(ring.coefficients, single.coefficients, rtol=0., atol=3e-14)


@pytest.mark.parametrize("nbeads", [1, 4])
def test_stationary_centroid_has_no_decoherence(nbeads):
    coherent = state(nbeads, momentum=0.)
    if nbeads == 4:
        coherent.p[:, 0, 0] = (2., -2., 1., -1.)
    corrected = coherent.copy()
    driver(nbeads=nbeads).step(coherent)
    driver(nbeads=nbeads, decoherence="edc").step(corrected)
    np.testing.assert_allclose(corrected.coefficients, coherent.coefficients, rtol=0., atol=2e-14)
    np.testing.assert_array_equal(corrected.q, coherent.q)
    np.testing.assert_array_equal(corrected.p, coherent.p)


def test_default_none_is_unchanged_and_ignores_unused_constant():
    default_driver = driver(RotatingMatrix())
    explicit_driver = driver(RotatingMatrix(), decoherence="none", decoherence_constant=object())
    default_state = state(coefficients=[1., 0.], active=0)
    explicit_state = default_state.copy()
    for _ in range(8):
        default_driver.step(default_state)
        explicit_driver.step(explicit_state)
    assert default_state.to_dict() == explicit_state.to_dict()
    assert default_driver.last_probabilities == explicit_driver.last_probabilities
    assert default_driver.rng.bit_generator.state == explicit_driver.rng.bit_generator.state


def test_edc_uses_the_post_hop_surface_and_post_rescaling_kinetic_energy(monkeypatch):
    potential = RotatingMatrix()
    dynamics = driver(potential, decoherence="edc")
    initial = state(coefficients=[1., 0.], active=0)
    electronic = np.sqrt((0.6, 0.4))*np.exp(1j*np.array((0.4, -0.2)))

    def certain_upward_hop(coefficients, old_q, new_q, active):
        u = potential.basis(new_q[:, 0, 0].mean())
        return u@electronic, np.array([0., 1.])

    monkeypatch.setattr(dynamics, "_electronic_step", certain_upward_hop)
    dynamics.step(initial)
    assert dynamics.last_hop.accepted and initial.active_surface == 1
    kinetic = np.sum(initial.p.mean(axis=0)**2/(2*dynamics.masses[:, None]))
    assert kinetic == pytest.approx(0.8, abs=2e-15)
    expected = electronic.copy()
    expected[0] *= np.exp(-dynamics.dt*0.2/(1.+0.1/0.8))
    expected[1] = electronic[1]/abs(electronic[1])*np.sqrt(1.-abs(expected[0])**2)
    u = potential.basis(initial.q[:, 0, 0].mean())
    np.testing.assert_allclose(initial.coefficients, u@expected, rtol=0., atol=3e-14)
    assert dynamics.last_probabilities == (0., 1.)


def test_edc_state_and_rng_snapshot_reproduce_future_steps():
    dynamics = driver(RotatingMatrix(), decoherence="edc")
    initial = state(coefficients=[1., 0.], active=0)
    for _ in range(7):
        dynamics.step(initial)
    restored = RPSHState.from_dict(initial.to_dict())
    resumed = driver(RotatingMatrix(), decoherence="edc")
    resumed.rng.bit_generator.state = copy.deepcopy(dynamics.rng.bit_generator.state)
    for _ in range(15):
        dynamics.step(initial)
        resumed.step(restored)
        assert dynamics.last_hop == resumed.last_hop
        assert dynamics.last_probabilities == resumed.last_probabilities
    assert restored.to_dict() == initial.to_dict()
    assert dynamics.rng.bit_generator.state == resumed.rng.bit_generator.state


def test_edc_covaries_under_constant_real_basis_rotation_and_global_phase():
    rotation, _ = np.linalg.qr(np.array([[1., 2., -1.], [2., 1., 0.5], [-0.3, 0.4, 1.]]))
    phase = np.exp(0.63j)
    initial = state()
    rotated = state(coefficients=phase*(rotation@initial.coefficients))
    driver(decoherence="edc").step(initial)
    driver(ConstantMatrix((0., 0.2, 0.7), rotation), decoherence="edc").step(rotated)
    np.testing.assert_allclose(rotated.coefficients, phase*(rotation@initial.coefficients),
                               rtol=0., atol=4e-14)
    np.testing.assert_allclose(rotated.q, initial.q, rtol=0., atol=2e-14)
    np.testing.assert_allclose(rotated.p, initial.p, rtol=0., atol=2e-14)


def test_single_surface_edc_is_identity():
    dynamics = driver(ConstantMatrix((0.4,)), decoherence="edc")
    initial = state(coefficients=np.array([np.exp(0.3j)]), active=0)
    expected = initial.coefficients*np.exp(-0.4j*dynamics.dt)
    dynamics.step(initial)
    np.testing.assert_allclose(initial.coefficients, expected, rtol=0., atol=2e-14)


def test_zero_active_amplitude_has_no_invented_phase():
    dynamics = driver(decoherence="edc")
    initial = state(coefficients=np.array([1., 0., 0.]), active=1)
    with pytest.raises(ValueError, match="without a phase"):
        dynamics._apply_decoherence(initial)


def test_subnormal_active_amplitude_preserves_its_phase_without_overflow():
    dynamics = driver(decoherence="edc")
    initial = state(coefficients=np.array([np.sqrt(0.3), 1e-320+1e-320j, np.sqrt(0.7)]), active=1)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        dynamics._apply_decoherence(initial)
    assert np.vdot(initial.coefficients, initial.coefficients).real == pytest.approx(1., abs=2e-14)
    assert np.angle(initial.coefficients[1]) == pytest.approx(np.pi/4, abs=2e-15)


def test_failure_after_edc_rolls_back_state_rng_and_diagnostics(monkeypatch):
    dynamics = driver(decoherence="edc")
    initial = state()
    dynamics.last_hop = HopResult(0, 1, False, True, "centroid", 0.)
    dynamics.last_probabilities = (0.1, 0., 0.2)
    before = initial.to_dict()
    rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    hop, probabilities = dynamics.last_hop, dynamics.last_probabilities
    apply = dynamics._apply_decoherence

    def fail_after_correction(trial):
        apply(trial)
        assert not np.array_equal(trial.coefficients, initial.coefficients)
        dynamics.rng.random()
        raise RuntimeError("correction validation failed")

    monkeypatch.setattr(dynamics, "_apply_decoherence", fail_after_correction)
    with pytest.raises(RuntimeError, match="correction validation failed"):
        dynamics.step(initial)
    assert initial.to_dict() == before
    assert dynamics.rng.bit_generator.state == rng
    assert dynamics.last_hop is hop
    assert dynamics.last_probabilities is probabilities


@pytest.mark.parametrize("mode", ["unknown", "", None, 1])
def test_invalid_decoherence_mode_fails(mode):
    with pytest.raises(ValueError, match="decoherence must"):
        driver(decoherence=mode)


@pytest.mark.parametrize("constant", [0., -0.1, np.inf, np.nan, True, "bad", 0.1+0j, [0.1]])
def test_edc_requires_positive_finite_real_constant(constant):
    with pytest.raises(ValueError, match="decoherence_constant"):
        driver(decoherence="edc", decoherence_constant=constant)


@pytest.mark.parametrize("name,value", [("decoherence", "none"), ("decoherence_constant", 0.2)])
def test_decoherence_parameters_are_immutable(name, value):
    dynamics = driver(decoherence="edc")
    setattr(dynamics, name, value)
    with pytest.raises(ValueError, match="immutable"):
        dynamics.step(state())
