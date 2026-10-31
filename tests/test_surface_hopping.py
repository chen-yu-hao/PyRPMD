"""Centroid RPSH limits, fewest-switches current and hop energy accounting."""
import copy
import json

import numpy as np
import pytest
from scipy.linalg import expm

from pyrpmd.dynamics import Dynamics, INTEGRATORS, State
from pyrpmd.nonadiabatic import DiabaticResult
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.potentials.base import PotentialError
from pyrpmd.surface_hopping import RPSHDynamics, RPSHState


class AvoidedCrossing:
    def evaluate(self, q):
        x = q[:, 0, 0]
        v = np.zeros((len(q), 2, 2))
        v[:, 0, 0], v[:, 1, 1] = .3*x, -.3*x
        v[:, 0, 1] = v[:, 1, 0] = .2
        g = np.zeros((*v.shape, *q.shape[1:]))
        g[:, 0, 0, 0, 0], g[:, 1, 1, 0, 0] = .3, -.3
        return DiabaticResult(v, g)


class RotatingTwoState:
    """Constant eigenvalues and a known rotating adiabatic frame."""
    gap = .3
    slope = .4

    def basis(self, x):
        c, s = np.cos(self.slope*x), np.sin(self.slope*x)
        return np.array([[c, -s], [s, c]])

    def evaluate(self, q):
        v = np.empty((len(q), 2, 2))
        g = np.zeros((*v.shape, *q.shape[1:]))
        generator = np.array([[0., -1.], [1., 0.]])*self.slope
        for b, x in enumerate(q[:, 0, 0]):
            u = self.basis(x)
            v[b] = (u*np.array([-.5*self.gap, .5*self.gap]))@u.T
            g[b, :, :, 0, 0] = generator@v[b]-v[b]@generator
        return DiabaticResult(v, g)


@pytest.mark.parametrize("beads", [1, 4])
@pytest.mark.parametrize("scheme", INTEGRATORS)
def test_single_surface_limit_reuses_real_mass_rpmd(beads, scheme):
    class HarmonicMatrix:
        def evaluate(self, q):
            return DiabaticResult(.2*np.sum(q*q, axis=(1, 2))[:, None, None],
                                   .4*q[:, None, None])
    rng = np.random.default_rng(41)
    initial = State(rng.normal(size=(beads, 2, 3)), rng.normal(size=(beads, 2, 3)))
    ordinary = initial.copy()
    dyn = RPSHDynamics(HarmonicMatrix(), [2., 3.], 4., beads, .02, 1,
                       native=False, integrator=scheme)
    r = dyn.initialize(initial)
    reference = Dynamics(HarmonicPotential(.4), [2., 3.], 4., beads, .02,
                         native=False, integrator=scheme)
    for _ in range(10):
        dyn.step(r)
        reference.step(ordinary)
    np.testing.assert_allclose(r.q, ordinary.q, atol=1e-14)
    np.testing.assert_allclose(r.p, ordinary.p, atol=1e-14)
    assert dyn.energy(r) == pytest.approx(reference.energy(ordinary), abs=1e-13)
    assert dyn.last_probabilities == (0.,)
    assert dyn.last_hop is None


def test_centroid_electronics_matches_independent_rotating_frame_and_probability_current():
    pot = RotatingTwoState()
    q0, q1 = np.zeros((4, 1, 3)), np.zeros((4, 1, 3))
    q1[:, 0, 0] = .3
    # Nonzero spread leaves centroid electronics invariant.
    q0[:, 0, 1] = q1[:, 0, 1] = [-.3, -.1, .1, .3]
    dt, velocity = .6, .5
    d = np.array([[0., -pot.slope], [pot.slope, 0.]])
    exact_c = expm((-1j*np.diag([-.5*pot.gap, .5*pot.gap])-velocity*d)*dt)@np.array([1., 0.])
    exact_z = pot.basis(.3)@exact_c
    errors = []
    for substeps in [8, 16, 32]:
        dyn = RPSHDynamics(pot, [2.], 4., 4, dt, 2, native=False, electronic_substeps=substeps)
        final_z, probabilities = dyn._electronic_step(np.array([1.+0j, 0j]), q0, q1, 0)
        errors.append(np.linalg.norm(final_z-exact_z))
        assert np.vdot(final_z, final_z).real == pytest.approx(1., abs=2e-14)
        assert probabilities[1] == pytest.approx(abs(exact_c[1])**2, rel=3e-4)
        assert probabilities[0] == 0.
    assert 3.8 < errors[0]/errors[1] < 4.2
    assert 3.8 < errors[1]/errors[2] < 4.2


def hop_state(policy="ring", frustrated="keep", speed=2.):
    q = np.zeros((4, 1, 3))
    q[:, 0, 0] = [-.3, -.1, .2, .6]
    p = np.array([[[speed-.3, .2, .7]], [[speed-.1, -.3, .1]],
                  [[speed+.1, .4, -.1]], [[speed+.3, -.3, -.7]]])
    d = RPSHDynamics(AvoidedCrossing(), [2.], 4., 4, .01, 2,
                      native=False, rescaling=policy, frustrated_hops=frustrated)
    return d, d.initialize(State(q, p))


@pytest.mark.parametrize("policy", ["ring", "centroid"])
def test_upward_hop_conserves_selected_energy_and_preserves_internal_modes(policy):
    d, s = hop_state(policy)
    before = s.copy()
    total = d.energy(s)
    centroid_energy = .5*np.sum(s.p.mean(axis=0)**2/d.masses[:, None])
    eigenvalues = d._adiabatic(s.q.mean(axis=0, keepdims=True))[0][0]
    centroid_energy += eigenvalues[0]
    result = d.attempt_hop(s, 1)
    assert result.accepted and not result.frustrated and s.active_surface == 1
    np.testing.assert_array_equal(s.q, before.q)
    np.testing.assert_allclose(s.p-s.p.mean(axis=0), before.p-before.p.mean(axis=0), atol=5e-16)
    assert (s.step, s.time) == (before.step, before.time)
    if policy == "ring":
        assert d.energy(s) == pytest.approx(total, abs=4e-15)
        assert abs(result.ring_energy_change) < 4e-15
    else:
        after = .5*np.sum(s.p.mean(axis=0)**2/d.masses[:, None])+eigenvalues[1]
        assert after == pytest.approx(centroid_energy, abs=2e-15)
        assert result.ring_energy_change == pytest.approx(d.energy(s)-total, abs=2e-15)
        assert abs(result.ring_energy_change) > 1e-3


@pytest.mark.parametrize("frustrated", ["keep", "reverse"])
def test_frustrated_hops_have_explicit_energy_conserving_policy(frustrated):
    d, s = hop_state(frustrated=frustrated, speed=.03)
    before = s.copy()
    total = d.energy(s)
    result = d.attempt_hop(s, 1)
    assert result.frustrated and not result.accepted and s.active_surface == 0
    assert d.energy(s) == pytest.approx(total, abs=2e-15)
    if frustrated == "keep":
        np.testing.assert_array_equal(s.p, before.p)
    else:
        assert s.p[:, 0, 0].mean() == pytest.approx(-before.p[:, 0, 0].mean(), abs=1e-15)
        np.testing.assert_array_equal(s.p[:, :, 1:], before.p[:, :, 1:])
        np.testing.assert_allclose(s.p-s.p.mean(axis=0), before.p-before.p.mean(axis=0), atol=2e-16)


def test_downward_hop_releases_ring_energy_at_zero_centroid_velocity():
    d, s = hop_state(speed=0.)
    s.active_surface = 1
    total = d.energy(s)
    result = d.attempt_hop(s, 0)
    assert result.accepted
    assert d.energy(s) == pytest.approx(total, abs=2e-15)


def test_constant_orthogonal_diabatic_basis_change_preserves_hops_and_trajectory():
    pot = AvoidedCrossing()
    angle = .73
    u = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    class ChangedBasis:
        def evaluate(self, q):
            r = pot.evaluate(q)
            return DiabaticResult(u.T@r.matrix@u,
                                   np.einsum("si,bstac,tj->bijac", u, r.gradients, u))
    d1, s1 = hop_state(speed=0.)
    d2 = RPSHDynamics(ChangedBasis(), [2.], 4., 4, .01, 2, native=False, rescaling="ring")
    s1.active_surface = 1
    s2 = RPSHState(s1.q, s1.p, coefficients=u.T@s1.coefficients, active_surface=1)
    assert d1.attempt_hop(s1, 0).accepted
    assert d2.attempt_hop(s2, 0).accepted
    np.testing.assert_allclose(s1.p, s2.p, atol=5e-16)
    for _ in range(5):
        d1.step(s1)
        d2.step(s2)
        np.testing.assert_allclose(d1.last_probabilities, d2.last_probabilities, atol=2e-16)
    np.testing.assert_allclose(s1.q, s2.q, atol=1e-15)
    np.testing.assert_allclose(s1.p, s2.p, atol=1e-15)
    np.testing.assert_allclose(u.T@s1.coefficients, s2.coefficients, atol=3e-15)


def test_complex_soc_and_degenerate_hopping_fail_explicitly():
    class Matrix:
        def __init__(self, v): self.v = np.asarray(v)
        def evaluate(self, q):
            return DiabaticResult(np.broadcast_to(self.v, (len(q), 2, 2)),
                                   np.zeros((len(q), 2, 2, *q.shape[1:])))
    s = State(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)))
    for matrix, message in (([[0., .1j], [-.1j, 0.]], "complex SOC"),
                            ([[0., 0.], [0., 0.]], "degenerate")):
        d = RPSHDynamics(Matrix(matrix), [2.], 4., 1, .01, 2, native=False)
        with pytest.raises(PotentialError, match=message): d.initialize(s)


@pytest.mark.parametrize("failure", [PotentialError, KeyboardInterrupt])
def test_rng_and_all_state_roll_back_after_a_failed_hop(monkeypatch, failure):
    d, s = hop_state()
    before, rng = s.to_dict(), copy.deepcopy(d.rng.bit_generator.state)
    old_probabilities = d.last_probabilities
    monkeypatch.setattr(d, "_electronic_step", lambda *args: (args[0].copy(), np.array([0., 1.])))
    def fail(trial, *args):
        trial.p[:] = 0.
        trial.active_surface = 1
        raise failure("hop provider failure")
    monkeypatch.setattr(d, "_hop", fail)
    with pytest.raises(failure, match="provider failure"): d.step(s)
    assert s.to_dict() == before
    assert d.rng.bit_generator.state == rng
    assert d.last_hop is None and d.last_probabilities == old_probabilities


def test_state_and_rng_snapshot_reproduce_future_stochastic_sequence():
    d, s = hop_state()
    for _ in range(3): d.step(s)
    encoded = json.dumps(s.to_dict())
    saved_rng = copy.deepcopy(d.rng.bit_generator.state)
    other, _ = hop_state()
    restored = RPSHState.from_dict(json.loads(encoded))
    other.rng.bit_generator.state = saved_rng
    for _ in range(6):
        d.step(s)
        other.step(restored)
        assert d.last_hop == other.last_hop
        assert d.last_probabilities == other.last_probabilities
    assert s.to_dict() == restored.to_dict()
    assert d.rng.bit_generator.state == other.rng.bit_generator.state


@pytest.mark.parametrize("field,value", [("coefficients", [0., 0.]), ("coefficients", [1., np.nan]),
                                          ("active_surface", True), ("active_surface", 2)])
def test_invalid_state_is_not_silently_normalized(field, value):
    kwargs = {"coefficients": [1., 0.], "active_surface": 0, field: value}
    with pytest.raises(ValueError): RPSHState(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), **kwargs)


@pytest.mark.parametrize("field,value", [("rescaling", "all"), ("frustrated_hops", "auto"),
                                          ("gap_tolerance", 0.), ("electronic_substeps", 0), ("dt", -.1)])
def test_invalid_dynamics_parameters(field, value):
    args = dict(potential=AvoidedCrossing(), masses=[2.], beta=4., nbeads=4, dt=.01, nstates=2, native=False)
    args[field] = value
    with pytest.raises(ValueError): RPSHDynamics(**args)


def test_parameter_mutation_and_zero_active_population_are_rejected():
    d, s = hop_state()
    d.electronic_substeps += 1
    with pytest.raises(ValueError, match="immutable"): d.step(s)
    d, s = hop_state()
    _, u, _ = d._adiabatic(s.q.mean(axis=0, keepdims=True))
    s.coefficients[:] = u[0, :, 1]
    with pytest.raises(ValueError, match="zero electronic population"): d.step(s)


def test_unresolved_fewest_switches_probability_is_rejected_without_clipping():
    d = RPSHDynamics(RotatingTwoState(), [2.], 4., 1, .5, 2, native=False, electronic_substeps=1)
    s = d.initialize(State(np.zeros((1, 1, 3)), [[[16., 0., 0.]]]))
    before = s.to_dict()
    rng = copy.deepcopy(d.rng.bit_generator.state)
    with pytest.raises(ValueError, match="probabilities exceed one"): d.step(s)
    assert s.to_dict() == before
    assert d.rng.bit_generator.state == rng


def test_clock_overflow_cannot_commit_a_surface_hopping_step():
    class ZeroMatrix:
        def evaluate(self, q):
            return DiabaticResult(np.zeros((len(q), 1, 1)), np.zeros((len(q), 1, 1, *q.shape[1:])))
    d = RPSHDynamics(ZeroMatrix(), [2.], 4., 1, 1e308, 1, native=False)
    s = d.initialize(State(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), time=1e308))
    before, rng = s.to_dict(), copy.deepcopy(d.rng.bit_generator.state)
    with pytest.raises(ValueError, match="clock"): d.step(s)
    assert s.to_dict() == before
    assert d.rng.bit_generator.state == rng
