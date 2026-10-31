"""Enumerated electronic-path and Hamiltonian tests of original KC-RPMD."""

import copy
import itertools

import numpy as np
import pytest

from pyrpmd.kc_rpmd import (KCCalibration, KCRPMDDynamics, KCRPMDPotential,
                            KCRPMDSampler, KCRPMDState)
from pyrpmd.potentials.base import PotentialError
from test_isomorphic import TwoStateModel


CALIBRATION = KCCalibration(3.0, 5.0)


def enumerated_weight(model, q, y, beta, a, b):
    count = len(q)
    matrices = model.evaluate(q).matrix
    center = model.evaluate(q.mean(axis=0, keepdims=True)).matrix[0]
    w = (center[0, 0] - center[1, 1]) / center[0, 1]
    weights = np.zeros(3)
    for states in itertools.product((0, 1), repeat=count):
        term = 1.0
        for bead in range(count):
            i, j = states[bead], states[(bead+1) % count]
            term *= np.exp(-beta/count * matrices[bead, i, i])
            if i != j:
                term *= -beta/count * matrices[bead, i, j]
        sector = 0 if all(s == 0 for s in states) else 2 if all(s == 1 for s in states) else 1
        if sector == 1:
            term *= np.sqrt(a/np.pi) * CALIBRATION.mean_gap_gradient * np.exp(-a*w*w)
        theta = (-1, 0, 1)[sector]
        term /= 1 + np.exp(2*b*(abs(y-theta)-0.5))
        weights[sector] += term
    return weights


@pytest.mark.parametrize("count", [2, 3, 5])
@pytest.mark.parametrize("y", [-1.2, 0.13, 0.84])
def test_kc_augmented_path_matches_explicit_electronic_enumeration(count, y):
    q = np.random.default_rng(4).normal(scale=0.3, size=(count, 2, 3))
    beta, a, b = 2.7, 0.15, 7.0
    model = TwoStateModel()
    result = KCRPMDPotential(model, beta, count, calibration=CALIBRATION, a=a, b=b).evaluate(q, y)
    weights = enumerated_weight(model, q, y, beta, a, b)
    assert result.log_electronic_weight == pytest.approx(np.log(weights.sum()), abs=2e-14)
    assert result.potential_energy == pytest.approx(-count/beta*np.log(weights.sum()), abs=2e-14)
    np.testing.assert_allclose(result.sector_weights, weights/weights.sum(), atol=3e-15)


def test_kc_nuclear_and_auxiliary_forces_match_energy_derivatives():
    q = np.random.default_rng(9).normal(scale=0.15, size=(4, 2, 3))
    y = 0.17
    potential = KCRPMDPotential(TwoStateModel(), 3, 4, calibration=CALIBRATION, a=0.2, b=6)
    result = potential.evaluate(q, y)
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical = (potential.evaluate(plus, y).potential_energy - potential.evaluate(minus, y).potential_energy)/2e-6
        assert result.nuclear_forces[index] == pytest.approx(-numerical, abs=3e-9)
    numerical_y = (potential.evaluate(q, y+1e-6).potential_energy - potential.evaluate(q, y-1e-6).potential_energy)/2e-6
    assert result.auxiliary_force == pytest.approx(-numerical_y, abs=3e-9)


def test_kc_weak_coupling_kinks_do_not_use_cancelling_trace_subtraction():
    q = np.zeros((8, 1, 3))
    # At the crossing the weight of a kink pair is O(K^2), far below the
    # machine precision of the no-kink trace.  Augmented paths preserve it.
    q[:, 0, 0] = 1/6
    model = TwoStateModel(coupling=1e-10)
    potential = KCRPMDPotential(model, 3, 8, calibration=CALIBRATION, a=0.1, b=80)
    result = potential.evaluate(q, 0)
    weights = enumerated_weight(model, q, 0, 3, 0.1, 80)
    assert result.sector_weights[1] > 0.99
    assert result.log_electronic_weight == pytest.approx(np.log(weights.sum()), abs=2e-12)


def test_kc_calibrated_auxiliary_mass_and_sum_bead_convention():
    beta, gap_slope, coupling, count = 3.0, 0.6, 0.2, 4
    calibration = KCCalibration(gap_slope/coupling, 1/coupling)
    expected = beta**3/(2*np.pi**3) * gap_slope**2
    assert calibration.auxiliary_mass(beta) == pytest.approx(expected)
    dynamics = KCRPMDDynamics(TwoStateModel(), [1], beta, count, 0.001,
                              calibration=calibration, a=0.1, b=8, native=False)
    assert dynamics.auxiliary_mass == pytest.approx(count*expected)
    q = np.zeros((count, 1, 3))
    state = KCRPMDState(q, q, y=0.1, p_y=count*expected*0.3)
    potential_energy = dynamics.potential.evaluate(q, state.y).potential_energy
    assert (dynamics.energy(state)-potential_energy)/count == pytest.approx(0.5*expected*0.3**2)


def test_kc_second_order_energy_and_reversibility():
    q = np.random.default_rng(2).normal(scale=0.15, size=(4, 1, 3))
    start = KCRPMDState(q, np.full_like(q, 0.06), y=0.17, p_y=0.01)
    errors = []
    for dt in (0.01, 0.005):
        dynamics = KCRPMDDynamics(TwoStateModel(), [1], 3, 4, dt,
                                  calibration=CALIBRATION, a=0.1, b=8, native=False)
        state, count = start.copy(), round(0.1/dt)
        energy = dynamics.energy(state)
        for _ in range(count):
            dynamics.step(state)
        errors.append(abs(dynamics.energy(state)-energy))
        backward = KCRPMDDynamics(TwoStateModel(), [1], 3, 4, -dt,
                                  calibration=CALIBRATION, a=0.1, b=8, native=False)
        for _ in range(count):
            backward.step(state)
        np.testing.assert_allclose(state.q, start.q, atol=3e-13)
        np.testing.assert_allclose(state.p, start.p, atol=3e-13)
        assert state.y == pytest.approx(start.y, abs=3e-13)
        assert state.p_y == pytest.approx(start.p_y, abs=3e-13)
    assert errors[1] < errors[0]/3.7


@pytest.mark.parametrize("coupling", [0.0, -0.2, 4.0])
def test_kc_rejects_outside_primitive_positive_coupling_domain(coupling):
    potential = KCRPMDPotential(TwoStateModel(coupling), 3, 4,
                                calibration=CALIBRATION, a=0.1, b=8)
    with pytest.raises(PotentialError, match="coupling|primitive"):
        potential.evaluate(np.zeros((4, 1, 3)), 0)


@pytest.mark.parametrize("kwargs", [{"a": 1.1}, {"a": 0}, {"b": 0}, {"nbeads": 1}])
def test_kc_requires_valid_explicit_constraint_parameters(kwargs):
    settings = {"beta": 3, "nbeads": 4, "calibration": CALIBRATION, "a": 0.1, "b": 8}
    settings.update(kwargs)
    with pytest.raises(ValueError):
        KCRPMDPotential(TwoStateModel(), **settings)


def test_kc_sampler_reproducibility_and_failure_transaction():
    q = np.zeros((4, 1, 3))
    states = [KCRPMDState(q, q, y=0.1), KCRPMDState(q, q, y=0.1)]
    for state in states:
        dynamics = KCRPMDDynamics(TwoStateModel(), [1], 3, 4, 0.01,
                                  calibration=CALIBRATION, a=0.1, b=8, native=False)
        sampler = KCRPMDSampler(dynamics, position_step=0.03, auxiliary_step=0.05, seed=6)
        for _ in range(20):
            sampler.step(state)
        sampler.draw_momenta(state)
    np.testing.assert_array_equal(states[0].q, states[1].q)
    np.testing.assert_array_equal(states[0].p, states[1].p)
    assert states[0].y == states[1].y and states[0].p_y == states[1].p_y
    assert states[0].step == 0 and states[0].time == 0

    class Failed(TwoStateModel):
        calls = 0
        def evaluate(self, q):
            self.calls += 1
            if self.calls == 3:
                raise PotentialError("proposal failure")
            return super().evaluate(q)
    dynamics = KCRPMDDynamics(Failed(), [1], 3, 4, 0.01,
                              calibration=CALIBRATION, a=0.1, b=8, native=False)
    sampler = KCRPMDSampler(dynamics, position_step=0.03, auxiliary_step=0.05, seed=6)
    state = states[0].copy()
    rng, before = copy.deepcopy(sampler.rng.bit_generator.state), state.copy()
    with pytest.raises(PotentialError, match="proposal"):
        sampler.step(state)
    assert sampler.rng.bit_generator.state == rng
    assert sampler.attempted == sampler.accepted == 0
    np.testing.assert_array_equal(state.q, before.q)
    assert state.y == before.y


def test_kc_failed_propagation_is_transactional():
    class Failed(TwoStateModel):
        calls = 0
        def evaluate(self, q):
            self.calls += 1
            if self.calls == 3:
                raise PotentialError("force failure")
            return super().evaluate(q)
    q = np.zeros((4, 1, 3))
    state = KCRPMDState(q, np.full_like(q, 0.02), y=0.2, p_y=0.04)
    before = state.copy()
    dynamics = KCRPMDDynamics(Failed(), [1], 3, 4, 0.01,
                              calibration=CALIBRATION, a=0.1, b=8, native=False)
    with pytest.raises(PotentialError, match="force failure"):
        dynamics.step(state)
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert state.y == before.y and state.p_y == before.p_y
    assert state.step == before.step and state.time == before.time

