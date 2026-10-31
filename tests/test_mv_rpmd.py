"""Projected-trace, nonseparable force and trajectory controls for MV-RPMD."""

import copy
import itertools

import numpy as np
import pytest
from scipy.linalg import expm

from pyrpmd.dynamics import Dynamics, State
from pyrpmd.mv_rpmd import (MappingNodeError, MVRPMDDynamics, MVRPMDPotential,
                            MVRPMDSampler, MVRPMDState)
from pyrpmd.nonadiabatic import DiabaticResult
from pyrpmd.potentials.base import PESResult, PotentialError
from test_isomorphic import ConstantMatrix, TwoStateModel


def example_state(nbeads=3):
    rng = np.random.default_rng(17)
    return MVRPMDState(rng.normal(scale=0.1, size=(nbeads, 1, 3)),
                       rng.normal(scale=0.04, size=(nbeads, 1, 3)),
                       mapping_q=rng.normal(size=(nbeads, 2)),
                       mapping_p=rng.normal(size=(nbeads, 2)))


def test_mv_nuclear_and_mapping_gradients_are_hamiltonian_derivatives():
    state = example_state()
    potential = MVRPMDPotential(TwoStateModel(), 3, 3, 2)
    value = potential.evaluate(state)
    for key, expected in (("q", -value.nuclear_forces),
                           ("mapping_q", value.mapping_q_gradient),
                           ("mapping_p", value.mapping_p_gradient)):
        for index in np.ndindex(getattr(state, key).shape):
            plus, minus = state.copy(), state.copy()
            getattr(plus, key)[index] += 1e-6
            getattr(minus, key)[index] -= 1e-6
            numerical = (potential.evaluate(plus).potential_energy
                          - potential.evaluate(minus).potential_energy) / 2e-6
            assert expected[index] == pytest.approx(numerical, abs=6e-8, rel=4e-7)


def test_mv_gaussian_projection_integrates_to_exact_electronic_trace():
    # Two-point Gauss-Hermite integrates every quadratic mapping variable
    # exactly.  This independent finite integral tests the SEO projection,
    # ordering of noncommuting matrices and retention of negative weights.
    q = np.array([[[-0.8, 0, 0]], [[0.6, 0, 0.2]]])
    model, beta, count = TwoStateModel(), 2.1, 2
    potential = MVRPMDPotential(model, beta, count, 2)
    total, negative = 0.0, 0
    population_numerator = np.zeros(2)
    for values in itertools.product((-1/np.sqrt(2), 1/np.sqrt(2)), repeat=8):
        values = np.asarray(values).reshape(2, 2, 2)
        state = MVRPMDState(q, np.zeros_like(q), mapping_q=values[0], mapping_p=values[1])
        result = potential.evaluate(state)
        total += result.sign * np.exp(result.log_absolute_theta) / 256
        population_numerator += result.sign * np.exp(result.log_absolute_theta) * result.populations / 256
        negative += result.sign < 0
    product = np.eye(2)
    for matrix in model.evaluate(q).matrix:
        product = product @ expm(-beta/count * matrix)
    assert total == pytest.approx(0.5**count * np.trace(product), abs=4e-14)
    np.testing.assert_allclose(population_numerator / total,
                               np.diagonal(product) / np.trace(product), atol=4e-14)
    assert negative > 0


def test_mv_population_sum_and_cyclic_phase_gauge_invariance():
    state = example_state(4)
    potential = MVRPMDPotential(TwoStateModel(), 3, 4, 2)
    result = potential.evaluate(state)
    assert result.populations.sum() == pytest.approx(1, abs=2e-13)
    z = state.mapping_q + 1j*state.mapping_p
    z *= np.exp(1j*np.array([0.1, -0.7, 1.2, 2.0]))[:, None]
    changed = MVRPMDState(np.roll(state.q, 1, axis=0), np.roll(state.p, 1, axis=0),
                          mapping_q=np.roll(z.real, 1, axis=0),
                          mapping_p=np.roll(z.imag, 1, axis=0))
    other = potential.evaluate(changed)
    assert other.potential_energy == pytest.approx(result.potential_energy, abs=2e-13)
    np.testing.assert_allclose(other.populations, result.populations, atol=2e-13)
    np.testing.assert_allclose(other.nuclear_forces, np.roll(result.nuclear_forces, 1, axis=0), atol=2e-13)


def test_mv_one_state_nuclear_limit_is_ordinary_rpmd():
    class Harmonic:
        def evaluate(self, q):
            return PESResult(np.sum(0.15*q*q, axis=(1, 2)), -0.3*q)
    class OneState:
        def evaluate(self, q):
            value = Harmonic().evaluate(q)
            return DiabaticResult(value.energy[:, None, None], -value.forces[:, None, None])
    q = np.random.default_rng(12).normal(scale=0.2, size=(3, 1, 3))
    p = np.full_like(q, 0.04)
    mapping = MVRPMDState(q, p, mapping_q=np.full((3, 1), 1.3), mapping_p=np.full((3, 1), 0.7))
    nuclear = State(q, p)
    mv = MVRPMDDynamics(OneState(), [1.7], 3, 3, 0.01, 1, native=False)
    ordinary = Dynamics(Harmonic(), [1.7], 3, 3, 0.01, native=False)
    for _ in range(5):
        mv.step(mapping)
        ordinary.step(nuclear)
    np.testing.assert_allclose(mapping.q, nuclear.q, atol=5e-15)
    np.testing.assert_allclose(mapping.p, nuclear.p, atol=5e-15)


def test_mv_reversibility_mapping_norm_and_second_order_energy():
    start = example_state()
    errors = []
    norm = start.mapping_q**2 + start.mapping_p**2
    for dt in (0.004, 0.002):
        dynamics = MVRPMDDynamics(TwoStateModel(), [1], 3, 3, dt, 2, native=False)
        state, count = start.copy(), round(0.016/dt)
        energy = dynamics.energy(state)
        initial_sign = dynamics.potential.evaluate(state).sign
        for _ in range(count):
            dynamics.step(state)
        errors.append(abs(dynamics.energy(state)-energy))
        np.testing.assert_allclose((state.mapping_q**2 + state.mapping_p**2).sum(axis=1),
                                   norm.sum(axis=1), atol=1e-10)
        assert dynamics.potential.evaluate(state).sign == initial_sign
        backward = MVRPMDDynamics(TwoStateModel(), [1], 3, 3, -dt, 2, native=False)
        for _ in range(count):
            backward.step(state)
        for key in ("q", "p", "mapping_q", "mapping_p"):
            np.testing.assert_allclose(getattr(state, key), getattr(start, key), atol=2e-11)
    assert errors[1] < errors[0]/3.7


def test_mv_node_and_failed_implicit_solver_leave_state_unchanged():
    q = np.zeros((1, 1, 3))
    nodal = MVRPMDState(q, q, mapping_q=[[0.5]], mapping_p=[[0.5]])
    with pytest.raises(MappingNodeError, match="node"):
        MVRPMDPotential(ConstantMatrix([[0]]), 2, 1, 1).evaluate(nodal)
    state, before = example_state(), example_state()
    dynamics = MVRPMDDynamics(TwoStateModel(), [1], 3, 3, 0.01, 2,
                              native=False, max_evaluations=1)
    with pytest.raises(PotentialError, match="converge"):
        dynamics.step(state)
    for key in ("q", "p", "mapping_q", "mapping_p"):
        np.testing.assert_array_equal(getattr(state, key), getattr(before, key))
    assert state.step == 0 and state.time == 0


def test_mv_negative_weights_are_not_clipped():
    q = np.zeros((1, 1, 3))
    state = MVRPMDState(q, q, mapping_q=[[0.2]], mapping_p=[[0.1]])
    result = MVRPMDPotential(ConstantMatrix([[0.4]]), 2, 1, 1).evaluate(state)
    assert result.sign == -1
    assert result.log_absolute_theta == pytest.approx(np.log(0.45)-0.8)


def test_mv_sampler_is_reproducible_and_does_not_advance_physical_time():
    states = [example_state(), example_state()]
    samplers = []
    for state in states:
        dynamics = MVRPMDDynamics(TwoStateModel(), [1], 3, 3, 0.001, 2, native=False)
        sampler = MVRPMDSampler(dynamics, position_step=0.02, mapping_step=0.08, seed=14)
        for _ in range(20):
            sampler.step(state)
        sampler.draw_momenta(state)
        samplers.append(sampler)
    for key in ("q", "p", "mapping_q", "mapping_p"):
        np.testing.assert_array_equal(getattr(states[0], key), getattr(states[1], key))
    assert states[0].step == 0 and states[0].time == 0
    assert 0 < samplers[0].accepted <= samplers[0].attempted == 20


def test_mv_sampler_failure_rolls_back_rng_and_counters():
    class Failed(TwoStateModel):
        calls = 0
        def evaluate(self, q):
            self.calls += 1
            if self.calls == 2:
                raise PotentialError("proposal failure")
            return super().evaluate(q)
    state = example_state()
    before = state.copy()
    sampler = MVRPMDSampler(MVRPMDDynamics(Failed(), [1], 3, 3, 0.01, 2, native=False),
                            position_step=0.02, mapping_step=0.04, seed=5)
    rng = copy.deepcopy(sampler.rng.bit_generator.state)
    with pytest.raises(PotentialError, match="proposal"):
        sampler.step(state)
    assert sampler.rng.bit_generator.state == rng
    assert sampler.attempted == sampler.accepted == 0
    for key in ("q", "p", "mapping_q", "mapping_p"):
        np.testing.assert_array_equal(getattr(state, key), getattr(before, key))

