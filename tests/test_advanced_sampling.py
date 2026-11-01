"""Detailed balance, kinetics clocks and real RETIS trajectory validation."""

import copy
import numpy as np
import pytest

from pyrpmd.biased_kinetics import estimate_biased_rate, rescale_biased_time
from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import FreePotential, HarmonicPotential
from pyrpmd.replica_exchange import ReplicaExchange, reduced_configurational_energy
from pyrpmd.retis import ReplicaExchangeTIS
from pyrpmd.transition_interface import TISPath


def test_temperature_swap_ratio_contains_quantum_spring_action():
    a = Dynamics(HarmonicPotential(k=0.4), [2.0], 3.0, 4, 0.03, native=False)
    b = Dynamics(HarmonicPotential(k=0.4), [2.0], 7.0, 4, 0.03, native=False)
    rng = np.random.default_rng(154)
    states = [State(rng.normal(size=(4, 1, 3)), rng.normal(size=(4, 1, 3))) for _ in range(2)]
    exchange = ReplicaExchange([a, b], states, seed=31)

    def action(dyn, q):
        physical = 0.2 * np.sum(q * q)
        springs = 0.5 * 2 * (4 / dyn.beta)**2 * np.sum((q - np.roll(q, -1, axis=0))**2)
        return dyn.beta / 4 * (physical + springs)

    qi, qj = states[0].q, states[1].q
    expected = action(a, qi) + action(b, qj) - action(a, qj) - action(b, qi)
    assert exchange.log_acceptance_ratio(0, 1) == pytest.approx(expected, abs=1e-12)
    classical_only = (a.beta - b.beta) / 4 * 0.2 * (np.sum(qi**2) - np.sum(qj**2))
    assert abs(classical_only - expected) > 0.1
    exchange.states.reverse()
    assert exchange.log_acceptance_ratio(0, 1) == pytest.approx(-expected, abs=1e-12)


def test_accepted_swap_preserves_reduced_kinetic_distribution_and_slot_clocks():
    dynamics = [Dynamics(FreePotential(), [2.], beta, 1, .01, native=False) for beta in [2., 8.]]
    a = State(np.zeros((1, 1, 3)), np.array([[[1., 2., 3.]]]), step=4, time=.04)
    b = State(np.ones((1, 1, 3)), np.array([[[4., 5., 6.]]]), step=4, time=.04)
    exchange = ReplicaExchange(dynamics, [a, b], seed=8)
    kinetic_before = sum(d.beta * np.sum(s.p**2) / 4 for d, s in zip(dynamics, exchange.states))
    move = exchange.attempt(0, 1)
    assert move.accepted and move.log_acceptance_ratio == 0
    np.testing.assert_array_equal(exchange.states[0].q, b.q)
    np.testing.assert_array_equal(exchange.states[0].p, 2 * b.p)
    np.testing.assert_array_equal(exchange.states[1].p, a.p / 2)
    kinetic_after = sum(d.beta * np.sum(s.p**2) / 4 for d, s in zip(dynamics, exchange.states))
    assert kinetic_after == kinetic_before
    assert [(s.step, s.time) for s in exchange.states] == [(4, .04), (4, .04)]
    np.testing.assert_array_equal(a.q, 0)  # constructor owns copies


def test_hamiltonian_exchange_uses_four_potential_evaluations():
    dynamics = [Dynamics(HarmonicPotential(k=k), [1.], 2, 1, .01, native=False) for k in [.3, 1.2]]
    states = [State(np.full((1, 1, 3), x), np.zeros((1, 1, 3))) for x in [.2, .7]]
    exchange = ReplicaExchange(dynamics, states)
    expected = -2 * (0.3 - 1.2) * 3 * (.7**2 - .2**2) / 2
    assert exchange.log_acceptance_ratio(0, 1) == pytest.approx(expected)


def test_replica_advance_rolls_back_all_slots_and_rngs_on_late_failure(monkeypatch):
    dyn = [Dynamics(HarmonicPotential(), [1.], beta, 2, .01, native=False, seed=i) for i, beta in enumerate([2., 3.])]
    states = [State(np.zeros((2, 1, 3)), d.sample_momenta()) for d in dyn]
    exchange = ReplicaExchange(dyn, states)
    rngs = [copy.deepcopy(d.rng.bit_generator.state) for d in dyn]
    def fail(*args, **kwargs):
        raise ArithmeticError("late PES failure")
    monkeypatch.setattr(dyn[1].potential, "evaluate", fail)
    with pytest.raises(ArithmeticError, match="late PES"):
        exchange.advance(2)
    for old, current, d, rng in zip(states, exchange.states, dyn, rngs):
        np.testing.assert_array_equal(current.q, old.q)
        np.testing.assert_array_equal(current.p, old.p)
        assert current.step == old.step
        assert d.rng.bit_generator.state == rng


def test_biased_time_has_correct_sign_and_is_gauge_invariant():
    times = np.array([0., .5, 1., 2.])
    bias = np.array([0., .2, .4])
    expected = np.sum(np.diff(times) * np.exp(3 * bias))
    result = rescale_biased_time(times, bias, beta=3, transition_bias=0., transition_unbiased=True)
    assert result["rescaled_time_au"] == pytest.approx(expected)
    shifted = rescale_biased_time(times, bias - 7, beta=3, transition_bias=-7., transition_unbiased=True)
    assert shifted["rescaled_time_au"] == pytest.approx(expected)
    assert result["acceleration_factor"] > 1
    for options in [{"transition_unbiased": False}, {"transition_unbiased": True, "deposited_in_transition": True}]:
        with pytest.raises(ValueError):
            rescale_biased_time(times, bias, beta=3, transition_bias=0, **options)


def test_censored_rate_retains_unreactive_observation_time():
    result = estimate_biased_rate([1., 2., 5.], events=[True, True, False])
    assert result["rate_au"] == .25
    assert result["events"] == 2 and result["censored"] == 1
    assert result["ks_refitted_bootstrap_pvalue"] is None
    none = estimate_biased_rate([2., 4.], events=[False, False])
    assert none["rate_au"] == 0 and none["mean_first_passage_au"] is None


def test_complete_exponential_fit_uses_refitted_parametric_bootstrap():
    data = np.random.default_rng(42).exponential(scale=10, size=80)
    result = estimate_biased_rate(data, seed=23, bootstrap=50)
    assert result == estimate_biased_rate(data, seed=23, bootstrap=50)
    assert result["rate_au"] == pytest.approx(1 / data.mean())
    assert 0 < result["ks_refitted_bootstrap_pvalue"] <= 1


def test_retis_real_paths_are_valid_and_samples_follow_swaps():
    dynamics = Dynamics(FreePotential(), [1.], 2., 1, .1, native=False)
    sampler = ReplicaExchangeTIS(dynamics, Cartesian(0), basin_a=-1., basin_b=1.,
                                  interfaces=[-.5, .5], max_path_steps=100, seed=17)
    state = State(np.zeros((1, 1, 3)), np.array([[[1., 0., 0.]]]))
    initial = sampler.generate_path(state, 1)
    result = sampler.sample_interfaces([initial, initial], n_samples=20, burn_in=2)
    assert len(result.interfaces) == 2
    np.testing.assert_array_equal(result.walker_labels.sum(axis=1), 1)
    assert result.swap_attempts[0] == 22
    assert result.swap_accepts[0] == 22
    for index, ensemble in enumerate(result.interfaces):
        sampler.validate_path(ensemble.final_path, index)
        assert len(ensemble.crossing) == 20
    assert result.summary()["requires_independent_flux"]


def test_retis_swap_checks_both_ensemble_memberships():
    dynamics = Dynamics(FreePotential(), [1.], 2., 1, .1, native=False)
    sampler = ReplicaExchangeTIS(dynamics, Cartesian(0), basin_a=-1, basin_b=1,
                                  interfaces=[-.5, .5], max_path_steps=100)
    # Direct exchange checks the indicator only; externally supplied path
    # dynamics is validated once by sample_interfaces/validate_path.
    values = np.array([-1., -.4, -1.])
    q = np.zeros((3, 1, 1, 3)); q[:, 0, 0, 0] = values
    low = TISPath(q, np.zeros_like(q), values, .1)
    values = np.array([-1., .8, 1.])
    q = q.copy(); q[:, 0, 0, 0] = values
    high = TISPath(q, np.zeros_like(q), values, .1)
    paths = [low, high]
    assert not sampler.exchange_paths(paths, 0, 1)
    assert paths[0] is low and paths[1] is high
