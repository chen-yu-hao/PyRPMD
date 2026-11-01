"""Static exchange is invalid if a borrowed physical component becomes dynamic."""
import copy

import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials.builtin import FreePotential
from pyrpmd.potentials.cache import StepEvaluationCache
from pyrpmd.potentials.contraction import DifferencePotential, SplitPotential
from pyrpmd.potentials.lifecycle import CommitteePotential, DeltaLearningPotential
from pyrpmd.replica_exchange import ReplicaExchange


@pytest.mark.parametrize("composition", ["committee", "delta", "difference", "rpc", "nested", "cache"])
@pytest.mark.parametrize("marker", ["stateful", "time_dependent", "history_dependent"])
@pytest.mark.parametrize("declaration", ["attribute", "metadata", "capabilities"])
def test_dynamic_borrowed_member_cannot_change_exchange_hamiltonian(composition, marker, declaration):
    class CountingFree(FreePotential):
        calls = 0

        def evaluate(self, q):
            self.calls += 1
            return super().evaluate(q)

    member = CountingFree()
    other = FreePotential()
    wrappers = {
        "committee": lambda: CommitteePotential([member, other]),
        "delta": lambda: DeltaLearningPotential(other, member),
        "difference": lambda: DifferencePotential(member, other),
        "rpc": lambda: SplitPotential(other, member, 4, 2),
        "nested": lambda: DeltaLearningPotential(other, CommitteePotential([member, FreePotential()])),
        "cache": lambda: StepEvaluationCache(CommitteePotential([member, other])),
    }
    provider = wrappers[composition]()
    dynamics = [Dynamics(provider, [1.0], beta, 4, 0.01, native=False) for beta in (1.0, 2.0)]
    states = [State(np.full((4, 1, 3), x), np.zeros((4, 1, 3))) for x in (0.1, 0.3)]
    exchange = ReplicaExchange(dynamics, states, seed=67)
    saved_rng = copy.deepcopy(exchange.rng.bit_generator.state)
    if declaration == "attribute":
        setattr(member, marker, True)
    else:
        member.metadata = copy.deepcopy(member.metadata)
        target = member.metadata if declaration == "metadata" else member.metadata.setdefault("capabilities", {})
        target[marker] = True
    with pytest.raises(ValueError, match="time-independent"):
        exchange.attempt(0, 1)
    assert member.calls == 0
    assert exchange.rng.bit_generator.state == saved_rng
    assert not np.any(exchange.attempts)
    for actual, expected in zip(exchange.states, states):
        np.testing.assert_array_equal(actual.q, expected.q)
        np.testing.assert_array_equal(actual.p, expected.p)


def test_recursive_potential_composition_rejects_cycle_before_evaluation():
    provider = CommitteePotential([FreePotential(), FreePotential()])
    provider.members = (provider, FreePotential())
    dynamics = [Dynamics(provider, [1.0], beta, 1, 0.01, native=False) for beta in (1.0, 2.0)]
    states = [State(np.zeros((1, 1, 3)), np.zeros((1, 1, 3))) for _ in dynamics]
    with pytest.raises(ValueError, match="acyclic"):
        ReplicaExchange(dynamics, states)


def test_static_nested_components_keep_exact_exchange_ratio():
    from pyrpmd.potentials.builtin import HarmonicPotential

    # Committee k=(.4+1.2)/2=.8; delta reference k=.2; total k=1.
    provider = DeltaLearningPotential(HarmonicPotential(k=0.2),
        CommitteePotential([HarmonicPotential(k=0.4), HarmonicPotential(k=1.2)]))
    dynamics = [Dynamics(provider, [1.0], beta, 1, 0.01, native=False) for beta in (1.0, 2.0)]
    states = [State(np.full((1, 1, 3), x), np.zeros((1, 1, 3))) for x in (0.1, 0.3)]
    exchange = ReplicaExchange(dynamics, states)
    expected = (1.0 - 2.0) * 0.5 * 3 * (0.1**2 - 0.3**2)
    assert exchange.log_acceptance_ratio(0, 1) == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("indirect", [False, True])
def test_cache_cycles_fail_before_forwarded_metadata_recurses(indirect):
    cache = StepEvaluationCache(FreePotential())
    if indirect:
        committee = CommitteePotential([FreePotential(), FreePotential()])
        cache._provider = committee
        committee.members = (cache, FreePotential())
    else:
        cache._provider = cache
    dynamics = [Dynamics(cache, [1.0], beta, 1, 0.01, native=False) for beta in (1.0, 2.0)]
    states = [State(np.zeros((1, 1, 3)), np.zeros((1, 1, 3))) for _ in dynamics]
    with pytest.raises(ValueError, match="acyclic"):
        ReplicaExchange(dynamics, states)


def test_mutual_cache_cycle_reaches_static_exchange_preflight():
    first = StepEvaluationCache(FreePotential())
    second = StepEvaluationCache(first)
    first._provider = second
    dynamics = [Dynamics(first, [1.0], beta, 1, 0.01, native=False) for beta in (1.0, 2.0)]
    states = [State(np.zeros((1, 1, 3)), np.zeros((1, 1, 3))) for _ in dynamics]
    with pytest.raises(ValueError, match="acyclic"):
        ReplicaExchange(dynamics, states)


@pytest.mark.parametrize("changed", ["potential", "bias"])
def test_advance_preflights_all_slots_before_any_provider_side_effect(changed):
    from pyrpmd.bias import CompositeBias

    class CountingFree(FreePotential):
        calls = 0

        def evaluate(self, q):
            self.calls += 1
            return super().evaluate(q)

    class CountingBias:
        calls = 0
        history_dependent = False

        def energy_force(self, q):
            self.calls += 1
            return 0.0, np.zeros_like(q)

    members = [CountingFree(), CountingFree()]
    providers = [members[0], CommitteePotential([members[1], FreePotential()])]
    bias = CountingBias()
    dynamics = [Dynamics(provider, [1.0], beta, 1, 0.01, native=False, seed=seed)
                for provider, beta, seed in zip(providers, (1.0, 2.0), (41, 42))]
    states = [State(np.zeros((1, 1, 3)), np.ones((1, 1, 3))) for _ in dynamics]
    exchange = ReplicaExchange(dynamics, states, biases=[None, CompositeBias(bias)], seed=43)
    rngs = [copy.deepcopy(dyn.rng.bit_generator.state) for dyn in dynamics]
    stats = [copy.deepcopy(dyn.propagation_stats) for dyn in dynamics]
    exchange_rng = copy.deepcopy(exchange.rng.bit_generator.state)
    if changed == "potential":
        members[1].time_dependent = True
    else:
        bias.history_dependent = True
    with pytest.raises(ValueError, match="time-independent"):
        exchange.advance(2)
    assert [member.calls for member in members] == [0, 0]
    assert bias.calls == 0
    assert exchange.rng.bit_generator.state == exchange_rng
    assert not np.any(exchange.attempts)
    for current, original, dyn, rng, stat in zip(exchange.states, states, dynamics, rngs, stats):
        np.testing.assert_array_equal(current.q, original.q)
        np.testing.assert_array_equal(current.p, original.p)
        assert (current.step, current.time) == (original.step, original.time)
        assert dyn.rng.bit_generator.state == rng
        assert dyn.propagation_stats == stat
