"""Shared block contracts apply to the whole operator state, not just q/p."""
import copy

import numpy as np
import pytest

from pyrpmd import Dynamics, ForceSplitting, State
from pyrpmd.potentials import HarmonicPotential, FreePotential
from pyrpmd.potentials.contraction import SplitPotential
from pyrpmd.propagation import NativeUnavailableError
from test_native_propagation import native_library, CompiledPotential


def initial():
    rng = np.random.default_rng(9)
    return State(rng.normal(size=(3, 3, 3))*.2,
                 rng.normal(size=(3, 3, 3))*.1, step=5, time=.123)


def make(thermostat="nhc", *, splitting=False, native=False):
    potential = HarmonicPotential(.2)
    split = None
    if splitting:
        potential = SplitPotential(potential, HarmonicPotential(.03), 3, 1)
        split = ForceSplitting(3)
    return Dynamics(potential, [1., 2., 3.], 3., 3, .01, native=native,
                    thermostat=thermostat, force_splitting=split, seed=37)


class FailingObserver:
    dimension = 1

    def __init__(self, fail):
        self.calls, self.fail = 0, fail

    def value(self, q):
        self.calls += 1
        if self.calls == self.fail:
            raise ValueError("observation failed")
        return np.array([q.mean()])


@pytest.mark.parametrize("splitting", [False, True])
@pytest.mark.parametrize("thermostat", ["nhc", "svr", {"mode": "gle", "gle": {"A": [[1., 2.], [-2., 1.]]}}])
def test_whole_block_failure_restores_complete_operator_state(splitting, thermostat):
    if thermostat != "nhc":
        pytest.importorskip("pyrpmd._thermal")
    dynamics = make(thermostat, splitting=splitting)
    state = initial()
    dynamics.step(state)
    before, operators = state.copy(), dynamics.operator_state_dict()
    stats = copy.deepcopy(dynamics.propagation_stats)
    with pytest.raises(ValueError, match="observation failed"):
        dynamics.propagate(state, 7, observe=FailingObserver(3))
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert (state.step, state.time) == (before.step, before.time)
    assert dynamics.operator_state_dict() == operators
    assert dynamics.propagation_stats == stats


@pytest.mark.parametrize("splitting", [False, True])
def test_prefix_observer_failure_retains_matching_bath_and_physical_prefix(splitting):
    dynamics, reference = make(splitting=splitting), make(splitting=splitting)
    state, expected = initial(), initial()
    reference.step(expected)
    result = dynamics.propagate_block(state, 5, cv=FailingObserver(2),
                                      record_states=True, raise_on_error=False)
    assert result.completed == 1 and "observation failed" in result.error
    np.testing.assert_array_equal(state.q, expected.q)
    np.testing.assert_array_equal(state.p, expected.p)
    assert dynamics.operator_state_dict() == reference.operator_state_dict()
    for _ in range(3):
        dynamics.step(state)
        reference.step(expected)
    np.testing.assert_array_equal(state.q, expected.q)
    np.testing.assert_array_equal(state.p, expected.p)


@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab", "velocity-verlet"])
def test_common_split_block_matches_compatibility_adapter(integrator):
    from pyrpmd import MTSDynamics
    potential = SplitPotential(HarmonicPotential(.2), HarmonicPotential(.03), 3, 1)
    args = (potential, [1., 2., 3.], 3., 3, .01)
    common = Dynamics(*args, force_splitting=ForceSplitting(3),
                      thermostat="nhc", native=False, integrator=integrator)
    adapter = MTSDynamics(*args, inner_steps=3, thermostat="nhc",
                          native=False, integrator=integrator)
    state, expected = initial(), initial()
    common.propagate(state, 8)
    adapter.propagate(expected, 8)
    np.testing.assert_array_equal(state.q, expected.q)
    np.testing.assert_array_equal(state.p, expected.p)
    assert common.force_evaluations == adapter.force_evaluations == {"fast": 25, "slow": 9}


def test_deterministic_chain_does_not_consume_random_seed():
    dynamics = make()
    rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    dynamics.propagate(initial(), 6)
    assert dynamics.rng.bit_generator.state == rng


@pytest.mark.parametrize("nbeads,scale", [(1, 1.), (3, 0.)])
def test_composed_trpmd_bath_free_limit_retains_exact_map_and_rng(nbeads, scale):
    args = (FreePotential(), [2.], 3., nbeads, .01)
    options = dict(native=False, seed=28)
    dynamics = Dynamics(*args, thermostat={"mode": "trpmd", "pile_scale": scale}, **options)
    reference = Dynamics(*args, **options)
    state = State(np.full((nbeads, 1, 3), .1), np.full((nbeads, 1, 3), .2))
    expected = state.copy()
    rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    for _ in range(4):
        dynamics.step(state)
        reference.step(expected)
    np.testing.assert_array_equal(state.q, expected.q)
    np.testing.assert_array_equal(state.p, expected.p)
    assert dynamics.rng.bit_generator.state == rng


def test_strict_native_rejects_missing_binding_without_rng_change():
    dynamics = make(native=True)
    class PythonHarmonic(HarmonicPotential):
        """Keep exercising an explicitly Python-only provider."""
    dynamics.potential = PythonHarmonic(k=.01)
    dynamics.loop_backend = "native"
    state, before = initial(), initial()
    operators = dynamics.operator_state_dict()
    with pytest.raises(NativeUnavailableError):
        dynamics.propagate(state, 3)
    np.testing.assert_array_equal(state.q, before.q)
    assert dynamics.operator_state_dict() == operators


def test_native_provider_rebinding_never_reuses_old_force_cache(native_library):
    pytest.importorskip("pyrpmd._thermal")
    dynamics = Dynamics(CompiledPotential(native_library), [1., 2., 3.], 3., 3, .01,
                        thermostat="nhc", potential_identity="harmonic-reference")
    state = initial()
    dynamics.step(state)
    before, operators = state.copy(), dynamics.operator_state_dict()
    dynamics.potential = CompiledPotential(native_library)
    with pytest.raises(ValueError, match="provider is immutable"):
        dynamics.step(state)
    np.testing.assert_array_equal(state.q, before.q)
    assert dynamics.operator_state_dict() == operators


def test_portable_native_checkpoint_requires_explicit_potential_identity(native_library):
    pytest.importorskip("pyrpmd._thermal")
    args = (CompiledPotential(native_library), [1., 2., 3.], 3., 3, .01)
    first, second = Dynamics(*args, thermostat="nhc"), Dynamics(*args, thermostat="nhc")
    first.step(initial())
    before = second.operator_state_dict()
    with pytest.raises(ValueError, match="physics/backend differs"):
        second.load_operator_state_dict(first.operator_state_dict())
    assert second.operator_state_dict() == before


def test_zero_length_strict_native_composed_block_is_a_noop(native_library):
    dynamics = Dynamics(CompiledPotential(native_library), [1., 2., 3.], 3., 3, .01,
                        thermostat="nhc", loop_backend="native")
    state, before = initial(), initial()
    operators = dynamics.operator_state_dict()
    result = dynamics.propagate_block(state, 0)
    assert result.success and result.completed == 0 and result.steps.size == 0
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert (state.step, state.time) == (before.step, before.time)
    assert dynamics.operator_state_dict() == operators


def test_sample_step_preserves_explicit_bath_free_trpmd_dict():
    args = (FreePotential(), [1., 2., 3.], 3., 3, .01)
    dynamics, reference = Dynamics(*args, native=False), Dynamics(*args, native=False)
    state, expected = initial(), initial()
    rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    for _ in range(4):
        dynamics.sample_step(state, thermostat={"mode": "trpmd", "pile_scale": 0.})
        reference.step(expected)
    np.testing.assert_array_equal(state.q, expected.q)
    np.testing.assert_array_equal(state.p, expected.p)
    assert dynamics.rng.bit_generator.state == rng


@pytest.mark.parametrize("spec,friction", [({"mode": "trpmd", "pile_scale": 0.}, 2.),
                                         ({"mode": "pile-l"}, 2.)])
def test_sample_step_rejects_conflicting_friction_before_state_changes(spec, friction):
    dynamics = make()
    state, before = initial(), initial()
    operators = dynamics.operator_state_dict()
    with pytest.raises(ValueError, match="friction_scale"):
        dynamics.sample_step(state, thermostat=spec, friction_scale=friction)
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert dynamics.operator_state_dict() == operators


@pytest.mark.parametrize("record_centroids", [False, True])
def test_sparse_native_whole_block_bounds_dense_temporary_records(native_library, monkeypatch, record_centroids):
    from pyrpmd import propagation
    from pyrpmd.native_ensemble import NativeEnsemble, _COLUMNS
    args = (CompiledPotential(native_library), [1., 2., 3.], 3., 3, .01)
    dynamics = Dynamics(*args, thermostat="nhc", block_steps=65536)
    reference = Dynamics(*args, thermostat="nhc", block_steps=65536)
    state, expected = initial(), initial()
    wanted = reference.propagate(expected, 8, sample_interval=8, record_centroids=record_centroids)
    calls, original = [], NativeEnsemble.run_prefix
    # Use a three-frame budget to exercise the same bound without allocating
    # hundreds of megabytes. It must constrain internal, not public, records.
    monkeypatch.setattr(propagation, "_COMPOSED_NATIVE_BUFFER_BYTES", 3 * (len(_COLUMNS) + 7*3) * 8)
    def bounded(engine, steps, **options):
        calls.append(steps)
        assert steps <= 3
        return original(engine, steps, **options)
    monkeypatch.setattr(NativeEnsemble, "run_prefix", bounded)
    actual = dynamics.propagate(state, 8, sample_interval=8, record_centroids=record_centroids)
    assert calls == [3, 3, 2]
    np.testing.assert_array_equal(state.q, expected.q)
    np.testing.assert_array_equal(state.p, expected.p)
    np.testing.assert_array_equal(actual.energies, wanted.energies)
    assert (actual.minimum_energy, actual.maximum_energy) == (wanted.minimum_energy, wanted.maximum_energy)
