"""Independent review regressions for the electronic-bath lifecycle."""
import copy

import numpy as np
import pytest

from pyrpmd.dynamics import State
from pyrpmd.electronic_friction import (
    ConstantElectronicFriction, ElectronicFrictionDynamics, FrictionResult,
)
from pyrpmd.potentials.builtin import FreePotential
from pyrpmd.system import System


class ZeroNoise:
    def normal(self, size):
        return np.zeros(size)


@pytest.mark.parametrize("timing", ["before", "after"])
@pytest.mark.parametrize("override", ["instance", "class"])
def test_spatial_evaluate_override_is_not_replaced_by_cached_constant(timing, override, monkeypatch):
    provider = ConstantElectronicFriction(np.eye(3)*.25)
    def spatial(q):
        return FrictionResult(q[:, 0, 0, None, None]**2*np.eye(3))
    def replace():
        if override == "instance":
            monkeypatch.setattr(provider, "evaluate", spatial)
        else:
            monkeypatch.setattr(ConstantElectronicFriction, "evaluate", lambda self, q: spatial(q))
    if timing == "before": replace()
    driver = ElectronicFrictionDynamics(FreePotential(), [2.], 3., 2, .01, provider, native=False)
    if timing == "after": replace()
    driver.rng = ZeroNoise()
    q = np.array([[[1., 0., 0.]], [[3., 0., 0.]]])
    p = np.array([[[2., 3., 4.]], [[4., 1., 2.]]])
    pbar, internal = p.mean(axis=0), p-p.mean(axis=0)
    driver._centroid_ou(q, p, .2)
    # The independently solved scalar ODE has mean physical eta=(1²+3²)/2.
    np.testing.assert_allclose(p.mean(axis=0), pbar*np.exp(-.2*5/2), atol=2e-15)
    np.testing.assert_allclose(p-p.mean(axis=0), internal, atol=1e-15)


def test_friction_provider_identity_cannot_silently_change_a_cached_bath():
    original = ConstantElectronicFriction(np.eye(3))
    driver = ElectronicFrictionDynamics(FreePotential(), [2.], 3., 1, .01, original, native=False)
    with pytest.raises(AttributeError):
        driver.friction = ConstantElectronicFriction(np.eye(3)*2)
    assert driver.friction is original


def test_constructor_does_not_borrow_mutable_tensor_storage():
    tensor = np.eye(3)*.7
    provider = ConstantElectronicFriction(tensor)
    tensor[:] = 0.
    np.testing.assert_array_equal(provider.evaluate(np.zeros((1, 1, 3))).tensor[0], np.eye(3)*.7)


def test_nonfinite_clock_rolls_back_the_complete_electronic_friction_step():
    driver = ElectronicFrictionDynamics(FreePotential(), [2.], 3., 1, 1e308,
                                        ConstantElectronicFriction(np.zeros((3, 3))), native=False)
    state = State(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), time=1e308)
    before, rng = state.copy(), copy.deepcopy(driver.rng.bit_generator.state)
    with pytest.raises(FloatingPointError, match="clock"):
        driver.step(state)
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert (state.step, state.time) == (before.step, before.time)
    assert driver.rng.bit_generator.state == rng


def test_interrupt_during_block_observation_preserves_rng_and_state():
    class InterruptingObservation:
        dimension = 1
        def value(self, q):
            raise KeyboardInterrupt("interrupted observable")
    driver = ElectronicFrictionDynamics(FreePotential(), [2.], 3., 3, .1,
                                        ConstantElectronicFriction(np.eye(3)), native=False)
    state = State(np.zeros((3, 1, 3)), np.ones((3, 1, 3)))
    before, rng = state.copy(), copy.deepcopy(driver.rng.bit_generator.state)
    with pytest.raises(KeyboardInterrupt, match="observable"):
        driver.propagate(state, 2, observe=InterruptingObservation())
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert (state.step, state.time) == (before.step, before.time)
    assert driver.rng.bit_generator.state == rng


def test_failing_friction_cleanup_still_closes_the_nuclear_potential(tmp_path, monkeypatch):
    from pyrpmd.workflows import materials
    from pyrpmd import electronic_friction
    closed = []
    class PES(FreePotential):
        def close(self): closed.append("potential")
    class Friction(ConstantElectronicFriction):
        def close(self):
            closed.append("friction")
            raise RuntimeError("friction cleanup failure")
    monkeypatch.setattr(materials, "make_task_potential", lambda *args: PES())
    monkeypatch.setattr(electronic_friction, "make_electronic_friction", lambda *args: Friction(np.zeros((3, 3))))
    system = System(("H",), [2.], [[0., 0., 0.]])
    config = {
        "task": "diffusion", "beta": 3., "beads": 1, "dt_au": .01, "native": False,
        "potential": {"provider": "builtin", "name": "free"},
        "sampling": {"equilibration_steps": 0, "production_steps": 3, "sample_interval": 1, "replicas": 1},
        "electronic_friction": {"gamma": 0.},
    }
    with pytest.raises(RuntimeError, match="cleanup failure"):
        materials._sample_replica({"config": config, "system": system, "output": tmp_path, "replica": 0})
    assert closed == ["friction", "potential"]


def test_public_module_propagate_preserves_stochastic_transaction():
    from pyrpmd.propagation import propagate
    class BrokenObservation:
        dimension = 1
        def value(self, q): raise RuntimeError("observer failed")
    driver = ElectronicFrictionDynamics(FreePotential(), [2.], 3., 3, .1,
                                        ConstantElectronicFriction(np.eye(3)), native=False)
    state = State(np.zeros((3, 1, 3)), np.ones((3, 1, 3)))
    before, rng = state.copy(), copy.deepcopy(driver.rng.bit_generator.state)
    stats = copy.deepcopy(driver.propagation_stats)
    with pytest.raises(RuntimeError, match="observer failed"):
        propagate(driver, state, 2, observe=BrokenObservation())
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert (state.step, state.time) == (before.step, before.time)
    assert driver.rng.bit_generator.state == rng
    assert driver.propagation_stats == stats


@pytest.mark.parametrize("entry", ["block", "constraint"])
def test_direct_deterministic_helpers_reject_electronic_friction_before_mutation(entry):
    from pyrpmd.block_propagation import propagate_block
    from pyrpmd.workflows.rate import constrained_step
    driver = ElectronicFrictionDynamics(FreePotential(), [2.], 3., 3, .1,
                                        ConstantElectronicFriction(np.eye(3)), native=False)
    state = State(np.zeros((3, 1, 3)), np.ones((3, 1, 3)))
    before, rng = state.copy(), copy.deepcopy(driver.rng.bit_generator.state)
    with pytest.raises(ValueError, match="stochastic|deterministic"):
        if entry == "block": propagate_block(driver, state, 2)
        else: constrained_step(driver, state, None)
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert (state.step, state.time) == (before.step, before.time)
    assert driver.rng.bit_generator.state == rng


def test_explicit_null_friction_is_not_silently_treated_as_absent():
    from pyrpmd.electronic_friction import validate_electronic_friction
    system = System(("H",), [2.], [[0., 0., 0.]])
    assert validate_electronic_friction({"task": "diffusion"}, system) is None
    with pytest.raises(ValueError, match="table"):
        validate_electronic_friction({"task": "diffusion", "electronic_friction": None}, system)
