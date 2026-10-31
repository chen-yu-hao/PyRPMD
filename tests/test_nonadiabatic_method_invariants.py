"""State ownership, native dispatch and failure invariants of the model APIs."""

import copy

import numpy as np
import pytest

from pyrpmd.dynamics import native_capabilities
from pyrpmd.isomorphic import IsomorphicEhrenfestDynamics, IsomorphicEhrenfestState
from pyrpmd.kc_rpmd import (KCCalibration, KCRPMDDynamics, KCRPMDResult,
                            KCRPMDSampler, KCRPMDState)
from pyrpmd.mv_rpmd import MVRPMDDynamics, MVRPMDSampler
from test_isomorphic import TwoStateModel
from test_mv_rpmd import example_state


def make_method(method, *, native=False, dt=0.001, model=None):
    model = TwoStateModel() if model is None else model
    q = np.random.default_rng(7).normal(scale=0.1, size=(3, 1, 3))
    p = np.full_like(q, 0.02)
    if method == "iso":
        return (IsomorphicEhrenfestDynamics(model, [1], 3, 3, dt, native=native),
                IsomorphicEhrenfestState(q, p, amplitudes=[0.8, 0.6j]))
    if method == "mv":
        return (MVRPMDDynamics(model, [1], 3, 3, dt, 2, native=native), example_state())
    return (KCRPMDDynamics(model, [1], 3, 3, dt, calibration=KCCalibration(3, 5),
                           a=0.1, b=8, native=native),
            KCRPMDState(q, p, y=0.1, p_y=0.01))


@pytest.mark.parametrize("method", ["iso", "mv", "kc"])
@pytest.mark.parametrize("parameter", ["dt", "beta", "masses"])
def test_changed_integrator_cannot_use_a_stale_native_context(method, parameter):
    dynamics, state = make_method(method)
    old = state.copy()
    if parameter == "masses":
        dynamics.integrator.masses = 2*dynamics.integrator.masses
    else:
        setattr(dynamics.integrator, parameter, 2*getattr(dynamics.integrator, parameter))
    with pytest.raises(ValueError, match="immutable"):
        dynamics.step(state)
    np.testing.assert_array_equal(state.q, old.q)
    np.testing.assert_array_equal(state.p, old.p)
    assert state.step == old.step and state.time == old.time


@pytest.mark.parametrize("method", ["iso", "mv", "kc"])
def test_overflowed_clock_is_not_committed(method, monkeypatch):
    dynamics, state = make_method(method, dt=6e307)
    state.time = 1.5e308
    before = state.copy()
    # Isolate the clock transaction from the unrelated enormous-dt integration
    # failure.  A successful no-op map still must not commit an invalid clock.
    monkeypatch.setattr(dynamics.integrator, "free_step", lambda q, p: None)
    if method in ("iso", "mv"):
        monkeypatch.setattr(dynamics, "_potential_step", lambda state, duration: None)
    else:
        monkeypatch.setattr(dynamics.potential, "evaluate", lambda q, y:
                            KCRPMDResult(0, np.zeros_like(q), 0, 0, np.array([1., 0, 0])))
    with pytest.raises(ValueError, match="clock"):
        dynamics.step(state)
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert state.step == before.step and state.time == before.time


def test_isomorphic_amplitudes_cannot_alias_nuclear_coordinates():
    dynamics, state = make_method("iso")
    state.q.ravel()[:4] = [1, 0, 0, 0]
    state.amplitudes = state.q.ravel()[:4].view(complex)
    with pytest.raises(ValueError, match="share memory"):
        dynamics.step(state)


@pytest.mark.parametrize("method", ["iso", "mv", "kc"])
def test_native_and_numpy_free_ring_maps_give_the_same_method_step(method):
    if not native_capabilities()["available"]:
        pytest.skip("compiled nuclear kernel is optional")
    native, native_state = make_method(method, native=True)
    numpy, numpy_state = make_method(method, native=False)
    native.step(native_state)
    numpy.step(numpy_state)
    for key in ("q", "p", "amplitudes", "mapping_q", "mapping_p"):
        if hasattr(native_state, key):
            np.testing.assert_allclose(getattr(native_state, key), getattr(numpy_state, key), atol=2e-12)
    if method == "kc":
        assert native_state.y == pytest.approx(numpy_state.y, abs=2e-13)
        assert native_state.p_y == pytest.approx(numpy_state.p_y, abs=2e-13)


@pytest.mark.parametrize("method", ["mv", "kc"])
def test_sampler_interrupt_restores_rng_state_and_counters(method):
    class Interrupted(TwoStateModel):
        calls = 0
        def evaluate(self, q):
            self.calls += 1
            if self.calls == (2 if method == "mv" else 3):
                raise KeyboardInterrupt("interrupted proposal")
            return super().evaluate(q)
    dynamics, state = make_method(method, model=Interrupted())
    if method == "mv":
        sampler = MVRPMDSampler(dynamics, position_step=0.02, mapping_step=0.03, seed=12)
    else:
        sampler = KCRPMDSampler(dynamics, position_step=0.02, auxiliary_step=0.03, seed=12)
    rng, before = copy.deepcopy(sampler.rng.bit_generator.state), state.copy()
    with pytest.raises(KeyboardInterrupt):
        sampler.step(state)
    assert sampler.rng.bit_generator.state == rng
    assert sampler.attempted == sampler.accepted == 0
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)

