"""Configured native baths must retain custom Python free half maps."""
from types import MethodType

import numpy as np
import pytest

pytest.importorskip("pyrpmd._thermal")
from pyrpmd.dynamics import CayleyIntegrator, Dynamics, State
from pyrpmd.propagation import NativeUnavailableError
from test_native_propagation import CompiledPotential, native_library


def _dynamics(potential, thermostat="pile-l", nbeads=3):
    return Dynamics(potential, [1., 2., 3.], 2., nbeads, .01,
                    thermostat=thermostat, seed=61,
                    potential_identity="test-half-step-harmonic")


def _state(nbeads=3):
    rng = np.random.default_rng(72)
    q, p = rng.normal(scale=.2, size=(2, nbeads, 3, 3))
    return State(q, p, step=7, time=.123)


def _advance(dynamics, state, api, backend):
    if api == "step":
        dynamics.loop_backend = backend
        for _ in range(3):
            dynamics.step(state)
        return dynamics.last_step_backend
    return getattr(dynamics, api)(state, 3, backend=backend).backend


@pytest.mark.parametrize("api", ["step", "propagate", "propagate_block"])
@pytest.mark.parametrize("scope", ["instance", "class"])
@pytest.mark.parametrize("backend", ["auto", "native"])
def test_composed_pile_half_step_override_is_never_bypassed(
    native_library, monkeypatch, api, scope, backend,
):
    potential = CompiledPotential(native_library)
    dynamics = _dynamics(potential)
    reference = _dynamics(potential)
    actual, expected = _state(), _state()
    original = CayleyIntegrator.half_step
    calls = []

    def changed(self, q, p):
        calls.append(self)
        original(self, q, p)
        # This is a different requested map, not just an observation hook.
        q[:, 0, 0] += .004

    if scope == "class":
        monkeypatch.setattr(CayleyIntegrator, "half_step", changed)
    else:
        for target in (dynamics, reference):
            monkeypatch.setattr(target.integrator, "half_step",
                                MethodType(changed, target.integrator))

    before = dynamics.operator_state_dict()
    if backend == "native":
        with pytest.raises(NativeUnavailableError, match="Python force/operator scheduling"):
            _advance(dynamics, actual, api, backend)
        assert calls == []
        assert potential.lib.get_calls() == 0
        assert dynamics.operator_state_dict() == before
        np.testing.assert_array_equal(actual.q, expected.q)
        np.testing.assert_array_equal(actual.p, expected.p)
        assert (actual.step, actual.time) == (expected.step, expected.time)
        return

    assert _advance(dynamics, actual, api, backend) == "python"
    assert calls == [dynamics.integrator] * 6
    assert _advance(reference, expected, api, "python") == "python"
    assert calls == [dynamics.integrator] * 6 + [reference.integrator] * 6
    np.testing.assert_array_equal(actual.q, expected.q)
    np.testing.assert_array_equal(actual.p, expected.p)
    assert (actual.step, actual.time) == (expected.step, expected.time)
    assert dynamics.operator_state_dict() == reference.operator_state_dict()


@pytest.mark.parametrize("thermostat,nbeads", [
    ("nhc", 3), ("trpmd", 1), ({"mode": "trpmd", "pile_scale": 0.}, 3),
])
def test_unused_half_step_override_does_not_disable_native_execution(
    native_library, monkeypatch, thermostat, nbeads,
):
    potential = CompiledPotential(native_library)
    dynamics = _dynamics(potential, thermostat, nbeads)
    reference = _dynamics(potential, thermostat, nbeads)
    actual, expected = _state(nbeads), _state(nbeads)

    def unused(self, q, p):
        raise AssertionError("this plan must not invoke the half-step operator")

    monkeypatch.setattr(CayleyIntegrator, "half_step", unused)
    assert _advance(dynamics, actual, "propagate", "native") == "native"
    assert _advance(reference, expected, "propagate", "python") == "python"
    np.testing.assert_allclose(actual.q, expected.q, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(actual.p, expected.p, atol=2e-12, rtol=2e-12)

