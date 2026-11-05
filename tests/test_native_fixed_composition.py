"""Native fixed-atom lowering preserves the shared map and transaction contracts."""
import copy

import numpy as np
import pytest

from pyrpmd.bias import UmbrellaBias
from pyrpmd.block_propagation import PropagationError
from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import Dynamics, INTEGRATORS, native_capabilities

from test_native_propagation import CompiledPotential, native_library, state


pytestmark = pytest.mark.skipif(not native_capabilities()["available"], reason="native build required")


def _system(potential, scheme="cayley-bcb", *, native=True):
    initial = state(8)
    reference = initial.q[0].copy()
    initial.q[:, 0] = reference[0]
    initial.p[:, 0] = 0.0
    return Dynamics(potential, [1800.0] * 3, 220.0, 8, .2,
                    native=native, integrator=scheme, fixed_atoms=[0],
                    fixed_positions=reference), initial


def _assert_fixed(dynamics, snapshot):
    np.testing.assert_array_equal(snapshot.q[:, 0],
                                  np.broadcast_to(dynamics.fixed_positions[0], snapshot.q[:, 0].shape))
    np.testing.assert_array_equal(snapshot.p[:, 0], 0.0)


@pytest.mark.parametrize("scheme", INTEGRATORS)
@pytest.mark.parametrize("api", ["propagate", "propagate_block"])
@pytest.mark.parametrize("biased", [False, True])
def test_fixed_native_blocks_match_independent_numpy_map(native_library, monkeypatch, scheme, api, biased):
    native_pes = CompiledPotential(native_library)
    dynamics, actual = _system(native_pes, scheme)
    reference, expected = _system(CompiledPotential(native_library), scheme, native=False)
    cv = Cartesian(1, "y")
    bias = UmbrellaBias(cv, [.3], [[.07]]) if biased else None
    for _ in range(13):
        reference.step(expected, bias)
    python_calls = 0
    evaluate = native_pes.evaluate

    def watched(q):
        nonlocal python_calls
        python_calls += 1
        return evaluate(q)

    monkeypatch.setattr(native_pes, "evaluate", watched)
    if api == "propagate":
        result = dynamics.propagate(actual, 13, bias=bias, observe=cv,
                                    sample_interval=3, backend="native")
    else:
        result = dynamics.propagate_block(actual, 13, bias=bias, cv=cv,
                                          record_states=True, backend="native")
    assert result.backend == "native"
    assert python_calls == 0
    _assert_fixed(dynamics, actual)
    np.testing.assert_allclose(actual.q, expected.q, atol=3e-13, rtol=3e-13)
    np.testing.assert_allclose(actual.p, expected.p, atol=5e-12, rtol=5e-13)
    assert actual.step == expected.step
    assert actual.time == expected.time


@pytest.mark.parametrize("nonfinite", [0, 1])
def test_fixed_native_failure_preserves_valid_prefix_without_python_retry(native_library, nonfinite):
    potential = CompiledPotential(native_library)
    dynamics, actual = _system(potential)
    actual.step, actual.time = 11, .7
    expected = actual.copy()
    dynamics.propagate_block(expected, 2, backend="native", record_energy=False)
    potential.lib.reset(6, nonfinite)
    with pytest.raises(PropagationError) as raised:
        dynamics.propagate_block(actual, 8, backend="native", record_energy=False,
                                  record_states=True, reuse_forces=False)
    assert raised.value.result.completed == 2
    assert potential.lib.get_calls() == 6
    np.testing.assert_array_equal(actual.q, expected.q)
    np.testing.assert_array_equal(actual.p, expected.p)
    assert (actual.step, actual.time) == (expected.step, expected.time)
    _assert_fixed(dynamics, actual)


def test_fixed_native_transactional_block_rolls_back_entire_state(native_library):
    potential = CompiledPotential(native_library)
    dynamics, actual = _system(potential)
    before, rng = actual.copy(), copy.deepcopy(dynamics.rng.bit_generator.state)
    potential.lib.reset(6, 0)
    with pytest.raises(RuntimeError):
        dynamics.propagate(actual, 8, backend="native")
    assert potential.lib.get_calls() == 6
    np.testing.assert_array_equal(actual.q, before.q)
    np.testing.assert_array_equal(actual.p, before.p)
    assert (actual.step, actual.time) == (before.step, before.time)
    assert dynamics.rng.bit_generator.state == rng
    assert dynamics.propagation_stats["native_steps"] == 0


def test_fixed_projection_requires_native_api_marker_before_state_mutation(native_library, monkeypatch):
    dynamics, actual = _system(CompiledPotential(native_library))
    before = actual.copy()
    monkeypatch.setattr(dynamics.integrator._native, "FIXED_ATOMS_API", 0)
    for advance in (dynamics.propagate, dynamics.propagate_block):
        with pytest.raises(RuntimeError, match="fixed-atom projection"):
            advance(actual, 1, backend="native")
    np.testing.assert_array_equal(actual.q, before.q)
    np.testing.assert_array_equal(actual.p, before.p)


def test_mobile_constraint_with_fixed_substrate_retains_python_fallback(native_library):
    dynamics, actual = _system(CompiledPotential(native_library))
    actual.q[:, 1, 0] = 0.0
    actual.p[:, 1, 0] -= actual.p[:, 1, 0].mean()
    constraint = Cartesian(1, "x")
    expected = actual.copy()
    reference, _ = _system(CompiledPotential(native_library), native=False)
    reference.propagate_block(expected, 3, constraint=constraint, backend="python")
    result = dynamics.propagate_block(actual, 3, constraint=constraint, backend="auto")
    assert result.backend == "python"
    _assert_fixed(dynamics, actual)
    np.testing.assert_allclose(actual.q, expected.q, atol=3e-13, rtol=3e-13)
    np.testing.assert_allclose(actual.p, expected.p, atol=5e-12, rtol=5e-13)
