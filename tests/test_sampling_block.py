"""Focused E0/E1 and fallback checks for the NP2 sampling block."""

import copy

import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.sampling_block import propagate_sampling_block
import pyrpmd.sampling_block as sampling_block


requires_native = pytest.mark.skipif(
    not native_capabilities()["available"], reason="native build required"
)


def _pair(beads=8, atoms=2, seed=19):
    rng = np.random.default_rng(seed)
    q = rng.normal(scale=0.2, size=(beads, atoms, 3))
    p = rng.normal(scale=0.3, size=q.shape)
    return State(q, p, step=4, time=0.25)


@pytest.mark.parametrize("thermostat,kwargs", [
    ("pile-l", {}),
    ("trpmd", {"friction_scale": 0.5}),
])
@requires_native
def test_native_sampling_block_matches_repeated_steps(thermostat, kwargs):
    actual, expected = _pair(), _pair()
    dynamics = Dynamics(HarmonicPotential(0.4), [1.0, 2.0], 3.0, 8, 0.01,
                        native=True, seed=101)
    reference = Dynamics(HarmonicPotential(0.4), [1.0, 2.0], 3.0, 8, 0.01,
                         native=True, seed=101)
    result = propagate_sampling_block(
        dynamics, actual, 6, observe=np.array([1, 0, 1, 0, 1, 1], dtype=np.uint8),
        record_step_energies=True, thermostat=thermostat, **kwargs)
    reference_energies = []
    for _ in range(6):
        if thermostat == "trpmd":
            reference.trpmd_step(expected, friction_scale=kwargs["friction_scale"])
        else:
            reference.sample_step(expected)
        reference_energies.append(reference.energy(expected))
    assert result.backend == "native"
    assert result.completed == 6
    assert dynamics.propagation_stats["python_steps"] == 0
    np.testing.assert_array_equal(actual.q, expected.q)
    np.testing.assert_array_equal(actual.p, expected.p)
    assert dynamics.rng.bit_generator.state == reference.rng.bit_generator.state
    np.testing.assert_array_equal(result.step_energies, reference_energies)
    np.testing.assert_array_equal(result.energies, np.asarray(reference_energies)[[0, 2, 4, 5]])


@requires_native
def test_native_sampling_block_bias_and_observation_mask_are_bitwise_stable():
    from pyrpmd.bias import UmbrellaBias
    from pyrpmd.cv import Cartesian

    actual, expected = _pair(beads=4, atoms=1), _pair(beads=4, atoms=1)
    bias = UmbrellaBias(Cartesian(0, 0), [0.1], [[1.7]])
    dynamics = Dynamics(HarmonicPotential(0.2), [1.0], 2.5, 4, 0.01,
                        native=True, seed=202)
    reference = Dynamics(HarmonicPotential(0.2), [1.0], 2.5, 4, 0.01,
                         native=True, seed=202)
    result = propagate_sampling_block(
        dynamics, actual, 5, bias=bias, cv=Cartesian(0, 0),
        observe=np.array([0, 1, 0, 1, 1], dtype=np.uint8),
    )
    for _ in range(5):
        reference.sample_step(expected, bias)
    assert result.backend == "native"
    np.testing.assert_array_equal(actual.q, expected.q)
    np.testing.assert_array_equal(actual.p, expected.p)
    assert dynamics.rng.bit_generator.state == reference.rng.bit_generator.state
    assert result.steps.tolist() == [6, 8, 9]
    assert result.values.shape == (3, 1)


def test_unsupported_ring_falls_back_and_native_backend_rejects():
    actual, expected = _pair(beads=12, atoms=1), _pair(beads=12, atoms=1)
    dynamics = Dynamics(HarmonicPotential(0.4), [1.0], 3.0, 12, 0.01,
                        native=True, seed=303)
    reference = Dynamics(HarmonicPotential(0.4), [1.0], 3.0, 12, 0.01,
                         native=True, seed=303)
    result = propagate_sampling_block(dynamics, actual, 3)
    for _ in range(3):
        reference.sample_step(expected)
    assert result.backend == "python"
    assert "power-of-two" in result.fallback_reason
    np.testing.assert_array_equal(actual.q, expected.q)
    np.testing.assert_array_equal(actual.p, expected.p)
    assert dynamics.rng.bit_generator.state == reference.rng.bit_generator.state

    strict = Dynamics(HarmonicPotential(0.4), [1.0], 3.0, 12, 0.01,
                      native=True, loop_backend="native", seed=303)
    with pytest.raises(RuntimeError, match="native sampling block required"):
        propagate_sampling_block(strict, _pair(beads=12, atoms=1), 1)


@requires_native
def test_noise_budget_chunking_preserves_state_and_rng(monkeypatch):
    actual, expected = _pair(beads=8, atoms=2), _pair(beads=8, atoms=2)
    dynamics = Dynamics(HarmonicPotential(0.4), [1.0, 2.0], 3.0, 8, 0.01,
                        native=True, seed=404)
    reference = Dynamics(HarmonicPotential(0.4), [1.0, 2.0], 3.0, 8, 0.01,
                         native=True, seed=404)
    monkeypatch.setattr(sampling_block, "MAX_PREDRAWN_BYTES", actual.p.nbytes * 2)
    result = propagate_sampling_block(dynamics, actual, 5, record_energy=False)
    for _ in range(5):
        reference.sample_step(expected)
    assert result.backend == "native"
    np.testing.assert_array_equal(actual.q, expected.q)
    np.testing.assert_array_equal(actual.p, expected.p)
    assert dynamics.rng.bit_generator.state == reference.rng.bit_generator.state
