"""Common native lowering honors only the observations requested by its caller."""
import numpy as np
import pytest

pytest.importorskip("pyrpmd._thermal")
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.native_ensemble import NativePeriodicPES
from test_native_periodic_sync import cell_library


def configured(library, *, beads=1, mass=1.):
    # The fixture has constant molecular energy and zero force at every finite q.
    provider = NativePeriodicPES(library, "evaluate", 1)
    provider.lib.set_failure(-1)
    return Dynamics(provider, [mass], 2., beads, .01, seed=27,
                    thermostat={"mode": "langevin", "gamma": 0.},
                    potential_identity="constant-energy-native-observation-test")


def prohibit_python_steps(dynamics, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("common native lowering called a Python step or force provider")

    monkeypatch.setattr(dynamics._composition, "step", forbidden)
    monkeypatch.setattr(dynamics.potential, "evaluate", forbidden)


@pytest.mark.parametrize("mask,record_energy,check_energy,completed", [
    ([False, False, True, False, True], True, False, 2),
    ([False]*5, True, False, 5),
    ([True, False, True, False, True], False, False, 5),
    ([False]*5, False, True, 0),
])
def test_common_prefix_sparse_and_disabled_energy_contract(
        cell_library, monkeypatch, mask, record_energy, check_energy, completed):
    dynamics = configured(cell_library)
    state = State(np.zeros((1, 1, 3)), np.full((1, 1, 3), 1e200), 7, 13.5)
    before = dynamics.operator_state_dict()
    reference = configured(cell_library)
    expected = state.copy()
    if completed:
        reference.propagate_block(expected, completed, observe=[False]*completed,
                                  record_energy=False, backend="native")
    prohibit_python_steps(dynamics, monkeypatch)
    result = dynamics.propagate_block(
        state, 5, observe=mask, record_energy=record_energy, check_energy=check_energy,
        backend="native", raise_on_error=False)
    assert result.backend == "native" and result.completed == completed
    np.testing.assert_array_equal(state.q, expected.q)
    np.testing.assert_array_equal(state.p, expected.p)
    assert state.step == expected.step and state.time == expected.time
    np.testing.assert_array_equal(result.steps, 8+np.flatnonzero(mask[:completed]))
    assert result.centroid_momenta.shape == (sum(mask[:completed]), 1, 3)
    if completed == 5:
        assert result.error is None
    else:
        assert "nonfinite total ring energy" in result.error
    if not record_energy:
        assert np.isnan(result.energies).all()
    if not completed:
        assert dynamics.operator_state_dict() == before


def test_common_prefix_unrequested_radius_does_not_reject_finite_energy(cell_library, monkeypatch):
    dynamics = configured(cell_library, beads=3, mass=1e-30)
    q = np.zeros((3, 1, 3));q[:, 0, 0] = [-1e160, 0., 1e160]
    state = State(q, np.zeros_like(q))
    prohibit_python_steps(dynamics, monkeypatch)
    result = dynamics.propagate_block(state, 3, backend="native", check_energy=True)
    assert result.completed == 3 and result.error is None
    assert np.isfinite(result.energies).all()
    assert result.diagnostics is None


def test_common_whole_block_unrequested_radius_does_not_reject_finite_energy(cell_library, monkeypatch):
    dynamics = configured(cell_library, beads=3, mass=1e-30)
    q = np.zeros((3, 1, 3));q[:, 0, 0] = [-1e160, 0., 1e160]
    state = State(q, np.zeros_like(q))
    prohibit_python_steps(dynamics, monkeypatch)
    result = dynamics.propagate(state, 3, backend="native", sample_interval=2)
    assert result.backend == "native"
    assert state.step == 3 and np.isfinite(result.energies).all()
    np.testing.assert_array_equal(result.steps, [2, 3])


def test_direct_composed_step_does_not_request_energy_or_moments(cell_library, monkeypatch):
    dynamics = configured(cell_library)
    state = State(np.zeros((1, 1, 3)), np.full((1, 1, 3), 1e200))

    def forbidden(*args, **kwargs):
        raise AssertionError("native step evaluated the Python force provider")

    monkeypatch.setattr(dynamics.potential, "evaluate", forbidden)
    dynamics.loop_backend = "native"
    dynamics.step(state)
    assert state.step == 1 and state.time == .01
    assert np.isfinite(state.q).all() and np.isfinite(state.p).all()
    assert dynamics.last_step_backend == "native"
