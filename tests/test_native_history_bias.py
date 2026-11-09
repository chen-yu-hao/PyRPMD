"""Native tagged descriptors for fixed and history-dependent biases."""

import numpy as np
import pytest

from pyrpmd.bias import (
    CompositeBias,
    GridMetadynamicsBias,
    MetadynamicsBias,
    WallBias,
)
from pyrpmd.block_propagation import (
    _check_bias_snapshot,
    bias_descriptors,
)
from pyrpmd.cv import Cartesian, CVStack
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.potentials import HarmonicPotential

_native = pytest.importorskip("pyrpmd._native")


pytestmark = pytest.mark.skipif(
    not native_capabilities()["available"], reason="native build required"
)


def _dynamics():
    return Dynamics(HarmonicPotential(k=0.2), [1.0], 2.0, 3, 0.002)


def _state(q=(0.7, 0.4, 0.0)):
    coordinates = np.repeat(np.asarray([[q]], dtype=float), 3, axis=0)
    momenta = np.zeros_like(coordinates)
    momenta[:, 0, 0] = 0.1
    return State(coordinates, momenta)


def _compare_native_and_python(bias, *, q=(0.7, 0.4, 0.0), steps=4):
    native_dynamics = _dynamics()
    python_dynamics = _dynamics()
    native_state, python_state = _state(q), _state(q)
    native = native_dynamics.propagate_block(
        native_state, steps, bias=bias, backend="native"
    )
    reference = python_dynamics.propagate_block(
        python_state, steps, bias=bias, backend="python"
    )
    assert native.backend == "native"
    assert reference.backend == "python"
    np.testing.assert_allclose(native_state.q, python_state.q, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(native_state.p, python_state.p, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(native.energies, reference.energies, rtol=1e-12, atol=1e-12)


def test_wall_and_composite_history_biases_match_python():
    cv = CVStack([Cartesian(0, 0)])
    wall = WallBias(cv, 0.3, lower=-0.2, upper=0.2)
    _compare_native_and_python(wall, q=(0.7, 0.0, 0.0))

    hills = MetadynamicsBias(cv, [0.4], 0.1, 2.0, mode="standard")
    for step, center in enumerate(([0.5], [0.3], [0.7]), 1):
        hills.deposit(center, step=step, time=0.1 * step)
    _compare_native_and_python(CompositeBias(wall, hills), q=(0.7, 0.0, 0.0))


@pytest.mark.parametrize("mode", ["standard", "well_tempered"])
def test_periodic_hills_match_python(mode):
    cv = CVStack([Cartesian(0, 0), Cartesian(0, 1)])
    cv.periods = (1.0, None)
    hills = MetadynamicsBias(
        cv,
        [0.4, 0.2],
        0.1,
        2.0,
        mode=mode,
        bias_factor=4.0 if mode == "well_tempered" else None,
    )
    for step, center in enumerate(([0.95, 0.2], [0.05, 0.1], [0.7, 0.4]), 1):
        hills.deposit(center, step=step, time=0.1 * step)
    _compare_native_and_python(hills)


def test_periodic_two_dimensional_grid_matches_python():
    cv = CVStack([Cartesian(0, 0), Cartesian(0, 1)])
    cv.periods = (1.0, None)
    grid = GridMetadynamicsBias(
        cv,
        [0.4, 0.2],
        0.1,
        2.0,
        mode="well_tempered",
        bias_factor=4.0,
        grid_ranges=[[0.0, 1.0], [-1.0, 1.0]],
        grid_bins=[6, 7],
    )
    grid.deposit([0.95, 0.2], step=1, time=0.1)
    grid.deposit([0.05, 0.1], step=2, time=0.2)
    _compare_native_and_python(grid)


def test_native_descriptors_are_detached_and_snapshot_expiry_is_rejected(monkeypatch):
    dynamics = _dynamics()
    cv = CVStack([Cartesian(0, 0)])
    hills = MetadynamicsBias(cv, [0.4], 0.1, 2.0, mode="standard")
    hills.deposit([0.2], step=1, time=0.1)
    descriptor = bias_descriptors(hills, dynamics)[0]
    assert descriptor["hill_count"] == 1
    assert isinstance(descriptor["centers"], tuple)
    hills.deposit([0.3], step=2, time=0.2)
    with pytest.raises(RuntimeError, match="snapshot expired"):
        _check_bias_snapshot(hills, [descriptor])

    grid = GridMetadynamicsBias(
        cv, [0.4], 0.1, 2.0, mode="standard",
        grid_ranges=[[-1.0, 1.0]], grid_bins=8,
    )
    grid.deposit([0.2], step=1, time=0.1)
    grid_descriptor = bias_descriptors(grid, dynamics)[0]
    grid.deposit([0.3], step=2, time=0.2)
    with pytest.raises(RuntimeError, match="snapshot expired"):
        _check_bias_snapshot(grid, [grid_descriptor])

    # Exercise the shared insertion point in propagate_block itself: the
    # native call completes, then an event mutates the live history before the
    # state clock is committed.
    live = MetadynamicsBias(cv, [0.4], 0.1, 2.0, mode="standard")
    live.deposit([0.2], step=1, time=0.1)
    original_advance = _native.advance

    def advance_then_deposit(*args):
        result = original_advance(*args)
        live.deposit([0.3], step=2, time=0.2)
        return result

    monkeypatch.setattr(_native, "advance", advance_then_deposit)
    state = _state()
    with pytest.raises(RuntimeError, match="snapshot expired"):
        dynamics.propagate_block(state, 1, bias=live, backend="native")
    assert state.step == 0


def test_subclass_and_missing_capability_retain_python_contract(monkeypatch):
    cv = CVStack([Cartesian(0, 0)])
    base = MetadynamicsBias(cv, [0.4], 0.1, 2.0, mode="standard")
    base.deposit([0.2], step=1, time=0.1)

    class DerivedMetadynamics(MetadynamicsBias):
        pass

    derived = DerivedMetadynamics(cv, [0.4], 0.1, 2.0, mode="standard")
    derived.deposit([0.2], step=1, time=0.1)
    dynamics = _dynamics()
    assert dynamics.propagate_block(_state(), 1, bias=derived).backend == "python"

    monkeypatch.setattr(_native, "NATIVE_HISTORY_BIAS_API", 0)
    state = _state()
    result = dynamics.propagate_block(state, 1, bias=base)
    assert result.backend == "python"
    assert "history-bias" in result.fallback_reason
    untouched = _state()
    with pytest.raises(RuntimeError, match="native loop required"):
        dynamics.propagate_block(untouched, 1, bias=base, backend="native")
    np.testing.assert_array_equal(untouched.q, _state().q)
    np.testing.assert_array_equal(untouched.p, _state().p)
