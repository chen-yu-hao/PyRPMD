"""NHC remains available without native extensions and keeps its saved backend."""
import copy
import sys

import numpy as np
import pytest

import pyrpmd
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.thermostats import NoseHooverChain, PythonNHCBath


def hide_thermal(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyrpmd._thermal", None)
    monkeypatch.delattr(pyrpmd, "_thermal", raising=False)


def setup(beads=3):
    q, p = np.random.default_rng(71).normal(scale=.2, size=(2, beads, 3, 3))
    q[:, 2] = [.1, -.2, .3]
    p[:, 2] = 0.
    common = dict(masses=[1., 2., 3.], beta=2., nbeads=beads, dt=.01,
                  native=False, fixed_atoms=[2], fixed_positions=q[0], seed=371)
    return common, State(q, p)


def make(common):
    return Dynamics(HarmonicPotential(k=.7), **common,
                    thermostat={"mode": "nhc", "tau": .4, "chain_length": 3, "substeps": 2})


@pytest.mark.parametrize("beads", [1, 3, 8])
def test_composed_nhc_without_thermal_matches_original_chain(monkeypatch, beads):
    hide_thermal(monkeypatch)
    common, state = setup(beads)
    actual = make(common)
    reference_dyn = Dynamics(HarmonicPotential(k=.7), **common)
    reference_bath = NoseHooverChain(reference_dyn, tdamp=.4, chain_length=3, substeps=2)
    reference = state.copy()
    rng = copy.deepcopy(actual.rng.bit_generator.state)
    for _ in range(30):
        actual.step(state)
        reference_bath.step(reference)
    helper = next(iter(actual._composition.baths.values()))
    assert isinstance(helper, PythonNHCBath)
    assert helper.state()["format"] == "pyrpmd.python-nhc-bath.v1"
    assert helper.chain.state_dict() == reference_bath.state_dict()
    np.testing.assert_array_equal(state.q, reference.q)
    np.testing.assert_array_equal(state.p, reference.p)
    assert actual.rng.bit_generator.state == rng


def test_python_nhc_checkpoint_keeps_python_backend_when_extension_returns(monkeypatch):
    common, state = setup()
    a = make(common)
    with monkeypatch.context() as missing:
        hide_thermal(missing)
        for _ in range(6):
            a.step(state)
        saved = a.operator_state_dict()
    b = make(common)
    b.load_operator_state_dict(saved)
    assert isinstance(next(iter(b._composition.baths.values())), PythonNHCBath)
    resumed = state.copy()
    for _ in range(9):
        a.step(state)
        b.step(resumed)
    np.testing.assert_array_equal(state.q, resumed.q)
    np.testing.assert_array_equal(state.p, resumed.p)
    assert a.operator_state_dict() == b.operator_state_dict()


def test_python_nhc_failure_restores_chain_and_physical_state(monkeypatch):
    hide_thermal(monkeypatch)
    common, state = setup()
    dyn = make(common)
    dyn.step(state)
    before, operators = state.copy(), dyn.operator_state_dict()
    original = dyn.potential.evaluate
    calls = 0

    def fail_second(q):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("controlled force failure")
        return original(q)

    monkeypatch.setattr(dyn.potential, "evaluate", fail_second)
    with pytest.raises(RuntimeError, match="controlled force failure"):
        dyn.step(state)
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert (state.step, state.time) == (before.step, before.time)
    assert dyn.operator_state_dict() == operators


def test_python_nhc_operator_rolls_back_overflow_without_momentum_writes():
    common, state = setup(1)
    dyn = Dynamics(HarmonicPotential(k=.7), **common)
    helper = PythonNHCBath(dyn, state.q, .005, {"mode": "nhc", "tau": .001})
    before = helper.state()
    p = state.p.copy()
    with pytest.raises(FloatingPointError):
        helper.apply_thermostat(state.q, p, 10.)
    np.testing.assert_array_equal(p, state.p)
    assert helper.state()["chain"] == before["chain"]


def test_python_nhc_rejects_malformed_snapshot_without_changing_bath():
    common, state = setup()
    dyn = Dynamics(HarmonicPotential(k=.7), **common)
    helper = PythonNHCBath(dyn, state.q, .005, {"mode": "nhc", "tau": .4})
    helper.apply_thermostat(state.q, state.p)
    before = helper.state()
    bad = copy.deepcopy(before)
    bad["chain"]["eta"][0] = 1j
    with pytest.raises(ValueError, match="finite real"):
        helper.restore(bad)
    assert helper.state()["chain"] == before["chain"]
    bad = copy.deepcopy(before)
    bad["identity"]["masses"][0] = 2.
    with pytest.raises(ValueError, match="parameters"):
        helper.restore(bad)
    assert helper.state()["chain"] == before["chain"]
