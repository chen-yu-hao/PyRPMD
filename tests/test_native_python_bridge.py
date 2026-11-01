"""Explicit Python force boundary uses the same native operators and rollback."""
import numpy as np
import pytest

pytest.importorskip("pyrpmd._thermal")
from pyrpmd.native_ensemble import NativeEnsemble
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.potentials.base import PESResult


class PeriodicFree:
    def __init__(self):
        self.cell = np.eye(3)*2
        self.calls = 0
        self.fail_on = None

    def set_cell(self, cell):
        self.cell = cell.copy()

    def evaluate(self, q):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("controlled Python PES failure")
        return PESResult(np.zeros(len(q)), np.zeros_like(q),
                         virial=np.zeros((len(q), 3, 3)))


def make(potential, **kwargs):
    options = dict(q=np.ones((3, 2, 3))*.2, p=np.ones((3, 2, 3))*.1,
                   masses=[1., 2.], beta=2., dt=.01, seed=72,
                   thermostat="pile-l", potential=potential,
                   potential_identity="test-model-v1")
    options.update(kwargs)
    return NativeEnsemble(**options)


@pytest.mark.parametrize("thermostat", ["none", "pile-l", "svr", "nhc"])
def test_python_force_bridge_preserves_native_maps_and_counts(thermostat):
    native = make({"kind": "harmonic", "k": .7}, thermostat=thermostat)
    hybrid = make(HarmonicPotential(k=.7), thermostat=thermostat, allow_python=True)
    first, second = native.run(31, stride=3), hybrid.run(31, stride=3)
    np.testing.assert_allclose(first["samples"], second["samples"], atol=1e-12)
    for key in ("q", "p", "aux", "energy", "force"):
        np.testing.assert_allclose(native.state()[key], hybrid.state()[key], atol=1e-12)
    assert second["backend"] == "native-operators-python-pes"
    assert second["python_step_callbacks"] == 31
    assert first["python_step_callbacks"] == 0
    assert native.state()["rng"] == hybrid.state()["rng"]


@pytest.mark.parametrize("barostat", ["bzp", "mttk", "rgb"])
def test_python_npt_uses_same_native_cell_map_and_restart(barostat):
    spec = dict(mode=barostat, tau=1., gamma=.3)
    if barostat == "rgb":
        spec.update(stress=(np.eye(3)*.5).tolist(), reference_cell=(np.eye(3)*2).tolist())
    else:
        spec["pressure"] = .5
    options = dict(cell=np.eye(3)*2, barostat=spec)
    native = make({"kind": "free"}, **options)
    potential = PeriodicFree()
    hybrid = make(potential, allow_python=True, **options)
    expected = native.run(35, stride=4)
    first = hybrid.run(11, stride=4)
    saved = hybrid.state()
    last = hybrid.run(24, stride=4)
    np.testing.assert_array_equal(np.concatenate([first["samples"], last["samples"]]), expected["samples"])
    np.testing.assert_array_equal(potential.cell, hybrid.state()["cell"])
    hybrid.restore(saved)
    np.testing.assert_array_equal(potential.cell, saved["cell"])
    repeated = hybrid.run(24, stride=4)
    np.testing.assert_array_equal(repeated["samples"], last["samples"])


def test_python_failure_rolls_back_cell_physical_aux_and_rng():
    potential = PeriodicFree()
    hybrid = make(potential, allow_python=True, cell=potential.cell,
                  thermostat={"mode": "nm-gle", "gle": {"A": [[1., 2.], [-2., 1.]]}},
                  barostat={"mode": "mttk", "pressure": .5, "tau": 1.})
    hybrid.run(2)
    saved = hybrid.state()
    potential.fail_on = potential.calls + 3
    with pytest.raises(RuntimeError, match="controlled Python PES failure"):
        hybrid.run(9)
    assert hybrid.state()["digest"] == saved["digest"]
    np.testing.assert_array_equal(potential.cell, saved["cell"])
    potential.fail_on = None
    hybrid.run(9)


def test_python_set_state_failure_is_atomic():
    potential = PeriodicFree()
    hybrid = make(potential, allow_python=True, cell=potential.cell,
                  barostat={"mode": "mttk", "pressure": .5, "tau": 1.})
    saved = hybrid.state()
    potential.fail_on = potential.calls + 1
    with pytest.raises(RuntimeError, match="controlled Python PES failure"):
        hybrid.set_state(saved["q"]+.4, saved["p"], 10)
    assert hybrid.state()["digest"] == saved["digest"]
    np.testing.assert_array_equal(potential.cell, saved["cell"])


def test_python_periodic_bridge_requires_explicit_virial():
    class MissingVirial(PeriodicFree):
        def evaluate(self, q):
            return PESResult(np.zeros(len(q)), np.zeros_like(q))

    potential = MissingVirial()
    with pytest.raises(RuntimeError, match="physical configurational virials"):
        make(potential, allow_python=True, cell=potential.cell,
             barostat={"mode": "mttk", "pressure": .5, "tau": 1.})
    np.testing.assert_array_equal(potential.cell, np.eye(3)*2)


def test_python_npt_requires_cell_setter_and_explicit_identity():
    with pytest.raises(ValueError, match="set_cell"):
        make(HarmonicPotential(), allow_python=True, cell=np.eye(3),
             barostat={"mode": "mttk", "pressure": .5})
    with pytest.raises(ValueError, match="potential_identity"):
        make(HarmonicPotential(), allow_python=True, potential_identity=None)


def test_callback_receives_copied_buffers():
    class MutatingPotential:
        def evaluate(self, q):
            q[:] = 12345.
            return PESResult(np.zeros(len(q)), np.zeros_like(q))

    clean = make({"kind": "free"}, thermostat="none")
    mutating = make(MutatingPotential(), allow_python=True, thermostat="none")
    clean.run(4)
    mutating.run(4)
    np.testing.assert_array_equal(clean.state()["q"], mutating.state()["q"])
    np.testing.assert_array_equal(clean.state()["p"], mutating.state()["p"])
