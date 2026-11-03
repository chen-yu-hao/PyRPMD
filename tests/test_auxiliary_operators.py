import copy

import numpy as np
import pytest

from pyrpmd.auxiliary import (
    AuxiliaryCompatibility,
    AuxiliaryLayout,
    AuxiliarySplitPoint,
    validate_auxiliary_operator,
)
from pyrpmd.dynamics import CayleyIntegrator
from pyrpmd.engine import ForceSplitting
from pyrpmd.input import InputError
from pyrpmd.nonadiabatic import (
    DiabaticResult,
    NRPMDAuxiliaryOperator,
    NRPMDDynamics,
    NRPMDState,
    _mint,
)
from pyrpmd.nonadiabatic_input import resolve_nonadiabatic_job
from pyrpmd.potentials.base import Potential, coordinates
from pyrpmd.potentials.builtin import HarmonicPotential
from pyrpmd.potentials.contraction import SplitPotential


class LinearDiabatic(Potential):
    nstates = 2
    natoms = 1
    metadata = {"provider": "t26-linear", "batch_independent": True}

    def __init__(self, slope=0.021, coupling=0.007):
        self.slope = float(slope)
        self.coupling = float(coupling)
        self.calls = 0

    def evaluate(self, q):
        q = coordinates(q)
        self.calls += 1
        x = q[:, 0, 0]
        matrix = np.zeros((len(q), 2, 2))
        matrix[:, 0, 0] = self.slope * x
        matrix[:, 1, 1] = -self.slope * x
        matrix[:, 0, 1] = matrix[:, 1, 0] = self.coupling
        gradients = np.zeros((len(q), 2, 2, 1, 3))
        gradients[:, 0, 0, 0, 0] = self.slope
        gradients[:, 1, 1, 0, 0] = -self.slope
        return DiabaticResult(matrix, gradients)

    def to_dict(self):
        return {"model": "t26-linear", "slope": self.slope, "coupling": self.coupling}


class PeriodicDiabatic(Potential):
    nstates = 2
    natoms = 1
    metadata = {"provider": "t26-periodic", "periodic": True}

    def __init__(self, length, height=0.03):
        self.length = float(length)
        self.height = float(height)
        self.cells = []

    def set_cell(self, cell):
        self.cells.append(np.array(cell, copy=True))

    def evaluate(self, q):
        q = coordinates(q)
        phase = 2 * np.pi * q[:, 0, 0] / self.length
        matrix = np.zeros((len(q), 2, 2))
        matrix[:, 0, 0] = 0.5 * self.height * (1 - np.cos(phase))
        gradients = np.zeros((len(q), 2, 2, 1, 3))
        gradients[:, 0, 0, 0, 0] = np.pi * self.height * np.sin(phase) / self.length
        return DiabaticResult(matrix, gradients)

    def to_dict(self):
        return {"model": "t26-periodic", "length": self.length, "height": self.height}


class ZeroDiabatic(Potential):
    nstates = 2
    natoms = 1
    metadata = {"provider": "t26-zero", "batch_independent": True}

    def evaluate(self, q):
        q = coordinates(q)
        return DiabaticResult(
            np.zeros((len(q), 2, 2)), np.zeros((len(q), 2, 2, 1, 3))
        )

    def to_dict(self):
        return {"model": "t26-zero"}


class CountingPotential(ZeroDiabatic):
    def __init__(self):
        self.calls = 0
        self.cells = 0

    def evaluate(self, q):
        self.calls += 1
        return super().evaluate(q)

    def set_cell(self, cell):
        self.cells += 1


class DummyAuxiliary:
    nstates = 3
    split_points = frozenset({AuxiliarySplitPoint.SYMMETRIC_AROUND_FREE})
    compatibility = AuxiliaryCompatibility(
        rationale="Dummy validation operator has no physical couplings."
    )

    def __init__(self, layout, count):
        self.layout = layout
        self.values = np.zeros((count, 3))

    @property
    def state_arrays(self):
        return {"values": self.values}

    def propagate(self, dt, q):
        self.values += dt

    def nuclear_force(self, q):
        return np.zeros_like(q)

    def energy(self, q):
        return 0.0

    def state_dict(self):
        return {"version": 1, "values": self.values.tolist()}

    def restore_state_dict(self, state):
        self.values[:] = state["values"]

    def identity(self):
        return {"kind": "dummy", "layout": self.layout.value}


def _focused_state(q, p=None):
    q = np.asarray(q, dtype=float)
    p = np.zeros_like(q) if p is None else np.asarray(p, dtype=float)
    mapping_q = np.tile([np.sqrt(3.0), 1.0], (len(q), 1))
    mapping_p = np.zeros_like(mapping_q)
    return NRPMDState(q, p, mapping_q=mapping_q, mapping_p=mapping_p)


def test_protocol_accepts_per_ring_and_per_bead_layouts():
    validate_auxiliary_operator(
        DummyAuxiliary(AuxiliaryLayout.PER_RING, 1), nbeads=8, checkpoint=True
    )
    validate_auxiliary_operator(
        DummyAuxiliary(AuxiliaryLayout.PER_BEAD, 8), nbeads=8, checkpoint=True
    )
    with pytest.raises(ValueError, match="leading dimension 8"):
        validate_auxiliary_operator(
            DummyAuxiliary(AuxiliaryLayout.PER_BEAD, 1), nbeads=8
        )


def test_nrpmd_auxiliary_core_is_bitwise_legacy_equivalent_for_10000_steps():
    rng = np.random.default_rng(7281)
    beads = 4
    masses = np.array([1836.15267343])
    beta, dt = 32.0, 0.013
    q = rng.normal(scale=0.2, size=(beads, 1, 3))
    p = rng.normal(scale=0.1, size=q.shape)
    x = rng.normal(size=(beads, 2))
    y = rng.normal(size=(beads, 2))
    modern = NRPMDState(q, p, mapping_q=x, mapping_p=y)
    legacy = modern.copy()
    dynamics = NRPMDDynamics(LinearDiabatic(), masses, beta, beads, dt, 2, native=False)
    legacy_free = CayleyIntegrator(
        masses, beta, beads, dt, native=False, scheme="cayley-bcb"
    )

    for _ in range(10_000):
        _mint(legacy, dynamics._evaluate(legacy.q), 0.5 * dt)
        legacy_free.free_step(legacy.q, legacy.p)
        _mint(legacy, dynamics._evaluate(legacy.q), 0.5 * dt)
        legacy.step += 1
        legacy.time += dt
        dynamics.step(modern)

    assert modern.step == legacy.step == 10_000
    assert modern.time == legacy.time
    for name in ("q", "p", "mapping_q", "mapping_p"):
        np.testing.assert_array_equal(getattr(modern, name), getattr(legacy, name))
    assert dynamics.energy(modern) == dynamics.energy(legacy)


def test_periodic_nrpmd_force_matches_finite_difference_and_translation():
    length = 7.0
    potential = PeriodicDiabatic(length)
    dynamics = NRPMDDynamics(
        potential, [2.0], beta=8.0, nbeads=1, dt=0.01, nstates=2,
        native=False, cell=np.diag([length, 10.0, 11.0]),
        pbc=(True, False, False),
    )
    state = _focused_state([[[length + 0.37, 0.2, -0.1]]])
    operator = NRPMDAuxiliaryOperator(dynamics, state)
    force = operator.nuclear_force(state.q)[0, 0, 0]
    eps = 1e-6
    plus, minus = state.q.copy(), state.q.copy()
    plus[0, 0, 0] += eps
    minus[0, 0, 0] -= eps
    finite_difference = -(operator.energy(plus) - operator.energy(minus)) / (2 * eps)
    assert force == pytest.approx(finite_difference, rel=3e-9, abs=3e-11)
    shifted = state.q.copy()
    shifted[0, 0, 0] += length
    assert operator.energy(shifted) == pytest.approx(operator.energy(state.q), abs=1e-15)
    np.testing.assert_allclose(
        operator.nuclear_force(shifted), operator.nuclear_force(state.q), atol=1e-14
    )
    assert len(potential.cells) == 1


def test_nrpmd_rpc_mts_has_bounded_harmonic_energy_error():
    beads = 8
    scalar = SplitPotential(
        HarmonicPotential(k=0.07), HarmonicPotential(k=0.013),
        beads, contracted_beads=1,
    )
    dynamics = NRPMDDynamics(
        ZeroDiabatic(), [3.0], beta=12.0, nbeads=beads, dt=0.004,
        nstates=2, native=False, nuclear_potential=scalar,
        force_splitting=ForceSplitting(inner_steps=4),
    )
    rng = np.random.default_rng(91)
    state = _focused_state(
        rng.normal(scale=0.15, size=(beads, 1, 3)),
        rng.normal(scale=0.08, size=(beads, 1, 3)),
    )
    initial = dynamics.energy(state)
    maximum = 0.0
    for _ in range(3000):
        dynamics.step(state)
        maximum = max(maximum, abs(dynamics.energy(state) - initial))
    assert maximum / max(1.0, abs(initial)) < 2e-5


@pytest.mark.parametrize("thermostat", ["andersen", "pile-l"])
def test_nuclear_preparation_freezes_mapping_variables(thermostat):
    beads = 4 if thermostat == "pile-l" else 1
    dynamics = NRPMDDynamics(
        ZeroDiabatic(), [5.0], beta=10.0, nbeads=beads, dt=0.01,
        nstates=2, native=False, seed=782,
    )
    state = _focused_state(np.zeros((beads, 1, 3)))
    before_q, before_p = state.mapping_q.copy(), state.mapping_p.copy()
    dynamics.thermalize(state, 3, thermostat=thermostat)
    np.testing.assert_array_equal(state.mapping_q, before_q)
    np.testing.assert_array_equal(state.mapping_p, before_p)
    assert state.step == 3
    assert np.isfinite(state.q).all() and np.isfinite(state.p).all()


def test_andersen_preparation_momentum_variance_matches_m_p_over_beta():
    mass, beta, beads = 7.0, 9.0, 2
    dynamics = NRPMDDynamics(
        ZeroDiabatic(), [mass], beta=beta, nbeads=beads, dt=0.01,
        nstates=2, native=False, seed=4102,
    )
    samples = []
    for _ in range(1024):
        state = _focused_state(np.zeros((beads, 1, 3)))
        dynamics.thermalize(state, 0, thermostat="andersen", resample=True)
        samples.append(state.p.copy())
    values = np.asarray(samples).reshape(-1)
    expected = mass * beads / beta
    assert np.mean(values) == pytest.approx(0.0, abs=0.08 * np.sqrt(expected))
    assert np.var(values, ddof=1) == pytest.approx(expected, rel=0.08)


def test_npt_is_rejected_before_provider_side_effects():
    potential = CountingPotential()
    with pytest.raises(ValueError, match="does not whitelist barostat='bzp'"):
        NRPMDDynamics(
            potential, [1.0], 10.0, 1, 0.01, 2, native=False,
            cell=np.eye(3), pbc=(True, True, True),
            barostat={"mode": "bzp", "pressure": 0.0, "tau": 10.0},
        )
    assert potential.calls == 0
    assert potential.cells == 0


def test_input_rejects_nrpmd_npt_before_output_directory(tmp_path):
    output = tmp_path / "must-not-exist"
    raw = {
        "schema_version": 1, "task": "nonadiabatic", "method": "nrpmd",
        "temperature": 300.0, "beads": 1,
        "model": {"name": "tully1"}, "system": {"masses": [1.0]},
        "initial": {"kind": "point"}, "ensemble": {"steps": 0},
        "dynamics": {
            "native": False,
            "barostat": {"mode": "bzp", "pressure": 0.0, "tau": 10.0},
        },
        "output": {"directory": str(output)},
    }
    with pytest.raises(InputError, match="Unknown dynamics fields: barostat"):
        resolve_nonadiabatic_job(raw, base_directory=tmp_path)
    assert not output.exists()


def test_versioned_operator_checkpoint_roundtrip_and_legacy_rng_restore():
    dynamics = NRPMDDynamics(
        LinearDiabatic(), [2.0], 11.0, 2, 0.01, 2, native=False, seed=918
    )
    state = _focused_state(np.zeros((2, 1, 3)))
    saved = dynamics.operator_state_dict(state)
    assert saved["version"] == 2
    expected_q, expected_p = state.mapping_q.copy(), state.mapping_p.copy()
    state.mapping_q[:] = 0.0
    state.mapping_p[:] = 0.0
    dynamics.rng.random(10)
    result = dynamics.load_operator_state_dict(saved, state)
    assert result == {"legacy": False, "version": 2}
    np.testing.assert_array_equal(state.mapping_q, expected_q)
    np.testing.assert_array_equal(state.mapping_p, expected_p)
    expected_draw = np.random.default_rng()
    expected_draw.bit_generator.state = copy.deepcopy(saved["rng"])
    assert dynamics.rng.random() == expected_draw.random()

    mapping = (state.mapping_q.copy(), state.mapping_p.copy())
    legacy_rng = copy.deepcopy(np.random.default_rng(51).bit_generator.state)
    assert dynamics.load_operator_state_dict(legacy_rng, state)["legacy"] is True
    np.testing.assert_array_equal(state.mapping_q, mapping[0])
    np.testing.assert_array_equal(state.mapping_p, mapping[1])
