"""Analytic charging limits, matrix derivatives, and NRPMD integration."""

import numpy as np
import pytest

from pyrpmd.constants import EV_TO_AU
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.electrochemistry import (
    AdiabaticElectrodePotential, ConstantPotentialDiabaticPotential,
    ConstantPotentialDiabaticResult, ElectronReservoir, ElectronicPESResult,
)
from pyrpmd.nonadiabatic import DiabaticResult, NRPMDState
from pyrpmd.potentials.base import PotentialError
from pyrpmd.potentials.diabatic import DiabaticPythonPotential, check_diabatic_potential
from pyrpmd.thermal_nrpmd import ThermalNRPMDDynamics, ThermalNRPMDSampler, thermal_weight


class ChargingModel:
    """E_charge = k R²/2 + (N-a x)²/(2 C), using reusable buffers."""

    k, a, capacitance = 0.7, 0.3, 1.4

    def __init__(self, periodic=False):
        self.metadata = {"supports_pbc": periodic, "native_context": "do-not-forward"}
        self.bound = []
        self.closed = 0
        self.buffers = None

    def evaluate(self, q, electron_numbers):
        shape = q.shape
        if self.buffers is None or self.buffers[1].shape != shape:
            self.buffers = (np.empty(len(q)), np.empty_like(q), np.empty(len(q)))
        energy, forces, derivative = self.buffers
        excess = electron_numbers - self.a * q[:, 0, 0]
        energy[:] = self.k / 2 * np.sum(q*q, axis=(1, 2)) + excess**2 / (2*self.capacitance)
        forces[:] = -self.k * q
        forces[:, 0, 0] += self.a * excess / self.capacitance
        derivative[:] = excess / self.capacitance
        return ElectronicPESResult(energy, forces, derivative)

    def bind_system(self, system):
        self.bound.append(system)

    def close(self):
        self.closed += 1


class MatrixModel:
    """Linear two-state matrix with optional geometry-dependent complex coupling."""

    def __init__(self, complex_coupling=False, periodic=False):
        self.complex_coupling = complex_coupling
        self.metadata = {"supports_pbc": periodic, "native_context": "do-not-forward"}
        self.bound, self.closed = [], 0
        self.buffers = None

    def evaluate(self, q):
        dtype = complex if self.complex_coupling else float
        if self.buffers is None or self.buffers[1].shape != (len(q), 2, 2, *q.shape[1:]):
            self.buffers = (np.empty((len(q), 2, 2), dtype=dtype),
                            np.empty((len(q), 2, 2, *q.shape[1:]), dtype=dtype))
        matrix, gradient = self.buffers
        gradient.fill(0)
        matrix[:, 0, 0] = 0.2 + 0.4*q[:, 0, 0]
        matrix[:, 1, 1] = -0.1 - 0.2*q[:, 0, 0]
        matrix[:, 0, 1] = 0.05 + 0.03*q[:, 0, 1]
        gradient[:, 0, 0, 0, 0] = 0.4
        gradient[:, 1, 1, 0, 0] = -0.2
        gradient[:, 0, 1, 0, 1] = 0.03
        if self.complex_coupling:
            matrix[:, 0, 1] += 1j*(0.02 + 0.07*q[:, 0, 2])
            gradient[:, 0, 1, 0, 2] = 0.07j
        matrix[:, 1, 0] = matrix[:, 0, 1].conj()
        gradient[:, 1, 0] = gradient[:, 0, 1].conj()
        return DiabaticResult(matrix, gradient)

    def bind_system(self, system):
        self.bound.append(system)

    def close(self):
        self.closed += 1


def make_potential(*, voltage=0.0, matrix=None, charging=None, bounds=(-3, 3)):
    matrix = MatrixModel() if matrix is None else matrix
    charging = ChargingModel() if charging is None else charging
    bath = ElectronReservoir(voltage, 0.0, 0.13, "analytic-model-reference")
    electrode = AdiabaticElectrodePotential(charging, bath, electron_bounds=bounds)
    return ConstantPotentialDiabaticPotential(matrix, electrode)


@pytest.mark.parametrize("beads", [1, 4, 7])
def test_analytic_matrix_grand_potential_and_batch_independence(beads):
    potential = make_potential()
    q = np.random.default_rng(16).normal(scale=0.3, size=(beads, 2, 3))
    result = potential.solve(q)
    assert isinstance(result, ConstantPotentialDiabaticResult)
    mu = 0.13
    expected_n = 0.3*q[:, 0, 0] + 1.4*mu
    expected_omega = 0.35*np.sum(q*q, axis=(1, 2)) - mu*0.3*q[:, 0, 0] - 0.7*mu*mu
    expected_gradient = 0.7*q
    expected_gradient[:, 0, 0] -= 0.3*mu
    bare = MatrixModel().evaluate(q)
    np.testing.assert_allclose(result.electron_numbers, expected_n, atol=1e-14)
    np.testing.assert_allclose(result.electrode.pes.energy, expected_omega, atol=1e-14)
    np.testing.assert_allclose(result.electrode.pes.forces, -expected_gradient, atol=1e-14)
    np.testing.assert_allclose(result.chemical_potential_residuals_au, 0, atol=1e-14)
    for state in range(2):
        np.testing.assert_allclose(result.diabatic.matrix[:, state, state],
                                   bare.matrix[:, state, state] + expected_omega, atol=1e-14)
        np.testing.assert_allclose(result.diabatic.gradients[:, state, state],
                                   bare.gradients[:, state, state] + expected_gradient, atol=1e-14)
    np.testing.assert_array_equal(result.diabatic.matrix[:, 0, 1], bare.matrix[:, 0, 1])
    split = [potential.evaluate(frame[None]) for frame in q]
    np.testing.assert_allclose(np.concatenate([r.matrix for r in split]), result.diabatic.matrix)
    np.testing.assert_allclose(np.concatenate([r.gradients for r in split]), result.diabatic.gradients)


@pytest.mark.parametrize("complex_coupling", [False, True])
def test_all_matrix_gradients_match_finite_differences(complex_coupling):
    potential = make_potential(matrix=MatrixModel(complex_coupling=complex_coupling))
    q = np.random.default_rng(28).normal(scale=0.3, size=(3, 2, 3))
    error = check_diabatic_potential(potential, q, nstates=2, epsilon=2e-5)
    assert error["max_absolute_gradient_error"] < 2e-10


def test_voltage_changes_charge_and_common_force_with_correct_sign():
    q = np.random.default_rng(49).normal(scale=0.2, size=(3, 1, 3))
    before = make_potential(voltage=0.0).solve(q)
    after = make_potential(voltage=0.8).solve(q)
    delta_mu = -0.8*EV_TO_AU
    np.testing.assert_allclose(after.electron_numbers-before.electron_numbers, 1.4*delta_mu, atol=2e-14)
    difference = after.diabatic.gradients-before.diabatic.gradients
    expected = np.zeros_like(difference)
    expected[:, 0, 0, 0, 0] = expected[:, 1, 1, 0, 0] = -0.3*delta_mu
    np.testing.assert_allclose(difference, expected, atol=2e-14)
    # Common charging cannot manufacture a change in diabatic or adiabatic gaps.
    gaps_before = np.diff(np.linalg.eigvalsh(before.diabatic.matrix), axis=1)
    gaps_after = np.diff(np.linalg.eigvalsh(after.diabatic.matrix), axis=1)
    np.testing.assert_allclose(gaps_after, gaps_before, atol=2e-14)


def test_complex_couplings_and_python_matrix_adapter_are_preserved():
    bare = MatrixModel(complex_coupling=True)
    potential = make_potential(matrix=bare)
    q = np.random.default_rng(55).normal(scale=0.2, size=(4, 1, 3))
    result = potential.evaluate(q)
    original = bare.evaluate(q)
    np.testing.assert_array_equal(result.matrix[:, 0, 1], original.matrix[:, 0, 1])
    np.testing.assert_array_equal(result.gradients[:, 0, 1], original.gradients[:, 0, 1])
    assert np.iscomplexobj(result.matrix)
    adapted = DiabaticPythonPotential(potential).evaluate(q)
    np.testing.assert_array_equal(adapted.matrix, result.matrix)
    np.testing.assert_array_equal(adapted.gradients, result.gradients)
    with pytest.raises(PotentialError, match="real symmetric"):
        ThermalNRPMDSampler(potential, [1.5], 2.0, q, 2)


def test_results_do_not_alias_provider_buffers_or_later_results():
    potential = make_potential(matrix=MatrixModel(complex_coupling=True))
    q = np.random.default_rng(67).normal(scale=0.2, size=(2, 1, 3))
    result = potential.solve(q)
    arrays = [result.diabatic.matrix, result.diabatic.gradients,
              result.electrode.pes.energy, result.electrode.pes.forces,
              result.electron_numbers, result.chemical_potentials_au]
    saved = [value.copy() for value in arrays]
    for value in arrays:
        for buffer in (*potential.matrix_potential.buffers, *potential.electrode_potential.model.buffers):
            assert not np.shares_memory(value, buffer)
    # The adapter must not add its charging term to the underlying matrix buffers.
    untouched = MatrixModel(complex_coupling=True).evaluate(q)
    np.testing.assert_array_equal(potential.matrix_potential.buffers[0], untouched.matrix)
    potential.solve(q + 0.1)
    for value, expected in zip(arrays, saved):
        np.testing.assert_array_equal(value, expected)
    assert result.chemical_potential_residuals_au is not result.chemical_potentials_au


@pytest.mark.parametrize("states", [1, 2])
def test_thermal_scalar_force_is_independent_of_mapping_norm(states):
    class ConstantMatrix:
        def evaluate(self, q):
            return DiabaticResult(np.zeros((len(q), states, states)),
                                  np.zeros((len(q), states, states, *q.shape[1:])))

    potential = make_potential(matrix=ConstantMatrix())
    rng = np.random.default_rng(78)
    q, p = rng.normal(scale=0.1, size=(2, 4, 1, 3))
    nuclear = State(q, p)
    # Deliberately non-focused, non-unit, bead-dependent mapping norms.
    mapped = NRPMDState(q, p, mapping_q=2*rng.normal(size=(4, states)),
                        mapping_p=0.3*rng.normal(size=(4, states)))
    bare_dyn = Dynamics(potential.electrode_potential, [2.3], 3.0, 4, 0.02, native=False)
    matrix_dyn = ThermalNRPMDDynamics(potential, [2.3], 3.0, 4, 0.02, states, native=False)
    for _ in range(6):
        bare_dyn.step(nuclear)
        matrix_dyn.step(mapped)
    np.testing.assert_allclose(mapped.q, nuclear.q, atol=2e-14)
    np.testing.assert_allclose(mapped.p, nuclear.p, atol=2e-14)
    assert matrix_dyn.energy(mapped) == pytest.approx(bare_dyn.energy(nuclear), abs=2e-14)


def test_thermal_weight_contains_common_grand_potential_once_per_bead():
    potential = make_potential()
    rng = np.random.default_rng(91)
    state = NRPMDState(rng.normal(scale=0.1, size=(3, 1, 3)), np.zeros((3, 1, 3)),
                       mapping_q=rng.normal(size=(3, 2)), mapping_p=rng.normal(size=(3, 2)))
    beta = 2.4
    bare = thermal_weight(potential.matrix_potential, state, [1.8], beta, 2)
    charged = thermal_weight(potential, state, [1.8], beta, 2)
    assert bare.sign != 0
    assert charged.sign == bare.sign
    omega = potential.electrode_potential.evaluate(state.q).energy
    assert charged.log_density-bare.log_density == pytest.approx(-beta*np.mean(omega), abs=2e-14)
    np.testing.assert_allclose(charged.electronic_populations, bare.electronic_populations, atol=2e-14)


def test_electrode_failure_propagates_and_dynamics_rolls_back():
    potential = make_potential(bounds=(-0.2, 0.2))
    q = np.zeros((2, 1, 3))
    q[:, 0, 0] = 0.05
    p = np.zeros_like(q)
    p[:, 0, 0] = 10
    state = NRPMDState(q, p, mapping_q=np.ones((2, 2)), mapping_p=np.ones((2, 2)))
    saved = state.copy()
    dynamics = ThermalNRPMDDynamics(potential, [1.0], 2.0, 2, 0.1, 2, native=False)
    # Initial optimum is in bounds; the second force evaluation leaves the domain.
    potential.evaluate(q)
    with pytest.raises(PotentialError, match="not bracketed"):
        dynamics.step(state)
    for key in ("q", "p", "mapping_q", "mapping_p"):
        np.testing.assert_array_equal(getattr(state, key), getattr(saved, key))
    assert (state.step, state.time) == (saved.step, saved.time)


def test_malformed_matrix_and_failed_electronic_provider_are_rejected():
    class BadMatrix(MatrixModel):
        def evaluate(self, q):
            result = super().evaluate(q)
            result.matrix[:, 0, 1] += 1.0
            return result

    class BadCharge(ChargingModel):
        def evaluate(self, q, electron_numbers):
            result = super().evaluate(q, electron_numbers)
            result.status = np.ones(len(q), dtype=int)
            return result

    q = np.zeros((2, 1, 3))
    with pytest.raises(PotentialError, match="Hermitian"):
        make_potential(matrix=BadMatrix()).evaluate(q)
    with pytest.raises(PotentialError, match="frames failed"):
        make_potential(charging=BadCharge()).evaluate(q)


def test_periodic_binding_requires_and_forwards_both_components():
    system = {"cell": np.eye(3)*8, "pbc": [True, True, False]}
    matrix, charging = MatrixModel(periodic=True), ChargingModel(periodic=True)
    potential = make_potential(matrix=matrix, charging=charging)
    assert potential.metadata["supports_pbc"] is True
    potential.bind_system(system)
    assert matrix.bound == charging.bound == [system]
    for matrix_periodic, charge_periodic in ((False, True), (True, False)):
        matrix = MatrixModel(periodic=matrix_periodic)
        charging = ChargingModel(periodic=charge_periodic)
        potential = make_potential(matrix=matrix, charging=charging)
        with pytest.raises(PotentialError, match="both require"):
            potential.bind_system(system)
        assert matrix.bound == charging.bound == []


def test_components_are_borrowed_and_no_native_context_is_forwarded():
    potential = make_potential()
    assert "native_context" not in potential.metadata
    assert not hasattr(potential, "native_context")
    assert potential.metadata["constant_potential_matrix_model"] == "state_independent_charging"
    potential.close()
    assert potential.matrix_potential.closed == potential.electrode_potential.model.closed == 0


@pytest.mark.parametrize("which,flag", [
    ("matrix", "stateful"), ("matrix", "time_dependent"),
    ("charge", "history_dependent"), ("charge", "ring_polymer_coupled"),
])
def test_advertised_nonstatic_or_coupled_components_are_rejected(which, flag):
    matrix, charging = MatrixModel(), ChargingModel()
    (matrix if which == "matrix" else charging).metadata[flag] = True
    with pytest.raises(ValueError, match="independent|deterministic"):
        make_potential(matrix=matrix, charging=charging)


def test_only_explicit_adiabatic_electrode_component_is_accepted():
    with pytest.raises(TypeError, match="AdiabaticElectrodePotential"):
        ConstantPotentialDiabaticPotential(MatrixModel(), ChargingModel())
    with pytest.raises(TypeError, match="matrix_potential"):
        make_potential(matrix=object())
