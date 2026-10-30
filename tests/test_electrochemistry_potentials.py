"""Independent charging-model limits and the existing PES/batch contract."""

import numpy as np
import pytest

from pyrpmd.constants import EV_TO_AU
from pyrpmd.electrochemistry import (
    ElectronReservoir, ElectronicPESResult, AdiabaticElectrodePotential,
    FixedElectronNumberPotential, evaluate_electronic, build_adiabatic_electrode,
)
from pyrpmd.potentials import PotentialError, PythonPotential, check_potential, make_potential
from pyrpmd.system import System


class ChargingModel:
    """E=.5*k*q² + .5*a*N² - c*N*x, a positive coupled Hessian."""

    def __init__(self, k=0.7, a=1.3, c=0.4, number_gauge=0.0, offset=0.0):
        self.k, self.a, self.c = k, a, c
        self.number_gauge, self.offset = number_gauge, offset

    def evaluate(self, q, electron_numbers):
        n = electron_numbers
        e = 0.5 * self.k * np.sum(q**2, axis=(1, 2)) + 0.5 * self.a * n**2
        e += (self.number_gauge - self.c * q[:, 0, 0]) * n + self.offset
        f = -self.k * q
        f[:, 0, 0] += self.c * n
        return ElectronicPESResult(e, f, self.a * n - self.c * q[:, 0, 0] + self.number_gauge)


def reservoir(mu=0.23, voltage=0.0):
    return ElectronReservoir(voltage, 0.0, mu, "model-reference")


@pytest.mark.parametrize("p", [1, 4, 17])
def test_adiabatic_charging_exact_energy_force_charge_and_batch_partition(p):
    model = ChargingModel()
    bath = reservoir()
    pes = AdiabaticElectrodePotential(model, bath, electron_bounds=(-5, 5))
    q = np.random.default_rng(37).normal(size=(p, 2, 3)) * 0.3
    solved = pes.solve(q)
    optimum = (model.c * q[:, 0, 0] + bath.chemical_potential_au) / model.a
    expected = 0.5 * model.k * np.sum(q**2, axis=(1, 2)) - 0.5 * model.a * optimum**2
    forces = -model.k * q
    forces[:, 0, 0] += model.c * optimum
    np.testing.assert_allclose(solved.electron_numbers, optimum, atol=2e-15)
    np.testing.assert_allclose(solved.pes.energy, expected, atol=2e-15)
    np.testing.assert_allclose(solved.pes.forces, forces, atol=2e-15)
    np.testing.assert_allclose(solved.chemical_potentials_au, bath.chemical_potential_au, atol=2e-15)
    assert solved.iterations == 1
    split = [pes.evaluate(frame[None]) for frame in q]
    np.testing.assert_allclose(np.concatenate([r.energy for r in split]), solved.pes.energy)
    np.testing.assert_allclose(np.concatenate([r.forces for r in split]), solved.pes.forces)
    assert check_potential(pes, q)["max_absolute_force_error"] < 1e-10


def test_adiabatic_model_is_explicitly_not_shared_number_minimization():
    q = np.zeros((2, 1, 3))
    q[:, 0, 0] = [-1, 1]
    pes = AdiabaticElectrodePotential(ChargingModel(), reservoir(0), electron_bounds=(-5, 5))
    r = pes.solve(q)
    assert r.electron_numbers[0] == pytest.approx(-0.4 / 1.3)
    assert r.electron_numbers[1] == pytest.approx(0.4 / 1.3)
    # The *shared* optimizer would be exactly zero, with mean energy k/2.
    assert np.mean(r.pes.energy) == pytest.approx(0.7 / 2 - 0.4**2 / (2 * 1.3))
    assert pes.metadata["electrode_approximation"] == "independent_frame_adiabatic_minimum"


def test_electronic_gauge_covariance_and_voltage_sign():
    q = np.random.default_rng(59).normal(size=(3, 1, 3))
    baseline = AdiabaticElectrodePotential(ChargingModel(), reservoir(), electron_bounds=(-5, 5)).solve(q)
    shifted = AdiabaticElectrodePotential(
        ChargingModel(number_gauge=0.9, offset=0.7), reservoir(0.23 + 0.9), electron_bounds=(-5, 5),
    ).solve(q)
    np.testing.assert_allclose(shifted.electron_numbers, baseline.electron_numbers)
    np.testing.assert_allclose(shifted.pes.forces, baseline.pes.forces)
    np.testing.assert_allclose(shifted.pes.energy, baseline.pes.energy + 0.7)
    positive_voltage = AdiabaticElectrodePotential(ChargingModel(), reservoir(voltage=1.0), electron_bounds=(-5, 5)).solve(q)
    np.testing.assert_allclose(positive_voltage.electron_numbers, baseline.electron_numbers - EV_TO_AU / 1.3)


def test_nonlinear_convex_electronic_surface_matches_independent_root_and_force():
    from scipy.optimize import brentq

    class Quartic(ChargingModel):
        def evaluate(self, q, electron_numbers):
            result = super().evaluate(q, electron_numbers)
            result.energy += 0.2 * electron_numbers**4
            result.denergy_dnumber += 0.8 * electron_numbers**3
            return result

    q = np.array([[[-0.7, 0.2, 0]], [[2.0, -0.5, 0]]])
    pes = AdiabaticElectrodePotential(Quartic(), reservoir(0.11), electron_bounds=(-2, 3), chemical_potential_tolerance_au=1e-13)
    result = pes.solve(q)
    expected = [brentq(lambda n: 1.3 * n + 0.8 * n**3 - 0.4 * x - 0.11, -2, 3) for x in q[:, 0, 0]]
    np.testing.assert_allclose(result.electron_numbers, expected, atol=1e-12)
    assert check_potential(pes, q)["max_absolute_force_error"] < 1e-9


@pytest.mark.parametrize("bounds", [(0.5, 1), (-1, -0.1), (0, 1)])
def test_unbracketed_or_boundary_minimum_never_silently_clamps(bounds):
    pes = AdiabaticElectrodePotential(ChargingModel(), reservoir(0), electron_bounds=bounds)
    with pytest.raises(PotentialError, match="not bracketed"):
        pes.evaluate(np.zeros((1, 1, 3)))


def test_solver_failure_does_not_reuse_previous_geometry_solution():
    pes = AdiabaticElectrodePotential(ChargingModel(), reservoir(), electron_bounds=(-1, 1))
    q = np.zeros((2, 1, 3))
    good = pes.evaluate(q).energy.copy()
    bad = q.copy()
    bad[1, 0, 0] = 100
    with pytest.raises(PotentialError, match="not bracketed"):
        pes.evaluate(bad)
    np.testing.assert_array_equal(pes.evaluate(q).energy, good)


def test_insufficient_iteration_budget_and_nonmonotone_derivative_fail():
    class Quartic(ChargingModel):
        def evaluate(self, q, electron_numbers):
            r = super().evaluate(q, electron_numbers)
            r.denergy_dnumber += electron_numbers**3
            return r

    with pytest.raises(PotentialError, match="max_iterations"):
        AdiabaticElectrodePotential(Quartic(), reservoir(), electron_bounds=(-2, 4), max_iterations=1).evaluate(np.zeros((1, 1, 3)))

    class Nonmonotone(ChargingModel):
        def evaluate(self, q, electron_numbers):
            r = super().evaluate(q, electron_numbers)
            r.denergy_dnumber = electron_numbers + 20 * np.exp(-electron_numbers**2)
            return r

    with pytest.raises(PotentialError, match="not monotone"):
        AdiabaticElectrodePotential(Nonmonotone(), reservoir(0), electron_bounds=(-5, 5)).evaluate(np.zeros((1, 1, 3)))


@pytest.mark.parametrize("field,bad", [
    ("denergy_dnumber", [np.nan]), ("denergy_dnumber", [1j]),
    ("denergy_dnumber", [1, 2]), ("energy", [np.inf]),
    ("forces", [[[np.nan, 0, 0]]]), ("status", [1]),
])
def test_failed_provider_results_are_rejected(field, bad):
    class Bad(ChargingModel):
        def evaluate(self, q, electron_numbers):
            r = super().evaluate(q, electron_numbers)
            setattr(r, field, np.asarray(bad))
            return r

    with pytest.raises(PotentialError):
        evaluate_electronic(Bad(), np.zeros((1, 1, 3)), 0.0)


@pytest.mark.parametrize("number", [True, np.nan, [0, 1], [[0]], [1j]])
def test_bad_number_inputs(number):
    with pytest.raises(PotentialError):
        evaluate_electronic(ChargingModel(), np.zeros((1, 1, 3)), number)


def test_fixed_number_reuses_provider_validation_without_reservoir_shift():
    q = np.ones((3, 1, 3))
    model = ChargingModel()
    fixed = FixedElectronNumberPotential(model, 0.2)
    r = fixed.evaluate(q)
    exact = model.evaluate(q, np.full(3, 0.2))
    np.testing.assert_allclose(r.energy, exact.energy)
    assert check_potential(fixed, q)["max_absolute_force_error"] < 1e-9
    fixed.electron_number = 0.4
    assert not np.array_equal(fixed.evaluate(q).energy, r.energy)
    with pytest.raises(ValueError):
        fixed.electron_number = True


@pytest.mark.parametrize("arguments", [
    {"potential_V": True}, {"reference_potential_V": np.inf},
    {"chemical_potential_at_reference_au": np.nan}, {"potential_reference": " "},
])
def test_reservoir_requires_finite_explicit_reference(arguments):
    config = dict(potential_V=0.0, reference_potential_V=0.0, chemical_potential_at_reference_au=0.0, potential_reference="model")
    config.update(arguments)
    with pytest.raises(ValueError):
        ElectronReservoir(**config)


@pytest.mark.parametrize("options", [
    {"electron_bounds": (1, 1)}, {"electron_bounds": (True, False)},
    {"electron_bounds": (-np.inf, 1)}, {"number_tolerance": 0},
    {"chemical_potential_tolerance_au": np.nan}, {"max_iterations": True},
    {"max_iterations": 1.5},
])
def test_invalid_minimization_options(options):
    kwargs = dict(electron_bounds=(-1, 1))
    kwargs.update(options)
    with pytest.raises(ValueError):
        AdiabaticElectrodePotential(ChargingModel(), reservoir(), **kwargs)


def test_periodic_contract_requires_model_binding_and_never_claims_native_binding():
    system = System(("H",), [1837], [[0, 0, 0]], cell=np.eye(3) * 10, pbc=[True, True, False])
    pes = AdiabaticElectrodePotential(ChargingModel(), reservoir(), electron_bounds=(-1, 1))
    with pytest.raises(PotentialError, match="periodic|Periodic"):
        make_potential(pes, system)

    class Periodic(ChargingModel):
        metadata = {"supports_pbc": True, "native_binding": "must-not-propagate"}

        def bind_system(self, system):
            self.bound_cell = system.cell.copy()

    model = Periodic()
    bound = make_potential(AdiabaticElectrodePotential(model, reservoir(), electron_bounds=(-1, 1)), system)
    np.testing.assert_array_equal(model.bound_cell, system.cell)
    assert not hasattr(bound, "native_binding")
    assert "native_binding" not in bound.metadata


def test_factory_failure_closes_constructed_provider(monkeypatch):
    model = ChargingModel()
    closed = []
    model.close = lambda: closed.append(True)
    monkeypatch.setattr("pyrpmd.electrochemistry.potentials.import_object", lambda value: lambda **kw: model)
    with pytest.raises(ValueError):
        build_adiabatic_electrode(
            model={"factory": "unused:builder"},
            reservoir=dict(potential_V=0, reference_potential_V=0, chemical_potential_at_reference_au=0, potential_reference="model"),
            electron_bounds=(1, 1),
        )
    assert closed == [True]


@pytest.mark.parametrize("options", [
    {"units": {"positions": "angstrom"}}, {"units": {"energy": "eV"}}, {"derivative": "gradient"},
])
def test_au_electrode_factory_cannot_silently_rescale_internal_geometry_or_force(options):
    pes = AdiabaticElectrodePotential(ChargingModel(), reservoir(), electron_bounds=(-1, 1))
    with pytest.raises(PotentialError, match="atomic units"):
        PythonPotential(pes, **options)


def test_outer_factory_contract_failure_closes_its_owned_model(monkeypatch):
    model = ChargingModel()
    closed = []
    model.close = lambda: closed.append(True)
    pes = AdiabaticElectrodePotential(model, reservoir(), electron_bounds=(-1, 1))
    monkeypatch.setattr("pyrpmd.potentials.import_object", lambda value: lambda **kw: pes)
    with pytest.raises(PotentialError, match="atomic units"):
        make_potential({"provider": "python", "factory": "unused:builder", "units": {"energy": "eV"}})
    assert closed == [True]
