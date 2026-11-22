"""Exact coupled-electronic-state subspace rates and 1D propagation diabatization."""

import json

import numpy as np
import pytest

from pyrpmd.analysis.multistate_subspace import (
    MultistateRouteUnavailable,
    MultistateSubspaceModel,
    exact_multistate_rate,
    multistate_model_from_function,
    nonadiabatic_subspace_factor,
    propagation_diabatization,
)
from pyrpmd.analysis.subspace_model import SubspaceModel, model_from_function
from pyrpmd.analysis.subspace_quantum import exact_quantum_rate

KELVIN = 3.166811563e-6
BETA = 1.0 / (1000.0 * KELVIN)
ECKART_MASS, V0, A = 1060.0, 0.425 / 27.211386, 0.66
MASS = 2000.0


def eckart(points):
    return V0 / np.cosh(points[..., 0] / A) ** 2


def crossing(height, slope_width, coupling, coupling_width, phase=0.0):
    """Symmetric two-diabat crossing: reactant on state 0 at s -> -inf, product on state 1."""

    def function(points):
        s = points[..., 0]
        rise = np.tanh(s / slope_width)
        out = np.zeros(s.shape + (2, 2), dtype=complex if phase else float)
        out[..., 0, 0] = 0.5 * height * (1.0 + rise)
        out[..., 1, 1] = 0.5 * height * (1.0 - rise)
        out[..., 0, 1] = coupling * np.exp(-((s / coupling_width) ** 2)) * np.exp(1j * phase if phase else 0.0)
        out[..., 1, 0] = np.conj(out[..., 0, 1])
        return out

    return function


def rotated(model, unitary):
    values = np.einsum("ia,...ij,jb->...ab", unitary.conj(), model.diabatic, unitary)
    return MultistateSubspaceModel(model.axes, values, model.masses, model.dividing_surface)


@pytest.mark.parametrize("temperature", [1000.0, 300.0])
def test_one_state_reproduces_single_surface_dense_rate(temperature):
    beta = 1.0 / (temperature * KELVIN)
    axis = np.linspace(-8.0, 8.0, 321)
    single = exact_quantum_rate(model_from_function(eckart, [axis], masses=ECKART_MASS), beta)
    coupled = exact_multistate_rate(
        multistate_model_from_function(lambda p: eckart(p)[..., None, None], [axis], masses=ECKART_MASS), beta)
    assert coupled.rate == pytest.approx(single.rate, rel=1e-12)
    assert coupled.reactant_partition == pytest.approx(single.reactant_partition, rel=1e-12)
    assert coupled.channel_rates["adiabatic0"] == pytest.approx(coupled.rate, rel=1e-9)
    assert abs(coupled.diagnostics["channel_sum_residual"]) < 1e-9


def test_one_state_reproduces_well_and_two_dimensional_rates():
    def well(points):
        return 0.01 * ((points[..., 0] / 1.2) ** 2 - 1.0) ** 2 + 0.002 * points[..., 0]

    axis = np.linspace(-4.0, 4.0, 201)
    options = dict(masses=1836.0, reactant="well", reactant_reference=-1.2)
    single = exact_quantum_rate(model_from_function(well, [axis], **options), BETA)
    coupled = exact_multistate_rate(multistate_model_from_function(lambda p: well(p)[..., None, None], [axis],
                                                                   **options), BETA)
    assert coupled.rate == pytest.approx(single.rate, rel=1e-10)

    def bath(points):
        frequency = 0.004
        stiffening = 1.0 + 0.3 / np.cosh(points[..., 0] / A) ** 2
        return eckart(points) + 0.5 * ECKART_MASS * frequency**2 * points[..., 1] ** 2 * stiffening

    axes = [np.linspace(-6.0, 6.0, 41), np.linspace(-1.2, 1.2, 13)]
    single = exact_quantum_rate(model_from_function(bath, axes, masses=ECKART_MASS), BETA)
    coupled = exact_multistate_rate(multistate_model_from_function(lambda p: bath(p)[..., None, None], axes,
                                                                   masses=ECKART_MASS), BETA)
    assert coupled.rate == pytest.approx(single.rate, rel=1e-10)


def test_uncoupled_states_add_single_surface_fluxes_channel_by_channel():
    axis = np.linspace(-8.0, 8.0, 161)
    model = multistate_model_from_function(crossing(0.02, 0.5, 0.0, 1.0), [axis], masses=MASS)
    coupled = exact_multistate_rate(model, BETA, channels="diabatic")
    singles = [exact_quantum_rate(model.diabatic_surface(i), BETA, method="sylvester") for i in range(2)]

    def unshifted(flux, result):
        # Fluxes carry exp(beta * energy_shift); compare them on a common origin.
        return flux * np.exp(-BETA * result.diagnostics["energy_shift"])

    total = sum(unshifted(s.flux, s) for s in singles)
    assert unshifted(coupled.flux, coupled) == pytest.approx(total, rel=1e-9)
    for label, single in zip(model.state_labels, singles):
        channel = coupled.diagnostics["channel_flux"][label]
        assert unshifted(channel, coupled) == pytest.approx(unshifted(single.flux, single), rel=1e-8)


def test_constant_electronic_rotation_and_coupling_phase_leave_rates_invariant():
    axis = np.linspace(-8.0, 8.0, 161)
    real = multistate_model_from_function(crossing(0.02, 0.5, 0.003, 1.0), [axis], masses=MASS)
    reference = exact_multistate_rate(real, BETA)
    angle = 0.4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    unitary = rotation @ np.diag([1.0, np.exp(0.7j)])
    for transformed in (rotated(real, rotation), rotated(real, unitary)):
        result = exact_multistate_rate(transformed, BETA)
        assert result.rate == pytest.approx(reference.rate, rel=1e-9)
        for key, value in reference.channel_rates.items():
            assert result.channel_rates[key] == pytest.approx(value, rel=1e-8)
    # A constant phase on the coupling (e.g. a complex spin-orbit element) is a gauge choice.
    phased = multistate_model_from_function(crossing(0.02, 0.5, 0.003, 1.0, phase=1.1), [axis], masses=MASS)
    assert phased.is_complex
    result = exact_multistate_rate(phased, BETA, channels="diabatic")
    plain = exact_multistate_rate(real, BETA, channels="diabatic")
    assert result.rate == pytest.approx(plain.rate, rel=1e-9)
    for key, value in plain.channel_rates.items():
        assert result.channel_rates[key] == pytest.approx(value, rel=1e-8)


def test_weak_coupling_transfer_follows_golden_rule_scaling():
    # Product on diabat 1 requires an electronic transition; the direct diabat-0
    # passage needs E > 0.08 Eh and is suppressed by exp(-beta D) ~ 1e-11.
    axis = np.linspace(-8.0, 8.0, 161)
    rates = []
    for coupling in (2.5e-4, 5.0e-4):
        model = multistate_model_from_function(crossing(0.08, 0.5, coupling, 1.0), [axis], masses=MASS)
        result = exact_multistate_rate(model, BETA, channels="diabatic")
        assert abs(result.diagnostics["channel_sum_residual"]) < 1e-9
        assert result.channel_rates["state0"] < 1e-3 * result.channel_rates["state1"]
        rates.append(result.channel_rates["state1"])
    assert rates[1] / rates[0] == pytest.approx(4.0, rel=2e-3)


def test_strong_coupling_approaches_adiabatic_surface_with_diagonal_correction():
    axis = np.linspace(-12.0, 12.0, 241)
    deviations = []
    for coupling in (0.002, 0.008):
        model = multistate_model_from_function(crossing(0.04, 2.0, coupling, 4.0), [axis], masses=MASS)
        out = nonadiabatic_subspace_factor(model, BETA)
        values = model.diabatic
        angle = 0.5 * np.unwrap(np.arctan2(2.0 * values[:, 0, 1], values[:, 0, 0] - values[:, 1, 1]))
        derivative = np.gradient(angle, axis)
        corrected = SubspaceModel((axis,), model.adiabatic()[:, 0] + derivative**2 / (2.0 * MASS),
                                  np.array([MASS]), 0.0)
        reference = exact_quantum_rate(corrected, BETA, method="sylvester")
        assert 0.0 < out["factor"] < 1.0
        deviations.append(abs(out["multistate"].rate / reference.rate - 1.0))
    assert deviations[1] < 2e-3
    assert deviations[1] < 0.1 * deviations[0]


def test_two_dimensional_conical_intersection_is_resolved_and_invariant():
    frequency, height, coupling = 0.005, 0.03, 0.01

    def conical(points):
        s, y = points[..., 0], points[..., 1]
        rise = np.tanh(s / 0.7)
        harmonic = 0.5 * MASS * frequency**2 * y**2
        out = np.zeros(s.shape + (2, 2))
        out[..., 0, 0] = 0.5 * height * (1.0 + rise) + harmonic
        out[..., 1, 1] = 0.5 * height * (1.0 - rise) + harmonic
        out[..., 0, 1] = out[..., 1, 0] = coupling * y * np.exp(-((s / 1.5) ** 2))
        return out

    axes = [np.linspace(-6.0, 6.0, 33), np.linspace(-1.5, 1.5, 11)]
    model = multistate_model_from_function(conical, axes, masses=MASS)
    gaps = model.electronic_diagnostics()["adiabatic_gaps"][0]
    assert gaps["minimum_gap"] < 1e-12
    assert np.allclose(gaps["minimum_gap_point"], [0.0, 0.0])
    result = exact_multistate_rate(model, BETA)
    assert result.rate > 0
    assert abs(result.diagnostics["channel_sum_residual"]) < 1e-8
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    invariant = exact_multistate_rate(rotated(model, rotation), BETA, channels=None)
    assert invariant.rate == pytest.approx(result.rate, rel=1e-9)
    json.dumps(result.to_dict(), allow_nan=False)


def scrambled_adiabatic_data(model, seed=7):
    rng = np.random.default_rng(seed)
    energies, vectors = model.adiabatic_states()
    energies, vectors = energies.copy(), vectors.copy()
    for n in range(len(energies)):
        vectors[n] *= rng.choice([-1.0, 1.0], size=2)[None, :]
        if rng.random() < 0.3:
            energies[n] = energies[n, ::-1]
            vectors[n] = vectors[n][:, ::-1]
    overlaps = np.einsum("nia,nib->nab", vectors[:-1], vectors[1:])
    return energies, vectors, overlaps


def test_propagation_diabatization_recovers_rotated_diabatic_model_and_rate():
    axis = np.linspace(-8.0, 8.0, 161)
    model = multistate_model_from_function(crossing(0.02, 0.5, 0.003, 1.0), [axis], masses=MASS)
    energies, vectors, overlaps = scrambled_adiabatic_data(model)
    result = propagation_diabatization(energies, overlaps)
    frame = vectors[0]
    expected = np.einsum("ia,nij,jb->nab", frame, model.diabatic, frame)
    assert np.max(np.abs(result.diabatic - expected)) < 1e-12
    assert result.diagnostics["minimum_singular_value"] == pytest.approx(1.0, abs=1e-12)
    grid_only = MultistateSubspaceModel((axis,), model.diabatic, MASS, 0.0)
    rebuilt = MultistateSubspaceModel((axis,), result.diabatic, MASS, 0.0)
    assert exact_multistate_rate(rebuilt, BETA).rate == pytest.approx(exact_multistate_rate(grid_only, BETA).rate,
                                                                      rel=1e-9)
    anchored = propagation_diabatization(energies, overlaps, anchor=len(axis) - 1)
    last = anchored.diabatic[-1]
    assert np.allclose(np.diag(last), energies[-1], atol=1e-14)
    assert abs(last[0, 1]) < 1e-14
    json.dumps(result.to_dict(), allow_nan=False)


def test_diabatization_rejects_leaking_overlap():
    energies = np.zeros((3, 2))
    overlaps = np.stack([np.eye(2), np.diag([1.0, 0.5])])
    with pytest.raises(ValueError, match="singular value"):
        propagation_diabatization(energies, overlaps)


def test_invalid_models_and_budgets_are_refused():
    axis = np.linspace(-4.0, 4.0, 41)
    values = np.zeros((41, 2, 2))
    values[:, 0, 1] = 1e-3
    with pytest.raises(ValueError, match="Hermitian"):
        MultistateSubspaceModel((axis,), values, MASS, 0.0)
    with pytest.raises(ValueError, match="shape"):
        MultistateSubspaceModel((axis,), np.zeros((40, 2, 2)), MASS, 0.0)
    model = MultistateSubspaceModel((axis,), np.zeros((41, 2, 2)), MASS, 0.0)
    with pytest.raises(MultistateRouteUnavailable) as refused:
        exact_multistate_rate(model, BETA, dense_limit=50)
    assert refused.value.plan["basis_size"] == 82
    with pytest.raises(ValueError, match="channels"):
        exact_multistate_rate(model, BETA, channels="spin")
    with pytest.raises(TypeError, match="unknown options"):
        nonadiabatic_subspace_factor(model, BETA, method="flux_eigenstates")
