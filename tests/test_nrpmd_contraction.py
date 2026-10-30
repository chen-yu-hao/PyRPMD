"""Independent multistate force-integral checks for the MInt basis transform."""

import numpy as np
import pytest
from scipy.linalg import expm

from pyrpmd.nonadiabatic import DiabaticResult, NRPMDState, _mint


def problem(nstates, spectrum):
    rng = np.random.default_rng(431 + nstates)
    beads, atoms = 3, 2
    matrices = []
    for bead in range(beads):
        basis, _ = np.linalg.qr(rng.normal(size=(nstates, nstates)))
        energies = np.linspace(-0.6, 0.9, nstates)
        if spectrum == "degenerate":
            energies[:min(3, nstates)] = 0.2
        elif spectrum == "near_degenerate":
            energies[:min(3, nstates)] = 0.2 + np.arange(min(3, nstates)) * 1e-14
        matrices.append((basis * energies) @ basis.T + bead * 0.13 * np.eye(nstates))
    gradient = rng.normal(0, 0.2, (beads, nstates, nstates, atoms, 3))
    gradient = 0.5 * (gradient + gradient.swapaxes(1, 2))
    state = NRPMDState(
        rng.normal(size=(beads, atoms, 3)), rng.normal(size=(beads, atoms, 3)),
        step=7, time=0.21,
        mapping_q=rng.normal(size=(beads, nstates)),
        mapping_p=rng.normal(size=(beads, nstates)),
    )
    return state, DiabaticResult(np.asarray(matrices), gradient)


def independent_flow(state, result, duration):
    """Integrate forces from matrix exponentials, without eigenvectors/gaps."""
    z0 = state.mapping_q + 1j * state.mapping_p
    z_final = np.empty_like(z0)
    momentum = state.p.copy()
    nodes, weights = np.polynomial.legendre.leggauss(24)
    times = 0.5 * duration * (nodes + 1)
    weights = 0.5 * duration * weights
    for bead, matrix in enumerate(result.matrix):
        z_final[bead] = expm(-1j * duration * matrix) @ z0[bead]
        for time, weight in zip(times, weights):
            z = expm(-1j * time * matrix) @ z0[bead]
            density = np.outer(z.real, z.real) + np.outer(z.imag, z.imag) - np.eye(len(z))
            momentum[bead] -= 0.5 * weight * np.tensordot(
                density, result.gradients[bead], axes=([0, 1], [0, 1]),
            )
    return momentum, z_final


@pytest.mark.parametrize("nstates", [1, 2, 8, 16])
@pytest.mark.parametrize("spectrum", ["separated", "degenerate", "near_degenerate"])
@pytest.mark.parametrize("duration", [0.0, 0.41, -0.41])
def test_multistate_mint_matches_independent_exponential_force_quadrature(nstates, spectrum, duration):
    state, result = problem(nstates, spectrum)
    original = state.copy()
    momentum, mapping = independent_flow(original, result, duration)
    _mint(state, result, duration)
    np.testing.assert_allclose(state.p, momentum, rtol=2e-12, atol=3e-13)
    np.testing.assert_allclose(state.mapping_q + 1j * state.mapping_p, mapping, rtol=2e-12, atol=3e-13)
    np.testing.assert_array_equal(state.q, original.q)
    assert (state.step, state.time) == (original.step, original.time)
    np.testing.assert_allclose(
        np.sum(state.mapping_q**2 + state.mapping_p**2, axis=1),
        np.sum(original.mapping_q**2 + original.mapping_p**2, axis=1),
        rtol=3e-14, atol=3e-13,
    )


@pytest.mark.parametrize("spectrum", ["separated", "degenerate", "near_degenerate"])
def test_multistate_mint_reverse_flow_restores_nuclear_and_mapping_state(spectrum):
    state, result = problem(16, spectrum)
    original = state.copy()
    _mint(state, result, 0.73)
    _mint(state, result, -0.73)
    for field in ("p", "mapping_q", "mapping_p"):
        np.testing.assert_allclose(getattr(state, field), getattr(original, field), rtol=2e-12, atol=3e-13)
    np.testing.assert_array_equal(state.q, original.q)


def test_zero_mapping_vectors_keep_negative_mmst_population_and_trace_impulse():
    state, result = problem(8, "near_degenerate")
    state.mapping_q[:] = state.mapping_p[:] = 0
    before = state.p.copy()
    duration = 0.41
    _mint(state, result, duration)
    trace_force = 0.5 * np.trace(result.gradients, axis1=1, axis2=2)
    np.testing.assert_allclose(state.p, before + duration * trace_force, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(state.populations(), np.full(8, -0.5))
