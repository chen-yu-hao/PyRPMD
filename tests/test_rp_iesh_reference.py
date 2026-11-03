"""Independent finite-Fock/Fourier references for centroid RP-IESH.

The exact decoupled limit exercises a noncollapsed nuclear ring and nontrivial
two-electron evolution. It does not certify the approximation for a coupled
metallic bath; that finite-nucleus comparison is reported by the example.
"""
from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy.linalg import expm

from pyrpmd.iesh import NewnsAndersonModel
from pyrpmd.nonadiabatic_sampling import sample_harmonic_ring


_reference = runpy.run_path(str(Path(__file__).resolve().parents[1]
                               / "examples/nonadiabatic/rp_iesh_reference.py"))
FockSpace = _reference["FockSpace"]
quantum_reference = _reference["quantum_reference"]
initialize_diabatic_determinant = _reference["initialize_diabatic_determinant"]


def test_three_orbital_two_electron_lift_has_independent_fermion_signs():
    space = FockSpace(3, 2)
    matrix = np.array([[.3, .07, -.09], [.07, -.2, .04], [-.09, .04, .6]])
    # Basis |01>, |02>, |12>: the 0<->2 transfer passes occupied orbital 1.
    expected = np.array([[.1, .04, .09], [.04, .9, .07], [.09, .07, .4]])
    np.testing.assert_allclose(space.lift(matrix), expected, rtol=0., atol=3e-16)
    energies = np.linalg.eigvalsh(matrix)
    sums = sorted(energies[i]+energies[j] for i, j in space.basis)
    np.testing.assert_allclose(np.linalg.eigvalsh(space.lift(matrix)), sums, atol=2e-15, rtol=0.)
    np.testing.assert_array_equal(space.operators.diagonal(axis1=0, axis2=1).sum(axis=-1),
                                  2*np.eye(3))


@pytest.mark.parametrize("norbitals,nelectrons", [(3, 2), (4, 2), (4, 3)])
def test_fock_exponential_and_one_particle_density_match_slater_exactly(norbitals, nelectrons):
    space = FockSpace(norbitals, nelectrons)
    rng = np.random.default_rng(47+norbitals+nelectrons)
    raw = rng.normal(size=(norbitals, norbitals))+1j*rng.normal(size=(norbitals, norbitals))
    matrix = .2*(raw+raw.conj().T)
    orbitals, _ = np.linalg.qr(rng.normal(size=(norbitals, nelectrons))
                               +1j*rng.normal(size=(norbitals, nelectrons)))
    initial = space.slater_coefficients(orbitals)
    np.testing.assert_allclose(np.vdot(initial, initial), 1., rtol=0., atol=2e-15)
    duration = 1.7
    exact = expm(-1j*space.lift(matrix)*duration) @ initial
    propagated = expm(-1j*matrix*duration) @ orbitals
    np.testing.assert_allclose(exact, space.slater_coefficients(propagated), atol=2e-15, rtol=0.)
    density = space.one_particle_density(exact[:, None]*exact.conj()[None, :])
    np.testing.assert_allclose(density, propagated@propagated.conj().T, atol=2e-15, rtol=0.)
    assert np.max(abs(density.imag)) > .01  # Detect transposed-density conventions.
    np.testing.assert_allclose(np.trace(density), nelectrons, atol=2e-15, rtol=0.)


@pytest.mark.parametrize("beads", [1, 8])
def test_noncollapsed_ring_electronic_density_matches_two_electron_fourier(beads):
    from pyrpmd.rp_iesh import RPIESHDynamics

    mass, beta, omega, dt, steps = 1000., 80., .02, .2, 30
    model = NewnsAndersonModel(nband=2, band_min=-.1, band_max=.1,
                               level_slope=0., hybridization=.04,
                               bare_force_constant=mass*omega**2)
    space = FockSpace(3, 2)
    exact, reference_density, omitted = quantum_reference(
        model, space, points=96, dt=dt, duration=steps*dt,
        mass=mass, beta=beta, omega=omega, sample_stride=1)
    driver = RPIESHDynamics(model, [mass], beta, beads, dt, nelectrons=2,
                            native=False, seed=19, electronic_substeps=2)
    nuclear = sample_harmonic_ring([mass], beta, beads, [omega], seed=81)[0]
    state = initialize_diabatic_determinant(driver, nuclear, space, np.random.default_rng(43))
    if beads > 1:
        assert np.ptp(state.q[:, 0, 0]) > .05
    initial_q = state.q.copy()
    estimates = [driver.density_matrix(state)]
    for _ in range(steps):
        driver.step(state)
        estimates.append(driver.density_matrix(state))
        assert driver.last_hop is None
    assert np.max(abs(state.q-initial_q)) > .001
    np.testing.assert_allclose(estimates, reference_density, rtol=0., atol=8e-13)
    assert reference_density[-1, 2, 2].real > .01  # Two-electron transfer is nontrivial.
    np.testing.assert_allclose(np.trace(reference_density, axis1=-2, axis2=-1), 2., atol=8e-13, rtol=0.)
    assert np.max(exact.edge_population) < 1e-12
    np.testing.assert_allclose(exact.norm, 1., atol=5e-13, rtol=0.)
    assert omitted < 1e-12
    # With constant h the bare oscillator remains canonical. This independent
    # nuclear moment also detects accidentally adding U0 once per electron.
    grid = np.linspace(-2., 2., 96, endpoint=False)
    nuclear_probability = np.sum(abs(exact.final_wavefunctions)**2, axis=-1)
    final_q2 = (grid[1]-grid[0])*np.dot(exact.weights, nuclear_probability @ grid**2)
    harmonic_q2 = 1/(2*mass*omega*np.tanh(beta*omega/2))
    assert final_q2 == pytest.approx(harmonic_q2, abs=1e-8)


def test_coupled_reference_reports_distinct_errors_without_claiming_convergence():
    result = _reference["compare"](trajectories=3, beads=4, duration=.4, dt=.2,
                                    points=48, reference_dt=.1, seed=7)
    assert result["physical_converged"] is False
    assert result["nelectrons"] == 2 and result["population_normalization"] == 2
    np.testing.assert_allclose(result["rp_iesh_initial_mean"], [1., 1., 0.], rtol=0., atol=2e-15)
    np.testing.assert_allclose(sum(result["rp_iesh_final_mean"]), 2., rtol=0., atol=8e-13)
    np.testing.assert_allclose(sum(result["fourier_final_populations"]), 2., rtol=0., atol=8e-13)
    assert len(result["rp_iesh_final_sem"]) == 3
    assert all(value >= 0 for value in result["rp_iesh_final_sem"])
    for kind in ("time", "space", "box"):
        assert 0 <= result[f"fourier_{kind}_refinement_max_density_diff"] < 1e-5


@pytest.mark.parametrize("norbitals,nelectrons", [(True, 1), (3, True), (3, 4), (17, 1), (16, 8)])
def test_reference_dimension_and_integer_guards(norbitals, nelectrons):
    with pytest.raises(ValueError):
        FockSpace(norbitals, nelectrons)
