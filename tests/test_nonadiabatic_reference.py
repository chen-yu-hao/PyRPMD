"""Quantum reference checks independent of trajectory dynamics."""

import numpy as np
import pytest
from scipy.linalg import block_diag, expm

from pyrpmd.analysis.quantum_reference import SplitOperator1D, harmonic_canonical_ensemble


def normalized(values, dx):
    return values/np.sqrt(dx*np.sum(np.abs(values)**2))


def dense_kinetic(grid, mass):
    """Explicit DFT matrix, independent of the production FFT action."""
    n, dx = len(grid), grid[1]-grid[0]
    indices = np.arange(n)
    dft = np.exp(-2j*np.pi*np.outer(indices, indices)/n)/np.sqrt(n)
    modes = np.where(indices <= (n-1)//2, indices, indices-n)
    energies = (2*np.pi*modes/(n*dx))**2/(2*mass)
    return dft.conj().T@np.diag(energies)@dft


def test_constant_complex_matrix_matches_analytic_rabi_populations():
    grid = np.linspace(-8., 8., 64, endpoint=False)
    coupling, bias = 0.4+0.1j, 0.3
    matrix = np.array([[bias, coupling], [coupling.conjugate(), -bias]])
    potential = np.broadcast_to(matrix, (len(grid), 2, 2))
    psi = np.zeros((len(grid), 2), complex)
    psi[:, 0] = normalized(np.exp(-grid**2/4), grid[1]-grid[0])
    result = SplitOperator1D(grid, potential, 2., 0.07).propagate(psi, 30, sample_stride=3)
    frequency = np.sqrt(bias**2+abs(coupling)**2)
    expected = abs(coupling)**2/frequency**2*np.sin(frequency*result.time)**2
    np.testing.assert_allclose(result.populations[:, 1], expected, rtol=0., atol=5e-14)
    np.testing.assert_allclose(result.populations[:, 0], 1.-expected, rtol=0., atol=5e-14)
    np.testing.assert_allclose(result.norm, 1., rtol=0., atol=7e-14)


def test_free_gaussian_has_analytic_translation_spreading_and_unit_norm():
    grid = np.linspace(-20., 20., 128, endpoint=False)
    mass, sigma, momentum = 2., 0.9, 0.7
    psi = normalized(np.exp(-grid**2/(4*sigma**2)+1j*momentum*grid), grid[1]-grid[0])[:, None]
    before = psi.copy()
    solver = SplitOperator1D(grid, np.zeros((len(grid), 1, 1)), mass, 0.04)
    result = solver.propagate(psi, 30, sample_stride=5)
    probability = np.abs(result.final_wavefunctions[0, :, 0])**2
    mean = solver.dx*np.dot(grid, probability)
    variance = solver.dx*np.dot((grid-mean)**2, probability)
    duration = result.time[-1]
    assert mean == pytest.approx(momentum*duration/mass, abs=2e-12)
    assert variance == pytest.approx(sigma**2+(duration/(2*mass*sigma))**2, abs=3e-12)
    np.testing.assert_allclose(result.component_norms, 1., rtol=0., atol=5e-14)
    np.testing.assert_array_equal(psi, before)
    assert result.edge_population.max() < 1e-20


def test_split_operator_converges_quadratically_to_independent_dense_expm():
    grid, mass, duration = np.linspace(-np.pi, np.pi, 16, endpoint=False), 1.3, 0.4
    potential = np.zeros((len(grid), 2, 2))
    potential[:, 0, 0], potential[:, 1, 1] = 0.3*np.cos(grid), -0.2*np.cos(grid)
    potential[:, 0, 1] = potential[:, 1, 0] = 0.2+0.1*np.sin(grid)
    nuclear = normalized(np.exp(0.4*np.cos(grid)+2j*grid), grid[1]-grid[0])
    psi = nuclear[:, None]*np.array([np.sqrt(0.7), 1j*np.sqrt(0.3)])[None, :]
    full = np.kron(dense_kinetic(grid, mass), np.eye(2))+block_diag(*potential)
    exact = (expm(-1j*duration*full)@psi.reshape(-1)).reshape(psi.shape)
    errors = []
    for dt in (0.1, 0.05, 0.025):
        solver = SplitOperator1D(grid, potential, mass, dt)
        output = solver.propagate(psi, round(duration/dt))
        errors.append(np.sqrt(solver.dx)*np.linalg.norm(output.final_wavefunctions[0]-exact))
        assert abs(output.norm[-1]-1.) < 5e-14
    assert 3.8 < errors[0]/errors[1] < 4.2
    assert 3.8 < errors[1]/errors[2] < 4.2
    assert errors[-1] < 1e-4


def test_mixture_density_is_weighted_sum_of_pure_results():
    grid = np.linspace(-5., 5., 32, endpoint=False)
    potential = np.broadcast_to(np.array([[0.1, 0.2], [0.2, -0.1]]), (len(grid), 2, 2))
    nuclear = normalized(np.exp(-grid**2), grid[1]-grid[0])
    psi = np.zeros((2, len(grid), 2), complex)
    psi[0, :, 0], psi[1, :, 1] = nuclear, nuclear
    solver = SplitOperator1D(grid, potential, 1., 0.05)
    mixture = solver.propagate(psi, 7, weights=[0.25, 0.75], sample_stride=3)
    pure = [solver.propagate(component, 7, sample_stride=3) for component in psi]
    np.testing.assert_allclose(mixture.density, 0.25*pure[0].density+0.75*pure[1].density, atol=1e-14)
    np.testing.assert_array_equal(mixture.time, np.array([0, 3, 6, 7])*0.05)
    assert solver.step(psi[0]).shape == psi[0].shape
    assert solver.step(psi).shape == psi.shape


def test_finite_grid_harmonic_thermal_density_matches_dense_matrix_exponential():
    grid = np.linspace(-4., 4., 16, endpoint=False)
    dx, mass, omega, beta, center = grid[1]-grid[0], 1.2, 0.8, 0.7, 0.3
    thermal = harmonic_canonical_ensemble(grid, mass, omega, beta, center=center, tail_tolerance=0.)
    full = dense_kinetic(grid, mass)+np.diag(0.5*mass*omega**2*(grid-center)**2)
    exact = expm(-beta*full)
    exact /= np.trace(exact)
    actual = dx*np.einsum("m,mx,my->xy", thermal.weights, thermal.wavefunctions, thermal.wavefunctions.conj())
    np.testing.assert_allclose(actual, exact, rtol=0., atol=3e-14)
    np.testing.assert_allclose(dx*np.sum(abs(thermal.wavefunctions)**2, axis=1), 1., atol=2e-14)
    assert thermal.omitted_weight == 0.


def test_harmonic_reference_recovers_continuum_variance_and_controls_thermal_tail():
    grid = np.linspace(-8., 8., 64, endpoint=False)
    thermal = harmonic_canonical_ensemble(grid, 1., 1., 2., tail_tolerance=1e-10)
    dx = grid[1]-grid[0]
    variance = dx*np.einsum("m,mx,x->", thermal.weights, abs(thermal.wavefunctions)**2, grid**2)
    np.testing.assert_allclose(thermal.energies[:5], np.arange(5)+0.5, atol=2e-12)
    assert variance == pytest.approx(0.5/np.tanh(1.), abs=2e-9)
    assert 0. <= thermal.omitted_weight <= 1e-10
    assert thermal.weights.sum() == pytest.approx(1., abs=2e-15)


def setup_small():
    grid = np.arange(8.)/2
    potential = np.zeros((8, 1, 1))
    psi = np.ones((8, 1))/2
    return grid, potential, psi


@pytest.mark.parametrize("field,value", [("mass", 0.), ("mass", True), ("mass", np.inf),
                                          ("dt", 0.), ("dt", -0.1), ("dt", np.nan)])
def test_invalid_scalar_parameters_are_rejected(field, value):
    grid, potential, _ = setup_small()
    kwargs = dict(mass=1., dt=0.1)
    kwargs[field] = value
    with pytest.raises(ValueError):
        SplitOperator1D(grid, potential, **kwargs)


def test_nonuniform_grid_and_nonhermitian_absorber_are_rejected():
    grid, potential, _ = setup_small()
    bad = grid.copy(); bad[3] += 0.01
    with pytest.raises(ValueError, match="uniformly"):
        SplitOperator1D(bad, potential, 1., 0.1)
    with pytest.raises(ValueError, match="Hermitian"):
        SplitOperator1D(grid, potential.astype(complex)-0.01j, 1., 0.1)


@pytest.mark.parametrize("weights", [[-0.1, 1.1], [0.2, 0.2], [np.nan, 0.5], [1.], None, [True, False]])
def test_invalid_or_missing_mixture_weights_are_rejected(weights):
    grid, potential, psi = setup_small()
    solver = SplitOperator1D(grid, potential, 1., 0.1)
    with pytest.raises(ValueError, match="weights"):
        solver.propagate(np.stack((psi, psi)), 1, weights=weights)


def test_input_normalization_parameters_and_storage_are_strict():
    grid, potential, psi = setup_small()
    solver = SplitOperator1D(grid, potential, 1., 0.1)
    potential[:] = 100.
    assert np.all(solver.potential == 0)
    with pytest.raises(AttributeError, match="immutable"):
        solver.dt = 0.2
    with pytest.raises(ValueError):
        solver.potential.flags.writeable = True
    with pytest.raises(ValueError, match="unit dx-weighted norm"):
        solver.step(2*psi)
    with pytest.raises(ValueError, match="steps"):
        solver.propagate(psi, True)
    with pytest.raises(ValueError, match="sample_stride"):
        solver.propagate(psi, 1, sample_stride=0)
    result = solver.propagate(psi, 0)
    np.testing.assert_array_equal(result.time, [0.])
    np.testing.assert_allclose(result.populations, [[1.]])
