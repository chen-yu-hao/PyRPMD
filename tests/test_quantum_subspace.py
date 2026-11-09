"""Small-system regression tests; no exact quantum answers enter selection."""
import numpy as np
import pytest
from pyrpmd.analysis.quantum_subspace import select_subspace, memory_embed


def force_samples(couplings, seed=3):
    z = np.random.default_rng(seed).normal(size=(1000, 3))
    K = np.diag([-3.136, 2.25, 5.29])
    for j, (w2, c) in enumerate(zip([2.25, 5.29], couplings), 1):
        K[0, 0] += w2 * c * c
        K[j, 0] = K[0, j] = -w2 * c
    f = -z @ K.T
    f[:, 0] -= 1.6 * z[:, 0] ** 3
    return z, f


@pytest.mark.parametrize('couplings,dimension', [((.03, .02), 1), ((.8, .02), 2), ((.8, .5), 3)])
def test_selection_without_quantum_oracle(couplings, dimension):
    z, f = force_samples(couplings)
    result = select_subspace(z, f, target=np.eye(3)[0])
    assert result['selected_dimension'] == dimension
    assert result['force_fit_relative_residual'] < 1e-10
    assert result['residual_bath_nonlinearity_fraction'] < 1e-10


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_rotational_covariance(seed):
    z, f = force_samples((.8, .02), seed)
    rotation, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    U = select_subspace(z, f, target=np.eye(3)[0])['basis']
    V = rotation.T @ select_subspace(z @ rotation.T, f @ rotation.T, target=rotation[:, 0])['basis']
    np.testing.assert_allclose(U @ U.T, V @ V.T, atol=1e-11)


def test_dimension_budget_is_not_silently_exceeded():
    z, f = force_samples((.8, .5))
    result = select_subspace(z, f, max_dimension=2, target=np.eye(3)[0])
    assert result['selected_dimension'] == 2
    assert result['budget_exhausted']


def test_even_nonlinear_partner_is_detected():
    z = np.random.default_rng(6).normal(size=(3000, 3))
    x, y, b = z.T
    f = np.column_stack([-4*x-.35*y*y, -y-.7*x*y-.08*y**3, -2*b])
    result = select_subspace(z, f, max_dimension=2, target=np.eye(3)[0])
    assert result['nonlinear_rank'] == 2
    assert result['selected_dimension'] == 2
    limited = select_subspace(z, f, max_dimension=1, target=np.eye(3)[0])
    assert limited['selected_dimension'] == 1 and limited['budget_exhausted']


def test_harmonic_force_is_not_a_nonlinear_defect():
    z = np.random.default_rng(2).normal(size=(1000, 3))
    K = np.array([[2., .4, 0.], [.4, 3., .2], [0., .2, 4.]])
    target = np.array([2., 0., 0.])
    result = select_subspace(z, -z @ K, target=target)
    assert not result['quantum_correction_candidate']
    np.testing.assert_array_equal(target, [2., 0., 0.])


def test_harmonic_memory_matches_exact_schur_complement():
    K = np.array([[2., -.6], [-.6, 1.5]])
    gap = np.sqrt(2.-.36/1.5)
    spec = {'gap': np.array([gap]), 'matrix_coefficients': np.array([[[1/gap]]])}
    w = np.linspace(.02, 4., 400)
    result, diagnostics = memory_embed(spec, w, .04, K[:1, 1:], K[1:, 1:])
    exact = np.array([np.linalg.inv(K + (.04-1j*v)**2*np.eye(2))[0, 0] for v in w])
    np.testing.assert_allclose(result[:, 0, 0], exact, atol=1e-12)
    assert diagnostics['passivity_grid_passed']
    assert not diagnostics['physical_converged']


def test_optional_static_covariance_matches_zero_frequency():
    spec = {'gap': np.array([1.5]), 'matrix_coefficients': np.array([[[1/1.5]]])}
    result, _ = memory_embed(spec, np.array([0., .01]), 1e-6,
                             np.array([[.1]]), np.array([[2.]]), np.array([[.4]]), 2.)
    assert result[0, 0, 0].real == pytest.approx(.8, abs=1e-10)


def test_no_bath_preserves_quantum_response():
    gap = 1.2
    spec = {'gap': np.array([gap]), 'matrix_coefficients': np.array([[[1/gap]]])}
    w = np.linspace(.03, 3., 100)
    result, _ = memory_embed(spec, w, .04, np.empty((1, 0)), np.empty((0, 0)))
    np.testing.assert_allclose(result[:, 0, 0], 1/(gap*gap+(.04-1j*w)**2), atol=1e-12)


@pytest.mark.parametrize('w,damping,bath', [
    (np.array([1., 0.]), .04, np.array([[2.]])),
    (np.array([1., 2.]), 0., np.array([[2.]])),
    (np.array([1., 2.]), .04, np.array([[-1.]])),
])
def test_invalid_embedding_inputs(w, damping, bath):
    spec = {'gap': np.array([1.]), 'matrix_coefficients': np.array([[[1.]]])}
    with pytest.raises(ValueError):
        memory_embed(spec, w, damping, np.array([[.1]]), bath)


@pytest.mark.parametrize('target', [np.zeros(3), np.ones(2), np.array([np.nan, 0., 0.])])
def test_invalid_observable_direction(target):
    z, f = force_samples((.8, .02))
    with pytest.raises(ValueError):
        select_subspace(z, f, target=target)

from pyrpmd.analysis.quantum_subspace import select_quantum_subspace


def test_selector_captures_even_nonlinear_force_and_joint_coordinate_group():
    rng = np.random.default_rng(12)
    positions = rng.normal(size=(600, 3))
    forces = np.column_stack([
        -(positions[:, 0] + 0.8 * positions[:, 1] ** 2),
        -positions[:, 1],
        -positions[:, 2],
    ])
    selection = select_quantum_subspace(
        positions,
        forces,
        max_dimension=2,
        joint_coordinates=[(0, 1)],
        target_observables=positions[:, 0] ** 2,
    )
    assert selection.selected_indices == (0, 1)
    assert selection.basis.shape == (3, 2)
    assert selection.diagnostics["selection_is_error_bound"] is False
    assert selection.diagnostics["physical_converged"] is False


def test_selector_reports_nested_ranking_without_forcing_full_space():
    rng = np.random.default_rng(3)
    positions = rng.normal(size=(300, 5))
    forces = -positions.copy()
    forces[:, 0] -= 0.5 * positions[:, 2] ** 2
    selection = select_quantum_subspace(positions, forces, max_dimension=2)
    assert 1 <= len(selection.selected_indices) <= 2
    assert len(selection.diagnostics["ranking"]) == 5
