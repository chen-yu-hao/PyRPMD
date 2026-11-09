import numpy as np
import pytest

from pyrpmd.analysis import (
    flux_side_rate,
    hybrid_flux_side_correction,
    select_quantum_subspace,
)


def test_selector_catches_even_target_and_closes_joint_group():
    rng = np.random.default_rng(14)
    positions = rng.normal(size=(256, 3))
    forces = np.column_stack(
        [
            2.0 * positions[:, 0] + 0.8 * positions[:, 1] ** 2,
            1.5 * positions[:, 1] + 0.4 * positions[:, 0],
            4.0 * positions[:, 2],
        ]
    )
    result = select_quantum_subspace(
        positions,
        forces,
        max_dimension=2,
        joint_coordinates=[(0, 1)],
        target_observables=positions[:, 1] ** 2,
    )
    assert result.selected_indices == (0, 1)
    assert result.basis.shape == (3, 2)
    assert result.diagnostics["selection_is_error_bound"] is False
    assert result.diagnostics["target_feature_model"]
    assert np.all(np.isfinite(result.scores))


def test_selector_requires_budget_for_joint_group():
    positions = np.arange(40.0).reshape(20, 2)
    forces = np.column_stack([positions[:, 0] ** 2, positions[:, 1]])
    with pytest.raises(ValueError, match="joint coordinate group exceeds max_dimension"):
        select_quantum_subspace(positions, forces, max_dimension=1, joint_coordinates=[(0, 1)])


def test_flux_side_normalization_and_bootstrap():
    time = np.linspace(0.0, 1.0, 5)
    flux = np.array([[1.0, 1.0, 2.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0, 4.0]])
    side = np.ones_like(flux)
    result = flux_side_rate(
        time,
        flux,
        side,
        reactant_population=np.array([1.0, 3.0]),
        bootstrap=64,
        rng=np.random.default_rng(4),
        plateau_start=0.5,
        plateau_stop=1.0,
    )
    # Weighted mean flux is 2.75 and weighted mean population is 2.
    assert np.allclose(result.rate, [1.0, 1.0, 1.5, 1.5, 1.5])
    assert result.bootstrap_rate.shape == (64, 5)
    assert result.standard_error is not None
    assert result.diagnostics["plateau_detected"]
    assert result.physical_converged is False


def test_flux_side_defect_zero_and_full_subspace_limits():
    time = np.arange(4.0)
    full = np.array([1.0, 2.0, 3.0, 4.0])
    sub = np.array([0.5, 1.0, 2.0, 2.5])
    quantum = sub.copy()
    result = hybrid_flux_side_correction(time, full, sub, quantum)
    assert np.allclose(result["rate"], full)
    result = hybrid_flux_side_correction(time, full, full, quantum)
    assert np.allclose(result["rate"], quantum)
    assert result["diagnostics"]["exact_general_rate_claim"] is False


def test_matrix_flux_side_defect_transfer():
    time = np.arange(3.0)
    eye = np.eye(2)
    full = np.stack([eye, 2 * eye, 3 * eye])
    sub = np.stack([0.5 * eye, eye, 1.5 * eye])
    quantum = np.stack([0.75 * eye, 1.25 * eye, 2.25 * eye])
    result = hybrid_flux_side_correction(time, full, sub, quantum)
    assert result["rate"].shape == (3, 2, 2)
    assert np.allclose(result["rate"], full + quantum - sub)
    inverse = hybrid_flux_side_correction(
        time, full, sub, quantum, mode="inverse_defect", ridge=1e-3,
    )
    assert inverse["rate"].shape == (3, 2, 2)
    assert np.isfinite(inverse["rate"]).all()
