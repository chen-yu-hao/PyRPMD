"""Out-of-sample checks for the nonlinear/joint-coordinate selector.

These tests do not treat the selector score as an error bound.  They only make
sure that the feature rule is stable on independent samples and that the
selected joint coordinates explain a held-out nonlinear force better than a
deliberately incomplete candidate on a controlled model.
"""

import numpy as np

from pyrpmd.analysis.quantum_subspace import select_quantum_subspace


def _samples(seed, count=1200):
    rng = np.random.default_rng(seed)
    positions = rng.normal(size=(count, 3))
    forces = np.column_stack(
        [
            -(positions[:, 0] + 0.8 * positions[:, 1] ** 2),
            -positions[:, 1],
            -positions[:, 2],
        ]
    )
    return positions, forces


def _quadratic_design(values):
    values = np.asarray(values, dtype=float)
    columns = [np.ones(len(values))]
    columns.extend(values[:, i] for i in range(values.shape[1]))
    columns.extend(values[:, i] ** 2 for i in range(values.shape[1]))
    columns.extend(
        values[:, i] * values[:, j]
        for i in range(values.shape[1])
        for j in range(i + 1, values.shape[1])
    )
    return np.column_stack(columns)


def _held_out_force_error(train_positions, train_forces, test_positions, test_forces, indices):
    train = train_positions[:, indices]
    test = test_positions[:, indices]
    design_train = _quadratic_design(train)
    design_test = _quadratic_design(test)
    coefficients = np.linalg.lstsq(design_train, train_forces, rcond=None)[0]
    prediction = design_test @ coefficients
    return float(np.linalg.norm(prediction - test_forces) / np.linalg.norm(test_forces))


def test_selector_is_stable_on_independent_sampling_blocks():
    selections = []
    for seed in (1, 2, 3, 4):
        positions, forces = _samples(seed)
        result = select_quantum_subspace(
            positions,
            forces,
            max_dimension=2,
            joint_coordinates=[(0, 1)],
            target_observables=positions[:, 0] ** 2,
        )
        selections.append(result.selected_indices)
        assert result.diagnostics["physical_converged"] is False
        assert np.isfinite(result.scores).all()
    assert selections == [(0, 1)] * len(selections)


def test_joint_selection_improves_held_out_nonlinear_force_prediction():
    train_positions, train_forces = _samples(11)
    test_positions, test_forces = _samples(211)
    selected = select_quantum_subspace(
        train_positions,
        train_forces,
        max_dimension=2,
        joint_coordinates=[(0, 1)],
        target_observables=train_positions[:, 0] ** 2,
    )
    selected_error = _held_out_force_error(
        train_positions,
        train_forces,
        test_positions,
        test_forces,
        selected.selected_indices,
    )
    incomplete_error = _held_out_force_error(
        train_positions,
        train_forces,
        test_positions,
        test_forces,
        (0, 2),
    )
    assert selected.selected_indices == (0, 1)
    assert selected_error < 0.8 * incomplete_error
