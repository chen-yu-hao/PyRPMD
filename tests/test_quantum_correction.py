"""Algebraic and numerical checks for the response-level correction."""

import numpy as np
import pytest

from pyrpmd.analysis.quantum_correction import (
    causal_susceptibility,
    frequency_response,
    hybrid_quantum_correction,
    inverse_response_defect,
    regularized_inverse,
    run_quantum_correction,
)


def response_series(values, *, size=2):
    values = np.asarray(values, dtype=complex)
    return np.array([np.eye(size, dtype=complex) * value for value in values])


def test_causal_susceptibility_and_transform_match_exponential_reference():
    gamma, beta = 0.7, 2.0
    time = np.linspace(0.0, 40.0, 4001)
    correlation = np.exp(-gamma * time)
    expected_time = beta * gamma * correlation
    actual_time = causal_susceptibility(time, correlation, beta)
    np.testing.assert_allclose(actual_time[2:-2], expected_time[2:-2], rtol=0, atol=2e-5)
    frequencies = np.linspace(0.0, 2.0, 11)
    actual = frequency_response(time, actual_time, frequencies, damping=0.05)
    expected = beta * gamma / (gamma + 0.05 - 1j * frequencies)
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_zero_defect_recovers_full_response_exactly():
    frequencies = np.linspace(0.1, 4.0, 20)
    full = 1.2 + 0.1j * frequencies
    result = hybrid_quantum_correction(frequencies, full, full, full)
    np.testing.assert_allclose(result.inverse_defect, 0.0, atol=0.0)
    np.testing.assert_allclose(result.corrected_response, full, rtol=0, atol=1e-14)
    assert result.physical_converged is False
    assert result.diagnostics["physical_converged"] is False


def test_full_subspace_limit_recovers_quantum_response():
    frequencies = np.linspace(0.1, 3.0, 10)
    full = 1.3 + 0.2j * frequencies
    quantum = 0.8 + 0.05j * frequencies
    result = hybrid_quantum_correction(frequencies, full, full, quantum)
    np.testing.assert_allclose(result.corrected_response, quantum, rtol=0, atol=2e-14)


def test_inverse_defect_has_the_stated_sign_and_matrix_support():
    frequencies = np.array([0.3, 0.8])
    rpmd = response_series([1.0, 1.2])
    quantum = response_series([2.0, 2.4])
    defect = inverse_response_defect(rpmd, quantum)
    expected = response_series([-0.5, -5.0/12.0])
    np.testing.assert_allclose(defect, expected, atol=1e-14)


def test_matrix_correction_preserves_observable_basis_and_reports_rank():
    frequencies = np.array([0.4, 1.0, 2.0])
    full = np.array([[[2.0 + w, 0.2], [0.2, 1.0 + 0.5 * w]] for w in frequencies], dtype=complex)
    sub_rp = np.array([[[1.5 + w, 0.1], [0.1, 0.8 + 0.2 * w]] for w in frequencies], dtype=complex)
    sub_q = np.array([[[1.2 + w, 0.05], [0.05, 0.9 + 0.3 * w]] for w in frequencies], dtype=complex)
    result = hybrid_quantum_correction(frequencies, full, sub_rp, sub_q)
    assert result.corrected_response.shape == (3, 2, 2)
    assert result.diagnostics["corrected_effective_rank_min"] == 2
    assert result.diagnostics["full_condition_number_max"] > 1


def test_ridge_stabilizes_singular_response_and_is_recorded():
    frequencies = np.array([0.5, 1.0])
    singular = np.array([1.0, 0.0], dtype=complex)
    inverse = regularized_inverse(singular, ridge=0.1)
    assert np.all(np.isfinite(inverse.values))
    assert inverse.ridge == pytest.approx(0.1)
    result = hybrid_quantum_correction(frequencies, singular, singular, singular + 0.2,
                                       inverse_ridge=0.1, output_ridge=0.1)
    assert result.diagnostics["inverse_ridge"] == pytest.approx(0.1)
    assert np.all(np.isfinite(result.response))


def test_time_domain_workflow_records_grid_and_convention():
    time = np.linspace(0.0, 20.0, 2001)
    frequencies = np.linspace(0.1, 2.0, 16)
    base = np.exp(-0.8 * time)
    result = run_quantum_correction(
        time, frequencies, base, base, 1.05 * base + 0.02 * np.exp(-0.2 * time),
        beta=1.7, damping=0.03,
        provenance={"subspace": ["x"], "reference_kind": "exact-test"},
    )
    assert result.provenance["subspace"] == ["x"]
    assert result.provenance["correlation_convention"].startswith("Kubo")
    assert result.provenance["time_points"] == len(time)
    assert np.all(np.isfinite(result.corrected_response))


@pytest.mark.parametrize("bad", [np.ones(4), np.ones((4, 2, 3))])
def test_mismatched_response_axes_are_rejected(bad):
    frequencies = np.linspace(0.1, 1.0, 3)
    with pytest.raises(ValueError):
        hybrid_quantum_correction(frequencies, np.ones(3), bad, np.ones(3))
