"""Frequency-domain quantum correction workflow for RPMD observables.

The public functions implement the controlled response-level construction

    chi_mix(w)^(-1) = chi_rpmd,full(w)^(-1)
                    + chi_q,sub(w)^(-1) - chi_rpmd,sub(w)^(-1).

The low-dimensional quantum calculation supplies a *defect* relative to a
matched low-dimensional RPMD calculation. It does not replace full-system
bath dynamics and is not an identity for a coupled Hamiltonian. Results carry
``physical_converged=False`` until a caller supplies independent convergence
evidence.

All transforms use atomic units and the convention
``chi(w) = integral_0^infty exp((i*w-eta)*t) chi(t) dt``. For a Kubo
correlation ``C(t)``, the causal response is ``chi(t)=-beta*dC/dt`` for
``t>=0``. The t=0 sample receives the usual half trapezoid weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

_EPS = np.finfo(float).eps
_TRAPEZOID = getattr(np, "trapezoid", np.trapz)


def _readonly(value: Any, *, dtype=None) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    owned = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    owned.flags.writeable = False
    return owned


def _real_scalar(value: Any, name: str, *, nonnegative=False, positive=False) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "fiu" or not np.isfinite(raw):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _validate_time(time: Iterable[float], name="time") -> np.ndarray:
    raw = np.asarray(time)
    if raw.ndim != 1 or raw.size < 3 or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be a finite one-dimensional vector with at least three points")
    result = np.array(raw, dtype=float, copy=True)
    if result[0] < 0 or np.any(np.diff(result) <= 0):
        raise ValueError(f"{name} must be strictly increasing and nonnegative")
    return result


def _validate_frequencies(frequencies: Iterable[float]) -> np.ndarray:
    raw = np.asarray(frequencies)
    if raw.ndim != 1 or raw.size < 1 or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError("frequencies must be a finite one-dimensional vector")
    result = np.array(raw, dtype=float, copy=True)
    if np.any(result < 0) or (result.size > 1 and np.any(np.diff(result) <= 0)):
        raise ValueError("frequencies must be sorted and nonnegative")
    return result


def _validate_series(value: Any, ntime: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim not in (1, 3) or array.shape[0] != ntime:
        raise ValueError(f"{name} must have shape [time] or [time,n,n]")
    if array.ndim == 3 and array.shape[1] != array.shape[2]:
        raise ValueError(f"{name} matrix samples must be square")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return np.array(array, dtype=np.complex128, copy=True)


def _validate_frequency_response(value: Any, nfreq: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim not in (1, 3) or array.shape[0] != nfreq:
        raise ValueError(f"{name} must have shape [frequency] or [frequency,n,n]")
    if array.ndim == 3 and array.shape[1] != array.shape[2]:
        raise ValueError(f"{name} matrix samples must be square")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return np.array(array, dtype=np.complex128, copy=True)


def _same_shape(*arrays: np.ndarray) -> None:
    first = arrays[0].shape
    if any(array.shape != first for array in arrays[1:]):
        raise ValueError("all response objects must have the same frequency and observable axes")


def causal_susceptibility(time: Iterable[float], correlation: Any, beta: float) -> np.ndarray:
    """Convert a scalar or matrix Kubo correlation into ``chi(t)=-beta*dC/dt``."""
    t = _validate_time(time)
    value = _validate_series(correlation, len(t), "correlation")
    beta_value = _real_scalar(beta, "beta", positive=True)
    return _readonly(-beta_value * np.gradient(value, t, axis=0, edge_order=2))


def frequency_response(time: Iterable[float], response: Any, frequencies: Iterable[float], *, damping=0.0) -> np.ndarray:
    """Fourier-Laplace transform a causal response on a finite time window."""
    t = _validate_time(time)
    w = _validate_frequencies(frequencies)
    chi_t = _validate_series(response, len(t), "response")
    eta = _real_scalar(damping, "damping", nonnegative=True)
    # Damping is independent of frequency: exp((i*w-eta)*t), rather than
    # exp(w*(i-eta)*t), which would incorrectly remove damping at w=0.
    kernel = np.exp(1j * np.outer(w, t) - eta * t[None, :])
    if chi_t.ndim == 1:
        result = _TRAPEZOID(kernel * chi_t[None, :], t, axis=1)
    else:
        result = _TRAPEZOID(kernel[:, :, None, None] * chi_t[None, :, :, :], t, axis=1)
    return _readonly(result)


def _matrix_inverse(response: np.ndarray, *, ridge: float, rcond: float):
    if response.ndim == 1:
        scale = np.abs(response)
        threshold = rcond * np.maximum(scale, _EPS)
        if ridge:
            inverse = response.conj() / (scale * scale + ridge * ridge)
        else:
            inverse = np.divide(1.0, response, out=np.zeros_like(response), where=scale > threshold)
        return inverse, np.where(scale > threshold, 1.0, np.inf), (scale > threshold).astype(np.int64)
    inverse = np.empty_like(response)
    condition = np.empty(response.shape[0], dtype=float)
    rank = np.empty(response.shape[0], dtype=np.int64)
    for i, matrix in enumerate(response):
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        cutoff = rcond * (singular[0] if singular.size else 0.0)
        if ridge:
            factors = singular / (singular * singular + ridge * ridge)
        else:
            factors = np.divide(1.0, singular, out=np.zeros_like(singular), where=singular > cutoff)
        inverse[i] = (vh.conj().T * factors) @ u.conj().T
        rank[i] = int(np.count_nonzero(singular > cutoff))
        condition[i] = float(singular[0] / singular[-1]) if singular.size and singular[-1] > cutoff else np.inf
    return inverse, condition, rank


@dataclass(frozen=True)
class InverseResponse:
    """Regularized inverse response and numerical diagnostics."""

    values: np.ndarray
    condition_number: np.ndarray
    effective_rank: np.ndarray
    ridge: float
    rcond: float

    def __post_init__(self):
        object.__setattr__(self, "values", _readonly(self.values, dtype=np.complex128))
        object.__setattr__(self, "condition_number", _readonly(self.condition_number, dtype=float))
        object.__setattr__(self, "effective_rank", _readonly(self.effective_rank, dtype=np.int64))


def regularized_inverse(response: Any, *, ridge=0.0, rcond=1e-12) -> InverseResponse:
    """Compute a scalar or matrix response inverse with optional Tikhonov ridge."""
    eta = _real_scalar(ridge, "ridge", nonnegative=True)
    cutoff = _real_scalar(rcond, "rcond", nonnegative=True)
    if cutoff >= 1:
        raise ValueError("rcond must be smaller than one")
    array = np.asarray(response)
    if array.ndim not in (1, 3) or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("response must be a finite scalar or square-matrix frequency series")
    if array.ndim == 3 and array.shape[1] != array.shape[2]:
        raise ValueError("matrix response samples must be square")
    values, condition, rank = _matrix_inverse(np.asarray(array, dtype=np.complex128), ridge=eta, rcond=cutoff)
    return InverseResponse(values, condition, rank, eta, cutoff)


def inverse_response_defect(rpmd_sub_response: Any, quantum_sub_response: Any, *, ridge=0.0, rcond=1e-12) -> np.ndarray:
    """Return ``chi_Q,sub^-1 - chi_RPMD,sub^-1`` for matched responses."""
    rpmd = np.asarray(rpmd_sub_response)
    quantum = np.asarray(quantum_sub_response)
    if rpmd.shape != quantum.shape:
        raise ValueError("RPMD and quantum subspace responses must have identical shape")
    return _readonly(regularized_inverse(quantum, ridge=ridge, rcond=rcond).values
                     - regularized_inverse(rpmd, ridge=ridge, rcond=rcond).values,
                     dtype=np.complex128)


def _negative_real_fraction(response: np.ndarray) -> float:
    if response.ndim == 1:
        eigenvalues = response.real[:, None]
    else:
        hermitian = .5 * (response + np.swapaxes(response.conj(), -1, -2))
        eigenvalues = np.linalg.eigvalsh(hermitian)
    negative = np.clip(-eigenvalues, 0., None).sum()
    return float(negative / max(np.abs(eigenvalues).sum(), _EPS))


def _norm(response: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(response) ** 2)))


@dataclass(frozen=True)
class QuantumCorrectionResult:
    """Auditable response-level correction with explicit non-convergence status."""

    frequencies: np.ndarray
    rpmd_full_response: np.ndarray
    rpmd_sub_response: np.ndarray
    quantum_sub_response: np.ndarray
    inverse_defect: np.ndarray
    corrected_response: np.ndarray
    diagnostics: Mapping[str, Any]
    provenance: Mapping[str, Any]
    physical_converged: bool = False

    def __post_init__(self):
        object.__setattr__(self, "frequencies", _readonly(_validate_frequencies(self.frequencies), dtype=float))
        for name in ("rpmd_full_response", "rpmd_sub_response", "quantum_sub_response", "inverse_defect", "corrected_response"):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=np.complex128))
        if not isinstance(self.diagnostics, Mapping) or not isinstance(self.provenance, Mapping):
            raise TypeError("diagnostics and provenance must be mappings")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def response(self):
        return self.corrected_response


def hybrid_quantum_correction(
    frequencies: Iterable[float],
    rpmd_full_response: Any,
    rpmd_sub_response: Any,
    quantum_sub_response: Any,
    *,
    inverse_ridge=0.0,
    output_ridge=0.0,
    rcond=1e-12,
    provenance: Mapping[str, Any] | None = None,
) -> QuantumCorrectionResult:
    """Apply a low-dimensional inverse-response quantum defect.

    Zero ridges preserve the algebraic zero-defect and full-subspace limits for
    nonsingular inputs. Positive ridges are explicit noise-stabilization choices
    and must be included in convergence studies.
    """
    w = _validate_frequencies(frequencies)
    full = _validate_frequency_response(rpmd_full_response, len(w), "rpmd_full_response")
    sub_rp = _validate_frequency_response(rpmd_sub_response, len(w), "rpmd_sub_response")
    sub_q = _validate_frequency_response(quantum_sub_response, len(w), "quantum_sub_response")
    _same_shape(sub_rp, sub_q)
    if full.ndim != sub_rp.ndim:
        raise ValueError("full and subspace responses must use the same scalar or matrix representation")
    if full.ndim == 3 and full.shape[1:] != sub_rp.shape[1:]:
        raise ValueError("full and subspace response matrices must use the same observable basis")
    full_inverse = regularized_inverse(full, ridge=inverse_ridge, rcond=rcond)
    sub_rp_inverse = regularized_inverse(sub_rp, ridge=inverse_ridge, rcond=rcond)
    sub_q_inverse = regularized_inverse(sub_q, ridge=inverse_ridge, rcond=rcond)
    defect = sub_q_inverse.values - sub_rp_inverse.values
    corrected_inverse = full_inverse.values + defect
    corrected_inverse_result = regularized_inverse(corrected_inverse, ridge=output_ridge, rcond=rcond)
    corrected = corrected_inverse_result.values
    diagnostics = {
        "inverse_ridge": float(inverse_ridge), "output_ridge": float(output_ridge), "rcond": float(rcond),
        "full_condition_number_max": float(np.max(full_inverse.condition_number)),
        "sub_rpmd_condition_number_max": float(np.max(sub_rp_inverse.condition_number)),
        "sub_quantum_condition_number_max": float(np.max(sub_q_inverse.condition_number)),
        "corrected_condition_number_max": float(np.max(corrected_inverse_result.condition_number)),
        "full_effective_rank_min": int(np.min(full_inverse.effective_rank)),
        "sub_rpmd_effective_rank_min": int(np.min(sub_rp_inverse.effective_rank)),
        "sub_quantum_effective_rank_min": int(np.min(sub_q_inverse.effective_rank)),
        "corrected_effective_rank_min": int(np.min(corrected_inverse_result.effective_rank)),
        "inverse_defect_rms": _norm(defect),
        "relative_inverse_defect": _norm(defect) / max(_norm(full), _EPS),
        "negative_real_fraction_full": _negative_real_fraction(full),
        "negative_real_fraction_corrected": _negative_real_fraction(corrected),
        "physical_converged": False,
        "approximation": "matched-subspace-inverse-response-defect",
        "assumptions": "low-dimensional quantum-minus-RPMD inverse response defect is transferable; no exact full-system rate identity is assumed",
    }
    prov = {"reference_kind": "response-level-RPMD-plus-low-dimensional-quantum",
            "frequency_points": int(len(w)),
            "observable_shape": list(full.shape[1:]) if full.ndim == 3 else [],
            "physical_converged": False}
    if provenance is not None:
        if not isinstance(provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        prov.update(dict(provenance))
    return QuantumCorrectionResult(w, full, sub_rp, sub_q, defect, corrected, diagnostics, prov)


def run_quantum_correction(
    time: Iterable[float],
    frequencies: Iterable[float],
    rpmd_full_correlation: Any,
    rpmd_sub_correlation: Any,
    quantum_sub_correlation: Any,
    *,
    beta: float,
    damping=0.0,
    inverse_ridge=0.0,
    output_ridge=0.0,
    rcond=1e-12,
    provenance: Mapping[str, Any] | None = None,
) -> QuantumCorrectionResult:
    """Run correlation -> causal response -> hybrid inverse-response correction."""
    t = _validate_time(time)
    full_c = _validate_series(rpmd_full_correlation, len(t), "rpmd_full_correlation")
    sub_rp_c = _validate_series(rpmd_sub_correlation, len(t), "rpmd_sub_correlation")
    sub_q_c = _validate_series(quantum_sub_correlation, len(t), "quantum_sub_correlation")
    _same_shape(sub_rp_c, sub_q_c)
    if full_c.ndim != sub_rp_c.ndim:
        raise ValueError("full and subspace correlations must use the same scalar or matrix representation")
    if full_c.ndim == 3 and full_c.shape[1:] != sub_rp_c.shape[1:]:
        raise ValueError("full and subspace correlations must use the same observable basis")
    full = frequency_response(t, causal_susceptibility(t, full_c, beta), frequencies, damping=damping)
    sub_rp = frequency_response(t, causal_susceptibility(t, sub_rp_c, beta), frequencies, damping=damping)
    sub_q = frequency_response(t, causal_susceptibility(t, sub_q_c, beta), frequencies, damping=damping)
    metadata = {"time_start": float(t[0]), "time_end": float(t[-1]), "time_points": int(t.size),
                "time_step_median": float(np.median(np.diff(t))), "beta": _real_scalar(beta, "beta", positive=True),
                "damping": _real_scalar(damping, "damping", nonnegative=True),
                "correlation_convention": "Kubo; chi(t)=-beta*dC(t)/dt",
                "transform_convention": "integral_0^infty exp((i*w-damping)*t) chi(t) dt",
                "causality_check": "constructed on the nonnegative-time window",
                "causality_residual": 0.0}
    if provenance is not None:
        if not isinstance(provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        metadata.update(dict(provenance))
    return hybrid_quantum_correction(frequencies, full, sub_rp, sub_q,
                                     inverse_ridge=inverse_ridge, output_ridge=output_ridge,
                                     rcond=rcond, provenance=metadata)


__all__ = ["InverseResponse", "QuantumCorrectionResult", "causal_susceptibility", "frequency_response",
           "regularized_inverse", "inverse_response_defect", "hybrid_quantum_correction", "run_quantum_correction"]
