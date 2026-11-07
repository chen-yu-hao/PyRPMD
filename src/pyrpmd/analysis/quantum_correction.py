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
``chi(w) = integral_0^infinity exp((i*w-eta)*t) chi(t) dt``. For a Kubo
correlation ``C(t)``, the causal response is ``chi(t)=-beta*dC/dt`` for
``t>=0``. The t=0 sample receives the usual half trapezoid weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

_EPS = np.finfo(float).eps
try:
    # NumPy 2.0+ exposes ``trapezoid`` and recent releases may no longer
    # expose the deprecated ``trapz`` alias.  Resolve the fallback lazily so
    # importing this module remains compatible with both API generations.
    _TRAPEZOID = np.trapezoid
except AttributeError:  # pragma: no cover - exercised on older NumPy
    _TRAPEZOID = np.trapz


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
        raise ValueError(f"{name} must be strictly increasing and nonegative")
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
    if array.ndim == 3 and array.shape,²H†@LƒCStsvöÆF–öà