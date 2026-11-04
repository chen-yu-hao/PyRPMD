"""Generic online excited-state provider framework.

Backends solve one bead at a time in atomic units.  The provider owns strict
batch validation, process/bead-isolated scratch directories, state and phase
tracking, overlap-derived path couplings, status mapping, scientific identity,
and restartable electronic contexts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
import uuid
from typing import Any, Mapping, Sequence

import numpy as np

from ._util import canonical_json, finite_positive, immutable_array, positive_integer, sha256_text
from .tracking import (
    StateTrackingError,
    TrackingDiagnostics,
    align_states,
    path_derivative_couplings,
    time_derivative_coupling,
)



from ._types import BackendFailure, BackendResult, ExcitedStateStatus

def _validate_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    if isinstance(symbols, (str, bytes)):
        raise ValueError("symbols must be a nonempty sequence")
    try:
        result = tuple(symbols)
    except TypeError as exc:
        raise ValueError("symbols must be a nonempty sequence") from exc
    if not result or any(not isinstance(symbol, str) or not re.fullmatch(r"[A-Z][a-z]?", symbol) for symbol in result):
        raise ValueError("symbols must contain standard element symbols")
    return result


def _validate_backend_result(result: BackendResult, *, nstates: int, natoms: int) -> BackendResult:
    if not isinstance(result, BackendResult):
        raise BackendFailure(
            ExcitedStateStatus.ELECTRONIC_STRUCTURE_ERROR,
            "backend must return BackendResult",
        )
    energies = np.asarray(result.energies)
    gradients = np.asarray(result.gradients)
    if energies.shape != (nstates,):
        raise BackendFailure(
            ExcitedStateStatus.STATE_COUNT_MISMATCH,
            f"backend returned {energies.shape} energies; expected [{nstates}]",
        )
    if energies.dtype.kind not in "fiu" or not np.isfinite(energies).all():
        raise BackendFailure(ExcitedStateStatus.ELECTRONIC_STRUCTURE_ERROR, "backend energies are nonfinite or nonreal")
    if gradients.shape != (nstates, natoms, 3):
        raise BackendFailure(
            ExcitedStateStatus.GRADIENT_UNAVAILABLE,
            f"backend gradients have shape {gradients.shape}; expected [{nstates},{natoms},3]",
        )
    if gradients.dtype.kind not in "fiu" or not np.isfinite(gradients).all():
        raise BackendFailure(ExcitedStateStatus.GRADIENT_UNAVAILABLE, "backend gradients are nonfinite or nonreal")

    couplings = None
    if result.derivative_couplings is not None:
        couplings = np.asarray(result.derivative_couplings)
        if (
            couplings.shape != (nstates, nstates, natoms, 3)
            or couplings.dtype.kind not in "fciu"
            or not np.isfinite(couplings).all()
        ):
            raise BackendFailure(
                ExcitedStateStatus.COUPLING_UNAVAILABLE,
                f"backend derivative couplings must be finite [{nstates},{nstates},{natoms},3]",
            )
        if not np.allclose(couplings, -couplings.swapaxes(0, 1).conj(), rtol=0.0, atol=1e-9):
            raise BackendFailure(
                ExcitedStateStatus.COUPLING_UNAVAILABLE,
                "backend derivative couplings are not anti-Hermitian",
            )

    overlap = None
    if result.overlap is not None:
        overlap = np.asarray(result.overlap)
        if (
            overlap.shape != (nstates, nstates)
            or overlap.dtype.kind not in "fciu"
            or not np.isfinite(overlap).all()
        ):
            raise BackendFailure(
                ExcitedStateStatus.OVERLAP_UNAVAILABLE,
                f"backend overlap must be finite [{nstates},{nstates}]",
            )

    strengths = None
    if result.oscillator_strengths is not None:
        strengths = np.asarray(result.oscillator_strengths)
        if strengths.shape != (nstates,) or strengths.dtype.kind not in "fiu" or not np.isfinite(strengths).all():
            raise BackendFailure(
                ExcitedStateStatus.ELECTRONIC_STRUCTURE_ERROR,
                f"oscillator strengths must be finite [{nstates}]",
            )

    dipoles = None
    if result.transition_dipoles is not None:
        dipoles = np.asarray(result.transition_dipoles)
        if dipoles.shape != (nstates, 3) or dipoles.dtype.kind not in "fciu" or not np.isfinite(dipoles).all():
            raise BackendFailure(
                ExcitedStateStatus.ELECTRONIC_STRUCTURE_ERROR,
                f"transition dipoles must be finite [{nstates},3]",
            )

    if not isinstance(result.metadata, Mapping):
        raise BackendFailure(ExcitedStateStatus.ELECTRONIC_STRUCTURE_ERROR, "backend metadata must be a mapping")
    try:
        metadata = json.loads(canonical_json(dict(result.metadata)))
    except ValueError as exc:
        raise BackendFailure(
            ExcitedStateStatus.ELECTRONIC_STRUCTURE_ERROR,
            "backend metadata must contain finite JSON values",
        ) from exc
    return BackendResult(
        immutable_array(energies, dtype=np.float64),
        immutable_array(gradients, dtype=np.float64),
        result.payload,
        None if couplings is None else immutable_array(couplings),
        None if overlap is None else immutable_array(overlap),
        None if strengths is None else immutable_array(strengths, dtype=np.float64),
        None if dipoles is None else immutable_array(dipoles),
        metadata,
    )


def _json_array(array: np.ndarray | None):
    if array is None:
        return None
    value = np.asarray(array)
    if np.iscomplexobj(value):
        return {"complex": True, "real": value.real.tolist(), "imag": value.imag.tolist()}
    return {"complex": False, "data": value.tolist()}


def _from_json_array(value, *, dtype=None):
    if value is None:
        return None
    if not isinstance(value, Mapping) or not isinstance(value.get("complex"), bool):
        raise ValueError("invalid serialized numerical array")
    if value["complex"]:
        array = np.asarray(value.get("real"), dtype=float) + 1j * np.asarray(value.get("imag"), dtype=float)
    else:
        array = np.asarray(value.get("data"), dtype=dtype)
    if not np.isfinite(array).all():
        raise ValueError("serialized context contains nonfinite values")
    return array
