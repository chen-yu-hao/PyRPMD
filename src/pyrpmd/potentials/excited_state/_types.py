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


class ExcitedStateStatus(IntEnum):
    """Stable provider status codes; zero is the only usable frame."""

    SUCCESS = 0
    SCF_NOT_CONVERGED = 1
    EXCITED_STATE_NOT_CONVERGED = 2
    STATE_COUNT_MISMATCH = 3
    GRADIENT_UNAVAILABLE = 4
    COUPLING_UNAVAILABLE = 5
    OVERLAP_UNAVAILABLE = 6
    STATE_TRACKING_FAILED = 7
    EXTERNAL_PROGRAM_FAILED = 8
    OUTPUT_PARSE_ERROR = 9
    SCRATCH_ERROR = 10
    ELECTRONIC_STRUCTURE_ERROR = 11
    INVALID_WAVEFUNCTION = 12
    TIMEOUT = 13


class BackendFailure(RuntimeError):
    """A diagnosed backend failure that maps to one provider status code."""

    def __init__(self, status: ExcitedStateStatus | int, message: str):
        try:
            status = ExcitedStateStatus(int(status))
        except (TypeError, ValueError) as exc:
            raise ValueError("backend failure status must be an ExcitedStateStatus") from exc
        if status is ExcitedStateStatus.SUCCESS:
            raise ValueError("BackendFailure cannot use SUCCESS")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("backend failure message must be nonempty")
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class BackendResult:
    """Raw one-bead electronic data returned by a backend.

    The state order and phases are backend-local.  ``overlap`` (when present)
    has rows in the previous *raw* backend order and columns in this result's
    raw order.  ``payload`` contains the wavefunction/orbitals required for the
    next overlap and SCF guess.
    """

    energies: np.ndarray
    gradients: np.ndarray
    payload: object
    derivative_couplings: np.ndarray | None = None
    overlap: np.ndarray | None = None
    oscillator_strengths: np.ndarray | None = None
    transition_dipoles: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExcitedStateFrameContext:
    """Opaque process-local context returned to ``AdiabaticPythonPotential``."""

    fingerprint: str
    geometry: np.ndarray
    payload: object
    permutation: np.ndarray
    phases: np.ndarray
    oscillator_strengths: np.ndarray | None = None
    transition_dipoles: np.ndarray | None = None
    time_derivative_couplings: np.ndarray | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        geometry = immutable_array(self.geometry, dtype=np.float64)
        permutation = immutable_array(self.permutation, dtype=np.int64)
        phases = immutable_array(self.phases)
        if geometry.ndim != 2 or geometry.shape[-1] != 3 or not np.isfinite(geometry).all():
            raise ValueError("context geometry must be finite [N,3]")
        nstates = len(permutation)
        if (
            permutation.shape != (nstates,)
            or not np.array_equal(np.sort(permutation), np.arange(nstates))
        ):
            raise ValueError("context permutation must contain every raw state once")
        if phases.shape != (nstates,) or not np.isfinite(phases).all() or np.any(np.abs(phases) == 0):
            raise ValueError("context phases must be finite nonzero [S]")
        phases = immutable_array(phases / np.abs(phases))
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "permutation", permutation)
        object.__setattr__(self, "phases", phases)
        if self.oscillator_strengths is not None:
            strengths = immutable_array(self.oscillator_strengths, dtype=np.float64)
            if strengths.shape != (nstates,) or not np.isfinite(strengths).all():
                raise ValueError("context oscillator strengths must be finite [S]")
            object.__setattr__(self, "oscillator_strengths", strengths)
        if self.transition_dipoles is not None:
            dipoles = immutable_array(self.transition_dipoles)
            if dipoles.shape != (nstates, 3) or not np.isfinite(dipoles).all():
                raise ValueError("context transition dipoles must be finite [S,3]")
            object.__setattr__(self, "transition_dipoles", dipoles)
        if self.time_derivative_couplings is not None:
            tdc = immutable_array(self.time_derivative_couplings)
            if tdc.shape != (nstates, nstates) or not np.isfinite(tdc).all():
                raise ValueError("context time-derivative couplings must be finite [S,S]")
            object.__setattr__(self, "time_derivative_couplings", tdc)
        object.__setattr__(self, "diagnostics", copy.deepcopy(dict(self.diagnostics)))


@dataclass(frozen=True)
class ProviderStatistics:
    call_count: int
    frame_count: int
    failure_count: int
    wall_seconds: float
    backend_seconds: float
    status_counts: Mapping[str, int]


class ExcitedStateBackend(ABC):
    """One-bead backend protocol.

    Implementations must not change numerical precision, device, or thread
    settings.  All coordinates are bohr, energies hartree, gradients
    hartree/bohr, and derivative couplings bohr**-1.
    """

    name = "abstract"
    program_version = "unknown"

    @abstractmethod
    def calculate(
        self,
        geometry: np.ndarray,
        scratch: Path,
        previous_payload: object | None,
    ) -> BackendResult:
        """Solve one geometry and return raw electronic data."""

    def overlap(
        self,
        previous_payload: object,
        current_payload: object,
        scratch: Path,
    ) -> np.ndarray | None:
        """Optionally compute ``<old_raw_i|new_raw_j>``."""

        return None

    def initial_derivative_couplings(
        self,
        geometry: np.ndarray,
        payload: object,
        scratch: Path,
    ) -> np.ndarray | None:
        """Optionally provide full NAC vectors for a first frame."""

        return None

    @abstractmethod
    def scientific_identity(self) -> Mapping[str, Any]:
        """Return a complete JSON-serializable scientific identity."""

    def serialize_payload(self, payload: object) -> Mapping[str, Any]:
        raise TypeError(f"{type(self).__name__} does not support context serialization")

    def deserialize_payload(self, data: Mapping[str, Any]) -> object:
        raise TypeError(f"{type(self).__name__} does not support context serialization")

    def close(self) -> None:
        pass
