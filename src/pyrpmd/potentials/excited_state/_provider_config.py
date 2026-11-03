"""Configuration, identity, and previous-frame validation for excited-state providers."""

from __future__ import annotations

from collections import Counter
import copy
import json
import os
from pathlib import Path
import threading
from typing import Any, Sequence

import numpy as np

from ._scratch import ScratchManager
from ._types import (
    BackendFailure, BackendResult, ExcitedStateBackend, ExcitedStateFrameContext,
    ExcitedStateStatus, ProviderStatistics,
)
from ._util import canonical_json, finite_positive, positive_integer, sha256_text
from ._validation import _validate_symbols
from .tracking import TrackingDiagnostics


class _ProviderConfigurationMixin:
    """Batch callback satisfying :class:`pyrpmd.adiabatic.AdiabaticResult`.

    A backend calculation is independent for every bead.  Previous electronic
    state is accepted only through ``previous_context``; no hidden global last
    frame is used, so interleaved trajectories do not contaminate one another.
    """

    metadata = {
        "requires_atomic_units": True,
        "execution_device": "cpu",
        "thread_safety": "context_serial",
    }

    def __init__(
        self,
        backend: ExcitedStateBackend,
        symbols: Sequence[str],
        nstates: int,
        *,
        coupling_method: str = "npi",
        require_overlap: bool = True,
        require_couplings: bool = True,
        phase_overlap_tolerance: float = 1e-8,
        subspace_tolerance: float = 0.25,
        degeneracy_tolerance: float = 1e-6,
        ambiguity_tolerance: float = 0.05,
        time_step: float | None = None,
        scratch_root: str | os.PathLike[str] | None = None,
        scratch_cleanup: str = "always",
    ):
        if not isinstance(backend, ExcitedStateBackend):
            raise TypeError("backend must implement ExcitedStateBackend")
        self.backend = backend
        self.symbols = _validate_symbols(symbols)
        self.nstates = positive_integer(nstates, "nstates")
        if coupling_method not in ("analytic", "hst", "npi"):
            raise ValueError("coupling_method must be 'analytic', 'hst', or 'npi'")
        if not isinstance(require_overlap, (bool, np.bool_)):
            raise ValueError("require_overlap must be boolean")
        if not isinstance(require_couplings, (bool, np.bool_)):
            raise ValueError("require_couplings must be boolean")
        self.coupling_method = coupling_method
        self.require_overlap = bool(require_overlap)
        self.require_couplings = bool(require_couplings)
        self.phase_overlap_tolerance = finite_positive(phase_overlap_tolerance, "phase_overlap_tolerance")
        self.subspace_tolerance = finite_positive(subspace_tolerance, "subspace_tolerance")
        if self.subspace_tolerance >= 1:
            raise ValueError("subspace_tolerance must be below one")
        self.degeneracy_tolerance = finite_positive(
            degeneracy_tolerance, "degeneracy_tolerance", allow_zero=True
        )
        self.ambiguity_tolerance = finite_positive(
            ambiguity_tolerance, "ambiguity_tolerance", allow_zero=True
        )
        self.time_step = None if time_step is None else finite_positive(time_step, "time_step")

        backend_identity = dict(backend.scientific_identity())
        if not backend_identity:
            raise ValueError("backend scientific identity must be nonempty")
        # Validate and detach before constructing the scratch manager or
        # creating a path.  A backend cannot mutate an already-issued run ID.
        backend_identity = json.loads(canonical_json(backend_identity))
        self._identity = {
            "type": "excited_state_provider",
            "version": 1,
            "symbols": list(self.symbols),
            "nstates": self.nstates,
            "coupling_method": coupling_method,
            "require_overlap": self.require_overlap,
            "require_couplings": self.require_couplings,
            "phase_overlap_tolerance": self.phase_overlap_tolerance,
            "subspace_tolerance": self.subspace_tolerance,
            "degeneracy_tolerance": self.degeneracy_tolerance,
            "ambiguity_tolerance": self.ambiguity_tolerance,
            "time_step": self.time_step,
            "backend": backend_identity,
        }
        identity_json = canonical_json(self._identity)
        self.fingerprint = sha256_text(identity_json)
        self._scratch = ScratchManager(scratch_root, scratch_cleanup)
        self._lock = threading.RLock()
        self._closed = False
        self._calls = 0
        self._frames = 0
        self._failures = 0
        self._wall_seconds = 0.0
        self._backend_seconds = 0.0
        self._status_counts: Counter[str] = Counter()
        self.metadata = {
            **type(self).metadata,
            "provider": "excited_state",
            "backend": backend.name,
            "program_version": str(backend.program_version),
            "continuous_gauge": False,
            "scientific_identity": self.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._identity)

    def statistics(self) -> ProviderStatistics:
        with self._lock:
            return ProviderStatistics(
                self._calls,
                self._frames,
                self._failures,
                self._wall_seconds,
                self._backend_seconds,
                dict(sorted(self._status_counts.items())),
            )

    def _validate_previous(self, q: np.ndarray, previous_q, previous_context):
        if (previous_q is None) != (previous_context is None):
            raise ValueError("previous_q and previous_context must be supplied together")
        if previous_q is None:
            return None, None
        previous_q = np.asarray(previous_q)
        if (
            previous_q.shape != q.shape
            or previous_q.dtype.kind not in "fiu"
            or not np.isfinite(previous_q).all()
        ):
            raise ValueError("previous_q must be finite real and match q [B,N,3]")
        if not isinstance(previous_context, (tuple, list)) or len(previous_context) != len(q):
            raise ValueError("previous_context must contain one frame context per bead")
        checked = []
        for bead, context in enumerate(previous_context):
            if not isinstance(context, ExcitedStateFrameContext):
                raise ValueError("previous_context contains an invalid excited-state frame")
            if context.fingerprint != self.fingerprint:
                raise ValueError("previous_context belongs to a different scientific identity")
            if context.geometry.shape != q.shape[1:] or not np.array_equal(context.geometry, previous_q[bead]):
                raise ValueError("previous_q differs from previous_context geometry")
            if len(context.permutation) != self.nstates:
                raise ValueError("previous_context state count differs from provider")
            checked.append(context)
        return np.ascontiguousarray(previous_q, dtype=np.float64), tuple(checked)

    def _raw_overlap(
        self,
        result: BackendResult,
        old: ExcitedStateFrameContext,
        scratch: Path,
    ) -> np.ndarray | None:
        overlap = result.overlap
        if overlap is None:
            overlap = self.backend.overlap(old.payload, result.payload, scratch)
        if overlap is None:
            return None
        raw = np.asarray(overlap)
        if raw.shape != (self.nstates, self.nstates) or not np.isfinite(raw).all():
            raise BackendFailure(
                ExcitedStateStatus.OVERLAP_UNAVAILABLE,
                f"backend overlap must be finite [{self.nstates},{self.nstates}]",
            )
        # old reported ket_i = phase_i * old raw ket_perm_i, hence its bra
        # contributes conjugate(phase_i).
        return np.conjugate(old.phases)[:, None] * raw[old.permutation, :]

    def _diagnostic_message(self, diagnostics: TrackingDiagnostics) -> str:
        parts = []
        if diagnostics.near_degenerate:
            pairs = ",".join(f"{i}-{j}" for i, j in diagnostics.near_degenerate_pairs)
            parts.append(
                f"near-degenerate states {pairs} (minimum gap {diagnostics.minimum_energy_gap:.3e} hartree)"
            )
        if diagnostics.ambiguous:
            pairs = ",".join(f"old {i}->raw {j}" for i, j in diagnostics.ambiguous_pairs)
            parts.append(f"ambiguous maximum-overlap assignment: {pairs}")
        return "; ".join(parts)
