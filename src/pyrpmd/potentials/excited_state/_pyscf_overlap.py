"""Optional PySCF TDA/TDDFT online excited-state backend.

The backend is deliberately imported lazily.  It supplies total-state energies,
analytic nuclear gradients supported by PySCF, ground-to-state transition
properties, and adjacent-geometry overlaps of Casida auxiliary many-electron
wavefunctions.  For TDA the latter are CIS wavefunctions built from the PySCF
X amplitudes.  For full TDDFT the documented auxiliary ``X+Y`` single-
excitation representation is used; this approximation is part of the run
identity and is never presented as an analytic TDDFT NAC vector.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ._util import (
    canonical_json,
    finite_positive,
    immutable_array,
    positive_integer,
    sha256_text,
)
from .base import (
    BackendFailure,
    BackendResult,
    ExcitedStateBackend,
    ExcitedStateStatus,
)
from .tracking import align_reference_overlap


@dataclass(frozen=True)
class PySCFTDWavefunction:
    molecule: str
    geometry: np.ndarray
    mo_coeff: np.ndarray
    mo_occ: np.ndarray
    amplitudes: tuple[np.ndarray, ...]
    tda: bool

    def __post_init__(self):
        object.__setattr__(self, "geometry", immutable_array(self.geometry, dtype=np.float64))
        object.__setattr__(self, "mo_coeff", immutable_array(self.mo_coeff, dtype=np.float64))
        object.__setattr__(self, "mo_occ", immutable_array(self.mo_occ, dtype=np.float64))
        object.__setattr__(
            self,
            "amplitudes",
            tuple(immutable_array(amplitude, dtype=np.float64) for amplitude in self.amplitudes),
        )

    @property
    def mol(self):
        from pyscf import gto

        return gto.loads(self.molecule)


def _normalized_auxiliary_amplitudes(x, y, *, tda: bool) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2 or x.dtype.kind not in "fciu" or not np.isfinite(x).all():
        raise BackendFailure(
            ExcitedStateStatus.INVALID_WAVEFUNCTION,
            "PySCF response X amplitude must be a finite [occupied,virtual] matrix",
        )
    if tda:
        amplitude = np.asarray(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64)
    else:
        if np.isscalar(y):
            y = np.zeros_like(x)
        y = np.asarray(y)
        if y.shape != x.shape or y.dtype.kind not in "fciu" or not np.isfinite(y).all():
            raise BackendFailure(
                ExcitedStateStatus.INVALID_WAVEFUNCTION,
                "PySCF response Y amplitude must match X",
            )
        amplitude = x + y
    norm = float(np.sqrt(2.0 * np.vdot(amplitude, amplitude).real))
    if not np.isfinite(norm) or norm <= 1e-14:
        raise BackendFailure(
            ExcitedStateStatus.INVALID_WAVEFUNCTION,
            "PySCF response amplitude has zero or nonfinite norm",
        )
    amplitude = amplitude / norm
    if np.max(np.abs(np.imag(amplitude))) > 1e-12:
        raise BackendFailure(
            ExcitedStateStatus.INVALID_WAVEFUNCTION,
            "complex PySCF response amplitudes are not supported by the real overlap implementation",
        )
    return np.asarray(np.real(amplitude), dtype=np.float64)


def _state_expansions(
    amplitudes: tuple[np.ndarray, ...],
    nocc: int,
    nmo: int,
    *,
    coefficient_tolerance: float = 0.0,
):
    occupied = tuple(range(nocc))
    states: list[list[tuple[tuple[int, ...], tuple[int, ...], float]]] = [
        [(occupied, occupied, 1.0)]
    ]
    for amplitude in amplitudes:
        if amplitude.shape != (nocc, nmo - nocc):
            raise BackendFailure(
                ExcitedStateStatus.INVALID_WAVEFUNCTION,
                "response amplitude dimensions do not match occupied/virtual orbitals",
            )
        expansion = []
        for i, a in np.ndindex(amplitude.shape):
            coefficient = float(amplitude[i, a])
            if abs(coefficient) <= coefficient_tolerance:
                continue
            orbital = nocc + a
            alpha = list(occupied)
            alpha[i] = orbital
            beta = list(occupied)
            beta[i] = orbital
            # For a restricted singlet and PySCF's normalization, X_ia is the
            # coefficient of each alpha/beta singly-excited determinant.
            expansion.append((tuple(alpha), occupied, coefficient))
            expansion.append((occupied, tuple(beta), coefficient))
        if not expansion:
            raise BackendFailure(
                ExcitedStateStatus.INVALID_WAVEFUNCTION,
                "response state has no retained single-excitation coefficients",
            )
        states.append(expansion)
    return states


def auxiliary_wavefunction_overlap(
    old: PySCFTDWavefunction,
    new: PySCFTDWavefunction,
    *,
    max_determinant_pairs: int = 2_000_000,
) -> np.ndarray:
    """Return ``<old_i|new_j>`` for TDA/TDDFT auxiliary wavefunctions.

    Determinant overlaps are evaluated from the cross-geometry AO overlap, so
    orbital relaxation and moving atom-centered basis functions are included.
    ``max_determinant_pairs`` is a hard explicit cost bound; exceeding it fails
    rather than switching to a lower-fidelity approximation.
    """

    if not isinstance(old, PySCFTDWavefunction) or not isinstance(new, PySCFTDWavefunction):
        raise TypeError("PySCF overlap requires PySCFTDWavefunction payloads")
    if old.tda != new.tda or len(old.amplitudes) != len(new.amplitudes):
        raise BackendFailure(
            ExcitedStateStatus.INVALID_WAVEFUNCTION,
            "adjacent PySCF contexts use different response models or state counts",
        )
    max_determinant_pairs = positive_integer(
        max_determinant_pairs, "max_determinant_pairs"
    )
    from pyscf import gto

    old_mol, new_mol = old.mol, new.mol
    ao_overlap = gto.intor_cross("int1e_ovlp", old_mol, new_mol)
    orbital = old.mo_coeff.T @ ao_overlap @ new.mo_coeff
    if not np.isfinite(orbital).all():
        raise BackendFailure(
            ExcitedStateStatus.INVALID_WAVEFUNCTION,
            "nonfinite cross-geometry molecular-orbital overlap",
        )
    old_occ = np.flatnonzero(old.mo_occ > 1e-8)
    new_occ = np.flatnonzero(new.mo_occ > 1e-8)
    if (
        not np.allclose(old.mo_occ[old_occ], 2.0, rtol=0.0, atol=1e-8)
        or not np.allclose(new.mo_occ[new_occ], 2.0, rtol=0.0, atol=1e-8)
        or len(old_occ) != len(new_occ)
        or not np.array_equal(old_occ, np.arange(len(old_occ)))
        or not np.array_equal(new_occ, np.arange(len(new_occ)))
    ):
        raise BackendFailure(
            ExcitedStateStatus.INVALID_WAVEFUNCTION,
            "auxiliary overlap currently requires closed-shell contiguous doubly occupied orbitals",
        )
    nocc = len(old_occ)
    if old.mo_coeff.shape[1] != new.mo_coeff.shape[1]:
        raise BackendFailure(
            ExcitedStateStatus.INVALID_WAVEFUNCTION,
            "adjacent PySCF wavefunctions have different MO dimensions",
        )
    nmo = old.mo_coeff.shape[1]
    old_states = _state_expansions(old.amplitudes, nocc, nmo)
    new_states = _state_expansions(new.amplitudes, nocc, nmo)
    required = sum(len(left) * len(right) for left in old_states for right in new_states)
    if required > max_determinant_pairs:
        raise BackendFailure(
            ExcitedStateStatus.OVERLAP_UNAVAILABLE,
            f"auxiliary overlap requires {required} determinant pairs, above "
            f"max_determinant_pairs={max_determinant_pairs}",
        )

    cache: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}

    def determinant(rows: tuple[int, ...], columns: tuple[int, ...]) -> float:
        key = (rows, columns)
        value = cache.get(key)
        if value is None:
            value = float(np.linalg.det(orbital[np.ix_(rows, columns)]))
            cache[key] = value
        return value

    overlap = np.zeros((len(old_states), len(new_states)))
    for i, left in enumerate(old_states):
        for j, right in enumerate(new_states):
            value = 0.0
            for alpha_old, beta_old, coefficient_old in left:
                for alpha_new, beta_new, coefficient_new in right:
                    value += (
                        coefficient_old
                        * coefficient_new
                        * determinant(alpha_old, alpha_new)
                        * determinant(beta_old, beta_new)
                    )
            overlap[i, j] = value
    if not np.isfinite(overlap).all():
        raise BackendFailure(
            ExcitedStateStatus.OVERLAP_UNAVAILABLE,
            "nonfinite PySCF auxiliary wavefunction overlap",
        )
    return overlap


def _align_displaced_overlap(overlap: np.ndarray, *, subspace_tolerance: float) -> np.ndarray:
    """Backward-compatible private alias for the shared FD alignment helper."""

    try:
        return align_reference_overlap(
            overlap,
            phase_overlap_tolerance=1e-8,
            subspace_tolerance=subspace_tolerance,
        )
    except Exception as exc:
        if isinstance(exc, BackendFailure):
            raise
        raise BackendFailure(ExcitedStateStatus.STATE_TRACKING_FAILED, str(exc)) from exc
