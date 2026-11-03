"""Analytic two-state avoided-crossing backend used for examples and tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ._util import finite_positive, immutable_array, positive_integer
from .base import BackendResult, ExcitedStateBackend


@dataclass(frozen=True)
class ToyWavefunction:
    geometry: np.ndarray
    vectors: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "geometry", immutable_array(self.geometry, dtype=np.float64))
        object.__setattr__(self, "vectors", immutable_array(self.vectors, dtype=np.float64))


class AvoidedCrossingBackend(ExcitedStateBackend):
    """Two-state linear vibronic Hamiltonian with exact gradients and NACs.

    The active coordinate is one Cartesian component ``x`` and

    ``H = harmonic_energy I + [[slope*x, coupling], [coupling, -slope*x]]``.

    It is deliberately small but exercises the same overlap convention and
    derivative-coupling signs as an electronic-structure backend.
    """

    name = "toy-avoided-crossing"
    program_version = "1"

    def __init__(
        self,
        natoms: int,
        *,
        atom_index: int = 0,
        axis: int = 0,
        slope: float = 0.02,
        coupling: float = 0.005,
        harmonic: float = 0.001,
        phase_flip_threshold: float | None = None,
    ):
        self.natoms = positive_integer(natoms, "natoms")
        self.atom_index = positive_integer(atom_index, "atom_index", minimum=0)
        if self.atom_index >= self.natoms:
            raise ValueError("atom_index is outside natoms")
        self.axis = positive_integer(axis, "axis", minimum=0)
        if self.axis > 2:
            raise ValueError("axis must be 0, 1, or 2")
        self.slope = finite_positive(slope, "slope")
        self.coupling = finite_positive(coupling, "coupling")
        self.harmonic = finite_positive(harmonic, "harmonic", allow_zero=True)
        if phase_flip_threshold is not None:
            if isinstance(phase_flip_threshold, (bool, np.bool_)) or not np.isfinite(phase_flip_threshold):
                raise ValueError("phase_flip_threshold must be finite or None")
            phase_flip_threshold = float(phase_flip_threshold)
        self.phase_flip_threshold = phase_flip_threshold

    def _solve(self, geometry: np.ndarray):
        q = np.asarray(geometry)
        if q.shape != (self.natoms, 3) or q.dtype.kind not in "fiu" or not np.isfinite(q).all():
            raise ValueError(f"toy geometry must be finite [{self.natoms},3]")
        x = float(q[self.atom_index, self.axis])
        common = 0.5 * self.harmonic * float(np.sum(q * q))
        hamiltonian = np.array(
            [[self.slope * x, self.coupling], [self.coupling, -self.slope * x]],
            dtype=float,
        )
        electronic, vectors = np.linalg.eigh(hamiltonian)
        if self.phase_flip_threshold is not None and x >= self.phase_flip_threshold:
            vectors[:, 1] *= -1.0
        energies = electronic + common

        gradients = np.broadcast_to(self.harmonic * q, (2, self.natoms, 3)).copy()
        derivative_h = np.array([[self.slope, 0.0], [0.0, -self.slope]])
        for state in range(2):
            gradients[state, self.atom_index, self.axis] += float(
                vectors[:, state] @ derivative_h @ vectors[:, state]
            )

        nac = np.zeros((2, 2, self.natoms, 3))
        matrix_element = float(vectors[:, 0] @ derivative_h @ vectors[:, 1])
        nac[0, 1, self.atom_index, self.axis] = matrix_element / (
            electronic[1] - electronic[0]
        )
        nac[1, 0] = -nac[0, 1]

        dipole_operator = np.array([[1.0, 0.0], [0.0, -1.0]])
        transition = np.zeros((2, 3))
        transition[0, self.axis] = float(vectors[:, 0] @ dipole_operator @ vectors[:, 0])
        transition[1, self.axis] = float(vectors[:, 0] @ dipole_operator @ vectors[:, 1])
        strengths = np.zeros(2)
        strengths[1] = (2.0 / 3.0) * (electronic[1] - electronic[0]) * float(
            transition[1] @ transition[1]
        )
        return energies, gradients, nac, vectors, strengths, transition

    def calculate(
        self,
        geometry: np.ndarray,
        scratch: Path,
        previous_payload: object | None,
    ) -> BackendResult:
        energies, gradients, nac, vectors, strengths, transition = self._solve(geometry)
        payload = ToyWavefunction(geometry, vectors)
        overlap = None
        if previous_payload is not None:
            if not isinstance(previous_payload, ToyWavefunction):
                raise ValueError("toy previous payload has the wrong type")
            overlap = previous_payload.vectors.T @ vectors
        return BackendResult(
            energies,
            gradients,
            payload,
            derivative_couplings=nac,
            overlap=overlap,
            oscillator_strengths=strengths,
            transition_dipoles=transition,
            metadata={"model": "two-state-linear-vibronic"},
        )

    def overlap(
        self,
        previous_payload: object,
        current_payload: object,
        scratch: Path,
    ) -> np.ndarray:
        if not isinstance(previous_payload, ToyWavefunction) or not isinstance(current_payload, ToyWavefunction):
            raise ValueError("toy overlap requires ToyWavefunction payloads")
        return previous_payload.vectors.T @ current_payload.vectors

    def initial_derivative_couplings(
        self,
        geometry: np.ndarray,
        payload: object,
        scratch: Path,
    ) -> np.ndarray:
        return self._solve(geometry)[2]

    def scientific_identity(self) -> Mapping[str, Any]:
        return {
            "backend": self.name,
            "program_version": self.program_version,
            "natoms": self.natoms,
            "atom_index": self.atom_index,
            "axis": self.axis,
            "slope_hartree_per_bohr": self.slope,
            "coupling_hartree": self.coupling,
            "harmonic_hartree_per_bohr2": self.harmonic,
            "phase_flip_threshold_bohr": self.phase_flip_threshold,
        }

    def serialize_payload(self, payload: object) -> Mapping[str, Any]:
        if not isinstance(payload, ToyWavefunction):
            raise TypeError("toy payload has the wrong type")
        return {"geometry": payload.geometry.tolist(), "vectors": payload.vectors.tolist()}

    def deserialize_payload(self, data: Mapping[str, Any]) -> ToyWavefunction:
        if not isinstance(data, Mapping):
            raise ValueError("serialized toy payload must be a mapping")
        return ToyWavefunction(np.asarray(data.get("geometry")), np.asarray(data.get("vectors")))
