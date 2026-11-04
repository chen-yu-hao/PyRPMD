"""Provider hooks, finite-difference NAC, identity, and persistence for PySCF."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ._pyscf_overlap import (
    PySCFTDWavefunction, _align_displaced_overlap, auxiliary_wavefunction_overlap,
)
from ._types import BackendFailure, BackendResult, ExcitedStateStatus
from ._util import canonical_json, sha256_text


class _PySCFRuntimeMixin:
    def calculate(
        self,
        geometry: np.ndarray,
        scratch: Path,
        previous_payload: object | None,
    ) -> BackendResult:
        return self._solve(
            geometry,
            previous_payload,
            gradients=True,
            properties=self.transition_properties,
        )

    def overlap(
        self,
        previous_payload: object,
        current_payload: object,
        scratch: Path,
    ) -> np.ndarray:
        return auxiliary_wavefunction_overlap(
            previous_payload,
            current_payload,
            max_determinant_pairs=self.max_determinant_pairs,
        )

    def initial_derivative_couplings(
        self,
        geometry: np.ndarray,
        payload: object,
        scratch: Path,
    ) -> np.ndarray | None:
        if self.initial_nac == "unavailable":
            return None
        if not isinstance(payload, PySCFTDWavefunction):
            raise BackendFailure(
                ExcitedStateStatus.INVALID_WAVEFUNCTION,
                "initial PySCF payload has the wrong type",
            )
        q = np.asarray(geometry, dtype=float)
        derivative = np.empty((self.nstates, self.nstates, len(self.symbols), 3))
        epsilon = self.nac_finite_difference_step
        for atom in range(len(self.symbols)):
            for axis in range(3):
                plus, minus = q.copy(), q.copy()
                plus[atom, axis] += epsilon
                minus[atom, axis] -= epsilon
                plus_result = self._solve(
                    plus, payload, gradients=False, properties=False
                )
                minus_result = self._solve(
                    minus, payload, gradients=False, properties=False
                )
                plus_overlap = _align_displaced_overlap(
                    auxiliary_wavefunction_overlap(
                        payload,
                        plus_result.payload,
                        max_determinant_pairs=self.max_determinant_pairs,
                    ),
                    subspace_tolerance=self.finite_difference_subspace_tolerance,
                )
                minus_overlap = _align_displaced_overlap(
                    auxiliary_wavefunction_overlap(
                        payload,
                        minus_result.payload,
                        max_determinant_pairs=self.max_determinant_pairs,
                    ),
                    subspace_tolerance=self.finite_difference_subspace_tolerance,
                )
                derivative[:, :, atom, axis] = np.real(
                    (plus_overlap - minus_overlap) / (2.0 * epsilon)
                )
        derivative = 0.5 * (derivative - derivative.swapaxes(0, 1))
        diagonal = np.arange(self.nstates)
        derivative[diagonal, diagonal] = 0.0
        if not np.isfinite(derivative).all():
            raise BackendFailure(
                ExcitedStateStatus.COUPLING_UNAVAILABLE,
                "finite-difference PySCF overlap derivative is nonfinite",
            )
        return derivative

    def scientific_identity(self) -> Mapping[str, Any]:
        settings = {
            "reference": "RKS" if self.xc is not None else "RHF",
            "response": "TDA" if self.tda else "TDDFT",
            "xc": self.xc,
            "charge": self.charge,
            "spin": self.spin,
            "scf_conv_tol": self.scf_conv_tol,
            "td_conv_tol": self.td_conv_tol,
            "max_scf_cycles": self.max_scf_cycles,
            "max_td_cycles": self.max_td_cycles,
            "grid_level": self.grid_level,
            "max_memory_mb": self.max_memory_mb,
            "scf_initial_guess": "projected-adjacent-density-v1",
            "transition_properties": self.transition_properties,
            "initial_nac": self.initial_nac,
            "nac_finite_difference_step_bohr": self.nac_finite_difference_step,
            "finite_difference_subspace_tolerance": self.finite_difference_subspace_tolerance,
            "max_determinant_pairs": self.max_determinant_pairs,
            "overlap_model": (
                "cis-determinant-x-v1"
                if self.tda
                else "casida-auxiliary-determinant-x-plus-y-v1"
            ),
        }
        return {
            "backend": self.name,
            "program_version": self.program_version,
            "symbols": list(self.symbols),
            "nstates": self.nstates,
            "basis": self.basis,
            "basis_sha256": sha256_text(self._basis_json),
            "ecp": self.ecp,
            "ecp_sha256": None if self._ecp_json is None else sha256_text(self._ecp_json),
            "input_template_sha256": sha256_text(canonical_json(settings)),
            **settings,
        }

    def serialize_payload(self, payload: object) -> Mapping[str, Any]:
        if not isinstance(payload, PySCFTDWavefunction):
            raise TypeError("PySCF TDDFT payload has the wrong type")
        return {
            "molecule": payload.molecule,
            "geometry": payload.geometry.tolist(),
            "mo_coeff": payload.mo_coeff.tolist(),
            "mo_occ": payload.mo_occ.tolist(),
            "amplitudes": [amplitude.tolist() for amplitude in payload.amplitudes],
            "tda": payload.tda,
        }

    def deserialize_payload(self, data: Mapping[str, Any]) -> PySCFTDWavefunction:
        if not isinstance(data, Mapping) or not isinstance(data.get("molecule"), str):
            raise ValueError("invalid serialized PySCF TDDFT payload")
        if bool(data.get("tda")) != self.tda:
            raise ValueError("serialized PySCF response model differs from backend")
        payload = PySCFTDWavefunction(
            data["molecule"],
            np.asarray(data.get("geometry"), dtype=float),
            np.asarray(data.get("mo_coeff"), dtype=float),
            np.asarray(data.get("mo_occ"), dtype=float),
            tuple(np.asarray(value, dtype=float) for value in data.get("amplitudes", [])),
            self.tda,
        )
        if len(payload.amplitudes) != self.nstates - 1:
            raise ValueError("serialized PySCF payload state count mismatch")
        return payload
