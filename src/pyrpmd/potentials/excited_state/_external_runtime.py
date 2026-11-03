"""Calculation, finite-difference NAC, identity, and persistence for file backends."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ._external_common import ExternalWavefunction, _safe_command
from ._types import BackendFailure, BackendResult, ExcitedStateStatus
from ._util import canonical_json, sha256_text
from .tracking import StateTrackingError, align_reference_overlap


class _FileProgramRuntimeMixin:
    def calculate(
        self,
        geometry: np.ndarray,
        scratch: Path,
        previous_payload: object | None,
    ) -> BackendResult:
        q = np.asarray(geometry)
        if q.shape != (len(self.symbols), 3) or q.dtype.kind not in "fiu" or not np.isfinite(q).all():
            raise ValueError(f"geometry must be finite [{len(self.symbols)},3]")
        input_path = scratch / self.input_filename
        output_path = scratch / self.output_filename
        try:
            self._stage_identity_files(scratch)
            self.prepare_files(q, scratch)
            input_path.write_text(self.render_input(q), encoding="utf-8")
        except BackendFailure:
            raise
        except OSError as exc:
            raise BackendFailure(
                ExcitedStateStatus.SCRATCH_ERROR,
                f"cannot write {self.name} input files: {exc}",
            ) from exc
        old_dir = None
        if previous_payload is not None:
            if not isinstance(previous_payload, ExternalWavefunction):
                raise BackendFailure(ExcitedStateStatus.INVALID_WAVEFUNCTION, "invalid previous external wavefunction")
            old_dir = self._restore_previous(previous_payload, scratch)
        command = _safe_command(
            self.command,
            {
                "input": str(input_path),
                "output": str(output_path),
                "scratch": str(scratch),
                "previous": str(scratch if old_dir is None else old_dir),
            },
        )
        captured = self._run(command, scratch)
        # Some launchers write the requested output themselves (Q-Chem), while
        # others report everything on stdout (ORCA/OpenMolcas).  Never overwrite
        # a program-created output file.
        try:
            if output_path.is_file() and output_path.stat().st_size:
                text = output_path.read_text(encoding="utf-8", errors="replace")
            else:
                output_path.write_bytes(captured)
                text = captured.decode("utf-8", errors="replace")
        except OSError as exc:
            raise BackendFailure(
                ExcitedStateStatus.SCRATCH_ERROR,
                f"cannot read or retain {self.name} output: {exc}",
            ) from exc
        parsed = self.parse_output(text)
        payload = self._collect_artifacts(scratch)
        overlap = None if old_dir is None else self._run_overlap(old_dir, scratch)
        return parsed.as_backend_result(payload, overlap)

    def initial_derivative_couplings(
        self,
        geometry: np.ndarray,
        payload: object,
        scratch: Path,
    ) -> np.ndarray | None:
        if self.initial_nac == "unavailable":
            return None
        if not isinstance(payload, ExternalWavefunction):
            raise BackendFailure(
                ExcitedStateStatus.INVALID_WAVEFUNCTION,
                "initial external wavefunction payload has the wrong type",
            )
        q = np.asarray(geometry, dtype=float)
        root = scratch / "initial-nac-finite-difference"
        try:
            root.mkdir(mode=0o700)
        except OSError as exc:
            raise BackendFailure(
                ExcitedStateStatus.SCRATCH_ERROR,
                f"cannot create external finite-difference scratch: {exc}",
            ) from exc
        derivative = np.empty((self.nstates, self.nstates, len(self.symbols), 3))
        epsilon = self.nac_finite_difference_step
        for atom in range(len(self.symbols)):
            for axis in range(3):
                plus, minus = q.copy(), q.copy()
                plus[atom, axis] += epsilon
                minus[atom, axis] -= epsilon
                plus_dir = root / f"atom-{atom:06d}-axis-{axis}-plus"
                minus_dir = root / f"atom-{atom:06d}-axis-{axis}-minus"
                try:
                    plus_dir.mkdir(mode=0o700)
                    minus_dir.mkdir(mode=0o700)
                except OSError as exc:
                    raise BackendFailure(
                        ExcitedStateStatus.SCRATCH_ERROR,
                        f"cannot create external finite-difference bead scratch: {exc}",
                    ) from exc
                forward = self.calculate(plus, plus_dir, payload)
                backward = self.calculate(minus, minus_dir, payload)
                if forward.overlap is None or backward.overlap is None:
                    raise BackendFailure(
                        ExcitedStateStatus.OVERLAP_UNAVAILABLE,
                        "external initial NAC finite difference requires overlaps at both displacements",
                    )
                try:
                    forward_overlap = align_reference_overlap(
                        forward.overlap,
                        subspace_tolerance=self.finite_difference_subspace_tolerance,
                    )
                    backward_overlap = align_reference_overlap(
                        backward.overlap,
                        subspace_tolerance=self.finite_difference_subspace_tolerance,
                    )
                except (StateTrackingError, ValueError) as exc:
                    raise BackendFailure(
                        ExcitedStateStatus.STATE_TRACKING_FAILED, str(exc)
                    ) from exc
                derivative[:, :, atom, axis] = np.real(
                    (forward_overlap - backward_overlap) / (2.0 * epsilon)
                )
        derivative = 0.5 * (derivative - derivative.swapaxes(0, 1))
        diagonal = np.arange(self.nstates)
        derivative[diagonal, diagonal] = 0.0
        if not np.isfinite(derivative).all():
            raise BackendFailure(
                ExcitedStateStatus.COUPLING_UNAVAILABLE,
                "external finite-difference overlap derivative is nonfinite",
            )
        return derivative

    def scientific_identity(self) -> Mapping[str, Any]:
        return {
            "backend": self.name,
            "program_version": self.program_version,
            "symbols": list(self.symbols),
            "nstates": self.nstates,
            "method": self.method,
            "basis": self.basis,
            "basis_sha256": {
                "specification": sha256_text(self.basis),
                "files": list(self._basis_hashes),
            },
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "input_template_sha256": sha256_text(self.input_template),
            "ecp_sha256": list(self._ecp_hashes),
            "command": list(self.command),
            "overlap_command": None if self.overlap_command is None else list(self.overlap_command),
            "wavefunction_filenames": list(self.wavefunction_filenames),
            "initial_nac": self.initial_nac,
            "nac_finite_difference_step_bohr": self.nac_finite_difference_step,
            "finite_difference_subspace_tolerance": self.finite_difference_subspace_tolerance,
            "timeout_seconds": self.timeout_seconds,
            "max_artifact_bytes": self.max_artifact_bytes,
            "environment_keys": (
                None if self.environment is None else sorted(self.environment)
            ),
            "environment_sha256": (
                None
                if self.environment is None
                else sha256_text(canonical_json(dict(sorted(self.environment.items()))))
            ),
        }

    def serialize_payload(self, payload: object) -> Mapping[str, Any]:
        if not isinstance(payload, ExternalWavefunction):
            raise TypeError("external payload has the wrong type")
        return {
            "program": payload.program,
            "artifacts": {
                name: base64.b64encode(content).decode("ascii")
                for name, content in payload.artifacts.items()
            },
        }

    def deserialize_payload(self, data: Mapping[str, Any]) -> ExternalWavefunction:
        if not isinstance(data, Mapping) or not isinstance(data.get("artifacts"), Mapping):
            raise ValueError("invalid serialized external wavefunction")
        try:
            artifacts = {
                str(name): base64.b64decode(value, validate=True)
                for name, value in data["artifacts"].items()
            }
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid base64 external wavefunction artifact") from exc
        return ExternalWavefunction(str(data.get("program", "")), artifacts)
