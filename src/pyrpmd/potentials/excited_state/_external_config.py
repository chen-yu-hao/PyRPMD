"""Configuration and subprocess helpers for file-based excited-state backends."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import Mapping, Sequence

import numpy as np

from ._external_common import ExternalWavefunction, ParsedExternalResult, _identity_file_payloads, _safe_command
from ._types import BackendFailure, ExcitedStateStatus
from ._util import finite_positive, positive_integer


class _FileProgramConfigurationMixin:
    """Base class for ORCA/OpenMolcas/Q-Chem file interfaces.

    The class invokes an explicit argument vector with ``shell=False``.  Program
    output parsing and input generation are subclass responsibilities.  Binary
    wavefunction files are retained in the in-memory context up to an explicit
    byte limit so adjacent-frame overlaps do not depend on a scratch directory
    surviving cleanup.
    """

    input_filename = "job.inp"
    output_filename = "job.out"
    wavefunction_filenames: tuple[str, ...] = ()
    sidecar_filenames: tuple[str, ...] = ()

    def __init__(
        self,
        symbols: Sequence[str],
        nstates: int,
        *,
        program_version: str,
        method: str,
        basis: str,
        charge: int = 0,
        multiplicity: int = 1,
        command: Sequence[str],
        input_template: str,
        basis_files: Sequence[str | os.PathLike[str]] = (),
        ecp_files: Sequence[str | os.PathLike[str]] = (),
        overlap_command: Sequence[str] | None = None,
        initial_nac: str = "unavailable",
        nac_finite_difference_step: float = 2e-4,
        finite_difference_subspace_tolerance: float = 0.1,
        timeout_seconds: float = 3600.0,
        max_artifact_bytes: int = 256 * 1024 * 1024,
        environment: Mapping[str, str] | None = None,
    ):
        try:
            self.symbols = tuple(symbols)
        except TypeError as exc:
            raise ValueError("symbols must be a sequence") from exc
        if not self.symbols or any(
            not isinstance(symbol, str) or not re.fullmatch(r"[A-Z][a-z]?", symbol)
            for symbol in self.symbols
        ):
            raise ValueError("symbols must contain standard element symbols")
        self.nstates = positive_integer(nstates, "nstates")
        if not isinstance(program_version, str) or not program_version.strip():
            raise ValueError("program_version must be supplied explicitly")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("method must be nonempty")
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError("basis must be nonempty")
        if isinstance(charge, bool) or not isinstance(charge, (int, np.integer)):
            raise ValueError("charge must be an integer")
        self.charge = int(charge)
        self.multiplicity = positive_integer(multiplicity, "multiplicity")
        self.program_version = program_version.strip()
        self.method = method.strip()
        self.basis = basis.strip()
        self.command = tuple(command)
        self.overlap_command = None if overlap_command is None else tuple(overlap_command)
        if not isinstance(input_template, str) or not input_template.strip():
            raise ValueError("input_template must be nonempty")
        self.input_template = input_template
        self.timeout_seconds = finite_positive(timeout_seconds, "timeout_seconds")
        self.max_artifact_bytes = positive_integer(max_artifact_bytes, "max_artifact_bytes")
        self.environment = None if environment is None else dict(environment)
        if self.environment is not None and any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.environment.items()
        ):
            raise ValueError("environment must map strings to strings")
        if initial_nac not in ("unavailable", "finite_difference"):
            raise ValueError("initial_nac must be 'unavailable' or 'finite_difference'")
        self.initial_nac = initial_nac
        self.nac_finite_difference_step = finite_positive(
            nac_finite_difference_step, "nac_finite_difference_step"
        )
        self.finite_difference_subspace_tolerance = finite_positive(
            finite_difference_subspace_tolerance,
            "finite_difference_subspace_tolerance",
        )
        if self.finite_difference_subspace_tolerance >= 1.0:
            raise ValueError("finite_difference_subspace_tolerance must be below one")
        self._basis_hashes, basis_payloads = _identity_file_payloads(basis_files, "basis")
        self._ecp_hashes, ecp_payloads = _identity_file_payloads(ecp_files, "ECP")
        identity_payloads = (*basis_payloads, *ecp_payloads)
        identity_names = [name for name, _ in identity_payloads]
        if len(identity_names) != len(set(identity_names)):
            raise ValueError("basis and ECP files must have distinct basenames")
        reserved = {
            self.input_filename,
            self.output_filename,
            "overlap.dat",
            "previous",
            *self.wavefunction_filenames,
            *self.sidecar_filenames,
        }
        collision = sorted(set(identity_names).intersection(reserved))
        if collision:
            raise ValueError(
                "basis/ECP filenames collide with backend-managed files: "
                + ", ".join(collision)
            )
        self._identity_file_payloads = identity_payloads
        # Validate placeholders before any external program or scratch side effect.
        _safe_command(
            self.command,
            {
                "input": self.input_filename,
                "output": self.output_filename,
                "scratch": ".",
                "previous": "previous",
            },
        )
        if self.overlap_command is not None:
            _safe_command(
                self.overlap_command,
                {"old_dir": "old", "new_dir": ".", "matrix": "overlap.dat", "nstates": str(self.nstates)},
            )

    def render_input(self, geometry: np.ndarray) -> str:
        raise NotImplementedError

    def parse_output(self, text: str) -> ParsedExternalResult:
        raise NotImplementedError

    def _environment(self):
        if self.environment is None:
            return None
        result = os.environ.copy()
        result.update(self.environment)
        return result

    def _run(self, command: Sequence[str], scratch: Path):
        try:
            completed = subprocess.run(
                list(command),
                cwd=scratch,
                env=self._environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                shell=False,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendFailure(
                ExcitedStateStatus.TIMEOUT,
                f"{self.name} exceeded timeout_seconds={self.timeout_seconds:g}",
            ) from exc
        except OSError as exc:
            raise BackendFailure(
                ExcitedStateStatus.EXTERNAL_PROGRAM_FAILED,
                f"cannot start {self.name}: {exc}",
            ) from exc
        output = completed.stdout or b""
        if completed.returncode != 0:
            tail = output.decode("utf-8", errors="replace")[-1000:].strip()
            raise BackendFailure(
                ExcitedStateStatus.EXTERNAL_PROGRAM_FAILED,
                f"{self.name} exited with code {completed.returncode}: {tail}",
            )
        return output

    def _restore_previous(self, payload: ExternalWavefunction, scratch: Path) -> Path:
        if payload.program != self.name:
            raise BackendFailure(
                ExcitedStateStatus.INVALID_WAVEFUNCTION,
                f"previous wavefunction belongs to {payload.program}, not {self.name}",
            )
        destination = scratch / "previous"
        try:
            destination.mkdir(mode=0o700)
            for name, content in payload.artifacts.items():
                path = destination / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        except OSError as exc:
            raise BackendFailure(
                ExcitedStateStatus.SCRATCH_ERROR,
                f"cannot restore previous {self.name} wavefunction: {exc}",
            ) from exc
        return destination

    def _collect_artifacts(self, scratch: Path) -> ExternalWavefunction:
        artifacts: dict[str, bytes] = {}
        total = 0
        for name in self.wavefunction_filenames:
            path = scratch / name
            if not path.is_file():
                continue
            try:
                content = path.read_bytes()
            except OSError as exc:
                raise BackendFailure(
                    ExcitedStateStatus.SCRATCH_ERROR,
                    f"cannot retain {self.name} wavefunction artifact {name!r}: {exc}",
                ) from exc
            total += len(content)
            if total > self.max_artifact_bytes:
                raise BackendFailure(
                    ExcitedStateStatus.INVALID_WAVEFUNCTION,
                    f"retained wavefunction artifacts exceed max_artifact_bytes={self.max_artifact_bytes}",
                )
            artifacts[name] = content
        return ExternalWavefunction(self.name, artifacts)

    def _run_overlap(self, old_dir: Path, scratch: Path) -> np.ndarray | None:
        if self.overlap_command is None:
            return None
        matrix_path = scratch / "overlap.dat"
        command = _safe_command(
            self.overlap_command,
            {
                "old_dir": str(old_dir),
                "new_dir": str(scratch),
                "matrix": str(matrix_path),
                "nstates": str(self.nstates),
            },
        )
        self._run(command, scratch)
        if not matrix_path.is_file():
            raise BackendFailure(
                ExcitedStateStatus.OVERLAP_UNAVAILABLE,
                "external overlap helper did not create overlap.dat",
            )
        try:
            overlap = np.loadtxt(matrix_path, dtype=float)
        except (OSError, ValueError) as exc:
            raise BackendFailure(
                ExcitedStateStatus.OVERLAP_UNAVAILABLE,
                f"cannot parse external overlap matrix: {exc}",
            ) from exc
        if overlap.shape != (self.nstates, self.nstates) or not np.isfinite(overlap).all():
            raise BackendFailure(
                ExcitedStateStatus.OVERLAP_UNAVAILABLE,
                f"external overlap matrix must be finite [{self.nstates},{self.nstates}]",
            )
        return overlap

    def _stage_identity_files(self, scratch: Path) -> None:
        try:
            for name, content in self._identity_file_payloads:
                (scratch / name).write_bytes(content)
        except OSError as exc:
            raise BackendFailure(
                ExcitedStateStatus.SCRATCH_ERROR,
                f"cannot stage {self.name} basis/ECP files: {exc}",
            ) from exc

    def prepare_files(self, geometry: np.ndarray, scratch: Path) -> None:
        """Optionally write program-specific sidecar input files."""
