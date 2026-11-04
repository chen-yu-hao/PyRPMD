"""Strict file-based excited-state backend infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
import base64
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from ._util import canonical_json, finite_positive, positive_integer, sha256_bytes, sha256_text
from .base import (
    BackendFailure,
    BackendResult,
    ExcitedStateBackend,
    ExcitedStateStatus,
)
from .tracking import StateTrackingError, align_reference_overlap


@dataclass(frozen=True)
class ExternalWavefunction:
    program: str
    artifacts: Mapping[str, bytes]

    def __post_init__(self):
        if not isinstance(self.program, str) or not self.program:
            raise ValueError("external wavefunction program must be nonempty")
        copied: dict[str, bytes] = {}
        for name, content in dict(self.artifacts).items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or not name:
                raise ValueError("external wavefunction artifact names must be relative")
            if not isinstance(content, (bytes, bytearray)):
                raise ValueError("external wavefunction artifacts must contain bytes")
            copied[name] = bytes(content)
        object.__setattr__(self, "artifacts", copied)


@dataclass(frozen=True)
class ParsedExternalResult:
    energies: np.ndarray
    gradients: np.ndarray
    oscillator_strengths: np.ndarray | None = None
    transition_dipoles: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None

    def as_backend_result(self, payload: object, overlap: np.ndarray | None = None) -> BackendResult:
        return BackendResult(
            self.energies,
            self.gradients,
            payload,
            overlap=overlap,
            oscillator_strengths=self.oscillator_strengths,
            transition_dipoles=self.transition_dipoles,
            metadata={} if self.metadata is None else self.metadata,
        )


def parse_gradient_blocks(text: str, *, nstates: int, natoms: int) -> np.ndarray:
    """Parse explicit ``PYRPMD GRADIENT STATE k`` blocks from recorded output.

    The marker is designed for wrappers around programs whose native gradient
    formatting changes between releases.  Each following nonblank line must be
    ``atom_index gx gy gz``; exactly ``natoms`` rows are required.  Values are
    positive energy gradients in hartree/bohr.
    """

    gradients = np.full((nstates, natoms, 3), np.nan)
    lines = text.splitlines()
    for index, line in enumerate(lines):
        fields = line.strip().split()
        if len(fields) != 4 or fields[:3] != ["PYRPMD", "GRADIENT", "STATE"]:
            continue
        try:
            state = int(fields[3])
        except ValueError as exc:
            raise BackendFailure(ExcitedStateStatus.OUTPUT_PARSE_ERROR, "invalid gradient-state marker") from exc
        if not 0 <= state < nstates:
            continue
        rows = []
        index_base = None
        cursor = index + 1
        while cursor < len(lines) and len(rows) < natoms:
            value = lines[cursor].strip()
            cursor += 1
            if not value:
                continue
            parts = value.replace("D", "E").replace("d", "e").split()
            if len(parts) != 4:
                break
            try:
                atom = int(parts[0])
                vector = [float(component) for component in parts[1:]]
            except ValueError:
                break
            if index_base is None:
                if atom not in (0, 1):
                    raise BackendFailure(
                        ExcitedStateStatus.OUTPUT_PARSE_ERROR,
                        f"gradient atom index {atom} is inconsistent in state {state}",
                    )
                index_base = atom
            if atom != len(rows) + index_base:
                raise BackendFailure(
                    ExcitedStateStatus.OUTPUT_PARSE_ERROR,
                    f"gradient atom index {atom} is inconsistent in state {state}",
                )
            rows.append(vector)
        if len(rows) != natoms:
            raise BackendFailure(
                ExcitedStateStatus.GRADIENT_UNAVAILABLE,
                f"state {state} gradient has {len(rows)} atoms; expected {natoms}",
            )
        gradients[state] = rows
    missing = np.flatnonzero(~np.isfinite(gradients).all(axis=(1, 2)))
    if len(missing):
        raise BackendFailure(
            ExcitedStateStatus.GRADIENT_UNAVAILABLE,
            "missing gradient blocks for states " + ", ".join(map(str, missing)),
        )
    return gradients


def parse_transition_properties(text: str, *, nstates: int):
    strengths = np.zeros(nstates)
    dipoles = np.zeros((nstates, 3))
    seen = False
    for line in text.splitlines():
        fields = line.strip().split()
        if len(fields) != 10 or fields[:3] != ["PYRPMD", "PROPERTY", "STATE"]:
            continue
        try:
            state = int(fields[3])
            if not 0 <= state < nstates or fields[4] != "F" or fields[6] != "MU":
                raise ValueError
            strengths[state] = float(fields[5].replace("D", "E").replace("d", "e"))
            dipoles[state] = [
                float(value.replace("D", "E").replace("d", "e"))
                for value in fields[7:10]
            ]
        except (ValueError, IndexError) as exc:
            raise BackendFailure(ExcitedStateStatus.OUTPUT_PARSE_ERROR, "invalid transition-property line") from exc
        seen = True
    return (strengths, dipoles) if seen else (None, None)


def _safe_command(command: Sequence[str], replacements: Mapping[str, str]) -> list[str]:
    if isinstance(command, (str, bytes)):
        raise ValueError("external commands must be an argument sequence, not a shell string")
    result = []
    for argument in command:
        if not isinstance(argument, str) or not argument:
            raise ValueError("external command arguments must be nonempty strings")
        try:
            rendered = argument.format_map(replacements)
        except KeyError as exc:
            raise ValueError(f"unknown external command placeholder {exc.args[0]!r}") from exc
        if "\x00" in rendered:
            raise ValueError("external command contains a NUL byte")
        result.append(rendered)
    if not result:
        raise ValueError("external command must not be empty")
    return result


def _identity_file_payloads(paths: Sequence[str | os.PathLike[str]], label: str):
    records = []
    payloads = []
    names = set()
    for value in paths:
        path = Path(value)
        name = path.name
        if not name or name in names:
            raise ValueError(f"{label} files must have unique nonempty basenames")
        names.add(name)
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot read {label} identity file {path}: {exc}") from exc
        records.append({"name": name, "sha256": sha256_bytes(content)})
        payloads.append((name, content))
    return tuple(records), tuple(payloads)
