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



from ._types import BackendFailure, ExcitedStateStatus

class ScratchManager:
    """Process- and provider-isolated scratch lifecycle."""

    _POLICIES = frozenset({"always", "success", "never"})

    def __init__(self, root: str | os.PathLike[str] | None, cleanup: str):
        if cleanup not in self._POLICIES:
            raise ValueError("scratch cleanup must be 'always', 'success', or 'never'")
        if root is not None:
            root_path = Path(root).expanduser()
            if root_path.exists() and not root_path.is_dir():
                raise ValueError("scratch_root must be a directory path")
            self._root: Path | None = root_path
            self._owns_root = False
        else:
            self._root = None
            self._owns_root = True
        self.cleanup = cleanup
        self._instance = f"pid-{os.getpid()}-{uuid.uuid4().hex}"
        self._instance_path: Path | None = None
        self._lock = threading.Lock()
        self._closed = False

    @property
    def root(self) -> Path | None:
        return self._root

    def _ensure_instance(self) -> Path:
        with self._lock:
            if self._closed:
                raise BackendFailure(ExcitedStateStatus.SCRATCH_ERROR, "scratch manager is closed")
            if self._instance_path is not None:
                return self._instance_path
            try:
                if self._root is None:
                    self._root = Path(tempfile.mkdtemp(prefix="pyrpmd-excited-state-"))
                else:
                    self._root.mkdir(parents=True, exist_ok=True)
                self._instance_path = self._root / self._instance
                self._instance_path.mkdir(mode=0o700, exist_ok=False)
            except OSError as exc:
                raise BackendFailure(
                    ExcitedStateStatus.SCRATCH_ERROR, f"cannot create excited-state scratch: {exc}"
                ) from exc
            return self._instance_path

    def create_call(self, call_id: int, nbeads: int) -> tuple[Path, tuple[Path, ...]]:
        instance = self._ensure_instance()
        call = instance / f"call-{call_id:08d}-{uuid.uuid4().hex[:12]}"
        try:
            call.mkdir(mode=0o700)
            beads = []
            for bead in range(nbeads):
                path = call / f"bead-{bead:06d}"
                path.mkdir(mode=0o700)
                beads.append(path)
        except OSError as exc:
            shutil.rmtree(call, ignore_errors=True)
            raise BackendFailure(
                ExcitedStateStatus.SCRATCH_ERROR, f"cannot create bead scratch directories: {exc}"
            ) from exc
        return call, tuple(beads)

    def finish_call(self, call: Path, *, success: bool) -> None:
        if self.cleanup == "always" or (self.cleanup == "success" and success):
            shutil.rmtree(call, ignore_errors=False)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            instance = self._instance_path
            root = self._root
        if self.cleanup == "always":
            if instance is not None and instance.exists():
                shutil.rmtree(instance, ignore_errors=True)
            if self._owns_root and root is not None and root.exists():
                shutil.rmtree(root, ignore_errors=True)
            return
        if self.cleanup == "success":
            # Successful call directories have already been removed. Preserve
            # any failed calls for diagnosis, including after provider.close().
            try:
                if instance is not None and instance.exists() and not any(instance.iterdir()):
                    instance.rmdir()
                if self._owns_root and root is not None and root.exists() and not any(root.iterdir()):
                    root.rmdir()
            except OSError:
                # close() is best-effort; evaluate() reports cleanup failures
                # when they affect a scientific result.
                pass
