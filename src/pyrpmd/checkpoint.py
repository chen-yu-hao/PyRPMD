"""Versioned, atomic NPZ containers with per-array integrity verification.

Version 2 adds a reserved envelope without changing workflow payload names.
Legacy NPZ files remain readable; their lack of checksums is reported explicitly.
No object array or pickle is read or written by this module.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from . import __version__

SCHEMA = "pyrpmd.checkpoint"
SCHEMA_VERSION = 2
ENVELOPE = "_pyrpmd_checkpoint"


class CheckpointError(ValueError):
    """An unsupported or damaged checkpoint must not be used to propagate."""


def _array_description(array):
    array = np.asarray(array)
    if array.dtype.hasobject:
        raise CheckpointError("Checkpoint object arrays are forbidden")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def atomic_savez(path, *, kind="arrays", **arrays):
    """Commit a complete payload and its integrity envelope with one replace."""
    if ENVELOPE in arrays:
        raise CheckpointError(f"{ENVELOPE} is reserved for the container schema")
    if not isinstance(kind, str) or not kind:
        raise CheckpointError("Checkpoint kind must be a nonempty string")
    # Own the payload until the atomic replacement has completed.
    payload = {name: np.array(value, copy=True) for name, value in arrays.items()}
    envelope = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "writer_version": __version__,
        "arrays": {name: _array_description(value) for name, value in payload.items()},
    }
    payload[ENVELOPE] = np.asarray(json.dumps(envelope, sort_keys=True, allow_nan=False))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _payload_format(archive, names, *, legacy=False):
    """Read an optional payload schema independently of its NPZ container.

    The integrity-envelope version describes storage, not the meaning of a
    workflow's arrays. A future payload cannot become a v1 payload merely by
    being stored in a supported v2 container.
    """
    if "metadata" not in names:
        return None
    try:
        metadata = json.loads(str(archive["metadata"].item()))
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        payload_format = metadata.get("format")
    except (ValueError, TypeError, AttributeError) as error:
        raise CheckpointError("Invalid checkpoint payload metadata") from error
    if payload_format is None and "format" not in metadata:
        return None
    if not isinstance(payload_format, str) or not payload_format:
        raise CheckpointError("Checkpoint payload format must be a nonempty string")
    # PyRPMD's current workflow payloads are v1. User-defined kinds own their
    # own version policy and can be checked with expected_kind instead.
    match = re.search(r"\.v(\d+)$", payload_format) if payload_format.startswith("PyRPMD.") else None
    if match and int(match.group(1)) != 1:
        label = "legacy payload" if legacy else "checkpoint payload"
        raise CheckpointError(f"Unsupported {label} format: {payload_format}")
    return payload_format


def _validate(archive, *, expected_kind=None, require_integrity=False):
    names = set(archive.files)
    if ENVELOPE not in names:
        if require_integrity:
            raise CheckpointError("Legacy checkpoint has no integrity envelope")
        # Materialize every payload before admitting the checkpoint, including
        # ZIP CRC checks and allow_pickle=False validation.
        for name in names:
            _array_description(archive[name])
        payload_format = _payload_format(archive, names, legacy=True)
        if expected_kind is not None and payload_format != expected_kind:
            raise CheckpointError(
                f"Expected checkpoint kind {expected_kind!r}, got legacy payload {payload_format!r}"
            )
        return {
            "schema": "legacy.npz",
            "schema_version": 1,
            "kind": None,
            "integrity": "unverified_legacy",
        }
    try:
        envelope = json.loads(str(archive[ENVELOPE].item()))
    except (ValueError, TypeError) as error:
        raise CheckpointError("Invalid checkpoint integrity envelope") from error
    if not isinstance(envelope, dict) or envelope.get("schema") != SCHEMA:
        raise CheckpointError("Unrecognized checkpoint schema")
    if type(envelope.get("schema_version")) is not int or envelope["schema_version"] != SCHEMA_VERSION:
        raise CheckpointError(
            f"Unsupported checkpoint schema version: {envelope.get('schema_version')}"
        )
    if expected_kind is not None and envelope.get("kind") != expected_kind:
        raise CheckpointError(
            f"Expected checkpoint kind {expected_kind!r}, got {envelope.get('kind')!r}"
        )
    descriptions = envelope.get("arrays")
    if not isinstance(descriptions, dict) or set(descriptions) != names - {ENVELOPE}:
        raise CheckpointError("Checkpoint payload inventory does not match its envelope")
    for name, expected in descriptions.items():
        actual = _array_description(archive[name])
        if actual != expected:
            raise CheckpointError(f"Checkpoint integrity check failed for array {name!r}")
    payload_format = _payload_format(archive, names)
    if payload_format is not None and envelope.get("kind") != payload_format:
        raise CheckpointError("Checkpoint payload format does not match its envelope kind")
    return {**envelope, "integrity": "verified"}


class _Payload:
    """NpzFile-compatible view which hides only the reserved envelope."""

    def __init__(self, archive, envelope):
        self._archive = archive
        self.envelope = envelope
        self.files = [name for name in archive.files if name != ENVELOPE]

    def __getitem__(self, key):
        if key not in self.files:
            raise KeyError(key)
        return self._archive[key]

    def __contains__(self, key):
        return key in self.files

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def keys(self):
        return self.files.copy()

    def get(self, key, default=None):
        return self[key] if key in self else default

    def items(self):
        return ((key, self[key]) for key in self.files)


@contextmanager
def load_checkpoint(path, *, expected_kind=None, require_integrity=False):
    """Validate before yielding a read-only, no-pickle payload view."""
    archive = None
    try:
        archive = np.load(path, allow_pickle=False)
        if not hasattr(archive, "files"):
            raise CheckpointError("A checkpoint must be an NPZ archive")
        envelope = _validate(
            archive, expected_kind=expected_kind, require_integrity=require_integrity
        )
    except CheckpointError:
        if archive is not None and hasattr(archive, "close"):
            archive.close()
        raise
    except Exception as error:
        if archive is not None and hasattr(archive, "close"):
            archive.close()
        raise CheckpointError(f"Cannot read checkpoint {path}: {error}") from error
    try:
        yield _Payload(archive, envelope)
    finally:
        archive.close()


def inspect_checkpoint(path, *, require_integrity=False):
    """Return schema and array shapes, without printing coordinates or RNG data."""
    with load_checkpoint(path, require_integrity=require_integrity) as data:
        return {
            "path": str(Path(path).resolve()),
            "schema": data.envelope["schema"],
            "schema_version": data.envelope["schema_version"],
            "kind": data.envelope.get("kind"),
            "integrity": data.envelope["integrity"],
            "arrays": {
                name: {"shape": list(data[name].shape), "dtype": data[name].dtype.str}
                for name in data.files
            },
        }
