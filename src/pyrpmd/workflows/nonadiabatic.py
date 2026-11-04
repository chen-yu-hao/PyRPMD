"""Nonadiabatic workflow with versioned NRPMD auxiliary-state sidecars.

The historical trajectory container remains the authoritative atomic checkpoint
and is unchanged, so every pre-T26 checkpoint remains readable.  New NRPMD
checkpoints also publish a version-2 integrity-protected auxiliary sidecar.  A
missing or stale sidecar can result only from interruption after the primary
checkpoint was committed; the complete primary state remains resumable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import _nonadiabatic_legacy as _legacy
from ..checkpoint import CheckpointError, atomic_savez, load_checkpoint
from ..io import jsonable


# Retain the complete established workflow surface, including private helpers
# used by existing tests. Only the two persistence functions are interposed.
for _name in dir(_legacy):
    if not _name.startswith("__") and _name not in {"_save", "_load"}:
        globals()[_name] = getattr(_legacy, _name)


_OPERATOR_KIND = "pyrpmd.nrpmd_auxiliary_state.v2"
_OPERATOR_VERSION = 2
_ORIGINAL_SAVE = _legacy._save
_ORIGINAL_LOAD = _legacy._load


def _operator_path(path):
    path = Path(path)
    return path.with_name(path.stem + ".operator.npz")


def _save(path, config, identity, index, state, rng, history):
    """Write the unchanged primary checkpoint, then its derived v2 sidecar."""

    _ORIGINAL_SAVE(path, config, identity, index, state, rng, history)
    if config["method"] != "nrpmd":
        return
    metadata = {
        "format": _OPERATOR_KIND,
        "version": _OPERATOR_VERSION,
        "physics_fingerprint": identity,
        "trajectory": int(index),
        "method": "nrpmd",
        "step": int(state.step),
        "rng_in_primary_checkpoint": True,
    }
    atomic_savez(
        _operator_path(path), kind=_OPERATOR_KIND,
        metadata=np.asarray(json.dumps(jsonable(metadata), sort_keys=True, allow_nan=False)),
        mapping_q=state.mapping_q, mapping_p=state.mapping_p,
    )


def _load(path, config, identity, index, dynamics, rng):
    """Load the primary checkpoint and validate a same-step v2 sidecar if present."""

    state, history = _ORIGINAL_LOAD(path, config, identity, index, dynamics, rng)
    if config["method"] != "nrpmd":
        return state, history
    operator_path = _operator_path(path)
    if not operator_path.exists():
        # Legacy checkpoint, or an interruption after the authoritative primary
        # atomic rename but before the derived sidecar rename.
        return state, history
    with load_checkpoint(
        operator_path, expected_kind=_OPERATOR_KIND, require_integrity=True
    ) as saved:
        data = {key: saved[key] for key in saved.files}
    if set(data) != {"metadata", "mapping_q", "mapping_p"}:
        raise CheckpointError("NRPMD auxiliary checkpoint payload inventory mismatch")
    raw = data["metadata"]
    if raw.shape != () or raw.dtype.kind != "U":
        raise CheckpointError("invalid NRPMD auxiliary checkpoint metadata")
    metadata = json.loads(str(raw.item()))
    expected_keys = {
        "format", "version", "physics_fingerprint", "trajectory", "method",
        "step", "rng_in_primary_checkpoint",
    }
    if (set(metadata) != expected_keys
            or metadata["format"] != _OPERATOR_KIND
            or metadata["version"] != _OPERATOR_VERSION
            or metadata["physics_fingerprint"] != identity
            or type(metadata["trajectory"]) is not int
            or metadata["trajectory"] != index
            or metadata["method"] != "nrpmd"
            or type(metadata["step"]) is not int
            or metadata["rng_in_primary_checkpoint"] is not True):
        raise CheckpointError("NRPMD auxiliary checkpoint identity mismatch")
    if metadata["step"] < state.step:
        # A complete newer primary file is authoritative. The next normal save
        # regenerates the sidecar without changing trajectory state or RNG.
        return state, history
    if metadata["step"] != state.step:
        raise CheckpointError("NRPMD auxiliary checkpoint is ahead of primary checkpoint")
    for name in ("mapping_q", "mapping_p"):
        value = data[name]
        actual = getattr(state, name)
        if (value.dtype.kind != "f" or value.dtype.itemsize != 8
                or value.shape != actual.shape or not np.isfinite(value).all()
                or not np.array_equal(value, actual)):
            raise CheckpointError(f"NRPMD auxiliary checkpoint mismatch for {name}")
    return state, history


# Functions defined in the legacy module resolve globals in that module. Patch
# only its persistence hooks, leaving all propagation and analysis code intact.
_legacy._save = _save
_legacy._load = _load

__all__ = ["prepare_nonadiabatic", "run_nonadiabatic", "analyze_nonadiabatic"]
