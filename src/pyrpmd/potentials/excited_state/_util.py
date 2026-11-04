"""Small validation and hashing utilities for excited-state providers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def positive_integer(value: object, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def finite_positive(value: object, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    result = float(value)
    if not np.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return result


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("scientific identity must contain finite JSON values") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("text to hash must be a string")
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"identity file does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def immutable_array(value: object, *, dtype=None) -> np.ndarray:
    array = np.array(value, dtype=dtype, order="C", copy=True)
    # A bytes-backed view cannot later be made writable via setflags.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
