"""Restartable context persistence and lifecycle helpers."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

import numpy as np

from ._types import ExcitedStateFrameContext
from ._util import canonical_json
from ._validation import _from_json_array, _json_array


class _ProviderPersistenceMixin:
    def export_context(self, context) -> dict[str, Any]:
        if not isinstance(context, (tuple, list)) or not context:
            raise ValueError("context must contain at least one excited-state frame")
        frames = []
        for frame in context:
            if not isinstance(frame, ExcitedStateFrameContext) or frame.fingerprint != self.fingerprint:
                raise ValueError("context belongs to a different excited-state provider")
            frames.append(
                {
                    "geometry": frame.geometry.tolist(),
                    "permutation": frame.permutation.tolist(),
                    "phases": _json_array(frame.phases),
                    "oscillator_strengths": _json_array(frame.oscillator_strengths),
                    "transition_dipoles": _json_array(frame.transition_dipoles),
                    "time_derivative_couplings": _json_array(frame.time_derivative_couplings),
                    "diagnostics": copy.deepcopy(dict(frame.diagnostics)),
                    "payload": self.backend.serialize_payload(frame.payload),
                }
            )
        payload = {
            "format": "pyrpmd-excited-state-context-v1",
            "scientific_identity": self.to_dict(),
            "fingerprint": self.fingerprint,
            "frames": frames,
        }
        payload["sha256"] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
        return payload

    def import_context(self, snapshot: Mapping[str, Any]):
        if not isinstance(snapshot, Mapping):
            raise ValueError("context snapshot must be a mapping")
        data = copy.deepcopy(dict(snapshot))
        checksum = data.pop("sha256", None)
        if not isinstance(checksum, str) or hashlib.sha256(canonical_json(data).encode()).hexdigest() != checksum:
            raise ValueError("excited-state context checksum mismatch")
        if data.get("format") != "pyrpmd-excited-state-context-v1":
            raise ValueError("unsupported excited-state context format")
        if data.get("fingerprint") != self.fingerprint or data.get("scientific_identity") != self.to_dict():
            raise ValueError("excited-state context scientific identity mismatch")
        raw_frames = data.get("frames")
        if not isinstance(raw_frames, list) or not raw_frames:
            raise ValueError("excited-state context has no frames")
        frames = []
        for raw in raw_frames:
            if not isinstance(raw, Mapping):
                raise ValueError("invalid excited-state context frame")
            frames.append(
                ExcitedStateFrameContext(
                    self.fingerprint,
                    np.asarray(raw.get("geometry"), dtype=float),
                    self.backend.deserialize_payload(raw.get("payload")),
                    np.asarray(raw.get("permutation")),
                    _from_json_array(raw.get("phases")),
                    _from_json_array(raw.get("oscillator_strengths"), dtype=float),
                    _from_json_array(raw.get("transition_dipoles")),
                    _from_json_array(raw.get("time_derivative_couplings")),
                    raw.get("diagnostics", {}),
                )
            )
        return tuple(frames)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.backend.close()
        finally:
            self._scratch.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
