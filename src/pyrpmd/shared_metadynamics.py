"""Distributed multiple-walker metadynamics bias histories.

The serial implementation is retained verbatim in
:mod:`pyrpmd._shared_metadynamics_serial`.  ``SharedMetadynamicsBias`` adds a
thin MPI allgather operation whose batch is committed by the unchanged serial
algorithm.  ``AsynchronousMetadynamicsBias`` consumes immutable records from
per-walker logs and rebuilds a deterministic global hill order after every
merge.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import json

import numpy as np

from ._shared_metadynamics_serial import SharedMetadynamicsBias as _SerialSharedMetadynamicsBias
from .bias import MetadynamicsBias
from .path_integral import _integer
from .walker_exchange import allgather_centers, validate_record


class SharedMetadynamicsBias(_SerialSharedMetadynamicsBias):
    """Existing serial shared bias plus a rank-ordered MPI deposition."""

    def deposit_mpi(self, values, *, step, time, comm=None):
        centers = allgather_centers(values, walkers=self.walkers, comm=comm)
        return self.deposit_walkers(centers, step=step, time=time)


class AsynchronousMetadynamicsBias(MetadynamicsBias):
    """Eventually consistent metadynamics history from immutable hill records.

    The hill height is fixed by the writer against the history visible to that
    walker immediately before publication.  Merging never re-tempers a remote
    hill.  All readers sort the same committed event set by
    ``(step, time, walker, sequence)`` before rebuilding numeric buffers, so
    equal event sets produce equal force summation order.
    """

    def __init__(self, *args, walkers, records=None, **kwargs):
        self.walkers = _integer(walkers, "walkers", 2)
        self._records: dict[tuple[int, int], dict] = {}
        super().__init__(*args, **kwargs)
        if records is not None:
            self.merge_records(records)

    @property
    def records(self):
        """Detached canonical record envelopes suitable for JSON checkpoints."""
        return json.loads(json.dumps(self._ordered_records(), allow_nan=False))

    def prepare_local_payload(self, values) -> dict:
        """Prepare, but do not commit, one locally tempered hill payload."""
        center = np.asarray(values, dtype=np.float64)
        if center.shape != (self.cv.dimension,) or not np.isfinite(center).all():
            raise ValueError("one asynchronous deposition needs one finite [D] center")
        before = float(self.physical_energy(center))
        height = self.height
        if self.mode == "well_tempered":
            height *= np.exp(-self.beta * before / (self.bias_factor - 1.0))
        if not np.isfinite(height):
            raise ValueError("asynchronous metadynamics height is not representable")
        return {"center": center.tolist(), "height": float(height),
                "bias_before": before}

    def merge_records(self, records: Iterable[Mapping]) -> int:
        """Transactionally merge complete records and return the number added."""
        candidate = dict(self._records)
        changed = 0
        for raw in records:
            record = validate_record(raw, method="metadynamics")
            event = record["event"]
            if event["walker"] >= self.walkers:
                raise ValueError("metadynamics record walker is outside this ensemble")
            payload = event["payload"]
            if not isinstance(payload, Mapping) or set(payload) != {
                    "center", "height", "bias_before"}:
                raise ValueError("invalid asynchronous metadynamics payload fields")
            center = np.asarray(payload["center"])
            if (center.dtype.kind not in "iuf" or center.shape != (self.cv.dimension,)
                    or not np.isfinite(center).all()):
                raise ValueError("invalid asynchronous metadynamics center")
            for name in ("height", "bias_before"):
                value = float(payload[name])
                if not np.isfinite(value) or (name == "height" and value < 0):
                    raise ValueError("invalid asynchronous metadynamics payload")
            identity = (event["walker"], event["sequence"])
            previous = candidate.get(identity)
            if previous is not None:
                if previous != record:
                    raise ValueError("conflicting duplicate metadynamics event")
                continue
            candidate[identity] = record
            changed += 1
        if not changed:
            return 0
        self._validate_candidate(candidate)
        previous = self._records
        self._records = candidate
        try:
            self._rebuild()
        except BaseException:
            self._records = previous
            self._rebuild()
            raise
        return changed

    def _validate_candidate(self, candidate):
        for walker in range(self.walkers):
            events = sorted(
                (record["event"] for (owner, _), record in candidate.items()
                 if owner == walker), key=lambda event: event["sequence"])
            if [event["sequence"] for event in events] != list(range(len(events))):
                raise ValueError("metadynamics walker sequence has a gap")
            for previous, current in zip(events, events[1:]):
                if (current["step"] <= previous["step"]
                        or current["time"] <= previous["time"]):
                    raise ValueError("metadynamics walker clocks must strictly increase")

    def _ordered_records(self):
        return sorted(self._records.values(), key=lambda record: (
            record["event"]["step"], record["event"]["time"],
            record["event"]["walker"], record["event"]["sequence"]))

    def _rebuild(self):
        self._hills = []
        self._centers = np.empty((16, self.cv.dimension))
        self._heights = np.empty(16)
        for record in self._ordered_records():
            event, payload = record["event"], record["event"]["payload"]
            MetadynamicsBias._append(self, {
                "center": payload["center"], "height": payload["height"],
                "bias_before": payload["bias_before"],
                "step": event["step"], "time": event["time"],
                "walker": event["walker"],
            })


__all__ = ["AsynchronousMetadynamicsBias", "SharedMetadynamicsBias"]
