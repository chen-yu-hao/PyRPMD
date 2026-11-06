"""Synchronous and asynchronous shared metadynamics histories."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import json

import numpy as np

from .bias import MetadynamicsBias
from .path_integral import _integer
from .walker_exchange import allgather_centers, validate_record


class SharedMetadynamicsBias(MetadynamicsBias):
    """Deposit a complete walker batch against the same pre-deposition bias.

    All walkers see one frozen history for an entire MD step. At a deposition
    barrier, each contributes the configured hill height (WT-scaled using the
    bias BEFORE any hill in this batch). Total deposition grows with walker
    count; choose height/pace accordingly. There is one correlated bias history.
    """

    _shared_clocks = True

    def __init__(self, *args, walkers, **kwargs):
        self.walkers = _integer(walkers, "walkers", 2)
        super().__init__(*args, **kwargs)
        if self.hill_count % self.walkers:
            raise ValueError("shared history must end at a complete walker deposition")

    def deposit(self, *args, **kwargs):
        raise ValueError("shared bias requires deposit_walkers with all walkers")

    def truncate_history(self, count):
        """Rollback only to a complete synchronized deposition barrier."""
        if (isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer))
                or count % self.walkers):
            raise ValueError("shared history rollback must retain complete walker batches")
        super().truncate_history(count)

    def deposit_walkers(self, values, *, step, time):
        centers = np.asarray(values)
        if centers.dtype.kind not in "fiu" or centers.shape != (self.walkers, self.cv.dimension) or not np.isfinite(centers).all():
            raise ValueError("shared deposition needs finite [walkers,dimension] centers")
        before = np.asarray(self.physical_energy(centers), dtype=float)
        heights = np.full(self.walkers, self.height)
        if self.mode == "well_tempered":
            heights *= np.exp(-self.beta * before / (self.bias_factor - 1))
        old_count = self.hill_count
        try:
            for walker in range(self.walkers):
                self._append({"center": centers[walker], "height": heights[walker],
                    "step": step, "time": time, "bias_before": before[walker], "walker": walker})
        except BaseException:
            self.truncate_history(old_count)
            raise
        return self.hills[-self.walkers:]

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
