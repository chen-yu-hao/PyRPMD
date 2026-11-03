"""OPES biases, including deterministic shared-kernel multiple walkers.

The historical single-walker implementation is retained verbatim in
:mod:`pyrpmd._opes_serial`.  Shared histories deliberately remain
uncompressed: every committed kernel contributes to the KDE and the exact
normalization is rebuilt in deterministic event order.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import json

import numpy as np
from scipy.special import logsumexp

from ._opes_serial import OPESBias, OPESExploreBias, normalized_weights
from .path_integral import _integer
from .walker_exchange import allgather_centers, make_record, validate_record


class SharedOPESBias(OPESBias):
    """Fixed-bandwidth OPES with one shared, uncompressed kernel history.

    A synchronous batch computes every kernel weight from the same pre-commit
    bias and then commits kernels in walker order.  Asynchronous callers may
    instead publish ``prepare_local_payload`` through ``AsyncWalkerStore`` and
    feed the resulting immutable records to ``merge_records``.
    """

    def __init__(self, cv, sigma, beta, *, walkers, barrier, bias_factor=10.0,
                 mode="metad", state=None):
        self.walkers = _integer(walkers, "walkers", 2)
        super().__init__(cv, sigma, beta, barrier=barrier,
                         bias_factor=bias_factor, mode=mode,
                         adaptive_sigma=False, state=None)
        self.method = "opes_" + mode + "_shared_fixed_bandwidth"
        self._records: dict[tuple[int, int], dict] = {}
        self._next_sequences = np.zeros(self.walkers, dtype=np.int64)
        if state is not None:
            self._load_shared_state(state)

    @property
    def records(self):
        return json.loads(json.dumps(self._ordered_records(), allow_nan=False))

    def prepare_local_payload(self, values) -> dict:
        center = self._wrapped_center(values)
        before = float(self.physical_energy(center))
        log_weight = 0.0 if self.mode == "explore" else self.beta * before
        if not np.isfinite(log_weight):
            raise ValueError("shared OPES kernel weight is not representable")
        return {"center": center.tolist(), "log_weight": float(log_weight),
                "bias_before": before}

    def update_walkers(self, values, *, step, time):
        centers = np.asarray(values, dtype=np.float64)
        if (centers.shape != (self.walkers, self.cv.dimension)
                or not np.isfinite(centers).all()):
            raise ValueError("shared OPES update needs finite [walkers,dimension] centers")
        wrapped = np.asarray([self._wrapped_center(center) for center in centers])
        before = np.asarray(self.physical_energy(wrapped), dtype=np.float64)
        log_weights = (np.zeros(self.walkers) if self.mode == "explore"
                       else self.beta * before)
        if not np.isfinite(log_weights).all():
            raise ValueError("shared OPES kernel weights are not representable")
        records = []
        for walker in range(self.walkers):
            records.append(make_record(
                "opes", walker, int(self._next_sequences[walker]),
                step=step, time=time,
                payload={"center": wrapped[walker].tolist(),
                         "log_weight": float(log_weights[walker]),
                         "bias_before": float(before[walker])}))
        self.merge_records(records)
        return [{**record["event"], "log_z": self.log_z,
                 "weight_effective_sample_size": self.effective_sample_size}
                for record in records]

    def update_mpi(self, values, *, step, time, comm=None):
        centers = allgather_centers(values, walkers=self.walkers, comm=comm)
        return self.update_walkers(centers, step=step, time=time)

    def merge_records(self, records: Iterable[Mapping]) -> int:
        candidate = dict(self._records)
        changed = 0
        for raw in records:
            record = validate_record(raw, method="opes")
            event = record["event"]
            walker = event["walker"]
            if walker >= self.walkers:
                raise ValueError("OPES record walker is outside this ensemble")
            payload = event["payload"]
            if not isinstance(payload, Mapping) or set(payload) != {
                    "center", "log_weight", "bias_before"}:
                raise ValueError("invalid shared OPES payload fields")
            center = np.asarray(payload["center"])
            if (center.dtype.kind not in "iuf" or center.shape != (self.cv.dimension,)
                    or not np.isfinite(center).all()):
                raise ValueError("invalid shared OPES center")
            for name in ("log_weight", "bias_before"):
                if not np.isfinite(float(payload[name])):
                    raise ValueError("invalid shared OPES payload")
            if self.mode == "explore" and float(payload["log_weight"]) != 0.0:
                raise ValueError("shared OPES explore kernels must be unweighted")
            identity = (walker, event["sequence"])
            previous = candidate.get(identity)
            if previous is not None:
                if previous != record:
                    raise ValueError("conflicting duplicate OPES event")
                continue
            candidate[identity] = record
            changed += 1
        if not changed:
            return 0
        self._validate_candidate(candidate)
        previous = self._records
        previous_sequences = self._next_sequences.copy()
        self._records = candidate
        for walker in range(self.walkers):
            self._next_sequences[walker] = sum(
                owner == walker for owner, _ in candidate)
        try:
            self._rebuild()
        except BaseException:
            self._records = previous
            self._next_sequences = previous_sequences
            self._rebuild()
            raise
        return changed

    def _validate_candidate(self, candidate):
        for walker in range(self.walkers):
            events = sorted(
                (record["event"] for (owner, _), record in candidate.items()
                 if owner == walker), key=lambda event: event["sequence"])
            if [event["sequence"] for event in events] != list(range(len(events))):
                raise ValueError("OPES walker sequence has a gap")
            for previous, current in zip(events, events[1:]):
                if (current["step"] <= previous["step"]
                        or current["time"] <= previous["time"]):
                    raise ValueError("OPES walker clocks must strictly increase")

    def state_dict(self):
        return {"schema_version": 1, "method": self.method,
                "sigma": self.sigma.tolist(), "beta": self.beta,
                "barrier": self.barrier, "bias_factor": self.bias_factor,
                "mode": self.mode, "periods": list(self.periods),
                "walkers": self.walkers, "initial_bias": self._initial_bias,
                "records": self.records, "log_z": self.log_z,
                "compression": "none"}

    def _load_shared_state(self, state):
        if not isinstance(state, Mapping) or set(state) != set(self.state_dict()):
            raise ValueError("invalid shared OPES state fields")
        current = self.state_dict()
        for key in ("schema_version", "method", "sigma", "beta", "barrier",
                    "bias_factor", "mode", "periods", "walkers",
                    "initial_bias", "compression"):
            if state[key] != current[key]:
                raise ValueError("shared OPES state parameters differ: " + key)
        records = state["records"]
        if not isinstance(records, list):
            raise ValueError("shared OPES records must be a list")
        self.merge_records(records)
        expected = float(state["log_z"])
        if not np.isfinite(expected) or expected != self.log_z:
            raise ValueError("shared OPES state normalization is inconsistent")

    def _wrapped_center(self, values):
        center = self._values(values).copy()
        if center.shape != (self.cv.dimension,):
            raise ValueError("one shared OPES update needs one [D] center")
        for index, period in enumerate(self.periods):
            if period is not None:
                center[index] = (center[index] + period / 2) % period - period / 2
        return center

    def _ordered_records(self):
        return sorted(self._records.values(), key=lambda record: (
            record["event"]["step"], record["event"]["time"],
            record["event"]["walker"], record["event"]["sequence"]))

    def _rebuild(self):
        ordered = self._ordered_records()
        if not ordered:
            self._centers = np.empty((0, self.cv.dimension))
            self._log_weights = np.empty(0)
            self._log_center_sums = np.empty(0)
            self._steps = np.empty(0, dtype=np.int64)
            self._times = np.empty(0)
            self.log_z = 0.0
            return
        self._centers = np.asarray(
            [record["event"]["payload"]["center"] for record in ordered],
            dtype=np.float64)
        self._log_weights = np.asarray(
            [record["event"]["payload"]["log_weight"] for record in ordered],
            dtype=np.float64)
        self._steps = np.asarray(
            [record["event"]["step"] for record in ordered], dtype=np.int64)
        self._times = np.asarray(
            [record["event"]["time"] for record in ordered], dtype=np.float64)
        pair_logs, _ = self._kernel(self._centers, self._centers)
        self._log_center_sums = logsumexp(pair_logs, axis=0)
        self.log_z = float(
            logsumexp(self._log_weights + self._log_center_sums)
            - logsumexp(self._log_weights) - np.log(len(self._log_weights)))
        if not np.isfinite(self.log_z):
            raise ValueError("shared OPES normalization is not representable")


# Preserve the historical public import path despite the private source copy.
OPESBias.__module__ = __name__
OPESExploreBias.__module__ = __name__
normalized_weights.__module__ = __name__

__all__ = [
    "OPESBias", "OPESExploreBias", "SharedOPESBias", "normalized_weights",
]
