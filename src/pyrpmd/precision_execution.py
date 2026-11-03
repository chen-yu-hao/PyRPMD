"""Durable admission accounting for the serial precision rate executor.

One cost unit is one configuration admitted to ``PES.evaluate([B,N,3])``.
Admissions are fsynced *before* calling the provider. A crash between those two
operations conservatively retains the charge; checkpoint replay never refunds
work. This is a bound on admitted PES frames, not wall time or provider-internal
electronic iterations. Ordinary workflows do not install this meter.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import time

from .io import write_json
from .precision_ledger import (
    PrecisionBudgetError, PrecisionLedgerError, _digest, _exclusive_file_lock,
    _json_value,
)


EXECUTION_FORMAT = "PyRPMD.precision.execution.v1"


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_object)
    _json_value(value, str(path))
    if type(value) is not dict:
        raise PrecisionLedgerError(f"Expected a JSON object: {path}")
    return value


def read_sealed(path):
    value = read_json(path)
    checksum = value.pop("integrity_sha256", None)
    if checksum != _digest(value):
        raise PrecisionLedgerError(f"Precision file checksum differs: {path}")
    return value


def write_sealed(path, value):
    _json_value(value, "precision execution data")
    payload = {**value, "integrity_sha256": _digest(value)}
    parent = Path(path).parent
    created = []
    current = parent
    while not current.exists():
        created.append(current)
        current = current.parent
    # Windows may briefly deny replacement while a concurrent status reader
    # has the old file open. Retry that sharing conflict, never other I/O errors.
    for attempt in range(11):
        try:
            write_json(path, payload, sort_keys=True)
            if os.name != "nt":
                # Persist the rename and any newly created journal directories
                # before the admission can authorize a PES call.
                for directory in (parent, *(item.parent for item in created)):
                    fd = os.open(directory, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 10:
                raise
            time.sleep(.01)


def inside(directory, *parts):
    """Resolve only controller-owned paths, rejecting redirected state files."""
    root = Path(directory).resolve()
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root):
        raise PrecisionLedgerError("Precision state must remain inside its campaign directory")
    current = path
    while current != root:
        if current.is_symlink():
            raise PrecisionLedgerError(f"Precision state cannot be a symlink: {current}")
        current = current.parent
    return path


@contextmanager
def execution_lock(directory):
    """Serialize launches separately from the ledger's short transactions.

    Status readers never acquire this lock. Process death releases it on both
    POSIX and Windows; reusing an execution ID alone is not a launch lock.
    """
    path = inside(directory, "precision", "execution.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
                 | getattr(os, "O_BINARY", 0), 0o600)
    try:
        with _exclusive_file_lock(fd):
            yield
    finally:
        os.close(fd)


class ForceBudgetExceeded(PrecisionBudgetError):
    """The next complete PES batch cannot fit its immutable execution cap."""


class JobMeter:
    """A single job's write-ahead charges and immutable completion receipt.

    The caller must hold ``execution_lock`` for the entire lifetime. The
    independently readable JSON file is also the recovery journal.
    """

    def __init__(self, path, *, identity, max_cost, available_cost=None):
        self.path = Path(path)
        self.identity = deepcopy(identity)
        if type(max_cost) is not int or max_cost < 1:
            raise ValueError("job_max_cost must be a positive integer")
        if available_cost is not None and (type(available_cost) is not int or available_cost < 0):
            raise ValueError("available execution cost must be a nonnegative integer")
        self.limit = max_cost if available_cost is None else min(max_cost, available_cost)
        if type(self.limit) is not int or self.limit < 0:
            raise ValueError("available execution cost must be a nonnegative integer")
        if self.path.exists():
            self.data = read_sealed(self.path)
            if (self.data.get("format") != EXECUTION_FORMAT
                    or self.data.get("identity") != self.identity
                    or self.data.get("max_cost") != max_cost):
                raise PrecisionLedgerError("Job execution identity or budget differs")
            for key in ("charged_cost", "evaluation_batches"):
                if type(self.data.get(key)) is not int or self.data[key] < 0:
                    raise PrecisionLedgerError("Invalid saved PES admission count")
            if self.data["charged_cost"] > self.limit:
                raise PrecisionLedgerError("Saved execution exceeds its authorized budget")
            if self.data.get("state") not in ("running", "completed", "failed"):
                raise PrecisionLedgerError("Invalid saved job execution state")
            if self.data["state"] == "completed" and type(self.data.get("record")) is not dict:
                raise PrecisionLedgerError("Completed job is missing its immutable record")
        else:
            self.data = {
                "format": EXECUTION_FORMAT, "identity": self.identity,
                "max_cost": max_cost, "state": "running", "charged_cost": 0,
                "evaluation_batches": 0, "blocked_request": None,
                "failure": None, "record": None,
            }
            self._write(self.data)

    def _write(self, value):
        write_sealed(self.path, value)
        self.data = deepcopy(value)

    def charge(self, frames):
        if type(frames) is not int or frames < 1:
            raise ValueError("PES admission requires a positive frame count")
        if self.data["state"] != "running":
            raise PrecisionLedgerError("Cannot evaluate a terminal precision job")
        used = self.data["charged_cost"]
        if used + frames > self.limit:
            self._write({**self.data, "blocked_request": {
                "frames": frames, "charged_cost": used, "max_cost": self.limit,
            }})
            raise ForceBudgetExceeded(
                f"PES batch of {frames} frames exceeds precision job budget "
                f"({used}/{self.limit} frames already charged)"
            )
        # No provider call can occur unless publication of its charge succeeds.
        self._write({**self.data, "charged_cost": used + frames,
                     "evaluation_batches": self.data["evaluation_batches"] + 1})

    def complete(self, record):
        if type(record) is not dict:
            raise PrecisionLedgerError("A completed job record must be a dictionary")
        _json_value(record, "completed precision job record")
        if self.data["state"] == "completed":
            if self.data["record"] != record:
                raise PrecisionLedgerError("Conflicting immutable precision job receipt")
            return
        if self.data["state"] != "running" or self.data["blocked_request"] is not None:
            raise PrecisionLedgerError("Cannot complete a failed or budget-blocked job")
        self._write({**self.data, "state": "completed", "record": record})

    def fail(self, status, reason):
        value = {"status": status, "reason": reason}
        if self.data["state"] == "failed":
            if self.data["failure"] != value:
                raise PrecisionLedgerError("Conflicting immutable precision job failure")
            return
        if self.data["state"] != "running":
            raise PrecisionLedgerError("Cannot fail a completed precision job")
        self._write({**self.data, "state": "failed", "failure": value})


class MeteredPotential:
    """Expose only the ordinary PES contract; ownership stays with the lease."""

    def __init__(self, provider, meter):
        self.provider = provider
        self.meter = meter
        self.metadata = dict(getattr(provider, "metadata", {}) or {})
        required = getattr(provider, "required_bond_constraints", None)
        if required is not None:
            self.required_bond_constraints = deepcopy(required)
        # Do not forward native bindings or split-force entry points: those
        # would permit evaluation without passing the admission gate.
        for name in ("evaluate_fast", "evaluate_slow", "split_force"):
            if hasattr(provider, name):
                raise ValueError("Precision force metering does not support split/RPC providers")

    def evaluate(self, x):
        from .potentials.base import coordinates

        q = coordinates(x)
        self.meter.charge(len(q))
        return self.provider.evaluate(q)
