"""Crash-safe accounting between precision decisions and an external executor.

The ledger does not run dynamics.  Agent F must reconcile a durable ``running``
execution with its own checkpoint/idempotency key after a crash.  A plan has one
execution identity and one immutable terminal receipt; retrying a failed plan
requires a new plan (and a new reservation).  Costs in receipts are the *total*
cost of that execution, not an incremental retry charge.

Real costs are recorded even if they exceed an estimate or the maximum budget.
An overrun is therefore visible and prevents further reservations; it is never
discarded to make accounting appear to fit a budget.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


FORMAT = "PyRPMD.rate.precision-ledger.v1"


class PrecisionLedgerError(ValueError):
    """Conflicting, corrupt, or invalid precision accounting data."""


@contextmanager
def _exclusive_file_lock(fd):
    """Hold an exclusive inter-process lock on an open lock-file descriptor.

    POSIX uses advisory ``flock``. Windows has no ``fcntl``; it locks the
    first byte with ``msvcrt.locking`` and keeps retrying, matching the
    indefinite wait of ``flock(LOCK_EX)``.
    """
    try:
        import fcntl
    except ImportError:
        fcntl = None
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
        return
    try:
        import msvcrt
    except ImportError as exc:  # pragma: no cover - neither POSIX nor Windows
        raise RuntimeError("PrecisionLedger requires fcntl or msvcrt file locks") from exc
    while True:
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            # LK_LOCK itself retries for about ten seconds before failing.
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            break
        except OSError as exc:
            if exc.errno not in (errno.EDEADLK, errno.EACCES):
                raise
    try:
        yield
    finally:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


class PrecisionBudgetError(PrecisionLedgerError):
    """A proposed reservation would exceed the available execution budget."""


def _json_value(value, label="value"):
    """Validate strict JSON data without stringifying keys or losing precision."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise PrecisionLedgerError(f"{label} must contain only finite numbers")
        return
    if type(value) is list:
        for item in value:
            _json_value(item, label)
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise PrecisionLedgerError(f"{label} must have string dictionary keys")
        for item in value.values():
            _json_value(item, label)
        return
    raise PrecisionLedgerError(f"{label} must contain only JSON values")


def _dictionary(value, label):
    if type(value) is not dict:
        raise PrecisionLedgerError(f"{label} must be a dictionary")
    _json_value(value, label)
    return deepcopy(value)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _identifier(value, label):
    if type(value) is not str or not value.strip():
        raise PrecisionLedgerError(f"{label} must be a nonempty string")
    return value


def _cost(value, label):
    if type(value) not in (int, float):
        raise PrecisionLedgerError(f"{label} must be a finite nonnegative number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise PrecisionLedgerError(f"{label} is outside the finite range") from exc
    if not math.isfinite(result) or result < 0:
        raise PrecisionLedgerError(f"{label} must be a finite nonnegative number")
    return result


def _result_ids(values):
    if type(values) is not list:
        raise PrecisionLedgerError("result_ids must be a list")
    for value in values:
        _identifier(value, "result ID")
    if len(set(values)) != len(values):
        raise PrecisionLedgerError("A receipt cannot count the same result ID twice")
    return list(values)


class PrecisionLedger:
    """Persist a frozen specification, reservations, receipts, and decisions.

    ``directory`` is a run/controller directory; files live under ``precision``.
    ``specification`` must include the caller's frozen scientific/check design.
    Cost units are caller-defined, but must be identical in every plan/receipt.
    The implementation requires an exclusive file lock (POSIX ``flock`` or
    Windows ``msvcrt.locking``) and atomic rename on the local filesystem.
    Independent writers must all use this API.
    """

    def __init__(self, directory, *, specification: dict, max_cost: float):
        self.directory = Path(directory).resolve()
        self.path = self.directory / "precision" / "ledger.json"
        self.lock_path = self.path.with_name("ledger.lock")
        self.specification = _dictionary(specification, "specification")
        self.specification_id = _digest(self.specification)
        self.max_cost = _cost(max_cost, "max_cost")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.parent.resolve().is_relative_to(self.directory):
            raise PrecisionLedgerError("Precision ledger must remain inside its run directory")
        with self._locked():
            if self.path.is_symlink():
                raise PrecisionLedgerError("Precision ledger cannot be a symlink")
            if self.path.exists():
                self._read()
            else:
                self._write({
                    "format": FORMAT,
                    "specification": deepcopy(self.specification),
                    "specification_id": self.specification_id,
                    "max_cost": self.max_cost,
                    "revision": 0,
                    "plans": {},
                    "results": {},
                    "checks": {},
                    "stop": None,
                    "events": [],
                })

    @contextmanager
    def _locked(self):
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        fd = os.open(self.lock_path, flags, 0o600)
        try:
            with _exclusive_file_lock(fd):
                yield
        finally:
            os.close(fd)

    def _read(self):
        if self.path.is_symlink():
            raise PrecisionLedgerError("Precision ledger cannot be a symlink")
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if type(saved) is not dict:
                raise ValueError("ledger must be an object")
            checksum = saved.pop("integrity_sha256", None)
            _json_value(saved, "ledger")
            if checksum != _digest(saved):
                raise ValueError("ledger checksum differs")
            required = {"format", "specification", "specification_id", "max_cost",
                        "revision", "plans", "results", "checks", "stop", "events"}
            if set(saved) != required or saved["format"] != FORMAT:
                raise ValueError("unknown ledger format")
            if (_digest(saved["specification"]) != saved["specification_id"]
                    or saved["specification_id"] != self.specification_id):
                raise ValueError("frozen precision specification differs")
            if saved["max_cost"] != self.max_cost:
                raise ValueError("frozen precision budget differs")
            if any(type(saved[key]) is not dict for key in ("plans", "results", "checks")):
                raise ValueError("invalid ledger collections")
            if type(saved["revision"]) is not int or saved["revision"] < 0:
                raise ValueError("invalid ledger revision")
            if type(saved["events"]) is not list or len(saved["events"]) != saved["revision"]:
                raise ValueError("invalid ledger event history")
            self._budget(saved)
            return saved
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            raise PrecisionLedgerError(f"Cannot load precision ledger: {exc}") from exc

    def _write(self, state):
        saved = deepcopy(state)
        saved["integrity_sha256"] = _digest(state)
        data = json.dumps(saved, indent=2, sort_keys=True, allow_nan=False) + "\n"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=self.path.parent,
                                             prefix=".ledger-", suffix=".tmp", delete=False) as stream:
                temporary = stream.name
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
            # Windows cannot open a directory as a file to fsync it; os.replace
            # there is still an atomic rename of the fsynced temporary file.
            if os.name != "nt":
                fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    @staticmethod
    def _budget(state):
        spent = []
        reserved = []
        pending = []
        for plan_id, entry in state["plans"].items():
            _identifier(plan_id, "saved plan_id")
            if type(entry) is not dict or entry.get("state") not in {
                    "planned", "running", "completed", "failed"}:
                raise PrecisionLedgerError("Invalid saved plan state")
            if entry["plan"]["plan_id"] != plan_id:
                raise PrecisionLedgerError("Saved plan ID differs from its key")
            if entry["state"] in ("planned", "running"):
                reserved.append(_cost(entry["plan"]["estimated_cost"], "saved estimate"))
                if entry["state"] == "running":
                    pending.append(plan_id)
            else:
                spent.append(_cost(entry["receipt"]["actual_cost"], "saved actual cost"))
        try:
            spent_cost, reserved_cost = math.fsum(spent), math.fsum(reserved)
        except OverflowError as exc:
            raise PrecisionLedgerError("Accumulated budget is outside the finite range") from exc
        if not math.isfinite(spent_cost + reserved_cost):
            raise PrecisionLedgerError("Accumulated budget is outside the finite range")
        remaining = state["max_cost"] - spent_cost - reserved_cost
        return {
            "max_cost": state["max_cost"],
            "spent_cost": spent_cost,
            "reserved_cost": reserved_cost,
            "available_cost": max(0.0, remaining),
            "budget_exhausted": remaining <= 0,
            "budget_exceeded": spent_cost > state["max_cost"],
            "pending_reconciliation": sorted(pending),
        }

    @staticmethod
    def _event(state, kind, **details):
        state["revision"] += 1
        state["events"].append({"revision": state["revision"], "kind": kind, **details})

    @staticmethod
    def _entry(state, plan_id):
        _identifier(plan_id, "plan_id")
        try:
            return state["plans"][plan_id]
        except KeyError as exc:
            raise PrecisionLedgerError(f"Unknown precision plan: {plan_id}") from exc

    def snapshot(self):
        """Return independent JSON data, including derived truthful budget state."""
        with self._locked():
            state = self._read()
            state["budget"] = self._budget(state)
            return state

    def register_plan(self, plan: dict):
        """Durably reserve a plan once, before handing it to an executor."""
        plan = _dictionary(plan, "plan")
        for key in ("plan_id", "scientific_identity", "epoch_id"):
            _identifier(plan.get(key), key)
        if type(plan.get("actions")) is not list:
            raise PrecisionLedgerError("plan actions must be a list")
        plan["estimated_cost"] = _cost(plan.get("estimated_cost"), "estimated_cost")
        plan_id = plan["plan_id"]
        with self._locked():
            state = self._read()
            previous = state["plans"].get(plan_id)
            if previous is not None:
                if previous["plan"] != plan:
                    raise PrecisionLedgerError(f"Plan ID reused with different content: {plan_id}")
                return deepcopy(previous)
            if state["stop"] is not None:
                raise PrecisionLedgerError("Cannot register new work after a recorded stop")
            budget = self._budget(state)
            if budget["budget_exceeded"] or plan["estimated_cost"] > budget["available_cost"]:
                raise PrecisionBudgetError("Precision plan exceeds the unreserved budget")
            entry = {"plan": plan, "state": "planned", "execution_id": None, "receipt": None}
            state["plans"][plan_id] = entry
            self._event(state, "plan_registered", plan_id=plan_id)
            self._write(state)
            return deepcopy(entry)

    def start(self, plan_id, *, execution_id: str):
        """Record F's idempotency key; replays never launch another execution."""
        _identifier(execution_id, "execution_id")
        with self._locked():
            state = self._read()
            entry = self._entry(state, plan_id)
            if entry["execution_id"] is not None:
                if entry["execution_id"] != execution_id:
                    raise PrecisionLedgerError("Execution identity differs; reconcile the original execution")
                return deepcopy(entry)
            if state["stop"] is not None:
                raise PrecisionLedgerError("Cannot start new work after a recorded stop")
            budget = self._budget(state)
            if budget["spent_cost"] + budget["reserved_cost"] > budget["max_cost"]:
                raise PrecisionBudgetError("Actual costs left insufficient budget for reserved work")
            if any(other["execution_id"] == execution_id for other in state["plans"].values()):
                raise PrecisionLedgerError("Execution ID is already attached to another plan")
            entry["execution_id"] = execution_id
            entry["state"] = "running"
            self._event(state, "execution_started", plan_id=plan_id, execution_id=execution_id)
            self._write(state)
            return deepcopy(entry)

    def _finish(self, plan_id, receipt):
        with self._locked():
            state = self._read()
            entry = self._entry(state, plan_id)
            if entry["execution_id"] != receipt["execution_id"]:
                raise PrecisionLedgerError("Receipt execution ID differs from the recorded execution")
            if entry["receipt"] is not None:
                if entry["receipt"] != receipt:
                    raise PrecisionLedgerError("Conflicting replay of an immutable execution receipt")
                return deepcopy(entry)
            if entry["state"] != "running":
                raise PrecisionLedgerError("Record start before recording an execution receipt")
            for result_id in receipt["result_ids"]:
                if result_id in state["results"]:
                    raise PrecisionLedgerError(f"Result ID is already accounted for: {result_id}")
            for result_id in receipt["result_ids"]:
                state["results"][result_id] = {
                    "plan_id": plan_id,
                    "execution_id": receipt["execution_id"],
                    "scientific_identity": entry["plan"]["scientific_identity"],
                    "epoch_id": entry["plan"]["epoch_id"],
                    "execution_state": receipt["state"],
                }
            entry["receipt"] = receipt
            entry["state"] = receipt["state"]
            self._budget(state)  # Validate accumulated finite accounting before publication.
            self._event(state, "execution_" + receipt["state"], plan_id=plan_id,
                        execution_id=receipt["execution_id"])
            self._write(state)
            return deepcopy(entry)

    def complete(self, plan_id, *, execution_id: str, actual_cost: float,
                 result_ids: list[str], result: dict):
        """Commit a complete receipt once; actual cost may expose a budget overrun."""
        receipt = {
            "state": "completed",
            "execution_id": _identifier(execution_id, "execution_id"),
            "actual_cost": _cost(actual_cost, "actual_cost"),
            "result_ids": _result_ids(result_ids),
            "result": _dictionary(result, "result"),
        }
        return self._finish(plan_id, receipt)

    def fail(self, plan_id, *, execution_id: str, actual_cost: float, reason: str,
             result_ids: list[str] | None = None, result: dict | None = None):
        """Record a terminal failed/partial execution and release its reservation.

        Cost must first be reconciled with F.  Unknown cost leaves the execution
        running/reserved.  Partial result IDs are claimed for deduplication but
        explicitly carry ``execution_state='failed'``; this ledger does not
        promote them into scientific evidence.  A retry needs a new plan ID.
        """
        receipt = {
            "state": "failed",
            "execution_id": _identifier(execution_id, "execution_id"),
            "actual_cost": _cost(actual_cost, "actual_cost"),
            "result_ids": _result_ids([] if result_ids is None else result_ids),
            "result": _dictionary({} if result is None else result, "result"),
            "reason": _identifier(reason, "reason"),
        }
        return self._finish(plan_id, receipt)

    def record_check(self, check_id, payload: dict):
        """Persist one immutable check; identical interrupt/recovery replays are no-ops."""
        _identifier(check_id, "check_id")
        payload = _dictionary(payload, "check payload")
        with self._locked():
            state = self._read()
            if check_id in state["checks"]:
                if state["checks"][check_id] != payload:
                    raise PrecisionLedgerError("Check ID reused with different evidence or decision")
                return deepcopy(payload)
            if state["stop"] is not None:
                raise PrecisionLedgerError("Cannot add a new check after a recorded stop")
            state["checks"][check_id] = payload
            self._event(state, "check_recorded", check_id=check_id)
            self._write(state)
            return deepcopy(payload)

    def stop(self, status, reasons: list[str], conclusions: dict):
        """Record a terminal reason without changing independent scientific flags.

        Existing running work can still be reconciled using complete/fail after
        a stop.  The stop record itself is immutable and contains exactly the
        caller's conclusions, including any unmet numerical/precision criteria.
        """
        _identifier(status, "status")
        if type(reasons) is not list:
            raise PrecisionLedgerError("stop reasons must be a list")
        for reason in reasons:
            _identifier(reason, "stop reason")
        record = {"status": status, "reasons": list(reasons),
                  "conclusions": _dictionary(conclusions, "conclusions")}
        with self._locked():
            state = self._read()
            if state["stop"] is not None:
                if state["stop"] != record:
                    raise PrecisionLedgerError("Conflicting replay of an immutable stop record")
                return deepcopy(record)
            state["stop"] = record
            self._event(state, "stopped", status=status)
            self._write(state)
            return deepcopy(record)
