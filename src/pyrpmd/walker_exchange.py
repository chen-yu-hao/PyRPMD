"""Crash-safe exchange primitives for distributed multiple-walker biases.

Each walker is the sole writer of one JSON-lines file.  A record is committed
only when the complete checksum-protected line (including its newline) is
visible.  Readers leave an unterminated tail unconsumed, which makes a crash in
the middle of a write harmless.  Checkpoints persist the byte offsets returned
by :meth:`AsyncWalkerStore.poll` and therefore resume without rescanning or
silently skipping committed records.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

_HEX = frozenset("0123456789abcdef")

import numpy as np

_SCHEMA = 1


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("walker record must be finite JSON data") from error
    return text.encode("utf-8")


def _checked_index(value: Any, name: str, minimum: int = 0) -> int:
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or int(value) < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def make_record(method: str, walker: int, sequence: int, *, step: int,
                time: float, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical checksum envelope for one immutable bias event."""
    if method not in ("metadynamics", "opes"):
        raise ValueError("walker record method must be metadynamics or opes")
    walker = _checked_index(walker, "walker")
    sequence = _checked_index(sequence, "sequence")
    step = _checked_index(step, "step", 1)
    time = float(time)
    if not np.isfinite(time) or time <= 0:
        raise ValueError("walker record time must be positive and finite")
    if not isinstance(payload, Mapping):
        raise ValueError("walker record payload must be a mapping")
    event = {"schema": _SCHEMA, "method": method, "walker": walker,
             "sequence": sequence, "step": step, "time": time,
             "payload": dict(payload)}
    digest = hashlib.sha256(_canonical(event)).hexdigest()
    return {"event": event, "sha256": digest}


def validate_record(record: Any, *, method: str | None = None,
                    walker: int | None = None) -> dict[str, Any]:
    """Validate and detach one checksum envelope."""
    if not isinstance(record, Mapping) or set(record) != {"event", "sha256"}:
        raise ValueError("invalid walker record envelope")
    event, digest = record["event"], record["sha256"]
    if not isinstance(event, Mapping) or set(event) != {
            "schema", "method", "walker", "sequence", "step", "time", "payload"}:
        raise ValueError("invalid walker event fields")
    expected = hashlib.sha256(_canonical(event)).hexdigest()
    if not isinstance(digest, str) or digest != expected:
        raise ValueError("walker record checksum mismatch")
    if event["schema"] != _SCHEMA:
        raise ValueError("unsupported walker record schema")
    checked = make_record(str(event["method"]), event["walker"], event["sequence"],
                          step=event["step"], time=event["time"],
                          payload=event["payload"])
    if checked["sha256"] != digest:
        raise ValueError("walker record is not canonical")
    detached = json.loads(_canonical(checked).decode("utf-8"))
    if method is not None and detached["event"]["method"] != method:
        raise ValueError("walker record method differs from this store")
    if walker is not None and detached["event"]["walker"] != walker:
        raise ValueError("walker record appears in the wrong writer log")
    return detached


def encode_record(record: Mapping[str, Any]) -> bytes:
    """Encode one validated record as an indivisible JSON-lines payload."""
    checked = validate_record(record)
    return _canonical(checked) + b"\n"


def read_committed_records(path: str | os.PathLike[str], offset: int = 0,
                           *, method: str | None = None,
                           walker: int | None = None) -> tuple[list[dict[str, Any]], int]:
    """Read complete records starting at *offset* and ignore a truncated tail.

    A malformed *complete* line is corruption and fails explicitly.  An
    unterminated final line is left for the next poll and does not advance the
    returned offset.
    """
    offset = _checked_index(offset, "walker log offset")
    file = Path(path)
    if not file.exists():
        if offset:
            raise ValueError("walker checkpoint offset refers to a missing log")
        return [], 0
    size = file.stat().st_size
    if offset > size:
        raise ValueError("walker checkpoint offset is beyond the end of its log")
    records: list[dict[str, Any]] = []
    committed = offset
    with file.open("rb") as stream:
        stream.seek(offset)
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                break
            try:
                decoded = json.loads(line[:-1].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"malformed complete walker record at byte {start}") from error
            records.append(validate_record(decoded, method=method, walker=walker))
            committed = stream.tell()
    return records, committed


def ensure_manifest(directory: str | os.PathLike[str], identity: Mapping[str, Any]) -> Path:
    """Atomically create or verify the exchange directory's identity."""
    if not isinstance(identity, Mapping):
        raise ValueError("walker exchange identity must be a mapping")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": 1, "identity": dict(identity)}
    envelope = {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    encoded = _canonical(envelope) + b"\n"
    path = directory / "manifest.json"
    if not path.exists() and any(directory.glob("walker-*.jsonl")):
        raise ValueError(
            "walker exchange logs exist without a manifest; use a fresh directory")
    descriptor, temporary = tempfile.mkstemp(
        prefix="manifest.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A hard link publishes only the already-complete inode and fails
            # atomically when another walker won the creation race.
            os.link(temporary, path)
            if os.name == "posix":
                directory_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except FileExistsError:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("walker exchange manifest is malformed") from error
            if existing != envelope:
                raise ValueError(
                    "walker exchange directory belongs to different physics")
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return path


class _WriterLock:
    """Advisory lock whose ownership is released automatically on process exit."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self._stream.seek(0, os.SEEK_END)
                if self._stream.tell() == 0:
                    self._stream.write(b"0")
                    self._stream.flush()
                self._stream.seek(0)
                msvcrt.locking(self._stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            self._stream.close()
            raise RuntimeError("another process already owns this walker id") from error

    def close(self) -> None:
        if self._stream.closed:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self._stream.seek(0)
                msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        finally:
            self._stream.close()


@dataclass(frozen=True)
class PollResult:
    records: tuple[dict[str, Any], ...]
    offsets: dict[str, int]


class AsyncWalkerStore:
    """One-writer/many-reader append-only exchange directory.

    ``state_dict`` contains only committed reader offsets and the next local
    sequence number.  Bias events themselves belong in the bias checkpoint;
    records committed after that checkpoint are replayed from the saved
    offsets on restart.
    """

    def __init__(self, directory: str | os.PathLike[str], *, method: str,
                 walker: int, walkers: int, state: Mapping[str, Any] | None = None,
                 identity: Mapping[str, Any] | None = None):
        self.directory = Path(directory)
        self.method = str(method)
        if self.method not in ("metadynamics", "opes"):
            raise ValueError("walker store method must be metadynamics or opes")
        self.walker = _checked_index(walker, "walker")
        self.walkers = _checked_index(walkers, "walkers", 2)
        if self.walker >= self.walkers:
            raise ValueError("walker id must be smaller than walker count")
        self.directory.mkdir(parents=True, exist_ok=True)
        if identity is not None:
            ensure_manifest(self.directory, identity)
        self._lock = _WriterLock(self.directory / f"walker-{self.walker:04d}.lock")
        self._offsets = {str(index): 0 for index in range(self.walkers)}
        self._seen: dict[tuple[int, int], str] = {}
        self._next_sequence = 0
        if state is not None:
            self._load_state(state)
        own_path = self.log_path(self.walker)
        own_records, committed = read_committed_records(
            own_path, 0, method=self.method, walker=self.walker)
        # Only the owning writer may repair its own unterminated crash tail.
        # The advisory lock guarantees that no live writer can be truncated.
        if own_path.exists() and own_path.stat().st_size != committed:
            with own_path.open("r+b") as stream:
                stream.truncate(committed)
                stream.flush()
                os.fsync(stream.fileno())
        expected = 0
        self._own_records: list[dict[str, Any]] = []
        for record in own_records:
            event = record["event"]
            if event["sequence"] != expected:
                raise ValueError("walker log sequence has a gap or duplicate")
            self._own_records.append(record)
            expected += 1
        if self._next_sequence > expected:
            raise ValueError("walker checkpoint sequence is ahead of its log")
        self._next_sequence = expected

    def log_path(self, walker: int) -> Path:
        walker = _checked_index(walker, "walker")
        if walker >= self.walkers:
            raise ValueError("walker id is outside this store")
        return self.directory / f"walker-{walker:04d}.jsonl"

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    def ensure_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Idempotently commit one exact local record.

        This is the second phase of workflow publication: a checkpoint can
        retain the prepared record, then restart safely whether the crash was
        before, during or after its log write.
        """
        checked = validate_record(record, method=self.method, walker=self.walker)
        sequence = checked["event"]["sequence"]
        if sequence < len(self._own_records):
            existing = self._own_records[sequence]
            if existing != checked:
                raise ValueError("conflicting already-committed local walker record")
            self._next_sequence = len(self._own_records)
            return existing
        if sequence != len(self._own_records):
            raise ValueError("local walker record sequence has a gap")
        encoded = encode_record(checked)
        created = not self.log_path(self.walker).exists()
        descriptor = os.open(self.log_path(self.walker),
                             os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            written = os.write(descriptor, encoded)
            if written != len(encoded):
                raise OSError("short walker log write")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if created and os.name == "posix":
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        self._own_records.append(checked)
        self._next_sequence = len(self._own_records)
        return checked

    def append(self, payload: Mapping[str, Any], *, step: int,
               time: float) -> dict[str, Any]:
        record = make_record(self.method, self.walker, self._next_sequence,
                             step=step, time=time, payload=payload)
        return self.ensure_record(record)

    def poll(self) -> PollResult:
        accepted: list[dict[str, Any]] = []
        new_offsets = dict(self._offsets)
        new_seen = dict(self._seen)
        for index in range(self.walkers):
            key = str(index)
            records, offset = read_committed_records(
                self.log_path(index), self._offsets[key],
                method=self.method, walker=index)
            expected = 0
            while (index, expected) in new_seen:
                expected += 1
            for record in records:
                event = record["event"]
                identity = (event["walker"], event["sequence"])
                old = new_seen.get(identity)
                if old is not None:
                    if old != record["sha256"]:
                        raise ValueError("conflicting duplicate walker record")
                    continue
                if event["sequence"] != expected:
                    raise ValueError("walker log sequence has a gap or duplicate")
                new_seen[identity] = record["sha256"]
                accepted.append(record)
                expected += 1
            new_offsets[key] = offset
        # Reader progress is one transaction: corruption in any writer log
        # leaves every offset and duplicate guard at the previous checkpoint.
        self._seen = new_seen
        self._offsets = new_offsets
        accepted.sort(key=lambda item: (
            item["event"]["step"], item["event"]["time"],
            item["event"]["walker"], item["event"]["sequence"]))
        return PollResult(tuple(accepted), dict(self._offsets))

    def state_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "method": self.method,
                "walker": self.walker, "walkers": self.walkers,
                "next_sequence": self._next_sequence,
                "offsets": dict(self._offsets),
                "seen": {f"{walker}:{sequence}": digest
                         for (walker, sequence), digest in sorted(self._seen.items())}}

    def _load_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or set(state) != {
                "schema_version", "method", "walker", "walkers",
                "next_sequence", "offsets", "seen"}:
            raise ValueError("invalid walker exchange checkpoint fields")
        if (state["schema_version"] != 1 or state["method"] != self.method or
                state["walker"] != self.walker or state["walkers"] != self.walkers):
            raise ValueError("walker exchange checkpoint identity differs")
        self._next_sequence = _checked_index(state["next_sequence"], "next_sequence")
        offsets = state["offsets"]
        if not isinstance(offsets, Mapping) or set(offsets) != set(self._offsets):
            raise ValueError("walker exchange checkpoint has invalid offsets")
        self._offsets = {key: _checked_index(offsets[key], "walker log offset")
                         for key in self._offsets}
        seen = state["seen"]
        if not isinstance(seen, Mapping):
            raise ValueError("walker exchange checkpoint has invalid duplicate table")
        for key, digest in seen.items():
            try:
                first, second = str(key).split(":", 1)
                identity = (_checked_index(int(first), "seen walker"),
                            _checked_index(int(second), "seen sequence"))
            except (TypeError, ValueError) as error:
                raise ValueError("walker exchange checkpoint has invalid duplicate key") from error
            if (identity[0] >= self.walkers or not isinstance(digest, str) or
                    len(digest) != 64 or any(char not in _HEX for char in digest)):
                raise ValueError("walker exchange checkpoint has invalid duplicate value")
            self._seen[identity] = digest
        for walker in range(self.walkers):
            sequences = sorted(sequence for owner, sequence in self._seen if owner == walker)
            if sequences != list(range(len(sequences))):
                raise ValueError("walker exchange checkpoint duplicate table has a gap")

    def close(self) -> None:
        self._lock.close()

    def __enter__(self) -> "AsyncWalkerStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def mpi_world():
    """Return ``MPI.COMM_WORLD`` with an actionable optional-dependency error."""
    try:
        from mpi4py import MPI
    except ImportError as error:
        raise RuntimeError("walkers_mode='mpi-sync' requires mpi4py") from error
    return MPI.COMM_WORLD


def allgather_centers(center: Any, *, walkers: int, comm=None) -> np.ndarray:
    """Gather one finite CV center per rank in deterministic rank order."""
    if comm is None:
        comm = mpi_world()
    size = int(comm.Get_size())
    if size != walkers:
        raise ValueError(
            f"MPI walker count mismatch: sampling.replicas={walkers}, MPI size={size}")
    local = np.asarray(center, dtype=np.float64)
    if local.ndim != 1 or not np.isfinite(local).all():
        raise ValueError("one MPI walker must contribute one finite [D] center")
    gathered = np.asarray(comm.allgather(local), dtype=np.float64)
    if gathered.shape != (walkers, local.size) or not np.isfinite(gathered).all():
        raise ValueError("MPI allgather returned malformed walker centers")
    return gathered


__all__ = [
    "AsyncWalkerStore", "PollResult", "allgather_centers", "encode_record",
    "ensure_manifest", "make_record", "mpi_world", "read_committed_records",
    "validate_record",
]
