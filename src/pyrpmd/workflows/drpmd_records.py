"""Crash-consistent, append-only per-parent dRPMD trajectory records.

The small NPZ checkpoint commits a prefix of the JSONL ledger. A journal tail
written just before a crash is discarded on restart, and the corresponding
trajectory can be replayed from its separately committed launch/result files.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from pyrpmd.io import jsonable, _replace_text_file
from .drpmd_persistence import _load_npz

LEGACY_PARENT_KIND = "PyRPMD.drpmd.parent-checkpoint.v1"
PARENT_KIND = "PyRPMD.drpmd.parent-checkpoint.v2"
EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()


def load_parent_checkpoint(path):
    metadata, arrays = _load_npz(path)
    if metadata.get("format") not in (LEGACY_PARENT_KIND, PARENT_KIND):
        raise ValueError("Unsupported dRPMD parent checkpoint payload format")
    return metadata, arrays


def matches_parent_allocation(metadata, ids):
    if metadata["format"] == LEGACY_PARENT_KIND:
        return metadata.get("trajectory_ids") == ids
    return (metadata.get("trajectory_start") == ids[0]
            and metadata.get("trajectory_count") == len(ids))


def _record_line(record):
    portable = jsonable(record, dataclasses="asdict", callables=False, to_dict="ignore")
    return (json.dumps(portable, sort_keys=True, ensure_ascii=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _advance_digest(digest, line):
    return hashlib.sha256(bytes.fromhex(digest) + line).hexdigest()


class ParentRecordLedger:
    """The checkpoint's count/digest identifies its committed JSONL prefix."""

    def __init__(self, checkpoint_path):
        self.path = Path(checkpoint_path).with_suffix(".records.jsonl")
        self.count = 0
        self.digest = EMPTY_DIGEST

    def restore(self, metadata):
        if metadata["format"] == LEGACY_PARENT_KIND:
            records = metadata.get("records")
            if not isinstance(records, list):
                raise ValueError("Legacy parent checkpoint has no records list")
            # A crash in the middle of migration leaves v1 authoritative.
            self.replace(records)
            return records

        count, expected = metadata.get("record_count"), metadata.get("record_digest")
        if (type(count) is not int or count < 0 or not isinstance(expected, str)
                or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError("Invalid dRPMD parent record ledger checkpoint")
        if not self.path.exists():
            if count == 0 and expected == EMPTY_DIGEST:
                return []
            raise ValueError("dRPMD parent record ledger is missing")

        records, digest = [], EMPTY_DIGEST
        with self.path.open("rb+") as stream:
            for _ in range(count):
                line = stream.readline()
                if not line.endswith(b"\n"):
                    raise ValueError("dRPMD parent record ledger is truncated")
                try:
                    record = json.loads(line)
                except (UnicodeError, ValueError) as error:
                    raise ValueError("Invalid dRPMD parent record ledger entry") from error
                if not isinstance(record, dict):
                    raise ValueError("Invalid dRPMD parent record ledger entry")
                digest = _advance_digest(digest, line)
                records.append(record)
            if digest != expected:
                raise ValueError("dRPMD parent record ledger digest mismatch")
            # A valid, but uncheckpointed, append must not get counted twice.
            stream.truncate(stream.tell())
        self.count, self.digest = count, digest
        return records

    def append(self, record):
        line = _record_line(record)
        with self.path.open("ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        self.count += 1
        self.digest = _advance_digest(self.digest, line)

    def replace(self, records):
        """Atomic, one-time import of v1 data or completed dynamic children."""
        fd, temporary = tempfile.mkstemp(
            prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
        count, digest = 0, EMPTY_DIGEST
        try:
            with os.fdopen(fd, "wb") as stream:
                for record in records:
                    line = _record_line(record)
                    stream.write(line)
                    count += 1
                    digest = _advance_digest(digest, line)
                stream.flush()
                os.fsync(stream.fileno())
            _replace_text_file(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.count, self.digest = count, digest
