"""Immutable, bounded-memory trajectory chunks committed by a checkpoint.

The workflow's small atomic checkpoint owns the authoritative descriptor.
Writing a chunk does not commit a simulation step: only replacing that
checkpoint commits its referenced prefix. Unreferenced chunks left by a crash
are harmless and never become samples merely because they exist on disk.

Chunk files are content addressed and linked backwards. The checkpoint stores
one head digest rather than an ever-growing list of filenames. Readers verify
both the complete-file SHA256 and the checkpoint container's array checksums.
No paths from a descriptor are followed and no pickle/object arrays are used.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import tempfile
import uuid
from collections.abc import Mapping
from functools import partial
from pathlib import Path

import numpy as np

from .checkpoint import CheckpointError, atomic_savez, load_checkpoint
from .validation import integer_count

FORMAT = "PyRPMD.trajectory-stream.v1"
CHUNK_KIND = "PyRPMD.trajectory-chunk.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_STREAM_ID = re.compile(r"[0-9a-f]{32}\Z")
_FIELD = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_DESCRIPTOR_KEYS = {"format", "stream_id", "frames", "chunks", "head", "schema"}
_CHUNK_KEYS = {"format", "stream_id", "index", "start", "stop", "previous", "schema_sha256"}


class TrajectoryError(CheckpointError):
    """The selected committed trajectory cannot be used safely."""


_integer = partial(
    integer_count,
    maximum=np.iinfo(np.int64).max,
    error_type=TrajectoryError,
    message="{name} must be an integer >= {minimum}",
)


def _schema(value):
    if not isinstance(value, Mapping) or not value:
        raise TrajectoryError("trajectory schema must be a nonempty field mapping")
    if any(not isinstance(name, str) for name in value):
        raise TrajectoryError("trajectory field names must be strings")
    result = {}
    for name, spec in sorted(value.items()):
        if not isinstance(name, str) or not _FIELD.fullmatch(name) or name == "metadata":
            raise TrajectoryError("trajectory field names must be identifiers other than metadata")
        if not isinstance(spec, Mapping) or set(spec) != {"dtype", "shape"}:
            raise TrajectoryError(f"trajectory field {name!r} needs dtype and shape")
        try:
            dtype = np.dtype(spec["dtype"])
        except (TypeError, ValueError) as error:
            raise TrajectoryError(f"invalid trajectory dtype for {name!r}") from error
        if dtype.kind not in "biuf" or dtype.hasobject or dtype.fields is not None:
            raise TrajectoryError("trajectory fields must be real numerical arrays")
        if not isinstance(spec["shape"], (tuple, list)):
            raise TrajectoryError("trajectory field shape must be a list of dimensions")
        shape = [_integer(n, f"shape of {name}", 1) for n in spec["shape"]]
        result[name] = {"dtype": dtype.str, "shape": shape}
    return result


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _schema_hash(schema):
    return hashlib.sha256(_json(schema).encode("utf-8")).hexdigest()


def _descriptor(value):
    if not isinstance(value, Mapping) or set(value) != _DESCRIPTOR_KEYS or value.get("format") != FORMAT:
        raise TrajectoryError("unsupported or invalid trajectory descriptor")
    stream_id = value["stream_id"]
    if not isinstance(stream_id, str) or not _STREAM_ID.fullmatch(stream_id):
        raise TrajectoryError("invalid trajectory stream identity")
    frames = _integer(value["frames"], "trajectory frames")
    chunks = _integer(value["chunks"], "trajectory chunks")
    head = value["head"]
    if ((frames == 0) != (chunks == 0) or chunks > frames
            or (chunks == 0 and head is not None)
            or (chunks > 0 and (not isinstance(head, str) or not _DIGEST.fullmatch(head)))):
        raise TrajectoryError("trajectory descriptor has inconsistent prefix counts or head digest")
    return {"format": FORMAT, "stream_id": stream_id, "frames": frames,
            "chunks": chunks, "head": head, "schema": _schema(value["schema"])}


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_path(directory, digest):
    # A descriptor contains only a digest, never a path. Reject symlinks to
    # avoid accepting an external chunk as part of this trajectory directory.
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise TrajectoryError("invalid trajectory chunk digest")
    path = Path(directory) / f"{digest}.npz"
    if path.is_symlink():
        raise TrajectoryError("trajectory chunks must not be symbolic links")
    return path


def _load_chunk(directory, digest, descriptor, index, stop, fields=None):
    path = _chunk_path(directory, digest)
    try:
        if _file_digest(path) != digest:
            raise TrajectoryError(f"trajectory chunk file digest mismatch: {path.name}")
        with load_checkpoint(path, expected_kind=CHUNK_KIND, require_integrity=True) as saved:
            if set(saved.files) != set(descriptor["schema"]) | {"metadata"}:
                raise TrajectoryError("trajectory chunk field inventory differs from its schema")
            try:
                metadata = json.loads(str(saved["metadata"].item()))
            except (ValueError, TypeError, AttributeError) as error:
                raise TrajectoryError("invalid trajectory chunk metadata") from error
            if not isinstance(metadata, dict) or set(metadata) != _CHUNK_KEYS:
                raise TrajectoryError("invalid trajectory chunk metadata fields")
            start = _integer(metadata["start"], "chunk start")
            metadata_stop = _integer(metadata["stop"], "chunk stop", 1)
            metadata_index = _integer(metadata["index"], "chunk index")
            previous = metadata["previous"]
            if (metadata["format"] != CHUNK_KIND or metadata["stream_id"] != descriptor["stream_id"]
                    or metadata["schema_sha256"] != _schema_hash(descriptor["schema"])
                    or metadata_index != index or metadata_stop != stop or not start < stop
                    or (index == 0 and (start != 0 or previous is not None))
                    or (index > 0 and (start < index or not isinstance(previous, str)
                                      or not _DIGEST.fullmatch(previous)))):
                raise TrajectoryError("trajectory chunk linkage, schema, or frame range is inconsistent")
            count = stop - start
            arrays = {}
            for name, spec in descriptor["schema"].items():
                array = saved[name]
                if (array.dtype.str != spec["dtype"] or array.shape != (count, *spec["shape"])
                        or not np.isfinite(array).all()):
                    raise TrajectoryError(f"invalid trajectory chunk values for {name!r}")
                if fields is None or name in fields:
                    arrays[name] = array
            return metadata, arrays
    except TrajectoryError:
        raise
    except (OSError, CheckpointError) as error:
        raise TrajectoryError(f"cannot read trajectory chunk {path.name}: {error}") from error


def _walk_reverse(directory, descriptor, fields=None):
    digest, stop = descriptor["head"], descriptor["frames"]
    for index in range(descriptor["chunks"] - 1, -1, -1):
        metadata, arrays = _load_chunk(directory, digest, descriptor, index, stop, fields)
        yield digest, metadata, arrays
        digest, stop = metadata["previous"], metadata["start"]
    if digest is not None or stop != 0:
        raise TrajectoryError("trajectory chain does not terminate at its initial frame")


def validate_stream(directory, descriptor):
    """Verify the entire committed prefix with memory bounded by one chunk.

    The returned descriptor is normalized and independent of the caller's
    mutable mapping. Extra files are deliberately ignored, including chunks
    written after the most recent successful checkpoint replacement.
    """
    descriptor = _descriptor(descriptor)
    for _ in _walk_reverse(directory, descriptor, fields=set()):
        pass
    return descriptor


def _selected_fields(descriptor, fields):
    if fields is None:
        return set(descriptor["schema"])
    fields = list(fields)
    if any(not isinstance(f, str) for f in fields) or len(fields) != len(set(fields)):
        raise TrajectoryError("trajectory field selection must contain distinct names")
    unknown = set(fields) - set(descriptor["schema"])
    if unknown:
        raise TrajectoryError("unknown trajectory fields: " + ", ".join(sorted(unknown)))
    return set(fields)


def iter_chunks(directory, descriptor, *, fields=None, reverse=False):
    """Yield validated arrays for one chunk at a time.

    ``reverse=True`` follows the committed chain from newest to oldest without
    temporary storage. Chronological iteration first validates the chain into
    a temporary disk index of hashes, then reads it in reverse order. Neither
    mode retains a trajectory-sized in-memory list of chunks or filenames.
    Fields not requested are still integrity-checked, but not yielded.
    """
    descriptor = _descriptor(descriptor)
    fields = _selected_fields(descriptor, fields)
    if reverse:
        for _, _, arrays in _walk_reverse(directory, descriptor, fields):
            yield arrays
        return
    with tempfile.TemporaryFile(mode="w+b") as index_file:
        for digest, metadata, _ in _walk_reverse(directory, descriptor, fields=set()):
            index_file.write(digest.encode("ascii") + struct.pack("<Q", metadata["stop"]))
        start = 0
        for index in range(descriptor["chunks"]):
            index_file.seek((descriptor["chunks"] - 1 - index) * 72)
            digest = index_file.read(64).decode("ascii")
            stop = struct.unpack("<Q", index_file.read(8))[0]
            metadata, arrays = _load_chunk(directory, digest, descriptor, index, stop, fields)
            if metadata["start"] != start:
                raise TrajectoryError("trajectory chunks are not a continuous chronological prefix")
            start = metadata["stop"]
            yield arrays
        if start != descriptor["frames"]:
            raise TrajectoryError("trajectory chunks do not match the committed frame count")


def materialize_stream(directory, descriptor, *, fields=None):
    """Collect a committed prefix for explicit offline analysis.

    Unlike writing and iteration, this function intentionally allocates the
    selected full histories. Production workers should return the descriptor.
    """
    descriptor = _descriptor(descriptor)
    fields = _selected_fields(descriptor, fields)
    arrays = {name: np.empty((descriptor["frames"], *spec["shape"]), dtype=spec["dtype"])
              for name, spec in descriptor["schema"].items() if name in fields}
    # Reverse traversal supplies absolute frame ranges, so collecting does not
    # need a filename index or a temporary concatenation copy of every frame.
    for _, metadata, chunk in _walk_reverse(directory, descriptor, fields):
        for name, value in chunk.items():
            arrays[name][metadata["start"]:metadata["stop"]] = value
    return arrays


class TrajectoryStream:
    """Append bounded batches while an external checkpoint commits prefixes.

    ``schema`` maps field names to ``{"dtype": "<f8", "shape": [N, 3]}``;
    a scalar frame uses an empty shape. Construct with ``descriptor`` on
    restart. Chunk size is an execution/storage choice and may change on
    restart without changing a simulation's scientific identity.

    Call ``prepare_commit`` before atomically saving the simulation state and
    returned descriptor together. Never infer a committed prefix by listing
    files. There is one writer per replica; no concurrent writer lock is used.
    """

    def __init__(self, directory, *, schema=None, chunk_frames=256, descriptor=None, validate=True):
        self.directory = Path(directory)
        self.chunk_frames = _integer(chunk_frames, "trajectory chunk_frames", 1)
        if descriptor is None:
            normalized = _schema(schema)
            self._descriptor = {"format": FORMAT, "stream_id": uuid.uuid4().hex,
                                "frames": 0, "chunks": 0, "head": None, "schema": normalized}
        else:
            self._descriptor = validate_stream(self.directory, descriptor) if validate else _descriptor(descriptor)
            if schema is not None and _schema(schema) != self._descriptor["schema"]:
                raise TrajectoryError("trajectory restart schema differs from the committed schema")
        self.schema = _schema(self._descriptor["schema"])
        self._schema_digest = _schema_hash(self.schema)
        self._buffer = {name: np.empty((self.chunk_frames, *spec["shape"]), dtype=spec["dtype"])
                        for name, spec in self.schema.items()}
        self._size = 0

    @property
    def frames(self):
        return self._descriptor["frames"] + self._size

    @property
    def buffered_frames(self):
        return self._size

    @property
    def chunks(self):
        return self._descriptor["chunks"]

    def append(self, record):
        """Append exactly one finite frame, owning all caller buffers.

        Validation or chunk I/O failure leaves the accepted prefix and buffer
        count unchanged. A fully written but unlinked-to-checkpoint chunk may
        remain after an I/O failure; it is never followed on restart.
        """
        if not isinstance(record, Mapping) or set(record) != set(self.schema):
            raise TrajectoryError("trajectory record fields differ from the stream schema")
        values = {}
        for name, spec in self.schema.items():
            value = np.asarray(record[name])
            if (value.dtype.kind not in "biuf" or value.shape != tuple(spec["shape"])
                    or not np.isfinite(value).all()):
                raise TrajectoryError(f"invalid trajectory frame field {name!r}")
            dtype = np.dtype(spec["dtype"])
            if dtype.kind in "iu" and value.dtype.kind == "b":
                raise TrajectoryError(f"trajectory integer field {name!r} cannot be boolean")
            with np.errstate(over="ignore", invalid="ignore"):
                cast = value.astype(dtype)
            if (not np.isfinite(cast).all()
                    or (dtype.kind in "biu" and not np.array_equal(value, cast))):
                raise TrajectoryError(f"trajectory field {name!r} cannot be represented by its declared dtype")
            values[name] = cast
        old_size = self._size
        for name, value in values.items():
            self._buffer[name][old_size] = value
        self._size += 1
        try:
            if self._size == self.chunk_frames:
                self._flush()
        except BaseException:
            self._size = old_size
            raise

    def extend(self, arrays):
        """Append a legacy in-memory history without copying it as a whole."""
        if not isinstance(arrays, Mapping) or set(arrays) != set(self.schema):
            raise TrajectoryError("trajectory arrays differ from the stream schema")
        arrays = {name: np.asarray(value) for name, value in arrays.items()}
        counts = set()
        for name, value in arrays.items():
            if value.ndim != len(self.schema[name]["shape"]) + 1 or value.shape[1:] != tuple(self.schema[name]["shape"]):
                raise TrajectoryError(f"invalid trajectory history shape for {name!r}")
            counts.add(value.shape[0])
        if len(counts) != 1:
            raise TrajectoryError("trajectory history fields have different frame counts")
        for i in range(counts.pop()):
            self.append({name: value[i] for name, value in arrays.items()})

    def _flush(self):
        if not self._size:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        metadata = {"format": CHUNK_KIND, "stream_id": self._descriptor["stream_id"],
                    "index": self._descriptor["chunks"], "start": self._descriptor["frames"],
                    "stop": self.frames, "previous": self._descriptor["head"],
                    "schema_sha256": self._schema_digest}
        descriptor, temporary = tempfile.mkstemp(prefix=".trajectory-", suffix=".npz", dir=self.directory)
        os.close(descriptor)
        temporary = Path(temporary)
        try:
            atomic_savez(temporary, kind=CHUNK_KIND, metadata=_json(metadata),
                         **{name: value[:self._size] for name, value in self._buffer.items()})
            digest = _file_digest(temporary)
            final = _chunk_path(self.directory, digest)
            try:
                # A hard link publishes without replacing any immutable chunk.
                # The temporary file is on the same filesystem by construction.
                os.link(temporary, final)
            except FileExistsError:
                if _file_digest(final) != digest:
                    raise TrajectoryError("existing content-addressed chunk is corrupt")
            if os.name == "posix":
                directory_fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        # No fallible I/O follows this point. If publication or cleanup fails,
        # the previous descriptor and all buffered accepted frames stay intact.
        self._descriptor["head"] = digest
        self._descriptor["frames"] = metadata["stop"]
        self._descriptor["chunks"] += 1
        self._size = 0

    def prepare_commit(self):
        """Durably flush pending data and return the prefix for atomic commit."""
        self._flush()
        return _descriptor(self._descriptor)
