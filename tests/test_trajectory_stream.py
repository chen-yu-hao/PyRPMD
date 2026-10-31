"""Streaming storage survives interrupted checkpoint commits without replay drift."""

import copy
import hashlib
import json
import tracemalloc

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.trajectory import (
    CHUNK_KIND, TrajectoryError, TrajectoryStream, iter_chunks,
    materialize_stream, validate_stream,
)


SCHEMA = {
    "sample_steps": {"dtype": "<i8", "shape": []},
    "positions": {"dtype": "<f8", "shape": [2, 3]},
    "velocities": {"dtype": "<f8", "shape": [2, 3]},
    "cells": {"dtype": "<f8", "shape": [3, 3]},
    "pressures": {"dtype": "<f8", "shape": []},
}


def record(step):
    return {"sample_steps": np.int64(step), "positions": np.arange(6).reshape(2, 3) + step / 10,
            "velocities": np.ones((2, 3)) * step / 3, "cells": np.eye(3) * (5 + step / 100),
            "pressures": np.float64(step / 1000)}


def expected(steps):
    records = [record(i) for i in steps]
    return {name: np.asarray([r[name] for r in records], dtype=spec["dtype"]).reshape(-1, *spec["shape"])
            for name, spec in SCHEMA.items()}


def assert_arrays(actual, values):
    assert set(actual) == set(values)
    for name in values:
        np.testing.assert_array_equal(actual[name], values[name])


def test_empty_descriptor_and_scalar_fields(tmp_path):
    writer = TrajectoryStream(tmp_path, schema=SCHEMA, chunk_frames=3)
    descriptor = writer.prepare_commit()
    assert descriptor["frames"] == descriptor["chunks"] == 0
    assert descriptor["head"] is None
    assert not list(tmp_path.iterdir())
    assert list(iter_chunks(tmp_path, descriptor)) == []
    assert_arrays(materialize_stream(tmp_path, descriptor), expected([]))


def test_bounded_append_owns_buffers_and_reads_chronological_or_reverse_chunks(tmp_path):
    writer = TrajectoryStream(tmp_path, schema=SCHEMA, chunk_frames=3)
    for step in range(10):
        frame = record(step)
        writer.append(frame)
        frame["positions"][:] = -1e9
        assert writer.frames == step + 1
        assert writer.buffered_frames == (step + 1) % 3
    descriptor = writer.prepare_commit()
    assert descriptor["frames"] == 10 and descriptor["chunks"] == 4
    chunks = list(iter_chunks(tmp_path, descriptor))
    assert [len(chunk["sample_steps"]) for chunk in chunks] == [3, 3, 3, 1]
    assert_arrays({name: np.concatenate([c[name] for c in chunks]) for name in SCHEMA}, expected(range(10)))
    reverse = list(iter_chunks(tmp_path, descriptor, reverse=True, fields=["sample_steps"]))
    assert [c["sample_steps"].tolist() for c in reverse] == [[9], [6, 7, 8], [3, 4, 5], [0, 1, 2]]
    assert_arrays(materialize_stream(tmp_path, descriptor, fields=["positions"]),
                  {"positions": expected(range(10))["positions"]})


def test_restart_uses_checkpoint_prefix_and_ignores_orphan_tail(tmp_path):
    directory = tmp_path / "chunks"
    checkpoint = tmp_path / "state.npz"
    original = TrajectoryStream(directory, schema=SCHEMA, chunk_frames=3)
    original.extend(expected(range(5)))
    descriptor = original.prepare_commit()
    atomic_savez(checkpoint, kind="test.state.v1", trajectory=json.dumps(descriptor), step=4)
    committed = {p: p.read_bytes() for p in directory.glob("*.npz")}
    # These accepted-in-memory frames were durably written, but a power loss
    # happened before the workflow replaced its state/clock checkpoint.
    original.extend(expected(range(5, 12)))
    abandoned = original.prepare_commit()
    assert abandoned["frames"] == 12
    with load_checkpoint(checkpoint) as saved:
        descriptor = json.loads(str(saved["trajectory"]))
    # Changed storage block size is allowed. The old prefix is immutable.
    resumed = TrajectoryStream(directory, descriptor=descriptor, chunk_frames=2)
    resumed.extend(expected(range(5, 9)))
    new_descriptor = resumed.prepare_commit()
    assert_arrays(materialize_stream(directory, new_descriptor), expected(range(9)))
    assert_arrays(materialize_stream(directory, descriptor), expected(range(5)))
    for path, data in committed.items():
        assert path.read_bytes() == data
    assert len(list(directory.glob("*.npz"))) > new_descriptor["chunks"]


@pytest.mark.parametrize("failure_stage", ["save", "publish", "directory_sync"])
def test_failed_automatic_flush_does_not_accept_the_new_frame(tmp_path, monkeypatch, failure_stage):
    import pyrpmd.trajectory as module

    writer = TrajectoryStream(tmp_path, schema=SCHEMA, chunk_frames=2)
    writer.append(record(0))
    original_save, original_link, original_fsync = module.atomic_savez, module.os.link, module.os.fsync
    def failed(*args, **kwargs):
        raise OSError("injected storage failure")
    if failure_stage == "save":
        monkeypatch.setattr(module, "atomic_savez", failed)
    elif failure_stage == "publish":
        monkeypatch.setattr(module.os, "link", failed)
    else:
        # Fail only the directory sync after chunk publication, preserving the
        # container's own file/directory durability sequence.
        def after_save(*args, **kwargs):
            result = original_save(*args, **kwargs)
            monkeypatch.setattr(module.os, "fsync", failed)
            return result
        monkeypatch.setattr(module, "atomic_savez", after_save)
    with pytest.raises(OSError, match="injected storage failure"):
        writer.append(record(1))
    assert writer.frames == writer.buffered_frames == 1 and writer.chunks == 0
    monkeypatch.setattr(module, "atomic_savez", original_save)
    monkeypatch.setattr(module.os, "link", original_link)
    monkeypatch.setattr(module.os, "fsync", original_fsync)
    writer.append(record(1))
    assert_arrays(materialize_stream(tmp_path, writer.prepare_commit()), expected(range(2)))


def test_descriptor_size_and_live_buffer_do_not_grow_with_trajectory_length(tmp_path):
    schema = {"sample_steps": {"dtype": "<i8", "shape": []}}
    writer = TrajectoryStream(tmp_path, schema=schema, chunk_frames=4)
    for i in range(40):
        writer.append({"sample_steps": i})
        assert writer.buffered_frames < 4
    small = writer.prepare_commit()
    for i in range(40, 400):
        writer.append({"sample_steps": i})
        assert writer.buffered_frames < 4
    large = writer.prepare_commit()
    assert len(json.dumps(large)) - len(json.dumps(small)) < 5
    assert sum(len(c["sample_steps"]) for c in iter_chunks(tmp_path, large)) == 400


def test_writing_does_not_retain_a_frame_history_in_memory(tmp_path):
    # A 15 MB logical history with one reused caller buffer. Tracemalloc tracks
    # NumPy data allocations as well as Python containers, so a retained list
    # of frame copies would violate this bound even after compression on disk.
    schema = {"positions": {"dtype": "<f8", "shape": [128, 3]}}
    frame = np.zeros((128, 3))
    tracemalloc.start()
    try:
        writer = TrajectoryStream(tmp_path, schema=schema, chunk_frames=32)
        for i in range(5000):
            frame[0, 0] = i
            writer.append({"positions": frame})
        descriptor = writer.prepare_commit()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert descriptor["frames"] == 5000
    assert peak < 4 * 1024 ** 2


def test_legacy_arrays_migrate_without_changing_values_or_integer_clock(tmp_path):
    arrays = expected(range(17))
    writer = TrajectoryStream(tmp_path, schema=SCHEMA, chunk_frames=4)
    writer.extend(arrays)
    descriptor = writer.prepare_commit()
    restored = TrajectoryStream(tmp_path, descriptor=descriptor, schema=SCHEMA, chunk_frames=7)
    assert restored.frames == 17 and restored.buffered_frames == 0
    assert_arrays(materialize_stream(tmp_path, descriptor), arrays)
    arrays["positions"][:] = -9
    assert_arrays(materialize_stream(tmp_path, descriptor), expected(range(17)))


@pytest.mark.parametrize("mutation", [
    {"head": "../escape"}, {"head": "/tmp/escape"}, {"frames": True},
    {"chunks": -1}, {"format": "PyRPMD.trajectory-stream.v2"},
    {"stream_id": "../../external"}, {"unknown": 1}, {"frames": 0},
])
def test_descriptor_validation_rejects_unsafe_or_inconsistent_inputs(tmp_path, mutation):
    writer = TrajectoryStream(tmp_path, schema=SCHEMA)
    writer.append(record(0))
    descriptor = writer.prepare_commit()
    descriptor.update(mutation)
    with pytest.raises(TrajectoryError):
        validate_stream(tmp_path, descriptor)


def test_missing_truncated_and_rewritten_chunk_are_detected(tmp_path):
    writer = TrajectoryStream(tmp_path, schema=SCHEMA, chunk_frames=2)
    writer.extend(expected(range(4)))
    descriptor = writer.prepare_commit()
    path = tmp_path / f"{descriptor['head']}.npz"
    contents = path.read_bytes()
    path.write_bytes(contents[:len(contents) // 2])
    with pytest.raises(TrajectoryError, match="digest mismatch"):
        materialize_stream(tmp_path, descriptor)
    path.write_bytes(contents)
    path.unlink()
    with pytest.raises(TrajectoryError, match="cannot read"):
        validate_stream(tmp_path, descriptor)


def _rewrite_head(tmp_path, descriptor, mutate):
    old_path = tmp_path / f"{descriptor['head']}.npz"
    with load_checkpoint(old_path) as saved:
        values = {name: saved[name] for name in saved.files}
    metadata = json.loads(str(values["metadata"]))
    mutate(metadata, values)
    values["metadata"] = json.dumps(metadata)
    temporary = tmp_path / "rewritten.npz"
    atomic_savez(temporary, kind=CHUNK_KIND, **values)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.rename(tmp_path / f"{digest}.npz")
    result = copy.deepcopy(descriptor)
    result["head"] = digest
    return result


@pytest.mark.parametrize("field,value", [
    ("start", 3), ("stop", 8), ("index", 0), ("previous", "../escape"),
    ("stream_id", "0" * 32), ("schema_sha256", "0" * 64),
])
def test_valid_npz_and_file_hash_cannot_hide_wrong_chain_metadata(tmp_path, field, value):
    writer = TrajectoryStream(tmp_path, schema=SCHEMA, chunk_frames=2)
    writer.extend(expected(range(4)))
    descriptor = _rewrite_head(tmp_path, writer.prepare_commit(), lambda m, a: m.update({field: value}))
    with pytest.raises(TrajectoryError, match="inconsistent|frame range|invalid trajectory chunk values"):
        validate_stream(tmp_path, descriptor)


def test_corrupted_old_prefix_detected_even_when_latest_chunk_is_intact(tmp_path):
    writer = TrajectoryStream(tmp_path, schema=SCHEMA, chunk_frames=2)
    writer.extend(expected(range(2)))
    first = writer.prepare_commit()
    writer.extend(expected(range(2, 6)))
    final = writer.prepare_commit()
    (tmp_path / f"{first['head']}.npz").write_bytes(b"broken older chunk")
    with pytest.raises(TrajectoryError, match="digest mismatch"):
        TrajectoryStream(tmp_path, descriptor=final)
    # Forward iteration validates the complete prefix before exposing a frame.
    with pytest.raises(TrajectoryError, match="digest mismatch"):
        next(iter_chunks(tmp_path, final))


def test_symlink_chunk_cannot_escape_stream_directory(tmp_path):
    writer = TrajectoryStream(tmp_path / "stream", schema=SCHEMA)
    writer.append(record(0))
    descriptor = writer.prepare_commit()
    path = tmp_path / "stream" / f"{descriptor['head']}.npz"
    external = tmp_path / "external.npz"
    path.rename(external)
    path.symlink_to(external)
    with pytest.raises(TrajectoryError, match="symbolic"):
        validate_stream(path.parent, descriptor)


@pytest.mark.parametrize("field,value", [
    ("sample_steps", 1.25), ("sample_steps", True), ("pressures", np.nan),
    ("positions", np.ones((3, 2))), ("velocities", np.array([["a"] * 3] * 2)),
])
def test_invalid_frame_does_not_change_buffer_or_prefix(tmp_path, field, value):
    writer = TrajectoryStream(tmp_path, schema=SCHEMA, chunk_frames=3)
    writer.append(record(0))
    invalid = record(1)
    invalid[field] = value
    with pytest.raises(TrajectoryError):
        writer.append(invalid)
    assert writer.frames == writer.buffered_frames == 1
    assert_arrays(materialize_stream(tmp_path, writer.prepare_commit()), expected([0]))


def test_writer_does_not_mutate_callers_descriptor_or_schema(tmp_path):
    schema = copy.deepcopy(SCHEMA)
    writer = TrajectoryStream(tmp_path, schema=schema)
    schema["positions"]["shape"][0] = 99
    writer.append(record(0))
    descriptor = writer.prepare_commit()
    original = copy.deepcopy(descriptor)
    restored = TrajectoryStream(tmp_path, descriptor=descriptor)
    restored.append(record(1))
    restored.prepare_commit()
    assert descriptor == original


def test_schema_and_field_selection_fail_early(tmp_path):
    for invalid in ({"x": {"dtype": "O", "shape": []}},
                    {"metadata": {"dtype": "f8", "shape": []}},
                    {"x": {"dtype": "f8", "shape": [0]}}):
        with pytest.raises(TrajectoryError):
            TrajectoryStream(tmp_path, schema=invalid)
    writer = TrajectoryStream(tmp_path, schema=SCHEMA)
    descriptor = writer.prepare_commit()
    with pytest.raises(TrajectoryError, match="unknown"):
        materialize_stream(tmp_path, descriptor, fields=["missing"])
    with pytest.raises(TrajectoryError, match="distinct"):
        list(iter_chunks(tmp_path, descriptor, fields=["positions", "positions"]))
    changed = copy.deepcopy(SCHEMA)
    changed["positions"]["shape"] = [3, 3]
    with pytest.raises(TrajectoryError, match="schema differs"):
        TrajectoryStream(tmp_path, schema=changed, descriptor=descriptor)
