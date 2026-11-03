"""Cross-platform native publication contracts."""

import os
from pathlib import Path

import numpy as np
import pytest

def test_native_trajectory_flush_works_with_windows_descriptor_contract(tmp_path, monkeypatch):
    pytest.importorskip("pyrpmd._thermal")
    from pyrpmd import native_ensemble

    real_fsync = os.fsync
    flushed = []

    def windows_commit(descriptor):
        # A zero-byte write is harmless and independently detects a read-only
        # descriptor on POSIX too, reproducing Windows _commit's requirement.
        os.write(descriptor, b"")
        flushed.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(native_ensemble.os, "fsync", windows_commit)
    kwargs = dict(q=np.zeros((4, 2, 3)), masses=[1., 2.], beta=2., dt=.025,
                  potential={"kind": "harmonic", "k": .7}, seed=41)
    streamed = native_ensemble.NativeEnsemble(**kwargs)
    control = native_ensemble.NativeEnsemble(**kwargs)
    path = tmp_path / "trajectory.npy"
    result = streamed.run(12, stride=3, trajectory=path)
    expected = control.run(12, stride=3)
    assert len(flushed) == 1
    np.testing.assert_array_equal(result["samples"], expected["samples"])
    for name in ("q", "p"):
        np.testing.assert_array_equal(streamed.state()[name], control.state()[name])
    np.testing.assert_array_equal(np.load(path), expected["samples"])


@pytest.mark.parametrize("malformed", [False, True])
def test_native_publication_unlinks_before_mapping_and_closes_failed_maps(tmp_path, monkeypatch, malformed):
    pytest.importorskip("pyrpmd._thermal")
    from pyrpmd import native_ensemble

    load, unlink = np.load, os.unlink
    opened = []

    def track_load(*args, **kwargs):
        value = load(*args, **kwargs)
        if isinstance(value, np.memmap):
            opened.append(value)
            if malformed:
                return value.reshape(-1)
        return value

    def windows_unlink(path, *args, **kwargs):
        for value in opened:
            if (not value._mmap.closed and Path(path).exists()
                    and os.path.samefile(path, value.filename)):
                raise PermissionError("Windows cannot delete a mapped file")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(native_ensemble.np, "load", track_load)
    monkeypatch.setattr(native_ensemble.os, "unlink", windows_unlink)
    engine = native_ensemble.NativeEnsemble(np.zeros((4, 2, 3)), [1., 2.], 2., .025,
                                            potential={"kind": "harmonic", "k": .7}, seed=41)
    before = engine.state()
    path = tmp_path / "trajectory.npy"
    if malformed:
        with pytest.raises(IndexError):
            engine.run(12, stride=3, trajectory=path)
        assert opened and all(value._mmap.closed for value in opened)
        assert not path.exists()
        for key in ("q", "p"):
            np.testing.assert_array_equal(engine.state()[key], before[key])
    else:
        result = engine.run(12, stride=3, trajectory=path)
        assert path.exists() and len(opened) == 1
        np.testing.assert_array_equal(result["samples"], load(path))
    assert not list(tmp_path.glob("*.tmp"))


def test_native_failed_load_does_not_delete_concurrently_replaced_destination(tmp_path, monkeypatch):
    pytest.importorskip("pyrpmd._thermal")
    from pyrpmd import native_ensemble

    path = tmp_path / "trajectory.npy"
    replacement = tmp_path / "other-writer.npy"
    replacement.write_bytes(b"another writer's data")

    def race_and_fail(*args, **kwargs):
        os.replace(replacement, path)
        raise OSError("injected load failure after replacement")

    monkeypatch.setattr(native_ensemble.np, "load", race_and_fail)
    engine = native_ensemble.NativeEnsemble(np.zeros((4, 2, 3)), [1., 2.], 2., .025,
                                            potential={"kind": "harmonic", "k": .7}, seed=41)
    before = engine.state()
    with pytest.raises(OSError, match="after replacement"):
        engine.run(12, stride=3, trajectory=path)
    assert path.read_bytes() == b"another writer's data"
    assert not list(tmp_path.glob("*.tmp"))
    for key in ("q", "p"):
        np.testing.assert_array_equal(engine.state()[key], before[key])
