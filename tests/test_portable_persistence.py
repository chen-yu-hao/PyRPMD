"""Cross-platform native publication contracts."""

import os

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
