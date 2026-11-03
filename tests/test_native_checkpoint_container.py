"""Native v1 restart payloads share the common atomic storage implementation."""

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from pyrpmd import checkpoint
from pyrpmd.native_ensemble import NativeEnsemble


FORMAT = "pyrpmd.native-ensemble.v1"
ARRAYS = ("q", "p", "aux", "energy", "force", "virial", "cell", "cell_p")


def make_engine(mode="pile-l", *, seed=31):
    pytest.importorskip("pyrpmd._thermal")
    thermostat = mode
    options = {}
    if mode == "andersen":
        thermostat = {"mode": mode, "interval": 3}
    elif mode == "nhc":
        thermostat = {"mode": mode, "tau": .7, "chain_length": 4}
    elif mode == "nm-gle":
        thermostat = {"mode": mode, "gle": {"A": [[1., 2.], [-2., 1.]]}}
    elif mode == "bzp":
        thermostat = "pile-l"
        options = {"barostat": {"mode": "bzp", "pressure": .1, "tau": 3.},
                   "cell": np.eye(3) * 4}
    return NativeEnsemble(
        np.arange(24, dtype=float).reshape(4, 2, 3) * .001,
        [1., 2.], beta=2., dt=.025, seed=seed, thermostat=thermostat,
        potential={"kind": "free"} if mode == "bzp" else {"kind": "harmonic", "k": .7},
        **options,
    )


def legacy_payload(state):
    """The pre-T16b on-disk format, without invoking the current writer."""
    return {"metadata": np.asarray(json.dumps(
        {key: value for key, value in state.items() if key not in ARRAYS},
        sort_keys=True,
    )), **{key: state[key] for key in ARRAYS}}


@pytest.mark.parametrize("mode", ["pile-l", "andersen", "nhc", "nm-gle", "bzp"])
def test_old_native_payload_resumes_exactly_and_upgrades_storage(tmp_path, mode):
    uninterrupted, interrupted = make_engine(mode), make_engine(mode)
    reference = uninterrupted.run(11, stride=2, record_centroids=True)
    first = interrupted.run(5, stride=2, record_centroids=True)
    legacy_path = tmp_path / "legacy.npz"
    payload = legacy_payload(interrupted.state())
    np.savez(legacy_path, **payload)
    original_bytes = legacy_path.read_bytes()

    resumed = make_engine(mode, seed=999)
    resumed.load(legacy_path)
    assert legacy_path.read_bytes() == original_bytes
    assert resumed.state()["digest"] == interrupted.state()["digest"]
    assert resumed.fingerprint == interrupted.fingerprint

    upgraded_path = tmp_path / "upgraded.npz"
    resumed.save(upgraded_path)
    with checkpoint.load_checkpoint(upgraded_path, expected_kind=FORMAT,
                                    require_integrity=True) as saved:
        assert saved.envelope["schema_version"] == 2
        for key, expected in payload.items():
            np.testing.assert_array_equal(saved[key], expected)
    with zipfile.ZipFile(upgraded_path) as archive:
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())

    continued = make_engine(mode, seed=17)
    continued.load(upgraded_path)
    last = continued.run(6, stride=2, record_centroids=True)
    for key in ("samples", "centroids"):
        np.testing.assert_array_equal(np.concatenate([first[key], last[key]]), reference[key])
    assert continued.state()["digest"] == uninterrupted.state()["digest"]


@pytest.mark.parametrize("storage", ["legacy", "envelope", "refreshed_envelope"])
def test_corruption_rejected_before_mutating_native_state(tmp_path, storage):
    engine = make_engine()
    engine.run(3)
    before = engine.state()["digest"]
    path = tmp_path / "state.npz"
    payload = legacy_payload(engine.state())
    if storage == "legacy":
        payload["q"][0, 0, 0] += .1
        np.savez(path, **payload)
        message = "checksum"
    elif storage == "refreshed_envelope":
        payload["q"][0, 0, 0] += .1
        checkpoint.atomic_savez_uncompressed(path, kind=FORMAT, **payload)
        message = "checksum"
    else:
        engine.save(path)
        with np.load(path, allow_pickle=False) as archive:
            payload = {name: archive[name] for name in archive.files}
        payload["q"][0, 0, 0] += .1
        np.savez(path, **payload)
        message = "integrity"
    with pytest.raises(ValueError, match=message):
        engine.load(path)
    assert engine.state()["digest"] == before


@pytest.mark.parametrize("existing", [False, True])
def test_failed_native_checkpoint_publish_preserves_previous_file(tmp_path, monkeypatch, existing):
    engine = make_engine()
    path = tmp_path / "state.npz"
    if existing:
        engine.save(path)
    original_bytes = path.read_bytes() if existing else None
    engine.run(3)
    before = engine.state()["digest"]

    def fail_replace(source, destination):
        assert Path(destination) == path
        raise OSError("injected checkpoint publish failure")

    monkeypatch.setattr(checkpoint.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected checkpoint publish failure"):
        engine.save(path)
    assert engine.state()["digest"] == before
    assert (path.read_bytes() if path.exists() else None) == original_bytes
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("writer,compression", [
    (checkpoint.atomic_savez, zipfile.ZIP_DEFLATED),
    (checkpoint.atomic_savez_uncompressed, zipfile.ZIP_STORED),
])
def test_common_writers_share_payload_and_integrity_contract(tmp_path, writer, compression):
    path = tmp_path / "arrays.npz"
    # Storage controls must not swallow names formerly accepted as payloads.
    payload = {"q": np.arange(12.).reshape(2, 2, 3), "compress": False,
               "compression": "payload", "compressed": np.arange(3)}
    writer(path, kind="test.state.v1", **payload)
    with checkpoint.load_checkpoint(path, require_integrity=True) as saved:
        assert set(saved.files) == set(payload)
        for key, expected in payload.items():
            np.testing.assert_array_equal(saved[key], expected)
    with zipfile.ZipFile(path) as archive:
        assert all(info.compress_type == compression for info in archive.infolist())
    with pytest.raises(checkpoint.CheckpointError, match="object arrays"):
        writer(tmp_path / "unsafe.npz", unsafe=np.array([{}], dtype=object))


def test_native_loader_rejects_foreign_container_kind(tmp_path):
    engine = make_engine()
    before = engine.state()["digest"]
    path = tmp_path / "foreign.npz"
    checkpoint.atomic_savez_uncompressed(path, kind="another.engine.v1",
                                       **legacy_payload(engine.state()))
    with pytest.raises(checkpoint.CheckpointError, match="Expected checkpoint kind"):
        engine.load(path)
    assert engine.state()["digest"] == before
