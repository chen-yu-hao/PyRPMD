"""End-to-end asynchronous walker workflows through public workflow APIs."""
from __future__ import annotations

import json
import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.system import System
from pyrpmd.walker_exchange import AsyncWalkerStore
import pyrpmd.workflows.shared_metadynamics as shared_workflow
from pyrpmd.workflows.metadynamics import (
    reanalyze_metadynamics, run_metadynamics)
from pyrpmd.workflows.opes import reanalyze_opes, run_opes


def system():
    return System(("H",), np.array([1.0]), np.array([[0.1, 0.0, 0.0]]))


def metadynamics_config():
    return {
        "task": "metadynamics", "beta": 2.0, "beads": 2, "dt_au": 0.02,
        "seed": 43, "native": False,
        "potential": {"provider": "builtin", "name": "harmonic",
                      "params": {"k": 0.2}},
        "sampling": {"equilibration_steps": 1, "production_steps": 6,
                     "sample_interval": 2, "andersen_interval": 3,
                     "replicas": 2},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0,
                 "axis": "x"}],
        "metadynamics": {"shared_walkers": True,
                         "walkers_mode": "async-files",
                         "mode": "well_tempered", "sigma": [0.3],
                         "height": 0.04, "pace": 2, "bias_factor": 5.0},
        "analysis": {"bins": [5], "bootstrap": 0},
        "output": {"checkpoint_steps": 3, "store_configurations": False},
    }


def opes_config():
    return {
        "task": "opes", "beta": 2.0, "beads": 2, "dt_au": 0.02,
        "seed": 44, "native": False,
        "potential": {"provider": "builtin", "name": "harmonic",
                      "params": {"k": 0.2}},
        "sampling": {"equilibration_steps": 1, "production_steps": 6,
                     "sample_interval": 2, "andersen_interval": 3,
                     "replicas": 2},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0,
                 "axis": "x"}],
        "opes": {"walkers_mode": "async-files", "sigma": [0.3],
                 "barrier": 1.2, "pace": 2, "bias_factor": 5.0},
        "analysis": {"method": "kde", "measure": "cv_density",
                     "bins": [5], "bootstrap": 0},
        "output": {"checkpoint_steps": 3},
    }


def _run_walker(monkeypatch, walker, callable_, *args, **kwargs):
    monkeypatch.setenv("PYRPMD_WALKER_ID", str(walker))
    monkeypatch.setenv("PYRPMD_WALKERS", "2")
    return callable_(*args, **kwargs)


def test_async_metadynamics_walkers_complete_and_reanalyze_after_truncated_tail(
        tmp_path, monkeypatch):
    config = metadynamics_config()
    output = tmp_path / "meta"
    first = _run_walker(
        monkeypatch, 0, run_metadynamics, config, system(), output)
    assert first["status"] == "walker_complete_pending_merge"
    result = _run_walker(
        monkeypatch, 1, run_metadynamics, config, system(), output)
    assert result["walkers_mode"] == "async-files"
    assert result["converged"] is False
    hills = result["hills"]
    log = output / "walkers" / "metadynamics" / "walker-0000.jsonl"
    with log.open("ab") as stream:
        stream.write(b'{"event":{"schema":1')
    restored = reanalyze_metadynamics(config, system(), output)
    assert restored["hills"] == hills
    repaired = _run_walker(
        monkeypatch, 0, run_metadynamics, config, system(), output)
    assert repaired["hills"] == hills
    assert not log.read_bytes().endswith(b'{"event":{"schema":1')


def test_async_opes_walkers_share_uncompressed_kernels_and_reanalyze(
        tmp_path, monkeypatch):
    config = opes_config()
    output = tmp_path / "opes"
    first = _run_walker(monkeypatch, 0, run_opes, config, system(), output)
    assert first["status"] == "walker_complete_pending_merge"
    result = _run_walker(monkeypatch, 1, run_opes, config, system(), output)
    assert result["walkers_mode"] == "async-files"
    assert result["compression"] == "none"
    assert result["kernels"] == 6
    restored = reanalyze_opes(config, system(), output)
    assert restored["kernels"] == result["kernels"]
    assert restored["log_z"] == result["log_z"]
    assert restored["converged"] is False



def _checkpoint_payload(path, kind):
    with load_checkpoint(path, expected_kind=kind, require_integrity=True) as data:
        return {key: data[key].copy() for key in data.files}


def test_pending_metadynamics_publication_resumes_bitwise_without_loss_or_duplicate(
        tmp_path, monkeypatch):
    config = metadynamics_config()
    reference = tmp_path / "reference"
    _run_walker(monkeypatch, 0, run_metadynamics, config, system(), reference)
    _run_walker(monkeypatch, 1, run_metadynamics, config, system(), reference)

    interrupted = tmp_path / "interrupted"
    original = AsyncWalkerStore.ensure_record
    failed = False

    def fail_before_publish(self, record):
        nonlocal failed
        if self.walker == 0 and not failed:
            failed = True
            raise RuntimeError("injected failure before log publication")
        return original(self, record)

    monkeypatch.setattr(AsyncWalkerStore, "ensure_record", fail_before_publish)
    with pytest.raises(RuntimeError, match="before log publication"):
        _run_walker(
            monkeypatch, 0, run_metadynamics, config, system(), interrupted)
    partial_path = (interrupted / "checkpoints" /
                    "shared_metadynamics_async_0000.npz")
    partial = _checkpoint_payload(partial_path, shared_workflow.ASYNC_KIND)
    assert json.loads(str(partial["pending_record"])) is not None

    monkeypatch.setattr(AsyncWalkerStore, "ensure_record", original)
    _run_walker(monkeypatch, 0, run_metadynamics, config, system(), interrupted)
    _run_walker(monkeypatch, 1, run_metadynamics, config, system(), interrupted)

    for walker in range(2):
        name = f"shared_metadynamics_async_{walker:04d}.npz"
        expected = _checkpoint_payload(
            reference / "checkpoints" / name, shared_workflow.ASYNC_KIND)
        actual = _checkpoint_payload(
            interrupted / "checkpoints" / name, shared_workflow.ASYNC_KIND)
        assert actual.keys() == expected.keys()
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        expected_log = (reference / "walkers" / "metadynamics" /
                        f"walker-{walker:04d}.jsonl")
        actual_log = (interrupted / "walkers" / "metadynamics" /
                      f"walker-{walker:04d}.jsonl")
        assert actual_log.read_bytes() == expected_log.read_bytes()
