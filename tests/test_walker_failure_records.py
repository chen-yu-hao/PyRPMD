"""Failed coupled histories remain visible until their own checkpoint completes."""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.dynamics import Dynamics
from pyrpmd.system import System
from pyrpmd.workflows.biased_sampling import history_failures
from pyrpmd.workflows.metadynamics import run_metadynamics
from pyrpmd.workflows.opes import run_opes


def _system():
    return System(("H",), np.array([1.0]), np.array([[0.1, 0.0, 0.0]]))


def _config(method, mode):
    config = {
        "task": method, "beta": 2.0, "beads": 2, "dt_au": 0.02,
        "seed": 43, "native": False,
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 0.2}},
        "sampling": {"equilibration_steps": 1, "production_steps": 6,
                     "sample_interval": 2, "andersen_interval": 3, "replicas": 2},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
        "analysis": {"bins": [5], "bootstrap": 0},
        "output": {"checkpoint_steps": 3},
    }
    if method == "opes":
        config["opes"] = {"sigma": [0.3], "barrier": 1.2, "pace": 2, "bias_factor": 5.0}
        config["analysis"].update(method="kde", measure="cv_density")
        if mode != "independent":
            config["opes"]["walkers_mode"] = mode
    else:
        config["metadynamics"] = {
            "mode": "well_tempered", "sigma": [0.3], "height": 0.04,
            "pace": 2, "bias_factor": 5.0,
        }
        if mode == "multiple":
            config["metadynamics"].update(multiple_walkers=True, block_steps=2)
        elif mode != "independent":
            config["metadynamics"].update(shared_walkers=True, walkers_mode=mode)
    return config


def _fail_third_step(monkeypatch):
    original = Dynamics.step

    def fail(self, state, *args, **kwargs):
        result = original(self, state, *args, **kwargs)
        if state.step == 3:
            raise RuntimeError("synthetic PES domain failure")
        return result

    monkeypatch.setattr(Dynamics, "step", fail)


def _assert_active(output, *, mode, elapsed, checkpoint_elapsed=None):
    failures = history_failures(output)
    assert failures
    for record in failures:
        assert record["status"] == "history_failed"
        assert "synthetic PES domain failure" in record["error"]["message"]
        assert record["state"]["elapsed_steps"] == elapsed
        assert record["state"]["step"] == elapsed
        assert record["checkpoint"]["committed_elapsed_steps"] == (
            elapsed if checkpoint_elapsed is None else checkpoint_elapsed)
        if mode != "independent":
            assert record["history"]["walkers_mode"] == mode
        with load_checkpoint(record["checkpoint"]["path"]) as saved:
            assert int(saved["elapsed"]) == record["checkpoint"]["committed_elapsed_steps"]
    return failures


def _assert_resolved(output, old):
    assert history_failures(output) == []
    for record in old:
        persisted = json.loads(open(record["file"]).read())
        assert persisted["status"] == "superseded"
        assert persisted["error"] == record["error"]
        assert persisted["state"] == record["state"]
        assert persisted["resolution"]["elapsed_steps"] == 7


@pytest.mark.parametrize("method,mode", [
    ("metadynamics", "independent"), ("opes", "independent"),
    ("metadynamics", "serial"), ("metadynamics", "multiple"),
    ("metadynamics", "async-files"), ("opes", "async-files"),
])
def test_failed_history_is_preserved_and_only_successful_resume_supersedes(
        tmp_path, monkeypatch, method, mode):
    config = _config(method, mode)
    run = run_opes if method == "opes" else run_metadynamics
    if mode == "async-files":
        monkeypatch.setenv("PYRPMD_WALKER_ID", "0")
        monkeypatch.setenv("PYRPMD_WALKERS", "2")
    with monkeypatch.context() as patch:
        _fail_third_step(patch)
        with pytest.raises(RuntimeError, match="synthetic PES domain failure"):
            run(config, _system(), tmp_path)
    failures = _assert_active(tmp_path, mode=mode, elapsed=2)
    if mode == "multiple":
        assert failures[0]["state"]["role"] == "local_state_in_uncommitted_shared_block"
    run(config, _system(), tmp_path)
    _assert_resolved(tmp_path, failures)


class _RootComm:
    """Exercise rank-zero collective control flow without an optional MPI runtime."""
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 2

    def bcast(self, value, root=0):
        return value

    def gather(self, value, root=0):
        return [value, value]

    def allgather(self, value):
        return [value, value]


@pytest.mark.parametrize("method,module", [
    ("metadynamics", "_distributed_meta_mpi"), ("opes", "_distributed_opes_mpi"),
])
def test_mpi_collective_failure_is_recorded_after_rollback_and_recovers(
        tmp_path, monkeypatch, method, module):
    workflow = importlib.import_module("pyrpmd.workflows." + module)
    monkeypatch.setattr(workflow, "mpi_world", lambda: _RootComm())
    config = _config(method, "mpi-sync")
    with monkeypatch.context() as patch:
        _fail_third_step(patch)
        with pytest.raises(RuntimeError, match="synthetic PES domain failure"):
            workflow._run_mpi(config, _system(), tmp_path)
    failures = _assert_active(tmp_path, mode="mpi-sync", elapsed=2)
    workflow._run_mpi(config, _system(), tmp_path)
    _assert_resolved(tmp_path, failures)


def test_multiple_walker_record_distinguishes_local_and_durable_clock(tmp_path, monkeypatch):
    config = _config("metadynamics", "multiple")
    config["metadynamics"]["block_steps"] = 4
    with monkeypatch.context() as patch:
        _fail_third_step(patch)
        with pytest.raises(RuntimeError, match="synthetic PES domain failure"):
            run_metadynamics(config, _system(), tmp_path)
    failure = history_failures(tmp_path)[0]
    assert failure["state"]["elapsed_steps"] == 2
    assert failure["checkpoint"]["committed_elapsed_steps"] == 0
    assert failure["state"]["role"] == "local_state_in_uncommitted_shared_block"
    run_metadynamics(config, _system(), tmp_path)
    _assert_resolved(tmp_path, [failure])


@pytest.mark.parametrize("method", ["metadynamics", "opes"])
def test_finishing_async_peer_does_not_supersede_failed_walker(tmp_path, monkeypatch, method):
    config = _config(method, "async-files")
    run = run_opes if method == "opes" else run_metadynamics
    monkeypatch.setenv("PYRPMD_WALKER_ID", "1")
    monkeypatch.setenv("PYRPMD_WALKERS", "2")
    with monkeypatch.context() as patch:
        _fail_third_step(patch)
        with pytest.raises(RuntimeError, match="synthetic PES domain failure"):
            run(config, _system(), tmp_path)
    failed = history_failures(tmp_path)
    assert failed[0]["history"]["walker"] == 1
    monkeypatch.setenv("PYRPMD_WALKER_ID", "0")
    run(config, _system(), tmp_path)
    assert history_failures(tmp_path) == failed
    monkeypatch.setenv("PYRPMD_WALKER_ID", "1")
    run(config, _system(), tmp_path)
    _assert_resolved(tmp_path, failed)


def test_failure_record_io_does_not_replace_original_sampling_error(tmp_path, monkeypatch):
    from pyrpmd.workflows import _walker_failures
    config = _config("metadynamics", "serial")
    _fail_third_step(monkeypatch)

    def fail_write(*args, **kwargs):
        raise OSError("failure-record disk unavailable")

    monkeypatch.setattr(_walker_failures, "write_report_json", fail_write)
    with pytest.raises(RuntimeError, match="synthetic PES domain failure") as caught:
        run_metadynamics(config, _system(), tmp_path)
    assert any("failure-record disk unavailable" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("method", ["metadynamics", "opes"])
def test_async_publication_failure_keeps_pending_checkpoint_and_failure_record(
        tmp_path, monkeypatch, method):
    from pyrpmd.walker_exchange import AsyncWalkerStore
    config = _config(method, "async-files")
    run = run_opes if method == "opes" else run_metadynamics
    monkeypatch.setenv("PYRPMD_WALKER_ID", "0")
    monkeypatch.setenv("PYRPMD_WALKERS", "2")

    def fail_publish(*args, **kwargs):
        raise OSError("synthetic log publication failure")

    with monkeypatch.context() as patch:
        patch.setattr(AsyncWalkerStore, "ensure_record", fail_publish)
        with pytest.raises(OSError, match="synthetic log publication failure"):
            run(config, _system(), tmp_path)
    failures = history_failures(tmp_path)
    assert len(failures) == 1
    record = failures[0]
    assert record["state"]["elapsed_steps"] == 3
    assert record["checkpoint"]["committed_elapsed_steps"] == 3
    assert record["bias_state"]["pending_record"] is not None
    run(config, _system(), tmp_path)
    _assert_resolved(tmp_path, failures)
