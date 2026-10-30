"""Real loaded BLAS controls in spawn workers and their caller scope."""

import json
import os
from pathlib import Path
import time

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from pyrpmd.parallel import ProcessBeadPotential, run_tasks
from pyrpmd.parallel.execution import _bead_init
from pyrpmd.potentials import PESResult


BLAS_ENV = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _threads(user_api="blas"):
    return {pool["filepath"]: pool["num_threads"] for pool in threadpool_info()
            if pool["user_api"] == user_api}


# In a spawned interpreter this records NumPy's already loaded thread pool,
# before _execute_batch or the bead initializer can apply any limits.
IMPORTED_THREADS = _threads()


class WorkerRendezvous:
    """Make two actual workers overlap without a socket-based Manager."""

    def __init__(self, directory):
        self.directory = str(directory)
        self.generation = 0

    def wait(self, timeout):
        self.generation += 1
        directory = Path(self.directory)
        (directory / f"{self.generation}-{os.getpid()}").touch()
        deadline = time.monotonic() + timeout
        while len(list(directory.glob(f"{self.generation}-*"))) < 2:
            if time.monotonic() >= deadline:
                raise TimeoutError("Both spawn workers did not reach the probe")
            time.sleep(0.01)


def _snapshot(barrier):
    if barrier is not None:
        barrier.wait(timeout=30)
    return {"pid": os.getpid(), "imported": IMPORTED_THREADS,
            "threads": _threads(), "environment": {key: os.environ.get(key) for key in BLAS_ENV}}


class ThreadProbePotential:
    def __init__(self, barrier):
        self.barrier = barrier

    def evaluate(self, q):
        snapshot = json.dumps(_snapshot(self.barrier))
        return PESResult(np.zeros(len(q)), np.zeros_like(q),
                         messages=tuple(snapshot for _ in q))


@pytest.fixture
def multithreaded_blas(monkeypatch):
    if not _threads():
        pytest.skip("No controllable BLAS thread pool is available")
    for name in BLAS_ENV:
        monkeypatch.setenv(name, "4")
    with threadpool_limits(limits=4, user_api="blas"):
        if any(value != 4 for value in _threads().values()):
            pytest.skip("This BLAS runtime cannot provide four threads")
        yield


def _check_two_workers(snapshots):
    assert len({snapshot["pid"] for snapshot in snapshots}) == 2
    assert all(snapshot["pid"] != os.getpid() for snapshot in snapshots)
    for snapshot in snapshots:
        assert snapshot["imported"]
        assert set(snapshot["imported"].values()) == {4}
        assert set(snapshot["threads"].values()) == {1}
        assert set(snapshot["environment"].values()) == {"1"}


def test_spawn_batches_limit_already_loaded_blas_in_each_worker(tmp_path, multithreaded_blas):
    before = _threads()
    environment = {key: os.environ.get(key) for key in BLAS_ENV}
    barrier = WorkerRendezvous(tmp_path)
    snapshots = run_tasks(_snapshot, [barrier, barrier],
                          {"backend": "process", "trajectory_workers": 2})
    _check_two_workers(snapshots)
    assert _threads() == before
    assert {key: os.environ.get(key) for key in BLAS_ENV} == environment


def test_persistent_bead_workers_keep_limits_for_multiple_evaluations(tmp_path, multithreaded_blas):
    before = _threads()
    environment = {key: os.environ.get(key) for key in BLAS_ENV}
    potential = ProcessBeadPotential(ThreadProbePotential(WorkerRendezvous(tmp_path)), None, 2)
    try:
        previous_pids = None
        for _ in range(2):
            result = potential.evaluate(np.zeros((2, 1, 3)))
            snapshots = [json.loads(message) for message in result.messages]
            _check_two_workers(snapshots)
            pids = {snapshot["pid"] for snapshot in snapshots}
            if previous_pids is not None:
                assert pids == previous_pids
            previous_pids = pids
    finally:
        potential.close()
    assert _threads() == before
    assert {key: os.environ.get(key) for key in BLAS_ENV} == environment


@pytest.mark.parametrize("fail", [False, True])
def test_in_process_batch_restores_loaded_threads_and_environment(
        monkeypatch, multithreaded_blas, fail):
    # A single actual task uses the caller even when more workers are requested.
    monkeypatch.delenv("BLIS_NUM_THREADS")
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    before, openmp = _threads(), _threads("openmp")
    environment = {key: os.environ.get(key) for key in BLAS_ENV}

    def task(_):
        assert set(_threads().values()) == {1}
        assert all(os.environ[key] == "1" for key in BLAS_ENV)
        assert os.environ["OMP_NUM_THREADS"] == "3"
        assert _threads("openmp") == openmp
        if fail:
            raise RuntimeError("task failure")
        return "complete"

    if fail:
        with pytest.raises(RuntimeError, match="task failure"):
            run_tasks(task, [None], {"backend": "process", "trajectory_workers": 2})
    else:
        assert run_tasks(task, [None], {"backend": "process", "trajectory_workers": 2}) == ["complete"]
    assert _threads() == before
    assert {key: os.environ.get(key) for key in BLAS_ENV} == environment
    assert _threads("openmp") == openmp


def test_bead_initializer_failure_restores_limits(monkeypatch, multithreaded_blas):
    before = _threads()
    environment = {key: os.environ.get(key) for key in BLAS_ENV}

    def failed_provider(*args):
        assert set(_threads().values()) == {1}
        raise RuntimeError("provider failure")

    monkeypatch.setattr("pyrpmd.potentials.make_potential", failed_provider)
    with pytest.raises(RuntimeError, match="provider failure"):
        _bead_init({}, None)
    assert _threads() == before
    assert {key: os.environ.get(key) for key in BLAS_ENV} == environment
