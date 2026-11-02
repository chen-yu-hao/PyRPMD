"""Bounded CPU queue tests, including actual spawn workers and rendezvous."""

from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from multiprocessing.util import Finalize
from pathlib import Path
import os
import pickle
import time

import pytest

from pyrpmd.parallel import process_queue


def _square(value):
    return value * value


def _wait_for(path):
    deadline = time.monotonic() + 30
    while not path.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError(f"rendezvous timed out: {path.name}")
        time.sleep(0.01)


def _rendezvous(task):
    index, directory = task
    directory = Path(directory)
    if index == 0:
        (directory / "blocked").touch()
        _wait_for(directory / "release")
    elif index == 1:
        _wait_for(directory / "blocked")
    elif index == 2:
        (directory / "release").touch()
    return index


def _fail_initializer():
    raise RuntimeError("initializer failed deliberately")


def _unpicklable(_):
    return lambda: None


def _fail_task(value):
    if value == 0:
        raise ValueError("task failed deliberately")
    return value


def _record_worker(directory):
    path = Path(directory) / str(os.getpid())
    path.with_suffix(".started").touch()
    Finalize(None, path.with_suffix(".closed").touch, exitpriority=10)


def test_real_spawn_refills_while_earlier_task_is_blocked(tmp_path):
    # With static even/odd batches, tasks 0 and 2 occupy the same worker and
    # cannot pass this rendezvous. No wall-clock speedup threshold is asserted.
    assert process_queue.ordered_process_map(
        _rendezvous, [(i, str(tmp_path)) for i in range(5)], max_workers=2,
    ) == list(range(5))


@pytest.mark.parametrize("size,workers", [(0, 2), (1, 4), (7, 2)])
def test_empty_underfilled_and_ordered_results(size, workers):
    assert process_queue.ordered_process_map(
        _square, iter(range(size)), max_workers=workers,
    ) == [i * i for i in range(size)]


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_worker_count_validation(workers):
    with pytest.raises(ValueError, match="positive integer"):
        process_queue.ordered_process_map(_square, [], max_workers=workers)


@pytest.mark.parametrize("fail", [False, True])
def test_worker_finalizers_run_after_success_or_task_failure(tmp_path, fail):
    kwargs = dict(max_workers=2, initializer=_record_worker, initargs=(str(tmp_path),))
    if fail:
        with pytest.raises(ValueError, match="task failed deliberately"):
            process_queue.ordered_process_map(_fail_task, range(8), **kwargs)
    else:
        assert process_queue.ordered_process_map(_square, range(8), **kwargs) == [i*i for i in range(8)]
    started = {p.stem for p in tmp_path.glob("*.started")}
    assert started
    assert started == {p.stem for p in tmp_path.glob("*.closed")}


def test_initializer_failure_propagates():
    with pytest.raises(BrokenProcessPool):
        process_queue.ordered_process_map(
            _square, range(4), max_workers=2, initializer=_fail_initializer,
        )


def test_unpicklable_result_propagates():
    with pytest.raises((AttributeError, TypeError, pickle.PicklingError)):
        process_queue.ordered_process_map(_unpicklable, range(4), max_workers=2)


class _FakePool:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs, self.submitted = kwargs, []
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, function, task):
        future = Future()
        future.compute = lambda: function(task)
        self.submitted.append(future)
        return future


def test_twenty_thousand_tasks_have_bounded_futures(monkeypatch):
    maximum = 0

    def finish_one(pending, *, return_when):
        nonlocal maximum
        maximum = max(maximum, len(pending))
        assert return_when == process_queue.FIRST_COMPLETED
        # Deliberately complete the newest task first; results stay ordered.
        future = list(pending)[-1]
        future.set_result(future.compute())
        return {future}, set(pending) - {future}

    monkeypatch.setattr(process_queue, "ProcessPoolExecutor", _FakePool)
    monkeypatch.setattr(process_queue, "wait", finish_one)
    assert process_queue.ordered_process_map(_square, range(20000), max_workers=4) == [i*i for i in range(20000)]
    assert maximum == 4
    assert _FakePool.instances[-1].kwargs["mp_context"].get_start_method() == "spawn"


def test_ready_failure_prevents_refill_and_cancels_pending(monkeypatch):
    def finish_two(pending, *, return_when):
        first, failed, _ = list(pending)
        first.set_result(0)
        failed.set_exception(ValueError("queue failure"))
        return {first, failed}, set(pending) - {first, failed}

    monkeypatch.setattr(process_queue, "ProcessPoolExecutor", _FakePool)
    monkeypatch.setattr(process_queue, "wait", finish_two)
    with pytest.raises(ValueError, match="queue failure"):
        process_queue.ordered_process_map(_square, range(20000), max_workers=3)
    submitted = _FakePool.instances[-1].submitted
    assert len(submitted) == 3
    assert submitted[-1].cancelled()
