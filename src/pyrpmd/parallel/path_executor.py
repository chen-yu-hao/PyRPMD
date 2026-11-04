"""Persistent local task execution for stateful path-sampling coordinators.

The caller owns admission, ordering and scientific state.  Futures may finish
in any order; their completion never mutates a coordinator's path bank.  This
uses the same provider leases, worker cleanup and BLAS limits as ``run_tasks``.
"""

from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import ExitStack
import multiprocessing as mp

from .execution import (
    _blas_thread_limit, _context, _optional_world, _process_dynamic_init,
    execution_plan,
)


class PathTaskExecutor:
    """Submit CPU path moves without a batch barrier.

    GPU/bead pools and MPI are deliberately rejected: the existing collective
    executor cannot safely admit an evolving set of individually locked paths.
    A one-worker configuration executes in process and returns ordinary futures.
    """

    def __init__(self, config=None, *, workers=None):
        config = dict(config or {})
        if workers is not None:
            if "trajectory_workers" in config and config["trajectory_workers"] != workers:
                raise ValueError("workers and parallel.trajectory_workers disagree")
            config["trajectory_workers"] = workers
        config.setdefault("scheduling", "dynamic")
        self.plan = execution_plan(config)
        if (self.plan.backend not in ("serial", "process")
                or self.plan.mode != "serial" or self.plan.bead_workers != 1
                or self.plan.pes_ranks != 1 or self.plan.scheduling != "dynamic"):
            raise ValueError("infinite RETIS requires local dynamic CPU path workers")
        self._resources = None
        self._pool = None

    def __enter__(self):
        if self._resources is not None:
            raise RuntimeError("path executor is already open")
        world = _optional_world()
        if world is not None and world.Get_size() > 1:
            raise ValueError("infinite RETIS does not support a multi-rank MPI launch")
        resources = ExitStack()
        try:
            if self.plan.trajectory_workers > 1:
                self._pool = ProcessPoolExecutor(
                    max_workers=self.plan.trajectory_workers,
                    mp_context=mp.get_context("spawn"),
                    initializer=_process_dynamic_init, initargs=(self.plan,),
                )
                resources.callback(self._pool.shutdown, wait=True, cancel_futures=True)
            else:
                resources.enter_context(_blas_thread_limit())
                resources.enter_context(_context(self.plan, 0))
        except BaseException:
            resources.close()
            self._pool = None
            raise
        self._resources = resources
        return self

    def submit(self, function, task):
        if self._resources is None:
            raise RuntimeError("path executor is not open")
        if self._pool is not None:
            return self._pool.submit(function, task)
        future = Future()
        try:
            future.set_result(function(task))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def __exit__(self, *exc):
        try:
            return self._resources.__exit__(*exc)
        finally:
            self._resources = self._pool = None
