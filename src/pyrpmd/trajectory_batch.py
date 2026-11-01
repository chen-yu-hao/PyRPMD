"""Share independent PES frames without introducing another propagation loop.

Each task keeps its ordinary scalar dynamics, thermostat state, random stream,
stopping rules and checkpoint.  A rendezvous combines one outstanding force
request from each live task.  Only the physical PES frame axis is concatenated;
ring-polymer springs, biases and thermostats remain local to each task.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import numbers
import threading
import copy

import numpy as np

from .batched import validate_batch_potential
from .potentials.base import PESResult, coordinates, validate_result


_PROVIDER_KEY = "_trajectory_batch_potential"
# Each cooperative lane has one blocked worker thread. Bound this resource
# independently of the number of trajectories stored in array-based APIs.
MAX_TRAJECTORY_BATCH_SIZE = 256


def trajectory_batch_size(config):
    """Validate the execution-only batch size before dispatching any work."""
    parallel = config.get("parallel", {})
    size = parallel.get("trajectory_batch_size", 1)
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, numbers.Integral) or size < 1:
        raise ValueError("parallel.trajectory_batch_size must be an integer >= 1")
    if size > MAX_TRAJECTORY_BATCH_SIZE:
        raise ValueError(f"parallel.trajectory_batch_size must be <= {MAX_TRAJECTORY_BATCH_SIZE}")
    if size > 1:
        if parallel.get("bead_workers", 1) != 1:
            raise ValueError("trajectory batching requires parallel.bead_workers=1")
        if config.get("dynamics", {}).get("loop_backend", "auto") == "native":
            raise ValueError("trajectory batching uses Python PES orchestration; use loop_backend='auto' or 'python'")
        if config.get("nonadiabatic"):
            raise ValueError("trajectory batching requires a scalar, independent-frame PES")
        ensemble = config.get("ensemble", {})
        if (config.get("barostat") or config.get("dynamics", {}).get("barostat")
                or isinstance(ensemble, dict) and (
                    ensemble.get("barostat") or ensemble.get("type", ensemble.get("kind")) == "npt")):
            raise ValueError("trajectory PES batching currently requires a fixed cell")
    return int(size)


class _LanePotential:
    """A borrowed lane; never expose a native binding that bypasses the queue."""

    def __init__(self, batch, index, channel):
        self._batch, self._index, self._channel = batch, index, channel
        self.metadata = dict(getattr(batch.providers[channel], "metadata", {}) or {})
        self.metadata.update(trajectory_batch_safe=True, trajectory_batch_lane=index)

    def evaluate(self, q):
        return self._batch.evaluate(self._index, q, self._channel)

    def close(self):
        # The group owns the physical provider, including when a task fails.
        pass


class _Rendezvous:
    def __init__(self, providers, count):
        self.providers = providers
        self.condition = threading.Condition()
        self.active = set(range(count))
        self.pending = {}
        self.pes_calls = 0
        self.pes_frames = 0
        self.aborted = None

    def evaluate(self, index, q, channel):
        q = coordinates(q).copy()
        future = Future()
        with self.condition:
            if self.aborted is not None:
                raise self.aborted
            if index not in self.active or index in self.pending:
                raise RuntimeError("invalid concurrent trajectory PES request")
            self.pending[index] = (channel, q, future)
            self.condition.notify_all()
        return future.result()

    def finish(self, index):
        with self.condition:
            self.active.discard(index)
            self.condition.notify_all()

    def abort(self, error):
        """Release all waiters if thread creation or the coordinator fails."""
        with self.condition:
            self.aborted = error
            waiting = list(self.pending.values())
            self.pending.clear()
            self.active.clear()
            self.condition.notify_all()
        for _, _, future in waiting:
            future.set_exception(error)

    def serve(self):
        while True:
            with self.condition:
                # No timeout or arrival-order heuristic: a batch is reproducible
                # even when checkpoint I/O and completed restarts take different
                # amounts of wall time. Departing lanes always notify this wait.
                self.condition.wait_for(lambda: not self.active or self.active <= self.pending.keys())
                if not self.active:
                    return
                pending = [self.pending.pop(index) for index in sorted(self.active)]
            # Different force-splitting stages can rendezvous together. Batch
            # only equal physical providers, never mix fast and slow forces.
            channels = {}
            for channel, q, future in pending:
                channels.setdefault(channel, []).append((q, future))
            for channel, requests in channels.items():
                self._evaluate_channel(channel, requests)

    def _evaluate_channel(self, channel, requests):
        try:
            shapes = [q.shape for q, _ in requests]
            if any(shape[1:] != shapes[0][1:] for shape in shapes):
                raise ValueError("trajectory PES frames must have the same atom shape")
            q = np.concatenate([q for q, _ in requests], axis=0)
            self.pes_calls += 1
            self.pes_frames += len(q)
            result = validate_result(self.providers[channel].evaluate(q), q.shape, require_success=False)
            start = 0
            answers = []
            for shape in shapes:
                end = start + shape[0]
                # Copy before releasing a worker: many providers reuse their
                # result arrays on the next call, including native providers.
                answers.append(PESResult(
                    result.energy[start:end].copy(), result.forces[start:end].copy(),
                    result.status[start:end].copy(), result.messages[start:end],
                    None if result.virial is None else result.virial[start:end].copy(),
                ))
                start = end
        except BaseException as exc:
            for _, future in requests:
                future.set_exception(exc)
        else:
            for (_, future), answer in zip(requests, answers):
                future.set_result(answer)

def _provider_tree(potential, providers, path=()):
    """Retain each ring contraction locally and batch only physical leaves."""
    from .potentials.contraction import DifferencePotential, SplitPotential

    if type(potential) in (SplitPotential, DifferencePotential):
        methods = ("evaluate", "evaluate_fast", "evaluate_slow", "_coordinates")
        if any(name in potential.__dict__ for name in methods):
            raise ValueError("custom split-potential methods cannot use trajectory batching")
        names = ("fast", "slow") if type(potential) is SplitPotential else ("target", "reference")
        return (potential, {name: _provider_tree(getattr(potential, name), providers, path + (name,))
                            for name in names})
    validate_batch_potential(potential)
    providers[path] = potential
    return path


def _lane_provider(tree, batch, index):
    if isinstance(tree, tuple) and len(tree) == 2 and isinstance(tree[1], dict):
        original, children = tree
        # Copy the wrapper, retaining its read-only contraction. Its physical
        # components become lane providers; no ring is ever flattened with a
        # different trajectory before contraction or force lifting.
        provider = copy.copy(original)
        for name, child in children.items():
            setattr(provider, name, _lane_provider(child, batch, index))
        return provider
    return _LanePotential(batch, index, tree)


def validate_trajectory_potential(potential):
    """Validate physical leaves, including explicit local RPC wrappers."""
    _provider_tree(potential, {})


def run_trajectory_batch(worker, tasks, potential, *, stats=None):
    """Run existing task workers with shared, independent physical PES calls.

    ``worker(task)`` obtains its potential through :func:`task_provider` or
    :func:`make_trajectory_potential`.  Tasks must be independent and must not
    share mutable dynamics, RNGs or checkpoint paths.  All workers finish (and
    can save their accepted prefixes) before the first task exception is raised.
    Per-frame failures retain the provider's lane-local status and message;
    malformed whole-batch results fail all requests in that call, without retry.
    The caller retains ownership of ``potential``.
    """
    tasks = tuple(tasks)
    if not tasks:
        return []
    if len(tasks) > MAX_TRAJECTORY_BATCH_SIZE:
        raise ValueError(f"a cooperative trajectory batch supports at most {MAX_TRAJECTORY_BATCH_SIZE} lanes")
    providers = {}
    tree = _provider_tree(potential, providers)
    batch = _Rendezvous(providers, len(tasks))

    def run(index, task):
        try:
            return worker(dict(task, **{_PROVIDER_KEY: _lane_provider(tree, batch, index)}))
        finally:
            batch.finish(index)

    with ThreadPoolExecutor(max_workers=len(tasks), thread_name_prefix="pyrpmd-trajectory") as pool:
        try:
            futures = [pool.submit(run, index, task) for index, task in enumerate(tasks)]
            batch.serve()
        except BaseException as exc:
            # ThreadPoolExecutor waits for workers on exit. Release their PES
            # futures first, including if submitting a later worker failed.
            batch.abort(exc)
            raise
        if stats is not None:
            stats.update(pes_calls=batch.pes_calls, pes_frames=batch.pes_frames)
        # Read every result before propagating an exception. No thread or PES
        # request is abandoned when a different lane fails or finishes early.
        values, errors = [], []
        for future in futures:
            try:
                values.append(future.result())
            except BaseException as exc:
                values.append(None)
                errors.append(exc)
        if errors:
            raise errors[0]
        return values


def make_trajectory_potential(task, factory=None):
    """Acquire a scalar workflow provider or the lane assigned by its group."""
    if _PROVIDER_KEY in task:
        return task[_PROVIDER_KEY]
    if factory is None:
        from .parallel import make_task_potential as factory
    return factory(task["config"]["potential"], task["system"])


@contextmanager
def task_provider(task, factory=None):
    """Lease a task provider while preserving the ordinary provider cache."""
    if _PROVIDER_KEY in task:
        yield task[_PROVIDER_KEY]
    else:
        if factory is None:
            from .parallel import task_potential as factory
        config = task["config"]
        with factory(config["potential"], task["system"],
                     reuse=config.get("parallel", {}).get("reuse_potentials", True)) as provider:
            yield provider


def _run_group(group):
    from .parallel import task_potential
    from .parallel.execution import _potential_cache_token

    tasks = group["tasks"]
    first = tasks[0]
    identity = _potential_cache_token((first["config"]["potential"], first["system"]))
    for task in tasks:
        trajectory_batch_size(task["config"])
        if _potential_cache_token((task["config"]["potential"], task["system"])) != identity:
            raise ValueError("trajectory batch tasks must share the physical PES and system")
    config = first["config"]
    with task_potential(config["potential"], first["system"],
                        reuse=config.get("parallel", {}).get("reuse_potentials", True)) as provider:
        return run_trajectory_batch(group["worker"], tasks, provider)


def map_trajectory_tasks(worker, tasks, config, executor=None, *, batch_size=None):
    """Group scalar workers while retaining serial/process/MPI task transport."""
    from .parallel import run_tasks

    size = trajectory_batch_size(config) if batch_size is None else trajectory_batch_size(
        {**config, "parallel": {**config.get("parallel", {}), "trajectory_batch_size": batch_size}})
    mapper = executor.map if executor is not None else lambda function, jobs: run_tasks(
        function, jobs, config.get("parallel", {}))
    if size == 1:
        return mapper(worker, tasks)
    tasks = list(tasks)
    groups = [{"worker": worker, "tasks": tasks[start:start + size]}
              for start in range(0, len(tasks), size)]
    results = mapper(_run_group, groups)
    return None if results is None else [item for group in results for item in group]
