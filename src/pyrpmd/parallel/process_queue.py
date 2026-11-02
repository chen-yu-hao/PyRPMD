"""Bounded first-free-worker dispatch with ordered results.

This module uses only the standard library. Workers are processes, not threads;
scientific state and checkpoint ownership belong to the submitted task. At most
one future per worker is outstanding, so a slow earlier task cannot prevent
refilling other workers and large ensembles do not create an unbounded queue.
"""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing as mp


def ordered_process_map(function, tasks, *, max_workers, initializer=None, initargs=()):
    """Run importable tasks in a spawn pool, preserving input result order.

    Stop submitting when a task, serialization or worker-initialization error is
    observed. Cancel work that has not started, then wait for running workers to
    finish before propagating the exception. This is not recovery from a hung
    task or a node crash, and does not split an individual long trajectory.
    """
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    tasks = list(tasks)
    if not tasks:
        return []
    count = min(max_workers, len(tasks))
    results = [None] * len(tasks)
    pending = {}
    next_index = 0
    with ProcessPoolExecutor(
        max_workers=count, mp_context=mp.get_context("spawn"),
        initializer=initializer, initargs=initargs,
    ) as pool:
        try:
            while next_index < count:
                pending[pool.submit(function, tasks[next_index])] = next_index
                next_index += 1
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Inspect the whole completed set before refilling. An error
                # alongside a successful result must not dispatch another task.
                for future in sorted(done, key=pending.__getitem__):
                    results[pending.pop(future)] = future.result()
                while len(pending) < count and next_index < len(tasks):
                    pending[pool.submit(function, tasks[next_index])] = next_index
                    next_index += 1
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return results
