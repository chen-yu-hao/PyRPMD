"""Repeat selected pytest cases under reproducible, same-CPU contention.

Each case receives an independent parametrized fixture invocation. Failures are
retained; this is not a retry runner. Run from an environment with PyRPMD and its
test dependencies installed. All competing workers are reaped on exit.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import select
import subprocess
import sys
import time


DEFAULT_NODES = (
    "tests/test_h3_p256_parent_campaign.py::"
    "test_worker_failure_stops_admissions_and_drains_all_owned_processes",
    "tests/test_h3_p256_parent_campaign.py::"
    "test_successful_worker_can_be_replaced_before_another_worker_fails",
    "tests/test_h3_quantum_isolated_continuation.py::"
    "test_independent_worker_kernel_caps_survive_blocked_owner_alarm",
)


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=positive_integer, default=100)
    parser.add_argument("--load-workers", type=positive_integer, default=3)
    parser.add_argument("--cpu", type=int, help="one CPU in the current affinity mask")
    parser.add_argument("--junitxml", required=True, help="path for every individual result")
    parser.add_argument("nodes", nargs="*", help="pytest node IDs; defaults to the four T21 cases")
    args = parser.parse_args(argv)
    if sys.platform != "linux" or not hasattr(os, "sched_getaffinity"):
        parser.error("these process-control tests require Linux CPU affinity")
    affinity = os.sched_getaffinity(0)
    cpu = min(affinity) if args.cpu is None else args.cpu
    if cpu not in affinity:
        parser.error(f"CPU {cpu} is outside the allowed affinity {sorted(affinity)}")

    import pytest

    class RepeatEachTest:
        @pytest.fixture(autouse=True, params=range(1, args.repetitions + 1),
                        ids=lambda number: f"run-{number:03d}")
        def stress_repetition(self, request):
            return request.param

    junitxml = str(Path(args.junitxml).resolve())
    previous_directory = Path.cwd()
    workers = []
    started = time.monotonic()
    try:
        os.chdir(Path(__file__).resolve().parents[2])
        os.sched_setaffinity(0, {cpu})
        for _ in range(args.load_workers):
            worker = subprocess.Popen(
                [sys.executable, "-c", "print('ready', flush=True)\nwhile True: pass\n"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            workers.append(worker)
            if not select.select([worker.stdout], [], [], 10)[0]:
                raise RuntimeError("competing CPU worker did not start")
            if worker.stdout.readline() != b"ready\n":
                raise RuntimeError("competing CPU worker did not acknowledge startup")
        print(f"LOAD: affinity={[cpu]}, busy_workers={len(workers)}, "
              f"repetitions={args.repetitions}", flush=True)
        result = pytest.main(
            ["-q", "--tb=short", "--junitxml=" + junitxml, *(args.nodes or DEFAULT_NODES)],
            plugins=[RepeatEachTest()])
        if not all(worker.poll() is None for worker in workers):
            raise RuntimeError("competing CPU load ended before the tests completed")
        return int(result)
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
            worker.stdout.close()
            worker.stderr.close()
        os.sched_setaffinity(0, affinity)
        os.chdir(previous_directory)
        print(f"LOAD cleanup: {len(workers)} workers reaped; "
              f"elapsed={time.monotonic() - started:.3f}s", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
