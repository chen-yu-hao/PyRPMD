# T21: deterministic H3 process-control regressions

## Scope and baseline

This work starts from `main` commit
`f87705115c7496821b283a0966c991abd541231c` (the T01–T15 integration).
Only the two assigned test modules and the new validation script/report change.
The campaign scheduler, worker-limit installation, scientific implementation,
run identities and checkpoint formats are unchanged.

The unmodified two-module baseline was:

```text
137 passed in 15.54s
```

## Root causes

### Worker failure versus successful completion

`fake_commands(..., fail_seed=1, delay=.1)` started two separate Python
interpreters which each slept for `.1` seconds. That did not establish which
process would exit first. If seed 2 completed successfully while seed 1 was
still running, admitting seed 3 was correct. The test nevertheless asserted
that only seeds 1 and 2 could have been launched.

`dispatch_jobs()` checks its active children before admissions and again after
constructing the next command. The reported `[1, 2, 3]` outcome did not establish
that it had admitted a task after observing a failed worker.

The repaired test uses real inert children blocked in `signal.pause()`. Once
both initial workers have been registered, its event hook sends `SIGKILL` to
seed 1. `waitid(WEXITED | WNOWAIT)` confirms the exact terminal signal without
reaping the child on the scheduler's behalf. The original exact launched and
unstarted seed assertions remain. Additional assertions require the exact
worker error, no verified results, a released lease/removed lock and all owned
PIDs already reaped.

A complementary regression permits only seed 2 to finish successfully. It
waits for verification and the resulting admission of seed 3 before killing
seed 1. It requires exactly `[1, 2, 3]`, only seed 2 verified, and seeds 4–8
unstarted. The existing `test_exited_failed_worker_blocks_the_next_free_slot_admission`
continues to cover a failure observed while a free slot remains.

Callback exceptions are caught by the scheduler, so both new arrangements
assert the complete, single expected worker error. An assertion failure inside
an event hook cannot masquerade as successful failure handling. The campaign
watchdog remains five seconds.

### CPU budget versus wall-clock budget

The old CPU case simultaneously installed a one-second `RLIMIT_CPU` and a
three-second `ITIMER_REAL`, with a four-second owner deadline and six-second
parent timeout. One second of actual CPU execution can take more than three
wall-clock seconds under contention. Correct enforcement then terminates the
worker with `SIGALRM`, contradicting the test's required `SIGKILL`.

This was reproduced on the unmodified tests with three busy processes sharing
the test's single logical CPU. All three CPU-case invocations returned `-14`
(`SIGALRM`) rather than `-9` (`SIGKILL`):

```text
3 failed, 6 passed in 21.49s
LOAD cleanup: 3 workers reaped; elapsed=21.914s
```

The CPU case still uses the actual one-second CPU hard/soft limit and requires
exactly `SIGKILL`. Its worker wall cap and owner deadline are now both 120
seconds, beyond the parent's 60-second watchdog. This establishes a budget
ordering for every successful test invocation, rather than assuming a CPU
second fits into a particular smaller wall interval. A parent timeout raises
`TimeoutExpired`; its cleanup kill cannot pass the signal assertion. The child
also reports its real CPU limits, unblocked alarm mask and active real timer,
all of which the parent verifies. No installed production timer is cancelled.

The wall case retains the `.12`-second timer and exact `SIGALRM` assertion but
waits in `signal.pause()` rather than spinning. Its signal remains direct
evidence that the inherited blocked alarm was unblocked. It does not require
a receipt to be printed before a legitimately immediate alarm.

## Reproducible load protocol

Install the test extra and native extensions, then run from the repository:

```bash
python -m pip install -e '.[test]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/h3/stress_process_tests.py \
    --repetitions 100 --load-workers 3 --cpu 0 \
    --junitxml /tmp/t21-stress-100.xml
```

Omit `--cpu` to select the first CPU in the current affinity mask. The driver
pins itself, the tests, their children and three continuously busy processes
to the same CPU. Each busy process acknowledges startup before collection;
all must still be alive after pytest completes and are then terminated/reaped.
OpenMP and OpenBLAS are each restricted to one thread.

An autouse parametrized pytest fixture produces 100 independently executed
invocations per case, with distinct `run-001` through `run-100` IDs and ordinary
fixture teardown. Every failure is retained in pytest's exit code and JUnit
output. There are no failed-test reruns, accepted alternative signals or
weakened seed assertions. The defaults also cover the complementary admission
test and the wall-time parameter, for 400 total cases.

For the baseline reproduction, use the same driver against the unmodified
test modules, explicitly select the two original function IDs (the new
complementary case is absent on `main`), and set `--repetitions 3`.

## Execution environment

- Linux `6.18.57`, x86_64, glibc 2.39; nine visible logical CPUs, affinity 0–8.
  CPU model and host CPU quota are not exposed by this managed environment.
- Pressure validation uses CPU 0 and three additional busy processes. Full
  suite validation uses CPUs 1–8, so its workers do not compete with those
  pressure processes on CPU 0.
- Python 3.12.14; pytest 8.3.5; NumPy 2.3.5; SciPy 1.17.0; threadpoolctl 3.6.0.
- All three C++17/OpenMP extensions (`_native`, `_thermal`, `_electronic`) were
  built and imported successfully. The workspace installation used
  `python -m pip install -e '.[test]' --no-build-isolation --no-index` after
  preparing the dependencies.
- h5py was unavailable in the initial runtime and PyPI was outside its network
  policy. Verification used the official upstream `h5py 3.17.0.dev0` wheel
  with HDF5 2.2.0, from commit
  `f5918a1cf81cfac6656c77083f0d255fd985f625`,
  [Actions run 41408262053](https://github.com/h5py/h5py/actions/runs/41408262053),
  artifact `13125546970`. Its ZIP SHA256 was verified as
  `b52db0124f665271e65f751549c0368ed6021de1527c0172bf1927e41267bb21`.
  This is explicitly a pre-release dependency; the two process-control
  regressions do not depend on HDF5.

## Results

The same-CPU acceptance run completed with no failures, errors or skips:

| Case | Invocations | Passed | Failures/errors/skips | Observed elapsed seconds per case |
| --- | ---: | ---: | ---: | ---: |
| Failure stops admissions and drains all owned processes | 100 | 100 | 0 / 0 / 0 | 0.012–0.135 |
| Success admits a replacement before another worker fails | 100 | 100 | 0 / 0 / 0 | 0.144–0.256 |
| Independent wall limit with an inherited blocked alarm | 100 | 100 | 0 / 0 / 0 | 0.337–0.538 |
| Independent CPU limit with an inherited blocked alarm | 100 | 100 | 0 / 0 / 0 | 4.188–4.859 |

Counts and elapsed ranges were read from all 400 JUnit test cases. Every group
contains all 100 requested invocations; none was retried or omitted. In
particular, every CPU invocation required more than four wall seconds, beyond
the old three-second alarm that caused the baseline failures.

Raw acceptance output:

```text
LOAD: affinity=[0], busy_workers=3, repetitions=100
........................................................................ [ 18%]
........................................................................ [ 36%]
........................................................................ [ 54%]
........................................................................ [ 72%]
........................................................................ [ 90%]
........................................                                 [100%]
- generated xml file: /workspace/scratch/ab03f1d2d0e3/t21-work/logs/stress-100.xml -
400 passed in 538.34s (0:08:58)
LOAD cleanup: 3 workers reaped; elapsed=538.695s
```

These are verification elapsed times under deliberately competing load, not
simulation speedup measurements.

### Complete suite

The full suite ran with native extensions enabled, one OpenMP/BLAS thread,
using CPUs 1–8 while the pressure run occupied CPU 0:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 1-8 python -m pytest -q \
  --junitxml /tmp/t21-full-suite.xml
```

Raw final result (exit status 1):

```text
FAILED tests/test_ipi_clients.py::test_existing_unix_address_is_not_removed_on_bind_failure
1 failed, 10071 passed, 340 skipped, 49 warnings in 1358.40s (0:22:38)
```

The single failure is the pre-existing environment restriction independently
reproduced below, not a new failure. Full-suite JUnit confirms that all **138**
cases in the two assigned modules passed (62 parent-campaign cases and 76
isolated-continuation cases). The extra passing case relative to the original
137-case baseline is the new success-before-failure regression. Existing
optional-dependency/platform skips are preserved; no test was newly skipped.
The full suite was not clean in this sandbox, and is not reported as clean.

### Environment-only full-suite failure

The managed sandbox rejects construction of `socket(AF_UNIX, SOCK_STREAM)`
with `PermissionError: [Errno 1] Operation not permitted`. The existing
`tests/test_ipi_clients.py::test_existing_unix_address_is_not_removed_on_bind_failure`
therefore cannot reach its bind/ownership assertion. Its test and implementation
are byte-identical to the starting commit and were not changed here.

To distinguish this from a regression, the original `src/`, the socket test,
`conftest.py` and `pyproject.toml` were exported from verified Git blobs at
`f87705115c7496821b283a0966c991abd541231c` into a separate baseline directory.
`PYTHONPATH` selected that original source directory; the imported socket module
path was checked. The isolated baseline command was:

```bash
PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q \
  tests/test_ipi_clients.py::test_existing_unix_address_is_not_removed_on_bind_failure
```

It failed identically at Python's `_socket.socket.__init__`, before any bind:

```text
E       PermissionError: [Errno 1] Operation not permitted
FAILED tests/test_ipi_clients.py::test_existing_unix_address_is_not_removed_on_bind_failure
1 failed in 0.39s
```

No sandbox bypass, new skip or modification to the socket code/test was made.

## Packaging and limits

`MANIFEST.in` already includes `benchmarks/h3/*.py` and
`benchmarks/h3/*.md`, so the new driver and this report enter source archives
without a manifest edit. No files outside the assigned test modules and new
H3 validation files were modified.

These are Linux process-control tests, as before. Finite repetition is evidence
for the stated environment/load; it cannot guarantee scheduling latency on
every possible host. A machine starved beyond the explicit watchdog still
fails visibly. This work makes no simulation-performance or physical-convergence
claim and requires no README or scientific-reference addition.
