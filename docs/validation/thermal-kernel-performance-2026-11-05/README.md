# T49 PR #115 thread-count repair and default-build validation

This record addresses the [maintainer's review](https://github.com/chen-yu-hao/PyRPMD/pull/115#issuecomment-6653360884).
Runtime repair: `1d9936d`, rebased onto `main@67415fd`. Reference:
`main@801953d`; the thermal, composition and Dynamics sources are unchanged
between that reference and the current main. Measurements were collected on
2026-11-05, Asia/Shanghai.

**The thread-count defect is fixed.** `threads=1` creates no OpenMP workers,
including when `OMP_NUM_THREADS` is absent. The six existing benchmark gates
(which use the optional Philox stream for throughput/scaling) pass on this
host's default build. The unchanged default MT stream still reaches only
36.2%/47.3% of matched NVE throughput at N=512/4096. The public Dynamics
four-thread speedup is 2.381x, below 2.5x; the direct native engine reaches
2.974x. These limits remain visible for the maintainer's acceptance decision.
No general default-stream or cross-machine performance acceptance is claimed.

## Repair and actual thread counts

`thermal::Config` now carries a positive thread count, defaulting to one.
Every thermal parallel region uses `num_threads(c.threads)`, and scratch
storage follows that executor's count. No process-global OpenMP setting is
changed. Both the full native executor and elementary thermostat helpers
inherit `Dynamics.integrator.threads`. Direct NativeEnsemble construction and
its native input schema accept `threads=...`. Existing Dynamics workflows
inherit `parallel.threads_per_rank`; the native PIMD workflow retains its
one-thread-per-trajectory restriction.

Thread counts are excluded from the physics fingerprint, RNG ABI and saved
state. Invalid counts and multi-thread requests against a serial build are
rejected before native executor construction or force evaluation.
`THERMAL_KERNEL_API=3` rejects
old extensions that would ignore the new option. The default random stream
and all thermal arithmetic are unchanged by this repair.

`tests/test_native_thermal_threads.py` reads `/proc/self/task` in isolated
processes after real propagation. It covers the direct API, full native
loop, Python loop with native thermostat helpers, and both PIMD workflow
engines through the real input resolver. Explicit two-thread requests are
checked against `OMP_NUM_THREADS=1`; one-thread requests are checked against
both unset and four-thread environments. A coordinated four-process test
checks that each trajectory creates only one OS thread, with no timing
assertions or mock executor. Actual process-count tests require Linux;
portable tests still cover validation, identity and stochastic replay.

Before the repair, the four-process regression failed: each worker had eight
threads despite requesting one. The reproducer capped the failing runtime
with `OMP_THREAD_LIMIT=8`; see `threads-before.log`. After repair all four
workers have one thread. Performance measurements below do not use that cap
or the test helper's affinity restriction.

## Default-build protocol

Host: `cluster`, Intel Xeon Platinum 8176, 2 sockets, 28 cores/socket,
112 logical CPUs. Python 3.11.5, NumPy 1.26.4, GCC 12.2.0.
Both checkouts were rebuilt with the repository defaults:
`-O3 -std=c++17 -fopenmp`, without `-march=native`, an explicit FMA flag,
or a contraction override. Full compiler invocations are retained in
`build-baseline-default.log` and `build-openmp.log`.

All default-build performance runs use the full inherited affinity 0–111;
no `taskset` or OpenMP binding is applied. `OMP_NUM_THREADS`, `OMP_PROC_BIND`,
`OMP_PLACES`, `OMP_DYNAMIC` and `OMP_THREAD_LIMIT` are unset.
`OPENBLAS_NUM_THREADS=1`; the requested thermal thread count is an explicit
API argument recorded in each JSON. Baseline and candidate runs execute
sequentially on the same host. Correctness suites were not run concurrently
with these measurements. This remains a shared host; raw repeats and separate
NVE controls are retained rather than assuming identical instantaneous load.

`thermal_throughput.py` uses P=32 (plus the N=3, P=1 control), masses 2,
beta 2, harmonic k=0.5, dt=0.01, native built-in binding, and Cayley BCB/BCOCB.
Construction/warm-up are excluded. Each public call advances 64 steps and
records centroid/energy data. Each entry retains seven timing repeats;
steps/s is based on their median. Values here are atomic units except timing.

## Same-host throughput and limits

All columns below are single-thread public Dynamics steps/s. Each stream's
PILE-L/NVE ratio uses its own adjacent NVE control, measured in the same run.

| N | P | Main NVE | Main PILE-L | MT NVE | MT PILE-L | Philox NVE | Philox PILE-L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 1 | 266911.95 | 60000.88 | 346028.78 | 155559.10 | 357359.03 | 161570.51 |
| 3 | 32 | 59152.07 | 14068.02 | 71485.94 | 26695.11 | 65824.47 | 22251.63 |
| 64 | 32 | 3768.62 | 904.35 | 4098.08 | 1875.58 | 4331.13 | 2155.12 |
| 512 | 32 | 441.65 | 114.85 | 509.02 | 184.41 | 521.58 | 280.19 |
| 4096 | 32 | 55.25 | 13.31 | 57.70 | 27.31 | 61.79 | 32.28 |

| Script gate | Measured | Target | Result |
| --- | --- | --- | --- |
| Optional Philox PILE-L/NVE, N=512 | 53.72% | >=50% | pass |
| Optional Philox PILE-L/NVE, N=4096 | 52.24% | >=50% | pass |
| Direct NativeEnsemble, Philox, 4/1 threads | 2.974x | >=2.5x | pass |
| Default MT two-step non-native/native time, N=4096 | 9.295% | <=10% | pass |
| Small default MT PILE-L/main, N=3, P=1 | 2.593x | >=0.98x | pass |
| Small default MT PILE-L/main, N=3, P=32 | 1.898x | >=0.98x | pass |

`performance-gates.json` retains all six checks, the two default MT ratios
(36.23% and 47.33%), and the public Philox speedup (2.381x). The latter three
do not reach 50%/2.5x. The existing script's `all_performance_gates_passed`
field refers only to the six explicitly listed checks, not to those additional
measurements or an approval to merge.

Direct native Philox runs use eight-step calls and rise from 37.648 to
111.974 steps/s at four threads. Public Dynamics uses 64-step calls and rises
from 32.281 to 76.875 steps/s. The protocols differ, so these are separate
scaling measurements. Short-call profiling counts all time outside the C++
run as overhead: the default MT call falls from 650.14 ms on main to 86.38 ms
(79.04 ms native); optional Philox takes 67.99 ms (60.13 ms native), with
13.06% non-native/native overhead. The 10% gate above uses the unchanged default
stream, as in the original validation.

## Four-process review case

`thermal_processes.py` reproduces N=512, P=32, PILE-L, `threads=1`, with
`OMP_NUM_THREADS` unset. Spawned workers synchronize before each of seven
64-step repeats. These are per-worker rates, not summed throughput or timings
from the serial matrix above. All measured processes start and end with one
OS thread.

| Build | Workers | Per-worker steps/s |
| --- | --- | --- |
| Main | 1 | 129.58 |
| Candidate | 1 | 232.83 |
| Main | 4 | 126.70, 127.11, 112.43, 108.90 |
| Candidate | 4 | 229.28, 229.58, 207.52, 207.16 |

The JSON files contain all repeats, process thread counts, source identity,
compiler, affinity and thread environment. They establish the repaired
one-thread-per-worker behavior without extrapolating the maintainer's
four-core-machine timings to this different host.

## Build, replay and regression evidence

- OpenMP default build: `480 passed, 1 skipped in 49.28s`.
- No OpenMP: `477 passed, 4 skipped in 46.68s`.
- FMA (`-march=native -mfma -ffp-contract=fast`):
  `480 passed, 1 skipped in 46.37s`.
- Pure Python wheel, verified to contain no native shared libraries:
  `40 passed, 13 skipped, 18 subtests passed in 9.21s`.
- Both streams retain complete state equality at requested 1/2/4/8 threads
  in OpenMP and FMA builds, even with `OMP_NUM_THREADS=1`. Serial builds test
  one thread and reject larger requests.
- Newly generated default-build main checkpoints: 118/118 exactly replayed
  at both requested 1 and 4 threads; `legacy-reference.json` and
  `legacy-replay-{1,4}.json`. Fingerprints remain unchanged across thread counts.
- Strict Sphinx build passed (`docs-build.log`).
- Full non-campaign suite (`tests-full.log`), exact original summary:

  ```text
  9470 passed, 236 skipped, 1 deselected, 47 warnings, 410 subtests passed in 1758.83s (0:29:18)
  ```

Compiler and test logs are retained separately for all three native builds.
The full suite uses OMP/BLAS one thread and affinity 32–47, after the unbound
performance runs, outside the sandbox to allow the existing i-PI socket tests.
The earlier tuned-build, detailed component, synthetic distribution and FMA
main-checkpoint evidence remains in the 2026-11-04 directory; it is not
presented as a new default-build measurement.

## Reproduction

Use one interpreter/environment and independently build both checkouts. Run
the current branch's committed benchmark scripts against each source path:

```bash
unset CXXFLAGS CFLAGS PYRPMD_NO_OPENMP
python setup.py build_ext --inplace --force
# Repeat that build from the baseline checkout.
PYRPMD_THERMAL_PYTHON=python PYRPMD_THERMAL_VALIDATION=/tmp/thermal-builds \
  bash benchmarks/efficiency/validate_thermal_builds.sh
unset OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES OMP_DYNAMIC OMP_THREAD_LIMIT
export OPENBLAS_NUM_THREADS=1
PYTHONPATH=/absolute/baseline/src python benchmarks/efficiency/thermal_throughput.py \
  --baseline --threads 1 --profile --repeats 7 --seconds .35 --output baseline-dynamics.json
python benchmarks/efficiency/thermal_throughput.py \
  --baseline --threads 1 --profile --repeats 7 --seconds .35 --output candidate-mt-dynamics.json
python benchmarks/efficiency/thermal_throughput.py \
  --baseline --threads 1 --rng philox-v1 --profile --repeats 7 --seconds .35 --output candidate-philox-dynamics.json
for threads in 1 4; do
  python benchmarks/efficiency/thermal_throughput.py --api native --atoms 4096 --beads 32 \
    --thermostats pile-l --rng philox-v1 --threads "$threads" --block-steps 8 \
    --repeats 7 --seconds .35 --output candidate-philox-native-$threads.json
done
python benchmarks/efficiency/thermal_throughput.py --atoms 4096 --beads 32 \
  --thermostats pile-l --rng philox-v1 --threads 4 --repeats 7 --seconds .35 \
  --output candidate-philox-dynamics-4.json
for workers in 1 4; do
  PYTHONPATH=/absolute/baseline/src python benchmarks/efficiency/thermal_processes.py \
    --workers "$workers" --output baseline-processes-$workers.json
  python benchmarks/efficiency/thermal_processes.py \
    --workers "$workers" --output candidate-processes-$workers.json
done
python benchmarks/efficiency/summarize_thermal.py /directory/containing/the/above/jsons
PYTHONPATH=/absolute/baseline/src python benchmarks/efficiency/thermal_contract.py \
  --write /tmp/reference --output legacy-reference.json
for threads in 1 4; do
  python benchmarks/efficiency/thermal_contract.py --compare /tmp/reference \
    --threads "$threads" --output legacy-replay-$threads.json
done
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q -m "not campaign"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

## Coordination and disclosure

The earlier `_builtin_native.py` cache scope extension was explicitly
authorized by the user; this repair makes no further changes to that file.
The two minimal forwarding changes in `engine.py` implement the review's
request to honor the already-public Dynamics thread setting. The public
default RNG remains `mt19937_64`. No GPU, external PES scaling or real-material
physical-convergence claim is made. T27 remains gated on the user's explicit
confirmation that T49 has merged.

OpenAI Codex implemented the repair and ran this validation in response to
the maintainer's posted review. The maintainer's final acceptance has not been
asserted. This follows `docs/publication-readiness.md`.
