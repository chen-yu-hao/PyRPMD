# T49 thermal kernel validation

Date: 2026-11-04 (Europe/Berlin). Baseline: `main@801953d`; implementation branch:
`agent/t49-thermal-kernel-performance`, based on last verified `main@cb28829`.
The final SSH refresh confirmed `main@cb28829`; the branch is current with this baseline.

**Performance acceptance: 6/6 gates passed.** The user-authorized harmonic
binding cache and the fused preparation of initial-energy terms reduce the
measured two-step overhead to 9.36% of native time (maximum 10%). The final
full non-campaign suite passes 9399 tests; all three builds pass 456 focused
tests each. No real-material physical-convergence claim is made.

## Hardware and build

All before/after measurements use host `cluster`, Intel Xeon Platinum 8176
(2 sockets, 28 physical cores/socket, 112 logical CPUs), with benchmark affinity
restricted to CPUs 24–27. `OMP_PROC_BIND=true`, `OMP_PLACES=cores`, and
`OPENBLAS_NUM_THREADS=1`; the JSONs record the requested OpenMP thread count.
Background correctness tests use CPUs 32–47 on the other socket. This is a
shared host, so raw repeats and the unchanged NVE controls are retained.

Python 3.11.5, NumPy 1.26.4, SciPy 1.17.1 and GCC 12.2.0. Matched performance
builds use `-O3 -std=c++17 -fopenmp -march=native -ffp-contract=off` in both
checkouts. FMA validation uses `-march=native -mfma -ffp-contract=fast`; the
OpenMP-disabled build omits `-fopenmp`. Source and benchmark hashes, interpreter,
compiler text, affinity and all individual wall times are in each JSON.

## Public Dynamics throughput

Harmonic potential, masses 2 electron masses, beta 2 hartree^-1, k=0.5,
dt=0.01 atomic time, Cayley BCB/BCOCB, native built-in binding. Each timed call
advances 64 steps, records centroid/energy data, and checks endpoint energy.
Warm-up and construction are excluded. Values are steps/s based on seven raw
wall-time repeats. The last column uses the candidate's own NVE control.

| N | P | Baseline NVE | Baseline PILE-L | Candidate MT PILE-L | Candidate Philox PILE-L | Philox / NVE |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 1 | 293,360.56 | 57,622.84 | 161,531.91 | 135,583.80 | 0.483 |
| 3 | 32 | 55,060.65 | 16,658.42 | 24,056.51 | 17,272.42 | 0.301 |
| 64 | 32 | 3,568.68 | 1,058.69 | 1,832.24 | 1,836.54 | 0.550 |
| 512 | 32 | 440.92 | 127.31 | 235.08 | 253.49 | 0.610 |
| 4096 | 32 | 54.35 | 15.32 | 29.01 | 31.73 | 0.551 |

`baseline-dynamics.json`, `candidate-mt-dynamics.json` and
`candidate-philox-dynamics.json` contain every raw repeat. The short-call
profile uses repeated `propagate(state, 2)` calls; native time is the cumulative
C++ run call, and all remaining time, including snapshots, validation, copies
and Python reductions, is counted as overhead.

## Scaling and component profile

For N=4096, P=32, direct `NativeEnsemble` Philox PILE-L rises from
35.225 to 91.897 steps/s at four threads
(2.609x).
The complete `Dynamics.propagate` path rises from 31.731 to 89.353 steps/s
(2.816x). Both exceed the 2.5x target.
The two public single-thread PILE-L/NVE ratios are 0.610 and 0.551 at
N=512 and N=4096. Legacy small-system PILE-L improves by 2.803x (P=1)
and 1.444x (P=32).

`summarize_thermal.py` checks matching hardware/software/affinity, produces
`throughput.csv`, and writes every pass/fail gate to `performance-gates.json`.
The complete direct-engine matrix covers N=64/1000/4096, P=8/32/128,
PILE-L, PILE-G, GLE, synthetic PIGLET, NHC, BZP and NVE controls. Each JSON
has `complete=true`; incomplete development runs are not mixed into these data.

The following isolated probes are milliseconds per operation at N=4096, P=32.
They use `thermal_kernels.cpp` with the same compiler flags. The generic four
transforms are measured separately from the actual fused middle map; the
Philox middle map uses two packed transforms. Timings overlap and are not
summed. No claim of a memory-bandwidth bottleneck is needed for the passed
scaling gate.

| Operation | MT, 1 thread | Philox, 1 thread | Philox, 4 threads |
| --- | --- | --- | --- |
| Four generic transforms | 14.450 | 14.344 | 7.011 |
| PES | 1.898 | 1.820 | 0.907 |
| Two kicks | 0.487 | 0.519 | 0.126 |
| Noise generation | 7.733 | 13.359 | 3.563 |
| Noise application | 0.241 | 0.256 | 0.065 |
| Bath including noise | 12.350 | 27.775 | 7.582 |
| Observation | 5.954 | 3.520 | 1.012 |
| State copy | 1.603 | 1.608 | 1.583 |
| Actual middle map | 27.167 | 29.925 | 8.710 |
| Complete native step | 32.075 | 31.474 | 9.767 |

STREAM-style triad bandwidth (24 logical bytes/element, 192 MiB working set): 13.40 GB/s, 12.93 GB/s, 42.57 GB/s in the same column order. These are bandwidth probes, not trajectory bandwidth.

## Correctness and numerical contracts

- Legacy checkpoints: 118/118 exactly replayed at 1 and 4 OpenMP threads.
  Both Cayley and exact normal-mode integrators, every native thermostat,
  small/odd/large rings, and BZP/MTK/RGB/SCR pressure control are covered.
- Matched FMA baseline checkpoints: 118/118 exactly replayed. Legacy OU, NHC
  and piston expressions retain their original association because extracting
  coefficients can change FMA rounding despite identical RNG states.
- Philox: tests compare complete q/p/auxiliary/cell/RNG state at 1/2/4/8 threads,
  checkpoint replay into another initial seed, nested/owned rollback tokens,
  failed-block rollback, stale-extension rejection and a real TOML PIMD workflow.
- Harmonic finite-P energy and normal-mode distribution checks are recorded in
  `statistics-philox.json`. These are synthetic implementation controls;
  `physical_converged=false` is retained.
- The complete matrix measures canonical synthetic GLE/PIGLET-shaped parameters;
  it does not validate calibrated PIGLET or chemical dynamics. No i-PI or GPU
  performance comparison is claimed.

## Build and test evidence

- `build-openmp.log`, `build-no-openmp.log`, `build-fma.log`: full compiler logs.
- `tests-openmp.log`, `tests-no-openmp.log`, `tests-fma.log`: 456 passed in each build.
- `build-no-native.log`, `tests-no-native.log`: a pure Python wheel with no native
  shared libraries; `40 passed, 13 skipped, 18 subtests passed in 8.97s`.
- `tests-full.log`: final full-suite output. Original summary line:

  ```text
  9399 passed, 235 skipped, 1 deselected, 47 warnings, 410 subtests passed in 1718.78s (0:28:38)
  ```

  Optional dependencies including Torch, ASE and PySCF were absent in this
  isolated environment; detailed skip reasons are retained in the raw log.
- `docs-build.log`: Sphinx with `-W --keep-going`.

The no-native test for a forced native loop originally required the error for
an unsupported PES even when no native kernel was installed. This same failure
was reproduced on `801953d`; its assertion now selects the actual capability
reason and still verifies that q/p/clock/RNG remain unchanged.

## Reproduction

Create an isolated Python environment with the dependencies in `pyproject.toml`,
then run the same committed scripts against the baseline and candidate:

```bash
python -m pip install -e . --no-build-isolation
PYRPMD_THERMAL_PYTHON=python bash benchmarks/efficiency/validate_thermal_builds.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=true OMP_PLACES=cores \
  CXXFLAGS='-march=native -ffp-contract=off' taskset -c 24-27 python \
  benchmarks/efficiency/thermal_throughput.py --baseline --repeats 7 --seconds .35 \
  --profile --output candidate-mt-dynamics.json
# Add --rng philox-v1 for the versioned stream. Use the same file with
# PYTHONPATH=/absolute/baseline/src for the independently built baseline.
for t in 1 2 4; do
  OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=true OMP_PLACES=cores \
    CXXFLAGS='-march=native -ffp-contract=off' taskset -c 24-27 python \
    benchmarks/efficiency/thermal_throughput.py --api native --rng philox-v1 \
    --block-steps 8 --repeats 3 --seconds .15 --output native-$t.json
done
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q -m "not campaign"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

`thermal_contract.py --write /tmp/reference --output reference.json` generates
baseline checkpoints; `--compare /tmp/reference --output replay.json` verifies
exact arrays, metadata and digest in the candidate. Repeat with matched FMA
builds for the FMA check. NPZs are not committed; reports contain their SHA256s.
`thermal_statistics.py` produces the harmonic controls. `thermal_kernels.cpp`
provides isolated component timings and STREAM-style bandwidth; component
measurements overlap and are not added to claim whole-run throughput.

## Coordination and retained limits

The user explicitly authorized the G2 scope extension for
`src/pyrpmd/potentials/_builtin_native.py`. The applied cache compares detached
original parameter arrays and preserves the prior expanded key, binding identity
and validation whenever parameters or provider capabilities change. No further
scope authorization is pending. Initial energy terms are prepared in one native
pass, then reduced by NumPy in the original order; numerical/restart tests cover
OpenMP, no-OpenMP and FMA builds.

The user authorized SSH publication. GitHub SSH authentication was verified as
`chen-yu-hao`, and the final main refresh used the SSH remote.

Only final script-produced records belong in this directory. Early probes,
interrupted suites, provisional Philox mappings and pre-fix FMA results are
retained separately in `/tmp/t49-development-results`, `/tmp/t49-before-authorized-cache`
and `/tmp/t49-authorized-cache-pass1` and excluded from final acceptance. The prior
incomplete record is also retained in Git at `6451297`.
No trajectories, compiled libraries, checkpoint NPZs or encoded source payloads
are included. `artifact-inventory.json` lists the included files and byte sizes;
the record is below the 1 MB card limit.
