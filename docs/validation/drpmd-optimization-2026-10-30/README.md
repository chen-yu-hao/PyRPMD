# dRPMD native outcomes and propagation optimization

This record compares the optimized implementation with pristine commit
`af90f20d05368fc910c493977cf29856300b5864`. The comparison preserves each
workload's physical parameters, classifier checks, persistence, output stride
and checkpoint cadence.

## Changes under test

- Exact built-in explicit connectivity classifiers and the benchmark's
  identity-aware H3 exchange classifier can perform outcome checks, candidate
  persistence and early termination inside the C++ block.
- Values close to strict bond, separation or outward-motion thresholds return
  to the original Python predicate at the original scheduled check. The
  conservative roundoff band selects the execution backend; it does not change
  a physical threshold.
- Ordinary native dRPMD blocks reuse endpoint forces. A block of L steps needs
  L+1 PES batches instead of 2L. The two half-kicks are unchanged; the cache ends
  at every block boundary. Bias, constraints and fragment projections retain
  their reference force schedule.
- Classifier-error rollback retains one pre-final q/p state and its exact clock
  instead of a full per-step bead-state history. Every accepted step still
  contributes to the energy-deviation statistic and the original output stride.
- Python classifiers share centroid/bond calculations where safe. Pilot channel
  discovery, custom callbacks, subclasses and overridden hooks remain Python.
  Older compiled extensions retain the original callback path.
- An unchanged terminal checkpoint is written once.

## Regression and package gates

| Gate | Result | Evidence |
| --- | --- | --- |
| Baseline targeted native/dRPMD/fused suites | 70 passed | [baseline-tests.txt](baseline-tests.txt) |
| Optimized complete pytest suite | 805 passed, 3 skipped, 146 subtests passed | [candidate-tests.txt](candidate-tests.txt) |
| Source and platform wheel build | Passed | [package-validation.json](package-validation.json) |
| Installed wheel exercised outside the checkout | Passed | [wheel-smoke.json](wheel-smoke.json) |
| Six same-machine BKMP2 comparisons | Exact arrays, status and stopping steps | [results.json](results.json) |
| Completed and interrupted checkpoint recovery | Exact for both checkouts | [results.json](results.json) |

The three pytest skips are one explicitly opt-in real NiAlH EAM potential check
and two optional PyTorch cases. The local environment did not have mpi4py; actual
multi-rank MPI smoke programs are delegated to the repository's existing CI.
The new classifier tests cover both H3 product identities, generic product
channels, nonreactive and unresolved states, strict thresholds, persistence
across blocks/checkpoints, accepted failure prefixes, custom callback state
changes and Unicode labels including embedded NUL.

The wheel smoke imports the independently installed package, checks all three
integrators, runs and resumes the existing network/FES/metadynamics/PCET examples,
and runs dRPMD through the alternative-integrator Python control paths. Separate
source tests cover the native outcome API and cached-force call counts.

## Timings

Each entry is the median of five measured runs after one warmup, with randomized
baseline/candidate order. Both checkouts were force-built with the same Python
and command. They used the same pinned real BKMP2 shared library. Python was
3.12.14, NumPy 2.3.5, Linux x86_64; workers were pinned to CPU 0 and OpenMP,
OpenBLAS and MKL thread counts were one. Build logs and complete source, harness,
extension and PES SHA256 values are retained with the report.

| Workload | Baseline median (s) | Optimized median (s) | Ratio |
| --- | ---: | ---: | ---: |
| Complete dRPMD, check every step | 2.773786 | 0.378896 | 7.32x |
| Complete dRPMD, check every 5 steps | 1.007577 | 0.398244 | 2.53x |
| Core, 16 beads, 5 steps/block | 0.098683 | 0.071280 | 1.38x |
| Core, 16 beads, 256 steps/block | 0.064310 | 0.032435 | 1.98x |
| Core, 32 beads, 5 steps/block | 0.136845 | 0.089681 | 1.53x |
| Core, 32 beads, 256 steps/block | 0.109965 | 0.061202 | 1.80x |

The complete jobs include PES initialization, two parent chains, eight H3
collisions, classification, trajectory/checkpoint I/O and summary generation.
Both versions use 16 beads, timestep 0.1 fs, maximum propagation time 200 fs,
persistence 5 fs, checkpoint interval 500 steps, trajectory stride 10 and
configured block length 256. Check intervals 1 and 5 are two distinct comparison
cases; the interval is identical within each baseline/candidate pair.

Each core timing advances 2,000 steps and retains per-step energy and centroid
observations. Core initialization and file I/O are outside the timed region.
The candidate explicitly enables `reuse_forces=True`; the general public API
continues to default to false.

These short runs show measurable host variation. For example, every-step full
jobs ranged from 2.188192 to 3.009405 seconds in the baseline and 0.357426 to
0.428097 seconds in the candidate. All five measurements are preserved; the
table uses medians rather than the fastest run. The figures apply to this
H3/BKMP2 workload and environment and are not universal production speedups.

## Numerical scope and provenance

All six cases have bitwise-identical saved numeric arrays; reported maximum
absolute q/p/energy/time differences are zero. Each version's measured repeats
also reproduce its warmup data digest exactly. Outcome status, channel, reason,
stop step, time and observation settings match across implementations.

The H3 sample has seven nonreactive, one unresolved, zero failed and zero product
trajectories at each check interval. Product-path correctness is covered by the
explicit classifier/workflow regressions; these eight random collisions do not
establish reactive statistics or convergence of a chemical rate.

Restart checks run outside timing. They cover completed outputs and an
interruption after a persisted checkpoint, using nonaligned checkpoint interval
37, output stride 7 and block length changed to 7 on resume. Arrays and
status/stopping records match uninterrupted runs for both implementations.

The final benchmark ran on 2026-10-30 from 12:57:03 to 12:58:05 UTC. It was
measured before committing the candidate, so the candidate manifest records the
base commit plus a dirty working tree. Its `sources_sha256` and
`harness_sha256` identify the exact tested code in this change. Untracked build
symlinks/compiler temporaries shown in that snapshot are local build inputs or
intermediates and are not part of the commit.

## Reproduce

From the optimized checkout, use a Python environment with the repository's test
dependencies and a C++/Fortran toolchain:

```bash
git worktree add --detach ../PyRPMD-baseline af90f20d05368fc910c493977cf29856300b5864
python benchmarks/h3/build_pes.py
python benchmarks/h3/drpmd_optimization.py \
  --baseline ../PyRPMD-baseline \
  --candidate . \
  --library "$PWD/benchmarks/h3/build/libbkmp2.so" \
  --build --repeats 5 \
  --output runs/drpmd-optimization
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q
```

Use a new output directory for every benchmark invocation. The harness defaults
to one permitted CPU and records the chosen affinity. It rejects source,
extension or PES changes during measurement and retains every raw timing.
