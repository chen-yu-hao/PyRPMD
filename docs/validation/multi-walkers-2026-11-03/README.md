# T38 distributed multiple-walker validation — 2026-11-03

## Scope and honesty boundary

This record covers deterministic shared-history checks, append-only log fault
injection, a two-dimensional toy statistical campaign, and an artificial
variable-latency scheduler benchmark. It does **not** establish molecular
free-energy convergence, CV adequacy, timestep/bead convergence, or production
speedup for a real PES. All reported workflow summaries keep
`physical_converged = false`.

Baseline: `main@ceee59cd5f702165b56c5e541a503d8810fb9111`.
Working branch: `agent/t38-multi-walkers`.

## Local correctness checks

The execution environment could not resolve GitHub from the shell, so the full
source archive and MPI launcher were unavailable locally. The new core modules
were exercised against byte-for-byte copies of the baseline MetaD/OPES bias
classes in a minimal import harness. This is narrower than the repository test
suite and is recorded as such.

```text
$ OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=. python -m pytest -q tests/test_multi_walker_exchange.py
...................                                                      [100%]
19 passed in 0.98s
```

The tests cover 2/4-walker rank-order parity, an independent weighted-KDE
formula, exact shared OPES normalization, deterministic merge order, sequence
gaps, checksums, truncated tails, committed offsets, duplicate live writer
IDs, an OS-terminated writer with surviving-peer progress, missing-log offset
rejection, writer restart repair, and idempotent prepared-record publication.

```text
$ OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=. python -m pytest -q test_multi_walker_campaign.py -m campaign
.                                                                        [100%]
1 passed, 1 warning in 7.22s
```

The campaign warning is only the minimal harness lacking the repository's
registered `campaign` marker; `pyproject.toml` in the full tree registers it.

The repository-level public-workflow and real-MPI tests are
`tests/test_distributed_walker_workflows.py` and
`tests/test_mpi_shared_walkers.py`. Their authoritative results are the PR CI
checks; local full-suite/MPI execution was not claimed here.

## Two-dimensional double-well campaign

Command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python docs/validation/multi-walkers-2026-11-03/double_well_campaign.py \
  --output docs/validation/multi-walkers-2026-11-03/double_well_campaign.json
```

The toy potential is

\[
U(x,y)=2.5(x^2-1)^2+0.65(y-0.35x)^2.
\]

Four overdamped walkers used the same noise per paired seed. Synchronous MetaD
committed rank-ordered batches every 30 steps. The asynchronous model polled
every 23 steps and preserved each writer's locally visible tempering history.
Across six paired seeds, the RMS difference between the mean FES estimates was
`2.585233321895051e-4`; the combined standard error estimated from run-to-run
surface spread was `9.543816222814713e-2`. The predefined two-standard-error
limit was `1.9087632445629427e-1`, so the mode difference was inside this toy
statistical resolution.

For OPES-explore, 160 uncompressed shared kernels agreed with an independent
fixed-bandwidth KDE formula to `7.105427357601002e-15` maximum absolute error
when supplied the same centers. With independent single-chain and four-walker
sample sets over eight replicates, the mean-surface RMS difference was
`1.322486680526311`, below the two-standard-error limit
`3.3234980203426403`. The large uncertainty is reported rather than hidden;
this is a small estimator campaign, not a convergence result.

Raw data: `double_well_campaign.json`. Reproducer:
`double_well_campaign.py`.

## Artificial slow-PES scheduler benchmark

Command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python docs/validation/multi-walkers-2026-11-03/slow_pes_benchmark.py \
  --walkers 4 --steps 120 --pace 10 --repeats 3 \
  --output docs/validation/multi-walkers-2026-11-03/slow_pes_benchmark.json
```

Hardware reported by Python: Linux 6.18.57 x86-64, glibc 2.41, five logical
CPUs exposed to the container, `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`. Median wall times were:

| mode | median seconds | artificial evaluations/s |
| --- | ---: | ---: |
| `mpi-sync` barrier model | 1.352742891 | 354.8346 |
| `async-files` log model | 1.234893750 | 388.6974 |

The observed asynchronous/ synchronous ratio was `1.0954x`. Each evaluation
was a deterministic log-normal sleep sequence shared by both modes. The number
measures barrier sensitivity and log overhead only; it is not a benchmark of
MPI, force evaluation, or molecular dynamics. Raw data:
`slow_pes_benchmark.json`.

## Fault model exercised

- An incomplete final JSON line remains uncommitted and leaves the byte offset
  unchanged; only its owner repairs the tail after taking the writer lock.
- Complete malformed or checksum-mismatched records fail explicitly.
- A second live process with the same walker ID fails before writing.
- A writer is terminated by the operating system after leaving a partial tail;
  its lock is released, another walker continues, and restart repairs the tail
  without changing the committed global history.
- A checkpoint offset that refers to a missing log fails explicitly.
- Checkpointed offsets and duplicate guards resume without reapplying records.
- A post-step `pending_record` can be interrupted before publication; restart
  idempotently commits the exact record and reproduces uninterrupted
  checkpoints and logs byte-for-byte in the public-workflow test.
- Missing manifests with pre-existing logs and mismatched scientific
  fingerprints are rejected.

## Known limits

Shared OPES is deliberately uncompressed. Memory scales linearly and exact
normalization rebuilds quadratically with kernel count. Distributed OPES is
limited to fixed widths and KDE `cv_density`; distributed MetaD excludes bias
grids and frozen-bias production. Async file visibility depends on the shared
filesystem and provides statistical, not trajectory-wise, equivalence.
