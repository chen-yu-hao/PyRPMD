# Event-bounded native blocks for diffusion NVE production (2026-11-05)

Short H + CH4 (CBE/PES-2008) NVE execution probe run through `pyrpmd run`.
It measures execution only; it is not a diffusion or rate measurement.

## Inputs

`inputs/original-hch4-nve-p32.inp` is the input as supplied. The other two
inputs change only the PES library path (a local build of the same pinned
source, receipt in `raw/libcbe2008.build.json`) and, for P=128, `beads` and the
output directory. Physics, timestep, step counts, sample interval and
checkpoint interval are unchanged: 256 production steps, `sample_interval=16`,
`checkpoint_steps=256`, `block_steps=256`, `native="required"`,
`loop_backend="native"`.

## Method

* "before" is commit `ca093eb` (source snapshot, same compiled `_native`).
  "after" is this change. Both use the same native extension binary.
* One warm-up run per configuration, then 5 timed runs. Five additional
  "before" runs were interleaved with the "after" runs (10 "before" runs total).
  Every run writes a fresh output directory.
* `timed_cli.py` calls `pyrpmd.cli.main(["run", ...])` in-process and wraps
  `Dynamics.propagate`, `Dynamics.propagate_block`, `_native.run_block` and
  `_native.advance` with timers and step counters. "propagation" is the sum of
  the two Python-level methods, "CLI" is `cli.main`, "process" is the whole
  interpreter including imports.
* Single thread (`threads_per_rank=1`, `OMP_NUM_THREADS=1`) in a 4-vCPU cloud container; runs were sequential.

## Results (medians; ranges in `raw/summary.txt`)

| P | Native calls before | Native calls after | Propagation s before / after | CLI s before / after | Process s before / after |
|---|---|---|---|---|---|
| 32 | 256 x `run_block(1)` | 16 x `advance(16)` | 0.0571 / 0.0256 (2.23x) | 0.511 / 0.472 (1.08x) | 0.766 / 0.740 (1.03x) |
| 128 | 256 x `run_block(1)` | 16 x `advance(16)` | 0.1906 / 0.0984 (1.94x) | 0.649 / 0.561 (1.16x) | 0.908 / 0.810 (1.12x) |

Blocks are 16 steps because each saved sample (every 16 steps) needs Python
estimators; `block_steps=256` is an upper bound. PES batches during production
fall from 2 x 256 = 512 to 16 x 17 = 272, plus the unchanged 16 observation
evaluations. Propagation is a small part of this 256-step probe; the remaining
CLI time is setup, checkpoint/output and analysis, which this change does not
touch.

## Numerical comparison

`raw/numerical-comparison.json` compares every array in the checkpoint and
sample files (q, p, step, time, elapsed, RNG, propagation counters, centroid
positions/velocities, observables, ring radii, sample steps/times) plus the
summary's NVE diagnostics and diffusion result. All 12 before/after pairs and
both resume cases are bit-identical:

* `A_old_then_new`: pre-change code interrupted after production step 99,
  resumed by the new code (first block 13 steps, to the sample at step 112).
* `B_new_then_new`: new code interrupted after its 7th block (q/p already
  advanced by the native call, not committed), resumed from step 96.
