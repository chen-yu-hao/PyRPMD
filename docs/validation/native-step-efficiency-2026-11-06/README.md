# Native step efficiency: bit-identical H+CH₄ umbrella/flux propagation

This record covers a performance-only change to the native propagation loops
and the CBE/PES-2008 wrapper. **No numerical result changes**: every state,
energy, CV observation, status and failure prefix is bit-identical to the
base commit `18d4556`, with and without FMA contraction. No sampling run, PMF
or rate was computed; the timings are microbenchmarks on a shared cloud VM.
The raw summaries are in [results.json](results.json).

## What changed

| Cost before | Change |
|---|---|
| Every `propagate_block` step copied the full `q` into a trial state; each block copied the caller's `q`/`p` in and back out | The caller's arrays and one private array form a double buffer. A step reads the accepted half and writes the other; acceptance swaps pointers. The free-ring step and first half kick write the trial half directly. The block end copies once, only if the accepted state finished in the private half. |
| Umbrella, flux metric and the observed CV each evaluated the same reaction coordinate, and the observation also built a `[P,N,3]` Jacobian | A `BiasStack` evaluates each distinct centroid coordinate once per force call (value, gradient and metric Hessian-vector product, as RPMDrate reuses ξ, ∇ξ, ∇²ξ). The saved-step observation reuses the endpoint value. Centroid bias forces are added in one pass over the force rows. |
| The CBE wrapper used `IEEE_ARITHMETIC`, so gfortran saved and restored the FP environment on every call; separate per-frame finite, distance and output passes | One test per atom pair (`0.04 < d² <= huge`) covers finiteness and the 0.2 bohr guard; only failing frames repeat the original classification. Negation and output finiteness share one pass. |
| Large-P steps were dominated by data movement: a latency-bound per-coordinate Thomas solve with integer modulo, early-exit `isfinite` scans, repeated checks | The cyclic Cayley solve sweeps whole bead rows (vectorized across coordinates). Finite checks are branch-free vectorized scans; checks implied by earlier ones are skipped. Per-block Python descriptor lowering is shared between umbrella, flux and observation. |
| `Dynamics.propagate` (`run_block`, Andersen thermalization) called the PES twice per step | Optional block-local endpoint-force reuse, enabled for stateless providers without fragment projection, as `propagate_block` already did. |

The PILE-L/TRPMD `advance_sampling` loop received the same double buffering
and shared-coordinate path.

## Bit-identity evidence

* `benchmarks/efficiency/native_ab_parity.py` compares this build with the
  base build on identical inputs: 209 comparisons (free-ring step for all three
  integrators, `advance` with and without bias, mixed umbrella/flux/wall stacks,
  other observations, `previous`, centroid RATTLE, fragments, momentum refresh,
  `advance_sampling` with fixed atoms, and `run_block`), **0 mismatches** with
  both the built-in harmonic PES and the real CBE/PES-2008 library. Against an
  FMA-compiled reference the same script reports 201 mismatches, showing it is
  sensitive to last-bit changes.
* Additional scratch comparisons, including 84 blocks that fail mid-block
  after odd and even numbers of accepted steps: 0 mismatches. Building base
  and candidate with `-mfma -mavx2 -ffp-contract=fast` (411 FMA instructions in
  the candidate module) also gives 0 mismatches in all 272 comparisons.
* Old and new CBE wrappers agree bit for bit on 334 frames in 8 batches,
  including NaN/±Inf coordinates, coincident atoms, distances exactly at and one
  ulp around 0.2 bohr, overflowing coordinates and all-rejected batches. With
  the rebuilt library, `benchmarks/hch4/verify.py` passes all 7 electronic
  gates and the 11 real-library tests pass.

## Timing

Built-in harmonic PES (almost free), so this is the loop's own cost:
H+CH₄ geometry and four-channel coordinate, check_energy on every step and an
observation every 20 steps. Best of 3 alternating rounds × 9 repeats.

| P | base unbiased | new unbiased | base umbrella+flux | new umbrella+flux |
|---:|---:|---:|---:|---:|
| 1 | 1.08 µs | 0.60 µs | 1.78 µs | 0.93 µs |
| 16 | 4.18 µs | 2.00 µs | 5.19 µs | 2.85 µs |
| 64 | 16.7 µs | 6.56 µs | 19.0 µs | 8.83 µs |
| 128 | 33.2 µs | 13.8 µs | 42.3 µs | 19.9 µs |
| 256 | 67.8 µs | 25.6 µs | 73.6 µs | 32.0 µs |
| 512 | 148 µs | 51.4 µs | 160 µs | 64.6 µs |

Real CBE/PES-2008 (base with its own wrapper), ξ=0.9 umbrella + flux metric,
same observation cadence, minimum per-step time over 7 alternating rounds:

| API, block | P=1 | P=16 | P=128 | P=512 |
|---|---:|---:|---:|---:|
| `propagate_block`, 256 steps | 4.34 → 3.20 µs (1.36×) | 41.8 → 38.6 µs (1.08×) | 325 → 299 µs (1.08×) | 1282 → 1232 µs (1.04×) |
| `propagate_block`, 10-step blocks | 15.0 → 11.3 µs (1.33×) | 57.4 → 50.8 µs (1.13×) | 366 → 338 µs (1.09×) | 1440 → 1405 µs (1.02×) |
| `propagate` (`run_block`) | 8.40 → 4.83 µs (1.74×) | 80.6 → 40.9 µs (1.97×) | 624 → 306 µs (2.04×) | 2469 → 1351 µs (1.83×) |

At large P the PES evaluation itself is about 95% of a step, so the end-to-end
gain is bounded by it. Single-call wrapper overhead over raw upstream
`get_potential` fell from about 0.5 µs to 0.08 µs per call in an interleaved
C measurement; repeated measurements on this VM vary by tens of percent.

## Limits

* Shared-VM timings: medians differ from minima by up to 25%; only the 7-round
  configurations are tabulated. No GPU, MPI or multi-thread timing was run.
* Fixed-bias FES/PMF sampling still ends a native block at each Andersen
  collision (existing tests pin this schedule), so short collision intervals
  still pay one extra PES call and the Python block setup per collision. The
  existing pre-drawn `refresh` descriptor could remove this in a separate change.
* An existing CBE build must be rebuilt: its receipt records the wrapper hash.

## Reproduce

```bash
python setup.py build_ext --inplace
python -m pytest -q tests/test_native_step_buffers.py tests/test_native_propagation.py
OMP_NUM_THREADS=1 python benchmarks/efficiency/native_step_overhead.py --output overhead.json
# Reference: build the base commit in another worktree and pass its extension.
OMP_NUM_THREADS=1 python benchmarks/efficiency/native_ab_parity.py \
  --reference /path/to/base/src/pyrpmd/_native.cpython-313-x86_64-linux-gnu.so \
  [--cbe-library /path/to/libcbe2008.so]
```

Environment: Intel Xeon @ 2.10 GHz (4 vCPU), Linux 6.18, Python 3.13.18,
NumPy 2.5.3, g++/gfortran 13.3.0. CBE library SHA-256: new wrapper build
`2bacd906…276c`, base wrapper build `9d1b535c…5df0`; wrapper.f90 SHA-256
`ef28beba…21e5` (base `872b7775…5bf8`).
