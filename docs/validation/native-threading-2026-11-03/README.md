# Native threading and periodic Lennard-Jones validation — 2026-11-03

This record covers the T27 implementation. All timings are software timings,
not evidence of physical convergence. The machine was a shared Linux container
on an Intel Xeon Platinum 8573C, x86-64, with affinity to five logical CPUs and
a cgroup quota of `400000 100000` (four CPUs). Software: Linux 6.18.57,
Python 3.13.5, NumPy 2.3.5, GCC/G++ 14.2.0. `OPENBLAS_NUM_THREADS=1` and
`OMP_DYNAMIC=FALSE` were used.

## Scope and implementation

- The Cayley map transposes coordinate tiles into its existing workspace so
  bead values are contiguous during the cyclic solve. The small `P=1` path and
  the arithmetic of each independent coordinate are unchanged.
- The exact normal-mode map for power-of-two `P >= 32` packs `q + i p`, retains
  one OpenMP team across batched FFT stages, and performs one forward and one
  inverse complex FFT instead of four transforms. Odd and small rings retain
  the existing direct spectral path.
- The fused block loop parallelizes independent physical kicks. Large energy
  accounting uses fixed 4096-element chunks and sums chunks in a fixed order;
  `q,p` arithmetic remains coordinate-local and deterministic. Single-threaded
  and small-system energy accounting retains the original loop.
- Built-in C ABI v1 models parallelize independent frames without changing a
  frame's accumulation order.
- `provider="lj"` supplies a fixed-cell, three-dimensional periodic,
  truncated-and-energy-shifted Lennard-Jones model through C ABI v2. It uses an
  exact bounded triclinic minimum-image search, deterministic per-frame Verlet
  lists, physical forces, and `-dE/dstrain` virials. Native and NumPy backends
  are explicit; no fallback changes the selected backend.
- PILE-L noise generation and random-number ordering were not changed. Only the
  two real Fourier transforms surrounding the already supplied noise were
  algebraically packed.

## Correctness commands and raw summaries

The reconstructed local source subset was compiled both with and without
OpenMP:

```text
g++ -std=c++17 -O3 -fPIC -shared -fopenmp ... native/cayley.cpp
# success
g++ -std=c++17 -O3 -fPIC -c ... native/cayley.cpp
# success
```

Targeted tests on the locally rebuilt extension:

```text
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q tests/test_builtin_lj.py \
  tests/test_native_normal_mode_packed_fft.py -k 'not trajectory'
.......s.......                                                          [100%]
14 passed, 1 skipped, 1 deselected in 0.17s

PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q tests/test_native_normal_mode_packed_fft.py
......                                                                   [100%]
6 passed in 0.07s
```

The optional skip was ASE, which was not installed in this container. The
thread-count trajectory test requires the complete package rather than the
minimal reconstructed source subset and is delegated to repository CI.

Independent numerical checks performed by the tests and local probes:

| Check | Result |
| --- | --- |
| LJ force vs central coordinate finite difference | maximum absolute error `1.63e-10 Ha/bohr` |
| LJ virial vs all nine homogeneous-strain derivatives | maximum absolute error `6.4e-11 Ha` |
| Native vs NumPy LJ sample frames | energy, force and virial maximum difference `0.0` in the sampled frames |
| Packed two-real PILE map vs independent NumPy FFT | maximum tested difference below `9e-16` for P=1,2,3,4,5,8,16,32 |
| Exact normal-mode free map vs NumPy spectral solution | P=32 and P=128 passed at `3e-12` absolute/relative tolerance |
| Cayley q,p across 1,2,4,8 threads | identical SHA-256 hashes in every matrix case below |

The test suite also covers shifted-cutoff continuity, an FCC nearest-neighbour
lattice sum, triclinic minimum images, neighbor-list reuse/rebuild, unit parsing,
and explicit rejection of incomplete PBC and ambiguous cutoffs. When ASE is
installed, the same test compares each triclinic frame with ASE
`LennardJones(..., smooth=False)`.

## Timing method

The committed public driver is
`benchmarks/efficiency/benchmark_native_threading.py`. The raw JSON in this
directory was collected with a benchmark-only, non-committed wrapper around the
same production `Cayley` and built-in harmonic kernels. Each step contains two
harmonic PES evaluations, two physical half-kicks, one free-ring map and full
ring-polymer energy accounting. Construction is excluded. Three short repeats
were used to bound campaign time; therefore the matrix is useful for identifying
regimes and determinism, not for high-confidence microbenchmark statistics.

The task-card baseline, measured before this work on a four-core container at
`N=1000, P=32`, was 1.69 ms/step at one thread and 1.30 ms/step at four threads
(1.30x). The current short campaign is:

| N | P | 1 thread (us) | 2 | 4 | 8 | 4-thread speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 8 | 24.51 | 36.99 | 60.19 | 235.82 | 0.41x |
| 64 | 32 | 130.02 | 117.82 | 130.14 | 1013.69 | 1.00x |
| 64 | 128 | 629.85 | 491.97 | 472.73 | 1567.31 | 1.33x |
| 1000 | 8 | 420.83 | 382.16 | 301.88 | 1256.91 | 1.39x |
| 1000 | 32 | 2128.07 | 1125.52 | 1489.13 | 2514.58 | 1.43x |
| 1000 | 128 | 10416.63 | 6327.99 | 5892.55 | 5464.68 | 1.77x |
| 8000 | 8 | 3611.86 | 2094.83 | 2169.97 | 2937.74 | 1.66x |
| 8000 | 32 | 20058.00 | 12921.21 | 8197.09 | 12309.16 | 2.45x |
| 8000 | 128 | 108897.79 | 76464.70 | 54989.65 | 58687.47 | 1.98x |

Raw samples and state hashes are in `matrix-t1.json`, `matrix-t2.json`,
`matrix-t4.json`, and `matrix-t8.json`. Every row has one state hash across all
thread counts. Eight threads oversubscribe the four-CPU quota and are retained
only because the acceptance matrix requests them.

The single-threaded normal-mode requirement is met in this campaign:

| N | P | Cayley BCB (us) | normal-mode BAB (us) | ratio |
|---:|---:|---:|---:|---:|
| 64 | 32 | 130.02 | 134.13 | 1.032 |
| 64 | 128 | 629.85 | 672.54 | 1.068 |

A preliminary phase probe at `N=1000, P=32` identified the free-ring map as the
largest component, followed by the built-in PES. Representative one-thread/four-thread values were approximately 1140/571 us for the free map,
234/107 us for one PES phase, and 46/26 us for one kick. Observation and energy
accounting remained a serially significant fraction before the fixed-chunk
change. These probes motivated tiled bead access, frame parallelism, parallel
kicks and fixed-order energy chunks.

## Bandwidth evidence and scaling limit on this host

The requested 2.5x speedup at `N=1000, P=32` was not reproduced on this shared
container. A persistent OpenMP three-stream kernel was measured to distinguish
algorithmic work from host bandwidth/scheduling limits:

| Working set kernel | 1 thread | 4 threads | ratio |
| --- | ---: | ---: | ---: |
| 96,000 doubles per array (about 2.3 MB traffic set) | 34.6 GB/s | 88.4 GB/s | 2.55x |
| 384,000 doubles per array (about 9.2 MB traffic set) | 22.2 GB/s | 27.3 GB/s | 1.23x |
| 16,000,000 doubles per array (DRAM) | 10.0 GB/s | 5.0 GB/s | 0.50x |

The fused `N=1000, P=32` state uses several arrays of 96,000 doubles, so its
active traffic spans the cache-size transition represented by the first two
rows. The observed upper bound for a perfectly parallel three-stream operation
on this host is 2.55x, while the more representative multi-array working set is
1.23x; barriers, cyclic solves, scalar frame energies and observations reduce
that bound further. Host load was visibly non-stationary, including regressions
when adding threads. The code therefore does not claim a portable 2.5x result;
the committed driver should be rerun on an otherwise idle physical machine.

## Small-system path

For `N=3, P=1`, the new OpenMP thresholds are not entered, the tiled Cayley path
is bypassed, and the original serial drift, kick and energy loops are retained.
The only constructor addition is a precomputed coordinate-mass vector. A stable
before/after percentage could not be produced in the shared reconstructed
checkout, so the 2% regression criterion remains a CI/reviewer measurement
rather than a claimed local result.

## Limitations and interpretation

- This container did not provide a complete clone, so the full repository test
  suite and Sphinx build are reported by GitHub CI rather than this local
  subset. The PR must not be merged if CI adds a failure.
- ASE comparison is optional and was skipped locally.
- The periodic LJ neighbor construction is deterministic Verlet `O(N^2)` rather
  than a linked-cell implementation. Force evaluation scales with the retained
  neighbor count. This favors correctness and triclinic generality; very large
  homogeneous systems should use a production external or ML potential.
- The shifted potential is energy-continuous but not force-shifted; the force is
  discontinuous at the cutoff by design and this is stated in the public guide.
- No timing implies physical convergence, equilibrated condensed matter, or a
  production-quality material model. `physical_converged` remains false in the
  benchmark metadata.
