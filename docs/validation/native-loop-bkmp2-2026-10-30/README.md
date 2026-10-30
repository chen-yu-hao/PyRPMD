# PyRPMD full-loop benchmark — real H3/BKMP2, 2026-10-30

**Validated scope:** BCB/NVE time stepping on the actual Fortran BKMP2 potential, 1500 K, dt 0.05 fs, H+H2 initial geometry, 16 or 32 ring-polymer beads, one CPU affinity core. No umbrella, recrossing, thermostat, classification or HDF5 I/O in timed region.

- Source code: benchmarks/loop_native/benchmark.py (Python), loop_c.c, loop_cpp.cpp, loop_fortran.f90 and cayley_common.h.
- [GitHub Actions run](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41001918841), 1 successful CPU benchmark job.
- [Complete downloadable artifact containing timing JSON, correctness checks, commands and author-source build record](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41001918841/artifacts/12954280017).
- [Ordinary 3-job regression CI](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41001918769), all successful.

## Real BKMP2 measurements

Same Python/NumPy/Fortran compiled physical PES. C and C++ use **identical** scalar cyclic Cayley code; Fortran implements the same cyclic solver independently. Python baseline is the real PyRPMD Dynamics.step path with compiled C++ Cayley map, Python ctypes BKMP2 adapter and safety checks. Native loops issue 2 PES batches per BCB step without re-entering Python. All native programs compiled -O3 -fno-fast-math with Ubuntu 24.04 GCC/G++/gfortran 13.3.0; OpenMP/BLAS thread counts 1. Each measurement consists of 2500 steps repeated 3 times; values are **medians**.

| Beads | Python steps/s | C steps/s | C++ steps/s | Fortran steps/s | Fastest/native speedup over Python |
|---:|---:|---:|---:|---:|---|
| 16 | 4,620.18 | 35,153.98 | **35,206.76** | 34,979.42 | C++; **7.620x** |
| 32 | 4,119.98 | **19,308.72** | 19,284.19 | 19,184.66 | C; **4.687x** |

Median seconds per 2500 steps: 16 beads (Python 0.541104, C 0.071116, C++ 0.071009, Fortran 0.071471); 32 beads (Python 0.606799, C 0.129475, C++ 0.129640, Fortran 0.130312). Native language differences <1% are too small for a stable ranking from three repetitions.

**Numerics:** 20 checks per potential (P=1,2,8,16,32 times Python/C/C++/Fortran) against independent NumPy/FFT Cayley over nine steps, all passed. Real BKMP2 max |delta q|=1.954e-14 bohr; max |delta p|=3.537e-12 atomic units. This is a short-term integrator check, not a proof of whole-trajectory energy-drift equivalence. The same author-referenced BKMP2 library was used in all real cases.

## Synthetic harmonic control (not chemistry)

For 10,000 steps × 3 repetitions (median steps/s):

| Beads | Python | C | C++ | Fortran |
|---:|---:|---:|---:|---:|
| 16 | 13,616 | 852,540 | 817,119 | 801,704 |
| 32 | 12,754 | 352,084 | 339,502 | 359,322 |

## Interpretation and limits

Fusing the outer loop and real Fortran PES calls into native code reduced dispatch, validation and copying costs by about 4.7–7.6 times. This should not be assigned solely to the Python interpreter. The 3 native languages are practically tied for H3 under the tested compilers and flags. Existing **C++17 integration** is the sensible extension target because its ecosystem is already part of PyRPMD; this is an architecture choice, **not** proof that C++ universally wins.

The test libraries are currently **benchmark-only**, not production dRPMD/FES/PMF/recrossing backends. Exact matching of biases, thermostats, constraint projection, trajectory classification, callbacks, checkpoints and numerical error semantics is required before moving those workflow loops into C++ or Fortran. No physical chemical rate convergence is claimed.

Reproduction: install PyRPMD; build the pinned Fortran PES with benchmarks/h3/build_pes.py; execute benchmarks/loop_native/benchmark.py with --potential real --sizes 16 32 --steps 2500 --repeats 3 --pin-cpu. Use the GitHub workflow for the recorded environment.


## Independent default-branch rerun — hardware/run variability

The squash-merged main commit ca1e86a was independently rerun on a fresh GitHub-hosted runner. [Workflow 41002549955](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41002549955) passed the same 20 short-trajectory correctness checks per PES mode; [raw artifact](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41002549955/artifacts/12953488774).

Run-specific medians, real Fortran BKMP2, 2500 steps × 3 repeats, one pinned CPU:

| Beads | Python steps/s | C steps/s | C++ steps/s | Fortran steps/s | Fastest/native over Python |
|---:|---:|---:|---:|---:|---:|
| 16 | 8032.25 | 42692.84 | 42586.58 | 42581.54 | 5.315x |
| 32 | 6762.50 | 23739.06 | 23676.90 | 23718.80 | 3.510x |

The original experimental-branch run and the main rerun differ materially in **absolute throughput and native/Python speedup** despite the same algorithm and runner class. Do not pool repetitions across runners as though they were same-hardware trials. Together the two successful runs show ~5.32–7.62x at 16 beads and ~3.51–4.69x at 32 beads, while C/C++/Fortran remain effectively tied within each run. This is the stronger supported statement than citing only the largest speedup. These runs are not a controlled study of compiler/hardware topology effects.
