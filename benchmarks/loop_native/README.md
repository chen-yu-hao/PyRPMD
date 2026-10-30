# PyRPMD: full time-step loop language benchmark

The benchmark asks a **specific** question: when the PES and Cayley algorithm are
identical, is keeping the **entire BCB/NVE time-step loop** in Python, C, C++ or
Fortran fastest on a small realistic ring polymer?

- `benchmark.py` builds three independent shared libraries with GCC/G++/gfortran
  `-O3 -fno-fast-math`. Each receives the same `float64` `[P,3,3]` q/p, the same
  masses, `beta`, `dt`, and number of steps, returns final q/p, and performs
  **two force evaluations per step**. Native libraries call the **same**
  `pyrpmd_bkmp2_batch` in the compiled author-derived Fortran BKMP2 PES, except
  in a **separate, clearly labelled synthetic harmonic** microbenchmark.
- `C` and `C++` compile the exact same scalar Cayley implementation from
  `cayley_common.h`. They differ in **language/compiler and outer loop**, not
  in the numerical scheme. The `Fortran` implementation independently follows
  the same Cayley tridiagonal/rank-two cyclic correction; all languages handle
  `P=1` and `P=2` identically.
- `Python` is the **actual** `pyrpmd.dynamics.Dynamics.step()` code with its
  normal compiled C++ Cayley kernel (when available) and physical BKMP2 adapter.
  This baseline includes safety validation, state copies, ctypes dispatch and
  Python-level event-loop overhead. Optional `NumPy` uses its independent FFT
  Cayley implementation.
- Before any timing, compares `P=1,2,8,16,32`, nine steps, against the
  independent NumPy reference. Records max absolute q and p discrepancies and
  refuses to time failures. Results use repeated **median wall time**, a
  one-core CPU affinity option, and retain the original samples (not only
  speedups). Large/chaotic trajectories are *not* asserted bit-identical.
- Force calculations, spring propagation and momentum kicks **stay in the
  timed section**. Compilation/loading and job setup stay **outside**. No MPI,
  OpenMP, GPU, thermostat, RATTLE, bias, classifier, or HDF5 is included.

## Reproduce the real BKMP2 experiment

From the repository root on Linux with `gcc`, `g++`, `gfortran`, NumPy and a
working PyRPMD install:

```bash
python -m pip install -e '.[test]'
python benchmarks/h3/build_pes.py
python benchmarks/loop_native/benchmark.py --potential real \
  --sizes 16 32 --steps 2500 --repeats 3 --pin-cpu \
  --output runs/loop-language/real.json
```

The GitHub workflow `.github/workflows/loop-language-benchmark.yml` executes
this for real BKMP2 and uploads the JSON and logs. Do not call a faster timing
"language X is universally faster": compilers, flags, PES complexity, ring
size, hardware and memory layout strongly affect the ranking.

## Additional toy kernel control

```bash
python benchmarks/loop_native/benchmark.py --potential toy \
  --sizes 16 32 --steps 10000 --repeats 3 --pin-cpu \
  --output runs/loop-language/toy.json
```

This control uses a separable harmonic reference, **not a chemical PES**. It
helps establish how different language overhead is when the PES is negligible.

## Why this benchmark is not the full PyRPMD implementation

The native libraries prove the performance of native tight loops on identical
unbiased full-ring Hamiltonian trajectories. They do *not* yet implement the
Python user-defined reaction-coordinate, custom classifier, thermostat, or
umbrella biases. Selecting a fused native back end for general rate/dRPMD/FES
requires matching their per-step observables and checkpoint/resume behavior,
and a verified change must not silently bypass user callbacks. The first
optimization target is therefore fast, uninterrupted NVE propagation with
native PES; more complete workflows require their own tests.
