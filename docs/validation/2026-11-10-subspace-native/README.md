# Subspace native and parallel validation — 2026-11-10

Base: `f410b97807d99d429ab1b63c6b1c7ff93b424b5c`, tree
`a38181fd46b33547d73b2b56999ae0b8965d7ed0`. The source distribution used
locally came from the prior packaging job's merge commit
`1b6efb43aaa051f2f09631a7f0ce4c3f93fe7650`; GitHub's Git data confirmed the
**same tree**, not merely a similarly named branch.

## Implemented and tested

C++17 cubic B-spline energy/gradient and fused exact-normal-mode ring propagation;
OpenMP over complete rings, canonical radix-2 FFT for P >= 32, NumPy fallback;
unchanged RNG/HMC adaptation, paired flux and equivalent pair bootstrap;
scaled blocked triangular Sylvester with LAPACK fallback; ordered whole-solver
process/MPI tasks; checksum-checked completed-stage reuse; dense working-array
admission estimates. See [execution guide](../../subspace-performance.md).

Local environment: Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, GCC/OpenMP,
OpenBLAS 0.3.30 with one BLAS thread. Five CPUs in affinity, **four-core CPU
quota**. This is not a dedicated performance machine.

## Correctness and safety

- Original correction/transfer baseline: **31 passed** (50.22 s).
- Final selected regression: **203 passed, 1 skipped, 23 subtests passed**
  (78.19 s). The skip is optional ASE, not a native test.
- Independently compiled **without OpenMP**: **44 passed, 8 skipped** (2.88 s);
  the skips are explicitly threaded cases. Missing-extension fallback is also tested.
- Source distribution includes the new native source, Python adapters, tests,
  MPI smoke, benchmark and documentation. No binaries are checked in.

The final selected command was:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4 \
python -m pytest -q \
  tests/test_subspace_correction.py tests/test_subspace_transfer.py \
  tests/test_subspace_native.py tests/test_subspace_linalg.py \
  tests/test_subspace_execution.py tests/test_centroid_path.py \
  tests/test_input_cli.py tests/test_cli_ux.py tests/test_cli_native.py \
  tests/test_parallel.py tests/test_mbar_bootstrap_parallel.py \
  tests/test_native_propagation.py tests/test_native_python_bridge.py
```

Coverage includes 1D–3D gradients/boundaries/nonuniform knots, P=1/2/3/8/32/64,
constrained/unconstrained propagation, thread/block invariance, noncanonical
basis fallback, buffer safety, failed-proposal isolation, full rate/error and
kappa parity, large/singular/scaled Sylvester systems, CAP/threshold/slice
preservation, real spawn static/dynamic execution, worker errors, cache
corruption/scientific invalidation and provider lifetime on failure.

### Existing environment-dependent failures are not hidden

A broader command also included `tests/test_native_thermal_threads.py`:
**215 passed, 11 failed, 2 skipped, 23 subtests passed**. All 11 failures were
existing Linux `/proc/self/task` absolute thread-count assertions (observed
5/6 versus expected 1, or analogous worker counts). Re-running **that unchanged
file against unmodified main** reproduced **the same 11 failures**, with
12 passed and 1 skipped. No assertion was weakened or removed. These failures
are recorded separately from the selected green regression, not called passes.
The full repository suite was not run.

## Fixed-workload solver benchmarks

Raw repeats, numerical values, library metadata and implementation hashes are in
[results.json](results.json). Each run creates a fresh gridded Eckart model;
P=16, 16 TI windows, 8 walkers/window, 20 burn-in and 60 production sweeps,
128 paired launches, 80 bootstrap draws, fixed seed 19. 1D has 321 grid points;
2D has 81 x 9 = 729 points. These are **performance pilots**, not converged rates.
The timer includes both numerical solvers and C_S/error assembly, **not** the
source-run projection, CLI startup, compilation or disk publication.

| Case | Original main median | Native 4-thread first pass | Final rebuild repeat | Final speedup |
| --- | ---: | ---: | ---: | ---: |
| 1D C_S solver | 1.5984 s | 0.3079 s | 0.3305 s | 4.84x |
| 2D C_S solver | 5.6368 s | 1.6510 s | 1.7002 s | 3.32x |

Three repeats per timing. First-pass ratios were 5.19x and 3.41x. Final-rebuild
1D native-single-thread median was 0.5829 s; 2D was 2.1575 s. The final 2D run
used `OMP_WAIT_POLICY=PASSIVE`; an earlier combined recheck exceeded its shell
wall-time limit before completing the four-thread case and produced no final
2D report. Only completed runs are tabulated. Do not extrapolate these small
solver ratios to the 7099-point molecular dense solve or to 32/64 cores.

Original/native C_S agree to about 2e-14 relative in the 1D pilot and machine
precision in 2D; the original/native RPMD rates and statistical errors agree
within floating-point tolerance. Tests also compare full kappa(t), mean forces,
QTST and pair-bootstrap errors rather than only the ratio.

Reproduce using `benchmarks/subspace_performance.py --dimension 1|2 --threads 4
--repeats 3 --output result.json`. For the original source, run the same script
with that checkout's `src` on `PYTHONPATH` and `--legacy`.

## Boundaries and remaining validation

Real MPI execution is **not locally tested**: this runtime has neither mpi4py
nor an MPI launcher. Read-only CI includes two-rank static and three-rank dynamic
smokes for parity, cache reuse and collective failure. A queued workflow is not
passing evidence. The MPI layer distributes complete solver tasks (at most four
for a 1D/2D comparison), not a distributed matrix or within-solver ensemble.

Not implemented: GPU kernels; distributed dense Schur/Sylvester; per-window or
per-pair MPI sharding inside one solver; parallel warm-start projection; in-HMC
or in-Schur checkpoints; a new Philox sampling protocol. HMC sweep/adaptation
and RNG control remain Python. Model projection remains serial. Existing
iterative quantum restrictions are retained, not bypassed for performance.
No production H + H2/Cl + HBr convergence study or physical-convergence claim
is made. Corrected rates retain their original publication gates and
`physical_converged=false`.
