# Native and parallel subspace correction

The correction remains `C_S = k_QM(H_S) / k_RPMD(H_S)`. The execution controls
below do not change the Hamiltonian, temperature, bead count, normalization,
paired-flux estimator or physical-convergence flag. See
[subspace correction](subspace-correction.md) for scientific settings.

## Local CPU execution

A normal source installation builds `pyrpmd._subspace_native` (C++17, Python
buffer API; no NumPy C ABI or pybind11 dependency). Linux builds use OpenMP.
`PYRPMD_NO_NATIVE=1` omits it; `PYRPMD_NO_OPENMP=1` builds a serial kernel.

Add to the standalone `pyrpmd subspace-correct correct.toml` specification:

```toml
[execution]
backend = "process"
workers = 2
scheduling = "dynamic"
rpmd_threads = 2
quantum_threads = 2

[rpmd]
backend = "auto"        # auto | native | numpy
block_steps = 256

[quantum]
sylvester_solver = "auto"  # auto | lapack | blocked
sylvester_block_size = 64
# max_memory_mb = 4096

[output]
reuse = true
```

This permits two whole solver tasks at once. For one dimension these are the
QM numerator and RPMD denominator; for a requested 1D/2D comparison there are
four tasks. Model projection finishes on the root process before numerical
tasks are submitted. The source PES provider is closed even if projection
fails. Only the root publishes final correction artifacts.

Choose workers and per-task threads to fit allocated CPUs and memory. RPMD
workers limit BLAS to one thread and use OpenMP across independent complete
rings. Quantum workers apply `quantum_threads` to BLAS, then restore limits.
There is no claim that concurrent dense solves outperform one large threaded
solve. `backend="serial", workers=1` retains sequential solver tasks while
allowing native RPMD threads. With tiny tasks process-start overhead dominates.

Python API:

```python
from pyrpmd.analysis.subspace_correction import subspace_correction

correction = subspace_correction(
    model, beta, nbeads=32,
    rpmd_options={"backend": "native", "threads": 4, "block_steps": 256},
    quantum_options={"sylvester_solver": "auto"},
    execution={"backend": "process", "workers": 2, "quantum_threads": 2},
)
```

Use an importable script and a `if __name__ == "__main__":` guard when spawning
processes. Explicit `rpmd_options['threads']` overrides `execution.rpmd_threads`.
Parallel/cached APIs require an unmodified static-grid `SubspaceModel`.
Serial uncached APIs still support arbitrary Python potential callbacks.

## Native RPMD contract

C++ evaluates the same SciPy-constructed cubic B-spline coefficients and knots
in one fused energy/gradient pass for 1D-3D grids. Coordinates outside the box
are clamped; gradients in clamped directions are zero; the existing outside
fraction is retained. Native calls release the GIL and own private workspaces.
Buffer shapes, dtype, contiguity, alignment, writable outputs and aliasing are
checked before entering a parallel region.

HMC leapfrog sequences and real-time trajectory blocks run in C++. The HMC
sweep loop, NumPy RNG draw order, random trajectory lengths, Metropolis test
and global burn-in step-size adaptation remain on their existing Python clock.
This intentionally does **not** introduce a new Philox sampling protocol or
independently adapted windows. A failed proposal cannot mutate accepted state;
it is replayed through the reference path to preserve rejection semantics.

All complete rings are independent OpenMP work items. The real-mode FFT is
used for canonical power-of-two rings with P >= 32; other sizes/bases retain
the matrix transform. A single trajectory's time steps remain sequential.
Opposite-momentum children retain their pairing. Bootstrap resamples pairs,
not individual plus/minus children; pre-averaging their tail contributions
avoids large per-replicate temporary arrays without changing the estimator.

`auto` falls back to NumPy for unsupported callbacks/builds and reports why.
`native`, or `threads>1` without supported OpenMP, fails rather than silently
running a different potential. Use `backend="numpy", threads=1` for comparison.
Floating-point ordering may differ from NumPy/FFT or between BLAS libraries;
there is no cross-platform bitwise-equivalence promise. The fixed-seed native
thread/block invariance and reference numerical parity are regression-tested.

## Dense quantum solver

The Hermitian eigensolve and complex Schur decomposition remain SciPy/LAPACK.
The triangular equation `T**H X - X T = scale*C` has a blocked backend: BLAS-3
off-diagonal updates and LAPACK `ztrsyl` on each diagonal tile. Local scaling
is propagated to the complete RHS and solved prefix. Perturbed/singular tiles,
non-finite updates or cumulative scaling underflow fall back to the full
reference LAPACK solve and retain its status. `auto` uses blocks above 256 rows.

CAP values, energy floor, truncated DVR support, slice normalization and the
iterative-route restrictions are unchanged. Absorber/threshold callbacks may
be evaluated once on the exact DVR nodes and transported as arrays; this is
not an interpolation or alteration of the operator.

`max_memory_mb` rejects a dense solve whose working-array estimate
`128 * N * N / 2**20` MiB exceeds the budget, before constructing the dense
Hamiltonian. Vendor LAPACK workspaces and other process memory are additional:
this is an admission estimate, **not an enforced peak-RSS cap**. An execution
`memory_budget_mb` also conservatively serializes local process tasks and
applies the limit per dense solver; under MPI it is a per-rank budget.

## Completed-stage reuse

The pipeline stores solver results under `OUTPUT/solver-cache`. Set
`output.reuse=true` or `execution.reuse=true` to reuse them. An explicit
`execution.cache_directory` selects a different location; use an absolute
shared path for MPI. The Python API caches only when that directory is supplied.

Fingerprints include model grids/values, masses, surface/reference,
normalization, provenance, beta, RPMD beads, numerical settings and seed,
implementation source hashes, native binary hash and NumPy/SciPy versions.
NPZ results contain all dataclass fields, including time/kappa arrays, and load
with `allow_pickle=False`. Atomic files plus SHA-256 manifests detect damage;
missing, changed or corrupted records are recomputed. Old `C_S_1d.json` files
are **not** trusted merely because temperature agrees. Model projection still
runs to establish the current numerical Hamiltonian before cache lookup.

This is completed-solver reuse, **not** an in-HMC, in-trajectory or Schur
checkpoint. Concurrent writes to the same cache key can cause a safe cache miss;
cache files are not a multi-writer database.

## MPI scope

Set `[execution] backend="mpi"` and call the CLI on every rank:

```bash
mpiexec -n 2 pyrpmd subspace-correct correct.toml
```

Static scheduling uses each rank as a whole-solver slot. Dynamic scheduling
reserves one coordinator, so two slots need three ranks. The task list and
results are ordered; root-only preparation/finalization use collective error
reporting. Use identical environments and shared model/output/cache paths.
A multi-rank launch with a non-MPI backend is rejected, preventing duplicate
writes. The Python pipeline must not be wrapped in a root-only call.

MPI here distributes complete QM/RPMD tasks: it does **not** shard one dense
matrix, one HMC chain or a single solver's recrossing ensemble across nodes.
For 1D/2D there are at most four numerical tasks. OpenMP supplies parallelism
within each RPMD task. GPU propagation and parallel warm-start grid relaxation
are not implemented in this change.

## Validation and benchmarking

Run the native/reference, process/cache and blocked-solver regressions:

```bash
OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_subspace_native.py tests/test_subspace_linalg.py \
  tests/test_subspace_execution.py tests/test_subspace_correction.py \
  tests/test_subspace_transfer.py

PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python benchmarks/subspace_performance.py \
  --dimension 2 --threads 4 --repeats 3 --output subspace-performance.json

mpiexec -n 2 python tests/mpi_subspace.py --scheduling static
mpiexec -n 3 python tests/mpi_subspace.py --scheduling dynamic
```

The benchmark starts at `subspace_correction(model, ...)`, includes both solvers
and factor/error assembly, and excludes source-run model projection, interpreter
startup, compilation and disk publication. It uses identical low-statistics
science across implementations, not a converged molecular rate. `--legacy`
allows the original checkout on `PYTHONPATH`; `--rpmd-only` isolates the RPMD
half. Compare raw repeats and numerical results, not just CPU utilization.
See the [dated validation record](validation/2026-11-10-subspace-native/README.md)
for measured results and explicit untested boundaries.
