# H+H2 subspace hybrid execution validation

The same BKMP2 gridded models at 300 K and P=32 were evaluated with the Python
trajectory reference, C++/OpenMP, process task concurrency and MPI. All modes
use the optimized SciPy quantum solver. The table measures native dynamics and
resource scheduling; it does not separately measure the quantum rewrite against
an old revision.

Host: Intel Xeon Platinum 8176, two sockets, 56 physical cores / 112 logical CPUs.
Runs used affinity CPUs 0–31 (32 physical cores), CPython 3.11.18, NumPy 2.4.6,
SciPy 1.17.1 and MPICH 4.1.1. The existing PyRPMD conda environment was used;
MPI/test dependencies were placed in a task-local temporary directory.
These are individual wall-time measurements on a shared host.

## Workload and measurements

There are 24 TI windows, 32 walkers/window, 40 burn-in and 80 production HMC
sweeps, 8 nominal leapfrog steps, 256 paired recrossing starts and 100 bootstrap
replicates per model. Both models use seed 20261110. The 1D Hamiltonian is an
adiabatic projection of a relaxed collinear path; the 2D Hamiltonian is the
collinear BKMP2 system in mass-weighted coordinates. RPMD uses the fine spline
grids; quantum nodes are subsets. The dense 2D DVR retains 3624 points.
See `benchmarks/h3_subspace/parallel.py` and the JSON records for all parameters.

| Mode | Total 1D+2D (s) | 1D RPMD (s) | 2D RPMD (s) | 2D QM (s) |
| --- | ---: | ---: | ---: | ---: |
| python1 | 209.599 | 8.700 | 48.882 | 151.515 |
| native1 | 168.365 | 3.581 | 10.098 | 154.624 |
| native32 | 54.808 | 0.475 | 1.136 | 52.433 |
| tasks32 | 50.596 | 0.927 | 2.010 | 49.898 |
| mpi32 | 63.194 | 1.024 | 2.033 | 62.963 |
| native32-unrestricted | 250.665 | 5.492 | 6.791 | 237.704 |

`native32` uses 32 OpenMP trajectory threads and up to 8 BLAS threads, executing
stages sequentially. `tasks32` runs up to four processes with eight threads
each. `mpi32` uses four ranks on this one host with eight threads each; this
validates distributed scheduling, not inter-node hardware performance. MPI rank zero also computes a task. MPI has a 16 GiB per-node budget (4 GiB/rank); other runs use an
8 GiB total budget. Memory estimates gate task admission, rather than imposing
an OS RSS limit.

The `native32-unrestricted` tuning control used 32 BLAS threads and the default
OpenMP waiting policy. Its complex Schur stage took 211.249 s versus 84.324 s
with one thread. Final runs cap BLAS at eight threads and use
`OMP_WAIT_POLICY=PASSIVE`, allowing idle OpenMP workers to sleep during Python
sampling and analysis. The 1D/2D RPMD stages then use actual teams of 32 threads
and take 0.475/1.136 s, versus 8.700/48.882 s in the Python reference.
The complete process run is 4.14 times faster than the Python reference and
3.33 times faster than native single-thread execution.

## Numerical and operational checks

- Factors, component rates and reported standard errors agree across modes to
  relative tolerance 1e-9; observed factor differences are below 3e-12.
- Native mean forces, walker standard errors and free energies are bitwise
  identical at 1, 8 and 32 threads. Python/native trajectory comparisons also
  cover beads 1, 2, 5, 8 and 32, constrained/unconstrained motion and 1–3D
  spline values/analytic gradients, including knots and clamped boundaries.
- Each native run reports 855872/1123680 native polymer-steps in 1D/2D,
  zero Python steps and zero Python potential callbacks. OpenMP team sizes,
  BLAS limits and CPU/wall times are in each result's diagnostics.
- `simd.txt` records GCC's vectorization of the propagation loops using
  16-byte vectors. Builds use C++17, `-O3 -fopenmp`, without fast-math.
- MPI uses four distinct worker PIDs. A fresh launcher invocation reuses all
  four completed tasks in 0.007 s of scheduler time; model preparation and MPI
  startup are outside that timer. Cache tests also cover interruption after
  quantum completion, seed changes, CPU layout changes and rejection of NaN
  options that could otherwise alias a cached null/default.
- 61 relevant non-MPI tests and four MPI integration cases passed. MPI tests
  cover success/restart, worker failure, preparation failure and the collective
  saved-run pipeline. Dense solver tests compare the assembled Hamiltonian to
  explicit Kronecker sums, check Sylvester residuals against LAPACK, exercise
  its scaling/near-degeneracy fallback and reject over-budget allocations.

The raw development measurements started from main `f410b978`; the committed
implementation is `ed5dcc9`. MPI records include that commit and an implementation
digest. The later cache guard changes validation of nonfinite input identity;
it does not change the measured numerical kernels.

## Scientific scope

These are execution and numerical-equivalence tests. The grids, bead number,
TI sampling and recrossing duration have not been converged. Spline outside
fractions are approximately 4.96% (1D) and 2.29% (2D). `physical_converged` remains
false in every correction; these factors are not accepted absolute-rate
predictions. MPI recovery is between completed QM/RPMD tasks. An interrupted
in-progress task restarts from its seed; live recovery after rank loss and
MPI-distributed dense matrices are not implemented.

## Reproduce

After installing PyRPMD with its native extension and `mpi`/`test` extras and
building BKMP2 with the checksum-pinned source:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
python benchmarks/h3_subspace/parallel.py --mode python --cpus 1 --output python1.json
python benchmarks/h3_subspace/parallel.py --mode native --cpus 1 --output native1.json
python benchmarks/h3_subspace/parallel.py --mode native --cpus 32 --output native32.json
python benchmarks/h3_subspace/parallel.py --mode tasks --cpus 32 --output tasks32.json
mpiexec -n 4 python benchmarks/h3_subspace/parallel.py --mode mpi --cpus 32 \
  --memory-limit-mb 16384 --checkpoint-dir mpi-cache --output mpi32.json
# Repeat the same MPI command with a different output filename to check reuse.
python -m pytest -q tests/test_subspace_native.py tests/test_subspace_execution.py \
  tests/test_subspace_correction.py tests/test_subspace_transfer.py tests/test_h3_collinear_subspace.py
```

Bind the launcher to 32 allocated physical cores using the host's scheduler or
MPI binding options. The tests only require two CPUs; the benchmark requires
32 CPUs for the 32-core variants. No performance threshold is enforced in CI.
