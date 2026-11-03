# Efficient phase projection for tunnelling splittings

The [open-path ground-doublet calculation](low-temperature.md) removes the
geometric imaginary-time translation tangent from its mass-weighted fluctuation
operator. This operation now uses one Householder reflection and a symmetric
rank-two update. It avoids an explicit orthonormal-complement matrix and the two
dense matrix products used previously. The remaining eigensolve is still dense;
the limit of 4096 active interior coordinates remains in force.

## Mathematical operation

Let $H$ be the symmetric fluctuation operator and $t$ its normalized geometric
phase tangent. Choose the normalized Householder vector $u$ from
$t + \operatorname{sign}(t_0)e_0$, taking the sign as positive when $t_0=0$.
The reflection $R=I-2uu^T$ maps the phase direction onto coordinate zero. Deleting
row and column zero of $RHR$ therefore produces a representation of the same
restricted operator as $Q^THQ$, where the columns of $Q$ are any orthonormal
basis for $t^\perp$.

With $w=Hu-(u^THu)u$, the transformed operator is

$$
RHR = H - 2(uw^T+wu^T).
$$

The implementation forms the remaining lower triangle using BLAS `dsyr2` and
passes that triangle to `numpy.linalg.eigvalsh`. Forming the restriction takes
$O(D^2)$ work and one owned dense matrix plus $O(D)$ vector storage, where $D$ is
the number of active interior coordinates. The eigensolve still takes
$O(D^3)$ work. A negative or nearly zero eigenvalue is never selected as a
substitute for the geometric phase direction.

The sampling model, action, Hessian, phase-residual threshold and result fields
are unchanged. This algebraic change introduces no new scientific option or
run identity. Results agree to floating-point accuracy; bitwise-identical
eigenvalues are not expected from a different orthogonal reduction.

## Single-thread measurements

The benchmark uses the mass-weighted fluctuation matrix of an analytic quartic
kink, with mass 1822, barrier height 0.01 and duration $28/\omega$, all in atomic
units. Both methods include the complete dense symmetric eigensolve. Input
construction and one warmup per method are excluded. Five timed repetitions
alternate method order; values below are medians.

Environment: AMD EPYC 9V74, Linux x86-64, Python 3.12.14, NumPy 2.3.5,
SciPy 1.17.0, OpenBLAS 0.3.30. OpenMP and both NumPy/SciPy BLAS pools use one
thread. Measurement date: 2026-11-03.

| Active interior coordinates | Previous QR wall time (ms) | Householder wall time (ms) | Wall speedup | Previous CPU time (ms) | Householder CPU time (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 4.587 | 2.870 | 1.60× | 4.585 | 2.869 |
| 512 | 27.614 | 14.608 | 1.89× | 27.612 | 14.606 |
| 1024 | 173.666 | 82.949 | 2.09× | 173.663 | 82.947 |
| 2048 | 1311.994 | 722.290 | 1.82× | 1311.551 | 722.266 |

At $D=2048$, peak Python/NumPy allocation tracked by `tracemalloc` falls from
100,615,200 to 33,572,226 bytes. These are additional tracked allocations during
the kernel call, excluding the pre-existing input matrix. They are not process
RSS and may exclude BLAS/LAPACK workspace allocations or allocator caches.
The largest absolute spectral difference across the four cases is
$2.89\times10^{-15}$ in the mass-weighted operator's units.

These measurements describe phase removal plus the eigensolve, not the complete
optimization, a molecular speedup, or physical convergence. Hessian evaluation,
optimization and other prefactor work can dominate an actual calculation.

Reproduce from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python examples/instanton_phase_projection_benchmark.py \
  --sizes 256 512 1024 2048 --repeats 5 --output phase-projection.json
```

The recorded inputs, timing samples, loaded BLAS pools and allocation diagnostics
are available in the {download}`measurement JSON <validation/splitting-phase-projection-2026-11-03.json>`.

## Verification

Before changing `main@8cc402f`:

```text
$ OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q tests/test_tunnelling.py tests/test_instanton.py tests/test_low_temperature.py
58 passed in 2.47s
```

After the change:

```text
$ OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q tests/test_tunnelling_phase_projection.py tests/test_tunnelling.py tests/test_instanton.py tests/test_low_temperature.py
81 passed in 3.07s
```

The warning-as-error documentation build also passed:

```text
$ OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m sphinx -W --keep-going -b html docs docs/_build/html
build succeeded.

The HTML pages are in docs/_build/html.
```

The 23 added cases compare against an independently constructed complete QR
basis for random symmetric matrices, including negative eigenvalues, C and
Fortran layouts, noncontiguous views and read-only inputs. Axis-aligned and
nearly axis-aligned tangents, tangent sign and scale, and an analytic degenerate
complement spectrum cover Householder edge cases.

The existing physical checks remain in the same run: the quartic instanton's
analytic action and prefactor, independent one-dimensional Schrödinger
diagonalization, cancellation of a separable harmonic bath, a curved path,
isotope dependence, and separate duration/grid refinement. Their conservative
`physical_converged=False` semantics remain unchanged.
