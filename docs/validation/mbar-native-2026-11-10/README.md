# Native streaming MBAR validation

The pinned upstream baseline is `f410b97807d99d429ab1b63c6b1c7ff93b424b5c` (main including #171). The measured solver/kernel implementation is commit `bc775cc`; subsequent PR changes add integration tests, the MPI executor guard and this report, without changing the timed kernel. These are synthetic numerical workloads, not new molecular trajectories or converged chemical rate calculations.

## Repeated complete computations

Each row reports three fresh Python processes on the same Xeon Platinum 8176 host, CPU affinity 0–3, `OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=OMP_NUM_THREADS=1`, Python 3.11.18 and NumPy 2.4.6. The native build uses GCC, `-O3 -std=c++17 -fopenmp`, without fast-math. Native per-fit thread counts are explicit, independent of the OpenMP environment default. All 21 raw reports are retained in `raw/`.

`solve`: 48 distinct umbrellas, 96,000 samples, potential construction plus a complete converged MBAR solve, target log weights and overlap. `rate`: 24 umbrellas, 36,000 samples, complete rate-PMF analysis including eight circular-block bootstrap fits and paired Gaussian UI calculations. Rate parallelism uses independent draw threads; each native draw has one native thread. Peak RSS is the maximum resident memory of the whole fresh process, including Python/SciPy and outputs.

| Scope | Backend | Threads/workers | Median seconds | Min–max seconds | Median RSS / MiB | Median CPU/wall |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| rate | baseline | 1 | 5.581 | 5.509–5.858 | 157.3 | 0.99 |
| rate | baseline | 4 | 2.830 | 2.798–2.832 | 368.1 | 3.03 |
| rate | native | 1 | 3.422 | 3.279–3.604 | 90.9 | 1.00 |
| rate | native | 4 | 1.286 | 1.164–1.371 | 99.6 | 2.80 |
| solve | baseline | 1 | 10.230 | 9.826–67.083 | 416.6 | 1.00 |
| solve | native | 1 | 1.996 | 1.955–2.016 | 90.6 | 1.00 |
| solve | native | 4 | 0.570 | 0.532–0.581 | 90.6 | 3.97 |

The complete four-worker rate computation is 2.20× faster than upstream's four-worker bootstrap and reduces peak RSS by 72.9% (368.1 → 99.6 MiB). The complete native single-fit solve is 5.12× faster on one core than upstream and scales another 3.50× from one to four cores. The four-core fit reduces peak RSS by 78.3% (416.6 → 90.6 MiB). These measurements do not imply the same gains for an entire molecular-dynamics campaign.

The baseline complete-solve timings include one 67.08 s outlier. It is retained in the range and raw reports; the median is used for comparisons. This was not an exclusive machine. Native timings are not advertised as a 64-core benchmark. The four-thread solve reports `threads_used=4`, positive `native_calls`, and CPU/wall≈3.97: parallel execution was measured, not inferred from configuration.

## Numerical gates

Every repetition converged and every rate run retained all eight bootstrap draws. Free energies, target weights/log weights, overlaps, PMF curves, bootstrap barriers and uncertainties were checked against the pinned baseline with atol=3e-9, rtol=3e-8. The largest observed absolute difference across recorded outputs was 1.22e-14. Complete arrays were compared before aggregation; only compact reports are committed. The deterministic sample generator and pinned baseline loader are in `benchmarks/efficiency/mbar_streaming.py`.

The existing analytic/non-Gaussian distribution tests and dense `method="legacy"` comparison remain independent numerical checks. Added tests cover 1D/2D/3D harmonic problems, mixed/off-diagonal stiffness, periodic CVs, duplicate states, unequal counts, extreme state offsets, disconnected overlap, native-vs-NumPy equivalence, actual OpenMP workers, missing-extension fallback, no K×N materialization, malformed native buffers, threaded and process executor bootstrap parity, failure draw ordering, workflow executor routing, and the root-only MPI collective guard.

The full installed-wheel smoke test passed from a temporary wheel extraction outside the source tree. A separate build of `native/mbar.cpp` without `-fopenmp` completed a native solve and correctly reported one thread. Runtime/default solver fallbacks do not install dependencies. The final focused suite passed **442 tests** on NumPy 2.4.6 (13 existing deprecation warnings); its full command selection is the MBAR, rate/FES, surface/site, adaptive and structure regression modules listed in `test-command.txt`; output is in `pytest.log`.

## Reproduce

From the repository root, using an existing environment with runtime dependencies and built extensions:

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0-3 \
  python benchmarks/efficiency/mbar_streaming.py --backend baseline \
  --states 48 --samples-per-state 2000 --output /tmp/mbar-baseline.json
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0-3 \
  python benchmarks/efficiency/mbar_streaming.py --backend native --threads 4 \
  --states 48 --samples-per-state 2000 --output /tmp/mbar-native.json
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0-3 \
  python benchmarks/efficiency/mbar_streaming.py --backend native --threads 4 \
  --states 24 --samples-per-state 1500 --bootstrap 8 --output /tmp/mbar-rate.json
```

Use `--backend baseline --threads 4` for the matching upstream parallel rate benchmark. Run each backend separately and repeat three times. The baseline ref is an argument so a reviewer can explicitly choose another comparison revision; changing it requires rerunning the numerical gates.
