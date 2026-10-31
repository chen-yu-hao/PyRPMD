# Grid-bias validation and CPU timing

The grid implementation is checked against the direct Gaussian sum, independent
finite differences of the applied interpolation energy, and workflow restarts.
The focused suite (`test_grid_metadynamics.py`, `test_grid_bias_review.py`,
`test_metadynamics.py`, and `test_surface_workflow.py`) reports **100 passed** on Python 3.12.14.

The tests cover one/two dimensions, narrow/broad periodic kernels, both centroid
and bead-average nonlinear CV Jacobians, two grid resolutions, periodic seams,
physical `P*V` scaling, well-tempered heights, out-of-domain errors, grid cache
identity, and exact replay. Failures deliberately injected **after** deposition
resume with identical q/p, RNG, hills, samples and grid coefficients. Fixed
substrate runs exercise all three deterministic integrators with a partially
periodic PES. Corrupted fixed coordinates/momenta, state shapes and clocks are
rejected even for completed checkpoints. Every hill timestamp must match its step and
sampling timestep, including direct and hill-only grid restores. Independent review regressions also
check signed bias undershoots on coarse grids, the corresponding WT hill
height, rejection of finite coefficient-cache tampering, and ownership of
returned deposition records. Direct Gaussian histories retain their
nonnegative-bias guard.

The executable `examples/metadynamics/grid_fixed_slab.inp` passed input/PES
validation, ran 500 steps with 20 deposited hills and 80 saved samples, and
completed a restart. This is an analytic teaching PES; no molecular convergence
or chemical free-energy validation follows from that smoke test.

## Bias-only timing

Run from the checkout:

```bash
PYTHONPATH=src python benchmarks/materials/grid_bias.py \
  --output docs/validation/materials-extensions-2026-10-31/grid-bias-benchmark.json
```

The archived [JSON](grid-bias-benchmark.json) records Python/NumPy/platform,
seed, bias and benchmark source SHA256, all five wall-time and process-CPU-time
repeats, setup times and numerical errors.
Each repeat evaluates the physical energy and gradient at 100 scalar CV points.
Synthetic standard-MetaD histories use identical prefixes, with `height=0.01`
hartree and `sigma=0.4` bohr. The grid has 80 intervals along each axis over
`[-2,2]` bohr, so its spacing is `sigma/8`. The result uses the median repeat.

| CV dimensions | Hills | Direct CPU (µs) | Grid CPU (µs) | CPU speedup | Wall-time speedup | Grid replay CPU (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 100 | 66.8 | 48.6 | 1.38 | 1.38 | 0.00571 |
| 1 | 1,000 | 314.2 | 46.1 | 6.82 | 6.82 | 0.03516 |
| 1 | 10,000 | 3,319.1 | 81.0 | 40.99 | 41.04 | 0.4637 |
| 2 | 100 | 108.1 | 306.6 | 0.35 | 0.35 | 0.0297 |
| 2 | 1,000 | 587.3 | 194.5 | 3.02 | 3.02 | 0.2354 |
| 2 | 10,000 | 4,011.7 | 199.6 | 20.10 | 20.10 | 1.554 |

The two-dimensional grid is slower for 100 hills. Timings from a shared
runtime vary with CPU availability and frequency; raw repeats are retained.
The algorithm uses the same 16 interpolation terms regardless of history
length. The single-dimensional grid uses four terms. Process CPU time is
recorded separately from wall time to expose scheduling effects.

At 10,000 hills, the maximum energy errors on 256 independent CV points are
`8.50e-6` hartree in 1D and `2.74e-6` hartree in 2D. Maximum gradient errors are
`5.29e-4` and `1.60e-4` hartree/bohr, respectively. Dividing by the maximum
absolute reference gradient gives `1.97e-5` and `2.23e-5`; these are normalized
maximum errors, not pointwise relative errors near force zeros.

This experiment measures the CPU bias kernel and history replay only. It
excludes the PES, integrator, checkpoint and trajectory IO. A larger grid makes
deposition and storage more expensive; a molecular run dominated by its PES
will not obtain the lookup speedup as its total simulation speedup. Grid
convergence remains separate from bead, timestep and statistical convergence.
