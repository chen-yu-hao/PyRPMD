# Normal-cut zero-point projections (issue #156)

This note records the reviewed definitions of the three
`project_path_subspace` zero-point choices. No result here establishes
physical convergence of a rate.

## Quadratic potential definition

For a fixed arc length `s`, the normal displacement is
`x(s,Q) = x(s) + M^(-1/2) N(s) Q`. It is linear in `Q`, so its Hessian is

`K_normal = N.T @ M^(-1/2) @ H_cartesian @ M^(-1/2) @ N`.

Both `zero_point="harmonic"` and `zero_point="rph"` use this quadratic
normal-plane potential. The selector does not subtract a gradient-curvature
term and does not add reaction-path kinetic or mode couplings. Those are
separate from the potential Taylor expansion. The previously proposed
subtraction changed the actual normal-cut frequency and has been removed.
Consequently the RPH and harmonic projected energies agree for the same
path and Hessian; no separate RPH improvement in the transfer factor is
claimed.

An analytic counterexample is a circular path of radius `R=5` on
`V(r)=F*(r-R)+k*(r-R)^2/2`, with `F=0.10` and `k=0.02`. A radial normal cut
has second derivative exactly `k`, independent of `F/R`. The correct
quadratic zero-point energy is `sqrt(k)/2 = 0.0707106781` for both selectors.
The regression checks an independent finite difference of the potential
and the entire projected curve. An analytic Hessian is sufficient; the RPH
selector does not require a separate gradient.

## Independent normal-cut DVR

`zero_point="dvr"` diagonalizes each Hessian normal-mode cut with a unit-mass
Colbert-Miller sinc DVR. It adds each cut's ground energy relative to the
path energy, or its thermal quantum free energy for
`environment="free_energy"`. This is a sum of independent one-dimensional
cuts, not a coupled multidimensional transverse DVR. Strong intermode
anharmonicity or degenerate-mode ambiguity requires a higher-dimensional
calculation.

The default normal half-width is eight oscillator lengths of the actual
positive normal frequency. Only non-positive Hessian eigenvalues use the
explicit fallback box heuristic. In review, a soft harmonic mode with
`omega=1e-5` had incorrectly received the much narrower `omega=0.01` box:
its zero-point energy was `1.845995e-4` instead of `5e-6` hartree. The corrected
automatic box reproduces the harmonic zero-point energy. Non-positive
curvatures remain diagnostic failures requiring an explicit box-convergence
study, even if their anharmonic cuts happen to be confining.

`dvr_points`, `dvr_half_width` and `dvr_spacing` remain numerical convergence
controls. `unstable_complement_points` counts path points with at least one
negative Hessian eigenvalue, while `minimum_complement_curvature` records a
Hessian eigenvalue; it never mixes that value with a DVR energy.

## Cache and workflow behavior

The construction records `zero_point_definition="linear-normal-cuts-v2"`.
Saved RPH/DVR factors from the old subtraction or box rule are not reused.
Historical harmonic projections remain compatible. Fractional `dvr_points`
values are rejected rather than silently truncated by the pipeline.

The benchmark workflow carries the selector through the existing curvature
matrix. Harmonic and RPH now define identical potential curves, so differences
between their rates can only be sampling noise. Old pilot RPH rates from the
removed subtraction are superseded and are not evidence for this revision.
DVR, exact-rate grids, boxes, absorbers, truncation margins, bead count and
RPMD statistics must all be converged before a physical transfer claim.

## Validation

Run with one BLAS thread:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_subspace_model_zero_point.py tests/test_subspace_correction.py \
  tests/test_subspace_transfer.py
python tools/check_workflow_policy.py
```

The tests include analytic quadratic and finite-temperature harmonic limits,
normal-cut grid refinement, a curved radial-potential derivative, a Hessian-only
RPH route, soft positive modes, negative-curvature diagnostics and rejection of
obsolete numerical definitions in saved correction factors. No new H + H2
production calculation is reported by this review.

Review validation completed with **46 passed** for the three suites above
(Python 3.11, NumPy 1.26.4, SciPy 1.12.0; 69.57 seconds), followed by the
additional fractional-grid pipeline regression: **1 passed, 15 deselected**.
The source was unchanged between these runs. Workflow policy and
`git diff --check` passed.
