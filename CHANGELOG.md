# Changelog

User-visible release notes are maintained here. The current working tree has
not been published as a new wheel release; no release date or DOI is assigned.

## Unreleased

- Subspace correction: plan the solves before running them (issue #195). Each
  model's solver plan (`solver_plans` in `subspace_pipeline.json` and the
  native `status.json`) records grid shape, retained points, route and memory.
  A 2D model whose energy truncation, energy floor and slice normalization need
  the dense route but whose grid exceeds `dense_limit` is refused before
  solving (`solver` gate, report status `solver_capacity_exceeded`, with
  `required_dense_limit` and recommendations), instead of ending the stage in
  a generic `ValueError`. `exact_quantum_rate` raises the same plan as
  `DenseRouteUnavailable` (a `ValueError`); `quantum_solver_plan()` computes it
  without solving, and the task scheduler budgets the same estimate. The 2D
  product absorber no longer jumps to the end of the path when
  `product_drop_kt` kT exceeds the available drop (H + H₂ above ~320 K): it
  starts where the channel has settled within `reference_kt` kT of its
  asymptote, a threshold continuous in temperature; grids where the channel
  can drop 15 kT are unchanged. In the sequential route a dimension whose solve
  raises is recorded in `solve_failures` (status `failed` if none succeeds)
  without discarding the other dimensions. The outside-grid gate reason lists
  the TI, surface and recrossing fractions and the dominant one.

- Subspace correction in `task = "rate"` runs now uses the CPUs the input's
  global `[parallel]` table already allocates (`trajectory_workers ×
  bead_workers × threads_per_rank`, or `threads_per_rank` of MPI rank zero),
  capped by CPU affinity, and runs its QM/RPMD tasks through the existing task
  scheduler. Defaults (one CPU) keep the sequential route. The optional
  `[rate.subspace_correction.execution]` table overrides `cpus`, `workers`,
  `blas_threads` and `memory_limit_mb`. Resources are excluded from the
  factor-cache identity; factors are unchanged.

- Add `pyrpmd status RUN_DIR` and `--watch` with committed-step counters,
  percentages, stage ETA and stale-heartbeat detection. Runner, resume and
  extension paths atomically publish advisory `progress.json`; only the main
  process / MPI root appends structured events to `run.log.jsonl` and stderr.
  Stdout remains result JSON. Explicit subspace postprocessing now updates
  these two advisory root files while preserving scientific source artifacts.
  Unknown/adaptive budgets remain null; execution completion is not scientific
  convergence. Checkpoint formats, numerical work and physics identities are
  unchanged. See [Monitoring](docs/manual/progress.md).

- Add machine-readable stable, beta and research workflow classifications and
  a compatibility policy for public stable inputs, summaries and checkpoints.
- Add analytic tutorial inputs and comparison scripts, keeping the short
  examples identifiable as smoke tests.
- Make the default library CI exclude campaign imports and retain a separate
  scheduled/manual campaign suite.
- Add contribution, citation, convergence and issue-reporting guidance.
- Prepare an additive validation-archive inventory and migration plan. Existing
  tracked scientific data remain in place; no archive has been published.
- Subspace correction: add validity gates for the automatic path subspace
  (issue #182). Path continuations no longer have a step budget
  (`path.max_steps` is optional; the default was 600) and report how they
  terminated; convergence is judged over an arc length instead of ten points.
  Unstable transverse Hessians of the path model are classified and reject
  the model instead of silently losing their zero-point term, and
  `dimension = "auto"` then adds the 2D model. Grid exits of the ring
  polymers are resolved by stage (`grid_exit`). A model that fails a gate
  writes `valid: false`, `factor: null` and `diagnostic_factor` and is never
  applied; `subspace_pipeline.json` gains `status`, `validity` and
  `dimension_escalation`, and `applied_dimension`/`corrected_rate` are null
  when no model passes. New `[rate.subspace_correction.validity]` table,
  `hessian_displacement` and `path.asymptote_patience`. The finite-difference
  normal-plane Hessian now displaces the lightest atom by 5e-3 bohr, which
  changes `C_S` of existing inputs within the finite-difference error.
  `relaxed_path` defaults to `max_steps=None`; a non-positive subspace rate
  now yields an invalid correction record instead of `FloatingPointError`.

## 0.2.0 source baseline

The project metadata declares version 0.2.0. Its existing implementation notes
are preserved in [docs/releases/0.2.0.md](docs/releases/0.2.0.md). This entry
does not assert that a corresponding PyPI or GitHub release was published.
