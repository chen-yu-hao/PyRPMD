# Changelog

User-visible release notes are maintained here. The current working tree has
not been published as a new wheel release; no release date or DOI is assigned.

## Unreleased

- Subspace correction: the 2D plane model's reactant absorber now continues
  past the start of the path. When the reference slice lies near the reactant
  end, the absorber region reaches beyond the path; grid points there took the
  arc length of the path's end point, so the absorber ramp stopped far below
  its strength and the grid ended 40 units past the path. Reactant waves
  reflected off the grid edge, and k_QM was 22% low on the curved-valley
  validation model. Validation against exact rates on 2D, 3D and rotated 8D
  model potentials:
  [report](docs/validation/2026-11-13-subspace-rph-2d/README.md).

- Subspace correction: reference slices must also be quantum-free. The
  quantum and the RPMD rate are normalized by the quantum and the centroid
  reactant density on the slice; on a slope these differ at O(hbar^2). The
  slice now needs `beta^2|V''|/12 + beta^3 V'^2/24` below
  `quantum.reference_density_tolerance` (1e-2) besides the 0.3 kT energy
  rule. The plane model also takes the free channel that starts at the
  reactant end instead of the last flat point before the transition state.
- Subspace correction: `exact_quantum_rate` checks a supplied absorber on
  the dividing surface instead of against end caps it does not use; the RPH
  model's channel stays open beyond its box for ring polymers.

- **Compatibility:** the subspace correction now builds its reaction path from
  the minimum-energy path by default (`path.source = "mep"`; previously
  `"centroid"`), and its 2D model is the reaction-path Hamiltonian on that
  path (`subspace.model_2d = "rph"`; previously the fixed plane through the
  transition state, still available as `"plane"`). Corrections computed with
  the old defaults are not reused. Set `path.source = "centroid"` and
  `subspace.model_2d = "plane"` to reproduce them.
- Subspace correction: new 2D reaction-path Hamiltonian model
  (`analysis/subspace_rph.py`). Axis 0 is the MEP arc length, axis 1 the
  displacement along a Bishop-transported normal that carries the curvature
  near the barrier; the ribbon is unrolled into a plane where the kinetic
  energy `p_s^2/(2(1 - kappa q)^2) + p_q^2/2` is flat, so the existing 2D
  quantum and RPMD solvers include corner cutting. Model potentials in
  abstract coordinates can drive the pipeline with `run.rigid_body = False`.
- Subspace correction: both 2D models now place the reactant reference slice
  on a DVR grid node. The quantum rate was normalized on the nearest grid line
  and the RPMD TI at the exact reference, i.e. on different slices wherever
  the channel is only flat to `reference_kt`.

- Subspace correction: new path sources that do not rely on centroid
  geometries, which are thermal averages (a librating bond's centroid is
  shorter than the bond). `path.source = "mep"` refines the rate run's
  transition state to a first-order saddle and follows the IRC from it;
  `"pmf_minimum"` takes the minimum of `V` on every umbrella window's `xi`
  surface over the full PMF range; `"centroid_all"` uses the centroids of
  every window. New controls `mep_displacement`, `ts_gradient_tolerance` and
  `pmf_constraint_tolerance`. Tested on LEPS H + H2 with and without
  ion–molecule complexes.

- Subspace correction: a reactant continuation that falls into a
  pre-reaction complex climbs out of it along decreasing rate coordinate `xi`
  (hyperplane steps along `-grad xi`, the minima of `V` on the umbrella
  windows' `xi` surfaces) and ends at `xi = 0`, the rate's own `R_inf`
  reactant reference (`coordinate_target_reached`, accepted as
  `reactant_reference`; `path.reactant_xi` moves the target). No long-range
  model is needed, so neutral and ionic reactions use the same rule. The 1D
  model's reactant reference slice must lie in the free reactant channel
  (path potential within `quantum.reference_kt` of the reactant end, as in the
  2D model): a sloped default slice is moved into the flat channel, and a
  model without one fails the path gate. Termination records gain `minima` and
  `exit_coordinate` (issue #191).

- Subspace correction: path continuations stop in a local minimum
  (`minimum_reached`) instead of climbing out of it along the incoming tangent
  (issue #191). Ion–molecule reactions with pre- and post-reaction complexes
  previously wandered for thousands of steps and ended with
  `asymptote_not_approached`, or passed through a well to the asymptote on its
  far side. A product well at least `quantum.product_drop_kt` below the
  barrier top is an accepted terminus, and the 1D model then absorbs products
  inside it (`exact_quantum_rate` accepts a per-end `cap_width` pair); a
  reactant complex is reported with its reason and still fails the path gate.
  Termination records gain `accepted`, `reason`, `max_energy`,
  `drop_below_top` and `gradient_norm`; the path gate now reads "did not
  reach an accepted terminus".

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
