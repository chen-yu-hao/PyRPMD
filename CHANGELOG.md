# Changelog

User-visible release notes are maintained here. The current working tree has
not been published as a new wheel release; no release date or DOI is assigned.

## Unreleased

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
