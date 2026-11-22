# Changelog

User-visible release notes are maintained here. The current working tree has
not been published as a new wheel release; no release date or DOI is assigned.

## Unreleased

- ChemPotPy DPEMs (research, optional dependency): `ChemPotPyDPEM` now bridges
  any entry of ChemPotPy's `multi_state_list` (`system`, `surface`, `symbols`;
  e.g. `OH3_PIP_FFW1_2019_DPEM`, `NH3_GEN_FFW1_2006_DPEM`), with `nstates`
  read from the Fortran source and its hash in the identity. `gradient =
  "auto"` (default for these entries) uses upstream analytic `ug` only after a
  one-time check against `u` and a five-point difference, because energy-only
  entries return an all-zero matrix and gradient from `ug`; otherwise it
  differentiates `u`. `matrices()` returns matrices with one call per frame
  and is used by the subspace path source. Tables
  `{provider = "chempotpy", system, surface, ...}` work in `diabatic_adiabat`
  and `[electronic] source = "diabatic"`. The default NaH2 5G entry and its
  pinned hash, symbols check and STOP screening are unchanged.
- Diabatic potential energy matrices (DPEMs) in the subspace pipeline and the
  rate run (research): `[electronic] source = "diabatic"` evaluates a DPEM
  (`potential = {model = "nah2_leps_5g_1992" | "tully1" | "lvc" | ...}`,
  `{provider = "diabatic_python", factory = ...}`, or a live object) directly
  along the 1D path, with no diabatization and `points = "grid"` by default.
  The new scalar provider `provider = "diabatic_adiabat"` (`diabatic = {...}`,
  `state = 0`) exposes one adiabat of a DPEM with Hellmann-Feynman forces, so
  the ground-state RPMD rate and `eta_NA` can come from the same surface.
  `pyrpmd.potentials.diabatic.make_diabatic_potential` builds a DPEM from a
  table or passes a live object through.
- Conical intersections along the reaction path in the subspace pipeline
  (research): an `[electronic]` table in `subspace-correct` (and
  `[rate.subspace_correction.electronic]` in `rate` inputs) computes the
  nonadiabatic factor `eta_NA` of the 1D path model and applies
  `C_S * eta_NA`. Electronic geometries lie on the same arc-length spline as
  the 1D model; energies and adjacent-frame overlaps come from an
  `ExcitedStateBackend` (`source = "backend"`), a Python callable or a
  precomputed `.npz` (`source = "file"`, which first writes
  `electronic_1d_request.npz` and reports `electronic_data_required`). The
  propagation-diabatized matrix is shifted so that adiabat `state` equals
  `V_S^(0)` exactly. New gates `electronic_leakage`, `electronic_resolution`
  and `ground_state_mismatch`; new report fields `nonadiabatic` and
  `applied_factors`; new files `electronic_1d.npz` (cache) and
  `multistate_1d.npz`. With `[electronic]`, only dimensions with a valid
  `eta_NA` are applied (2D is recorded as unsupported). Pipelines without the
  table are unchanged.
- Coupled electronic states in a quantum subspace (research, Python API):
  `pyrpmd.analysis.multistate_subspace` adds `MultistateSubspaceModel` (a
  Hermitian diabatic `S x S` matrix, real or complex, on a 1-3D sinc-DVR grid)
  and `exact_multistate_rate`, the dense Miller-Schwartz-Tromp/Sylvester rate
  with a nuclear dividing surface summed over electronic states and product
  flux resolved into adiabatic or diabatic channels from one shared Schur form
  (`channel_sum_residual` checks the channel sum rule).
  `nonadiabatic_subspace_factor` returns the coupled-state rate over the exact
  rate on one Born-Oppenheimer adiabat of the same subspace, and
  `propagation_diabatization` builds 1D diabatic matrices from adiabatic
  energies and raw adjacent-frame overlaps (the excited-state
  `BackendResult.overlap` convention). Not yet wired into `subspace-correct`;
  dense route only. See `docs/multistate-subspace.md`.
- Subspace correction triatomic RPH construction, normal-cut DVR robustness,
  and batched grid evaluation: orient 2D `rph_model` axis 0 along the bisector
  of the reactant reference and product absorber tangents (with oblique endcap
  padding and wall-masked 2D `record_edge_slope` floor refinement) so symmetric
  $120^\circ$ triatomic exchange paths advance cleanly; bound `half_width`
  normal-curvature sampling to the active barrier/channel region to prevent
  asymptotic soft-mode grid bloat; add vectorized `RateRun.energies` and
  chunked `project_rph_values` evaluation with recursive bisection/scalar
  fallback (30.0x RPH grid speedup on 3D H + H$_2$ BKMP2); and cap adaptive
  `zero_point = "dvr"` half-widths / rigid-body displacements while sealing
  non-finite outer cut wings outward.
- Subspace correction performance and multi-core scheduling:
  `_thermal_frequency` in `pyrpmd.analysis.subspace_rpmd` now requires all
  three central-difference stencil points to lie inside the thermally
  accessible region (falling back to the center-point mask only on narrow
  grids), eliminating spurious curvature spikes across `wall_energy` boundaries
  (5.25x 2D RPMD speedup with statistically invariant `kappa` and `C_S` on
  200 K H + H2, P = 128). In `pyrpmd.analysis.subspace_execution` and
  `pyrpmd.workflows.rate_subspace`, local `process` execution now overlaps
  lighter stages alongside heavy 2D tasks with workload-aware dynamic thread
  allocation (avoiding stranding heavy tasks at `cpus // workers` threads while
  preserving thread-count determinism and task checkpoint fingerprints), and
  `execution.cpus = "auto"` resolves process CPU affinity for both `process`
  and `mpi` backends (`pyrpmd subspace-correct` CLI included).
- Subspace correction validity gates and bilingual documentation: two validity
  gates for failures found on the 6D/8D complex-PES benchmark. `surface`
  rejects an RPMD factor whose dividing surface lies more than
  `validity.max_submerged_kt` (1 kT) below the centroid free-energy maximum
  between the reactant reference and the surface: through a barrier submerged
  below the asymptote, QTST exceeds what the entrance channel allows and the
  finite-time transmission coefficient never removes the excess. On the 6D
  submerged-barrier model the exact-reference RPMD rate was 2300 times the
  free-particle flux and its `ln C_exact = -8.67` an artifact; with the RPMD
  surface at the entrance-channel bottleneck it is `-0.89 +- 0.02`. `ribbon`
  rejects an RPH 2D model whose path bends more tightly than its transverse mode
  is wide (`kappa / sqrt(2 omega_n)` above `validity.max_curvature_width`, 0.5),
  where the ribbon coordinates fold inside the wavefunction; on the complex
  models (0.56 in 8D, 0.68 in 6D) its quantum rate was 66% high (8D) and a
  factor of 6 low (6D). `ring_polymer_rate` records `free_energy_bottleneck`;
  `rph_model` records `ribbon_fold`. Both gates are `reject`/`allow` policies
  (`submerged_surface`, `folded_ribbon`). Restructure `docs/subspace-correction.md`
  and add the full Chinese guide `docs/zh/subspace-correction.md` with
  workflow cross-links across EN/ZH manuals.
- Potentials and Torch workflows: add first-class zero-copy JAX DLPack potential
  provider (`provider = "jax-dlpack"`, `JAXDLPackPotential`) and device-resident
  `sampling.engine = "torch"` gas-phase rate workflow (`gpu_rate.py`,
  `torch_rate.py`, issue #205). `JAXDLPackPotential` enforces `jax_enable_x64`
  and `float64` precision on `"cpu"` or `"cuda[:index]"` without implicit device
  or dtype fallback, compiles scalar $E(q)$ or `(energy, forces)` / mapping
  callables via `jax.jit` / `jax.vmap` / `jax.value_and_grad` (including
  periodic `(positions, cell, pbc)` and JAX AD strain virials plus
  `max_frames_per_call` chunking), and hardens PyTorch/XLA DLPack handoff across
  non-default `torch.cuda.Stream`s (`__dlpack__(stream=...)`, `record_stream`,
  cross-stream CUDA event ordering, and retained output buffers).
  `JaxPIPPotential` (`provider = "jaxpip"`) delegates its NumPy and `TorchPES`
  DLPack paths to `JAXDLPackPotential` while preserving `pyrpmd.jaxpip-fit/1`
  manifest and bounds checks, and `sampling.engine = "torch"` under
  `task = "rate"` and `pimd`/`diffusion` accepts `jax-dlpack`, `jaxpip`,
  `torch`, batched ML providers (`*-batched`, `metatomic`), and analytic
  `free`/`harmonic`/`periodic_cosine` models while rejecting host-only or
  `float32` configurations at preflight.
- Subspace correction & instanton anchor: added steepest-descent RPMD defect
  `steepest_descent_defect` and `steepest_descent_rpmd_defect` (`eps_SD = 2*pi /
  (beta * hbar * eta_1)` below `T_c`, `1` at or above `T_c`) in
  `pyrpmd.instanton`, bare-potential skeleton extraction
  `SubspaceModel.skeleton()` (`bare_values` populated with zero extra PES
  evaluations in `project_linear_subspace`, `project_path_subspace`, and
  `project_rph_values(..., return_bare=True)`), and `subspace_instanton_defect`,
  `full_instanton_defect`, `instanton_anchor`, plus `anchor_log_shift` in
  `apply_subspace_correction` (`pyrpmd.analysis`). Setting
  `rate.subspace_correction.anchor = "skeleton"` (or
  `subspace.anchor = "skeleton"` in `run_subspace_pipeline`) anchors `C_S` to
  the full-dimensional instanton via `ln C_S + A_H - A_{S^0}` without
  double-counting the adiabatic zero-point / free-energy dressing, and refuses
  uncontrolled shifts when `T` lies between the full and skeleton crossover
  temperatures. Validated across 95 RPMD tasks on 2D valley models M1–M4
  (`benchmarks/subspace_instanton`,
  [report](docs/validation/2026-11-14-subspace-instanton-anchor/README.md)) and
  across 6D–10D complex potential energy surfaces (`benchmarks/subspace_complex`);
  see [docs/subspace-instanton-anchor.md](docs/subspace-instanton-anchor.md).
- Subspace correction on complex and high-dimensional PESs: barrier dividing
  surfaces and path peaks now select interior saddles by topographic prominence
  rather than global `argmax`, supporting submerged central barriers
  (`V_TS < V_asymptote`) flanked by pre- and post-reaction ion–dipole complexes;
  2D saddle initialization tracks the connected valley from `(s_div, 0)` so
  off-channel L-shaped complex wells do not pull the seed away from the barrier;
  and `subspace_instanton_defect` / `full_instanton_defect` use dome-bounded
  partitioned rational-function optimization (P-RFO) on the folded half-ring
  action with Bofill Hessian updates, rigid-body translation/rotation projection,
  and sparse `LinearOperator` + `eigsh` spectra when `nbeads * D > 2048`.

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
- Native ONNX PES (`native/onnx`): add optional `-DPYRPMD_ONNX_CUDA_KERNELS=ON`
  custom float64 CUDA `ScatterND` operator (`none`, `add`, `mul`, `max`, `min`
  reductions with `atomicAdd(double*, double)` accumulation and out-of-bounds
  index detection) bound to `CUDAExecutionProvider` via a load-time custom
  domain rewrite for float64 `ScatterND` nodes, keeping CPU builds free of CUDA
  Toolkit dependencies and reporting `capabilities.float64_cuda_scatter_nd` in
  provider metadata. `tools/export_jaxpip_onnx.py` also canonicalizes dynamic
  batch shape helpers so exported `PolynomialLinearModel` and
  `PolynomialNeuralNetwork` graphs execute on `cuda:<id>` with
  `allow_cpu_fallback=false` and zero CPU fallback nodes (issue #202).

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
