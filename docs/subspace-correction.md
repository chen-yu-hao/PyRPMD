# Subspace quantum correction of RPMD rates

RPMD treats the environment of a reaction well (zero-point energy, adiabatic following,
recrossing) but misses deep tunnelling, resonances and coherence along the reaction
coordinate. The subspace correction keeps the full-dimensional RPMD rate and fixes only
that defect with an exact quantum calculation in one to three coordinates:

```latex
k \approx k_{\mathrm{RPMD}}(\text{full system}) \times C_S, \qquad
C_S = \frac{k_{\mathrm{QM}}(H_S)}{k_{\mathrm{RPMD}}(H_S)}
```

Numerator and denominator use the same subspace Hamiltonian `H_S` and the same reactant
normalization, so `C_S` contains only the RPMD method error inside `S`. The transfer is
multiplicative: the absolute rate of `H_S` may differ from the full system's by orders of
magnitude, but a shared prefactor cancels in the ratio. `C_S` is exact when `S` is the
whole system and an approximation otherwise. Nothing in this workflow marks a rate as
physically converged.

## Optional native `rate` stage (one `.inp`)

Use the normal rate input and runner; a second correction input is not needed:

```toml
# All ordinary system, potential, reaction, umbrella and recrossing sections
# remain in this same file.
task = "rate"

[rate.subspace_correction]
enabled = true
mode = "apply"             # default; alternatively "factor_only"
dimension = "auto"         # alternatively 1, 2, or [1, 2]
environment = "adiabatic"  # also "free_energy" or "frozen"
threshold = 0.5
reuse = true

[rate.subspace_correction.quantum]
refinement = 3
truncation_kt = 30.0
# Converge truncation_min (hartree) for stiff modes, especially at low T.
dense_limit = 8000

[rate.subspace_correction.rpmd]
walkers = 32
sweeps = 800
trajectories = 1024
seed = 17

# Validity gates (defaults shown). "allow" is an explicit, recorded override.
[rate.subspace_correction.validity]
unconverged_path = "reject"   # both path continuations must end at an accepted terminus
unstable_bath = "reject"      # the transverse Hessian of the path model must be stable
outside_grid = "reject"       # the RPMD estimator must stay on the V_S grid
max_outside_fraction = 0.01
on_invalid = "stop"           # or "diagnose": solve a rejected model anyway, never apply it
```

```bash
pyrpmd run calculation.inp
# Re-run only the postprocessing using the same input/output directory:
pyrpmd run calculation.inp --stage subspace
# Ordinary checkpoint restart also runs the enabled postprocessing at the end:
pyrpmd run calculation.inp --resume
```

A complete teaching input is [examples/rate/subspace.inp](../examples/rate/subspace.inp).
`enabled` defaults to false. When true, input resolution automatically enables
`output.centroid_path` **before** umbrella sampling. PMF and recrossing still use
the existing native rate executor and its ordinary backends. The subsequent
subspace solve runs once on the root rank, using the existing SciPy/NumPy quantum
and subspace RPMD implementations; this does not claim a new C++/MPI solver.

`mode="apply"` computes the factor and multiplies the saved RPMD estimate.
`mode="factor_only"` computes the same quantum/RPMD ratio without requiring a
full-system rate estimate or writing a corrected rate. It still needs the saved
system/PES and complete centroid moments: it is not a TS-only or sampling-free
calculation. In a full run, this mode does not skip the normal rate stages; use
`--stage subspace` to avoid rerunning those stages on existing data.

Temperature, bead count, masses, PES and output location are inherited from the
rate run. `[rate.subspace_correction.path]`, `.quantum` and `.rpmd` accept the
pipeline numerical controls; dimensional values use **atomic units**, not
unit-bearing strings. Native `.rpmd.windows` is an integer count (at least four).
The subspace section rejects unknown fields and does not accept `source`,
`output`, or a separate `nbeads`. Path environments are `frozen`, `adiabatic` and
`free_energy`; `relaxed` is only available for linear projections in the API.

The first native integration supports gas-phase `task="rate"` with **local
umbrella PMF sampling**, including the independent `[rate.pmf]` umbrella stage.
It rejects external PMFs, metadynamics and OPES because they lack the required
window centroid moments. `surface_rate` and rate-extension generations are not
supported by this native stage; the independent application API remains available.

### Outputs, restart and failure handling

All postprocessing artifacts live in `RUN/subspace/`: `path.npy`,
`C_S_1d.json` and/or `C_S_2d.json`, `subspace_pipeline.json`, `request.json`,
`status.json`, and, in apply mode, `rate_quantum_corrected.json`. The full-run
return value and `results/summary.json` include a `subspace_correction` status
record, without replacing the original rate fields. The root `summary.json`,
PMF, recrossing files, manifest and saved inputs are not rewritten by this stage.
A stage-only run writes only under `RUN/subspace`; its `status.json` is authoritative
if a previous full-run summary contains an older stage status.

Changing subspace grid, dimension or sampling controls does not change the RPMD
physics fingerprint. Editing the `.inp`/`.toml` to change these controls is allowed
when resolved physics still matches; PES/model file hashes remain enforced.
`--stage subspace` checks saved input/manifest identity before evaluating the PES
and refuses changed temperature, beads, reaction, sampling or system settings.
It is incompatible with `--resume` and warm-start options. No sampling or TS
preparation is performed on this path. Missing or incomplete centroid moments
must be regenerated by sampling; enabling the flag after a checkpoint cannot
reconstruct earlier statistics.

`reuse=true` for the native stage requires matching source data, numerical
settings, implementation hashes, bead count, temperature and saved factor checksums.
A changed request or damaged record is recomputed, never reused on temperature
alone. This is completed-result reuse, not an in-solver checkpoint. Reusing a factor
still rebuilds/checks the path and applies it to the current saved rate estimate.

An optional-stage exception in a full rate run is recorded as `status="failed"`
and emits a warning, while the uncorrected rate remains available. An explicit
`--stage subspace` failure records the diagnostic and returns a nonzero CLI exit.
Always check the latest `status.json`: an earlier corrected artifact can remain
on disk after a failed retry or a switch to factor-only mode. Neither a successful
command nor `status="complete"` establishes numerical or physical convergence.
Unpublishable source rates remain unpublished, and all corrected results retain
`physical_converged=false`.

The standalone `pyrpmd subspace-correct correct.toml` command is retained for
compatibility and calls the same numerical pipeline.

### Validity gates

A positive `k_QM(H_S) / k_RPMD(H_S)` is not yet a correction factor: the ratio
is only meaningful if `H_S` is a valid model. Three gates are evaluated, and a
model that fails one is recorded but never applied.

| Gate | Passes when | Evidence in `subspace_pipeline.json` |
| --- | --- | --- |
| `path` | both continuations ended at an accepted terminus — `asymptote_converged`, the reactant reference `xi <= reactant_xi` after climbing out of a complex, or a product well `product_drop_kt` below the barrier — and the 1D reactant reference slice lies in the free reactant channel | `path.reactant_termination`, `path.product_termination` (status, `accepted`, `reason`, steps, arc length, trailing energy range, end energy, drop below the barrier top, `minima`, `exit_coordinate`); `reactant_reference` |
| `bath` | no normal-plane curvature below `-soft_mode_frequency**2` along the 1D path model | `models.1.model_construction.significant_unstable` (arc length, imaginary frequency, character) |
| `grid` | at most `max_outside_fraction` of the ring-polymer estimator saw `V_S` outside its grid | `corrections.D.rpmd.diagnostics.grid_exit` |

**Path.** The continuations have no step budget. How far an asymptote lies is a
property of the potential — an ion–dipole tail needs far more steps than a
neutral one — and a path that stops at a count is not a path to the asymptote.
A continuation ends when the energy varies by less than `asymptote_tolerance`
over the last `10 * step` of arc length. The window is an arc length rather than
ten points, so halved steps at a bend cannot imitate a flat tail. It ends
unconverged with `asymptote_not_approached` when that trailing energy range has
not fallen by 10% during `asymptote_patience` (50) consecutive windows — a bound
coordinate, a potential leaving its fitted domain, a path that wanders — with
`step_collapsed` when the tangent cannot be resolved at the smallest step, and
with `potential_failure` on a failed or non-finite evaluation. `path.max_steps`
is optional; if set, reaching it is `max_steps_exhausted` and fails the gate.

A descending continuation that runs into a local minimum — a pre- or
post-reaction complex, typical of ion–molecule reactions with a submerged
barrier — stops there with `minimum_reached` (issue #191). Steepest descent
cannot leave a minimum, and climbing on along the incoming tangent only climbs
a wall: such paths used to wander for thousands of steps and end with
`asymptote_not_approached`, or pass through the well and reach the asymptote on
its far side. A minimum is judged per side. On the product side it is accepted
(`accepted: "product_well"`) when it lies at least `quantum.product_drop_kt`
(15 kT) below the highest energy on the path: products are committed there,
which is all the correction needs, and the 1D model then absorbs products only
inside the committed part of the well (`models.1.product_absorber`).

On the reactant side a minimum is not an end: both rates are normalized on a
slice of the *free* reactant channel, which lies beyond the complex. The
reactant continuation records the complex (`minima`) and climbs out of it along
decreasing rate coordinate `xi`: each point is the minimum of `V` on the
hyperplane one step further along `-grad xi` (mass-weighted, rigid-body modes
removed). These are, to first order, the minima of `V` on the `xi = const`
surfaces of the umbrella windows; `xi` decreases monotonically, so the path
cannot turn back into the well. It ends with `coordinate_target_reached`
(`accepted: "reactant_reference"`) at `xi <= path.reactant_xi` — by default
`xi = 0`, where the fragments are `R_inf` apart, the reactant reference of the
rate itself — or earlier with `asymptote_converged` if the energy is already
flat. `xi` is defined for every rate run, so the same rule serves neutral and
ionic reactions and needs no model of the long-range tail. Called without a
reaction coordinate (the library API), a reactant continuation that stops in a
complex still fails the gate with the reason `bound pre-reaction complex`.

The 1D model puts its reactant reference slice `absorber_fraction + 0.05` of
the path length from the reactant end, behind the reactant absorber. As for the
2D model, the slice must lie in the free reactant channel: the potential along
the path there within `quantum.reference_kt` (0.3 kT) of its value at the
reactant end. (The bare potential is used rather than `V_S`: the zero-point
energy of soft fragment rotations is noisy at that scale.) If the default slice
is not free, it moves to the last point of the flat channel that starts at the
reactant end and the reactant absorber shrinks in front of it
(`models.1.model_construction.reactant_reference.rule = "free_channel"`); if
there is no such channel (`"not_free"`), the path gate fails.

**Bath.** A vibrationally adiabatic model needs a bath that is bound in the plane
normal to the path. A negative normal-plane curvature has no zero-point energy;
the term is omitted, which lowers `V_S` exactly where the model has failed.
Imaginary frequencies below `soft_mode_frequency` (2.3e-4 Eh, 50 cm⁻¹) are soft
modes — free relative rotations of separating fragments, whose curvature tends
to zero with either sign; they are counted (`soft_complement_points`) and
`dropped_zero_point_bound` bounds the omitted energy. Larger ones fail the gate
and are classified from the vibrational Hessian in the basis of the tangent and
the normal plane: `tangent_misaligned` (one unstable vibrational mode, the
reaction mode, leaking into the bath because the path tangent is tilted away
from it by `tangent_to_lowest_mode_degrees`) or `ridge` (two or more: a
transverse coordinate is not a bath mode). Either way that coordinate belongs in
`S`, so `dimension = "auto"` then also builds the 2D model
(`dimension_escalation`). The finite-difference Hessian displaces the lightest
atom by `hessian_displacement` (5e-3 bohr); the former fixed mass-weighted step
moved hydrogen by 2e-5 bohr, where the gradient difference of a soft mode is
comparable to the evaluation noise of a fitted potential.

**Grid.** Outside its grid a gridded `V_S` is a constant with zero force.
`spline_outside_fraction` counts every gradient evaluation — rejected HMC
proposals, equilibration, and recrossing trajectories that left through a flat
channel end and are simply committed — so it cannot say how much of the
estimator was affected. `grid_exit` resolves it: per-stage counts (`stages`),
the bead points of the accepted TI and dividing-surface samples outside the grid
(`ti_sample_outside_fraction`, `surface_sample_outside_fraction`), and for the
recrossing trajectories the flux share that crossed the dividing surface again
after leaving, or whose centroid left where `V_S` still rises outwards by more
than 0.1 kT per thermal length (`recrossing_contaminated_flux_fraction`). The
gate uses the largest of the three (`estimator_outside_fraction`).

When a gate fails, `C_S_Dd.json` carries `valid: false`, `factor: null`, the
ratio as `diagnostic_factor`, the gate record in `validity` and both component
rate results; `apply_subspace_correction` refuses it. With `on_invalid = "stop"`
a model whose path or bath gate fails is not solved at all. If no model passes,
the report has `status = "invalid_model"`, no corrected rate is written (a
corrected rate left by an earlier run is removed), `status.json` of the native
stage carries the same status, and `pyrpmd subspace-correct` exits nonzero.

## One command from a rate run

```bash
pyrpmd subspace-correct correct.toml
```

`correct.toml` names a finished rate run (`[source] rate_run = "..."`) and optional
`[path]`, `[subspace]`, `[quantum]`, `[rpmd]` and `[output]` settings
(`benchmarks/clhbr_subspace/correct.toml` is a complete example). The rate run must have
been made with

```toml
[output]
centroid_path = true
```

which makes every umbrella window of the rate PMF stage accumulate, for each recorded ring
polymer aligned onto the transition-state geometry (mass-weighted Kabsch, after assigning
the nearest equivalent reaction path): the mean centroid, the centroid covariance, the
bead-about-centroid covariance `Sigma(xi)` and the mean interatomic distances. The moments
are additive, so replicas and checkpoint resumes combine exactly; they are written to
`centroid_path.npz` and read back with `mean_path_from_rate_run(run_dir)`. The switch is
analysis-only (it is not part of the physics fingerprint and does not change the
trajectories) but disables one fused native fast path, so it is off by default.
Extend the umbrella windows past the dividing surface (e.g. `stop = 1.45`) if the
reaction path bends after the transition state.

The command then runs:

1. **Path source.** `[path] source = "centroid"` is the historical mean-centroid path.
   `"bead_weighted"` shifts each locked-window centroid along the transverse principal
   eigenvector of the bead covariance `Sigma(xi)` by its bead RMS `sqrt(lambda_max)`, with
   the sign chosen from the local curvature. `"instanton"` uses the folded half-ring from
   `pyrpmd.instanton.optimize_instanton` below the crossover temperature and records a
   centroid fallback at or above `T_c` (or when the orbit fails to converge). `"least_action"`
   scans a one-parameter interpolation between the sampled MEP and its endpoint chord and
   selects the smallest one-dimensional WKB action. All sources retain their selection and
   fallback diagnostics in `subspace_pipeline.json`.
2. **Mean path and path to both asymptotes.** For the centroid source, windows where the
   aligned centroids stay compact (spread at most twice that near the transition state) are
   kept; near the reactant asymptote the fragments rotate freely and a mean geometry has no
   meaning. `xbar(xi)` is fitted by local quadratic regression in `xi` (window means carry
   transverse noise). For a linear reference the mean path is built from mean bond lengths
   along the axis: the Kabsch rotation about the axis is undetermined, and axial projections
   of librating light atoms are shorter than the bonds. Every selected core is then continued
   by steepest-descent path following (minimum on a sphere around each point, forward hemisphere, rigid-body modes
   excluded; uphill segments use the hyperplane minimum), with step halving at sharp turns.
   The continuation runs until the energy is flat over ten step lengths of arc. A product
   continuation may stop in a deep post-reaction complex; a reactant continuation climbs out
   of a pre-reaction complex along decreasing `xi` to `xi = 0`. There is no step budget, and
   each end records how it terminated and whether that end is accepted (see
   [Validity gates](#validity-gates)).
3. **Curvature × width.** From `Sigma(xi)` along the sampled windows
   (`rho = sqrt(2 n^T Sigma n)` along the curvature normal) and, along the whole path, with
   the harmonic bead width of the mode along the normal. Above the threshold (default 0.5)
   a 2D subspace is added.
4. **1D model** along the path (normal-cut vibrationally adiabatic, rigid-body modes
   removed) and its `C_S`. With large curvature the normal cuts cross inside the bend and
   the adiabatic potential develops a cusp; `C_S(1D)` is then not trustworthy, which is
   what the curvature test says.
5. **2D model**: the plane through the transition state spanned by the rate dividing-surface
   normal (mass-weighted `grad xi`, so the 2D dividing line is the rate's `xi = 1` surface)
   and the principal direction of the path within the complement; for a collinear atom
   transfer this is exactly the collinear plane. The curvature-normal candidate is reported
   with its overlap with that plane. Absorbing regions are keyed to arc length along the
   path, so hairpin channels that leave the grid on the same side are both absorbed; the
   rate is normalized by the quantum line density on the reference slice, matching RPMD
   constrained TI along axis 0; an energy-truncated DVR keeps the dense exact route
   affordable. Both halves of `C_S` share one `V_S`: it is projected on a grid
   `refinement` (3) times finer than the DVR; the DVR uses every third node and RPMD the
   spline of the fine grid (a spline through DVR nodes alone rings across narrow valleys).
   The box only extends to a reference slice where the reactant channel is within
   `reference_kt` (0.3 kT) of its asymptote — `C_S` needs both rates on the same slice,
   not at infinity — and products are absorbed once the path is `product_drop_kt` (15 kT)
   below its maximum. The default grid is anisotropic: a third of the narrowest
   in-plane vibrational width along the transfer axis, a quarter of the shortest de
   Broglie wavelength along the path axis.
6. **Nested sensitivity** `ln(C_2D/C_1D)` selects the dimension, and
   `rate_quantum_corrected.json` is written next to the run. Every intermediate result is
   in `subspace_pipeline.json`.

**Exo- or endothermic reactions.** The box holds deep states of the lower channel whose net
flux vanishes but whose Boltzmann weight is enormous (`exp(25)` for Cl + HBr at 300 K); any
absorber reflection then swamps the reactive flux. Both models therefore restrict the
thermal flux operator to energies above the higher channel threshold minus one `kT`
(`exact_quantum_rate(..., energy_floor=...)`), which is exact in the continuum limit because
the cumulative reaction probability vanishes below that threshold. Without it the Cl + HBr
1D rate changes by 10% and the 2D rate by orders of magnitude when the absorber changes.

**Cl + HBr (BrHCl_LEPS1, 300 K)** is worked through in
[docs/validation/2026-11-08-clhbr-subspace](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-08-clhbr-subspace/README.md):
curvature × width is 1–12 at the H-transfer corner, the 1D model develops a cusp there
with the first, kinked path (normal cuts cross; with the corrected path C_S(1D) = 0.99), and
in the collinear plane RPMD matches the exact rate within its statistical error
(C_S(2D) = 1.02 ± 0.29 in the pipeline run, 0.9 ± 0.2 with a second RPMD run), with the 2D
quantum rate converged to 2.5% in the grid. The corrected rate is RPMD × C_S(2D).

## Workflow

1. **Build `H_S`** (`pyrpmd.analysis.subspace_model`). Coordinates are mass weighted and
   axis 0 is the reaction coordinate; the dividing surface is a value on axis 0.
   - `model_from_function(V, axes, masses=..., dividing_surface=...)` — the Hamiltonian
     already has 1–3 coordinates.
   - `project_linear_subspace(V, grad, reference, masses, basis, axes, environment=...)`
     — a linear subspace of a larger system. The complement is `"frozen"`,
     `"relaxed"` (minimized) or `"adiabatic"` (relaxed plus its harmonic zero-point
     energy, the default).
   - `project_path_subspace(V, grad, path, masses, axis)` — a curved 1D path, for
     example the ring-polymer mean path from constrained-centroid sampling
     (`ring_polymer_mean_path`), with a normal-cut adiabatic bath and arc-length
     coordinate.  `zero_point="harmonic"` retains the historical projected-Hessian
     frequencies; `zero_point="rph"` uses the same quadratic reaction-path
     potential Hessian `N.T @ H @ N` on the linear normal cut (no additional
     gradient--curvature subtraction); and `zero_point="dvr"` solves
     independent sinc-DVR cuts along the local normal modes.  For
     `environment="free_energy"`, the DVR option uses the thermal cut free energy
     instead of the ground-state energy.  `dvr_points`, `dvr_half_width`, and
     `dvr_spacing` are recorded in the model provenance so the normal-cut
     convergence can be reported.
2. **Exact quantum rate** `exact_quantum_rate(model, beta)`
   (`pyrpmd.analysis.subspace_quantum`). Thermal flux–flux formula with an absorbing
   potential on a sinc-DVR grid:
   - `method="sylvester"` (default up to 1600 grid points) solves
     `H†X − XH = iF` once; there is no time cutoff, so long-lived resonances are exact.
   - `method="flux_eigenstates"` (larger 2D/3D grids) finds the significant thermal flux
     eigenstates by Lanczos with a Chebyshev Boltzmann operator and propagates them with a
     split operator. Norm still trapped at `max_time` is assigned by the absorption
     branching ratio of the last half of the run; the estimates at T/2, 3T/4 and T are
     reported as `diagnostics["extrapolation"]`.
3. **RPMD rate of the same `H_S`** `ring_polymer_rate(model, beta, nbeads)`
   (`pyrpmd.analysis.subspace_rpmd`): centroid thermodynamic integration (exact mean
   force, HMC with exact free ring-polymer propagation and Metropolis test) plus
   recrossing trajectories. The reported error combines walker statistics, a TI
   quadrature estimate and the transmission bootstrap.
4. **Correction factor** `subspace_correction(model, beta, nbeads)` returns a
   `SubspaceCorrection` (`factor`, `log_standard_error`, both sub-results) that can be
   saved with `.write(path)`. Use the bead number of the full RPMD run.
5. **Apply it.** `apply_subspace_correction(rate, correction, standard_error=..., beta=...)`
   for a rate in hand, or `correct_rate_run(run_dir, correction)` for a saved PyRPMD
   `rate`/`surface_rate` run. The latter checks `beta` against `input/resolved.json`,
   never touches the source files, records their SHA-256 and writes
   `rate_quantum_corrected.json`. A run whose rate is not publishable keeps
   `corrected_rate_*` empty and reports only the corrected estimate.

```python
import numpy as np
from pyrpmd.analysis import (project_linear_subspace, subspace_correction,
                             correct_rate_run)

sub = project_linear_subspace(V, grad, x_ts, masses, basis_mw, [np.linspace(-150, 150, 301)],
                              environment="adiabatic", dividing_surface=0.0)
corr = subspace_correction(sub, beta, nbeads=32)
corr.write("C_S_300K.json")
correct_rate_run("runs/rate-300K", corr)
```

## Hybrid CPU and MPI execution

The gridded RPMD solver uses a C++17 extension by default when available. It
copies SciPy's cubic B-spline knots and coefficients and evaluates the same
potential and analytic gradient in C++. HMC proposal trajectories and paired
recrossing trajectories release the GIL and run without Python callbacks.
OpenMP distributes independent ring polymers across threads; SIMD covers the
normal-mode updates. Free ring propagation uses the existing FFT implementation
(direct Fourier transforms for non-power-of-two bead counts).

Python still owns the NumPy random stream, randomized proposal lengths, the
Metropolis test, shared burn-in step-size adaptation, walker statistics, TI,
and paired-trajectory bootstrap. Thread count does not change random draws or
reduction order. The FFT and spline arithmetic can differ slightly from the
Python reference; long chaotic trajectories need statistical comparison as
well as short trajectory comparisons. The physical definitions, error formulas
and existing result fields are retained.

For a single calculation, pass `threads=32` to `ring_polymer_rate`. Select
`backend="python"` for the reference implementation or `backend="native"` to
require the compiled kernel. `backend="auto"` reports a fallback reason if the
extension is missing or a model has Python potential callbacks. Callbacks are
never silently replaced by an interpolated potential: use an explicit
`SubspaceModel` containing the intended grid to enable native propagation.

The quantum solver retains SciPy eigensolvers and complex Schur decomposition.
Dense DVR assembly avoids full Kronecker temporaries. The Sylvester equation is
solved in triangular blocks with LAPACK `ztrsyl` and BLAS matrix updates;
LAPACK scaling or near-degeneracy triggers the original full solve. A dense
memory estimate is checked before allocation. This changes neither the DVR
grid nor the flux operator, absorbing potential or reactant normalization.

Add an execution section to a pipeline TOML to overlap QM/RPMD and, when both
dimensions are requested, all four stages of `C_S(1D)` and `C_S(2D)`:

```toml
[execution]
backend = "process"
cpus = 32
workers = 4
blas_threads = 8
memory_limit_mb = 8192
resume = true
# checkpoint_dir = "subspace-tasks"  # default: output directory / tasks

[rpmd]
backend = "native"
```

`cpus` is the total CPU budget and `workers` the maximum number of concurrent
tasks. Each process receives `floor(cpus / workers)` OpenMP threads
(using fewer workers if there are fewer unfinished tasks). Memory estimates
include dense linear algebra or RPMD state, side histories and bootstrap
temporaries; tasks are admitted only while their combined estimates fit the
budget. BLAS uses at most `blas_threads` (default 8) and never exceeds that
process's CPU allocation. Large thread teams can slow the sequential stages
of complex Schur decomposition; the cap is independently adjustable for the
installed BLAS and matrix size. Set `OMP_WAIT_POLICY=PASSIVE` before launching
mixed Python/OpenMP workloads so idle dynamics threads sleep during sampling
and analysis. These are conservative working-memory estimates, not OS memory limits.
Set the budget below the job's memory allocation to leave room for Python,
loaded libraries and the coordinator. BLAS thread limits are scoped inside
separate processes, avoiding conflicting thread limits in Python thread pools.

The same scheduler is available through Python:

```python
from pyrpmd.analysis import subspace_correction, subspace_corrections

# Guard process-spawning code with if __name__ == "__main__" in a script.
corr = subspace_correction(model, beta, 32,
    execution={"backend": "process", "cpus": 32, "workers": 2})
corrections = subspace_corrections({
    "1d": {"model": model1, "beta": beta, "nbeads": 32},
    "2d": {"model": model2, "beta": beta, "nbeads": 32},
}, execution={"cpus": 32, "workers": 4, "memory_limit_mb": 8192})
```

For multiple nodes, install the existing `mpi` extra and set
`execution.backend = "mpi"`. Launch the same command on all ranks, for example
with two ranks per node on two nodes:

```bash
mpiexec -n 4 --map-by ppr:2:node:PE=16 pyrpmd subspace-correct correct.toml
```

The launcher example uses Open MPI; use the equivalent binding options for
your MPI implementation and scheduler. `cpus` and `memory_limit_mb` are
**per-node** budgets in MPI mode. Each shared-memory node divides them among
its ranks; `workers` applies only to the local process backend. Rank zero
builds models and writes the pipeline report, dynamically dispatches tasks,
and also computes. Only gridded models and quantum grid arrays travel between
ranks; no live PES object or pipeline closure is sent. All ranks must call
the Python API. Rank zero alone supplies `cases` to `subspace_corrections`;
other ranks may pass `None`. There are no MPI collectives inside time steps.
A single correction currently supplies two distributed tasks; two dimensions
supply four. This is task-level scaling, not distributed dense linear algebra.

Each completed quantum or RPMD result is atomically checkpointed, including
the RPMD time series. Identity includes the grids, masses, temperature,
beads, potential metadata, sampled absorber/truncation arrays, numerical and
sampling settings, seed, solver implementation and NumPy/SciPy versions.
Changing thread counts or rank placement can reuse results. Changed physical
or sampling inputs cannot. With an execution section the task cache replaces
the older `output.reuse` shortcut, which checked temperature alone. A failed
or interrupted task reruns from its original seed; completed tasks survive a
job/launcher restart. This is restart between tasks, not continuation within
an unfinished HMC chain or recovery of a failed MPI rank inside a live job.

Diagnostics include actual OpenMP team size, native/Python polymer-step
counts, Python potential callbacks, fallback reasons, per-task CPU/wall time,
BLAS limits, memory estimates and checkpoint reuse. `physical_converged`
remains false. The reproducible H+H2 benchmark is
`benchmarks/h3_subspace/parallel.py`; its fixed 300 K/P32 budget tests execution
and numerical compatibility, and does not establish a converged reaction rate.

## Choosing the subspace

`S` must contain the RPMD defect, not merely the strongly coupled modes. The research behind
this module found that static coupling strength is the wrong criterion: a mode that follows a
smooth path adiabatically is handled by RPMD itself, so adding it to `S` changes `C_S` by only
a few percent. Two checks decide:

- `curvature_width_criterion(curvature, width)` — the path curvature in mass-weighted
  coordinates times the transverse quantum width (√2 × bead RMS). Values ≪ 1 let a 1D path
  carry the coupling; values near 1 mean the vibration straddles the bend (collinear H + H₂:
  0.80–1.01) and the transverse coordinate belongs in `S`. The 0.5 threshold is provisional.
- `nested_subspace_sensitivity([C_1D, C_2D, ...])` — the change of `ln C_S` when a coordinate
  is added. It is the final verdict and costs one larger quantum calculation.

[Subspace transfer theory](subspace-transfer-theory.md) derives why: static environment effects
(zero-point energy, imaginary-time corner cutting), which curvature × width measures, cancel from
the transfer error; what remains is the environment's influence on RPMD's dynamical error. It
also gives the first-order-exact `environment="free_energy"` projection and the finite-bead
factor `harmonic_bead_factor` for environment modes.

## Numerical parameters to converge

| Parameter | Where | Symptom when too small |
| --- | --- | --- |
| Box length and absorbing-potential width/strength | `exact_quantum_rate(cap_width=, cap_strength=)` | Reflection from the absorber; at 200 K a 16 bohr Eckart box is 0.7% high, 32 bohr 0.06% |
| Grid spacing | model axes | Rate drifts when refined |
| Truncation energy (energy-truncated DVR) | `exact_quantum_rate(truncation_energy=)`, pipeline `truncation_kt` and `truncation_min` (Eh) | The removed points are hard walls. The rate converges only once the margin above the local valley floor exceeds about six quanta of the stiffest transverse mode (walls at ~5 ground-state widths). A separable 2D Eckart model (transverse 0.010 Eh) at 200 K is 12% low with 0.016 Eh above barrier + ZPE, 1% low at 0.03 Eh, and exact (1 − 4×10⁻⁶) at 0.07 Eh. The diagnostic `truncation_margin` reports the smallest margin. The 30 kT default is too small for stiff modes at low temperature: set `truncation_min` to about six quanta |
| Propagation time and step | `max_time`, `dt` (iterative route) | `relative_extrapolation_spread` large |
| TI windows | `ring_polymer_rate(windows=)` | `ti_quadrature_log_error` comparable to the statistical error |
| Beads | `nbeads` | `C_S` should correct the same finite-`P` estimator as the full run |
| Sampling | `walkers`, `sweeps`, `trajectories` | `log_standard_error` |

Ring polymers that leave the model grid sample a constant continuation of the spline.
`spline_outside_fraction` counts all such evaluations, including committed recrossing
trajectories that have left through a flat channel end; the part that reaches the estimator
is `grid_exit.estimator_outside_fraction`, which must stay below `max_outside_fraction`
(see [Validity gates](#validity-gates)). Enlarge the grid if it does not.

## Validation

`examples/subspace_correction/eckart_bath_2d.py`: symmetric Eckart barrier (V₀ = 0.425 eV,
a = 0.66 bohr, m = 1060) with a vibration that softens by 30% and shifts by 0.2 bohr at the
barrier. The 1D subspace is projected from the 2D potential (relaxed vibration plus its
zero-point energy). Results and settings:
[docs/validation/2026-11-07-subspace-correction](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-07-subspace-correction/README.md).

Unit tests (`tests/test_subspace_correction.py`) check the exact solver against the analytic
Eckart transmission, the iterative route against the dense route, a separable 2D rate against
its 1D rate, single-bead RPMD against classical TST, the projections against analytic
adiabatic potentials, and the provenance and temperature checks of `correct_rate_run`.

## Relation to the other quantum-correction tools

[Quantum correction](quantum-correction.md) covers response-level (spectral) corrections, the
online Ehrenfest propagator and flux-side curve transfers. For thermal rate constants use this
module. `hybrid_flux_side_correction(..., mode="ratio")` and
`adaptive_quantum.matched_rate_defect(..., mode="ratio")` apply the same multiplicative
transfer to curves or plateau values computed elsewhere; their additive default is not
invariant to the subspace model's absolute rate.


The native `[rate.subspace_correction]` table also accepts the path-model
selectors `curvature = "none" | "small"` and
`zero_point = "harmonic" | "rph" | "dvr"`, plus `dvr_points`,
`dvr_half_width` and `dvr_spacing`. Its `[rate.subspace_correction.path]`
table accepts `source = "centroid" | "bead_weighted" | "instanton" |
"least_action"` and the corresponding `least_action_points`, `instanton_beads`
and `instanton_max_iterations` controls, and `reactant_xi` (default 0, must be
below 1): the value of `xi` at which a reactant continuation that climbed out of
a pre-reaction complex ends. `crossover_temperature`, when supplied
as an instanton fallback, is in kelvin. These choices enter the factor-cache
identity; changing them requires a new factor calculation. The standalone
`subspace-correct` entry point uses the same content-based cache checks.
