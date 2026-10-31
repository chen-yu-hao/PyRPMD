# Free-energy surfaces and enhanced sampling

## One FES entry point

Select the sampler with `task="fes"` and `[fes] method="umbrella"`,
`"metadynamics"` or `"opes"`. Omitting `fes.method` preserves the historical
umbrella default. The common CV definitions, sampling controls and analysis
settings may live in `[fes.cvs]` / `[[fes.cvs]]`, `[fes.sampling]` and
`[fes.analysis]`; their existing top-level forms remain accepted. Put bias
parameters in the selected method's subtable:

```toml
task = "fes"

[fes]
method = "metadynamics"

[fes.sampling]
equilibration_steps = 100
production_steps = 600
sample_interval = 5
replicas = 2

[fes.metadynamics]
cvs = ["x", "y"]
mode = "well_tempered"
sigma = ["0.25 bohr", "0.25 bohr"]
height = "0.02 kcal/mol"
pace = 20
bias_factor = 8.0

[fes.analysis]
method = "bias_estimate"
bins = [24, 24]
bootstrap = 0
```

The system, potential, temperature, beads, dynamics, output and parallel
sections retain their existing meanings. Define `x` and `y` with ordinary CV
specifications. The complete examples below supply those definitions.
Specifying both top-level and nested versions of a section is an error.
An inactive method's table is rejected, even if empty; fields from another
method are not silently accepted. For example, umbrella `stiffness` does not
belong in `[fes.metadynamics]`, and `[fes.opes]` cannot accompany
`method="umbrella"`. Adaptive-umbrella and replica-exchange controls apply to
the umbrella method only.

| Method | Supported CV dimensions | Reconstruction | `uncertainty_kind` |
| --- | --- | --- | --- |
| `umbrella` | 1–3 | MBAR with bin-volume correction | `block_bootstrap` |
| `metadynamics` | 1–3 | Standard or well-tempered bias estimate | `replica_spread` |
| `opes` | 1–2 | KDE or frozen-bias reweighted histogram | `replica_spread` |

All three use the same CV selection/unit validation, propagation and sampling
clock, atomic checkpoint boundaries, and replica dispatcher. The bias-update
policy and reconstruction remain specific to the algorithm. This preserves
the existing seed identities, thermostat scheduling, native fixed-bias block
path, and pre-deposition observation semantics.

### Compatibility and continuation

`task="metadynamics"` and `task="opes"` still work as deprecated aliases and
emit a `FutureWarning` with the replacement syntax. Their old top-level bias
tables remain supported. Existing examples retain some of these spellings to
exercise compatibility. New checkpoint identities include the method and
normalize the two aliases; changing the method cannot reuse a checkpoint.
The compatibility reader also checks the exact historical fingerprint, so
old independent umbrella, MetaD and OPES checkpoints can continue with their
saved coordinates, momenta, random-generator state and bias history. Resume
authenticates an old run manifest before migrating the saved entry spelling.
It does not relax checks on physical parameters, CVs or seeds.

### A common result contract

Every method writes the following layout:

| File | Contents |
| --- | --- |
| `fes.npz` | Authoritative, integrity-checked `FESResult`, including its metadata |
| `fes.csv` | Common grid/free-energy/uncertainty/mask table |
| `fes.json` | Schema, method, units and method-specific provenance |
| `summary.json` | Human-readable run diagnostics and file references |
| `results/summary.json` | CLI runner summary, including software/run metadata |

Umbrella `fes.h5` and method-specific sampling histories remain supplementary
diagnostics. The core contract contains `edges_0`, …, `grid` (cell centers),
`free_energy`, `uncertainty`, `mask`, `uncertainty_mask`, `cv_units`,
`energy_unit`, `method` and `uncertainty_kind`. Energies are in hartree; CV
units are canonical length, angle or dimensionless units. Unsupported bins
are masked. Unavailable uncertainty is NaN with an uncertainty mask, not zero.
Historical `standard_error` and `replica_spread` arrays remain compatibility
extras; new consumers use `uncertainty` and inspect its kind.

```python
from pyrpmd.analysis import FESResult

surface = FESResult.read("runs/my-fes")
supported = ~surface.mask
print(surface.method, surface.uncertainty_kind, surface.cv_units)
```

`analyze`, reports, comparison and projection use this contract; legacy file
formats are adapted in its reader. `analyze --recompute` still reconstructs
from authoritative sampling checkpoints and can change analysis settings
without propagating dynamics. Reports display orthogonal slices of 3D grids.
Cross-method comparisons can overlay surfaces when the physical system, CVs,
measure and static walls match. Each curve retains its method and uncertainty
meaning; samples and errors from different algorithms are not pooled.

Umbrella errors come from the existing correlated block bootstrap, including
synchronized resampling within replica-exchange ladders. A bin's primary
`uncertainty` (and compatibility `standard_error`) is unavailable if a requested
draw lacks support there or the bootstrap reference is unsupported. Conditional
`supported_draw_standard_error` and support counts remain diagnostics; they do
not replace the primary error. MetaD/OPES errors
are the sample standard deviation of independent replica estimates after
gauge alignment on supported bins. They are descriptive spread, not a
standard error of the mean or a calibrated confidence interval. A single
history has unavailable spread. Shared MetaD walkers form one correlated
history and do not supply independent replica errors.

### Three-dimensional umbrella sampling

Use three selected scalar CVs (or a CV stack totaling three components),
three grid axes, and a scalar, three-component diagonal, or 3×3 positive
definite stiffness matrix in `[fes.umbrella]`. Analysis bins and ranges must
have three entries. The MBAR estimator and bootstrap use the complete 3D
histogram and bin volumes; no 2D projection is substituted during sampling.
The special `rate_flux` measure retains its existing scalar/2D rate-coordinate
restrictions. See [`harmonic3d.inp`](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/fes/harmonic3d.inp):

```bash
pyrpmd run examples/fes/harmonic3d.inp --output runs/umbrella3d
pyrpmd analyze runs/umbrella3d --recompute --bins 6 6 6 --bootstrap 10
```

For FES PILE-L and fixed-window replica exchange, see the
[replica-exchange guide](replica-exchange.md).
Probability-based [OPES](opes.md), the optional [PLUMED bridge](plumed.md),
and [transition-interface path sampling](transition-interface-sampling.md)
have separate guides, examples and estimator contracts.

These workflows sample ring-polymer configuration distributions. Biased
trajectory time is not a reaction rate, and no infrequent-metadynamics rate
estimator is provided. Production rate coefficients still use the rate
workflow with its prescribed PMF measure and unbiased recrossing dynamics.

## Adaptive umbrella placement

Add `[adaptive_umbrella]` to a one-, two- or three-dimensional `task="fes"` job, or to
a `task="rate"` PMF job:

```toml
[adaptive_umbrella]
enabled = true
pilot_steps = 1000
# Omit pilot_equilibration_steps to inherit sampling.equilibration_steps.
max_rounds = 8
max_windows = 128
max_pilot_steps = 16000
target_overlap = 0.03
min_effective_samples = 20.0
min_spacing = 0.000001
```

The initial `[umbrella]` specification defines the domain and positive definite
stiffness matrix. Each pilot round independently equilibrates every window,
then samples its fixed bias. The pilot-only MBAR calculation constructs an
overlap graph. New schema-3 plans use the normalized score
`sqrt(O_ij * O_ji / (O_ii * O_jj))`, recorded as `normalized_mbar_cosine`, with
an edge when the score is at least `target_overlap`. The score lies in [0, 1];
identical window distributions score 1 even when there are many windows.
This avoids the `1 / number_of_windows` dilution of the former transition
score for identical windows. It is neither a replica-exchange acceptance
probability nor a guarantee of global mixing. `target_overlap=0.03` remains a
heuristic screening threshold that should be checked against independent
results for the system. Replicas
at the same center are pooled for the *overlap solver*, while each trajectory
retains its own autocorrelation diagnostic. Window effective sample counts
sum `N/g` across independent replicas. The slowest coordinate or centered
coordinate-square trace sets `g`, so rapid sign oscillations do not conceal
slow amplitude relaxation. Periodic CVs use first and second sine/cosine
harmonics, avoiding false decorrelation at the coordinate branch cut.
The production FES/PMF effective sample counts and bootstrap correlation blocks
use the same diagnostics; constant derived observables are ignored. This does
not alter the MBAR weights or free-energy point estimate.

If the graph is disconnected at the requested threshold, the algorithm
inserts midpoints along the shortest links between components. Distances use
the dimensionless metric `sqrt(beta * delta.T @ K @ delta)`; `min_spacing`
uses this metric too. Periodic displacements follow the shortest wrapped
arc. The stiffness stays fixed. Insufficient per-window ESS doubles the next pilot's sampling duration. An
unmet coverage target also doubles it once no further windows can be inserted;
a placement-only refinement otherwise keeps the current length. Each
round uses a separate deterministic seed. `max_pilot_steps` caps the duration
per window/replica (default: 16 times `pilot_steps`); reaching the cap without
adequate ESS or coverage and without further insertions stops instead of repeating equally
short pilots. Defaults allow 8 rounds and 128 windows, including a 96-window
initial grid. These are resource limits, not a recommendation to use all windows.

For a declared domain, optionally add a spatial coverage target to the same
table (the example below has two Cartesian CVs):

```toml
coverage_bins = [4, 8]
coverage_ranges = [["-0.5 bohr", "0.5 bohr"], ["-1 bohr", "1 bohr"]]
min_bin_effective_samples = 5.0
target_coverage = 0.95
```

`coverage_bins` and `coverage_ranges` must be supplied together. Numeric
bounds use canonical CV units. This pilot grid is independent of the final
`analysis.bins`: choose enough cells to resolve the domain's sampling holes
without requiring an impractical number of pilot samples. The default omits
this additional target; graph connectivity alone does not test domain coverage.
For each cell, the diagnostic is `(sum w)^2 / sum(g_stream * w^2)`, using
unbiased MBAR weights and the per-stream correlation factor. Empty cells count
as unsupported. Deficient cells propose additional windows at their centers,
subject to `max_windows` and `min_spacing`. Existing centers are not duplicated;
longer pilots can improve their support. `adaptive-plan.json` retains the full
cell counts, effective sample estimates, coverage fraction and decisions.

Once all enabled criteria pass, the window set is frozen. **Fresh production streams
start from newly equilibrated states and a different seed. No pilot samples
enter the production MBAR or UI estimate.** The saved `adaptive-plan.json`
contains a checksum, initial/final windows, every decision, diagnostics and
seeds. `adaptation/round_*/checkpoints/` contains the pilot states;
`checkpoints/window_*.npz` contains only final production. Interruptions resume
the current pilot or the frozen production without redesigning an already
committed stage. Changing analysis bins/bootstrap does not resample pilots.
Existing version-1 plans retain their saved defaults and geometric schedule.
Version-1 and version-2 plans retain the legacy `mbar_symmetric_transition`
score `sqrt(O_ij * O_ji)` and their committed decisions. The same numerical
threshold has a different meaning under the two metrics; use a new output
directory to assess a fresh plan with normalized overlap.

Exhausting `max_rounds`, or finding no admissible midpoint while the overlap
graph is disconnected, stops with `adaptation_incomplete`; production is not
silently launched. `max_windows` limits insertion: a connected set may still
pass ESS/coverage targets after longer pilots without adding more windows.
A reached pilot
target is a local overlap/ESS/coverage diagnostic, not proof of equilibration,
adequate CVs, final histogram coverage, or physical convergence. The local ESS
does not include uncertainty in fitted MBAR offsets or unseen slow variables.
The algorithm refines a supplied
domain and does not automatically discover unspecified distant basins.

Previously frozen plans keep their saved decisions on restart; their summaries
identify legacy coordinate-only ESS diagnostics. Use a fresh output directory
to reevaluate them under the new moment and optional coverage criteria.

An incomplete pilot can be continued with explicit, **increased** budgets:

```bash
pyrpmd resume runs/adaptive --adaptive-max-rounds 12 --adaptive-max-windows 192 --adaptive-max-pilot-steps 32000
```

The checksummed plan records each extension. Completed rounds, seeds, physical
inputs, overlap/ESS targets, and production settings stay fixed. Allow at least
one more round when increasing another exhausted limit. Frozen plans cannot
be reopened for adaptation; version-1 plans support rounds/windows extensions
but have no `max_pilot_steps` setting. Python uses
`resume_run(path, adaptive_budget={"max_rounds": 12})`.

When production duration is omitted, enabled adaptive jobs and metadynamics
now start with **1000 fs**, rather than the ordinary pilot's 200 fs. Explicit
`sampling.production`/`production_steps` always take precedence. This is only
a starting budget: an accepted pilot can still be longer than production or
have better mixing. The adaptive summary rechecks overlap at the pilot target,
ESS and any explicitly requested coverage grid on the independent production
streams. Read `adaptive_umbrella` in `RUN/results/summary.json` for FES or
`pmf.adaptive_umbrella` in the same final-summary file for gas-phase rates.
Gas-phase runs also retain `adaptive_umbrella` in `RUN/rate_pmf.json` and
`RUN/umbrella/summary.json`. Its fields distinguish the stages:

| `adaptive_umbrella` field | Interpretation |
| --- | --- |
| `pilot_target_met` | The pilot passed the requested selection criteria |
| `target_met` | Compatibility alias for `pilot_target_met` |
| `overlap_metric` | Normalized schema-3 metric or the saved legacy metric |
| `production_target_met` | Production independently passed all enabled local criteria |
| `production_overlap_connected_at_target` / `production_spanning_overlap` | Production connectivity and weakest score in the strongest spanning tree |
| `production_effective_samples_target_met` | Every production window passed the requested ESS |
| `pilot_coverage` / `production_coverage` | Separate diagnostics for an explicitly requested coverage grid |
| `coverage` | Compatibility alias for pilot coverage |

Final pilot and production lengths are reported separately. A failed production
check produces warnings; it does not automatically extend trajectories, delete
the estimate or establish `physical_converged`. A non-null FES or rate is not a
substitute for these checks.
Increase production in a fresh run and compare independent replicas and PMFs;
more windows alone do not increase each trajectory's independent sample count.

At the default 0.1 fs time step, `pilot_steps=1000` is only **100 fs**, and the
default `max_pilot_steps=16000` is 1.6 ps per window and replica. Pilot
equilibration inherits `sampling.equilibration_steps` unless supplied explicitly:
with otherwise default input it is 1000 steps, or 100 fs. The default save
interval is 1 fs and there is one replica per window. ESS 20 is a pilot screening
criterion; 20 default bootstrap draws are a short analysis budget. Neither
defines an accuracy for the PMF or rate. See the
[complete default table](manual/input.md#defaults) and
[scientific acceptance procedure](manual/science.md#assessing-adaptive-umbrella-results)
before choosing production settings.

The rate workflow retains its `rate_flux` metric bias during both stages;
the downstream PMF and recrossing preparation use the actual frozen window
centers from the sampled streams. Generic FES retains `cv_density`.
Rate PMFs default to Gaussian UI; adaptive MBAR overlap diagnostics do not change
that estimator or validate its Gaussian approximation. For a diagnostic comparison,
set `rate.pmf_method="mbar"` and keep `rate.compare_ui=true`. UI model tails can
extend beyond observed endpoint support; their warning must be addressed before
treating a rate as reliable. Histogram MBAR requires connected, sampled support
from the reference to the dividing surface and still needs bin convergence.
Rejected rate-PMF bootstrap draws make the primary barrier/rate uncertainty
unavailable; `supported_draw_*` diagnostics describe only the accepted subset.
Generic FES likewise withholds a bin's primary `standard_error` if any requested
bootstrap draw lacks support there, retaining `supported_draw_standard_error`
and `bootstrap_valid_per_bin` as conditional diagnostics.

For real H + H₂ settings, a reproducible fixed-96/adaptive comparison and an
unconfined ξ metadynamics diagnostic are provided in
[`benchmarks/h3/enhanced_sampling.py`](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/h3/enhanced_sampling.py).
See the [measured H₃ sampling audit](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/enhanced-sampling-2026-10-31/README.md).

```bash
pyrpmd validate examples/fes/adaptive_harmonic.inp
pyrpmd run examples/fes/adaptive_harmonic.inp --output runs/adaptive
pyrpmd resume runs/adaptive
pyrpmd analyze runs/adaptive --recompute --bins 30 --bootstrap 20
```

## Standard and well-tempered metadynamics

Use `task="fes"`, `[fes] method="metadynamics"`, ordinary CV specifications and an explicit hill
height/width. One, two or three CV dimensions are supported, including custom Python
CVs with declared positive periodicities and full ring-polymer Jacobians.
Fixed-cell periodic systems and fixed substrate atoms are supported. A cell-aware
PES is required, and all dynamics coordinates stay continuously unwrapped.
Optional one- and two-dimensional bias grids are described in [grid metadynamics](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/grid-metadynamics.md).

```toml
[fes.metadynamics]
cvs = ["x", "y"]
mode = "well_tempered"  # or "standard"
sigma = ["0.2 bohr", "0.2 bohr"]
height = "0.02 kcal/mol"
pace = 50               # completed production steps per hill
bias_factor = 8.0       # gamma > 1, omit for standard metadynamics
```

Numeric sigma values use the corresponding CV input units; `sigma_au` uses
canonical units. Height is a physical energy in hartree or an explicit energy
quantity, and `height_au` is also accepted. Supplying both forms is rejected.
The CV can be a function of the centroid, the average of a bead-wise function,
or a custom full-ring function. These definitions are distinct; the declared
value and full Jacobian are used without substituting one for another.

For a CV vector `s`, the physical bias is a sum of Gaussian hills. Periodic
dimensions use a smooth **sum over periodic Gaussian images**, normalized so
the value at a hill center equals its deposited height. An equivalent Fourier
sum is used for broad periodic kernels. This avoids a force discontinuity at
half a period. The ordinary nonperiodic Gaussian convention is
`exp(-delta**2 / (2*sigma**2))`.

The Python `MetadynamicsBias.hills` property returns a detached, serializable
history snapshot. Modify the running bias through `deposit()` or
`truncate_history()`, or construct a new bias with `hills=modified_history`
for reanalysis. Editing a snapshot does not change the running forces. This
keeps the cached Gaussian arrays consistent with the deposited history.

PyRPMD uses the sum-beads Hamiltonian sampled at `beta/P`. Therefore:

$$
H_P^{\rm biased}=H_P+P V(s,t),\qquad
F_{jac}^{\rm bias}=-P\sum_d\frac{\partial V}{\partial s_d}
                       \frac{\partial s_d}{\partial q_{jac}}.
$$

Standard metadynamics deposits a constant physical height `w0`.
Well-tempered metadynamics evaluates the **pre-deposition physical bias** at
the current center and deposits

$$
w(t)=w_0\exp\left[-\frac{\beta V(s(t),t^-)}{\gamma-1}\right],
\qquad \gamma=\frac{T+\Delta T}{T}>1.
$$

There is no extra factor of P in this exponent. The full-ring force and
Hamiltonian P factors are independently checked against finite differences.

### Sampling and event semantics

The ordinary `[sampling]` equilibration, production and save intervals apply.
Select `thermostat="andersen"`, `"nose-hoover-chain"` or `"pile-l"`.
PILE-L uses BCOCB with Cayley or BAOAB with exact normal modes. The NHC wraps
the selected deterministic map and supports velocity-Verlet as well. Initial
equilibration uses zero hills. The first hill is
deposited at production step `pace`, then at `2*pace`, and so on. Within each
integration step, both force evaluations see exactly the same bias history.
Only after the complete step succeeds is its hill committed. Saved sample
energies and physical bias values are evaluated before a coincident deposition.

History-dependent biases use the Python propagation loop, with a compiled
free-ring substep when the selected integrator supports it. `loop_backend="native"`
is rejected before simulation. By default, the hill sum is evaluated directly
and force cost grows with the number of hills. Optional
`metadynamics.grid.ranges` and `metadynamics.grid.bins` select a cubic/bicubic
Hermite bias whose energy/force lookup costs a fixed number of interpolation
terms. The force differentiates the interpolated energy, including across
periodic seams. The default spacing is at most `sigma/5`; grid refinement is
a separate numerical convergence check. Nonperiodic values outside the domain
raise an error and roll back the step. See [grid settings and cost](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/grid-metadynamics.md).

`sampling.replicas` creates independent metadynamics runs that can use ordinary
task parallelism. They do not share hills. A checkpoint atomically saves q/p,
RNG, NHC chain positions/momenta, absolute and production clocks, accumulated hills, observations and
propagation counts, with the common checkpoint checksum envelope. Failed steps
restore q/p, RNG and the entire thermostat before saving the valid prefix.

```toml
[sampling]
thermostat = "nose-hoover-chain"
thermostat_params = {tdamp="20 fs", chain_length=3, substeps=1, reset_after_equilibration=false}
```

The global chain samples `H_P` at `kT_P=P/beta`, with `3*P*N_mobile` degrees
of freedom and no COM subtraction. Frozen momenta are zero, and frozen atoms
contribute neither kinetic energy nor thermostat mass. `tdamp` is a time,
`chain_length` and `substeps` are positive integers. The default chain has three
members and uses a symmetric Suzuki–Yoshida composition. Optional
`reset_after_equilibration=true` zeros the chain at the start of production,
including when resuming exactly at that boundary. The physical state is retained.
This thermostat is for canonical sampling; real-time child trajectories are NVE.

Length-CV walls may constrain sampled domains, using the PLUMED quadratic
convention `W=kappa*delta^2` (no factor 1/2). They are always included in the
physical Hamiltonian as `P*W`. Declare wall CVs alongside the biased CVs; select
only the biased names in `metadynamics.cvs`.

```toml
[metadynamics]
cvs = ["r1", "r2", "height"]
sigma = ["0.2 angstrom", "0.2 angstrom", "0.2 angstrom"]
height = "0.020 eV"
pace = 400
bias_factor = 39
walls_during_equilibration = false
walls = [{cv="height", lower="0.5 angstrom", upper="7.2 angstrom", kappa="150 eV/angstrom^2"}]
```

Bare wall bounds and kappa are atomic units, independent of a CV's input unit.
Walls are active in equilibration by default; `false` activates them at the
first production step. Hill tempering uses only the MetaD bias. The reconstructed
FES describes the wall-defined ensemble; walls on additional CVs cannot in
general be removed by subtracting a function of the biased CVs.

Each `replicas/0000/HILLS.csv` is a human-readable view regenerated from its
authoritative `checkpoints/metadynamics_0000.npz`. A stale or missing CSV cannot
cause a duplicated hill during resume. HILLS files from different replicas
must not be concatenated to pretend they came from one shared-bias trajectory.

Numeric hill storage is reused between depositions; force evaluation still
performs the full Gaussian sum without a grid approximation or cutoff.
`bias.hills` now returns an owned serializable snapshot. Use `deposit()` to
append and `truncate_hills(count)` to roll back; editing a returned dictionary
or list does not change the live bias. Checkpoint JSON/HILLS formats are unchanged.

### Shared-bias multiple walkers

Opt in with `multiple_walkers=true` in `[fes.metadynamics]`. The walker count
is `sampling.replicas` (or `fes.sampling.replicas`). `block_steps` is the
positive number of integration steps between synchronizations and defaults
to `pace`. It is a scientific bias-sharing parameter and is included in the
fingerprint, unlike the dynamics backend's performance-only block size.

```toml
[fes.metadynamics]
mode = "well_tempered"
sigma = ["0.25 bohr", "0.25 bohr"]
height = "0.02 kcal/mol"
pace = 20
bias_factor = 8.0
multiple_walkers = true
block_steps = 100
```

At each absolute block boundary, every walker has completed the same number
of steps. During a block each walker sees the previous global history plus
its own new hills. The coordinator merges new hills in `(step, walker)`
order and distributes the resulting common bias before the next block.
Well-tempered heights retain the local pre-deposition bias used when they
were deposited; merging never recalculates them. The last block may be
shorter. Integration, sample, hill and synchronization clocks need not be
multiples of one another.

`checkpoints/metadynamics_walkers.npz` atomically stores one global hill
history and every walker's state, RNG, thermostat, samples and clocks.
`HILLS.csv` in the run root is its regenerable view and includes a `walker`
column. Shared checkpoint commits occur at synchronization boundaries;
`output.checkpoint_steps` does not create inconsistent per-walker histories.
If any walker fails, the next resume restarts the complete ensemble from
the last committed block. Serial and process scheduling use the same stream
identities. The full example is
[`multiple_walkers.inp`](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/metadynamics/multiple_walkers.inp).

The final bias is reconstructed once, using pooled walker coverage. Summary
metadata records `independent_histories=1`; `uncertainty` is unavailable for
this one shared ensemble. Run independent ensembles to assess spread. This
mode is separate from combining independently generated HILLS files, which
does not reproduce shared-bias sampling.

### Free-energy estimate and limits

`analysis.method="bias_estimate"` exports `fes.npz` and `fes.csv`:

$$
\begin{aligned}
F_{\rm standard}(s)&\approx -V(s,t)+C,\\
F_{\rm WT}(s)&\approx -\frac{\gamma}{\gamma-1}V(s,t)+C.
\end{aligned}
$$

These are finite-time, finite-width bias estimates. Standard metadynamics
continues to deposit constant hills and can oscillate about the free energy.
Neither formula proves convergence. All result summaries keep `converged=false`.
Unsampled bins are masked. Independent-replica surfaces are aligned using one
commonly sampled reference bin; the aggregate is restricted to common sampled
support. If no common reference exists, the aggregate is entirely masked.
`replica_spread` is descriptive, not a calibrated error bar.

`sampling_diagnostics` reports supported grid fractions for each replica and
their intersection, the fraction of samples inside the output grid, visited
CV extrema, total deposited height and last hill height. Small heights or many
hills do not establish convergence. `analysis.ranges` only selects histogram
bins: it **does not apply confining walls**. Auto ranges describe the visited
region and cannot diagnose coverage of an unspecified target domain.

In particular, the rational RPMDrate ξ coordinate can have a pole outside the
reaction corridor. Unconfined H + H₂ metadynamics can escape the corridor or
dissociate rather than explore the desired barrier. Simply extending time or
increasing hill height is not a remedy for an unsuitable CV/domain. Use an
appropriate nonsingular CV and a physically defined confined model for such
exploration. Length-CV walls are available as described above; they change the
sampled ensemble and do not remove a pole in a different CV. The current
metadynamics estimator uses `cv_density`, so its raw bias estimate cannot replace a `rate_flux`
umbrella PMF or supply a rate from biased trajectory times.

`analysis.bootstrap` must be zero; fixed-umbrella MBAR and ordinary equilibrium
bootstrap are not applied to these evolving-bias samples. Time-dependent
trajectory reweighting, well-tempered time-averaged estimators, automatic
convergence certification and infrequent-metadynamics kinetics are not
implemented. A periodic analysis range must span exactly one period.
The default limit is ten million output grid bins; `analysis.max_grid_points`
sets an explicit positive limit. Memory grows with the grid and replica count;
the direct hill sum is chunked but is not a grid-accelerated dynamics engine.

Use `pyrpmd project RUN --axes 2 --output NEW_DIRECTORY` to integrate a 3D
CV-density FES to its third coordinate. `--selection channel.npy` accepts a
Boolean domain mask of the source FES shape. Projection uses log-sum-exp with
bin-width volume factors, accepts nonuniform grids, and withholds any slice
with missing support. It does not add a radial Jacobian to a density PMF.
See the [Pt(111) example](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/surface/pt111/README.md) for the distinction
between conditional Boltzmann marginalization and the paper's hill-filtering
procedure. Projected bias estimates are not automatically substituted for a
statistically validated rate PMF.

```bash
pyrpmd run examples/metadynamics/standard.inp --output runs/meta1
pyrpmd run examples/metadynamics/well_tempered_2d.inp --output runs/meta2
pyrpmd resume runs/meta2
pyrpmd analyze runs/meta2 --recompute --bins 30 30
```

Python users can call `run_metadynamics`, `reanalyze_metadynamics`,
`MetadynamicsBias`, `GridMetadynamicsBias`, and `analysis.metadynamics.bias_fes` directly. The normal
`pyrpmd.run` / `JobSpec` entry points remain available.

## Choosing a PMF source for `rate`

Omitting `rate.pmf_source` (or setting it to `"umbrella"`) keeps the existing
umbrella-sampling rate workflow. To reuse a saved surface:

```toml
[rate.pmf_source]
directory = "runs/umbrella-rate-flux"
```

The default accepts only an umbrella `FESResult` with `measure="rate_flux"`,
one dimensionless centroid RPMDrate coordinate, and hartree energies. The
source must include its run manifest, resolved input and system description.
Temperature, beads, PES, system and reaction-coordinate identity must match
the destination calculation. Source files are hashed into the destination
manifest and checked again on resume. Parent and unbiased child dynamics
still run normally; the saved PMF replaces only biased PMF sampling.

Selecting MetaD/OPES requires both explicit
`allow_non_umbrella=true` and source metadata `reweighted=true`, in addition
to the same `rate_flux` and provenance checks. A raw `-V` or WT-scaled bias
surface is rejected with a message explaining that the PMF must be
reweighted. This selector does not implement time-dependent MetaD
reweighting; the current MetaD workflow does not produce a ready-to-use gas
rate PMF. OPES's frozen-bias CV-density reconstruction likewise does not
automatically acquire the rate-flux measure.

A stored pointwise uncertainty grid alone does not identify the covariance
between reference and barrier bins. External-source rate runs therefore
report unavailable PMF-barrier and total-rate uncertainty instead of
inventing an error from two independent endpoint errors.

## Refactoring regression evidence

[`unification_regression.py`](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/fes_methods/unification_regression.py)
runs the existing 2D umbrella, adaptive, replica-exchange, standard/WT/grid
MetaD and OPES examples in a baseline checkout and the current checkout.
It compares numerical FES arrays, trajectories, RNG state and bias histories
exactly, ignoring only migrated fingerprints, integrity envelopes and
producer metadata. It also creates real interrupted checkpoints with the
old code and continues them with the new code. Focused tests additionally
cover 3D analytic Gaussian cell integrals, cross-method validation, common
result readers, rate-source safeguards and shared-walker restart behavior.
These are implementation regressions, not claims of physical convergence.

## Method references

- J. Kästner and W. Thiel, *Analysis of the statistical error in umbrella sampling
  simulations by umbrella integration*, JCP **124**, 234106 (2006),
  [doi:10.1063/1.2206775](https://doi.org/10.1063/1.2206775).
- S. Park and W. Im, *Theory of Adaptive Optimization for Umbrella Sampling*,
  JCTC **10**, 2719–2728 (2014),
  [doi:10.1021/ct500504g](https://doi.org/10.1021/ct500504g). Its exchange-acceptance
  criteria are specific to replica exchange and are not the overlap score used here.
- A. Laio and M. Parrinello, *Escaping free-energy minima*, PNAS **99**, 12562–12566
  (2002), [doi:10.1073/pnas.202427399](https://doi.org/10.1073/pnas.202427399).
- A. Barducci, G. Bussi and M. Parrinello, *Well-Tempered Metadynamics: A Smoothly
  Converging and Tunable Free-Energy Method*, PRL **100**, 020603 (2008),
  [doi:10.1103/PhysRevLett.100.020603](https://doi.org/10.1103/PhysRevLett.100.020603).
- M. R. Shirts and J. D. Chodera, *Statistically optimal analysis of samples
  from multiple equilibrium states*, JCP **129**, 124105 (2008),
  [doi:10.1063/1.2978177](https://doi.org/10.1063/1.2978177).
- [PLUMED's official metadynamics tutorial](https://www.plumed.org/doc-v2.9/user-doc/html/master-_i_s_d_d-2.html)
  gives the physical hill-height, bias-factor and asymptotic FES conventions
  used for an independent formula cross-check. Code here is an independent
  implementation; no PLUMED code has been copied.
