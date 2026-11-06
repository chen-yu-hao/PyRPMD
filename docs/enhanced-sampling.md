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
| `metadynamics` | 1–3 | Standard or well-tempered bias estimate; optional frozen-bias reweighted histogram | `replica_spread` |
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

These workflows sample ring-polymer configuration distributions. Their ordinary
saved trajectories do not establish a reaction rate. A separate
[biased-kinetics API](tier2-sampling.md#kinetics-from-audited-biased-trajectories)
can rescale an audited infrequent-MetaD or OPES-flooding history with an explicit
transition-region bias reference. It does not convert these standard FES runs
to kinetics automatically. The gas-phase rate workflow retains its prescribed
PMF measure and unbiased recrossing dynamics. Its `[rate.pmf]` selector can use
umbrella sampling, frozen-bias MetaD/OPES reweighting or a compatible external
surface; see [choosing a rate PMF](rate-pmf-selection.md) for the separate PMF
and recrossing settings.

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

By default, `sampling.replicas` creates independent metadynamics runs that can
use ordinary task parallelism. They do not share hills. Select
`metadynamics.shared_walkers=true` for synchronized deposition or
`metadynamics.multiple_walkers=true` for block synchronization, as described
below; choose one sharing protocol for a run. An independent-run checkpoint atomically saves q/p,
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

Set `output.store_configurations = true` to retain the full ring positions
at every saved production sample. Checkpoints and returned streams then include
`sample_configurations[sample, bead, atom, xyz]` in bohr, aligned with
`sample_steps` and `samples`. This enables structural diagnostics and selection
of configurations for subsequent, independently equilibrated umbrella runs.
The default is false; storage grows as samples × beads × atoms × 3 float64
values. The option must remain unchanged when resuming, since missing history
cannot be recovered from the final state. These configurations are biased
samples and are not fixed-umbrella MBAR data.

Each `replicas/0000/HILLS.csv` is a human-readable view regenerated from its
authoritative `checkpoints/metadynamics_0000.npz`. A stale or missing CSV cannot
cause a duplicated hill during resume. HILLS files from different replicas
must not be concatenated to pretend they came from one shared-bias trajectory.

Numeric hill storage is reused between depositions; force evaluation still
performs the full Gaussian sum without a grid approximation or cutoff.
`bias.hills` now returns an owned serializable snapshot. Use `deposit()` to
append and `truncate_hills(count)` to roll back; editing a returned dictionary
or list does not change the live bias. Checkpoint JSON/HILLS formats are unchanged.

### Shared multiple walkers

This protocol synchronizes every deposition. Set `shared_walkers=true` in
`[fes.metadynamics]` (or the compatible top-level `[metadynamics]` table) and set
`sampling.replicas` to the number of walkers (at least two). This mode uses a
serial coordinator, direct Gaussian hills, and either PILE-L or Andersen
sampling. It supports standard and well-tempered metadynamics; bias grids and
Nose–Hoover chains are not supported in the shared workflow.

Every walker advances against the same fixed history. At each common deposition
step, all well-tempered heights are calculated from that history before any new
hill is added. Each walker contributes the configured height, so total
deposition per event grows with walker count. There is no implicit division of
the height by the number of walkers.

One atomic ensemble checkpoint retains every walker's q/p, RNG and clock,
the complete ordered hill batches, and saved observations. A failure in a later
walker rolls back the entire step. `pyrpmd resume` continues that ensemble, and
`pyrpmd analyze RUN --recompute` can reanalyze its saved shared history.
The output reports one correlated bias history and pools occupancy across the
walkers. Their spread is not an independent-replica uncertainty estimate;
uncertainty remains unavailable until an appropriate independent-run or joint
resampling analysis is supplied. The Python building block is
`pyrpmd.shared_metadynamics.SharedMetadynamicsBias`.

### Shared-bias multiple walkers

This protocol synchronizes at block boundaries and is distinct from the
synchronized-deposition protocol above. Opt in with `multiple_walkers=true`
in `[fes.metadynamics]`. The walker count
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
sampled ensemble and do not remove a pole in a different CV. The default
metadynamics bias estimator uses `cv_density`, so its raw bias estimate cannot
replace a `rate_flux` PMF or supply a rate from biased trajectory times. The
separate frozen-bias rate route explicitly samples the flux metric and
reweights stationary production; it requires sufficient support through the
reactant reference and dividing surface and does not allow static walls.

`analysis.bootstrap` must be zero; fixed-umbrella MBAR and ordinary equilibrium
bootstrap are not applied to these evolving-bias samples. Time-dependent
trajectory reweighting, well-tempered time-averaged estimators and automatic
convergence certification are not implemented. Infrequent-metadynamics
kinetics requires the separate audited-history API linked above; these sparse
FES samples do not supply its interval-resolved acceleration clock. A periodic
analysis range must span exactly one period.
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

New inputs select `umbrella`, `metadynamics`, `opes` or `external` with
`[rate.pmf] method`. The [rate PMF guide](rate-pmf-selection.md) provides complete
examples, inheritance rules and the training/freeze/production clock. Built-in
MetaD/OPES rate PMFs require `estimator="reweighted_histogram"`, explicit
`freeze_after_steps` and `freeze_equilibration_steps`, and `analysis.bootstrap=0`.
They include the rate-flux metric in propagation and use only stationary
fixed-bias samples with weights proportional to `exp(beta*V_frozen)`.
Correlated barrier and total-rate uncertainty remain unavailable.

To reuse a saved surface:

```toml
[rate.pmf]
method = "external"
directory = "runs/umbrella-rate-flux"
```

Legacy inputs that omit `[rate.pmf]` retain the existing umbrella-sampling
workflow and `[rate.pmf_source]` syntax. Do not combine the new selector with
`rate.pmf_method` or `rate.pmf_source`.

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
reweighted. This selector does not implement time-dependent MetaD reweighting.
An ordinary `cv_density` reconstruction retains its original measure; reuse
requires a surface sampled with `measure="rate_flux"` and reconstructed from
frozen-bias production. Selecting the built-in MetaD/OPES rate method performs
that protocol within the rate calculation.

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

## Centroid ABF and extended-system ABF

The additional `fes.method="abf"` and `"eabf"` selectors support one or two
nonperiodic centroid CVs. They produce the same `FESResult` NPZ/CSV/JSON layout
described above, with `measure="cv_density"` and `uncertainty_kind="replica_spread"`.
The evaluated CV Jacobian must have full row rank; repeated or functionally
dependent CVs do not define a regular density in the declared dimension.
The current implementation also rejects isolated rank-deficient points rather
than attempting a singular-coordinate estimator. Row normalization keeps this
geometrical check independent of the chosen CV units.
They require an unconstrained system with all atoms mobile, a canonical
sampling thermostat, and `loop_backend="auto"` or `"python"`. Fixed atoms,
periodic cells/CVs, pressure coupling, replica exchange and the rate-flux
measure are rejected. The existing `FluxMetricBias` is not involved.

### Local mean force and the full-ring Jacobian

Let `J[d,p,n,c]` be the public CV Jacobian and `G = sum_p J` the derivative
with respect to the centroid. Define the mass-weighted dual field

```{math}
v=(GM^{-1}G^{T})^{-1}GM^{-1}.
```

The local free-energy gradient is

```{math}
g_i(q)=-v_i\cdot\frac{1}{P}\sum_p F_p(q)
       -\frac{1}{\beta}\operatorname{div}_{\bar q}v_i.
```

`F_p` is the unbiased physical PES force, with no bead scaling. Ring spring
forces cancel under a common centroid displacement. The divergence supplies
the geometrical Jacobian/Fixman term for the probability-density measure:
for a radial CV in three dimensions it contributes `-2/(beta*r)`. Omitting it
would reconstruct a different free energy. Singular or ill-conditioned CV
metrics fail explicitly.

`derivative="analytic"` supports unmodified Cartesian components and a single
`Distance` or `COMDistance`. Other centroid CVs, including learned coordinates,
must explicitly request `derivative="finite_difference"`. That route takes
`2*3N` additional centroid-Jacobian evaluations per force observation, uses
`fd_step` (default `1e-5`) as a relative Cartesian displacement, and stores no
dense Hessian. Study this step size separately. eABF avoids these second
derivatives altogether.

The applied bias is **projected ABF**: online Welford statistics estimate the
mean force on the declared grid. Every `pace` production steps, a trapezoidal
edge integration in 1D or sparse least-squares Helmholtz projection in 2D
constructs a scalar free-energy estimate. The negative of its multilinear
interpolant is the bias potential; its analytic derivative supplies forces.
Consequently the energy and force remain consistent, including in 2D with
noisy, nonconservative raw mean forces. The local field ramps linearly with
`min(count/full_samples, 1)`. Evaluating a force never changes the history.
Grid resolution, ramp, update interval, derivative scheme and walls are all
scientific parameters. The method is not a native fused loop, although the
existing native free-ring substeps remain available.

```toml
[fes]
method = "abf"

[[fes.cvs]]
name = "x"
kind = "cartesian"
atom = 0
axis = "x"
estimator = "centroid"

[fes.abf]
cvs = ["x"]
ranges = [["-1.5 bohr", "1.5 bohr"]]
bins = [60]
full_samples = 200
pace = 100
wall_stiffness = ["1 hartree/bohr^2"]
derivative = "analytic"

[fes.sampling]
equilibration_steps = 1000
production_steps = 100000
sample_interval = 10
replicas = 4
thermostat = "pile-l"
gamma_centroid = "0.01 fs^-1"

[fes.analysis]
method = "mean_force"
min_count = 50
bootstrap = 0
```

Put `bins` and `ranges` in the method table: changing them changes sampling,
so they are not reanalysis overrides. Coordinate arrays also accept `_au`
spellings (`ranges_au`, `wall_stiffness_au`). Numerical values in ordinary
spellings follow the selected CV's declared input units; unit strings pass
through the common unit parser. Outside the grid the adaptive potential is
held fixed in the outward direction and the wall adds
`0.5*wall_stiffness*(value-boundary)^2`. The unbiased force estimator never
includes the adaptive or wall force.

### eABF and CZAR reconstruction

For `method="eabf"`, replace the ABF table with, for example:

```toml
[fes.eabf]
cvs = ["x"]
ranges = [["-1.5 bohr", "1.5 bohr"]]
bins = [60]
full_samples = 200
pace = 100
wall_stiffness = ["1 hartree/bohr^2"]
coupling = ["2 hartree/bohr^2"]
extended_mass = ["10 me"]
friction = "0.01 fs^-1"

[fes.analysis]
method = "czar"
min_count = 50
bootstrap = 0
```

The auxiliary variable `lambda` has its own momentum, mass and Langevin bath
at physical inverse temperature `beta`. A symmetric auxiliary OU/kick/drift
split surrounds the physical ring-polymer sampling step. Physical coordinates
feel only the harmonic coupling `W = coupling*(xi-lambda)^2/2`; their ring
Hamiltonian contains `P*W`. The adaptive force acts on `lambda`. Auxiliary
masses for a length CV have ordinary mass units; for other CV dimensions the
required units are `energy*time^2/CV^2`, with explicit `_au` inputs also accepted.

The auxiliary free energy is Gaussian-smoothed by the coupling. The reported
physical-CV surface instead uses the corrected z-averaged restraint estimator
(CZAR):

```{math}
\nabla A(z)=-\beta^{-1}\nabla\log\widetilde\rho(z)
             +\left\langle K(\lambda-z)\mid\xi=z\right\rangle.
```

This includes the physical CV's Jacobian through its sampled density without
evaluating a CV Hessian. Boundary bins and cells missing a complete centered
derivative stencil are masked; no pseudocounts bridge empty cells. Finite-bin
derivative errors, insufficient mixing and finite timestep still require
convergence studies. The auxiliary and physical histograms and conditional
restraint statistics are retained in the checkpoint.

### Analysis, uncertainty and replay

Every production step contributes to the force/histogram statistics;
`sample_interval` controls only the exported CV trace. Equilibration does not
train the adaptive bias. Checkpoints atomically store the MD coordinates,
momenta and clock, all thermostat/RNG state, online count/mean/M2 statistics,
the current projected bias, and eABF auxiliary state. A failed substep rolls
back these coupled states together. The usual `run`, `resume`, `prepare` and
`analyze --recompute` routes are supported:

```bash
pyrpmd run examples/abf_double_well.inp --output runs/abf
pyrpmd resume runs/abf
pyrpmd analyze runs/abf --recompute
```

Each replica is reconstructed independently. Only common support is averaged,
with a shared reference bin; disconnected islands are masked because their
relative offsets are unknown. Two or more independent histories provide a
descriptive replica standard deviation. A single history has NaN uncertainty;
per-bin force variance is not relabeled as a correlation-corrected FES error.
Both `converged` and `physical_converged` remain false. The tests compare
radial Jacobians, nonlinear vector CVs, double-well and Müller–Brown numerical
quadrature, CZAR Gaussian/entropy references, existing umbrella/MBAR estimates,
and exact interrupted RNG replay. These checks establish implementation
consistency, not convergence for a molecular production calculation.

## Steered MD and nonequilibrium work

`fes.method="steered"` uses a moving harmonic centroid restraint and writes
`steered.npz` plus JSON summaries. Its target is the **restrained endpoint
free-energy difference**, so it does not create a `FESResult` grid. The
previous common-grid output description applies to the surface estimators.
Unbiased-PMF/Hummer–Szabo reconstruction is not implemented.

```toml
[fes]
method = "steered"

[fes.steered]
cvs = ["x"]
start = ["-1 bohr"]
end = ["1 bohr"]
stiffness = ["2 hartree/bohr^2"]
direction = "both"
thermostat = "langevin"
gamma = "0.01 fs^-1"
initial_equilibrium = false

[fes.sampling]
equilibration_steps = 1000
production_steps = 5000
sample_interval = 10
replicas = 100

[fes.analysis]
method = "crooks"
measure = "restrained_endpoints"
bootstrap = 200
```

Supply the same CV/system/PES sections as for ordinary FES. An explicit
`centers`/`centers_au` schedule with `production_steps+1` entries can replace
`start`/`end`. The reverse ensemble follows the exact reversed schedule.
`replicas` is the number of independent pulls **per direction**. The steering
method owns its symmetric Langevin splitting; generic Andersen, PILE and
thermostat-parameter controls are rejected instead of silently ignored.

Work is done **on** the system. Energies and work use `H_P/P`, so Jarzynski uses
the physical `beta`, including when `P>1`. The output separates protocol work,
finite-step shadow work, and bath heat. Estimation uses
`total_work = protocol_work + shadow_work`; heat is excluded. This explicit
shadow correction removes deterministic integration-map bias from the
fluctuation relation, at the cost of two extra energy/PES evaluations per
pulling step. Fixed-center equilibration uses Metropolis-corrected reversible
dynamics; a finite equilibration duration still does not prove equilibrium.

```python
from pyrpmd.steered import jarzynski, crooks

forward = jarzynski(forward_total_work, beta, bootstrap=200, seed=42)
bidirectional = crooks(forward_total_work, reverse_total_work, beta,
                       bootstrap=200, seed=42)
```

The reverse work array retains its native B-to-A sign. `crooks` solves the
Crooks/Bennett equation, including unequal ensemble sizes. Diagnostics report
work variance, dissipated work, exponential-weight ESS and concentration,
second-cumulant and leading-order bias estimates, bootstrap uncertainty, and
bidirectional overlap. A bootstrap only describes observed independent
trajectories: it cannot discover unsampled low-work tails or initialization
bias. `initial_equilibrium` is a user declaration, and neither that declaration
nor a numerical estimator root sets `physical_converged=true`. Every pull has
an atomic, identity-checked checkpoint containing RNG, thermostat/MD state,
protocol position and the complete accumulated work/heat history.

## Thermal strings and learned coordinates

The existing `adaptive_string.py` optimizes a supplied Euclidean force oracle;
it is not a thermal sampler. The new
[finite-temperature string workflow](finite-temperature-string.md) adds
restrained centroid sampling, the averaged CV mass metric, block diagnostics,
fixed-endpoint path updates and exact iteration-boundary replay, while reusing
the existing arc-length reparameterization. At finite restraint stiffness it
targets a smoothed free-energy path; physical convergence remains unverified.
`PathCV` continues to describe progress/distance from supplied Cartesian
reference images; a CV-space string is not silently treated as Cartesian
reference structures.

The [mlcolvar route](mlcolvar.md) trains Deep-LDA or Deep-TICA, exports the full
preprocessing/network/projection graph as CPU float64 TorchScript, and checks
it with `TorchScriptCV`. The example and genuine optional end-to-end tests
include Cartesian finite-difference derivatives and ring-bead scaling. The
training dependency is optional; missing packages cause explicit pytest skips.
The [PLUMED guide](plumed.md) describes the separate audited file-RESTART
protocol for supported external bias histories. VES is not included here.

```{toctree}
:hidden:

finite-temperature-string
mlcolvar
```

### Additional method sources

- E. Darve, D. Rodríguez-Gómez and A. Pohorille, *Adaptive biasing force method
  for scalar and vector free energy calculations*, JCP **128**, 144120 (2008),
  [doi:10.1063/1.2829861](https://doi.org/10.1063/1.2829861).
- H. Alrachid and T. Lelièvre, *Long-time convergence of an adaptive
  biasing force method: Variance reduction by Helmholtz projection*,
  SMAI J. Comput. Math. **1**, 55–82 (2015),
  [doi:10.5802/smai-jcm.4](https://doi.org/10.5802/smai-jcm.4).
- A. Lesage, T. Lelièvre, G. Stoltz and J. Hénin, *Smoothed Biasing Forces Yield
  Unbiased Free Energies with the Extended-System Adaptive Biasing Force Method*,
  JPCB **121**, 3676–3685 (2017),
  [doi:10.1021/acs.jpcb.6b10055](https://doi.org/10.1021/acs.jpcb.6b10055).
- C. Jarzynski, *Nonequilibrium Equality for Free Energy Differences*,
  PRL **78**, 2690 (1997), [doi:10.1103/PhysRevLett.78.2690](https://doi.org/10.1103/PhysRevLett.78.2690).
- G. E. Crooks, *Entropy production fluctuation theorem and the nonequilibrium
  work relation for free energy differences*, PRE **60**, 2721 (1999),
  [doi:10.1103/PhysRevE.60.2721](https://doi.org/10.1103/PhysRevE.60.2721).
- C. H. Bennett, *Efficient estimation of free energy differences from Monte
  Carlo data*, J. Comput. Phys. **22**, 245–268 (1976),
  [doi:10.1016/0021-9991(76)90078-4](https://doi.org/10.1016/0021-9991(76)90078-4).
- D. A. Sivak, J. D. Chodera and G. E. Crooks, *Using Nonequilibrium Fluctuation
  Theorems to Understand and Correct Errors in Equilibrium and Nonequilibrium
  Simulations of Discrete Langevin Dynamics*, PRX **3**, 011007 (2013),
  [doi:10.1103/PhysRevX.3.011007](https://doi.org/10.1103/PhysRevX.3.011007).

## Variationally enhanced sampling (VES)

`task="fes"` with `[fes] method="ves"` optimizes a bias on one or two
nonperiodic **centroid** CV components. This is a native PyRPMD workflow; it
does not extend the external PLUMED bridge described above. It writes the
common `FESResult` files and supports the usual `run`, `resume`, and
`analyze --recompute` commands.

The current workflow requires a nonperiodic, unconstrained system with all
atoms mobile and independent selected CV gradients at the starting geometry.
Barostats, alternate path-integral methods, replica exchange,
and adaptive umbrella placement are not combined with VES. Use
`dynamics.loop_backend="auto"` or `"python"`; an explicitly forced native
loop is rejected because the adaptive VES updates run in Python.

```toml
[fes]
method = "ves"

[[fes.cvs]]
name = "x"
kind = "cartesian"
atom = 0
axis = "x"
estimator = "centroid"

[fes.ves]
cvs = ["x"]
ranges_au = [[-1.5, 1.5]]
orders = [4]
target = "uniform"
learning_rate_au = 0.05
pace = 50
quadrature_order = 32
wall_stiffness_au = [20.0]

[fes.sampling]
equilibration_steps = 500
production_steps = 10000
sample_interval = 10
replicas = 3
thermostat = "pile-l"
gamma_centroid = "1 autime^-1"

[fes.analysis]
method = "bias_estimate"
bins = [40]
bootstrap = 0
```

The complete illustrative inputs are
[`ves_double_well.inp`](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/ves_double_well.inp)
and
[`ves_well_tempered_2d.inp`](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/ves_well_tempered_2d.inp).
They use a toy model, electron-mass units, and an artificial temperature for
short runs; their durations are examples of the workflow, not a convergence
recommendation for molecular simulations.

### Basis, target, and optimizer

The bias is a tensor-product Legendre expansion on `ranges_au`, with maximum
degree `orders[d]` along each selected CV component. Mixed products are
included in two dimensions. The all-constant product is omitted because a
constant bias cannot change the sampled distribution. All `_au` fields use
atomic units: CV bounds use the CV's canonical units, `learning_rate_au` is in
hartree, and a wall stiffness is in hartree per squared CV unit.

For basis functions `f_i` and coefficients `alpha_i`, VES minimizes the
Valsson--Parrinello functional by matching biased-ensemble basis averages to
those of the target distribution:

```{math}
V(s;\alpha)=\sum_i\alpha_i f_i(s),\qquad
g_i=\langle f_i\rangle_p-\langle f_i\rangle_V,\qquad
H_{ij}=\beta\operatorname{Cov}_V(f_i,f_j).
```

The workflow uses averaged stochastic gradient descent with the full
covariance Hessian. It evolves instantaneous optimizer coefficients but applies
their running average to the dynamics. `pace` counts production MD steps per
coefficient update; `sample_interval` controls the saved CV trace, not the
optimizer's observation frequency. The target moments use tensor-product
Gauss--Legendre integration with `quadrature_order` points per axis. Both basis
order and quadrature order need a resolution study for a new system.

Defaults are `orders=4`, `pace=100`, `learning_rate_au=0.5/beta`,
`quadrature_order=32`, and `target_update_interval=100`. The quadrature order
must exceed the polynomial order on each axis. At most 255 nonconstant basis
functions are supported; output grids are evaluated in bounded chunks.

With `target="uniform"`, the target is constant within the specified box and
the variational estimate is `F = -V + constant`. For a well-tempered target,
replace the target setting and supply its controls:

```toml
# In [fes.ves]:
target = "well_tempered"
bias_factor = 4.0
target_update_interval = 10
```

`bias_factor` must be greater than one. `target_update_interval` counts
coefficient updates, not MD steps. The target starts uniform and is updated
from the current bias and the **previous** target:

```{math}
F_{m+1}(s)=-V_{m+1}(s)-\beta^{-1}\log p_m(s),\qquad
p_{m+1}(s)\propto\exp[-\beta F_{m+1}(s)/\gamma].
```

The relation `F = -gamma/(gamma-1) * V + constant` describes the self-consistent
limit. A finite optimization stage is not assumed to have reached that limit.
Target normalization is evaluated in the log domain.

### Finite domain and physical interpretation

The requested target and optimizer moments are conditional on being inside
the CV box. Observations outside it are excluded from the optimization and
counted explicitly in diagnostics; they are not clipped into edge bins or
silently added to the target moments. Harmonic walls outside each interval
provide confinement. Their stiffness remains part of the run identity.
The VES wall energy is `W = 0.5 * sum_d(k[d] * delta[d]**2)`, with `delta`
the signed distance outside each interval and zero inside it.

For force evaluation outside an interval, each one-dimensional basis function
uses the bounded tangent continuation
`L(b) + L'(b) * d / (1 + abs(d))`, where `b` is the nearest boundary and `d`
is the displacement in the mapped Legendre coordinate. This continuation
matches the value and first derivative at the boundary. It avoids uncontrolled
polynomial extrapolation; it does not turn an excursion into an in-domain
sample. The wall and continuation forces act on the centroid with the same
ring-polymer scaling as the other native FES biases.

This finite-box formulation estimates the conditional CV density in the
chosen region. A finite basis can only reproduce the shapes it represents.
Small optimization gradients therefore do not establish complete mixing,
adequate basis resolution, or a converged physical free energy.

### Frozen production, output, and continuation

`bias_estimate` is the default analysis. It reconstructs the finite-stage
variational surface from the saved coefficients and target. For a quantitative
check independent of that variational reconstruction, reserve a stationary
production tail and reweight its histogram:

```toml
# In [fes.ves]:
freeze_after_steps = 5000
freeze_equilibration_steps = 500

# In [fes.analysis]:
method = "reweighted_histogram"
bins = [16, 16]  # For the two-CV example.
bootstrap = 0
```

Freezing stops both coefficient and target updates. After the additional
frozen equilibration interval, reweighting uses only samples generated by that
fixed bias. Adaptive samples are not treated as samples from a stationary
ensemble. The frozen-tail estimator includes bin-volume normalization and
masks unsupported bins. Inspect coverage, weight concentration, independent
histories, and timestep/basis/domain dependence before interpreting an
agreement with a reference as physical convergence.

The usual common FES uncertainty fields are retained. Independent replicas
provide a descriptive spread of reconstructed surfaces; a single history does
not provide a calibrated FES uncertainty. Numerical optimizer diagnostics do
not set `physical_converged=true`.

Atomic checkpoints preserve coordinates, momenta, the thermostat and RNG state,
instantaneous and averaged coefficients, partial moment accumulators, the
optimization clock, the dynamic target, and frozen-tail statistics. Changing
scientific VES settings invalidates continuation; analysis settings can be
recomputed from the saved sampling state.

```bash
pyrpmd run examples/ves_double_well.inp --output runs/ves
pyrpmd resume runs/ves
pyrpmd analyze runs/ves --recompute
```

The reproducible [validation driver](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/ves_validation.py)
compares independently sampled one- and two-dimensional analytic models with
bin-integrated numerical quadrature and the existing umbrella/MBAR estimator.
Its optional `--md` stage separately checks frozen-bias MD production. Saved
[stochastic results](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/t24-ves/reference.json)
and a [comparison figure](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/t24-ves/stochastic_validation.png)
record the sampling budget, independent-history errors and model limitations.

### VES primary references

- O. Valsson and M. Parrinello, *Variational Approach to Enhanced Sampling and
  Free Energy Calculations*, PRL **113**, 090601 (2014),
  [doi:10.1103/PhysRevLett.113.090601](https://doi.org/10.1103/PhysRevLett.113.090601).
- O. Valsson and M. Parrinello, *Well-Tempered Variational Approach to Enhanced
  Sampling*, JCTC **11**, 1996--2002 (2015),
  [doi:10.1021/acs.jctc.5b00076](https://doi.org/10.1021/acs.jctc.5b00076).
- The PLUMED reference documentation describes the
  [linear expansion](https://www.plumed.org/doc-v2.10/user-doc/html/_v_e_s__l_i_n_e_a_r__e_x_p_a_n_s_i_o_n.html),
  [dynamic well-tempered target](https://www.plumed.org/doc-v2.10/user-doc/html/_t_d__w_e_l_l_t_e_m_p_e_r_e_d.html),
  and [averaged stochastic optimizer](https://www.plumed.org/doc-v2.10/user-doc/html/_o_p_t__a_v_e_r_a_g_e_d__s_g_d.html).

## Distributed multiple-walker MetaD and OPES

`walkers_mode` selects how a shared history is transported between walkers:

| value | execution | bias semantics | persistence |
| --- | --- | --- | --- |
| `serial` | existing single-process behavior | unchanged | unchanged historical checkpoint |
| `mpi-sync` | exactly one walker per MPI rank | one rank-ordered, synchronous batch | one atomic ensemble checkpoint |
| `async-files` | independent processes or MPI ranks | locally visible history, eventually merged | one atomic checkpoint and append-only log per walker |

The option is accepted in `[metadynamics]` and `[opes]`. Omitting it does not
insert a default into the resolved input and therefore does not change an
existing run identity. An explicit non-serial value is part of the scientific
fingerprint and cannot resume a serial checkpoint.

### Synchronous MPI

MetaD requires `shared_walkers = true`; `sampling.replicas` must equal the MPI
world size. At every deposition clock, each rank contributes one CV center with
`allgather`. All ranks evaluate the complete center array against the same
pre-commit bias, then commit in rank order. This is the existing serial
`SharedMetadynamicsBias.deposit_walkers` transaction, so a deterministic CPU
run has the same walker trajectories, hill records, observations, momenta and
RNG states as the serial coordinator. A failed rank is reported collectively;
the step, RNG and history are rolled back before an atomic checkpoint is
written.

```toml
[fes.sampling]
replicas = 4

[fes.metadynamics]
shared_walkers = true
walkers_mode = "mpi-sync"
mode = "well_tempered"
sigma = ["0.25 bohr", "0.25 bohr"]
height = "0.02 kcal/mol"
pace = 20
bias_factor = 8.0
```

Launch exactly four ranks:

```bash
mpiexec -n 4 python -m pyrpmd run input.inp --output runs/meta-mpi
```

The authoritative checkpoint is
`runs/meta-mpi/checkpoints/shared_metadynamics_mpi.npz`. The former serial
shared checkpoint remains `shared_metadynamics.npz` and its schema is not
changed.

OPES uses the same rank-count rule. A synchronous batch computes every kernel
weight from the common pre-update bias and commits kernels in rank order:

```toml
[fes.sampling]
replicas = 4

[fes.opes]
walkers_mode = "mpi-sync"
sigma = ["0.20 bohr", "0.20 bohr"]
barrier = "8 kcal/mol"
pace = 20
bias_factor = 10.0
```

Its checkpoint is `checkpoints/opes_shared_mpi.npz`.

### Asynchronous files

Each process owns one walker ID and one file under
`RUN/walkers/metadynamics/` or `RUN/walkers/opes/`. It may be launched by MPI,
or as independent scheduler jobs with explicit IDs:

```bash
PYRPMD_WALKER_ID=0 PYRPMD_WALKERS=4 python -m pyrpmd run input.inp --output runs/async &
PYRPMD_WALKER_ID=1 PYRPMD_WALKERS=4 python -m pyrpmd run input.inp --output runs/async &
PYRPMD_WALKER_ID=2 PYRPMD_WALKERS=4 python -m pyrpmd run input.inp --output runs/async &
PYRPMD_WALKER_ID=3 PYRPMD_WALKERS=4 python -m pyrpmd run input.inp --output runs/async &
wait
```

Use `walkers_mode = "async-files"` in the selected method table. The shared
directory has a checksummed manifest binding it to the method, walker count and
run fingerprint. `walker-NNNN.lock` rejects two live writers with the same ID.
Each writer appends checksum-protected JSON records only to
`walker-NNNN.jsonl`; readers sort the committed event set by
`(step, time, walker, sequence)` before rebuilding the numeric history.

A newline is the commit boundary. A complete malformed record or checksum
mismatch is corruption and stops the run. An unterminated final record is not
consumed and does not advance the saved byte offset; after taking the writer
lock, its owner truncates that crash tail before appending again. Every walker
checkpoint stores all reader offsets, duplicate guards, the next local
sequence, dynamics RNG and a possible `pending_record`. Deposition uses two
phases:

1. atomically checkpoint the post-step state and exact prepared record;
2. idempotently ensure that record is in the owner log, merge committed logs,
   clear `pending_record`, and atomically checkpoint again.

Consequently a crash before, during or after publication neither drops nor
duplicates a kernel. Other walkers do not wait for the failed process. The last
walker to finish writes the aggregate FES; an earlier finisher reports
`walker_complete_pending_merge`. Per-walker checkpoints are
`shared_metadynamics_async_NNNN.npz` and `opes_shared_async_NNNN.npz`.

Asynchronous MetaD preserves the height and `bias_before` calculated from the
writer's locally visible history. It never re-tempers a remote hill during
merge. Therefore it is eventually consistent and statistically comparable to
the synchronous algorithm, but it is not trajectory-wise or bitwise identical.

### Shared OPES kernel and normalization rules

Distributed OPES currently requires fixed bandwidth, KDE `cv_density`
analysis and no frozen-bias phase. Every kernel is retained: compression is
`none`, memory is linear in kernel count, and rebuilding the exact shared
normalization is quadratic in kernel count. This is explicit rather than a
silent approximation.

For kernel centers \(s_i\), log weights \(\ell_i\), and normalized kernel
\(K\), the shared estimate is

\[
P(s)=\frac{\sum_j e^{\ell_j}K(s,s_j)}{\sum_j e^{\ell_j}},
\qquad
Z=\frac{1}{N}\sum_i P(s_i).
\]

`log_z` is reconstructed from all uncompressed kernels in deterministic event
order. In `metad` mode, a synchronous batch uses
`log_weight = beta * V_before` with one common pre-commit history; asynchronous
writers use their locally visible `V_before`. In `explore` mode every shared
kernel has zero log weight. The checkpoint records the complete event set and
rejects a supplied `log_z` that is not exactly reproduced.

### Restrictions and interpretation

Distributed MetaD does not combine with the Hermite bias grid,
`multiple_walkers = true`, frozen-bias production, or non-legacy sampler
variants. Distributed OPES does not support adaptive widths, compression,
frozen-bias reweighting, or `rate_flux`. These combinations fail explicitly;
there is no serial or precision fallback.

The summaries report `converged = false` and `physical_converged = false`.
One shared history couples the walkers, so walker spread is not an independent
replica uncertainty. MPI equality tests establish deterministic implementation
parity, and the asynchronous toy campaign establishes only statistical
compatibility for its specified model and budget. Neither establishes CV
adequacy, equilibrium, bead convergence, timestep convergence, or a molecular
free-energy error bar.


### Serial MetaD history blocks

For a new serial standard or well-tempered MetaD run, the native path can keep
the frozen history in C++ between semantic events. A block ends at the next
hill (pace), the equilibrium or freeze boundary, a checkpoint, 4096 steps, or
the pre-drawn buffer limit. The observation mask includes both saved samples
and the hill endpoint, so sample_bias is evaluated from the same frozen bias
before the Python deposition transaction. Andersen uses the NP1 RefreshPlan;
PILE-L uses the NP2 sampling block when its native descriptor is available.
Walls are included through the NP4 detached descriptor.

The checkpoint carries native_path_contract = 1 for new runs. A checkpoint
created before this contract was introduced has no marker and continues through
the historical _history_step path, preserving its exact restart schedule.
Shared, asynchronous and custom-walker histories remain on their existing
paths. The block path records a valid prefix on a propagation failure and never
deposits a hill from a failed block.
