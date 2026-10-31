# Adaptive umbrellas and metadynamics

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

Add `[adaptive_umbrella]` to a one- or two-dimensional `task="fes"` job, or to
a `task="rate"` PMF job:

```toml
[adaptive_umbrella]
enabled = true
pilot_steps = 1000
pilot_equilibration_steps = 200
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
overlap graph with an edge when `sqrt(O_ij * O_ji) >= target_overlap`. Replicas
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
have better mixing. The adaptive summary reports production ESS per window,
`production_effective_samples_target_met`, final pilot length, and production
length separately, and warns about insufficient or shorter production.
Increase production in a fresh run and compare independent replicas and PMFs;
more windows alone do not increase each trajectory's independent sample count.

The rate workflow retains its `rate_flux` metric bias during both stages;
the downstream PMF and recrossing preparation use the actual frozen window
centers from the sampled streams. Generic FES retains `cv_density`.

For real H + H₂ settings, a reproducible fixed-96/adaptive comparison and an
unconfined ξ metadynamics diagnostic are provided in
[`benchmarks/h3/enhanced_sampling.py`](../benchmarks/h3/enhanced_sampling.py).
See the [measured H₃ sampling audit](validation/enhanced-sampling-2026-10-31/README.md).

```bash
pyrpmd validate examples/fes/adaptive_harmonic.inp
pyrpmd run examples/fes/adaptive_harmonic.inp --output runs/adaptive
pyrpmd resume runs/adaptive
pyrpmd analyze runs/adaptive --recompute --bins 30 --bootstrap 20
```

## Standard and well-tempered metadynamics

Use `task="metadynamics"`, ordinary CV specifications and an explicit hill
height/width. One, two or three CV dimensions are supported, including custom Python
CVs with declared positive periodicities and full ring-polymer Jacobians.
Fixed-cell periodic systems and fixed substrate atoms are supported. A cell-aware
PES is required, and all dynamics coordinates stay continuously unwrapped.
Optional one- and two-dimensional bias grids are described in [grid metadynamics](grid-metadynamics.md).

```toml
[metadynamics]
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

\[
H_P^{\rm biased}=H_P+P V(s,t),\qquad
F_{jac}^{\rm bias}=-P\sum_d\frac{\partial V}{\partial s_d}
                       \frac{\partial s_d}{\partial q_{jac}}.
\]

Standard metadynamics deposits a constant physical height `w0`.
Well-tempered metadynamics evaluates the **pre-deposition physical bias** at
the current center and deposits

\[
w(t)=w_0\exp\left[-\frac{\beta V(s(t),t^-)}{\gamma-1}\right],
\qquad \gamma=\frac{T+\Delta T}{T}>1.
\]

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
raise an error and roll back the step. See [grid settings and cost](grid-metadynamics.md).

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

### Free-energy estimate and limits

`analysis.method="bias_estimate"` exports `fes.npz` and `fes.csv`:

\[
\begin{aligned}
F_{\rm standard}(s)&\approx -V(s,t)+C,\\
F_{\rm WT}(s)&\approx -\frac{\gamma}{\gamma-1}V(s,t)+C.
\end{aligned}
\]

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
exploration. This implementation has no metadynamics wall option. The current
metadynamics estimator uses `cv_density`, so it cannot replace a `rate_flux`
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
See the [Pt(111) example](../examples/surface/pt111/README.md) for the distinction
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

## Method references

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
