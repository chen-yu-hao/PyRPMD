# Input specification v1

**English** · [中文](zh/input.md)

`.inp` and `.toml` files use TOML syntax. A `.py` input executes the module's `build()`
function, which returns a `JobSpec` or dictionary. Both routes use the same parser to
produce a normalized configuration. Relative paths inside an input are resolved from
**the input file's directory**. Command-line `--pes`, `--pes-manifest` and `--output`
paths are relative to the launch directory. Plugin entry points accept both
`module:object` and `file.py:object`.

## Common fields

| Field | Type, units and meaning |
| --- | --- |
| `schema_version` | Currently the integer 1 |
| `task` | `rate`, `drpmd`, `fes`, `surface_rate`, `network`, `metadynamics`, `opes`, `pcet`, `pimd`, `diffusion`, `site_rate` |
| `preset` | `pilot` (default) or `production` |
| `temperature` | Positive temperature, for example `"300 K"`; bare numbers also mean K |
| `beads` | Positive integer; `auto` uses P=32 only in a pilot and marks it unconverged |
| `seed` | Nonnegative integer; task random streams are derived from stable task IDs |
| `dynamics.timestep` | For example `"0.1 fs"`; a bare number is in atomic time units |
| `dynamics.integrator` | `cayley-bcb` (default), `normal-mode-bab`, or `velocity-verlet`; see [integrators](integrators.md) |
| `dynamics.native` | `true` prefers compiled kernels and allows serial NumPy fallback; `false` uses NumPy; `"required"` requires compiled kernels |
| `output.directory` | Run directory; a new job does not overwrite a nonempty directory |
| `output.checkpoint_interval` / `checkpoint_steps` | Time interval or positive integer step count; the step count takes precedence |
| `output.trajectory_stride` | Positive integer interval in steps between saved dRPMD frames |
| `output.store_configurations` | Save complete ring coordinates at each MetaD production sample; Boolean, default `false`, immutable on restart |
| `output.trajectory_storage` / `chunk_frames` | Materials defaults: `"stream"` and 256 frames; `"memory"` uses the older format |

`production` requires explicit temperature, bead count, dynamics and sampling settings;
dRPMD also requires explicit `bmax`. This checks input completeness. It does not select
more accurate parameters: omitted subfields within dynamics/sampling still use their
defaults. Scientific convergence requires a series of calculations. Unknown top-level
and common section fields raise errors. See the [manual's defaults](manual/input.md).

`pcet` uses the separate model input below. It does not use a molecular system, PES,
beads, propagation, or sampling presets. Its `temperature` also accepts a list for scans.

Times must be integer multiples of the timestep. To request a duration that is not an
exact multiple, choose the actual integer step count explicitly, for example
`equilibration_steps=333`. Internal units for bare numbers are bohr, hartree, atomic
time and electron mass. XYZ coordinates default to angstrom. Force constants accept
compound units such as `"10 kcal/mol/angstrom^2"`. All atom indices are **zero-based**.

## System

`[system]` accepts `transition_state="ts.xyz"`, `geometry="system.xyz"`, or
`reactants=["A.xyz","B.xyz"]` with `separation="12 angstrom"`. In dRPMD, separation is
the initial **radial distance** R0 between the two reactant centroid centers of mass.
Launches use `z=sqrt(R0^2-b^2)` to preserve that radial distance and require `bmax<R0`.

Python or TOML inputs may instead provide `symbols`, `geometry=[[...], ...]`, and
`masses` (electron masses) or `masses_amu`. `geometry_unit` overrides coordinate units.
Specify isotopes with, for example, `isotopes=[{atom=0, mass="2.01410177812 amu"}]`.
Masses are resolved before reactants are centered and placed. `charge` is an integer;
`multiplicity` is a positive integer. Their defaults are 0 and 1, and they must match
the selected electronic potential energy surface. Spin statistical weights are not
inferred from geometry.

Use `cell=[[...],[...],[...]]` (one vector per row), `cell_unit="angstrom"` and
`pbc=[true,true,false]` for a fixed cell. The default is full periodicity when a cell
is present and nonperiodic boundaries otherwise. See [materials](materials.md) for
cells, partial PBC, continuous centroid coordinates and supported potential interfaces.

`system.bond_constraints` accepts records such as
`[{atoms=[0,1], length="0.9572 angstrom"}]`. These per-bead constraints currently require
`velocity-verlet`, Andersen sampling, and fixed-cell `pimd` or `diffusion`; consult the
[bond-constraint guide](bond-constraints.md) for their supported combinations and constrained-ensemble scope.

## Periodic surfaces

Periodic surfaces use `system.cell` (3×3 row vectors), `cell_unit` (inherited from
`geometry_unit` by default: bohr for inline geometry, angstrom for XYZ),
`pbc=[true,true,false]`, and `fixed_atoms=[...]`. These fields enter the scientific
restart fingerprint. `task="surface_rate"` also requires `reactant_basin` and
`dividing_surface` in `[surface]`; `product_side` is `higher` (default) or `lower`.
Rates use a position-density PMF and first-order basin normalization, in s^-1.
Supported coordinates include built-in linear centroid coordinates, continuously
unwrapped `distance`, and `periodic_distance` satisfying the smooth-surface conditions.

`surface.normalization` explicitly specifies area/coverage conversion.
`surface.free_energy` can use a saved multidimensional FES, retaining the source
system's CVs, static walls and channel regions. Gas-phase `rate`, `drpmd` and `network`
reject periodic or fixed-atom systems. Surface FES calculations can use adaptive
umbrellas; `surface_rate` currently rejects `[adaptive_umbrella]`. `metadynamics`
supports 1–3 CVs, periodic/fixed-atom systems, NHC/PILE-L/Andersen and quadratic walls.
See [surface catalysis](surface-catalysis.md) for configuration, formulas and limitations,
and [grid metadynamics](grid-metadynamics.md) for 1D/2D bias grids.

## Thermalization and persistent parent trajectories

```toml
[sampling]
integrator = "cayley-bcb-andersen"
thermostat = "andersen"
equilibration = "5 ps"
interval = "0.2 ps"
production = "20 ps"
andersen_interval = 10
replicas = 2
parents = 8
```

`equilibration` is each chain's initial equilibration time. For dRPMD,
`interval`/`interval_steps` is **the time/number of steps advanced between successive
draws from the same parent trajectory**. It is not a temperature, and a new draw does
not restart thermalization from the initial geometry. `parents` sets the number of
parent chains; if omitted, it defaults to the trajectory worker count.
`collision.trajectories=N` is the total number of collisions, distributed as
`N//parents` plus the remainder. Set `parents` explicitly on the first run if you
intend to change the worker count on restart.

For FES, `replicas` sets the independent chains per window, `production` is the
production duration per chain, and `interval` is the CV sampling interval. FES,
gas-phase and site-rate workflows default to the selected dynamical integrator plus
Andersen. An omitted `sampling.integrator` is matched automatically; explicit
Andersen selection uses the canonical integrator name with `-andersen` appended.
FES also supports PILE-L. `pimd`/`diffusion` default to BCOCB/PILE-L with Cayley,
BAOAB/PILE-L with normal modes, and Andersen with Verlet.
`sampling.gamma_centroid` sets the PILE-L centroid friction. Sampling and child
trajectories must use the same timestep. Physical collisions, recrossing children and
the default `diffusion` production segment propagate without a thermostat.

`diffusion` may add `[electronic_friction]` for a physical electronic bath during
production, using Markovian centroid EF-RPMD. This section accepts `method="centroid"`
and `provider="constant"` or `"python"`. A constant provider requires exactly one of
the relaxation rate `gamma` and physical friction `tensor`; a Python provider uses
`factory` and `params`. Bare `gamma` values are inverse atomic time; strings such as
`"0.01 fs^-1"` are also accepted. `tensor` has shape `[3*N,3*N]` in electron
mass/atomic time. Fixed substrate atoms are allowed; friction acts only on mobile
degrees of freedom. Production exchanges energy with the electronic bath, so the NVE
drift criterion is not applied. Other tasks, explicit `loop_backend="native"` and
mixed mapping potentials reject this configuration. See [electronic friction](electronic-friction.md)
for the interface, correlated noise, model limitations and examples.

## Materials tasks

`pimd` and `diffusion` use the sampling times above and reject reaction, umbrella and
metadynamics biases. `diffusion` adds `[diffusion]` with `atom_indices`, `axes`,
`reference_atoms`, `fit_range`, `max_lag`, `bootstrap`, `seed` and `block_length`.
`max_lag`/`block_length` are measured in saved frames; `fit_range` contains two times.
All selection indices are zero-based. The linear-coordinate `site_rate` setup uses a
centroid CV, fixed umbrellas and `[recrossing]`, with explicit `cv`, `basin`,
`dividing_surface` and `direction` in `[site_rate]`. See [materials](materials.md)
for all three tasks and their results; [general-coordinate rates](general-coordinate-rates.md)
describes the nonlinear extension.

`pimd` and `diffusion` allow some atoms to be fixed. Measured diffusion atoms must be
mobile; reference atoms may be fixed. `pimd` may add `[barostat]` with `method="bzp"`,
`pressure="1 bar"`, `time_constant="1000 fs"` and optional `friction`. This requires
three-dimensional full periodicity, all-mobile atoms and a PES supporting variable
cells and physical virials. Isotropic pressure does not change cell shape. See
[NPT](materials-npt.md). Materials trajectories stream by default; see
[chunks and restart](streaming-trajectories.md).

## dRPMD

```toml
[collision]
trajectories = 10000
sampling = "thermal_flux"
bmax = "5 angstrom"
max_time = "2 ps"

[outcomes]
exit_distance = "12 angstrom"
persistence = "20 fs"
check_interval = 5
bond_cutoffs = { "O-H" = "1.3 angstrom", "H-H" = "0.9 angstrom" }
channels = { water = ["H2O", "O2"] }
```

The example channel name/composition illustrates syntax and must conserve the actual
reaction's atoms. `thermal_flux` samples the flux distribution E~Gamma(2,kBT).
`maxwell` samples the ordinary relative-speed distribution and weights each trajectory
by v. `fixed_energy` additionally requires `energy` and yields a scattering cross
section at that energy, not a thermal rate. Impact parameters are uniform in area,
and overall spatial orientations are isotropic.

When a minimal pilot omits classification settings, the program infers connectivity
cutoffs from covalent radii with a factor of 1.25 and reports provisional candidate
labels, inferred values and assumptions. It does not infer charge, bond order or
complete chemical identity. Thermal rates, cross sections and chemical-outcome bounds
are withheld. Production inputs require explicit cutoffs or a custom classifier.

The explicit built-in classifier uses centroid connectivity, molecular formulas,
separation and persistence. It cannot reliably distinguish all isomers of the same
composition or same-element exchange channels. Those cases require
`outcomes.factory="classify.py:build"`. The plugin returns `reactants`, a product
channel name, or an unresolved status and must define the relevant physical criteria.
See [potentials and plugins](potentials.md) for the protocol. Unresolved and failed
samples are counted separately; when present, the thermal-rate point estimate is
withheld and observed lower/possible upper bounds are supplied. Statistical confidence
intervals and these truncation bounds have different meanings.

## Standard RPMD rates

A minimal `rate` input prepares the calculation from a TS and PES. When the entrance
cannot be selected uniquely, provide:

```toml
[reaction]
fragments = [[0], [1, 2]]
forming_bonds = [[0, 1]]
breaking_bonds = [[1, 2]]
rinf = "12 angstrom"
electronic_weight = 1.0
symmetry_factor = 1.0
```

A single `task="rate"` uses one TS reference geometry and equal-length, paired lists
of forming/breaking bonds. Use `task="network"` below for different entrances or
inequivalent TS structures. `equivalent_paths` declares equivalent transition states
constructed by atom permutation. For example, the two H + H₂ exchange exits are:

```toml
[reaction]
fragments = [[0], [1, 2]]
forming_bonds = [[0, 1]]
breaking_bonds = [[1, 2]]
equivalent_paths = [[0, 2, 1]]
rinf = "10 bohr"
symmetry_factor = 1.0
electronic_weight = 1.0
```

Permutations are zero-based `old atom index → new atom index` mappings. The original
TS is already included; do not add the identity permutation. The program checks a
complete bijection, elements, isotope masses and entrance fragment partition, then
permutes both reference geometry and bonds. Declare distinct TS structures through
`network.transition_states`. Combination follows RPMDrate's
`min_bond(max_TS(s1))` exactly and is piecewise differentiable at branch boundaries.
Do not multiply by the path count again through `symmetry_factor` when equivalent
exits are explicitly included. Users still supply `electronic_weight` and any
additional `symmetry_factor` for the selected PES and channel definition.
`reaction.entry_side` selects either end of the numerical downhill paths.

The main `[recrossing]` fields are `parent_replicas`, `parent_samples`,
`equilibration`, `parent_interval`, `child_pairs`, `child_time`, `dividing_surface`,
`plateau_fraction`, `plateau_tolerance` and `bootstrap`. Corresponding `_steps` fields
can specify times. Each child pair propagates with opposite momenta; an uncomputed
κ=1 is not a substitute for recrossing. See `examples/rate/pilot.inp` for runnable syntax.

### Rate PMF backend and dividing surface

The default `rate.pmf_method="ui"` retains Gaussian umbrella integration. It assumes
nearly Gaussian window distributions; skewness, multiple modes and tail extrapolation
can introduce bias outside statistical error bars. UI can retain model values with a
warning when measured endpoint support is missing. This does not establish reliable
endpoint sampling. To select histogram MBAR and fix a dividing surface:

```toml
[rate]
pmf_method = "mbar"
dividing_surface = 1.00925
compare_ui = true

[analysis]
bins = 170
ranges = [[-0.30, 1.40]]
min_count = 5
bootstrap = 40
seed = 12345
```

The dividing surface and bin range above are H₃ example parameters; adapt them to
other reactions' coordinates and sampling support. With MBAR, `compare_ui` defaults
to true. UI uses the same samples and block-bootstrap draws, allowing paired
differences without repeating MBAR resampling. `bootstrap=0` is suitable for curve
diagnostics only and cannot provide a complete rate uncertainty. `bootstrap=1` cannot
estimate a variance and is rejected during input parsing.

If any rate-PMF bootstrap draw fails support or estimation, the main barrier and
final rate errors/confidence intervals are unavailable. `supported_draw_*` describes
only the accepted subset with complete support and valid fits; it is not an
unconditional uncertainty. Inspect requested, attempted, valid and rejected counts
and block status. More bootstrap draws do not replace more independent production data.

Omitting `dividing_surface` or using `"auto"` selects the peak automatically;
`rate.pmf_search_range=[0.8,1.1]` restricts the search. MBAR uses supported, unsmoothed
bin centers; the entire interval from reference point to dividing surface must have
continuous support. Bootstrap keeps the chosen surface fixed and excludes uncertainty
in automatic peak selection. Legacy `recrossing.dividing_surface` remains available;
inconsistent values specified in both places raise an error. Parent/child checkpoints
cannot be reused after changing the PMF method or dividing surface.

`rate_pmf.npz` and `rate_pmf.json` store densities, masks, uncertainties, endpoint
support, solver residuals and individual resampling statuses. Disconnected support,
missing endpoint/interior bins, and solver failure receive distinct failure statuses
and stop recrossing launch. Barriers are not interpolated across masks. See
[MBAR rate PMFs](mbar-rate-pmf.md).

## One- and two-dimensional FES

```toml
[[cvs]]
name = "r1"
kind = "distance"
atoms = [0, 1]
estimator = "centroid"
unit = "angstrom"

[[cvs]]
name = "r2"
kind = "distance"
atoms = [1, 2]
estimator = "centroid"
unit = "angstrom"

[umbrella]
cvs = ["r1", "r2"]
grid = [{start=0.8, stop=2.0, count=9}, {start=0.8, stop=2.0, count=9}]
stiffness = ["40 kcal/mol/angstrom^2", "40 kcal/mol/angstrom^2"]

[analysis]
method = "mbar"
measure = "cv_density"
bins = [40, 40]
bootstrap = 100
min_count = 5
```

Built-in CVs are `distance`, `bond_difference`, `angle`, `cartesian`, `com_distance`,
`rpmdrate_xi`, `surface_height`, `surface_lateral`, `periodic_distance` and `python`.
Geometric CVs support `centroid` or `bead_average`; custom CVs return their own full
estimator and derivatives. `stiffness` may be a scalar, diagonal array, or symmetric
positive-definite matrix. Numeric centers use the declared CV input units;
`centers_au` and bare numeric stiffness use atomic units. Unit-bearing strings are
checked dimensionally.

The FES is a density relative to the chosen CV measure. For example, a distance-CV
free energy naturally includes radial entropy; it is not automatically converted into
a Jacobian-corrected pair potential. Regions lacking samples or connected overlap
support are masked, not extrapolated. `analyze --recompute --bins ... --bootstrap ...`
re-estimates from saved samples without calling the PES.

The main per-bin `standard_error` is reported only when every requested bootstrap
draw supports that bin and correlation blocking is available. Otherwise it is NaN,
with `supported_draw_standard_error` and `bootstrap_valid_per_bin` retained for
accepted-subset diagnostics. Support for a point estimate does not establish that
its uncertainty is estimable; conditional errors do not replace unconditional intervals.

## Multiple entrances and distinct transition states

`task="network"` uses lists of `entrances`, `channels` and `transition_states` records
in `[network]`. Each TS explicitly references an entrance and channel label and
supplies its own structure and bonds. PES and sampling settings inherit or override
in the order global → entrance → TS. Each entrance specifies two fragments, `rinf`
and electronic statistical weight. All paths share an ordered atom/isotope inventory.
Each path undergoes ordinary `rate` validation and a full PMF/recrossing calculation.

Currently `network.aggregation` accepts only `"separate"`. Results preserve independent
entrance/TS normalizations and uncertainties. A channel name does not prove that
fluxes are disjoint. See [reaction networks](reaction-networks.md) for TOML/Python
syntax and a runnable four-path example.

## Adaptive umbrella sampling

Add `[adaptive_umbrella]` to `fes` or `rate`:

```toml
[adaptive_umbrella]
enabled = true
pilot_steps = 1000
# An omitted pilot_equilibration_steps inherits sampling.equilibration_steps.
max_rounds = 8
max_windows = 128
max_pilot_steps = 16000
target_overlap = 0.03
min_effective_samples = 20.0
min_spacing = 0.000001
```

`[umbrella]` still defines initial windows, ranges and stiffness. Adaptation inserts
windows and extends pilots using measured MBAR overlap and effective sample size.
Once windows are frozen, independent seeds are used for re-equilibration and
production. Pilot samples do not enter the final FES/PMF. Budget exhaustion fails
explicitly rather than bypassing overlap requirements. See [enhanced sampling](enhanced-sampling.md)
for parameter units, stage restart and applicability.

New schema-3 plans score overlap as `sqrt(O_ij * O_ji / (O_ii * O_jj))`, recorded as
`normalized_mbar_cosine`. Identical window distributions score 1, avoiding dilution
of the old score as window count increases. This is not a replica-exchange acceptance
probability; the default 0.03 is a heuristic screen, not proof of global mixing.
Existing schema-1/2 plans retain `sqrt(O_ij * O_ji)` (`mbar_symmetric_transition`) and
their committed decisions. Thresholds in the two definitions are not directly
comparable. Start a new output directory to reselect windows using the new metric.

For two-dimensional surfaces, explicitly enabling spatial coverage checks is
recommended. Add these entries to the same table:

```toml
coverage_bins = [4, 8]
coverage_ranges = [["-0.5 bohr", "0.5 bohr"], ["-1 bohr", "1 bohr"]]
min_bin_effective_samples = 5.0
target_coverage = 0.95
```

These ranges are examples and must match the actual CVs. Both grid parameters are
required together; checks are disabled by default. Local effective sample sizes use
MBAR weights and trajectory-correlation corrections; weak grid centers receive new
windows. This grid is independent of final analysis bins. Global window ESS also
checks coordinate second moments so fast oscillations do not hide slow amplitude
convergence. Production independently rechecks the same overlap, ESS and optional
coverage criteria. `pilot_target_met` and `production_target_met` report each stage;
legacy `target_met` aliases the pilot result only.

FES reports these fields under `adaptive_umbrella` in `RUN/results/summary.json`.
Gas-phase rates expose them under `pmf.adaptive_umbrella` in the same final summary;
`RUN/rate_pmf.json` and `RUN/umbrella/summary.json` retain `adaptive_umbrella` as well.
Inspect `production_overlap_connected_at_target`,
`production_effective_samples_target_met`, and the separate `pilot_coverage` and
`production_coverage` (`coverage` is the legacy pilot alias). A run without coverage
checks does not establish support over the whole domain. Failed production checks
warn but do not automatically extend sampling or delete the point estimate.
Independent repeats and physical convergence studies remain necessary.

Insufficient window ESS, or insufficient coverage when no more windows can be added,
doubles the next pilot duration. Inserting windows alone does not lengthen each
window. `max_pilot_steps` defaults to `16 * pilot_steps`. When production duration is
not explicitly set, adaptive and metadynamics jobs default to 1000 fs; other pilot
jobs retain 200 fs. Explicit durations are unchanged, and production ESS/coverage
still need separate checks. After exhausting the budget, increase limits with
`pyrpmd resume RUN --adaptive-max-rounds 12 --adaptive-max-windows 192`.
Completed pilots are reused. This interface cannot lower statistical thresholds or
reopen frozen windows.

At the default 0.1 fs timestep, 1000 initial steps are only 100 fs and 16000 maximum
steps are 1.6 ps per window/replica. Pilot equilibration inherits the default production
setting of 100 fs (1000 steps), samples are saved every 1 fs, and there is one replica
per window by default. ESS 20 is a window-selection screen; the default 20 bootstrap
draws are a short analysis budget. Neither guarantees physical precision. Gas-phase
rates still default to UI; MBAR overlap checks during adaptation do not switch the
final estimator to MBAR. See the [scientific convergence guide](manual/science.md)
for acceptance criteria and uncertainty budgets.

## Metadynamics

`task="metadynamics"` uses ordinary `[[cvs]]` and `[sampling]` and replaces umbrella
settings with:

```toml
[metadynamics]
cvs = ["x"]
mode = "well_tempered"
sigma = ["0.2 bohr"]
height = "0.02 kcal/mol"
pace = 50
bias_factor = 8.0

[analysis]
method = "bias_estimate"
measure = "cv_density"
bootstrap = 0
```

`mode="standard"` uses fixed hill heights and omits `bias_factor`. `sigma_au` and
`height_au` specify atomic units but must not duplicate the corresponding fields
without `_au`. `pace` is an interval in production steps. One to three CVs, periodic
CVs and independent replicas are supported; optional bias grids are limited to 1D/2D.
`loop_backend="native"` fails before propagation because bias history changes over
time. The compatible `auto` path may still use the chosen integrator's compiled
substeps. See [enhanced sampling](enhanced-sampling.md) for finite-time FES estimates
and HILLS/restart timing.

## Electrochemical PCET model

`task="pcet"` computes one-dimensional single-proton golden-rule rates for a fixed
local complex: reduction/oxidation rates in s⁻¹ and H/D KIEs. It does not propagate
RPMD trajectories. Instead of `[system]` and `[potential]`, supply `[pcet]`,
`[pcet.electrode]`, `[pcet.proton]` and the proton potentials of the two electronic states:

```toml
task = "pcet"
temperature = ["250 K", "300 K", "350 K"]

[pcet]
method = "golden_rule"
normalization = "local_first_order"
isotopes = ["H", "D"]
driving_force = "0.0 eV"
reorganization_energy = "0.7 eV"
electronic_coupling = "0.001 eV"

[pcet.electrode]
model = "wide_band"
density_of_states = "1 eV^-1"
potential_reference = "SHE"
reference_potential_V = 0.0
potentials_V = [-0.2, 0.0, 0.2]

[pcet.proton]
grid_min = "-1.5 angstrom"
grid_max = "1.5 angstrom"
grid_points = 401
states = 12

[pcet.proton.oxidized]
kind = "harmonic"
center = "-0.2 angstrom"
force_constant = "4 eV/angstrom^2"

[pcet.proton.reduced]
kind = "harmonic"
center = "0.2 angstrom"
force_constant = "5 eV/angstrom^2"
```

These are teaching parameters without a real-material assignment. `driving_force`
is the reduced-minus-oxidized minimum free-energy difference at the reference
potential, excluding the transferring proton's zero-point and thermal vibrational
free energies. Each proton potential has its own minimum removed; the eigenstate
solver supplies the proton quantum contribution. Increasing potential increases the
reduction free energy by `+e*(U-U_ref)`. `potential_reference` records the reference
scale only; it does not convert SHE/RHE or introduce pH dependence.

`[pcet.numerics]` accepts `quadrature_rtol`, `quadrature_atol`, `convergence_check`,
`relative_tolerance` and `absolute_tolerance_s`. Numerical convergence is reported
separately from applicability of the weak-coupling, wide-band and classical-environment
assumptions. See the [PCET specification](pcet.md) and [teaching examples](../examples/pcet/README.md)
for runnable inputs, tabulated potentials, rate formulas and output interpretation.

## Run and restart records

`manifest.json` v2 records program version, environment, parallel configuration,
units, scientific fingerprint and SHA256 hashes of local input/PES/plugin contents.
`input/resolved.json` is the normalized configuration; `input/system.json` stores
atoms and masses. New NPZ checkpoints use v2 containers with per-array integrity
checks and always read arrays with `allow_pickle=False`. RNG/progress state is stored
as JSON strings in the same container or explicit workflow metadata. See
[checkpoints](checkpoints.md) for legacy v1 reads and manifest migration. Arbitrary
transitive Python dependencies and remote PES service state are not automatically
included in the dependency closure; pin the execution environment separately.

PCET stores its one-dimensional model in `input/model.json`, without a molecular
`system.json`. Scan-point checkpoints are JSON with model identity and integrity
checks; they contain neither q/p nor RNG state.

## Continuous native propagation

`[dynamics]` accepts `loop_backend="auto"|"native"|"python"` and `block_steps=256`
(range 1..65536). `auto` uses C++17 continuous propagation for supported native PES/CV
combinations and retains Python elsewhere. `native` fails explicitly if the requested
block path is unsupported. `python` keeps per-step scheduling without disabling the
selected integrator's compiled single-step kernels.

Constrained recrossing parents, dRPMD collisions and native-capable FES/PMF sampling
use `propagate_block()`, limited to 4096 steps per call. Workflows cap larger
`block_steps` at that limit while preserving thermostat, classification, sampling and
checkpoint schedules. The accepted input range and default remain 1..65536 and 256.
Recrossing children and ordinary Andersen thermalization continue to use `propagate()`.
See [native propagation](native-propagation.md) and the [block API](block-propagation.md)
for supported combinations and the two APIs' failure semantics.
