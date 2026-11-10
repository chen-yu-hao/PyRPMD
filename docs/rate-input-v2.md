# Explicit rate inputs and regional umbrellas

**English** · [中文](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/zh/rate-input-v2.md)

[`examples/rate/research.inp`](../examples/rate/research.inp) is the recommended
input layout for a rate study. It declares the entrance fragments and bond
changes, checks the supplied TS, and gives each sampling stage its own controls.
Its analytic teaching PES and short pilot budgets are not production parameters.
The old [`pilot.inp`](../examples/rate/pilot.inp) remains a schema-1 compatibility
example. Schema 1 keeps its historical defaults and checkpoint interpretation;
`schema_version = 2` opts into explicit reaction identity and independent parent
sampling defaults.

## Reaction identity and TS preparation

```toml
schema_version = 2
task = "rate"

[system]
transition_state = "ts.xyz"

[reaction]
definition = "explicit"
reactant_fragments = [[0], [1, 2]]
product_fragments = [[0, 1], [2]]
forming_bonds = [[0, 1]]
breaking_bonds = [[1, 2]]
rinf = "5 angstrom"

[preparation]
ts_action = "validate"
check_channel = true
```

All indices use the same zero-based atom order as the full-system TS. Separate
reactant XYZ files are not required for this TS-based workflow. Fragment
partitions must cover the system without missing or repeated atoms.

`definition="explicit"` is the schema-2 default. Choose `definition="infer"`
explicitly to infer an entrance from the unstable mode and downhill traces;
[`minimal.inp`](../examples/rate/minimal.inp) demonstrates this convenience mode.
Inference requires a channel check, and ambiguous or inconsistent candidates do
not become ready runs.

`ts_action="validate"` checks the supplied coordinates for a stationary,
first-order saddle without moving them. `"refine"` permits P-RFO refinement
before checking; `"none"` explicitly skips the saddle check and is restricted to
a supplied TS, a complete explicit reaction and `check_channel=false`. Endpoint
search remains available through the existing atom-mapped endpoint input.

Knowing the reaction does not certify the geometry: explicit schema-2 inputs
still require numerical TS validation. `check_channel` independently requests
both downhill traces and comparison with the declared entrance, bond changes
and optional product partition. This connectivity check is not a claim of a
converged quantum-chemical IRC. Preparation reports `saddle_valid`,
`channel_consistent` and `refinement_performed` separately. Static input checking
does not evaluate a PES; use `validate --check-pes` or `prepare --pilot` for
numerical preparation.

## Temperature-normalized force constants: the RPMDrate convention

The new `force_constant` field defaults to the temperature-normalized coefficient
`a = k / T`, with temperature in kelvin and stiffness in atomic units. For the
dimensionless rate coordinate, its physical unit is hartree/K; the shorthand
`1/T` denotes this convention. The physical umbrella is

$$
W_i(\xi)=\tfrac12 k_i(\xi-\xi_i)^2,\qquad
k_i\,[\mathrm{hartree}]=a_i\,T\,[\mathrm K].
$$

This follows the RPMDrate example's
[`kforce = 0.1 * T`](https://github.com/chen-yu-hao/RPMDrate/blob/710670c6f753b1159c0147c817e82998c2cc8486/examples/H%2BH2/input.py).
Its [Fortran bias](https://github.com/chen-yu-hao/RPMDrate/blob/710670c6f753b1159c0147c817e82998c2cc8486/rpmdrate/_main.f90)
uses that physical stiffness directly. The coefficient is multiplied by T when
resolved; the dynamics does not divide the force constant by temperature.

| Input | Physical stiffness for dimensionless xi |
| --- | --- |
| `force_constant = 0.00016` | `0.00016 * T` hartree |
| `force_constant = "0.00016 1/T"` | Same explicit shorthand |
| `force_constant = "0.00016 hartree/K"` | Same explicit physical unit |
| `force_constant = "0.16 hartree"` | Fixed 0.16 hartree, independent of T |
| Legacy `stiffness = 0.16` or `kappa = 0.16` | Fixed 0.16 atomic stiffness units, unchanged |

Thus the new teaching template uses `0.00016` at 1000 K to retain the old
0.16-hartree spring. Copying RPMDrate's coefficient `0.1` would instead give
100 hartree at 1000 K; it is not an equivalent conversion. The omitted rate
default retains its old physical strength, `50 k_B T`, rather than silently
changing to a particular RPMDrate example's value.

For a dimensional CV, supply the corresponding stiffness/K unit, such as
`hartree/bohr^2/K`; the `1/T` shorthand is limited to dimensionless CVs. Bare
coefficients in `force_matrix` follow the same atomic-stiffness-per-kelvin rule.
The existing scalar/diagonal/full `stiffness` forms remain available for multi-CV
FES. The program applies the ring-polymer factor P internally; do not multiply
the input by the bead count.

**Check the width as well as the spacing.** For the dimensionless rate coordinate
the harmonic thermal width is
`sigma_xi = sqrt(k_B T / k) = sqrt(k_B / a)` when using a bare
`force_constant = a` (`k_B = 3.16681156e-6 hartree/K`). Thus the example
`a=0.00016` has `sigma_xi ≈ 0.141` and the barrier-region
`a=0.00032` has `sigma_xi ≈ 0.0995`, versus spacings of 0.05 and
0.01. These broad windows can violate Gaussian UI's assumption even with
adequate overlap and a small bootstrap error. As a narrower starting point,
tenfold stiffer `a=0.0016`/`0.0032` gives widths ≈0.0445/0.0315;
use correspondingly closer, overlapping windows (for example 0.02/0.01
spacing), then check convergence. RPMDrate's `a=0.1` is much stiffer
(`sigma_xi ≈ 0.00563`) and requires much closer centers.

`pyrpmd prepare` reports `thermal_sigma_xi` and
`nearest_center_spacing_xi` for every planned one-dimensional rate umbrella
window under `umbrella_preparation.window_plan`. The UI workflow additionally
compares its barrier to the already computed MBAR histogram point estimate,
emitting an explicit warning for a significant difference or an unsupported
comparison (see [PMF selection](rate-pmf-selection.md)). Width, spacing and
that comparison are diagnostics, not proof of physical convergence.

## Regional and explicit windows

```toml
[umbrella]
coordinate = "xi"
force_constant = 0.00016

[[umbrella.regions]]
id = "entrance"
start = 0.0
stop = 0.8
spacing = 0.05
include_endpoints = "left"

[[umbrella.regions]]
id = "barrier"
start = 0.8
stop = 1.05
spacing = 0.01
include_endpoints = "both"
force_constant = 0.00032
```

The first region generates 16 centers from 0 to 0.75; the second generates 26
from 0.8 to 1.05, for 42 windows. Endpoint inclusion is explicit (`left`, `right`,
`both`, `neither`), and a spacing must divide its range without silently adding
a shortened final interval. Duplicate centers and conflicting definitions are
rejected. Sampling distributions should still overlap; the rule concerns
conflicting input definitions, not statistical overlap.

Choose one center source: `centers`, `grid`, `regions`, `windows`, or the legacy
`start/stop/count` form. A new input cannot combine selectors and rely on an
implicit precedence. Regions are one-dimensional; arbitrary multidimensional
windows use explicit records and a positive-definite force matrix.

```toml
[[umbrella.windows]]
id = "near-saddle"
center = 0.95
force_constant = 0.00032

[umbrella.windows.sampling]
production = "40 fs"
```

Region/window records inherit common umbrella settings and can override their
own force constant or sampling budget. `regions` and `windows` are alternative
sources, not simultaneous layers. Different K values propagate into each
window's dynamics, saved metadata and UI/MBAR reconstruction. Fixed independent
windows support heterogeneous K; adaptive and replica-exchange combinations
that still require a shared K are rejected explicitly.
Replica exchange also retains its existing thermostat-clock interface: an
explicit new Andersen schedule is rejected there, including `schedule="fixed"`.
The operation-named Andersen examples in this guide target independent windows
and gas-phase rate sampling.
Ordinary run, analysis and resume support window-local sampling budgets.
The separate `extend --production-steps` ledger currently requires a common
sampling budget and rejects region/window `sampling` overrides before touching
the run. Different force constants with a common budget can be extended.

## Operation-named sampling and separate recrossing stages

```toml
[umbrella.sampling]
equilibration = "5 fs"
production = "20 fs"
sample_every = "0.2 fs"
chains_per_window = 1

[umbrella.sampling.thermostat]
type = "andersen"
scheme = "global_refresh"
schedule = "fixed"
refresh_every = "2 fs"

[recrossing.parent]
chains = 2
equilibration = "5 fs"
draw_every = "1 fs"
configurations_per_chain = 4

[recrossing.parent.thermostat]
type = "andersen"
scheme = "global_refresh"
schedule = "fixed"
refresh_every = "2 fs"

[recrossing.children]
method = "rpmd"
pairs_per_configuration = 2
duration = "10 fs"
```

At a 0.1-fs timestep, `sample_every="0.2 fs"` records CV observations every two
production steps, whereas `refresh_every="2 fs"` redraws global ring momenta
every twenty steps. `chains_per_window` counts independent random streams per
window, independently of execution workers. Durations accept unit strings or
`{steps=N}`; conflicting old/new declarations are rejected.

For a geometric Andersen clock, use `schedule="geometric"` and
`mean_refresh_interval="2 fs"` instead of `refresh_every`. The clock starts at
step zero and continues through equilibration and production. The first
production observation/draw follows one full observation/draw interval after
equilibration. Raw sample counts do not imply independent effective samples.

The constrained parent has its own clock. New stage inputs do not inherit the
umbrella thermostat; omitted parent-clock settings use its independent default
of ten fixed MD steps. The example draws 8 parent configurations and launches
16 plus/minus pairs, or 32 trajectories; paired trajectories are correlated.
Standard RPMD children do not inherit the sampling thermostat, umbrella spring
or dividing-surface constraint. Select `method="trpmd"` and its friction
explicitly when that different dynamics is intended.

`[umbrella.analysis]` accepts `estimator`, `integration_points`, `bins` and
`bootstrap_replicates`; `[recrossing.analysis]` controls the recrossing bootstrap
and plateau diagnostics separately. Rate-flux measures, statistical
normalization and paired estimators are unchanged.

## Validate, preview, run and resume

```bash
pyrpmd validate examples/rate/research.inp --check-pes
pyrpmd prepare examples/rate/research.inp --pilot --output preparation.json
pyrpmd run examples/rate/research.inp --output runs/research-layout
pyrpmd analyze runs/research-layout
pyrpmd resume runs/research-layout
```

Inspect `umbrella_preparation.window_plan`, the physical stiffnesses and the
separate saddle/channel statuses before a study. `recrossing_preparation` lists
planned parent configurations, pairs and individual trajectories. A purely
structural preview can resolve an explicit channel's window plan while
`ready=false` still records a deferred numerical TS check. During sampling,
`umbrella-plan.json` and each window checkpoint retain the resolved bias identity.
Changing a scientific window/stage setting cannot be
used to reinterpret a saved checkpoint; resource-only changes retain their
existing resume rules. Keep historical runs on their stored schema and inputs.
When manually converting old inputs, convert an old physical spring to `k/T`,
and copy a shared old Andersen clock to both new stage tables if the aim is to
preserve its old behavior. Do not edit an existing run's saved input in place.
