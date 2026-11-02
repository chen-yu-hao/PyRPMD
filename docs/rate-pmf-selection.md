# Choose the PMF method for an RPMD rate

**English** · [中文](zh/rate-pmf-selection.md)

For the standard gas-phase `task="rate"` workflow, use `[rate.pmf]` to choose
how to obtain the PMF. The subsequent constrained parent chains and paired
recrossing trajectories remain part of the same rate calculation.

| `rate.pmf.method` | `rate.pmf.estimator` | What happens |
| --- | --- | --- |
| `"umbrella"` | `"ui"` (default) or `"mbar"` | Sample fixed or adaptive umbrella windows, then reconstruct the PMF. |
| `"metadynamics"` | `"reweighted_histogram"` (default) | Learn a bias, freeze it, relax, then reweight fixed-bias production samples. |
| `"opes"` | `"reweighted_histogram"` (default) | Learn a probability bias, freeze it, relax, then reweight fixed-bias production samples. |
| `"external"` | Omit this field | Read a compatible saved `FESResult`; do not generate new PMF trajectories. |

`method` selects sampling or an existing source. `estimator` selects the
reconstruction. UI and MBAR both use umbrella samples. A raw MetaD bias
estimate or OPES KDE is not accepted as a rate PMF.

## Start with a complete example

The three examples use the existing analytic teaching PES and TS:

```bash
pyrpmd validate examples/rate/pmf_umbrella.inp --check-pes
pyrpmd validate examples/rate/pmf_metadynamics.inp --check-pes
pyrpmd validate examples/rate/pmf_opes.inp --check-pes

pyrpmd run examples/rate/pmf_umbrella.inp --output runs/rate-umbrella
```

Select either of the other input files in the same `run` command to exercise
that method. These are short input/workflow pilots, not a physical H + H2
benchmark or a convergence prescription. In particular, a short enhanced
sampling run can leave the reactant reference or barrier unsupported; that
condition withholds the rate instead of extrapolating a missing PMF.

## Separate PMF controls from recrossing controls

Top-level dynamics and sampling retain their existing rate meaning. Local
PMF tables override their shared defaults for the PMF stage only:

```toml
[dynamics]
integrator = "cayley-bcb"
timestep = "0.1 fs"

[sampling]
thermostat = "andersen"

[rate.pmf]
method = "umbrella"
estimator = "mbar"

[rate.pmf.dynamics]
integrator = "normal-mode-bab"
timestep = "0.2 fs"

[rate.pmf.sampling]
thermostat = "pile-l"
gamma_centroid = "0.02 fs^-1"
equilibration = "10 fs"
production = "200 fs"
interval = "1 fs"
replicas = 2

[rate.pmf.umbrella]
start = -0.05
stop = 1.10
count = 47
kappa = "0.16 hartree"

[rate.pmf.analysis]
bins = 80
bootstrap = 40
```

The PMF here uses normal-mode BAB with PILE-L (BAOAB sampling) at 0.2 fs.
Constrained parents and RPMD children use the root Cayley settings at 0.1 fs.
Keep the root sampler Andersen; select other canonical PMF thermostats in
`rate.pmf.sampling`. Omitting `sampling.integrator` lets the existing sampler
choose the matching splitting. For example, PILE-L pairs with Cayley/BCOCB or
normal-mode/BAOAB, and is incompatible with velocity-Verlet.

| Setting | Where to put it | Scope |
| --- | --- | --- |
| Temperature, beads, PES, system, reaction | Existing top-level fields/tables | Shared by both stages; no PMF-specific override. |
| Recrossing integrator and timestep | `[dynamics]` | Constrained parents and child trajectories. |
| Parent/child counts and durations | `[recrossing]` | Recrossing stage. |
| PMF integrator and timestep | `[rate.pmf.dynamics]` | PMF stage only. |
| PMF thermostat, time budget, replicas | `[rate.pmf.sampling]` | PMF stage only; unspecified settings inherit shared defaults. |
| Bias parameters | `[rate.pmf.umbrella]`, `[rate.pmf.metadynamics]`, or `[rate.pmf.opes]` | The selected method only. |
| Adaptive windows | `[rate.pmf.adaptive_umbrella]` | Umbrella method only. |
| Histogram and uncertainty settings | `[rate.pmf.analysis]` | PMF analysis, with method-specific compatibility checks. |
| Dividing surface/search range | Existing `[rate]` fields | Shared PMF endpoint and recrossing surface. |

Durations are converted using the timestep of their own stage. Explicit
`*_steps` values are counts of that stage's steps. If a local time option is
provided, it replaces the corresponding inherited duration/step option. The
PMF's own sampling and dynamics timesteps must agree.

Explicit `centers` may be nonuniform. The umbrella stiffness remains common
to all windows; a stiffness list describes CV dimensions, not per-window
force constants. REUS is not part of this gas-rate interface.

## MetaD and OPES: make the sampling clock explicit

For either enhanced method, specify both `freeze_after_steps` and
`freeze_equilibration_steps` in its method table. The learning period ends
on a deposition/update event, so `freeze_after_steps` must be a positive
multiple of `pace`.

For example, with a 0.2 fs PMF timestep:

```toml
[rate.pmf]
method = "metadynamics"

[rate.pmf.sampling]
equilibration_steps = 50
production_steps = 1000
interval_steps = 5
replicas = 2

[rate.pmf.metadynamics]
mode = "well_tempered"
sigma = 0.05
height = "0.02 kcal/mol"
pace = 20
bias_factor = 10.0
freeze_after_steps = 200
freeze_equilibration_steps = 100

[rate.pmf.analysis]
bins = 40
ranges = [[-0.15, 1.15]]
bootstrap = 0
```

The initial equilibration is separate. `production_steps` includes all three
subsequent phases:

| Phase | PMF steps | Duration in this example | Used in the reweighted PMF? |
| --- | ---: | ---: | --- |
| Initial equilibration | 50 | 10 fs | No |
| Learn the bias | 200 | 40 fs | No |
| Relax with the final fixed bias | 100 | 20 fs | No |
| Fixed-bias production | 700 | 140 fs | Yes |

Saving every 5 steps gives 140 retained samples per independent history.
Only samples strictly after the freeze-plus-relaxation boundary are retained.
The observation on the final bias-update step is excluded, even if the
additional relaxation is zero. At least two retained samples per history are
required to run the estimator; that is an input check, not a precision target.

For OPES, select `method="opes"` and replace the MetaD table with:

```toml
[rate.pmf.opes]
sigma = 0.05
barrier = "0.5 kcal/mol"
bias_factor = 10.0
pace = 20
freeze_after_steps = 200
freeze_equilibration_steps = 100
```

The complete files also specify the PMF timestep, the root recrossing
settings, system and potential. Remove the inactive method's parameter table
when switching methods; mismatched tables are errors.

### What is being reweighted?

The actual sampling forces include the RPMDrate flux-metric bias. During
fixed-bias production the historical bias is removed with weights
`exp(beta * V_frozen)`, while the rate-flux measure is retained. This is a
stationary reweighting protocol; biased trajectory times are not used as
physical reaction times. MetaD static walls and shared-walker histories are
not supported by this rate protocol.

For these two estimators, PMF `bootstrap` must be zero. Independent-history
spread and weight-only effective sample sizes are diagnostics, not calibrated
standard errors. PMF-barrier uncertainty and complete rate uncertainty are
reported unavailable; recrossing-only uncertainty, if available, holds the
PMF factor fixed. Increasing the number of histories does not change that
uncertainty contract.

`analysis.ranges` defines histogram support and does not apply a confining
wall. RPMDrate xi can be singular outside its reaction corridor, so global
exploration and the stationary sampling domain need system-specific checks.

## Reuse an external PMF

```toml
[rate.pmf]
method = "external"
directory = "../compatible-fes"
allow_non_umbrella = false
```

The directory is relative to the input file. It must contain the saved
`FESResult` and its input/system/manifest records. The source must use the
same temperature, beads, PES, system/reference geometry and one-dimensional
centroid RPMDrate coordinate, with hartree energies and `rate_flux` measure.
Non-umbrella sources require `allow_non_umbrella=true` and valid reweighting
provenance. Setting that flag does not perform reweighting.

Do not add an estimator or local PMF sampling/dynamics/analysis table to an
external source. Parent and child trajectories still run. Pointwise source
errors do not provide reference/barrier covariance, so the existing external
source uncertainty limits remain in effect.

## Existing inputs and continuation

Inputs without `[rate.pmf]` keep the previous umbrella workflow, defaults and
checkpoint identity. Existing `rate.pmf_method` and `rate.pmf_source` remain
supported. Choose one notation; combining those legacy selectors with the new
table is rejected. Likewise, do not specify the same method-parameter table
both at top level and inside `[rate.pmf]`.

Resume continues the saved protocol and its clocks, bias history and
thermostat state. Changing method, timestep, thermostat, reaction coordinate,
or the freezing schedule changes the scientific calculation and requires a
new run. Ordinary resume and increasing the production budget are distinct
operations; rate production extension has its own method restrictions.

Use `validate --check-pes` before `run`, and inspect the resolved PMF-stage
settings and the reported support/uncertainty diagnostics before interpreting
a rate. A successful input check does not establish physical convergence.

The `pmf_plan` in `validate` and `prepare` output shows the selected sampler,
estimator, separate PMF/recrossing clocks, phase durations and retained sample
count. A new local PMF run saves its effective settings in
`RUN/input/pmf-resolved.json`; `RUN/rate_pmf.json` reports endpoint support,
the barrier and its uncertainty status alongside the plan.
