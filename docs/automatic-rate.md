# Automatic AUS rate calculations

Set `rate.auto = true` to let the gas-phase rate workflow choose and refine its
sampling settings. The controller uses adaptive umbrella sampling (AUS), then
the existing Bennett–Chandler constrained-parent/paired-child calculation. It
uses histogram MBAR by default, so non-Gaussian windows do not require a
Gaussian UI model. An explicitly selected UI estimator remains supported. It
selects the dividing surface from the supported PMF maximum unless an explicit
surface was supplied.

The [automatic example](../examples/rate/automatic.inp) needs a temperature,
transition-state geometry and potential. It supplies the bead count explicitly:

```toml
schema_version = 2
task = "rate"
temperature = "300 K"
beads = 32

[system]
transition_state = "ts.xyz"

[potential]
provider = "python"
factory = "pes.py:build"

[rate]
auto = true
```

With schema 2 and no reaction definition, automatic mode requests reaction
inference and channel checking. A supplied reaction definition, entrance,
electronic weight, isotope masses, bead count, integrator and timestep keep
their usual meanings. Ambiguous reaction inference or an invalid saddle stops
the calculation. The checked preparation result is saved as usual.

```sh
pyrpmd validate automatic.inp --write-resolved resolved.json
pyrpmd run automatic.inp --output runs/automatic
pyrpmd resume runs/automatic
```

Static validation resolves the defaults without evaluating the PES. Use the
ordinary `prepare --pilot` or `validate --check-pes` interface to inspect
numerical preparation. The bundled example is an analytic teaching potential,
not a physical benchmark.

The [BKMP2 example](../examples/rate/bkmp2_auto.inp) exercises the same automatic
workflow for H + H2 at 1500 K and P=16 with a 0.05 fs step. It requires the
locally built, provenance-checked BKMP2 library described in `benchmarks/h3`.
Use a local filesystem for the calculation and archive the results afterwards
if the shared filesystem lacks HDF5 file locks.

## Precision and budgets

The only optional controller settings are the target relative standard error
and maximum number of refinement rounds. Use a table in place of `auto = true`:

```toml
[rate.auto]
target_relative_error = 0.10
max_rounds = 8
```

The target applies to statistical uncertainty in the final rate, including PMF
barrier and transmission-coefficient uncertainty. `max_rounds` bounds the
production refinement and recrossing refinement loops separately. AUS pilot
adaptation retains its own existing `adaptive_umbrella.max_rounds` budget
(default 8), window cap and pilot-duration cap. Budgets are finite; a stricter
error target can exhaust them without yielding an accepted rate.

Existing sampling settings are valid starting points. For example, an explicit
`sampling.production`, `sampling.replicas`, `recrossing.child_time` or
operation-named stage setting is preserved as the initial budget. Automatic
duration defaults are rounded to the nearest positive number of steps on the
corresponding stage clock; explicitly supplied durations still require an exact
integer number of steps. The mode requires at least two PMF replicas, two
independent parent chains and 100
bootstrap resamples; omitted values use two replicas, four parent chains and
200 resamples. It requires locally sampled umbrella PMFs and enabled AUS
spring adaptation. An external PMF, a different enhanced-sampling method, or
disabled statistical diagnostics is rejected before dynamics starts. Automatic
recrossing uses RPMD; TRPMD requires the manual workflow because its thermostat
does not satisfy this controller's conservative-child energy criterion.

`auto = false`, or `[rate.auto] enabled = false`, leaves ordinary manual rate
sampling in control.

## What is adjusted

The starting search interval is `xi = [0, 1.05]`. Initial umbrella centers and
histogram bounds extend beyond this interval so that the reference and search
endpoints lie inside sampled histogram centers. The default initial spacing
is approximately 0.05 and each window starts with the physical spring
`k = k_B T / 0.05^2`. A user-provided center plan or spring replaces that start.
The controller then uses observed overlap, effective samples and coordinate
coverage to refine the windows. Each window adapts its own spring from its
mean displacement and covariance. In one dimension, the local curvature
estimate is `F'' = 1/(beta * variance) - K`. A steep slope or unstable broad
distribution calls for a stronger restraint; a narrow distribution in a flat
region permits a softer spring. A window with inadequate effective samples
waits for a longer pilot. New windows inherit a positive interpolation of
neighboring springs. Each update is bounded between 0.5 and 2, with total
spring scales between 1/64 and 256 of their initial values. All decisions and
physical matrices are saved in the version-4 adaptive plan. Older saved plans
retain their original shared-spring policy. Changed springs require another
independent pilot before production; pilot samples never enter the PMF.

The default pilot lasts 200 fs after 100 fs of equilibration. Formal production
starts with two independent replicas per frozen window, 100 fs of
equilibration and 1000 fs of production, sampled every 1 fs. Pilot samples do
not enter the production PMF. Windows and springs are frozen before production;
insufficient production support or PMF precision causes additional sampling.
When correlated blocks are insufficient, auto records the PMF point estimate
and defers the expensive bootstrap until its existing block-support criterion
is met. Deferred uncertainty remains unavailable and cannot authorize recrossing.
The decision uses production diagnostics, not the success of the pilot. If
production exposes inadequate spatial coverage or overlap, the controller
refines the umbrella plan in a new independent generation; it discards the
previous generation from the final production estimate instead of mixing
different adaptive plans.
This refinement also checks actual unsupported PMF bins and endpoint brackets,
even when a coarser coverage grid passes. The automatic Andersen refresh
interval defaults to 20 fs in physical time; an explicit interval takes priority.

Recrossing starts with four independent parent chains, 16 configurations per
chain, two positive/negative momentum pairs per configuration and 50 fs child
trajectories. A drifting tail increases the child propagation time. A noisy
but approximately flat tail increases statistical sampling. Parent correlation
and the relative contributions of parent and child noise guide changes to
parent spacing, parent counts and child-pair counts. This is a diagnostic and
cost-based tuning strategy, not a guarantee of a globally optimal allocation.

Each recrossing round has its own settings, checkpoints and report under
`auto-rate/recrossing/`. Longer children are recalculated in a fresh round
because the existing child cache does not store a propagatable terminal state.
Related rounds are refinements of one estimate and are never pooled as
independent observations. Resume reuses completed work with the saved settings.

## Interpreting the result

`sampling_converged` means that the automatic statistical criteria were met:
supported PMF sampling and uncertainty, independent parent support, acceptable
child energy conservation, a statistically stable transmission plateau and
the requested total uncertainty. It does not establish convergence in bead
count, timestep, reactant separation, the chosen PMF estimator or the PES.
Those remain physical/model convergence studies, and the ordinary physical
`converged` flag is not promoted by sampling automation.

If a budget is exhausted or a scientific gate requires intervention, the
automatic result reports an incomplete status and null formal rate values.
Diagnostic estimates and the reasons for the next decision remain available
for inspection. Dynamics failures, non-finite values and inconsistent reaction
coordinates are not repaired merely by adding samples.
