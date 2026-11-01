# Preregistered eight-chain paired time-step control

**Prepared and tested on small fixtures; the 64 ps live control has not been
launched.** Its fixed [protocol](protocol.json) was sealed at 19:02 UTC on
2026-11-01, before any control data or aggregate original-cohort diffusion
comparison. The protocol SHA256 is
`d7a34a9f92604b38ef0a4ab2bc6271b20bd14b09e92b95b4312bd2efe0543747`.

The control tests the production integration time step on all eight original
100 ps configurations. It cannot repair a preparation gate failure or establish
equilibrium, transport convergence, or agreement with the paper. The original
production trajectories and every original failed diagnostic remain preserved.

## Fixed experiment

The fixed seeds are 1101–1108. Registration requires every original run to be
complete, its complete-cohort assessment to be hash-bound, and the original
coordinator to have a bound normal-exit proof with its leases released. It
selects exactly the original step-400000 preparation checkpoint before the
segment-0 NVE momentum draw. There is no selection of production endpoints,
favorable seeds, time windows or partial diffusion results.

| Branch | Time step | NVE duration per seed | Integration steps per seed | Recording stride | Recording times |
| --- | ---: | ---: | ---: | ---: | --- |
| `dt025` | 0.25 fs | 4 ps | 16,000 | 20 steps | 0–4 ps, every 5 fs |
| `dt0125` | 0.125 fs | 4 ps | 32,000 | 40 steps | 0–4 ps, every 5 fs |

Each branch has exactly 801 coincident position/velocity observations, including
time zero. Across all eight pairs the fixed budget is 64 ps and 384,000
integration steps. Both branches retain 216 H2O molecules, the original masses,
298 K initial Maxwell distribution, 0.997 g/cm³ box, full 9 Å PME with tolerance
`1e-5`, no dispersion correction and no bond constraints. NVE contains no
thermostat and no further momentum draw.

Observations and restart states use float64 in both branches. This common storage
precision isolates the time-step change from a change in quantization. It is an
implementation choice recorded before the full control; the original
production's recording precision is unchanged.

## One physical starting state for both branches

First, the source checkpoint is loaded into a Context with its **original** time
step. The frozen version 2 checkpoint contract checks exact raw positions and
half-step velocities, independently bounded force-cache roundoff, and the
corresponding canonical-momentum conversion. The periodic cell and freshly
serialized model/integrator must match the source.

The saved external NumPy RNG is then restored. Exactly one pending segment-0
Maxwell draw is made, with total physical COM momentum removed using the same
mass-weighted projection as the original production. Its canonical coordinates
and momenta and the RNG states before/after the draw are saved once. Both branches
read that same checksum-bound artifact. A resumed branch never repeats the draw.

Each branch creates a fresh Context. The physical initial momentum is converted
to that branch's half-step velocity using
`v_half = p/m − dt×F/(2m)`. Recorded momenta use the inverse conversion. This
matches the time offset described by the
[OpenMM VerletIntegrator theory documentation](https://docs.openmm.org/latest/userguide/theory/04_integrators.html).
The binary source Context is not loaded into an integrator with a different
time step. The 0.25 fs branch is a separate fresh-context replay, with its own
saved provenance and CPU-roundoff qualification.

The energy diagnostic uses coincident canonical kinetic energy plus physical
potential energy. OpenMM's stored half-step velocities are not treated as
coincident velocities when forming this energy. Each branch retains the original
4 ps numerical limits: maximum sampled energy change of 0.01 kcal/mol/water and
absolute linear drift of 0.001 kcal/mol/water/ps.

## Statistical rule fixed before execution

For each seed, the primary endpoint is the molecular-COM Green–Kubo integral
through 2 ps, with the velocity-dot correlation integral divided by three.
The secondary endpoint is the raw unwrapped molecular-COM MSD slope over
1–2 ps divided by six. All valid time origins and molecules contribute to their
means, but they are never counted as independent uncertainty units.

For each endpoint, form eight differences
`delta_i = D_i(dt0125) − D_i(dt025)`. The reported mean is `sum(delta_i)/8`, with
standard error `sample_std(delta_i, ddof=1)/sqrt(8)`. The two-sided 95% interval is
the mean plus/minus `t.ppf(0.975,7) × standard_error`; see the
[official SciPy Student-t API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html).
This is nominal small-sample t inference conditional on the eight original
preparations and their shared starting history. It is not a distribution-free
coverage guarantee or an independent check of equilibrium sampling.

The entire interval must lie within **−0.01 to +0.01 Å²/ps** to establish the
declared numerical equivalence. A wide interval containing zero is unresolved.
An interval wholly outside the margin indicates a difference exceeding that
margin. The VACF and MSD intervals are reported separately; an overall numerical
control pass requires both interval tests and every branch's fixed energy limits.

All eight complete pairs are required before any aggregate transport comparison.
The analyzer also verifies each pair's original seed/checkpoint identity, so
copying a seed's output cannot create another independent pair. No paper
reference value appears in the decision rule. Regardless of the numerical
outcome, `physical_converged`, `literature_reproduced`, and
`preparation_gate_failure_repaired` remain false.

## Execution, restart and failure handling

The implemented execution is sequential, using one CPU force thread and the
fixed seed order, with `dt025` followed by `dt0125` for each seed. It holds the
original cohort's exclusion leases and a new control-writer lease. There is no
parallel launch or extra live worker in this review stage.

The default invocation budget is 1800 wall seconds, with a maximum of 7200.
Checkpoint boundaries occur every 1 ps. A bounded invocation can retain an
incomplete branch, which resumes from its exact branch Context plus redundant
raw state. The half-step/canonical conversion uses that branch's own time step.
Completed branches return without creating another engine. Off-record-grid
checkpoint restarts preserve the common observation sequence without duplicated
or missing frames.

A source/model/initial-state mismatch, nonfinite state, or engine exception
retains all committed data and latches a control failure. No later seed or branch
is started automatically. Failed streams are not replaced. Wall budgets and
committed restarts provide the normal pause/resume path; abrupt process failure
does not turn a partial cohort into a usable comparison.

The original measured CPU cost projects roughly 1,960 CPU seconds, or 33
CPU-minutes, for all 16 branches. This is a conditional estimate before execution;
shared-host allocation and startup/checkpoint work affect actual wall time.
The full control waits for the original eight-chain completion and a separate
resource assignment.

## Focused validation

The current focused gate passes **13 tests in 5.05 s**. The real-engine fixtures
use eight molecules and only 0.8 fs control traces at 0.1/0.05 fs time steps.
They validate phase/state transfer and restart implementation; they are not
diffusion measurements or substitutes for the planned full control.

Tests establish the exact saved-RNG COM Maxwell draw, its reuse without a second
draw, equal canonical initial states, common recording times, off-grid Reference
restart, actual CPU force-cache validation, completed no-op resume and rejection
of changed control identity. Excluded metadata/statistics fixtures check
all-eight completion, failure retention, duplicate-seed rejection, sample
variance with seven degrees of freedom, and the distinction between an interval
containing zero and equivalence. No full 216-molecule time-step control ran.

An independent read-only review of the final source found no blocking
phase/normalization issue. It confirmed the common canonical initial state,
time-step-specific half-step conversion, no-redraw restart, all-16 completion
guard and paired uncertainty unit. The reviewer did not run extra OpenMM/PES
calculations or tests. The source-bound receipt retains those review limits.

Run the focused test file from the repository root:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1 \
  python -m pytest -q tests/test_water_paired_timestep.py
```

## Reviewable future commands

Once the original cohort is complete and the control has its resource assignment,
registration is read-only with respect to the original experiment:

```sh
python benchmarks/water/paired_timestep.py register \
  --original-root ORIGINAL_COHORT_ROOT \
  --output NEW_CONTROL_ROOT \
  --protocol docs/validation/literature-convergence-2026-11-01/water/timestep-paired-control/protocol.json \
  --assessment COMPLETE_ORIGINAL_ANALYSIS/report.json \
  --exit-proof OBSERVED_ORIGINAL_NORMAL_EXIT.json
```

Execution uses `run --registration NEW_CONTROL_ROOT/registration.json --wall-seconds 1800`.
Repeating that command resumes the fixed remaining work. Only after every pair
is complete does `analyze --registration NEW_CONTROL_ROOT/registration.json --output NEW_ANALYSIS_DIRECTORY`
produce paired statistics and full control curves. These commands describe the
prepared interface; no live registration or 64 ps execution was issued here.
