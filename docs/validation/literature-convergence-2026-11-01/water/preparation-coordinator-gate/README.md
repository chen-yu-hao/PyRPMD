# All-eight preparation coordinator: unlaunched review contract

**This is an orchestration implementation and fixture gate, not an adopted
continuation experiment.** No live 100 ps checkpoint selection, registration,
preparation-extension configuration, or long extension run was created.
The original fixed classical cohort continues under its existing frozen scheduler.

The new coordinator closes the cohort-level gap left intentionally outside the
[single-chain preparation runner](../preparation-continuation-gate/README.md).
It is a separate file; the active scheduler, native integrator, original
configuration, random-number phase and proposal files remain unchanged.

## What must be established before registration

All eight original seed runs must be complete, including their original 25 NVE
segments. A checksum-bound complete-cohort assessment must identify exactly those
eight original manifests. An explicit later decision record must adopt **only**
the fixed additional 400 ps of preparation for every seed. The coordinator does
not infer this decision from a diffusion value, a paper comparison, or a pass flag.

The old coordinator must also have actually finished. A bound normal-exit proof
records its observed process exit code 0, a drained-pool declaration, the old
registration hash, the complete journal hash and its final successful
`coordinator_finished` event. Registration then acquires **all eleven original
leases**: the coordinator, both legacy queues and all eight seed leases. This
proves their release; a last-written complete manifest alone is insufficient.
The files and prior histories remain unchanged while these existing advisory
leases are held.

Only after those conditions hold does registration select each exact original
step-400000 checkpoint, before an NVE draw (`active_segment=-1`). It binds the
whole checkpoint, original manifest, physical configuration, source files,
environment, model and integrator hashes, plus the reviewed version 2 proposal.
The first two original runs' distinct checkpoint-migration histories remain
preserved. Only their physical model/integrator XML must be common to all seeds;
the historical receipts are not incorrectly required to be identical.

Registration validates every source before publishing any configuration or
registration file. It creates no OpenMM Context and consumes no saved RNG draw.
The templates here deliberately remain invalid for execution:

- [adoption-record.template.json](adoption-record.template.json) has a pending
  decision, unresolved assessment/exit hashes and no review timestamp.
- [original-exit-proof.template.json](original-exit-proof.template.json) has no
  asserted process exit or final journal event.

Neither template is evidence that the original cohort has already completed.

## Fixed execution and failure behavior

The only worker program is the already tested preparation-only continuation
runner. Every chain advances from absolute step 400000 to step 2000000: an
additional 400 ps, reaching 500 ps total at the unchanged 0.25 fs time step.
The first pending preparation Maxwell draw and all later refreshes use the
original RNG and absolute schedule. No NVE or diffusion-analysis command exists
in the coordinator, and the worker stops before a terminal NVE momentum draw.

The fixed queues remain `1101→1105`, `1102→1106`, `1103→1107`, and `1104→1108`.
The coordinator permits one, two or four integration workers, at most 1800 s
per worker invocation and at most 7200 s per outer invocation. Scheduling budget
changes do not change the physical budget, seed order or saved state.

During execution it holds both the original exclusion leases and the new
coordinator/eight seed leases. Each child inherits six descriptors: original
coordinator, old odd/even queues, original own-seed lease, new coordinator and new
own-seed lease. These continue to exclude another writer if the coordinator
unexpectedly disappears while a child is alive. The lease helper deliberately
does not issue an explicit unlock when closing a parent's descriptor.

Before each call, the coordinator checks immutable inputs and the previously
committed branch prefix. Afterward it verifies that the prefix was retained and
publishes a durable per-seed receipt. A missing or self-consistently replaced
earlier branch is rejected. A nonzero native exit, corrupt restart, provenance
failure or journal error stops further dispatch. Existing children finish
normally; their state and logs are retained. A durable failure latch blocks
automatic retries, seed substitution and history replacement.

## Signals and ordinary shutdown

**SIGINT/SIGTERM directed to the coordinator PID** request a scheduling stop.
The signal handler only records the request; the main loop persists a
`STOP_AFTER_CURRENT` marker and an event, stops admission and drains every
already-started child normally. It restores the prior handlers after the pool
has closed. The CLI returns 130 for SIGINT or 143 for SIGTERM after draining.
The stop marker remains for explicit later scheduling review.

The handler context also surrounds exceptional pool shutdown: an ordinary
main-thread exception latches a failure before the pool waits, and another signal
during that drain does not terminate the children. A child admission already
racing with a signal request is treated as in flight and is drained; no queue
successor is admitted after the stop is latched.

Process-group signals are outside this graceful-stop guarantee. They can reach
the frozen native runner directly, which retains its original signal handling.
Use the coordinator PID or the scheduling stop marker for an orderly pause;
the coordinator never sends a termination signal to an integration child.

## Focused validation and independent review

The final focused gate passes **35 tests in 3.86 s**, entirely using metadata and
dispatch fixtures. The earlier 22- and 32-case passing gates are retained to
show the subsequent review-driven additions; these are successive versions of
one test file and must not be added as independent test counts.

Coverage includes all-eight source selection, an incomplete/missing source,
exact pre-NVE phase, altered time step/model/source/proposal/configuration,
distinct historical migration receipts, normal-exit and original-lease release
requirements, fixed queues, inherited leases, restart-prefix retention, no-op
completion, durable failure, stop markers and post-spawn journal failure.
The orchestrator never creates a Context in any fixture.

Three isolated process fixtures send real coordinator-only SIGINT or SIGTERM
while a real bounded Python stub child is alive. They verify that the child
survives the signal, old/new leases remain held, no successor seed starts, the
child exits normally before the coordinator returns, and prior signal handlers
are restored. One sends SIGTERM during exceptional pool shutdown after an
injected main-thread error. These stub processes perform **no molecular dynamics**.

The separate nine real Reference/CPU phase tests in the preparation-runner gate
were not repeated for these orchestration changes. Short method tests and
metadata fixtures do not demonstrate liquid equilibrium or paper agreement.

Run the new gate from the repository root:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1 \
  python -m pytest -q tests/test_water_preparation_coordinator.py
```

## Reviewable future interface

After the complete original assessment and a separate preparation-only adoption,
the registration interface is:

```sh
python benchmarks/water/native_preparation_coordinator.py register \
  --original-root ORIGINAL_COHORT_ROOT \
  --output NEW_PREPARATION_COHORT_ROOT \
  --proposal docs/validation/literature-convergence-2026-11-01/water/equilibration-extension-proposal-v2.json \
  --adoption-record COMPLETED_PREPARATION_ONLY_ADOPTION.json
```

The execution interface is `run --registration NEW_PREPARATION_COHORT_ROOT/registration.json`
with explicit scheduling budgets. Registration and execution use separate
subcommands; neither is run by documenting them here.

The original U/T/OH/HOH thresholds remain unchanged. The proposed new RDF and
coordination hard gates remain unadopted; full diagnostics are retained for a
separate uncertainty assessment. They are explicitly not diffusion evidence.
Finishing this additional preparation could supply data for that later review;
it cannot automatically certify equilibrium, select a favorable time window,
authorize production or replace the original cohort.
