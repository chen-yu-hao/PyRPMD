# Precision-driven thermal RPMD rate control

The precision layer reads existing rate diagnostics, selects pilot work, freezes
independent production, and records decisions and execution receipts.
`pyrpmd precision plan`, `run` and `status` expose a complete production workflow
with enforced PES-frame budgets and recovery in the original job directories.
The Python controller remains available for custom executors and pilot planning;
the precision commands do not change `studies.py` scheduling.

## Scope and acceptance

Version 1 supports thermal gas-phase `task="rate"`, local fixed umbrella PMFs,
uniform sampling per window, and unthermostatted RPMD children. Both the legacy
rate input and resolved `rate.pmf_stage` input are supported. The dividing surface
must be fixed after the pilot, before production. External/shared PMFs, adaptive
biases during production, REUS, MetaD/OPES, TRPMD, surface/site/nonadiabatic rates,
and per-window sampling overrides are explicitly rejected by the job bridge.

`accepted=True` requires a production look that simultaneously has:

1. The exact persisted scientific identity, sampling design and job prefix.
2. Complete, independently initialized production sources, separate from pilot
   sources and other production epochs.
3. Supported PMF, connected overlap, adequate effective samples and independent
   parents, a resolved recrossing plateau and complete energy diagnostics.
4. A total-rate confidence interval within the requested relative half width.
5. Established equivalence for every predeclared numerical comparison.

No numerical comparisons requested means numerical convergence is unestablished;
an empty list does not certify convergence. Passing a requested pair does not
establish arbitrary bead/timestep limits, PES validity, or other untested physics.

| Status / condition | Meaning |
| --- | --- |
| `statistical_target_met` | A scheduled independent production interval meets its target. The separate `accepted` flag also requires numerical equivalence. |
| `budget_exhausted` | The campaign or per-job allowance cannot accommodate the next authorized work. |
| `diagnostic_failed` | A known PMF, overlap, effective-sample, plateau or energy prerequisite failed. |
| `error_unestimable` | Required uncertainty, covariance, degrees of freedom or diagnostics are unavailable. |
| `numerical_convergence_not_established` | At least one comparison is missing, invalid, too uncertain, or outside its equivalence margin. |
| `check_not_due` | The replica count is not a predeclared look; no inferential interval is exposed. |
| `production_design_invalid` | Production provenance, epoch, sampling or random-stream identity does not match its freeze. |
| `check_plan_exhausted` | The final declared look has been used without acceptance. |
| `execution_failed` | A rate calculation failed before producing an authenticated complete replica; its consumed budget remains charged. |

`status` is a primary decision; `conditions` retains simultaneous failures.
`statistical_target_met`, `numerical_convergence_established`, `diagnostics_passed`
and `accepted` are separate fields. Budget stops preserve these conclusions.

## Command-line production workflow

The CLI starts from an existing thermal rate input and a **declared, fixed
production design**. Complete pilot tuning and choose the geometry, local
umbrella windows and dividing surface first. `plan` resolves the input, validates
every comparison and freezes all job IDs and random seeds. It reserves the first
look without evaluating a PES. `run` executes that look, records its complete
replicas, checks the original acceptance criteria, and continues to the next
declared look only when the controller permits it.

### Prepare the rate input

The metered executor currently requires explicit Python propagation and one
serial worker. In the existing rate input, use these execution settings:

```toml
[dynamics]
loop_backend = "python"

[parallel]
backend = "serial"
mode = "serial"
trajectory_workers = 1
bead_workers = 1
trajectory_batch_size = 1
```

Keep `recrossing.child_batch_pairs = 1` if that option is present. For a scoped
`rate.pmf` input, its resolved PMF stage must also have
`dynamics.loop_backend = "python"` and serial settings; set
`[rate.pmf.dynamics] loop_backend = "python"` when overriding that stage.
These commands reject native/auto loop backends, process/MPI execution,
trajectory batching and child-pair batching instead of changing the requested
execution route. The original scientific settings, precision and provider
device remain unchanged.

Production also requires a supplied, fixed geometry and an explicit reaction.
Inputs that still require numerical TS validation/refinement or endpoint
preparation are rejected during planning, before a PES is constructed. Perform
that preparation separately, for example with `pyrpmd prepare INPUT --pilot`,
inspect its diagnostics, and use the chosen geometry in the production input.
For a schema-2 input with an already prepared TS, the relevant controls are:

```toml
[reaction]
definition = "explicit"
# Keep the chosen fragments, forming_bonds, breaking_bonds and rinf here.

[preparation]
ts_action = "none"
check_channel = false
```

`ts_action = "none"` records that production will not repeat saddle/channel
checks; it does not certify that the supplied geometry passed them. Do not
include unresolved reactant/product endpoints in this production input. The
usual fixed-dividing-surface and local-PMF requirements also apply.

### Declare precision, sampling and budgets

Save a JSON specification such as the following as `precision.json`. This
example assumes that the base rate input has **16 beads and a 0.1 fs timestep**;
the comparisons request 32 beads and 0.05 fs, respectively. The sampling counts
and tolerances are illustrative, not a convergence recommendation for a
particular reaction.

```json
{
  "format": "PyRPMD.precision.request.v1",
  "target": {
    "relative_half_width": 0.20,
    "confidence": 0.95,
    "minimum_umbrella_ess": 20.0,
    "minimum_parent_chains": 2,
    "minimum_effective_parent_blocks": 2.0,
    "maximum_energy_error_kbt": 0.1
  },
  "checks": {"replicas": [2, 4], "max_epochs": 1},
  "sampling": {
    "umbrella_streams": 1,
    "umbrella_steps": 1000,
    "umbrella_burnin_steps": 100,
    "parent_chains": 2,
    "parent_samples": 5,
    "parent_burnin_steps": 100,
    "parent_interval_steps": 20,
    "child_pairs": 2,
    "child_steps": 200,
    "plateau_fraction": 0.25
  },
  "numerical": [
    {"axis": "beads", "value": 32, "relative_margin": 0.05},
    {"axis": "timestep", "value": "0.05 fs", "relative_margin": 0.05}
  ],
  "max_cost": 120000000,
  "job_max_cost": 10000000,
  "cost_unit": "force_evaluations",
  "seed": 4812
}
```

JSON and TOML specifications have the same field names. Unknown fields,
nonfinite values and invalid integer counts fail explicitly. Numeric timestep
comparison values are in atomic time units; a unit-bearing string uses the
normal unit parser. `umbrella_streams` counts independent streams **per window**,
and each production replica is a complete independent PMF plus recrossing job.
The CLI currently requires `checks.max_epochs = 1`; the lower-level Python API
retains its predeclared multiple-epoch interface.

`max_cost` is the campaign budget and `job_max_cost` is the hard cap for each
complete rate job, including each numerical-comparison job. Both are integers.
Counts above `2^53 - 1` are rejected to preserve exact integer ledger accounting.
The per-job cap is an explicit authorization, not an automatic prediction of
the required work. Choose it to cover the declared sampling, energy checks and
any recovery recomputation. The finer-timestep and larger-bead jobs may require
more frames than the baseline.

At a look with `r` new replicas and `j` comparisons, the CLI reserves
`r * (1 + j) * job_max_cost` before launching any of that increment. The example
therefore reserves 60,000,000 frames for its first six jobs. A later increment
uses only the newly required replica suffix. When an increment finishes, its
recorded charges replace the reservation and the unused allowance becomes
available again. If the next complete increment cannot fit, planning records
`budget_exhausted` even if a smaller, incomplete selection would fit. It never
changes the declared look or selects only the successful replicas.

### Plan, pause, resume and inspect

```bash
pyrpmd precision plan rate.inp --spec precision.json --output precision-study
pyrpmd precision status precision-study
pyrpmd precision run precision-study --max-jobs 1
pyrpmd precision status precision-study
pyrpmd precision run precision-study
```

All three commands print JSON. `--max-jobs 1` pauses after one newly completed
job, leaving the current increment reserved; it is useful for controlled
handoffs and does not change the frozen statistical design. Resume with the
same `run` command. If a keyboard interrupt occurs during `run`, the command
prints `status="interrupted"` and returns exit code 130. Ordinary input or
execution-interface errors return code 2. A recorded scientific or budget stop
is a valid JSON result, so exit code 0 alone does not mean that the rate was
accepted. Inspect `stop` and `latest_assessment.accepted`.

`status` reads the ledger and job journals without launching dynamics or making
a new plan. Its budget fields distinguish `spent_cost` already settled in
terminal increment receipts, `unsettled_cost` charged by the current increment,
and their sum `charged_cost`. `reserved_cost` includes the full outstanding
increment allowance; do not add it to `charged_cost` to estimate consumed work.
`available_cost` is the unreserved balance. Each job reports its state,
directory, frame charges and number of admitted evaluation batches.

An identical `plan` replay reuses the campaign. Changes to the frozen input,
sampling, checks, comparison settings, seed or budget are rejected. `run` also
checks the saved explicit input/model hashes. It recovers the saved design and
execution identity rather than resolving a modified input file into a new run.
Replaying a terminal campaign returns its existing conclusion without starting
another calculation.

## Modules

| Module | Responsibility |
| --- | --- |
| `pyrpmd.precision_artifacts` | Read and validate the active published generation, source hashes, PMF, overlap/ESS, parent/child/plateau and energy evidence. |
| `pyrpmd.precision_statistics` | Covariance-aware uncertainty propagation, physical-replica statistics, adjusted intervals and numerical equivalence. |
| `pyrpmd.precision` | Typed targets, identities, sampling/check designs, frozen production, replica evidence and pure acceptance. |
| `pyrpmd.precision_planning` | Pilot supplement selection, frozen production increments, durable checks and stopping. |
| `pyrpmd.precision_jobs` | Materialize concrete existing-workflow configurations and authenticate completed production artifacts. |
| `pyrpmd.precision_ledger` | Reservations, execution identities, actual costs, immutable results, check history and crash recovery. |
| `pyrpmd.precision_execution` | Durable PES-frame admissions, per-job receipts and the cross-platform campaign execution lock. |
| `pyrpmd.workflows.precision` | CLI registration, fixed production planning, metered rate execution, reconciliation and status. |

## Read existing calculations and choose pilot work

```python
from pyrpmd.precision import (
    CheckSchedule, NumericalRequirement, PrecisionSpec, PrecisionTarget,
    SamplingDesign, evidence_from_diagnostics,
)
from pyrpmd.precision_artifacts import read_rate_diagnostics
from pyrpmd.precision_jobs import identity_from_config
from pyrpmd.precision_planning import PrecisionController

# config and system are an existing resolved thermal rate configuration/System.
identity = identity_from_config(config, system)
spec = PrecisionSpec(
    identity=identity,
    target=PrecisionTarget(relative_half_width=0.10, confidence=0.95),
    checks=CheckSchedule((4, 8, 16), max_epochs=2),
    numerical=(
        NumericalRequirement("beads", identity.beads * 2, relative_margin=0.05),
        NumericalRequirement("timestep", identity.timestep_au / 2,
                             relative_margin=0.05),
    ),
    max_cost=100_000_000,
    cost_unit="force_evaluations",
    seed=4812,
)
sampling = SamplingDesign(
    umbrella_streams=4,        # independent replicas PER WINDOW
    umbrella_steps=20_000,
    umbrella_burnin_steps=5_000,
    parent_chains=8,
    parent_samples=50,
    parent_burnin_steps=5_000,
    parent_interval_steps=500,
    child_pairs=16,
    child_steps=2_000,
    plateau_fraction=0.25,
)
controller = PrecisionController("precision-study", spec)
snapshot = read_rate_diagnostics("existing-pilot-run")
pilot = evidence_from_diagnostics(snapshot, identity)

# Quotes come from the executor in the SAME cost unit, including recomputation.
plan = controller.plan_pilot(pilot, sampling, costs=execution_quotes)
```

The legacy reader keeps absent errors as `None`. It preserves the original
single-chain conditional uncertainty and PMF rejected-bootstrap-draw metadata.
It does not promote a recrossing-only error, a supported-draw conditional error,
or an unverified PMF/parent independence assumption into total-rate evidence.
Published extension receipts are authenticated; pending generations are ignored.
Copies and overlapping generations of one run cannot count as independent runs.

Pilot planning returns one justified action:

| Limitation | Action | Independence rule |
| --- | --- | --- |
| PMF support/overlap/ESS or dominant PMF variation | `add_umbrella_streams` | Add independent streams per fixed window; changing the bias requires a new identity. |
| Too few independent parent chains/blocks or dominant parent variation | `add_parent_chains` | Add actual independent chains, regardless of children per parent. |
| Validated conditional-child variance dominates | `add_children` | Retain the same parent clusters; `new_independent_parent_chains=0`. |
| Resolved tail time dependence or insufficient tail samples | `extend_child_horizon` | Replace the shorter estimate; overlapping trajectory prefixes are not new samples. |
| Energy integration failure | `compare_timestep` | Use a declared finer step with the same physical time durations. |
| Numerical equivalence remains unestablished | `compare_beads` / `compare_timestep` | Separate identity and complete controlled comparison. |

A failed plateau with unresolved noise does not automatically mean a short child
horizon. Without validated parent/child variance decomposition, the planner
cannot justify a children-only remedy. Missing errors alone do not identify a
sampling remedy and return `error_unestimable`. Legacy summaries usually lack
that variance decomposition; a producer must explicitly supply it.

Pilot actions carry target counts and scientific semantics for an executor to dispatch.
They do not themselves implement checkpoint continuation of existing child
trajectories. The CLI executes fresh, frozen production; it does not dispatch
these adaptive pilot supplements. Existing rate extension publication remains
unchanged.

## Python API: freeze and hand off production

```python
from pyrpmd.precision_jobs import materialize_rate_job, record_from_rate_job
from pyrpmd.precision import RateEvidence, evidence_from_replicas

frozen = controller.freeze(
    sampling,
    epoch_index=0,
    pilot_source_ids=pilot.source_ids,
    pilot_source_family_ids=pilot.source_family_ids,
)
plan = controller.plan_production(
    frozen, through_replicas=4, estimated_cost=first_increment_quote,
)
```

The freeze fixes all complete-rate replicas, numerical comparison jobs, seeds,
sampling counts, plateau choice and look sizes before production starts. An executor calls
`materialize_rate_job(config, system, frozen, job_id)` to obtain the concrete
config, expected manifest/model hashes and provenance. Its `config` is ready for
the existing `runner.run_resolved`; `run_kwargs={"resume": False}` describes the
first launch in a fresh output directory. Ordinary recovery may continue that
same registered job; it must not create a new independent sample.

For timestep comparisons, frozen per-job sampling counts preserve umbrella,
burn-in, parent-spacing and child-horizon physical durations. The bridge also
preserves save, thermostat and checkpoint periods. Nonintegral time conversions
are rejected instead of rounding to a confounded comparison. Beads, timestep or
bias changes produce distinct scientific identities and are never pooled as
additional samples of the baseline.

The execution-layer lifecycle is:

```python
# Persist BEFORE any launch. Reuse this execution identity when reconciling a crash.
controller.ledger.start(plan["plan_id"], execution_id=execution_id)

# Inside a custom executor, for every planned job:
job = materialize_rate_job(config, system, frozen, planned_job["job_id"])
# Execute job["config"] under plan["execution_budget"] in a fresh destination.
record = record_from_rate_job(completed_directory, job, frozen)

# After every job in the increment has completed, preserve all records.
portable_records = [
    {**r, "diagnostics": r["diagnostics"].to_dict()} for r in increment_records
]
controller.ledger.complete(
    plan["plan_id"], execution_id=execution_id,
    actual_cost=measured_actual_cost,
    result_ids=[r["source_id"] for r in increment_records],
    result={"job_ids": [r["job_id"] for r in increment_records],
            "records": portable_records},
)

# At each look, include the complete prefix from ALL completed increments.
records = []
for entry in controller.ledger.snapshot()["plans"].values():
    if (entry["state"] == "completed"
            and entry["plan"].get("production_design_id") == frozen.design_id):
        for saved in entry["receipt"]["result"]["records"]:
            records.append({**saved, "diagnostics": RateEvidence.from_dict(saved["diagnostics"])})
evidence = evidence_from_replicas(spec, frozen, records)
decision = controller.check(evidence, frozen=frozen)
```

These snippets show the custom-executor contract; scheduler variables and
resource measurement must come from that executor. Calling the planning API
does not launch dynamics. The CLI implements this lifecycle for its supported
serial Python route.
The next `through_replicas=8` plan dispatches only replicas 4–7, including their
declared numerical counterparts. Production never follows adaptive pilot
allocation choices in place. Revising that design uses a fresh independent
epoch and its preallocated portion of the same global error budget.

## Statistical model and repeated checks

For existing component errors, the first-order total log-rate variance is

\[
s^2_{\log k}=s^2_{\log k_{\mathrm{QTST}}}
             +s^2_{\log\kappa}
             +2\operatorname{Cov}(\log k_{\mathrm{QTST}},\log\kappa).
\]

Unknown covariance is not zero. Independence must be explicitly established, or
paired physical replicas/joint resamples must retain the dependence. PMF barrier
propagation similarly retains the covariance of the reference and barrier bins.
Joint-bootstrap standard deviations are not divided by the number of bootstrap
draws, and bootstrap count is never a physical degrees-of-freedom count.

The complete-production path estimates the **arithmetic mean** of the independent
replica rates `k_i = q_i * c_i`. It uses the covariance of paired product
fluctuations, not a geometric mean of rates. With `q_bar`, `c_bar` and `k_bar`,
the scaled components

\[
A_i=\frac{(q_i-\bar q)(c_i+\bar c)}{2\bar k},\qquad
B_i=\frac{(c_i-\bar c)(q_i+\bar q)}{2\bar k}
\]

sum to the rate fluctuation up to a constant. Their sample covariance of means
therefore reconstructs `Var(mean(k))/mean(k)^2`, including PMF/recrossing
dependence. This descriptive split is not a unique causal variance attribution
and does not provide a parent/child decomposition.

With `E` declared epochs, `L` looks and `J` numerical comparisons, each interval
uses `alpha_claim = (1-confidence) / (E*L*(1+J))`. This finite Bonferroni family
does not require independent looks. Restarting production never resets alpha.
The fixed-look interval uses the physical replica degrees of freedom and a
Student-t/log-delta approximation; it is explicitly labeled approximate. The
replicate estimator may retain finite-sampling or equilibration bias. Increasing
replicas does not validate that bias, the PMF model or plateau physics.

The relative precision criterion is the larger of the upper/lower interval
distances from the positive point estimate, divided by that estimate. For a
numerical margin `r`, the entire adjusted log-rate-ratio interval must lie
strictly within `[-log(1+r), log(1+r)]`. A zero estimated difference with a wide
interval fails. This uses a conservative two-sided interval for equivalence.

## Recovery, budgets and failures

### CLI execution accounting

The only CLI cost unit is currently `force_evaluations`, defined as **PES
configuration frames admitted to the public evaluation contract**. A call
`evaluate(x)` with shape `[B,N,3]` costs `B` units, including calls that obtain
energies for diagnostics. This is a count of frames, with no bead scaling. It
does not measure elapsed time, GPU work, SCF iterations or additional
evaluations performed inside a composite or external provider.

Before every supported provider call, the executor checks whether the entire
batch fits its job cap and atomically publishes the increased charge. Only then
does it call the provider. A batch that cannot fit is rejected without calling
the PES or splitting the batch, and the campaign records `budget_exhausted`.
Failed provider calls remain charged because they were admitted. No subsequent
job can spend beyond the reserved increment caps.
A per-job cap can stop the campaign while global balance remains; another
job's unused allowance does not increase an already frozen per-job cap.

The charge and the provider call cannot be committed as one atomic operation.
If a process dies after publishing a charge but before making its call, recovery
retains that charge conservatively. Thus the durable count bounds admitted
work; it is not an exact count of successfully returned frames in every crash
scenario. Recomputed steps after a checkpoint rollback are new computations
and consume further allowance. Replaying an already completed job receipt or
increment receipt consumes no additional allowance and creates no new sample.

The meter is installed only inside the precision executor through an optional
provider-lease scope. Ordinary workflows retain their provider identity,
native bindings, reuse and ownership behavior. Fused native propagation would
bypass `evaluate`, and the scope does not propagate into subprocesses; this is
why the initial CLI requires explicit serial Python loops and rejects
unmetered execution combinations. Split/RPC provider entry points are not
exposed through the metered wrapper. Publishing every admission also incurs
filesystem synchronization overhead, so this executor should not be presented
as a native-performance path.

### Durable files and reconciliation

All paths below are relative to the campaign directory:

| Path | Purpose |
| --- | --- |
| `precision/workflow.json` | Frozen resolved input, system, scientific specification, production jobs, explicit file hashes and execution caps. |
| `precision/ledger.json` | Increment reservations, execution IDs, immutable receipts, checks and stop conclusions. |
| `precision/jobs/<job_id>.json` | Per-job durable charges and authenticated completion or failure record. |
| `jobs/<job_id>/` | Standard rate manifest, input, PMF, parent/child checkpoints and results for that one frozen job. |

The campaign execution lock serializes launches and recovery across processes.
It is separate from the ledger's short transaction lock, so `status` need not
wait for dynamics to finish. Both locks use `flock` on POSIX and
`msvcrt.locking` on Windows; process exit releases them. Checksums, atomic
replacement and file fsync protect published metadata. The ledger and execution
journals additionally fsync their directories on POSIX, an operation Windows
does not support. These
contracts require local-filesystem lock and atomic-replacement semantics.

On `run`, existing work is reconciled before new work is admitted:

1. A completed job receipt is authenticated against its saved rate artifacts
   and reused without dynamics.
2. Published rate output without its job receipt is authenticated and the
   receipt is recovered, without rerunning the completed calculation.
3. An incomplete job resumes in its original directory with its original ID,
   seed and checkpoints. Initial metadata is staged before dynamics can start.
4. After every job in an increment completes, one immutable ledger receipt
   settles its total charges and claims its result IDs.
5. A crash after that receipt but before assessment resumes the same scheduled
   check. A crash between check and stop publication recovers the same stop.

Umbrella and parent checkpoints retain coordinates, momenta, RNG state and
sampling clocks. Children currently commit a checkpoint only after a complete
positive/negative trajectory pair. An interruption during a child pair may
therefore recompute that pair from its deterministic initial state; those
additional PES admissions remain charged, while the pair and production
replica are counted only once. Use `precision run CAMPAIGN` for campaign
recovery, so metering and receipts remain attached to the original execution.
Directories without a matching execution identity are not adopted as fresh
independent production.

### Lower-level ledger contract

Reopening the Python controller must use the same complete specification and
maximum budget. `controller.restore_frozen(epoch_index)` recovers its exact
persisted design. The CLI performs that recovery automatically for its one
frozen epoch.

`plan_id` and `execution_id` are durable idempotency keys. Replaying an identical
plan, receipt, check or stop is a no-op; conflicting payloads and duplicate
result IDs fail. A persisted `running` entry requires an executor to reconcile
its actual job/checkpoint state. The controller alone never guesses that it
should rerun work.
An interruption between check publication and stop publication is repaired on
replay. A later terminal stop is not overwritten by replaying an older check.

Quotes reserve budget; receipts charge actual cost once and release the
reservation. Reserved work is not prematurely stopped just because unreserved
balance is zero. Every plan carries `execution_budget`, which its executor must
enforce. The CLI enforces the frame caps described above; a custom executor
must implement its own measurement and admission rules in the declared cost
unit. The ledger itself does not meter another process's work. If a custom
executor reports an actual overrun, the ledger records it truthfully and blocks
further reservations.

Use `ledger.fail(..., actual_cost=..., reason=...)` only for a reconciled terminal
failure. Unknown-cost interrupted work stays running/reserved until its executor
reconciles it. The CLI retains recoverable interruptions and filesystem errors
in the same pending execution; diagnosed budget or calculation failures record
a terminal receipt and stop. Failed partial artifacts remain labeled failed and
cannot enter acceptance.
A failed replica is never dropped to narrow a confidence interval. A retry after
a terminal failure needs a new registered execution plan or independent epoch;
ordinary interrupted-job resume retains its existing identity. The CLI never
increases a cap, changes a freeze or clears a stop in place. Repeated campaigns
selected after seeing their results must not be treated as one prespecified
experiment; the multiple-epoch Python interface exists to allocate that error
budget in advance.

## Verification and references

Run the acceptance suite with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_precision_*.py
PYTHONPATH=src python examples/precision_control.py
```

The tests include PMF-, parent-, child-variance- and horizon-limited cases;
non-equivalence despite an insignificant difference; covariance and bootstrap
handling; reserved/spent budget boundaries; interrupted and replayed decisions;
source-family deduplication; frozen clocks; and actual small legacy/scoped rate
jobs imported through the existing runner. The CLI/execution tests additionally
exercise planning without PES calls, real rate continuation after a durable
checkpoint, direct-run trajectory/RNG comparisons, batch admission limits,
receipt recovery and repeat execution without duplicate charges or results.
They also run the nonzero analytic teaching exchange PES and verify that an
interrupted child pair adds recomputation charges without adding a sample.
These are control, execution and estimator regressions. A short test run that
stops with failed diagnostics does not establish physical convergence of a
molecular reaction rate.

Method references:

- [NIST: Bonferroni simultaneous intervals](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm).
- [NIST TN 1297: covariance-aware first-order uncertainty propagation](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-appendix-law-propagation-uncertainty).
- [SciPy: independent observations and physical t-test degrees of freedom](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html).
- [SciPy: paired bootstrap and approximate coverage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
- [statsmodels: equivalence testing and its nonequivalence null](https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ttost_ind.html).
