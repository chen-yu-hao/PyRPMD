# Precision-driven thermal RPMD rate control

The precision layer reads existing rate diagnostics, selects pilot work, freezes
independent production, and records decisions and execution receipts. It is a
Python API for the execution layer and CLI to call. It does not add scheduling
to `studies.py` or register CLI commands.

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
| `budget_exhausted` | The spent budget is exhausted, or the next justified complete action cannot fit its remaining allowance. |
| `diagnostic_failed` | A known PMF, overlap, effective-sample, plateau or energy prerequisite failed. |
| `error_unestimable` | Required uncertainty, covariance, degrees of freedom or diagnostics are unavailable. |
| `numerical_convergence_not_established` | At least one comparison is missing, invalid, too uncertain, or outside its equivalence margin. |
| `check_not_due` | The replica count is not a predeclared look; no inferential interval is exposed. |
| `production_design_invalid` | Production provenance, epoch, sampling or random-stream identity does not match its freeze. |
| `check_plan_exhausted` | The final declared look has been used without acceptance. |

`status` is a primary decision; `conditions` retains simultaneous failures.
`statistical_target_met`, `numerical_convergence_established`, `diagnostics_passed`
and `accepted` are separate fields. Budget stops preserve these conclusions.

## Modules

| Module | Responsibility |
| --- | --- |
| `pyrpmd.precision_artifacts` | Read and validate the active published generation, source hashes, PMF, overlap/ESS, parent/child/plateau and energy evidence. |
| `pyrpmd.precision_statistics` | Covariance-aware uncertainty propagation, physical-replica statistics, adjusted intervals and numerical equivalence. |
| `pyrpmd.precision` | Typed targets, identities, sampling/check designs, frozen production, replica evidence and pure acceptance. |
| `pyrpmd.precision_planning` | Pilot supplement selection, frozen production increments, durable checks and stopping. |
| `pyrpmd.precision_jobs` | Materialize concrete existing-workflow configurations and authenticate completed production artifacts. |
| `pyrpmd.precision_ledger` | Reservations, execution identities, actual costs, immutable results, check history and crash recovery. |

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

# Quotes come from F in the SAME cost unit and include complete recomputation.
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

Pilot actions carry target counts and scientific semantics for F to dispatch.
They do not themselves implement checkpoint continuation of existing child
trajectories. Existing rate extension publication remains unchanged.

## Freeze and hand off production to F

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
sampling counts, plateau choice and look sizes before production starts. F calls
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

# Inside F's existing scheduling/execution path, for every planned job:
job = materialize_rate_job(config, system, frozen, planned_job["job_id"])
# F executes job["config"] under plan["execution_budget"] in a fresh destination.
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

These snippets show the F integration contract; scheduler variables and resource
measurement come from F. This module does not launch costly jobs implicitly.
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

The ledger lives under `precision/ledger.json`. Checksums, an exclusive file
lock (`flock` on POSIX, `msvcrt.locking` on Windows), atomic replacement and
file fsync protect publication; POSIX also fsyncs the directory, which Windows
does not support. Reopening must
use the same complete specification and maximum budget. Use
`controller.restore_frozen(epoch_index)` to recover the exact persisted design.

`plan_id` and `execution_id` are durable idempotency keys. Replaying an identical
plan, receipt, check or stop is a no-op; conflicting payloads and duplicate
result IDs fail. A persisted `running` entry requires F to reconcile its actual
job/checkpoint state. The controller never guesses that it should rerun work.
An interruption between check publication and stop publication is repaired on
replay. A later terminal stop is not overwritten by replaying an older check.

Quotes reserve budget; receipts charge actual cost once and release the
reservation. Reserved work is not prematurely stopped just because unreserved
balance is zero. Every plan carries `execution_budget`, which **F must enforce**;
this planning process cannot meter another process's force evaluations. If an
actual overrun occurs it is recorded truthfully and further work is blocked.

Use `ledger.fail(..., actual_cost=..., reason=...)` only for a reconciled terminal
failure. Unknown-cost interrupted work stays running/reserved until F reconciles
it. Failed partial artifacts remain labeled failed and cannot enter acceptance.
A failed replica is never dropped to narrow a confidence interval. A retry after
a terminal failure needs a new registered execution plan or independent epoch;
ordinary interrupted-job resume retains its existing identity.

## Verification and references

Run the acceptance suite with:

```bash
PYTHONPATH=src python -m pytest -q tests/test_precision_*.py
PYTHONPATH=src python examples/precision_control.py
```

The tests include PMF-, parent-, child-variance- and horizon-limited cases;
non-equivalence despite an insignificant difference; covariance and bootstrap
handling; reserved/spent budget boundaries; interrupted and replayed decisions;
source-family deduplication; frozen clocks; and actual small legacy/scoped rate
jobs imported through the existing runner. These are control/estimator
regressions, not validation of a new molecular reaction rate.

Method references:

- [NIST: Bonferroni simultaneous intervals](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm).
- [NIST TN 1297: covariance-aware first-order uncertainty propagation](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-appendix-law-propagation-uncertainty).
- [SciPy: independent observations and physical t-test degrees of freedom](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html).
- [SciPy: paired bootstrap and approximate coverage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
- [statsmodels: equivalence testing and its nonequivalence null](https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ttost_ind.html).
