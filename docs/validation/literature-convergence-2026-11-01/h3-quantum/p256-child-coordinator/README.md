# Fixed P256 child campaign coordinator

This package supplies an isolated scheduling and integrity layer for the
[reviewed P256 branch producer and analyzer](../p256-child-tool/README.md).
The scientific producer, native propagator, force field, momentum law and
registered statistical rules remain unchanged. The coordinator does not invoke
the rate analyzer, compute a rate, or select work using intermediate results.

The implementation is `benchmarks/h3/p256_child_campaign.py`; its test module is
`tests/test_h3_p256_child_campaign.py`. It reuses the exact reviewed parent
dispatcher and exclusive-lease helper, whose SHA256 is
`7934c1d7f695ce43337d11171cbad8c46ff95311490b755bf4191a56e00c0da3`.
The [fixed eight-chain design](../p256-fixed-eight-chain-design/README.md) remains
the source of the scientific cohort and counts.

## Complete experiment and ordering

The fixed seeds are 825601, 825603, 825607, 825609, 825613, 825617, 825621 and
825623. Each per-seed worker runs these four branches sequentially:

| Order | Momentum-pair batch | Pair indices | Timestep | Required native steps |
| --- | ---: | --- | ---: | ---: |
| 1 | 0 | 0, 1 | 0.1 fs | 1,600,000 |
| 2 | 0 | 0, 1 | 0.05 fs | 3,200,000 |
| 3 | 1 | 2, 3 | 0.1 fs | 1,600,000 |
| 4 | 1 | 2, 3 | 0.05 fs | 3,200,000 |

Every branch uses all 200 registered configurations, both antithetic signs,
256 beads and the fixed 200 fs horizon. A seed therefore contributes 9,600,000
native steps. The complete eight-seed experiment has **32 branch reports,
32 settings files, 800 integrity-bound raw chunks and 76,800,000 native steps**.
Chunk identities contain the exact parent indices and completed parent sweeps.
The saved initial momenta and channel identities must match across the paired
timesteps within each momentum batch.

Only the complete matrix can produce `children-complete.json`. The manifest
records all report, settings and chunk hashes, the full parent manifest, the
complete P128 four-pair baseline, source/runtime/model identities, and worker
resource receipts. Its eligibility flag permits a separate full-cohort analysis;
it does not claim physical convergence or paper agreement. See
[output-contract.json](output-contract.json).

The parent files are located through the frozen design and the completed parent
manifest. Historical parent replays reside under `history-replay/seed-<seed>`;
new independent parents reside under `new-independent/seed-<seed>`. The
coordinator does not use the superseded flat parent-directory template.

## Binding and authorization

Live preflight requires the complete eligible, drained eight-chain parent
manifest and its exact file identities. It also binds the complete P128
four-pair comparison SHA256
`62ce4fea23a55955cf404ae8aa00a2c70204bf20dd48e74de520169a83724720`.
No partial P128 baseline or four-parent-chain subset can be substituted.

The driver pins the final branch-producer SHA256
`9c5c5106d8392ca577f8ea6b54f6e1828dcabb8f86a0e01b732c2e626aa15692`
and analyzer SHA256
`9e2fdacddcf2ef5b825e0eb0b65789e5f045a8505a8fbd09fe21b704ed4936b6`.
The authorization additionally binds the complete required source-file inventory,
actual loaded native-module file, interpreter/runtime identity, external model
and build receipt, synthetic test gate, independent coordinator review, scientific
review, and root's actual resource decision.

[execution-authorization.schema.json](execution-authorization.schema.json)
describes the required adoption document.
[resource-decision.schema.json](resource-decision.schema.json) describes the
resource receipt. Schema validation alone does not authorize a run. The program
checks each cited file and actual CPU availability, then verifies the complete
parent and baseline prerequisites. Root supplies the operational adoption only
after those inputs exist and prerequisite queue drainage is observed.

The fixed output root is:

```text
/workspace/scratch/a5e4820daedd/literature-convergence-quantum/p256-fixed-eight-chain-v1/recrossing
```

The coordinator registers the branch protocol once, before admitting workers
and while holding the exclusive lease. Each worker checks its actual parent PID,
an inherited coordinator invocation token, exact attempt/registration guards,
and an exclusive per-seed claim. Each worker is pinned to one approved CPU and
receives one-thread BLAS/OpenMP settings.

## Resource and failure policy

Root explicitly selected a **36,000-second whole-campaign wall ceiling** and an
initial allocation of two single-CPU workers. This is a separate child-campaign
choice; the parent-generation proposal's 7,200-second ceiling is not reused.
The reviewed authorization can bind one or two workers, with no runtime resizing.

The final 120 seconds are reserved inside the 36,000-second ceiling for owned
worker drainage, raw-data verification and receipts. The inherited dispatcher
checks stop signals, the deadline and already available nonzero worker exits at
each admission boundary. It does not admit replacement seeds after a failure.
Final source guards and the wall deadline are checked again after raw inventory
and resource-receipt verification.

On failure, interruption or deadline, the dispatcher stops admission and sends
SIGTERM only to its own worker process groups. It escalates to SIGKILL if needed
and reaps those processes. It does not wait for an unfinished physical branch
to complete naturally. The worker's signal handler attempts to preserve its
failure and CPU/wall receipt; already committed branch chunks remain unchanged.
A killed or unresponsive worker may not finish that receipt. The global failed
attempt then records the receipt as unavailable and cannot publish complete
eligibility. Reaped-child aggregate CPU is recorded when the coordinator survives.

The exclusive lease remains held until drainage is confirmed. An unconfirmed
drain or hard crash leaves a stale lock; the tool does not remove it automatically.
The unchanged branch producer does not acquire this new coordinator lease, so
no other dispatcher or direct producer may write the same output root.

This first implementation deliberately has **one-attempt semantics**. A nonempty
campaign root, an existing branch, or a repeated worker claim is rejected.
There is no automatic retry or resume CLI. A failed or incomplete attempt retains
its protocol, registration, logs, source identities and exact atomic chunk
prefixes. Any later recovery requires a separate reviewed adoption and may use
the unchanged producer's existing idempotent chunk handling.

## Verification scope

The [final synthetic gate](final/synthetic-gate.json) passed **73 distinct cases**
with zero failures or skips in 2.23 seconds. The initial gate passed 69 cases;
a final-publication guard regression brought the candidate to 70. Independent
review then identified a missing endpoint-clock consistency check. One malformed
clock case and two floating-point recurrence cases brought the final set to 73.
The repeated gates are not additional independent cases. Initial source and
receipts remain under `initial/`; the candidate source and receipts remain under
`candidate/`. The final [test log](final/pytest.log) and
[JUnit results](final/pytest.xml) retain the exact execution results.

The final clock guard compares each raw atomic-unit endpoint against the fixed
200 fs horizon. The frozen native kernel adds its timestep once per physical
step, so the allowance is derived from the positive-sum rounding bound
`gamma_(N+4) * abs(200 * FS_TO_AU)`, with
`gamma_k = k * float64_eps / (1 - k * float64_eps)` and the registered step count
`N`. Using epsilon rather than half-epsilon also allows for timestep and target
conversion rounding. The resulting allowances are about `3.68e-9` atomic time
units at 0.1 fs and `7.35e-9` at 0.05 fs. They were chosen from arithmetic before
observing any P256 child trajectories. Pure repeated-addition tests accept the
expected accumulated clock and reject a one-femtosecond displacement; no
dynamics were needed for this check.

The [independent read-only review](independent-review.json) binds the final
source, tests and 73-case gate. It records the original clock finding, its
resolution, the complete raw-file and process-lifecycle checks, and no remaining
blocking finding within that review's scope. Cross-version execution and any
bounded physical validation are separate gates; this package does not claim
those results in advance.

Tests use synthetic metadata and actual integrity envelopes, plus inert Python
subprocesses. They cover complete and malformed authorization, parent/source/
baseline identity, raw array schema and actual native-step metadata, paired
momentum hashes, all-eight/all-branch membership, missing and repeated chunks,
source mutation during finalization, worker failures, missing resource receipts,
signal receipt preservation, handler restoration, owner-token refusal, one-attempt
exclusion and retention of the lease when drainage is unconfirmed. They construct
no external potential and run no physical trajectory.

The separate [source-only read](final/actual-source-inventory.json) verified the
actual required code/native inventory and runtime against frozen physical hashes.
It did not register a
protocol or perform the full live-input preflight, which requires the complete
parent cohort and root's resource authorization. No partial rate data were
inspected to prepare this package.

The wrapper performs structural and identity checks without deriving rates or
plateau estimates. The independently reviewed analyzer remains responsible for
signed-flux replay, all-eight-chain statistics, paired time/timestep comparisons,
same-chain covariance, and the independent P256/P128 equivalence analysis.

## Invocation after separate adoption

From the original checkout, a concrete adopted authorization can first be
checked without creating the output root:

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python benchmarks/h3/p256_child_campaign.py preflight \
  --authorization /absolute/path/to/adopted-child-authorization.json
```

The separately authorized run uses the same document:

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python benchmarks/h3/p256_child_campaign.py run \
  --authorization /absolute/path/to/adopted-child-authorization.json
```

The private `_worker` entrypoint requires the live coordinator contract and is
not a stand-alone launch or recovery route. This fixed execution wrapper requires
Linux process groups, affinity and the recorded runtime. Its tests skip before
importing those process controls on unsupported platforms.
