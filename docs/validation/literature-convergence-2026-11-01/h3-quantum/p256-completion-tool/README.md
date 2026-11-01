# Complete-cohort P256 analysis admission

This is an additive orchestration tool for the already registered eight-chain
P256 child campaign. It performs no dynamics and introduces no estimator. At
the time this package was prepared, the child campaign was still running. No
active child chunk or branch report was read for this preparation, and the
scientific analyzer has not been invoked by this tool.

The implementation is
[`p256_complete_analysis.py`](../../../../../benchmarks/h3/p256_complete_analysis.py).
The fixed paths, identities, resources and receipt contract are in
[`policy.json`](policy.json). Root adoption and an independent review are both
required before launching even the metadata waiter. They are separate from this
preparation package.

## Admission conditions

Polling reads completion metadata only. The following evidence must agree
before the tool opens any final branch report or raw child file:

1. The global `children-complete.json` declares the exact eight registered seeds,
   32 branches, 800 chunks and 76,800,000 native map steps, with all owned workers
   drained, no errors or unstarted seeds, and no partial-result fallback.
2. The original root launcher has written `child-execution/completion.json`
   with integer exit code zero, unchanged launcher/coordinator sources, the
   registered launch identity and the final stdout hash.
3. Root has actually observed the return of exec session **30588** and written
   `child-execution/root-exec-exit.json`. The receipt binds the actual returned
   tool observation, final global manifest, coordinator registration and root
   launcher completion. An expected future exit or a declared counter is
   insufficient.
4. The original child campaign's lease is absent. The completed worker journal,
   eight execution receipts and eight owner claims must subsequently establish
   that the same registered workers were reaped normally.

After this gate, the tool verifies all 82 static input identities, all 32 final
settings files, all 32 final reports and **every one of the 800 raw file hashes**.
It also checks exact branch membership, chunk filenames, matched reports,
worker ownership, execution counts and the byte-identical attempt/global
completion manifests. This is an identity and completeness check; it does not
calculate partial scientific results. Array integrity and scientific statistics
remain the responsibility of the frozen analyzer and its existing readers.

The root-observed exit receipt must use schema
`pyrpmd.p256-child.root-observed-exec-exit.v1` and supply these exact fields:

| Field | Required value or identity |
| --- | --- |
| `exec_session_id` | integer `30588` |
| `exit_code`, `tool_result.exit_code` | integer `0` |
| `all_workers_drained` | `true` |
| `complete_registered_child_seeds` | integer `8` |
| `driver_sha256` | `201f93a71cccecd2b486d4653cc8666b2e08a7765ca449554844869f6c80de1b` |
| `observed_utc` | actual nonempty observation timestamp |
| `children_complete_sha256` | whole global completion file |
| `registration_sha256` | whole coordinator registration file |
| `root_completion_sha256` | whole root launcher completion file |
| `actual_tool_observation_sha256` | whole actual exec-observation file |

Malformed JSON that is still being written remains pending. A well-formed
failed root completion, changed source identity, incompatible final manifest,
missing raw member or mismatched hash fails admission.

Root's receipt is a **hash-bound attestation**, not a digitally signed artifact.
The wrapper checks its declared actual session and exit fields and binds the
saved tool-observation bytes. It does not independently obtain an exec result
from the hosting service or infer a session identity from partial stdout. Root's
actual observation is the trust boundary.

## One invocation, including on failure

The tool creates a new `analysis-execution` directory exclusively and retains it
permanently, including on interruption, verification failure or analyzer failure.
The analysis output directory must not already exist. No retry, resume,
fallback, cohort reduction or sampling extension is implemented.

The original analyzer is invoked as a subprocess with its exact registered
command and source identity:

```text
/workspace/scratch/a5e4820daedd/PyRPMD/.venv/bin/python
/workspace/scratch/a5e4820daedd/PyRPMD/benchmarks/h3/p256_recrossing_analyze.py
--data /workspace/scratch/a5e4820daedd/literature-convergence-quantum/p256-fixed-eight-chain-v1/recrossing
--output /workspace/scratch/a5e4820daedd/literature-convergence-quantum/p256-fixed-eight-chain-v1/analysis
```

The frozen analyzer compares the eight independent P256 parent chains against
the immutable P128 four-pair result. Matching seed numbers do not create a paired
P128/P256 statistical design. The wrapper accepts a complete, valid report whose
scientific control fails and preserves that failure; it never chooses a new
calculation because of the result.

The metadata wait is limited to 12 hours with 10-second polls. After eligibility
is proved, verification, analysis and finalization have a 600-second wall budget
on CPU 6 with one computational thread. Ten seconds are reserved for termination
and reaping. Hashing checks the deadline between 1 MiB blocks and files. A
post-readiness work alarm interrupts slow verification or import work before
the drain reserve. The analyzer also receives its own kernel real-time timer
before exec, so its timeout does not depend on the waiter surviving. SIGALRM is
deferred across `Popen` until the owned process reference is captured; the
analyzer explicitly unblocks its own timer. SIGINT and SIGTERM remain deferred
throughout owned-process drain, and a final outer cleanup confirms reaping even
if the work alarm interrupted the inner cleanup.
Only the wrapper's own analyzer process group may be terminated. Input hashes
are rechecked before and after the analyzer, and a failed or interrupted attempt
cannot release the permanent one-attempt claim.

The independent review must bind `policy_sha256`, `source_sha256` and
`approved_for_complete_analysis_once: true`. Root's separate adoption must use
schema `pyrpmd.h3.p256-complete-analysis-adoption.v1`, set `adopted: true`, bind
that review's path/size/hash and the exact policy/source identities, and copy
`analysis_resources` from the policy without alteration. Once root has created
that receipt, the intended launch command is:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=/workspace/scratch/a5e4820daedd/PyRPMD/src:/workspace/scratch/a5e4820daedd/PyRPMD \
/workspace/scratch/a5e4820daedd/PyRPMD/.venv/bin/python \
/workspace/scratch/a5e4820daedd/PyRPMD/benchmarks/h3/p256_complete_analysis.py \
--authorization /workspace/scratch/a5e4820daedd/PyRPMD/docs/validation/literature-convergence-2026-11-01/h3-quantum/p256-completion-tool/root-adoption.json
```

This command is documented for review; it has not been run during preparation.

## Validation and provenance

The final candidate passed **43 synthetic and inert-process tests**, with zero
skips, in 3.72 seconds on Python 3.12. The execution receipt reports 3.220486 CPU
seconds and 3.993003 wall seconds for the complete pytest subprocess. The
standard test file is
[`test_h3_p256_complete_analysis.py`](../../../../../tests/test_h3_p256_complete_analysis.py).
The raw pytest log, JUnit result, exact source/test snapshots and before/after
identities are retained in [`resource-gate`](resource-gate/execution.json). The
earlier 30-case and 38-case gates and their exact snapshots remain in
`initial-gate` and `final-gate`. Independent review of the 38-case candidate found
that the pre-launch hash loop and analyzer needed stronger timeout protection;
the final version adds that protection and five targeted inert tests.

Tests cover metadata-only polling, all final membership and hash checks using
800 tiny inert files, failed/root-mismatched receipts, absent or changed raw
files, permanent concurrent-attempt exclusion, post-analysis input changes,
nonzero exit, timeout and signal/journal failures during process drain. The
additional timeout checks cover independent kernel expiry without parent
polling, a slow verification, expiry between hash blocks and a work alarm during
inner process cleanup. The
real frozen analyzer was never executed by these tests. A synthetic analyzer
that reports a failed scientific control is intentionally accepted as a valid
completed analysis, without a second invocation.

The live scientific sources, original protocol, parent checkpoints and P128
baseline remain unchanged. Source/test hashes and all preparation files are
listed in the package inventory. This package contains neither raw scientific
trajectories nor external PES source or libraries.

## Later raw-data preservation

[`raw-archive-plan.json`](raw-archive-plan.json) proposes an explicit, bounded,
no-overwrite archive after complete execution and process drain. It is a plan,
not an archive receipt. It includes all 800 P256 chunks and their exact metadata,
references the already archived P256 parents and P128 baseline externally, and
excludes active umbrella/water data, third-party PES assets, native libraries and
Git-backed source duplication. Future extraction must reproduce member bytes
and validate every checkpoint before the archive can be called complete.

No P256 dynamical convergence or paper agreement is established by this
orchestration gate. Those conclusions remain contingent on the complete frozen
scientific analysis, including all registered uncertainty and equivalence tests.
