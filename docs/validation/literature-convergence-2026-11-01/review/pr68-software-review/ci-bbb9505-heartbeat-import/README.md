# Temporary worker import fixture correction

This follow-up changes one test fixture after five jobs at commit
`bbb950528db2f43f59cb24acea5097568cd393a3` failed at the same assertion. It does not change
production code, a scientific model, a statistical gate, or a sampling run.

## Observed failure

Each Python 3.11, 3.12, 3.13 and 3.14 CPU job recorded **1 failure, 7,472 passes,
225 skips and 478 passing subtests**. The fused native job recorded **1 failure,
7,501 passes, 196 skips and 486 passing subtests**. All failures are in
`test_real_capped_worker_self_heartbeat_is_accepted_before_next_admission`,
with `worker exited before initial resource observation`.

[Exact job identities, counts and log hashes](failed-ci-evidence.json) retain
all five decoded full logs. These are per-job counts; repeated versions do not
add distinct test identities. MPI and package-build steps after the failing CPU
pytest step were skipped, not passed.

## Narrow correction

The generated worker script lives outside the checkout. Its Python import path
previously depended on the caller supplying the checkout through
`PYTHONPATH`. The locally focused gate had supplied that variable; the GitHub
workflow does not. This is a code-level explanation for the observed early
exit, but the existing main logs do not expose child stderr, so it remains an
inference until the corrected fixture executes.

The fixture now clears inherited `PYTHONPATH`, explicitly inserts the
`repr`-quoted absolute checkout root before its `benchmarks` import, and
includes the drained children's log contents if the assertion fails. The
resource cap, initial heartbeat, positive RSS, complete observation, exact
child ownership, independent reaping, fixed eight seeds, inert duration and
6/8-second test budgets retain their previous behavior. No test is disabled.

[Independent static review](independent-static-review.json) found no blocker in
this delta. The reviewer independently read the exact published fixture and
both dispatch layers; the candidate's complete bytes were hashed by root,
not independently by that reviewer. The [supersession record](fixture-supersession.json)
binds both file versions and the frozen production source. All text outside
this one fixture is byte-identical.

## Validation boundary at publication

The managed execution environment was unavailable while this patch was
prepared. **The new candidate has not executed locally.** GitHub CI for the
new commit must supply the execution result; no pending run is counted as a
pass. The old failed logs and source are retained even if later CI passes.

The sixth-stage inventory and Sphinx audits describe their original snapshot.
They are not recast as audits of this follow-up. The new inventory binds only
the files in this correction. No completed local MC or P256 result is inferred
from a CI job or from workspace availability.
