# Independent umbrella coordinator review

The reviewed coordinator has **no remaining blocking finding** after correcting
one exceptional-shutdown exclusion bug. The old version restored SIGINT before
waiting for workers during an exception; an interrupt at that point could
release the directory lock while workers still wrote their checkpoints.

The corrected version keeps deferred handlers active through draining. Its
lock-release lease stays disabled until the pool confirms shutdown. If draining
cannot be confirmed, it preserves the lock and does not publish a resumable or
complete-cohort receipt. A persistent event-journal failure is handled after
workers have drained.

The independent bounded check ran the two new regressions against the reviewed
source: **2 passed, 25 deselected, no skips, 2.15 seconds, one CPU**. One test sends
an actual OS SIGINT during exceptional shutdown with a live fake worker; the
other injects an unconfirmed shutdown and verifies that the lock is retained.
No PES or MC sampling was run. The exact results are in
[failure-path-tests.log](failure-path-tests.log) and
[failure-path-tests.xml](failure-path-tests.xml).

Read-only inspection also covered the unchanged `one_seed` worker contract,
complete eight-seed precision-only continuation decision, next registered
prefix, preserved observations/RNG states, source and import-path guards,
atomic global publication, and explicit failure-receipt resumption. Each worker
owns one seed directory; only the coordinator publishes the global result.
Changing the worker-count control changes scheduling admissions only.

The operational exclusion of the older direct campaign CLI remains necessary:
it does not acquire this new lock and must not run concurrently. A stale lock
from a hard crash or unconfirmed drain must be inspected, not automatically
removed. These software checks do not establish physical convergence.

[review-receipt.json](review-receipt.json) binds the reviewed source and test
hashes, initial finding, correction, and scope. The four publication files are
listed in [publication-inventory.json](publication-inventory.json); existing
frozen scientific files and review manifests were not changed.
