# CI failures at commit 0c5fec and fixture repairs

Five complete failed job logs are retained: four Python CPU jobs and one fused
native propagation job. Every job exposed the inert HCH4 fixture's assumption
that pytest temporary directories live under the repository workspace; the
Python3.14 job additionally exposed clock-origin floating-point cancellation.
Both defects were deterministically reproduced without any PES or dynamics.

The repair changes only the two test fixtures. The live output-containment
check and the real calibration resource deadlines are unchanged. All101 cases
from these two modules pass on each of Python3.12,3.13 and3.14, with zero failures,
errors or skips; this is focused validation, not a new whole-suite claim.
The first local gate stopped before collecting tests because it inherited an
older campaign-summary hash. That failure is retained separately; its successor
verified the summary against the already-published commit before testing.

Historical inventories keep their original test hashes. The supersession
receipt links exact old bytes, repaired bytes and the new passing gate; it does
not rewrite a prior scientific run's provenance or acceptance criteria.
