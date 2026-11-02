# Corrected ordinary full-project gate on Python 3.14

The unfiltered ordinary pytest selection completed normally: **7,285 primary
tests passed, 145 skipped, and 486 subtests passed**. There were zero failed or
errored cases. The 145 optional-dependency/platform skips remain skips. Pytest
reported 985 warnings. The complete supervised run took 538.05 wall seconds; it
did not time out and its owned process was reaped.

This run used published commit `36ec289981639bbeb8feecf3b30400b4cbda9afe` plus
one explicitly recorded test-only change: the synthetic HCH4 wall-time fixture
pins its CPU clock so a long-lived pytest interpreter cannot exhaust a
production CPU budget before exercising the intended wall deadline. Its
executed SHA256 is `b19e275b7b6837584afeea8f17f8038ecf162be85f2bfa97447113ee1bd267e1`.
The full fixture supersession, prior snapshot and cross-Python followups are
preserved in the sibling evidence bundles. No production source changed.

The first full-project attempt remains an immutable failed record: 17 failures
comprised 13 denied local-INET sockets, one overlong Unix socket path, two
initial-BLAS-thread expectations under a two-CPU affinity, and the synthetic
wall fixture's inherited CPU time. The retained targeted followups passed 2
affinity cases and 22 IPC/wall cases after those concrete corrections. The
corrected full run used authorized local sockets, short temporary paths and
four allowed CPUs. It installed no dependency and rebuilt no native library.

Before/after source, native, imported-project-path and dependency guards all
passed. The source HEAD and sole tracked test delta remained unchanged for the
entire run. `root-exec-observation.json` preserves root's actual terminal
session 94078 response separately from the child-produced execution/report.

The JUnit suite `tests=7916` includes subtest accounting; the XML contains 7,430
distinct primary testcase elements, comprising 7,285 passes and 145 skips.
Do not report 7,916 distinct tests or add this overlapping full selection to
the earlier 596-case three-version method-test union. This local gate validates
the recorded source plus patch. Remote failures at the unpatched published
commit remain actual failures until a subsequent remote run validates a
published correction.

All logs, JUnit, runtime audits, source snapshots, failure history and exact
executed driver are retained. Scientific convergence is established only by
the separate complete-cohort physical benchmarks.
