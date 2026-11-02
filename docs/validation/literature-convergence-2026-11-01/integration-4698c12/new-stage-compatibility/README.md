# Additional stage methods on Python 3.12, 3.13 and 3.14

**All requested new cases passed without skips, failures or errors.** These
checks cover the newly frozen umbrella coordinator, water plotting eligibility
checks, preparation-only restart runner, and analytic RDF sampling-noise audit.
They reuse the existing interpreters, OpenMM installation and native binaries;
no package was installed, no extension was rebuilt, and no scientific source or
test tolerance was edited.

| Interpreter | New cases executed here | Existing new plotting cases reused | Earlier method cases reused | Distinct passing case union |
| --- | ---: | ---: | ---: | ---: |
| CPython 3.12.14 | 44 | 9 | 217 | **270** |
| CPython 3.13.18 | 53 | 0 | 217 | **270** |
| CPython 3.14.10 | 53 | 0 | 217 | **270** |

This task performed **150 test executions covering 53 distinct newly added
cases**. It did not rerun the earlier 217-case selection or Python 3.12's
already completed nine plotting checks. The resulting 270-case set is exactly
the same on all three interpreters. [case-coverage.json](case-coverage.json)
records the deduplicated identifiers and [summary.json](summary.json) binds the
old and new JUnit receipts used to construct each union. A repeated execution
is never counted as another distinct case.

## New checks and execution

The newly frozen selection consists of 27 umbrella scheduler tests, nine water
plotting/data-eligibility tests, nine small real-engine preparation/restart
tests, and eight analytic RDF-noise tests. The scheduler tests exercise actual
spawned fake workers, complete-cohort continuation, interrupted partial
progress, source/seed guards, exceptional event-journal failure, an OS SIGINT
during draining, and lock retention when shutdown is unconfirmed. The water
tests include real OpenMM Reference/CPU checkpoint and force-cache checks;
they do not run a long water campaign. The plotting tests validate data and
uncertainty contracts, rather than image rendering.

The three invocations ran **sequentially on CPU affinity `[8]`**, with
OMP/OpenBLAS/MKL/NumExpr/OpenMM thread limits set to one. JUnit suite times were
28.253, 28.143 and 29.223 seconds respectively. All three environments used
NumPy 2.5.4, SciPy 1.18.1, pytest 9.2.0 and OpenMM 8.6.1. Complete package
inventories before and after each invocation are retained and unchanged.

| Interpreter | JUnit | Original output | Execution receipt |
| --- | --- | --- | --- |
| 3.12 | [pytest.xml](py312/pytest.xml) | [pytest.log](py312/pytest.log) | [compatibility.json](py312/compatibility.json) |
| 3.13 | [pytest.xml](py313/pytest.xml) | [pytest.log](py313/pytest.log) | [compatibility.json](py313/compatibility.json) |
| 3.14 | [pytest.xml](py314/pytest.xml) | [pytest.log](py314/pytest.log) | [compatibility.json](py314/compatibility.json) |

[selection.json](selection.json) records the exact files and counts before
execution. [run_checks.py](run_checks.py) records native and Python imports in
the pytest process and a separately spawned interpreter. It rejects missing or
unexpected native binaries before any test can silently fall back or skip.
The actual `_native`, `_thermal` and `_electronic` paths and hashes are retained
for all three ABIs.

## Source and runtime identity

The checkout's Git HEAD was `c2c5ae2dce253f1407e7de25ec343ef5e1844a73`.
The eight explicitly frozen new source/test files had been copied into that
integration checkout after the commit; their hashes, together with the source
guard, define the tested worktree. This report does not claim they were already
part of that Git commit. The directory label `integration-4698c12` names the
earlier integration base.

The final guard confirms **791 integrated files, 555 original-checkout
protected files, and ten external model/binary files unchanged**. Original
simulation sources and binaries were not modified. The original Python 3.12
environment was used read-only, with imports explicitly directed to the
integration checkout. Installed distributions remained unchanged in all
three environments.

The six previously documented, owned, Git-ignored CPython 3.13/3.14 discovery
symlinks were present and reused. Their exact targets and binary hashes remained
unchanged. This task made **no link-creation, unlink or cleanup attempt**.
[abi-links-final.json](abi-links-final.json) records the final observed state.
The existing integrated CPython 3.12 native binary retains SHA256
`61dc8ec076a86176e3826415444e64428272441ae3ebe50779acc69bebd51f6f`.

Preflight initially detected a separate change in the integration-only
`quantum_convergence_plot.py`. The root confirmed its intended, frozen update:
bind the figure to its protocol and display the observed κ precision. Its
current source hash is bound to the separate freeze inventory in
[preflight-resolution.json](preflight-resolution.json), and the original
[preflight observation](preflight-observation.json) is retained. No physical
source guard was relaxed, and the new tests began only after that resolution.

## Earlier receipts and interpretation

The earlier Python 3.12 record retains its original nine missing-BKMP2 skips
and two scheduler-fixture failures, together with the separate successful
followups that resolved them. The aggregate uses those completed case-level
records; it does not rewrite the historical outputs. The earlier Python
3.13/3.14 217-case runs already had zero skips and failures. Their receipts
remain in [the earlier ABI method check](../python-313-314/README.md).

The independent preparation-runner review is already retained in the frozen
[preparation review](../../water/preparation-continuation-gate/independent-review.json)
and its 16-file inventory. It was not duplicated here. Its remaining conditions
still apply: all-eight adoption and single-writer scheduling are separate from
the generic single-chain runner, and RDF/coordination equivalence requires an
appropriate predeclared uncertainty policy.

These results validate the selected methods on regular GIL-enabled Linux
x86_64 CPython builds. They do not certify every platform, free-threaded Python,
GPU execution, the complete project suite, long-time liquid equilibrium,
transport coefficients, or a physical literature reproduction.

All new compact evidence files are listed in
[publication-inventory.json](publication-inventory.json). Raw test fixtures,
native libraries and trajectory checkpoints are not publication members.
