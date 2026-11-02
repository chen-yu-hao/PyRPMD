# PR 68 CI observation and failure diagnosis

This is one read-only observation of [draft PR 68](https://github.com/chen-yu-hao/PyRPMD/pull/68), recorded on **2026-11-01 at 21:15:18.824 UTC**. Its head was `36ec289981639bbeb8feecf3b30400b4cbda9afe`; the PR was open, unmerged and still a draft. The five failed PR jobs below actually checked out GitHub's synthetic merge commit `47860188e45fbc439bdd887edca6347d4e63d0ee`, as retained in their logs. The head SHA identifies the requested run cohort; it is not substituted for that checkout identity.

The exact-head Actions query returned **18 runs: 12 successful, 2 failed, 2 running, 1 queued and 1 cancelled**. Running, queued and cancelled runs are not counted as passes or failed tests. No workflow was triggered, cancelled or rerun during this inspection, and there was no repeated status polling. The two failed workflows' job lists and all five failed job logs were then retrieved to diagnose their failures.

## Actual failures

Every failed job's own complete log reports exactly this one failed case:

`tests/test_hch4_reference_pilot.py::test_wall_supervisor_retains_ledger_and_refuses_second_attempt`

Each traceback reaches `allocate_CPU_budget(..., time.process_time())` and raises `ValueError: preflight exhausted the total CPU budget`. The synthetic wall-time fixture inherited the long-lived pytest process's accumulated CPU time. Its remote source did not contain the later explicit small CPU-clock monkeypatch. This is a real test-fixture failure at the observed revision; the physical pilot's whole-invocation CPU accounting is not implicated by these traces.

| Workflow / job | Run ID | Job ID | Failed | Passed | Skipped | Subtests passed | Seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [Fused native propagation / fused](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41348459099/job/124973258623) | 41348459099 | 124973258623 | 1 | 7,225 | 196 | 486 | 474.55 |
| [PyRPMD tests / CPU 3.11](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41348459263/job/124973261632) | 41348459263 | 124973261632 | 1 | 7,196 | 225 | 478 | 277.22 |
| [PyRPMD tests / CPU 3.12](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41348459263/job/124973261515) | 41348459263 | 124973261515 | 1 | 7,196 | 225 | 478 | 607.79 |
| [PyRPMD tests / CPU 3.13](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41348459263/job/124973261538) | 41348459263 | 124973261538 | 1 | 7,196 | 225 | 478 | 524.56 |
| [PyRPMD tests / CPU 3.14](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41348459263/job/124973261478) | 41348459263 | 124973261478 | 1 | 7,196 | 225 | 478 | 291.86 |

No second failed test case appears in any of these complete logs. Subsequent MPI, build and fused-observation steps skipped after the failed unit-test step are not independent failures or passes. These run counts describe overlapping executions of the test suite and must not be summed as unique tests.

The separately retained local test-only correction changes the fixture to use a small explicit CPU clock while preserving the real monotonic timeout, child-drain and second-attempt-refusal assertions. Its corrected file SHA is `b19e275b7b6837584afeea8f17f8038ecf162be85f2bfa97447113ee1bd267e1`. That delta was absent from the observed remote checkout. Local focused passes or a later local full-suite result do not turn these recorded remote failures into successes. This observation does not establish how a future published revision will fare.

## Other observed states

The H3 PMF independent-seed study's pull-request and push runs were still running. The installed-source/platform-wheel push run was queued, and the push PyRPMD-tests run was cancelled. The corresponding pull-request installed-source/platform-wheel run succeeded: its fetched job list contains Ubuntu 24.04 Python 3.11–3.14 and macOS 15 Python 3.12. It contains no Windows job, so this observation supplies no Windows validation claim.

The full run cohort is in [observation.json](observation.json). Exact failed-step and selected successful installation-job details are in [selected-workflow-jobs.json](selected-workflow-jobs.json).

## Retained evidence and scope

[failure-classification.json](failure-classification.json) binds each complete decoded log's SHA-256 and byte count, retrieval time, API URL, job/run identity, traceback line range and pytest summary. Five adjacent `job-…-failure.txt` files retain the minimal exact tracebacks and summaries. Complete response files remain in the workspace at the paths explicitly recorded in that JSON; they are excluded from the compact publication inventory. The compact bundle contains no credentials, raw physical checkpoint arrays, potential library, new simulation result or revised source code.

The diagnosis was derived independently from each failed remote log, not inferred from an earlier local passing gate. Its scope is all failed jobs returned by the two failed workflows at this single exact-head observation.
