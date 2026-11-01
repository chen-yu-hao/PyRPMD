# PR 68 software and fixture review

This is a read-only review of commit `3ff0b674fc9b28e60da9261e08a08ca1ecf75335` and the pending water test files in the integration checkout. The exact inspected worktree hashes are in `fixture-portability.json`. It introduces no scientific source changes and repeats no long test or simulation.

## Remote CI observation

At **2026-11-01 17:24 UTC**, all ten returned pull-request-triggered workflow runs at `3ff0b674fc9b28e60da9261e08a08ca1ecf75335` were **queued**, with no conclusion. No remote pass or failure is established. `ci-status.json` records each run ID. The connector returns the first page of pull-request-triggered runs; this observation does not cover push-triggered runs or later commits.

## Clean-checkout fixture assessment

No unresolved fixture blocker was found for the documented `python -m pytest` CI commands. All thirteen new H3 test files match the hashes in the existing integration test report. That report records 114 passes and nine explicit skips when the optional BKMP2 binary was absent; the subsequent run with the exact pinned binary passed all nine remaining cases, giving **123 distinct tests passed**. These are existing integration results, not an additional test run performed by this review.

The nine model-dependent cases check for `benchmarks/h3/build/libbkmp2.so` before use. The other H3 cases construct analytical models or temporary checkpoint fixtures. The committed H3 source descriptor is available at import time, and importing its potential module does not require loading an absent shared library.

The pending water scheduler tests construct their own temporary registration, source transition, configuration and manifest data. They skip `fcntl` before importing POSIX-only modules. OpenMM engine tests guard the optional dependency before creating a context. The existing liquid checkpoint used by a water restart test is a **31,220-byte NPZ Git blob**, not an unavailable private archive. The two older water JSON fixtures are also committed; byte and checksum evidence is retained in the report.

CI invokes `python -m pytest` from the repository or unpacked source-archive root, which makes the benchmark namespace importable. This assessment does not claim support for running the bare console entry point from an unrelated directory.

## Limits

The water files marked as pending in the report were inspected from the integration worktree and were not all present in the queried PR commit. Later commits require their own CI observation. This review does not certify the full project suite on the new main branch, and it does not turn queued remote jobs into a passing gate.
