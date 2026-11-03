# Portable publication and rate archives

This T18 follow-up repairs three persistence problems found during the first
Windows library test job. It keeps existing scientific fingerprint algorithms,
checkpoint payloads and RNG state formats unchanged. The drive-letter source
hashing problem is a separate, unfinished part of T18.

## Native trajectory publication

Native trajectories are flushed through a writable, nontruncating descriptor.
The temporary file is still published using an exclusive hard link, so an
existing destination is never overwritten.

The temporary link is removed before NumPy opens the published file as a memory
map. Windows does not permit deleting a file, including another hard link to
it, while a mapping prevents deletion. If loading or subsequent preparation
fails, the unpublished mapping is closed and the engine state is restored.
Cleanup compares the destination's file identity with the file this call
published, preserving a different file installed by a racing writer.

Relevant platform contracts are Microsoft's
[`_commit`](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/commit?view=msvc-170)
and [DeleteFileW](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-deletefilew)
documentation. The local regression emulates the writable-descriptor requirement
with a harmless zero-byte write, then independently compares the trajectory and
final phase-space state with an in-memory native run. Further tests simulate
Windows mapped-file deletion, a malformed mapped payload and a concurrently
replaced destination.

## Portable, authenticated rate generations

New rate-generation checkpoint maps, sample-stream maps and receipt paths use
forward slashes on every platform. Readers also accept the backslash spelling
written by historical Windows runs.

The receipt checksum and every artifact digest are verified against the
original files. Path normalization happens in memory for lookup and identity
comparison; it never rewrites an authenticated receipt, generation description
or checkpoint. Consequently a completed historical generation can be reported
and replayed, and a pending historical generation can resume under its already
bound identity.

Absolute paths, drive names, parent traversal, empty components, alternate
streams and duplicate aliases such as `a/b` alongside `a\b` are rejected.
Symlink confinement remains checked against the generation root. Report readers
and continuation preflight share the same canonical spelling.

The end-to-end regressions generate real small rate jobs, convert their path
records into the historical Windows format with consistent hashes, and then
report, replay and resume them. They assert that the archived checkpoint and
generation bytes stay unchanged. These are persistence regressions on a cheap
free-particle model, not a molecular-rate convergence claim.

## Concurrent JSON publication

An uncontended JSON write still creates a unique temporary, writes and flushes
its complete payload, and performs one atomic replacement. On Windows only,
replacement errors with `winerror` 5, 32 or 33 receive at most five retries.
The requested backoff totals 155 ms; this does not bound operating-system call
latency or scheduling delay. Persistent failures propagate their original
exception and preserve the previous destination. A generic `EACCES`, other
Windows errors and failures on other platforms are not retried.

The access-denied case is based on an actual concurrent `os.replace` failure in
the [original Windows CI job](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41476577723/job/125383616891),
under `test_independent_json_writers_do_not_share_a_temporary_file[writer2]`.
The implementation never falls back to deleting the destination or copying
over it. Unit tests cover recovery from a transient conflict, bounded permanent
denial, unchanged destination bytes, temporary cleanup and unrelated errors.
The existing concurrency test still synchronizes the writers' first attempts;
a retry no longer waits for an already finished peer.

## Validation and integration

The focused baseline on `main@8cc402f` was:

```text
396 passed, 4 subtests passed in 48.92s
```

The final focused command is:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_native_ensemble_portability.py tests/test_native_pimd_followup.py \
  tests/test_native_ensembles.py tests/test_atomic_json.py tests/test_input_cli.py \
  tests/test_ml_lifecycle.py tests/test_drpmd_extend.py tests/test_rate_extend.py \
  tests/test_rate_extension_artifacts.py tests/test_report_rate_pmf_stage.py \
  tests/test_spectroscopy_workflow.py tests/test_spectroscopy_trace_identity.py \
  tests/test_checkpoint.py tests/test_portable_persistence.py \
  tests/test_portable_rate_archives.py tests/test_windows_publication_retry.py
```

```text
439 passed, 4 subtests passed in 54.87s
```

Local execution uses Linux x86-64, Python 3.12.14, NumPy 2.3.5 and one
OpenMP/BLAS thread. Windows-specific behavior is fault-injected locally; this
is not evidence that the full suite has run successfully on Windows. No
numerical-kernel speedup is claimed by this change.

The fixes described above were integrated in PR #86. The Windows skip
registry in `tests/conftest.py` still retains the native publication, archived
generation and concurrent JSON groups pending their separate cleanup. PR #85
removed the drive-letter/source-hash skip group and supplied the explicit
legacy compatibility policy below. Current status and pinned Windows results
are in [the development guide](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/development.md).

<a id="deferred-source-identity-repair"></a>

## Windows source-identity compatibility

Drive-letter plugin/model parsing and hashing are repaired in the integrated
source. Resume rejects older Windows runs when their saved provenance omitted
those source files; it reports that they must be rerun in a new directory.
Today's files cannot establish which bytes produced an old checkpoint, so no
automatic migration fabricates the missing source hashes.

This refusal is checked before migration or workflow dispatch. POSIX identity
values and ordinary checkpoints with complete provenance keep their existing
behavior. The precise affected-run predicate and regression evidence are in
[the Windows compatibility policy](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/development.md).
