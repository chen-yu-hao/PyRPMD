# Complete 100,000-sweep archive: explicit read-only lease exception

**Status: new admission design and inert tests; not adopted or executed.** The
original [28-case archive tool and plan](../archive-preparation/README.md) remain
unchanged. This version preserves their scientific inspection, archive and exact
replay functions. It changes the operational admission condition for this one
completed cohort and adds the resulting anomaly evidence to the archive.

## Why a separate admission is needed

Root actually observed sampling session 57942 and complete-analysis session 1094
return exit 0. The final all-eight scheduler manifest and its journal record
normal completion and the drain of all submitted workers. The complete analysis
was run once by the already registered waiter. Its scientific precision failure
remains part of the result; it is not a reason to omit data from this archive.

The original lease nevertheless survived at its expected path. A separately
reviewed, root-adopted `renameat2(RENAME_NOREPLACE)` operation moved that exact
directory to quarantine. Root verified immediate absence of the old path and
preservation of the original directory/owner inodes. A later independent
observation found a **new directory and owner inode at the old path**, containing
the exact same 68 owner bytes. The quarantined original still exists.

The [postcondition review](../stale-lease-postcondition-review/independent-review.json)
records the actual limitation. The cause is unexplained. Neither PID 4 nor
matching owner text proves an active writer, and successful immediate absence
does not prove durable absence. No further retirement is attempted. The original
receipts remain intact, including the successful immediate observation and the
subsequent failure of its durable postcondition.

The frozen v1 tool correctly refuses a present lease. Its source or policy must
not be silently changed to pass this situation. The separate
`quantum_umbrella_archive_v2.py` requires a new explicit root adoption for
**read-only archival of this already completed cohort**.

## Exact admission and continued checks

[policy.json](policy.json) binds the old plan/source/review, both actual terminal
tool-return objects, the complete global manifest, all-eight drain journal,
original execution/report and both lease identities. Admission parses the saved
tool returns: session IDs, integer exit 0, tool chunk and observed timestamps
must agree with root's statements. These are hash-bound root observations, not
digitally signed artifacts or an independent query of the tool service.

The root adoption must explicitly state every condition below:

- Sampling and analysis actually exited; all owned sampling workers drained.
- No new, legacy or direct sampler is running or pending, and no 300,000 stage
  has been admitted.
- No writer will be admitted until this archive attempt has ended.
- The original campaign remains unchanged. Any later continuation uses a new
  isolated working copy restored from a successfully verified 100,000 archive.
- The unexplained anomaly is acknowledged. Lease absence is **not claimed**.

The tool checks the exact present directory/owner inodes, device, mode, size,
mtime, ctime, owner hash and sole-member inventory at admission, before any raw
read, before packing, after extraction, before replay and before publication.
Both original and reappeared owner bytes must match the fixed policy. A missing,
replaced or modified directory also fails this particular admission: the tool
does not reinterpret an unexpected change as success. A new 300,000 global
manifest or per-seed stage directory is forbidden.

This is an explicit root no-writer trust boundary, reinforced by repeated
identity/hash checks. It is not a held sampler lock or a general authorization
to ignore a lease. Raw/source files are hashed before and after the operation;
every published archived member must equal those exact bytes.

## Data and numerical methods are unchanged

The new module imports the exact frozen v1 source and directly calls its:

- all-eight checkpoint/state/RNG/prefix inspector;
- fixed original whitelist containing exactly 33 NPZ files;
- streaming pack, exact-member extraction and byte checks;
- exact scientific JSON/array comparison after the unchanged full-cohort
  analyzer has been invoked once on the fresh extraction.

There is no monkeypatch of the predecessor and no new scientific estimator.
Only the two top-level report timestamps/runtime fields already allowed by v1
may differ in the replay. All gates, including the failed precision gate, must
replay exactly. Source code, native binaries and external PES files are excluded
from the raw archive and remain external hash-bound prerequisites.

Seven small files are added under `execution/lease-anomaly/`: both owner byte
copies, this policy, the first independent lifecycle review, the retirement
adoption, its immediate receipt and the subsequent independent postcondition
review. **No lease is written under the restored `data/campaign-run` path.**
The archived root adoption contains the complete new no-writer conditions.

The archive and member-inventory destinations remain those in v1. The new
evidence, extraction and replay directories have distinct v2 names. All outputs
are exclusive/no-overwrite. A persistent attempt claim and any partial archive
or failure receipt are retained; no automatic retry or continuation is allowed.

## Resources and tests

The unchanged envelope is one root-allocated CPU, one thread, low priority,
300 seconds of observed aggregate CPU, a 900-second work/drain wall envelope,
and 1 GiB observed working memory. CPU/RSS checks are cooperative observations;
the reviewed independent child alarm bounds the owned replay subprocess.
Signal deferral and confirmed child draining remain active through finalization.
No OS guarantee is claimed for uncatchable termination or filesystem stalls.

The [initial gate](initial-gate/execution.json) ran **65 cases: 37 new and the
28 unchanged predecessor cases**, with no failures or skips. It used CPU 5,
one thread, approximately 0.50 child CPU seconds and 0.52 seconds wall time.
No original raw checkpoint, real analyzer or PES was opened or executed.

New fixtures reject changed/replaced/symlinked owners, additional lease members,
new stage directories, incomplete root attestations, rehashed but semantically
wrong terminal records and failed/incomplete worker journals. They prove that
anomaly evidence cannot create a restored campaign lease. An injected phase
guard failure retains its once-only claim and failure before raw admission or
analyzer launch. The original scientific replay/byte-preservation cases remain
unchanged.

After independent review, root must create a **new** adoption document from the
invalid [template](root-adoption-template.json), choosing actual resources and
binding the review. A later execution would use:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
.venv/bin/python benchmarks/h3/quantum_umbrella_archive_v2.py \
  --authorization /absolute/path/to/new-root-adoption.json
```

The template is not an adoption. This design, its tests and the completed
100,000 result do not admit a 300,000-sweep run.
