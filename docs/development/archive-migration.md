# Prepared archive migration

This change prepares an inventory and a reviewable migration path. It does not
delete tracked files, rewrite history, publish data or claim a smaller clone.
The existing [data archive plan](../data-archive-plan.md) remains the full
preservation and retrieval contract.

The tracked inventory at commit `6584af21bfe45c46d1d54c017c3bbbf57646f66d`
contains 3,721 files / 177,598,809 bytes under `docs/validation` and 515 files /
16,002,859 bytes under `benchmarks`. These are uncompressed Git blob sizes.
The working tree may contain later changes; regenerate for a release candidate.

```bash
python tools/prepare_archive_manifest.py --root . --revision HEAD > ../pyrpmd-archive-candidate.json
```

The script reads immutable Git blobs, so it also works from a sparse checkout.
It records original paths, sizes, SHA-256 values and the source commit. It keeps
source/summary files in one provisional category and marks other payloads for
dependency review. This is not an approved deletion list. The existing
`tools/data_manifest.py` separately inventories a full worktree when needed.

Before publishing, select coherent campaign archives, retain the executable
analysis and overview, review model/data licenses, and identify small fixtures
or figures still needed by tests and the manual. Build copies in a fresh staging
directory, add archive checksums and pinned destination metadata, and present
those concrete artifacts for publication approval. After publication, retrieve
the archive independently, verify members and run the affected offline replay.
Only a later change can switch consumers to the verified catalog; existing
payloads and historical links remain available throughout this preparation.

The fast library CI runs `-m 'not campaign'`. A scheduled/manual campaign suite
retains archive-dependent regression coverage. Specialized scientific workflows
remain individually configured; this preparation does not rewrite all of them
or treat their existence as evidence that they have run successfully.
