# Explicit publication set for HCH4 and the other-reaction source audit

[publish-files.json](publish-files.json) lists each proposed Git-backed file by
repository-relative path, byte length and SHA-256. It includes the HCH4 source,
tests, reference extraction and compact verification reports. It does not
include third-party PES source/binaries, Python cache files or the eight raw
workflow checkpoints. The manifest does not contain its own hash; that digest
is supplied with the handoff to avoid a self-referential checksum.

## Raw checkpoints stay outside the publication set

[hch4/raw-data-index.json](hch4/raw-data-index.json) records the exact local
relative paths and hashes of the eight `.npz` files, totaling 175611 bytes.
They are three-step workflow outputs, not required test fixtures. No test,
input generator or numerical verification requires those preexisting files.
`verify_workflow.py` creates a fresh output directory, generates the short
trajectories from its explicit configuration/seeds, writes its own checkpoints
and verifies their reload. The replay requires the externally rebuilt pinned
PES, as documented; it does not require downloading a saved trajectory.

The raw files remain in this workspace and are **excluded by default from Git**.
The old execution receipts describe all files present during that execution,
including raw outputs and incidental Python bytecode. Those historical receipts
are provenance, not instructions to stage every named file. The explicit
publication manifest controls the proposed upload set. The workflow report,
source hashes and this raw index preserve the numerical results and their
identities without including raw trajectory states in Git.

## Standard pytest collection

The original seven native/mapping checks remain unchanged in
`benchmarks/hch4/test_native.py`. The new `tests/test_hch4_native.py` explicitly
imports their four test functions and the external-library fixture. Parameter
marks are retained, so the parametrized mapping test expands to four cases and
the standard test directory collects seven cases in total. This is the minimal
collection bridge; no verified benchmark source was moved or changed.

Paths remain valid: the fixture obtains its library path from
`PYRPMD_CBE2008_LIBRARY`, and the imported benchmark modules still resolve data
relative to their original `benchmarks/hch4` locations. No automatic source
download occurs. With that environment variable unset, external-PES cases skip
while the reference-only consistency check still runs.

```bash
PYRPMD_CBE2008_LIBRARY=/path/to/libcbe2008.so \
  python -m pytest tests/test_hch4_native.py tests/test_hch4_pes.py -q
```

This standard-directory command passed all **11** cases. Collection output and
the exact bridge/original-source hashes are retained in
[standard-test-integration.json](hch4/native/standard-test-integration.json).
The previous benchmark-path test execution and all native energy-gate outcomes
remain unchanged.
