# Immutable first quantum umbrella stage

The complete eight-chain **30,000-production-sweep** stage was frozen before
continuation. This operation only copied, hashed and verified files. It performed
no sampling, PES evaluation, new analysis, upload or Git mutation.

## Local artifacts

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `pyrpmd-h3-quantum-umbrella-stage-30000-2026-11-01.tar.gz` | 145,968,199 | `24c1390e11791c9ed2092f98eeba30776bbbb5d46bb8f9a7801b09d62a027c07` |
| `pyrpmd-h3-quantum-umbrella-stage-30000-2026-11-01.members.json` | 30,370 | `9c73b0d8ab8e2db20c64e933e798993337f7cd287e19e3f09d81f7a782f47f69` |

Both are in `/workspace/scratch/a5e4820daedd/`, created without replacement and
made read-only. The archive contains **123 files**, totaling 146,371,757 bytes
before compression. [member-inventory.json](member-inventory.json) is identical
to the adjacent inventory and lists every member, byte count, hash and category.

The archive includes all eight original checkpoints, settings, progress/run
reports, stage-30000 receipts and restart states, sampling logs, complete-cohort
analysis and WHAM outputs. Earlier one-/two-chain analyses and empty failed
initialization logs remain explicitly classified as historical diagnostics.
Protocol, schedule, analysis prespecification, failure/fix receipts and the
immutable precision-continuation decision are included under `reference/`.
No source code, external potential, compiled library, live recrossing data or
water data is included.

## Verified restart and prefix identities

[stage-snapshot.json](stage-snapshot.json), SHA-256
`0ea2cd78af59dafb91ab4f0846786b6b63a63457ecc4434e8489ec1478b6d9a4`,
records all full checkpoint/settings/report/stage-state identities and each
checkpoint array's dtype, shape and C-order byte hash. It is the baseline for a
separately reviewed scheduling wrapper. The eight seeds are 211017, 211019,
211021, 211023, 211027, 211031, 211033 and 211039.

Every checkpoint has 35,000 completed sweeps, including 5,000 warmup sweeps;
`samples` has shape `(30000,48)` and `diagnostics` has shape `(3000,48,7)`.
The unchanged settings retain the originally registered maximum budget of
300,000 production sweeps. Accordingly, the original sampler's `completed`
flag is false even though the registered 30,000-sweep stage is complete.

All eight `R`, `r`, `completed`, `rng`, `accepted`, `proposed`, `swaps` and
`lineage` arrays in each saved stage state match the corresponding checkpoint
arrays exactly. Both samples and diagnostic prefixes match their stage receipts.
Checkpoint hashes match the original run reports and complete-cohort analysis.
These tests were repeated using the bytes inside the finished archive. Every
archive member was checked and all original inputs were rehashed afterward.
The legacy NPZ format has no original per-array integrity envelope; this new
snapshot supplies identities without rewriting any original NPZ.

## Scientific status retained

The completed pooled WHAM QTST point is
`2.8007250268823905e-16 cm^3 molecule^-1 s^-1`. The registered primary log-rate
standard error is `0.07689760467310561`, above the `0.03` precision threshold.
The immutable decision selects all eight chains for the already registered
100,000 prefix because of this precision failure. The point's difference from
the paper does not trigger continuation. The archive is not a full RPMD-rate
validation and does not include or reuse a recrossing coefficient.

## Restore without changing the original experiment

First compare both artifact hashes with the table, then unpack into a new
directory:

```bash
sha256sum /workspace/scratch/a5e4820daedd/pyrpmd-h3-quantum-umbrella-stage-30000-2026-11-01.tar.gz
sha256sum /workspace/scratch/a5e4820daedd/pyrpmd-h3-quantum-umbrella-stage-30000-2026-11-01.members.json
mkdir ./restored-quantum-umbrella-stage-30000
tar -xzf /workspace/scratch/a5e4820daedd/pyrpmd-h3-quantum-umbrella-stage-30000-2026-11-01.tar.gz \
  -C ./restored-quantum-umbrella-stage-30000
```

`data/campaign-run/` contains the exact eight starting chains;
`data/full-cohort-stage-30000/` contains their complete analysis. A working
continuation must use a separate writable copy, matching registered source
hashes, the original schedule/protocol and the independently supplied BKMP2
library hash. Never recompress the archived NPZ files or overwrite the original
stage-30000 state/receipts. Samples remain ordered prefixes after extension;
the old whole-file hashes remain tied to this archive.

[archive-provenance.json](archive-provenance.json) and [archive.log](archive.log)
record the completed operation. [freeze_stage.py](freeze_stage.py) is the exact
executed packaging source and refuses to replace the archive. The new
[publishable-files.json](publishable-files.json) lists only this small metadata
set; add that inventory itself separately. Raw files and the local archive are
excluded from the Git publication inventory.
