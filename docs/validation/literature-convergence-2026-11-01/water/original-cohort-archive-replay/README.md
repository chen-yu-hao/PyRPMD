# Actual extraction and replay of the original eight-chain water archive

**The archive replay passed.** A fresh extraction of the sealed raw-data archive
reproduced the complete original analysis report byte for byte, including every
failed preparation and transport-window check. All 696 compact curve arrays
also matched in dtype, shape and bytes. This validates the archive's use as
analysis input; it does not change the original scientific eligibility result.

## What was executed

The one-shot [replay harness](replay.py) ran in the original frozen Python 3.12
environment on CPU 5 at nice level 10, with one thread, a 300 CPU-second hard
limit and a 600 wall-second limit. It completed in 29.732 CPU seconds and 31.636
wall seconds with a peak resident set of 198,192 KiB. The tool observed process
exit code 0; [execution.json](execution.json) records that observation and the
exact command. No dynamics steps, force evaluations or OpenMM contexts were
created.

The input was the 1,476,697,324-byte raw-data-only archive identified by SHA256
`b3dc0faf5ddb84d06182ebf788edc73dfa1a0ec674069f01e61eaa62918d21af` in the
[sealed archive index](../original-fixed-cohort/raw-data-index.json). The harness
checked the archive checksum, extracted all 1,685 regular-file members to a new
workspace directory and checked all 1,684 entries in its checksum list. This was
an actual extraction and reanalysis, beyond the earlier streamed member checks.

During analysis, a Python file-access audit rejected any access under the
original raw-data directory and rejected trajectory/configuration reads outside
the extracted payload. The recorded input reads were exactly the 1,600 extracted
trajectory chunks and eight extracted seed configurations. There were zero
attempts to read the original raw-data directory. A single separate read of the
newly generated `curves.npz` was allowed for the frozen analyzer's output
checksum. The full input-path and array-comparison record is in
[read-audit.json](read-audit.json).

## Explicit source reconstruction

The extracted protocol, checkpoint contract, equivalence definition and eight
configuration files were compared with their Git blobs at commit
`c2c5ae2dce253f1407e7de25ec343ef5e1844a73`, and the extracted copies were the
ones actually passed to the analyzer. The source patch
`checkpoint-contract-v2.patch`, intentionally absent from the raw-only archive,
was reconstructed from that same commit alongside those extracted inputs. Its
SHA256 is
`81f1f008131a07cebda1796d7c1cbcca930b7923d31c4cbecb7827d360c8d365`.
The original archive and repository files remained unchanged.

The analyzer and all six recorded dependencies were read from the original
checkout, with both actual imported module paths and file checksums checked.
Their before/after identities are recorded in [report.json](report.json).
The exact diffusion helper came from original commit
`3d096d5cfc5709f14a1ad5e3d184b108eef9f20e`; its hash is recorded separately from
the Git commit used to reconstruct the protocol inputs. The
[source-provenance record](../original-fixed-cohort/source-provenance.json)
documents these per-file source references.

## Exact comparison and limits

The replay report SHA256 is the same as the original:
`f544d5de9202a73f7013ee5191fed4e53e330a9eeb41c87770bfaaa3f7d4d1f8`.
The compact curve container also reproduced the original SHA256:
`1555714decf9bf34b73463d05283ada6a8c5aa875798ca8834e18eb65f2e17f2`.
Every physical, statistical and gate field was compared exactly, and the curve
arrays were additionally compared independently of the container checksum.

The original failed potential-energy preparation checks and the failed
0.5–1 ps MSD-window check remain failures. The literature-comparison field
remains null. This replay establishes recoverability of the saved numerical
analysis under the frozen environment; it does not establish equilibrium,
diffusion convergence, portability to every dependency version, or bitwise
future OpenMM CPU trajectory continuation.

This directory is an additive immutable evidence bundle. Its harness deliberately
uses a new, fixed output directory and refuses an existing one, so the published
execution cannot silently overwrite or repeat itself. General archive replay
instructions remain in the archive's README. The actual extracted raw payload
is retained outside Git and is not duplicated in this compact evidence bundle.
