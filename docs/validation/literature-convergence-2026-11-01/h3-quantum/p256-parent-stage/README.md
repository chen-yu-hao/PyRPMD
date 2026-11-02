# Completed P=256 parent verification and raw archive

All eight registered parent chains completed normally. This stage verifies their identities, saved configurations and execution lifecycle, and preserves their exact checkpoint bytes. It does not calculate a transmission coefficient or establish dynamical bead convergence.

The complete-data audit began only after the root execution receipt recorded session `97158` exiting with code 0. The observation was made at 20:07:49 UTC on 2026-11-01; it is an observed exit time, not a reconstructed native completion timestamp. The campaign wrapper reported 1080.757510934 seconds of wall time and unchanged sources. The complete manifest and registration hashes are recorded in [complete-audit.json](complete-audit.json).

## Verification result

The verifier accepted all eight checkpoints, each containing 200 saved P=256 configurations from completed sweeps 2100 through 22000 at intervals of 100 sweeps. The uncertainty unit remains an independent equilibrium chain: there are eight chains and 1,600 configurations, not 1,600 independent trajectories. Every checkpoint, settings fingerprint, reference, domain, report, geometry and registered source identity passed.

| Seed | Origin | Minimum BAR overlap | Minimum energy ESS | Half-series log partition difference |
| --- | --- | ---: | ---: | ---: |
| 825601 | Historical deterministic replay | 0.903773 | 898.2 | -0.034431 |
| 825603 | Historical deterministic replay | 0.898188 | 728.6 | +0.025280 |
| 825607 | Historical deterministic replay | 0.904029 | 1025.2 | -0.010783 |
| 825609 | Historical deterministic replay | 0.902070 | 734.4 | +0.021130 |
| 825613 | New independent chain | 0.902283 | 1078.0 | -0.013806 |
| 825617 | New independent chain | 0.911944 | 950.1 | +0.088025 |
| 825621 | New independent chain | 0.899359 | 829.3 | +0.014525 |
| 825623 | New independent chain | 0.899458 | 865.8 | -0.033648 |

These are the predeclared parent eligibility diagnostics, copied from the verified full reports, with thresholds 0.2 for minimum overlap, 200 for minimum ESS and 0.2 for the absolute half-series difference. The audit did not reestimate BAR. They do not replace the later complete-rate confidence interval or transmission precision requirements.

For the four historical chains, all 16 registered non-observation checkpoint payloads matched the original dtype, shape and bytes: 64 comparisons in the original audit and another 64 after extraction. Reference/domain bytes and the scientific report content also matched. The execution journal independently accounted for all eight successful admissions, eight exit-code-zero reaps, eight verification events, at most two concurrent workers and a final drained pool. No subset or unfinished worker was accepted.

The geometry and historical-array checks reuse the exact reviewed `p256_parent_campaign.py` verifier, SHA256 `7934c1d7f695ce43337d11171cbad8c46ff95311490b755bf4191a56e00c0da3`. This is a separate lifecycle, provenance and archive audit, not a second independently implemented geometry oracle. Its new synthetic tests passed **20/20**, without PES evaluations or physical propagation; see [synthetic-gate.json](synthetic-gate.json).

## Archive and exact replay

The local archive is `pyrpmd-h3-quantum-p256-parents-2026-11-01.tar.gz`, 47,557,304 bytes, SHA256 `719a52f69e778ce5e9c25c10f438e3f862e8006be49dba8b09954ccd32eb3d30`. Its 90 members contain exactly eight current raw checkpoint NPZ files and supporting settings, reports, registrations, execution receipts and archive metadata. [raw-data-index.json](raw-data-index.json) gives every raw member's original path, byte size and checksum; [member-inventory.json](member-inventory.json) covers every archived member. The external sidecar is 36,739 bytes with SHA256 `6364e84fa93788f4dbca4c0df9cddcc9cef4d9cb7a9e478aa073afc4c186e889`.

The archive was actually extracted into a fresh directory. Every member was hashed and all eight parent checks repeated there. Original and extracted verification results were identical after disregarding only their relocated path strings. [archive-verification.json](archive-verification.json) records the result, 3.399510 CPU seconds, 3.529462 seconds of packaging wall time and a peak RSS of 55,920 KiB. No PES calls, dynamics, upload or source modification occurred.

To reproduce the audit, obtain the exact registered repository sources and external dependencies listed in [external-prerequisites.json](external-prerequisites.json), verify both archive and member checksums, then extract to a new directory. The archive contains `parents/`, `root-execution/` and `metadata/`; the helper's `audit_complete(parent_root, completion_receipt)` accepts the extracted `parents/` and `root-execution/root-exec-exit.json`. The four original no-snapshot checkpoints are external prerequisites in the earlier equilibrium archive and are deliberately not duplicated. The member manifest retains original provenance paths while the audit checks extracted routes against the registered design. The recorded actual packaging command is in [archive-execution.json](archive-execution.json); its packaging entry point refuses to overwrite existing outputs.

The archive excludes PES source, model libraries, native binaries, Git-backed source code, the older historical raw checkpoints and all recrossing or umbrella campaign data. Its immutability means exclusive creation, frozen bytes and explicit checksums; no durable filesystem write-protection property is claimed.

## Path correction and scope

[manifest-path-correction.json](manifest-path-correction.json) preserves an additive correction to the older prose contract: historical parents use `history-replay/seed-{seed}` and new chains use `new-independent/seed-{seed}`. The frozen design and executed producer already used these routes. The obsolete flat prose template is not authoritative, and no earlier inventory was changed.

This completed archive makes the eight parent sources reviewable and reproducible. Child eligibility adoption, bounded implementation calibration and any P=256 real-time propagation remain separate decisions with their own receipts. No P=256 rate or bead-convergence result is claimed here.
