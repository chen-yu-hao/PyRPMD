# Planned incremental archive for the four-pair H3 child cohort

**Status: preparation only. No active child output has been inspected, packaged,
or analyzed for this plan.** The extra-pair campaign must finish its complete
registered inventory before archive execution or any new acceptance decision.

The registered refinement appends global momentum-pair indices **2 and 3** to
the original **0 and 1** for every parent, seed and timestep. It produces four
pairs per parent in the combined result. It does not add new independent parent
chains or replace parents that produced inconvenient trajectories.

## Fixed identity and scope

The same eight parent chains supply 200 stored configurations each: **1,600
configurations**, with completed Monte Carlo sweeps 2100 through 22000 at stride
100. The two timesteps, 0.1 and 0.05 fs, reuse the same keyed momenta and channel
assignment. Each pair contains plus/minus antithetic children. With both batches,
there are 12,800 children per timestep, or 25,600 across the paired timestep
calculations. **Eight parent chains remain the uncertainty units.**

The frozen base archive is an external prerequisite:

- File: `pyrpmd-h3-quantum-recrossing-stage-200-pairs-2-2026-11-01.tar.gz`
- SHA256: `d3205ebc4a5210af59d03ebdbf0737708278d83f1e4cb540a18818213d78cea8`
- Scope: the original 400 child NPZ files, their complete reports/settings,
  references, original precision decision and associated receipts.

The parent archive is a second prerequisite:

- File: `pyrpmd-h3-quantum-equilibrium-2026-11-01.tar.gz`
- SHA256: `051ca8d016c722b740bc340f07387b72cf14c3379c21394a171dd34f083e944d`
- The exact parent checkpoint/settings/report/reference-domain bytes are needed;
  equivalent arrays in recompressed NPZ files are not interchangeable identities.

The proposed new archive contains only incremental extra-pair raw output and new
complete-cohort analysis/metadata. It references the two existing archives by
whole-file and member identities. The original two-pair and 30000-sweep archives
remain unchanged. External PES files, binaries, Git-backed source, live umbrella
or water data, and duplicated base/parent raw data are excluded.

## Execution gate

Execution requires the original dispatcher to exit successfully and the global
`schedule-200-pairs-1-completed.json` to bind exactly the frozen schedule and all
eight seeds. Every extra run must be complete, source-unchanged and bound to its
settings and 25 expected chunk hashes. Any missing, extra or changed member stops
archiving. A stale completion file is not sufficient evidence that no dispatcher
is still writing; the root must confirm the running campaign has drained.

The combined analysis must use batches `[0, 1]`, horizon 200 fs and the complete
eight-chain inventory. `available_only` is forbidden for the final comparison.
The archive preserves all observed gate outcomes, including failure. The decision
to append pairs was already registered from the complete base precision result;
no partial extra-pair rate is a continuation or selection trigger.

## Independent raw checks

For each of the 400 new chunks, reload the original bytes with checkpoint
integrity required; verify its settings fingerprint, seed, parent indices and
completed sweeps, pair indices `[2, 3]`, full 1–200 fs observation grid, array
shapes, finite values and final integration-step counts. Verify the report's
whole-NPZ hash before and after packaging. The unextended 200-fs chunks must have
an empty predecessor hash; this is a new momentum batch, not a time extension.

Bind all new settings to the **same exact** parent checkpoint/settings/report
identities used by the base batch. Verify each archived parent integrity envelope
and raw domain-file hash. Reconstruct the deterministic channel choice and keyed
Maxwell momenta using global pair IDs 0–3, and verify identical momenta/channels
across the two timesteps. Do not treat the two timesteps as independent replicas.

Recompute each stored signed-flux numerator from the saved velocities, flux
metrics and plus/minus surface signs, and its denominator from `abs(v)/fs`.
For each chain, combine **numerator and denominator sums across both batches**;
do not average individual pair or batch kappas. Recompute the registered plateau,
paired timestep and eight-chain uncertainty calculations. Retain the same-chain
BAR/log-kappa covariance when reconstructing the full-rate interval.

Any retained `failure.json` and `failed-child-state.npz` are explicitly inventoried
as failure evidence and excluded from successful-child counts. They must never
be silently omitted. Unexpected paths require review before the whitelist is
finalized; no recursive inclusion of neighboring active campaigns is allowed.

## Actual extraction and replay required after packaging

1. Verify the whole SHA256 and sidecar member inventory of all prerequisite
   archives and the new incremental archive.
2. Extract the base and new child archives into one new, otherwise empty campaign
   directory. Their paths must combine into `recrossing-primary/seed-…` and
   `recrossing-primary/extra-pair-batch-1/seed-…` without overwriting any file.
3. Extract the required exact parent/reference members into a separate new
   repository-layout directory. Reject symlinks, absolute paths, traversal and
   unexpected duplicate members. Do not rewrite source NPZ or JSON bytes.
4. Reload **all 800** child chunks and the eight parent envelopes from the
   extracted files. Repeat parent, momentum/channel, signed-flux and native-step
   checks against the frozen identities.
5. Replay combined per-chain curves/statistics from the extracted data into a
   new output directory. Numerical results and gate outcomes must agree with the
   complete original combined analysis. Record timestamp/path differences as
   provenance differences only; do not compare a regenerated JSON's whole hash
   to its original as though metadata timestamps were unchanged.
6. Rehash the original raw files after the operation, retain the extraction/replay
   receipt, and freeze a new publication inventory containing only compact
   metadata and any new packaging/replay code. No upload is authorized here.

The raw replay does not evaluate the PES or advance a trajectory. A later
trajectory regeneration remains a separate experiment requiring the exact
external model and native/source identities. The estimated archive/replay work
is tens of CPU seconds and below 256 MiB working memory with streamed members;
this is a planning estimate based on the completed base archive's measured
16.5 CPU seconds, not a new execution measurement.

[plan.json](plan.json) binds the currently inspected immutable protocol, schedule,
decision and source identities. Its completion, analysis and archive fields
remain explicitly pending.
