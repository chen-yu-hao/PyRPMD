# Completed 100k umbrella archive review

The bounded independent review found no remaining archive or metadata identity blocker. It streamed the exact archive SHA256, checked all 83 small members against their original and restored copies, checked 40 small external source/identity files, and independently compared the nine saved scientific JSON replay outputs. The review did not decode any NPZ, run an analyzer, evaluate a PES, draw random numbers, or propagate dynamics.

The archive contains **116 members, including exactly 33 NPZ files**: eight complete checkpoints, sixteen 30k/100k restart states, and nine WHAM outputs. All eight seeds retain 100,000 production observations per window after the original 5,000 warmup sweeps. The complete executor's array-byte and RNG/prefix checks remain explicitly distinguished from this review's metadata checks.

**The registered precision gate still fails:** the primary log standard error is 0.050873748737126245, above 0.03. The archive and replay succeeding do not establish a converged QTST or full RPMD rate. The unexplained lease reappearance and the actual root no-writer/exit attestations remain visible; the review neither deletes leases nor claims that they disappeared.

For a future isolated continuation, preserve the byte-identical `data/campaign-run/stage-100000.json` completion manifest as the scheduler's prior-stage input. The archive-specific `raw-archive-v2/stage-snapshot.json` wrapper has a different schema and cannot be substituted. Record a new working-directory route separately without rewriting old metadata.

## Evidence

- [Independent review receipt](independent-review.json): scope, exact identities, conclusions and limits.
- [Bounded metadata checks](checks.json): 1,127 assertions and all 205 observed file identities; 0.340 CPU seconds for the independent check.
- [Check program](check_small_files.py): standard-library-only implementation, including an opaque streaming archive hash and no array decoding.
- [Completed executor receipt](../raw-archive-v2/execution.json): existing executor's complete scientific replay and checkpoint/RNG/prefix results.
- [Root actual exit observation](../archive-root-adoption-v2/root-exec-observation.json): session 46558 exited 0, observed at 21:40:29 UTC on 2026-11-01.

This review authorizes no new simulation, execution plan, continuation, upload, source edit or Git operation.
