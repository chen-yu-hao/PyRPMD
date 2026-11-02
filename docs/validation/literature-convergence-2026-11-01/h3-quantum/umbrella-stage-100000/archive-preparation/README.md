# Complete 100,000-sweep umbrella archive and replay preparation

This additive tool preserves the next already registered umbrella prefix. The
frozen sampler, scheduler, protocol, estimates and acceptance gates are unchanged.
Preparation has read only frozen sources, the prior 30,000-sweep metadata and
inert fixtures. It has not opened a current checkpoint or partial numerical
estimate, created a scientific archive, or invoked the real analyzer.

The fixed [plan](plan.json) requires both actual root observations: sampling
session **57942** and complete-analysis waiter **1094** must have returned integer
exit code zero. Root supplies a hash-bound attestation and the exact saved tool
observations, plus the global completion, final report and execution identities.
This is an explicit trust boundary, not a digital signature or an inference from
partial stdout. The final root adoption remains separate from preparation.

## Exact contents and replay

The new runner is
[`quantum_umbrella_archive.py`](../../../../../../benchmarks/h3/quantum_umbrella_archive.py).
It reuses the frozen `quantum_sampling_scheduler.inspect_seed` function against
the original 30,000-sweep snapshot. Each final seed must have 105,000 total
sweeps, including 5,000 warmup sweeps, samples of shape `(100000,48)` and
diagnostics of shape `(10000,48,7)`. Every original 30,000-sweep sample and
diagnostic prefix and its saved restart-state file must remain identical. The
new geometry, RNG, counters, swaps and lineage must match the new stage receipt
and global snapshot exactly.

The configuration still declares the preregistered maximum of 300,000 production
sweeps. The sampler's `completed` flag can therefore remain false even though
the complete 100,000-sweep stage is present. This archive tool never starts or
authorizes a 300,000-sweep extension.

The whitelist includes **33 scientific NPZ files**: eight current checkpoints,
sixteen old/new stage states, and nine full-analysis WHAM outputs. It also
includes exact settings, progress/run reports, stage receipts, retained sampling
stdout, scheduler metadata, the original complete analysis and actual execution
evidence. The final worker-control path is taken from the immutable original
launch receipt; it is outside `campaign-run` and is preserved at
`data/umbrella-worker-control.json` in the archive.

The earlier raw checkpoint files and historical failed/partial analyses remain
in the immutable 30,000-sweep archive, whose identity is retained as an external
prerequisite. PES source, compiled libraries, Git-backed code, recrossing, water
and other-reaction data are excluded from this new archive.

The runner copies original NPZ bytes without rewriting their ZIP contents,
verifies every archived member, extracts into a fresh directory, and repeats all
eight checkpoint/state/prefix audits on the extracted bytes. It then invokes the
**unchanged complete-cohort analyzer once** with the restored campaign and the
byte-identical archived protocol. `--available-only` is never used.

Replay must match all eight seed reports and the full comparison exactly. Only
the top-level `created_utc` and `elapsed_seconds` fields of the new comparison
report are excluded from that equality. Every array in all nine WHAM files must
match in dtype, shape and C-order bytes; NPZ container timestamps are not used as
a numerical equality test. A changed criterion, nested receipt timestamp, or even
a one-ULP array change fails the replay gate. A scientifically failed original
gate remains a failed gate in the valid replay; it does not trigger a new run.

## Bounds and failure handling

The proposed budget is one allocated CPU, one thread, 300 aggregate observed CPU
seconds and 900 wall seconds, including a ten-second drain reserve. Input size
is limited to 1 GiB and each included file to 96 MiB. The observed working-memory
budget is 1 GiB. These are archive/replay resources, not additional sampling.

The tool requires `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and
`NUMEXPR_NUM_THREADS` all to equal `1` before admission or scientific imports.
CPU and memory observations are checked during hashing/extraction and while the
owned analyzer runs. These CPU/RSS checks are cooperative observed budgets, not
kernel hard CPU/RSS quotas. The reviewed work alarm and independent analyzer timer are
reused from the frozen P256 completion wrapper; only the owned process group may
be stopped, and drain is confirmed before returning. Source and input hashes are
checked again after replay.

The operation relies on root's no-writer attestation plus repeated absence and
hash checks; it does not acquire or modify the frozen sampler's lock. Root must
withhold every later sampling extension until archival completes.

Every output and the execution directory is created exclusively. A failed or
interrupted attempt retains its marker and partial artifacts; there is no
automatic retry, replacement, tolerance widening or cohort reduction. The final
archive name is published only after archive, extraction, prefix and exact
scientific replay checks succeed. This is a no-overwrite/byte-identity convention,
not a claim of permanent operating-system write protection.

The previous 30,000-sweep complete analysis used **23.697437 CPU seconds** and
**23.820031 wall seconds**, according to its frozen execution receipt. Linear
scaling by samples suggests about **79 CPU seconds** for the 100,000-sweep
analyzer alone. That is a planning estimate; actual cost will be recorded.

## Root adoption and intended invocation

[root-adoption-template.json](root-adoption-template.json) is deliberately
non-executable: `adopted` is false and future identities are placeholders. Root
must create a **new** adoption file after both actual session returns, bind the
reviewed runner and plan hashes, allocate one available CPU, attest that no
300,000-sweep writer is pending/running, and fill every required exact-file
identity. The actual redirected stdout files are:

- `literature-convergence-quantum/umbrella-stage100000.log`
- `literature-convergence-quantum/umbrella-analysis-100000.log`

They must not be copied or hashed as final logs while their processes run.
An optional absent analysis stdout file is permitted only with an explicit
retained-tool-output scope; no full stdout may be invented.

The intended command, after root adoption, is:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/workspace/scratch/a5e4820daedd/PyRPMD/src:/workspace/scratch/a5e4820daedd/PyRPMD \
/workspace/scratch/a5e4820daedd/PyRPMD/.venv/bin/python \
/workspace/scratch/a5e4820daedd/PyRPMD/benchmarks/h3/quantum_umbrella_archive.py \
--authorization <new-root-adoption.json>
```

## Preparation tests

The standard test file is
[`test_h3_quantum_umbrella_archive.py`](../../../../../../tests/test_h3_quantum_umbrella_archive.py).
The current candidate passed **28 tests, zero skips, in 0.20 seconds**, using one
CPU 5 and one thread. The complete test subprocess used 0.497903 CPU seconds.
[The exact receipt](final-source-gate/execution.json), source/test snapshots,
stdout and JUnit evidence are retained. The original 25-case version and the
subsequent checkout-independent fixture correction remain in their own earlier
gate directories.

The final review also required that the original execution and comparison
source-hash maps match the frozen plan before raw admission. Two stale-digest
fixtures verify that a rehashed but scientifically incompatible receipt is
rejected at that boundary, before any replay is attempted.

These fixtures check actual tiny NPZ byte preservation, immutable destinations,
unsafe paths, root/session/hash mismatch refusal, all-eight inspection coverage,
exact replay acceptance and one-ULP/criterion/receipt changes, resource refusal
and thread admission. They do not open the real campaign, call a PES, or run the
actual scientific analyzer. Archive/replay success alone would establish
reproducibility of the registered equilibrium estimator, not quantum dynamical
bead convergence or a full RPMD rate.
