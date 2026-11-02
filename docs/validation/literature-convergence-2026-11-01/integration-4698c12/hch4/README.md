# HCH4 replay against the integrated main core

**Integration verification passed; the predeclared P128 short-NVE energy gate
remains failed. No thermal-rate calculation or long sampling was performed.**

The evidence directory label is `integration-4698c12`. The actual execution
HEAD at both start and finish was `3ff0b674fc9b28e60da9261e08a08ca1ecf75335`. Python was
`/workspace/scratch/a5e4820daedd/PyRPMD/.venv/bin/python` (3.12.14), with `PYTHONPATH=src:.` and NumPy
2.5.4. The package, HCH4 adapter and probe all imported from this isolated
integration checkout. The native extension hash was:

`61dc8ec076a86176e3826415444e64428272441ae3ebe50779acc69bebd51f6f`

The unchanged external CBE/PES-2008 library hash was:

`aa33e4f55246d5b99839cd3b08702f1d091a6b3dddac1d6c97d34bb542d95aaf`

## Results

| Check | Result |
|---|---|
| Standard `tests/` HCH4 collection/execution | 11 passed, no failures/errors/skips |
| Native/Python propagation | All 12 comparisons passed: P1/P128, 1/64 steps, unbiased/flux-umbrella/constrained-parent |
| Maximum position difference | 4.08562073062e-14 bohr |
| Maximum momentum difference | 5.00435248796e-11 atomic units |
| Maximum energy difference per bead | 7.43849426499e-15 hartree |
| Actual umbrella worker and completed-checkpoint reload | All four P1/P128, xi=0/1 cases passed; three native steps each, zero fallback |
| Frozen source verification | 202 guarded source/native files unchanged during execution and rechecked afterward |
| Total subprocess CPU time | 6.663896 seconds |

## Known failed energy check retained

The native verification command intentionally returned exit code **1**, and
`native-verification.json` retains `passed=false`. Its P128, 0.1 fs fixed
non-equilibrium 20 fs trajectory has energy span **1.8773813694206165e-4
hartree/bead**, above the unchanged **5e-5** limit. At 0.05 fs the span is
**4.6039519112250904e-5**. Both P1 cases pass. All four energy spans are exactly
the same reported floating-point values as in the original checkout's probe.
No threshold, initial state, time step, potential coefficient or reference was
retuned for this replay. Native/Python parity and source compatibility are
passing checks separate from thermal/physical convergence.

## Evidence and reproduction

[receipt.json](receipt.json) records imported paths, full source hashes, exact
commands/environment, exit codes, CPU/wall timing and comparison with the
original probe. [native-verification.json](native-verification.json) and
[workflow/report.json](workflow/report.json) contain the numerical checks.
[standard-tests.xml](standard-tests.xml) and [standard-tests.log](standard-tests.log)
record the actual standard-directory test run.

The lightweight [run_checks.py](run_checks.py) orchestrates the same three
commands and refuses to overwrite a completed receipt/workflow. Repeat from a
fresh sibling evidence directory if needed. No frozen benchmark implementation
or original publication manifest was modified.

[evidence-manifest.json](evidence-manifest.json) lists publishable new evidence
with repository-relative paths, SHA-256 and sizes. It excludes its own entry to
avoid a self-hash cycle. The eight generated raw checkpoints stay local and
are separately identified in [raw-data-index.json](raw-data-index.json); none
is required as a preexisting test fixture. They are excluded from Git by
default. No third-party PES source or compiled library is included.
