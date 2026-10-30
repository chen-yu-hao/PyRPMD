# Native propagation branch integration review — 2026-10-30

This review integrates the remaining native-propagation branches into `main`
while preserving their commit ancestry. The baseline was
`28eae0ae5efc5f8633cc772c991774a3f6f8ea52`.

## Branch inventory and conflict decisions

| Source | Reviewed head | Content and integration decision |
| --- | --- | --- |
| `feat/fused-native-propagation` | `a48847c4df0bf44aada6882f8d647311d5f577d4` | Actual C++17 fused propagation, explicit native PES binding, observations, production workflow integration and Python fallback; retained. |
| `feat/fused-native-propagation-20261030` | `3fa3c75cd0a2cfed7445214aad2d314bc0e2102a` | Source/PES preparation workflow only; retained its provenance artifacts in the complete validation workflow. |
| `feat/native-propagation` preparation | `aacc0f25e9c4b3bbe5e2b5afd5577e8470301c07` | Initially a validation workflow only; consolidated runner pinning, concurrency and source provenance, and removed the duplicate workflow and silent missing-benchmark skip. |
| `feat/native-propagation`, PR #3 | `ce398e9180408c9f75ad423127d69e4d31870b6e` | Arrived during the first integration's CI run with a complete second implementation. Retained its segmented API, native RATTLE, observation masks, bead-average geometric CVs, valid-prefix failure semantics and independent verification. |
| `bench/native-loop-bkmp2-20261030` | `23006115beb2c6ddbdb44775c9020b7581aa4c39` | Already squash-merged by PR #1; recorded its ancestry without reverting the later main-branch benchmark notes. |

The add/add conflict in `.github/workflows/fused-native.yml` retained the full
production validation rather than the preparation-only version. The benchmark
report conflict retained the later report already present on `main`.

PR #3 was reviewed before completing the integration. The two propagation APIs
are retained under their existing names: `propagate()` keeps whole-block atomic
rollback; `propagate_block()` commits accepted steps and returns a valid prefix
on numerical failure. Both use the original immutable Cayley context. Native
PES binding adapters preserve the same lifetime owner, context, close check and
lock across the APIs. The original 256-step default and 65536 input maximum are
preserved; callers of the segmented API cap individual requests at 4096.

The original `benchmarks/h3/fused_propagation.py` remains unchanged. PR #3's
complementary harness is retained as `benchmarks/h3/native_propagation.py`, with
execution-field compatibility, explicit result-count assertions and exact source
identity. Its tests retain the distinct `test_native_propagation.py` name. A
single permanent workflow runs both harnesses, both regression modules, actual
MPI, and source-package rebuild validation. JUnit and compiler/CPU environment
records are included; no source-transport or repository-writing workflow is kept.

## Findings repaired before merging

1. **Native context lifetime race.** `CABIPotential.close()` previously published
   `closed=True` after releasing its lock. An already exported binding could
   therefore admit a destroyed context. Closing now checks and publishes the
   lifecycle state under the same lock used by evaluation and native blocks.
   `evaluate()` also rechecks the lifetime after preparing input buffers. Two
   deterministic tests reproduce both interleavings without dereferencing a
   freed pointer; both fail against the original implementation.
2. **Incomplete source package.** The real fused harness imports
   `benchmarks.loop_native`, which was absent from the source distribution.
   `MANIFEST.in` now includes its Python, C, C++, Fortran and header files.
   Permanent CI extracts the sdist, rebuilds its native extension, verifies the
   loaded extension location, recompiles the pinned PES and runs fused tests.
3. **dRPMD restart accounting.** New trajectory checkpoints preserve cumulative
   backend counts; replay/rollback restores the counts with the physical state.
   Counters exclude unrelated trajectories on the same `Dynamics` object.
   Legacy checkpoints remain readable and report previously completed steps as
   `untracked_steps`, without inventing their original backend.
4. **Different native failure contracts.** The public atomic API retains its
   original contract. dRPMD collisions now consume the segmented API's accepted
   prefix, without retrying failed PES calls through Python. Custom classifiers
   remain Python callbacks at their original check boundaries. The reviewed
   checkpoint-counter and classifier-rollback protections remain required tests.
5. **Overlapping verification assets.** Both real-system harnesses and regression
   suites remain active. The renamed segmented harness checks exact counts of
   windows, parent configurations, child pairs and collision records before
   comparing results, so `zip` truncation cannot hide missing outputs.

## Scientific and compatibility review

The review checked the B-C-B map, sum-beads Hamiltonian, ring-spring energy,
centroid umbrella factors, flux-metric forces, equivalent-surface min/max and
tie selection, Fortran/C memory layout, finite-result checks, state rollback,
PES locking and explicit Python fallback.

FES blocks preserve thermostat, sample and checkpoint boundaries. dRPMD blocks
stop at classifier and checkpoint boundaries and preserve persistence semantics;
per-step records retain exact trajectory-output times inside a block.
Recrossing children retain every surface timepoint and paired weights.
Constrained parents use native centroid Cayley-RATTLE through the segmented API
when supported and retain the original Python algorithm for unsupported cases.
Custom or overridden Python PES/CV/bias methods remain active through fallback.

This integration does not establish physical rate convergence or certify GPU
hardware execution. The permanent checks cover CPU native propagation, actual
four-rank MPI workflows, Python compatibility and real H3/BKMP2 comparisons.

## Verification and reproducibility

New deterministic regressions are in `tests/test_cabi_lifecycle.py` and
`tests/test_drpmd_propagation_resume.py`. The latter covers intermediate and
final-step restart, legacy checkpoints and classifier-error rollback, comparing
exact final coordinates/momenta, sampled trajectories, energies and drift.
The reconciled local suite completed **166 tests and 87 subtests**, with two
optional Torch skips. The iPI socket test was excluded from this sandbox run
because local sockets require a separate execution permission; that test passed
separately, and remote CI runs the full suite. Additional regressions exercise
cross-API bindings, generic two-atom Fortran batches, custom provider/CV/dynamics
semantics, exact CV shapes, failed-step no-retry behavior and full-state
classifier rollback. Alias imports of the real BKMP2 adapter share a canonical
library lock across ordinary and native calls.

The first source-package audit rebuilt the unpacked extension and passed all
16 atomic-API tests, including the real BKMP2 flux-metric and Andersen RNG
comparison. After reconciling PR #3, the independent sdist audit verified both
headers, both Python API modules and both harnesses, rebuilt the extension from
the unpacked source, confirmed that this rebuilt library supplied both entrypoints,
and passed all **36 tests** in the two propagation suites with real BKMP2 present.
The real-system harness is `benchmarks/h3/fused_propagation.py`; it checks
1/2/8/16/32 beads, two-dimensional FES, rate-PMF windows, recrossing children and
dRPMD stopping, plus interruption/resume.

The additional `benchmarks/h3/native_propagation.py` validates the segmented
API's NVE, off-diagonal two-dimensional umbrella, flux metric, constrained
parent and recrossing observations. Its workflow study compares four FES
windows and twelve PMF windows, three constrained parent configurations/six
children, eight real H3 collisions with the actual Python exchange classifier,
and interrupted dRPMD restart after changing block size. Native step counters
are checked explicitly. Reports distinguish its `Dynamics.step()` core timing
baseline from compatibility-workflow timings and retain
`physical_convergence=false`.

The local reconciled-source run completed all ten segmented core cases at
16/32 beads with five timing repeats. Maximum core differences were
`1.56e-15 bohr` in q, `4.66e-13` in p, and `3.02e-14 hartree` in energy.
The workflow checks saved 160 FES samples and 480 PMF samples, compared all six
recrossing children, and executed 10275 native collision steps for the eight
H3 trajectories. Python and native both gave seven nonreactive trajectories,
one unresolved trajectory and no failed trajectories. Completed and interrupted
collision restarts were exactly reproducible after changing block size.
These local checks validate the working tree recorded by the reports' source
hashes. Remote checks validate their own recorded commits; the earlier remote
baseline below is identified separately.

## First integration validation, before PR #3 reconciliation

The earlier integration head was
`6e0ebdc5f738de56f3efc2ced2a51d030721b779`, with reviewed tree
`550d72d91452f3c47af207e6d8b053a90ed1b2ca`. Its PR test merge commit
`bf7b617c71f525b5727b56caafbdba4be67ec2ca` had the same tree. These runs establish
the pre-reconciliation baseline, not the final combined implementation:

- [Fusion and packaging](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41018479699):
  127 passed, 2 optional Torch skips, 87 subtests; actual 4-rank MPI; real BKMP2
  numerical/workflow comparisons; extracted-sdist rebuild and 16 passing fused
  tests. An initial CI host environment lacked setuptools; the dependency was
  declared explicitly and both PR and push reruns passed the complete gate.
- [CPU/Python/Torch](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41018479712):
  Python 3.11 and 3.12 each had 124 passes, 5 optional-dependency/PES skips and
  87 subtests; the Torch job had 10 passes and one optional ASE skip.
- [Independent-seed PMF campaign](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41018479689):
  all four cases and the aggregate report completed. The report has
  `passed=true`, `errors=0`, **`converged=false`**. It contains 848000 samples in
  424 streams; each case has 40 valid bootstrap and paired-bootstrap replicates.
  Artifact `12961236763` retains the complete campaign. This is a workflow and
  numerical-validation result, not evidence of converged physical rates.

Integration PR checks and the subsequent `main` checks provide the authoritative
results for their respective commit SHAs. The `Fused native propagation` artifact
contains exact source commit and archive, test logs, full real-system JSON,
pinned BKMP2 build provenance, both harnesses' reports, JUnit, compiler/CPU
environment, checkpoints, sdist and wheel. The original implementation's
earlier measurements remain separately identified in
[`fused-production-2026-10-30`](../fused-production-2026-10-30/README.md).
