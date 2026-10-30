# Native propagation branch integration review — 2026-10-30

This review integrates the remaining native-propagation branches into `main`
while preserving their commit ancestry. The baseline was
`28eae0ae5efc5f8633cc772c991774a3f6f8ea52`.

## Branch inventory and conflict decisions

| Source | Reviewed head | Content and integration decision |
| --- | --- | --- |
| `feat/fused-native-propagation` | `a48847c4df0bf44aada6882f8d647311d5f577d4` | Actual C++17 fused propagation, explicit native PES binding, observations, production workflow integration and Python fallback; retained. |
| `feat/fused-native-propagation-20261030` | `3fa3c75cd0a2cfed7445214aad2d314bc0e2102a` | Source/PES preparation workflow only; retained its provenance artifacts in the complete validation workflow. |
| `feat/native-propagation` | `aacc0f25e9c4b3bbe5e2b5afd5577e8470301c07` | Validation workflow only; consolidated runner pinning, concurrency and source provenance, and removed the duplicate workflow and silent missing-benchmark skip. |
| `bench/native-loop-bkmp2-20261030` | `23006115beb2c6ddbdb44775c9020b7581aa4c39` | Already squash-merged by PR #1; recorded its ancestry without reverting the later main-branch benchmark notes. |

The add/add conflict in `.github/workflows/fused-native.yml` retained the full
production validation rather than the preparation-only version. The benchmark
report conflict retained the later report already present on `main`.

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

## Scientific and compatibility review

The review checked the B-C-B map, sum-beads Hamiltonian, ring-spring energy,
centroid umbrella factors, flux-metric forces, equivalent-surface min/max and
tie selection, Fortran/C memory layout, finite-result checks, state rollback,
PES locking and explicit Python fallback.

FES blocks preserve thermostat, sample and checkpoint boundaries. dRPMD blocks
preserve classifier, persistence, trajectory-output and checkpoint boundaries.
Recrossing children retain every surface timepoint and paired weights.
Constrained parent RATTLE trajectories retain the reference implementation.
Custom or overridden Python PES/CV/bias methods remain active through fallback.

This integration does not establish physical rate convergence or certify GPU
hardware execution. The permanent checks cover CPU native propagation, actual
four-rank MPI workflows, Python compatibility and real H3/BKMP2 comparisons.

## Verification and reproducibility

New deterministic regressions are in `tests/test_cabi_lifecycle.py` and
`tests/test_drpmd_propagation_resume.py`. The latter covers intermediate and
final-step restart, legacy checkpoints and classifier-error rollback, comparing
exact final coordinates/momenta, sampled trajectories, energies and drift.

The source-package audit rebuilt the unpacked extension and passed all 16 fused
tests, including the real BKMP2 flux-metric and Andersen RNG comparison.
The real-system harness is `benchmarks/h3/fused_propagation.py`; it checks
1/2/8/16/32 beads, two-dimensional FES, rate-PMF windows, recrossing children and
dRPMD stopping, plus interruption/resume.

Integration PR checks and the subsequent `main` checks provide the authoritative
results for their respective commit SHAs. The `Fused native propagation` artifact
contains exact source commit and archive, test logs, full real-system JSON,
pinned BKMP2 build provenance, sdist and wheel. The original implementation's
earlier measurements remain separately identified in
[`fused-production-2026-10-30`](../fused-production-2026-10-30/README.md).
