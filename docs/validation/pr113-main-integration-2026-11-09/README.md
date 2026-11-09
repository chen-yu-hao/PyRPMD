# PR #113 current-main integration repair

Date: 2026-11-09. This record describes checks actually run in this repair, not a rerun of the historical November 4 full-suite or checkpoint campaign.

## Inputs and changes

- Original PR head: `4b3713463d1f26fcc63493165d8a414fb68c1d70`.
- Current main: `25aa08964b3acfcba8be5aa648873d052dfb38ff`.
- Independent main test repair #144: `47a87a020eb944bfdc995a300b42e73d164ccebb`, included as the second parent of the integration merge. It changes tests only.
- Merge current main into the existing PR branch with two parents; do not rewrite history. Preserve the public-module consolidation and transfer main's subsequent changes into the relocated implementations.
- Preserve warm-start ring initialization and imported hills, Andersen collision schedules and checkpoint state, native-path contract checks, and current native thermal rollback/observation implementation.
- Resolve distributed helper imports using explicit public-module dependencies and `initial_ring`.
- Correct `warm_start._metadynamics_parts` to import `resolve_metadynamics` from the retained public module rather than the removed private implementation. An initial integration test run exposed three failures through this import; the corrected run passes.
- Retain the existing OPES pure-bias registry fix and its four OPES/OPESExplore endpoint-reuse regression cases. The review finding was already fixed in the original PR head; this repair verifies and preserves it.
- #144 reconciles two stale joint quantum test expectations with #140's corrected Hamiltonian, without changing production forces or weakening tolerances. Parameterized P=1,3,5 tests independently assemble energy and finite-difference its gradient.

## Provenance

The local main source and native directories were reconstructed from repository CI source artifacts and verified against Git tree object hashes before building. The final uploaded source and native subtree hashes exactly match the local files used in the final completed test runs below (generated extensions/caches excluded):

- Main `src` tree: `996005ba367112ddfe2906f74ab18cf5aacc3c37`.
- Final candidate `src` tree: `f2c2ead1560fabf1e6affa8dcedc18187501e9cc`.
- Unchanged main/candidate `native` tree: `98f1bf564ea5779c81dcca410fbe64975941d2b8`.

An initial upload differed from the earlier tested source by two trailing newlines in each of the two distributed MPI helper files. These whitespace differences were identified by blob hashes; the local files were synchronized to the uploaded bytes and both final test groups were rerun. The source subtree now matches exactly.

Native extensions were built from retained current-main source using `python setup.py build_ext --inplace -j 4`. Generated binaries are not committed. Environment: Linux, Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, pytest 9.0.2. Tests used `PYTHONPATH=src`, `OMP_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`.

## Final completed regression results

```sh
python -m pytest -q \
  tests/test_code_structure.py tests/test_sampling_module_imports.py \
  tests/test_block_python_fast_path.py tests/test_quantum_joint_end_to_end.py
# 45 passed in 9.94s

python -m pytest -q \
  tests/test_metadynamics.py tests/test_opes.py tests/test_opes_variants.py \
  tests/test_shared_metadynamics.py tests/test_grid_metadynamics.py \
  tests/test_distributed_opes_sampler.py tests/test_metadynamics_cv_identity.py \
  tests/test_opes_initialization.py tests/test_warm_start.py \
  tests/test_andersen_schedule.py tests/test_enhanced_sampling_composition.py \
  tests/test_sampling_public_api.py tests/test_biased_sampling_interrupt.py \
  tests/test_fes_block_sampling.py tests/test_native_thermal_observer_refactor.py \
  tests/test_propagation_composition.py
# 455 passed, 1 skipped, 26 warnings in 28.11s
```

These two disjoint test groups total **500 passed, 1 skipped**. The skip is `tests/test_opes_initialization.py:133`: optional PLUMED Python wrapper unavailable. Warnings were retained. An attempted combined invocation exceeded the execution tool's 60-second limit before producing a summary; it is not counted as a completed pass. The two separate final invocations above completed successfully on the exact uploaded source.

The independent #144 file also passed against unmodified current-main production source: `8 passed in 2.65s`. Earlier candidate runs reported `37 passed` for structure/import/fast-path checks, `455 passed, 1 skipped` for the broad regression, and `15 passed` for quantum plus warm-start tests. Those earlier runs overlap the final groups and are not added to the 500-test total.

## Limits

The full repository suite, current-head remote CI, strict Sphinx build, and real multi-rank MPI execution were not completed locally in this repair. Current-head GitHub Actions results are separate; queued or running jobs are not passed jobs. Historical full-suite and checkpoint comparison evidence remains historical. This integration does not claim new performance acceptance, complete platform coverage, physical convergence, or independent human review. AI assisted implementation and validation.
