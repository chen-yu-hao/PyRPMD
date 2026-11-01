# Compatibility with the integrated main branch

GitHub recorded PR #62 as merged at 16:55 UTC on 2026-11-01. Main then
advanced to `4698c1202d0726bce7d63742e7cd31a6f137a847`, incorporating other
development branches as well. The running H3 and water calculations were
left in their original checkout: their recorded source and binary hashes
remain the authority for those datasets. Commit
`3d096d5cfc5709f14a1ad5e3d184b108eef9f20e` preserves the completed H3
methods and evidence on that original runtime tree.

The follow-up changes were applied without conflicts in a separate
checkout of the integrated main branch. No core source file or binary was
replaced in the running simulation checkout.

## H3 result

- **123 distinct method tests passed** on Python 3.12.14. The first invocation
  passed 114 and skipped nine because the external BKMP2 library was not yet
  present in the new checkout. After copying the identical pinned library
  and its build receipt, all nine skipped cases passed. Both original and
  follow-up outputs are retained; no test tolerance or implementation was
  changed to resolve those skips.
- The new main-branch native extension has SHA-256
  `61dc8ec076a86176e3826415444e64428272441ae3ebe50779acc69bebd51f6f`.
  A 400 fs excluded calibration was propagated with 0.1 and 0.05 fs steps.
  Its final coordinates, momenta and all stored observation arrays are
  **bitwise identical to the old runtime's corresponding arrays**.
- Splitting each new-runtime trajectory at 200 fs, storing state, and
  recreating the provider and integrator also reproduces the uninterrupted
  new-runtime calculation bitwise, including clocks.

These checks cover the benchmark methods, their source/checkpoint guards,
and the native propagation path used in the H3 experiment. They are not a
rerun of the full project test suite, the eight primary equilibrium chains,
or the complete recrossing campaign. Scientific claims remain tied to the
original executed source identities.

Machine-readable evidence:

- [Original compatibility check](compatibility.json)
- [Resolution of all nine optional-model skips](complete-check.json)
- [Initial test report](h3-method-tests.xml) and [retained log](h3-method-tests.log)
- [Nine model-dependent tests](model-dependent-tests.xml) and [log](model-dependent-tests.log)
- [Native restart and trajectory comparison](native-replay/native-restart-validation.json)
- [Executed compatibility driver](run_check.py)

## Reproduction

Build this checkout's extension in place, using its declared build
dependencies, and set `PYTHONPATH=src:.` so that the check loads this checkout
rather than an installed package from another revision. Set
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`. Build or supply the pinned
BKMP2 model under `benchmarks/h3/build/`, including its matching `.build.json`
receipt. The external PES source and binary are not redistributed here.

`run_check.py` takes three explicit input paths: `--library` for that
binary, `--parent` for the excluded calibration parent directory, and
`--old-replay` for the archived original native replay directory. It records
the input checksums, actual loaded runtime paths and hashes, test command,
and every compared array. The old replay is a numerical reproducibility
input; it is not a source of primary rate samples.

The first local build invocation lacked `setuptools`. Build-only
dependencies were subsequently installed into a separate directory
(`setuptools 84.2.1`, `wheel 0.48.0`, `packaging 26.3`), without changing the
running experiment environment. The new extension then built successfully.
