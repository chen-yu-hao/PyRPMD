# Python 3.13 and 3.14: integrated method compatibility

**Both interpreters passed the same 217 distinct method tests, with zero
failures, errors or skips.** The tested source is integrated commit
`c2c5ae2dce253f1407e7de25ec343ef5e1844a73`. No implementation, test selection
or tolerance was changed to obtain these results.

| Method family | Python 3.13.18 | Python 3.14.10 |
| --- | ---: | ---: |
| H3 equilibrium, deterministic oracles and recrossing methods | 123 passed | 123 passed |
| Original water reference, checkpoint and estimator methods | 64 passed | 64 passed |
| Four-worker water scheduling | 19 passed | 19 passed |
| Methane potential and native methods | 11 passed | 11 passed |
| **Total** | **217 passed; 0 skipped** | **217 passed; 0 skipped** |
| Pytest time | 26.505 s | 26.919 s |
| Native build plus test time | 36.501 s | 37.260 s |

There are 217 unique cases and 434 executions across the two interpreters.
Their complete ordered case-ID lists are identical. [selection.json](selection.json)
derives the 24 selected files from the existing compatibility receipts and
records their hashes before execution. [summary.json](summary.json) binds the
results and runtime evidence.

## Dependencies and execution

The existing isolated `unit-py313` and `unit-py314` environments retain
NumPy 2.5.4, SciPy 1.18.1, pytest 9.2.0 and their prior distributions. Only
**OpenMM 8.6.1** was added to each. The official
[PyPI release files](https://pypi.org/project/OpenMM/8.6.1/#files) provide
matching `cp313` and `cp314` manylinux_2_34_x86_64 wheels. The installs used
`--no-deps --only-binary=:all: --require-hashes`; the published checksums,
requirement file and pip installation reports are retained here.

The interpreters ran sequentially, each pinned to CPU 8, with one compiler
worker and OpenMP, OpenBLAS, MKL, NumExpr and OpenMM thread limits of one.
`PYTHONPATH=src:.` selected the integrated Python sources. The fixed real
BKMP2 and CBE/PES-2008 libraries and their build receipts were present, so
model-dependent checks did not disappear as optional skips. Small OpenMM
Reference/CPU checks executed; no new physical campaign or GPU run was launched.

All three native extensions, `_native`, `_thermal` and `_electronic`, were
built separately for each ABI in temporary workspace directories. Their
actual import paths and hashes were verified at pytest session start and
finish, and in an independently spawned interpreter. The child check used
the same environment and package lookup as the tests. Native execution was
therefore not silently replaced by an absent-extension fallback.

## Shared-state protection

The checks protected **777 integrated files, 547 original-checkout files,
and four external model/receipt files**. All protected bytes remained
unchanged. The existing integrated CPython 3.12 native binary retains SHA-256
`61dc8ec076a86176e3826415444e64428272441ae3ebe50779acc69bebd51f6f`.
The original simulation environment's complete distribution inventory also
matches its before-check snapshot.

One separately assigned, new plotting helper in the original checkout
changed during preparation. The first guard stopped before any build or
test. Its owner and root confirmed that it was outside the frozen physical
sources and selected methods; root authorized excluding only
`benchmarks/water/plot_native_campaign.py`. The initial guard, failure log,
precise difference and resolution remain available. No frozen source,
native binary or integration-source check was relaxed.

Temporary ABI-specific symlinks made the external native build products
discoverable to child interpreters. Unlink calls observed absence within
their processes, but later tool processes exposed the same six links again.
The exact owned links are Git-ignored and may remain under root's explicit
retention approval; the build products themselves stay outside the checkout.
[The final link-status receipt](abi-link-retained-status.json) supersedes a
global-cleanup interpretation of the driver's run-local flag. The CPython
3.12 files were never replaced, and no test was repeated for cleanup.

## Evidence and scope

- Python 3.13: [result](py313/compatibility.json),
  [JUnit](py313/pytest.xml), [test log](py313/pytest.log),
  [runtime imports](py313/pytest-runtime-finish.json), and
  [child imports](py313/child-import.json).
- Python 3.14: [result](py314/compatibility.json),
  [JUnit](py314/pytest.xml), [test log](py314/pytest.log),
  [runtime imports](py314/pytest-runtime-finish.json), and
  [child imports](py314/child-import.json).
- [Official wheel identities](official-openmm-wheels.json),
  [hash-pinned requirements](openmm-requirements.txt),
  [protected sources](guard-before.json),
  [preflight resolution](preflight-guard-resolution.json), and
  [executed driver](run_method_checks.py).

The initial default-sandbox pip failure is retained. A separate URI probe
identified an operating-system network denial; reviewed access then installed
the published wheels successfully. It was an environment-access failure,
not evidence of an unavailable wheel or incompatible OpenMM release.

This gate covers the specified methods on regular, GIL-enabled Linux x86_64
CPython. It is not a rerun of the complete project suite or a claim about
free-threaded Python, every operating system, GPU execution, or literature
convergence. Existing raw reports and the executed driver are evidence;
any future replay should use fresh output directories and preserve this record.
