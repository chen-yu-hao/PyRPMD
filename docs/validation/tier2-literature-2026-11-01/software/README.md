# Software validation receipts

These checks exercise the final production code used in the literature
validation update to PR #63. Results are local observations; GitHub Actions
on the new commit is a separate verification.

## Full suite and source identity

The final command completed with **4,712 passed, 97 skipped, 928 warnings,
357 subtests passed**, zero failures/errors, in 302.23 seconds:

```bash
PLUMED_KERNEL=/path/to/plumed-2.9.4/lib/libplumedKernel.so \
PYRPMD_REQUIRE_PLUMED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/path/to/plumed-python:/path/to/optional-deps:/path/to/deps:src \
python -m pytest -q --durations=15 --junitxml=full-suite.xml
```

The separate dependency directories are a feature of this test environment,
not a package requirement. Install the corresponding versions listed in
[runtime.json](runtime.json) for a comparable environment. The native C++
extension was compiled with OpenMP and all three integrators available.
Real CPU TorchScript, PLUMED, ASE, tblite and PySCF were installed and used.

[full-suite.log](full-suite.log) is the complete output and
[full-suite.xml](full-suite.xml) is machine-readable JUnit evidence. JUnit
includes subtests in its testcase total, so its 5,166 entries must not be
confused with 5,166 independent top-level passes. The 357 successful subtests
and overlapping focused runs are not added to the 4,712 count.

[frozen-test-sources.json](frozen-test-sources.json) records hashes at launch.
[full-suite-source-verification.json](full-suite-source-verification.json)
verifies all 345 production/test files and numerical drivers remained
unchanged. Only a saved-data plotting helper was subsequently extended,
without being imported by tests; that exception and both hashes are recorded.
The 928 warnings are deprecations from the installed TorchScript and ASE/
NumPy versions, not suppressed project failures.

### All 97 skips

| Reason | Count |
| --- | ---: |
| Optional pinned BrHCl PES not built | 53 |
| Optional pinned H3/BKMP2 PES not built | 20 |
| Optional JaxPIP environment absent | 8 |
| Optional Caracal native source/optimizer build absent | 8 |
| This BLAS runtime cannot supply four threads | 5 |
| Optional native ONNX plugin absent | 1 |
| Pinned NiAlH reference-potential opt-in absent | 1 |
| Independent optional PMF reference solver absent | 1 |

Exact per-test reasons remain in the raw log. Some separate pre-existing
GitHub workflows exercise those source-pinned optional benchmarks, but their
results are not counted as local passes here.

An earlier test run overlapped source changes while multiprocessing parents
and children imported different OPES schemas. Its process/serial check failed
with an invalid-state-fields error. The complete superseded
[intermediate log](intermediate-suite.log) and XML are retained. The final
frozen-source run and the independent OPES review both pass that same test.

## Real PLUMED and compatibility checks

PLUMED kernel and wrapper are version 2.9.4 at source commit
`52da2bb76d37dbad0d19f59c6fe0b2ab939e3ded`, with the OPES module compiled.
The final full suite includes the live fixed-width energy/force comparison
and actual input-file PLUMED workflow tests.

The independent [OPES review](opes-independent-review.json) reports
**74 passed, zero skipped**. It covers old empty and nonempty histories,
adaptive/explore histories, schema-2 first-weight validation, process-pool
parity, completed and partial legacy resumes, and ambiguous mixed-schema
preflight rejection. This overlaps the full suite.

The live input workflow runs one- and four-bead METAD jobs. It checks one
kernel-history update per accepted MD step, invariance to diagnostics
frequency, HILLS/COLVAR counts, preparation without history mutation, and
preservation of opaque history when unsupported resume is rejected.
`PYRPMD_REQUIRE_PLUMED=1` makes a missing runtime fail rather than skip.

## Real MPI

Eight project gates passed:

| Script | MPI ranks |
| --- | ---: |
| mpi_integration.py | 4 |
| mpi_dynamic_integration.py | 5 |
| mpi_workflow_smoke.py | 4 |
| mpi_adaptive_smoke.py | 4 |
| mpi_opes_smoke.py | 4 |
| mpi_materials_smoke.py | 4 |
| mpi_surface_smoke.py | 4 |
| mpi_surface_extensions_smoke.py | 4 |

Commands use `mpiexec --oversubscribe`, Open MPI 4.1.6 and mpi4py 4.1.2.
[mpi-summary.json](mpi-summary.json) records the seven workflow batch
commands; the basic integration gate has its separate log. The final
[OPES MPI log](mpi-opes-final.log) repeats its gate after the last collective
legacy-state preflight change. Serial equality, completed resume/offline
analysis, and unchanged checkpoint hashes all pass.

The previous remote Python 3.11 CI failed after 4,618 passing tests at MPI
allocation because the old ORTE mapping environment variable did not enable
oversubscription under PRRTE. The workflow now supplies the explicit option
documented by [Open MPI](https://docs.open-mpi.org/en/v5.0.7/man-openmpi/man1/mpirun.1.html).

## Documentation and distributions

English and Chinese Sphinx HTML builds passed with warnings treated as
errors; [English](docs-en.log) and [Chinese](docs-zh.log) logs are retained.

The final native wheel was built, installed offline without dependencies
into a fresh target, and imported outside the checkout. The
[installed-wheel receipt](installed-wheel.json) verifies package origin,
new and legacy OPES first weights, an independent two-state exponential,
native OpenMP/all three integrators and the CLI. Build/install logs are
retained. Source-distribution contents are independently checked so the
benchmark drivers, reports and curated raw evidence remain available to
users installing from source.

[sdist-verification.json](sdist-verification.json) records 460 staged
project/evidence files checked against the built archive, with zero missing
files, byte mismatches or unexpected validation files. Exactly 50 curated
raw sampling NPZ files are included; local restart caches, redundant OPES
outputs and PLUMED backup files are excluded. The receipt and copied
[build log](sdist-build.log) were added after that archive was created, as
the receipt explicitly states.

A separate publication-only sampling replay omitted all excluded files and
reproduced seven PNG figures and `RESULTS.md` byte-for-byte. Seven PDF figures
were also regenerated. All 36 current OPES saved bias states matched their
saved bias values, and 82 exported JSON/JSONL records reconstructed the
original arrays exactly. The [sampling audit](../sampling/artifact_audit.json)
and raw manifest retain the underlying checks and identities.

These checks validate software behavior and packaging. Statistical and
physical convergence statements come from the separate model reports.
