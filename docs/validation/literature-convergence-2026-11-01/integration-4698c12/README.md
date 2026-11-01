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

## Water compatibility and portable scheduler tests

All **64 distinct water method tests** have also passed on the integrated
main runtime. The first invocation passed 62 and failed two scheduler tests
because they depended on historical local protocol/checkpoint files. That
dependency would also prevent those tests from running in a clean checkout.

A [test-only patch](../water/scheduler-tests-portability.patch) now provides
temporary source, configuration, protocol and initial-state fixtures with
their complete checksums. The real scheduler registration, manifest and
committed-prefix checks still execute; controlled subprocess failures and
history replacements remain the tested behaviors. All three scheduler
tests pass after the patch. The original test file and original failed
output remain available. No production scheduler, physical model, RNG,
sampling protocol or trajectory was changed.

See the [combined coverage receipt](water-complete-check.json),
[original report](water-method-tests.xml),
[original log](water-method-tests.log),
[successful scheduler report](water-scheduler-portable-tests.xml), and
[successful scheduler log](water-scheduler-portable-tests.log).

The other passing checks include small-system OpenMM Reference/CPU
force-map, all-copy quantum checkpoint, geometric-clock and momentum
conventions, VACF recording, estimators and corruption guards. They do not
establish long liquid equilibration, diffusion convergence or GPU validity.

The later four-worker orchestration adds **19 distinct self-contained
tests**, which also pass in the integrated checkout (0.70 seconds). Its
[JUnit report](water-scheduling-v2-tests.xml),
[log](water-scheduling-v2-tests.log), and
[source/runtime receipt](water-scheduling-v2-check.json) are separate from
the original 64 tests. The scientific water protocol and native trajectory
sources are unchanged. See the [handoff evidence](../water/scheduling-v2/README.md)
for the observed normal old-child exits, preserved prefixes and new leases.

## H + CH₄ compatibility

The [separate methane compatibility record](hch4/README.md) verifies the
actual integrated imports and native binary at execution HEAD `3ff0b67`.
All 11 standard tests, twelve native/Python comparisons, and four actual
three-step umbrella/checkpoint probes pass. The observed maximum position,
momentum and per-bead energy differences are respectively 4.09×10⁻¹⁴ bohr,
5.01×10⁻¹¹ atomic units and 7.44×10⁻¹⁵ hartree. All 202 guarded source/native
files are unchanged over the check.

The independent P128, 0.1 fs short-NVE energy gate still fails, with exactly
the same 1.8773813694×10⁻⁴ hartree-per-bead span as the original runtime.
The failed command return code and `passed=false` report are retained. The
other three original NVE spans are also exactly reproduced. Compatibility
therefore preserves the known failure instead of converting it to a pass.
These checks consume 6.66 CPU seconds and do not run a thermal-rate campaign.

## Python 3.13 and 3.14

The same **217 distinct cases** (123 H3, 64 water, 19 water-coordinator and
11 methane) also pass independently on Python 3.13.18 and 3.14.10, with
zero failures, errors or skips. Each environment builds and loads its own
three native modules and uses OpenMM 8.6.1. The original Python 3.12
experiment environment and all protected simulation sources remain unchanged.
The [complete compatibility record](python-313-314/README.md) contains
compiler and dependency identities, actual loaded binary paths/checksums,
both JUnit reports and before/after source guards.

The initial broader guard noticed a concurrently developed plotting helper;
the explicit resolution excludes only that nonphysical, separately owned
new file. Initial network/install failures are retained. Six owned, ignored
ABI symlinks remained visible in the final tool-process check despite earlier
cleanup receipts; the final retained-link record is authoritative. They are
not included in Git. This compatibility result does not imply a full-suite
rerun or a rerun of the long physical experiments.

The later [new-stage compatibility record](new-stage-compatibility/README.md)
adds 53 distinct coordinator, plotting, preparation-restart and analytic
RDF-noise checks. Python 3.12 executes 44 new cases and reuses the already
completed [nine plotting cases](water-plot-py312.xml); Python 3.13 and 3.14
each execute all 53. The case-level union is exactly the same **270 distinct
passing cases on each interpreter**, without double-counting retries.
There are zero new failures, errors or skips. No rebuild, installation,
source change or ABI-link mutation occurs during this later gate. These
software checks do not supply liquid equilibration or diffusion evidence.

The subsequent
[materials and preparation-coordinator gate](materials-preparation-compatibility/README.md)
executes 45 further distinct cases on each interpreter: ten real EAM4 static
checks and 35 orchestration fixtures. Each run has zero failures, errors or
skips. Their union with the unchanged earlier receipts contains exactly
**315 distinct passing cases on each of Python 3.12, 3.13 and 3.14**.
The earlier 270 cases were not rerun, and no model/native module or dependency
was rebuilt. This bounded compatibility gate remains separate from the
F + H₂ author's standalone static program and all long scientific campaigns.

The [P256, methane-pilot and water-tool gate](four-pair-p256-compatibility/README.md)
adds **239 distinct cases on each interpreter**, again with zero failures,
errors or skips. The exact union is now **554 distinct passing cases per
interpreter**. These comprise 17 water control/plot cases, 30 methane pilot
cases, 61 P256 parent-coordinator cases, 58 parent-binding/statistical-analysis
cases and 73 child-coordinator cases. The earlier 315 cases were not rerun.
Before/after checks cover 1,005 integrated files, 769 original-runtime files,
15 external/model files and the same six ABI links. The catalog-only source
delta is explicit. This historical 554-case stage did not include the
separate parent archive-auditor fixtures.

The subsequent [auxiliary proof gate](auxiliary-proof-compatibility-v2/README.md)
executes **42 further distinct cases on each interpreter**, with zero
failures, errors or skips. These are 20 parent-archive lifecycle cases,
16 excluded native-calibration comparator/resource cases and six descriptive
water stability cases. Their exact union with the unchanged earlier receipts
is **596 distinct passing cases on each of Python 3.12, 3.13 and 3.14**;
the earlier 554 were not rerun. This is a bounded method-test union, not a
full-project-suite result.

The [first auxiliary attempt](auxiliary-proof-compatibility/README.md) is
retained: its Python 3.12 tests passed 41 cases and skipped one because the
runner's CPU affinity excluded the required CPU 6; its session-finish audit
also failed on an assumed pytest module name. The v2 runner uses collected
module objects and the required affinity. A separate, documented
[one-line test portability fix](../water/production-stability-portability/README.md)
skips only the rendering case if optional matplotlib is absent. All three
selected environments provide matplotlib and execute all six water cases.
The original physics, benchmark sources and running checkout remain unchanged.

## Further completed software gates

The [next additive gate](fifth-stage-compatibility/README.md) passes 135 new
cases per interpreter: 43 complete-P256-analysis admissions, 85 methane
efficiency cases and 7 analytic VACF–MSD identities. The subsequent
[archive-method gate](archive-method-compatibility/README.md) passes 65 new
cases, comprising the 28 original archive contracts and 37 explicit
read-only anomaly-admission cases. All three selected Python versions have
zero failures, errors or skips. The exact case-level union is **796 distinct
passing identities per interpreter**; the earlier cases were not rerun by
these additive gates. Imports, existing ABI products, recorded sources and
dependencies are checked in each actual test process and a fresh child.

Separately, the [corrected full-project Python 3.14 run](full-suite-36ec289-py314-v2/README.md)
passes **7,285 primary cases**, with **145 skips** and **486 passing subtests**,
zero failures or errors. It tests published head `36ec289` plus one validated
synthetic wall-fixture correction, before the new additive source files were
copied. Subtests are not counted as distinct primary test identities, and
the full-suite count is not added to the overlapping method-case union.
The [initial full-suite attempt](full-suite-36ec289-py314/README.md) remains
preserved with all 17 failures and their environmental/fixture diagnoses.

The test-only [wall-fixture correction](hch4-wall-fixture-supersession/README.md)
isolates its CPU clock from the outer pytest process lifetime while preserving
the real wall timeout, child draining and retry-refusal checks. All three
interpreters execute it. Production resource accounting and original active
physics sources are unchanged. Complete remote CI logs at head `36ec289`
show that same unpatched fixture as the sole failed case in five jobs;
those historical remote failures are retained and are not relabelled as passes.

## Documentation

The changed English manual passes the strict Sphinx HTML build with
`-W --keep-going`. The [log](documentation-build.log) and
[scope/source receipt](documentation-check.json) retain the command and its
limits. Historical validation records are intentionally outside Sphinx's
include patterns; local links in the changed aggregate evidence pages are
checked separately. No untranslated-manual rebuild or full-suite rerun is
implied by this documentation check.

The later complete-200fs documentation update has a separate
[strict build receipt](documentation-200fs-stage-check.json) and
[build log](documentation-200fs-stage-build.log), preserving the earlier
check's exact source scope. Its new local evidence links are included in
the [publication audit](publication-200fs-stage-audit.json).

The complete four-pair evidence update has its own final
[strict Sphinx receipt](documentation-four-pair-stage-v2-check.json),
[build log](documentation-four-pair-stage-v2-build.log) and
[publication audit](publication-four-pair-stage-audit-v2.json).
The initial [strict build](documentation-four-pair-stage-check.json) rejected
one live manual link to a historical evidence page excluded from Sphinx.
That link now uses the repository URL; the failed build and its exact source
hashes remain preserved. The earlier receipts retain their original scopes.

The relocated historical methane design snapshot retains its original relative
links. Its [explicit original-directory mapping](publication-four-pair-stage-historical-link-context.json)
records all seven destinations and their exact hashes. The audit reports these
seven historical resolutions separately; current-document links are checked
at their actual locations. The initial rejected link check is retained.
