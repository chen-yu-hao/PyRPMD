# Unified FES workflow regression / 统一 FES 工作流回归

## Final integration against main #44–#49

The final reference is `519f1f0221ebff945ed7e3e12f20a37f00212e89`, including
all six concurrently merged upstream PRs. **1,351 tests and 85 subtests passed,
with zero failures and zero skips** in the final scoped integration gate
(189.51 seconds). See [final JUnit](evidence/final-integration-tests.xml).
ASE 3.29.0 was installed using the declared optional dependency; the formerly
skipped six actual-ASE resume/identity cases ran in this gate. NumPy/SciPy
versions were unchanged. MPI remains unexecuted locally.

All seven unchanged examples and three actual interrupted continuations were
rerun in fresh output directories against this final main checkout:
**677 arrays and 55 checkpoints match exactly**. See
[final main comparison](evidence/final-main-comparison.json). Both checkouts
use the same native build and dependencies. English/Chinese HTML and
singlehtml strict builds all passed, with zero warnings.

The last integrations preserve actual periodic cell vectors, one-pass
materials stream validation and validated adaptive-plan caching. Current
energy-convention metadata is included in the shared fingerprint. Old
non-ASE physical identities remain compatible; historical ASE-backed energy
conventions are rejected before migration or propagation, as required by main.
The new report-cache cases cover schema 2/3, 2D/3D and frozen status per caller.

These final counts supersede earlier scoped totals below for the assembled
implementation; the earlier records document each integration stage and must
not be summed. The initial full-repository run is still reported separately,
not represented as a final full-repository pass.

## Scope and reference

The initial reference checkout is `c4fa743f552c1d71db35fab791a2e63409bc1c3e`.
That first regression predates concurrently merged upstream reliability fixes;
the later current-main comparison below separately verifies their preservation.
Both checkouts used Python 3.12, the same NumPy/SciPy/h5py environment, and
separately rebuilt C++ native extensions. Existing examples were run with
their original scientific settings. The comparison harness asserts that
each subprocess imports its selected checkout, not another editable install.

The refactoring shares CV validation, bias sampling/rollback, checkpoint
boundaries and dispatch; introduces `FESResult`; normalizes the FES entry;
and adds 3D umbrella sampling, synchronized MetaD walkers and optional saved
rate PMF sources. The regression checks preserve original sample clocks,
RNG histories and estimator arithmetic. They do not establish physical
convergence of these short examples.

## Existing examples: exact equality

| Existing input | Compared arrays | Compared checkpoints | Result |
| --- | ---: | ---: | --- |
| `examples/fes/harmonic2d.inp` | 179 | 18 | exact |
| `examples/fes/adaptive_harmonic.inp` | 87 | 8 | exact |
| `examples/fes/replica_exchange_2d.inp` | 51 | 2 | exact |
| `examples/metadynamics/standard.inp` | 23 | 1 | exact |
| `examples/metadynamics/well_tempered_2d.inp` | 37 | 2 | exact |
| `examples/metadynamics/grid_fixed_slab.inp` | 26 | 1 | exact |
| `examples/opes/harmonic_2d.inp` | 34 | 2 | exact |

The comparison includes free energies, bootstrap errors / replica spread,
masks, sample arrays, q/p, clocks, saved RNG, hill/kernel histories and
propagation counts. Only migrated fingerprints, integrity envelopes and
producer metadata are excluded. This is exact array equality with matching
NaNs, not a relaxed numerical tolerance.

## Actual old-code interruption and new-code continuation

The harness also runs the old implementation and raises an intentional
exception after reaching at least 200 integration steps. The new code then
calls public `resume_run` on that old run directory, without replacing its
checkpoint fingerprint. The resulting complete run is compared to the
uninterrupted old implementation.

| Method/input | Compared arrays | Compared checkpoints | Result |
| --- | ---: | ---: | --- |
| Umbrella `harmonic2d.inp` | 179 | 18 | exact |
| MetaD `standard.inp` | 23 | 1 | exact |
| OPES `harmonic_2d.inp` | 34 | 2 | exact |

Across the two tables, **673 array comparisons and 55 checkpoint comparisons
passed exactly**. See [machine-readable evidence](evidence/example-comparison.json).

## New functionality and stage gates

The new examples `examples/fes/harmonic3d.inp` and
`examples/metadynamics/multiple_walkers.inp` were run through public
run → resume → reanalysis → report. The 3D surface has shape `(6, 6, 6)`;
the three-walker MetaD surface has shape `(24, 24)`. Both retain identical
free energies and uncertainties after resume and reanalysis, including NaN
uncertainty for the one correlated shared-bias ensemble. See
[new-example evidence](evidence/new-examples.json).

The automated gates cover each stage:

| Stage | Validation focus |
| --- | --- |
| 1: common foundation | Native fixed-bias partial commits; CV units/dimensions; unchanged thermostat/RNG schedule; ordinary failures and asynchronous interrupts; native PILE-L and replica exchange |
| 2: common result | Uniform files and legacy adapters; metadata conflicts; unchanged bootstrap/spread arithmetic; reports and cross-method comparisons; 3D slices; projection and rate postprocessing |
| 3: common entry | Nested inputs; unknown/duplicate/cross-method rejection; old aliases and deprecation; authenticated run-manifest migration; exact legacy checkpoint compatibility |
| 4: capabilities | Independent Gaussian-cell integral check of 3D MBAR; 3D adaptive coverage; shared direct/grid walkers, real process workers, NHC/PILE-L/fixed slab, failed-block rollback; external rate-source identity/measure/reweighting checks and real parent/child propagation |

Shared walker tests deliberately use synchronization blocks that are not
multiples of deposition pace. They include failures after a hill is appended
and after both walkers have advanced, plus bitwise matching recovery of the
global grid. A one-walker shared run matches the independent algorithm.

### Test execution record

| Run | Outcome |
| --- | --- |
| Initial full repository suite | 3,851 passed; 132 skipped; 23 failures investigated; 264 subtests passed |
| Final integrated FES / report / rate-source suite | **804 passed**, 10 subtests passed, zero failures |
| Corrected public API / metadata / legacy-hash regressions | **78 passed**, zero failures |
| i-PI clients in an approved localhost-capable context | **19 passed** |
| Additional real socket wire cases | **2 passed** |

All 23 initial full-suite failures have passing targeted reruns: eight
assertions expected pre-unification task names, fingerprints or error text;
one exposed a dropped legacy `flux_axis` scalar, which was restored; fourteen
were sandbox socket permission failures before protocol execution. The same
socket failure was reproduced on the unchanged baseline before rerunning
the actual protocol tests in a supported execution context. The updated
assertions retain the old scientific rejection checks and explicitly verify
the new method-bearing identities / alias contract.

The test suites overlap, so their counts must not be added. Optional external
PESs and ONNX/Torch/ASE dependencies account for the recorded skips. The final
integrated suite includes the failure/interrupt protections added during
cross-review. See [test summary](evidence/test-summary.json),
[integrated-suite JUnit](evidence/final-fes-tests.xml) and
[corrected-regression JUnit](evidence/corrected-regressions.xml).

The local environment does not provide `mpiexec`/`mpi4py`; actual shared-walker
MPI execution was not run locally. Collective control flow and root-only
checkpoint writes were reviewed. Serial versus two real ProcessPool workers
was exercised. No MPI execution claim is made from the process tests.

## Concurrent main integration

While PR #50 was being published, main merged the state-preparation hardening
and onboarding/preflight changes from PRs #44 and #45. The PR now includes main
at `3c8fe4594c730a167b43e6069059a487bcf5c21a`, preserving both sets of changes.

The overlapping input changes exposed a semantic merge issue: treating every
canonical `task="fes"` request as an umbrella preflight would reject MetaD/OPES.
Validation now uses the common method dispatcher, and umbrella uses the new
complete static preflight for ranges, bins, sample budget and bootstrap.
Additional tests cover nested-field typo suggestions, direct resolved entry
validation and 3D preflight without constructing a potential.

After integration, the FES/report/rate-source suite plus CLI, UX, input,
preflight, resume and parallelism tests passed: **853 passed, 64 subtests
passed, 1 optional ASE skip, zero failures** (164.23 seconds). See
[post-merge JUnit](evidence/merged-fes-tests.xml). These overlap the earlier
gates; the counts are not additive.

Both enhanced-sampling guides are now included in Sphinx's published manual,
contents and language switcher. English/Chinese HTML and singlehtml builds all
passed with `-W --keep-going`, with no warnings. The four build commands follow
the repository's documentation CI; all 295 relative file links checked across
the edited Markdown guides resolve.

## Upstream adaptive reliability integration

Main subsequently merged PR #46 at
`2a4410496f0d91cb947f814745757a76e354044c`. Its normalized schema-3 adaptive
overlap and conservative bootstrap support policy are retained. Those are
upstream scientific corrections; the earlier c4fa743 comparison is the
pre-correction refactoring record, not a claim that the corrected uncertainty
policy is identical to c4fa743.

Seven unchanged examples and three actual interrupted legacy continuations
were rerun against this newer baseline. **677 arrays and 55 checkpoints match
exactly**, including the added conditional uncertainty arrays. See
[current-main comparisons](evidence/current-main-comparison.json).

The combined FES, report, rate, source, UI/MBAR, surface uncertainty, CLI,
preflight and parallelism suite passed: **929 passed, 79 subtests passed,
1 optional ASE skip, zero failures** in 230.00 seconds. See
[reliability integration JUnit](evidence/reliability-fes-tests.xml). Additional
adaptive integration (153 tests) and analysis-consumer checks (300 tests)
overlap this suite and are not added to its count. Both languages' HTML and
singlehtml strict builds passed again with no warnings.

Integration preserves primary NaNs when bootstrap draws lack support and
keeps accepted-subset errors explicitly conditional. Adaptive production
diagnostics are annotated before the common writer, so standalone NPZ,
fes.json and summary.json agree. An external rate PMF carries explicit
unavailable covariance metadata; re-referencing its curve does not silently
reuse marginal errors as errors of a difference.

## Numerical-boundary and persistence integration

PR #47 advanced main to `c328da2814a11ad9431e3b98cd8158ca6b2a1d41`.
Both native extensions were forcibly rebuilt in both checkouts. The seven
examples and three real interrupted continuations again match exactly:
**677 arrays, 55 checkpoints**, including errors and RNG/state histories.
See [boundary-audit comparison](evidence/audit-main-comparison.json).

The expanded gate, including native observation validation, dynamics boundary
checks, atomic writes, projection axes, saved-rate integrity and cached
recrossing, passed **1,178 tests and 79 subtests**, with one optional ASE skip
and zero failures (202.46 seconds). See [JUnit](evidence/audit-fes-tests.xml).
The final six nested-input and source-cache assertions were also included in
a separate [125-test entry/rate gate](evidence/audit-config-rate-tests.xml).
These counts overlap. All four strict documentation builds passed again.

The common JSON writer retains upstream's unique temporary file, fsync and
atomic replace. Projection keeps normalized axis order through FESResult,
rate readers check all archived members, and completed child checkpoints are
validated before constructing a potential. None of these integrations removes
scientific identity or archive-integrity rejection checks.

## Reproduce

Use a fresh output directory and the same Python/native build in both trees:

```bash
git worktree add --detach ../PyRPMD-baseline 519f1f0221ebff945ed7e3e12f20a37f00212e89
python setup.py build_ext --inplace
# Also run setup.py build_ext --inplace in ../PyRPMD-baseline.
python benchmarks/fes_methods/unification_regression.py \
  --baseline ../PyRPMD-baseline --output runs/fes-unification-regression
python -m pytest -q tests/test_fes_unification.py tests/test_fes_entry.py \
  tests/test_fes_result.py tests/test_shared_metadynamics.py \
  tests/test_biased_sampling_interrupt.py tests/test_rate_fes_source.py
```

## 中文说明

本次回归以 `c4fa743` 为基线，在相同依赖和原生编译条件下运行七个未修改的现有
examples。另用旧代码真实生成伞形、MetaD、OPES 的中断 checkpoint，再由新代码
从原目录续算。673 组数组、55 个 checkpoint 对比全部完全一致。

新三维伞形与共享 walker 算例均完成运行、续算、重分析和报告生成。误差字段统一为
`uncertainty`，但保留 bootstrap 与独立 replica spread 的不同含义；共享偏置的
walkers 只构成一份独立历史，因此不会产生虚假的零误差。原始 MetaD 偏置不能作为
rate PMF，外部来源必须通过测度、物理身份及重加权来源检查。

本地实跑覆盖串行与真实多进程，并不代表 MPI 已实跑或这些短算例已经物理收敛。

发布期间主线合入了入门与预检改动，本 PR 已保留并整合这些改动。合并后的扩大回归为
853 passed、64 subtests、1 项可选 ASE skip，无失败；中英文 HTML 与单页手册的
四项严格 Sphinx 构建全部通过且无警告。

主线随后合入的自适应与 bootstrap 科学修复已单独整合。与 `2a44104` 的新对照为
677 组数组、55 个 checkpoint 完全一致；相应集成套件 929 passed、79 subtests、
1 项可选 ASE skip，无失败。最初 c4fa743 的记录保留为这些上游修复之前的回归证据。

最终代码已整合主线 #44–#49：与 `519f1f0` 的 677 组数组和 55 个 checkpoint 完全
一致，最终组合套件 1,351 passed、85 subtests、零失败、零跳过。ASE 六项真实恢复
检查已补跑并通过；MPI 仍未在本地实跑。中英文 HTML 与单页手册四项严格构建通过。
