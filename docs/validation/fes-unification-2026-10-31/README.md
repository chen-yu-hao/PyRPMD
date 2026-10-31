# Unified FES workflow regression / 统一 FES 工作流回归

## Scope and reference

The reference checkout is `c4fa743f552c1d71db35fab791a2e63409bc1c3e`.
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

## Reproduce

Use a fresh output directory and the same Python/native build in both trees:

```bash
git worktree add --detach ../PyRPMD-baseline c4fa743f552c1d71db35fab791a2e63409bc1c3e
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
