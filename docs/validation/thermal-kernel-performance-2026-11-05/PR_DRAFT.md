## 改动摘要

Replace JSON conversion in propagation rollback snapshots with owned native
snapshots and reusable array buffers. Parallelize independent thermal work
and provide the optional, versioned `philox-v1` stream through
`native/philox.h`. The default `mt19937_64` stream and public checkpoint format
remain unchanged.

The review repair makes thermal executors honor `Dynamics(threads=...)`,
including elementary thermostat helpers in the Python loop. Direct
NativeEnsemble calls and the native input schema accept `threads`; all default
to one. Each OpenMP region uses the executor's count, avoiding the reviewed
process-pool oversubscription. Thread counts do not affect physics identity or
restart compatibility. Stale native extensions are rejected before propagation.

The new [default-build validation](https://github.com/chen-yu-hao/PyRPMD/blob/agent/t49-thermal-kernel-performance/docs/validation/thermal-kernel-performance-2026-11-05/README.md)
uses no `-march=native`, no core binding, and no `OMP_NUM_THREADS` setting.
Four concurrent `threads=1` workers each actually use one OS thread and reach
207–230 steps/s, compared with 109–127 on the same host's baseline. Optional
Philox reaches 53.7%/52.2% of NVE at N=512/4096. Its direct native four-thread
speedup is 2.974x; default MT two-step overhead is 9.295% of native time.

The default MT PILE-L/NVE ratios remain 36.2%/47.3%, and the public Dynamics
four-thread speedup is 2.381x. These do not meet 50%/2.5x and remain explicit
limits for the maintainer's acceptance decision. The benchmark script's six
passing gates select optional Philox for throughput/direct-engine scaling;
they do not imply that the default stream or public scaling reaches those
thresholds on every machine.

## 文件清单

- `native/thermal.hpp`, `native/thermal.cpp`, new `native/philox.h`: versioned
  RNG, deterministic threading, private rollback storage and physical-state
  transfer. The review repair adds per-executor thread counts and explicit
  `num_threads` clauses; the original FMA-sensitive arithmetic is preserved.
- `src/pyrpmd/native_ensemble.py`, `propagation.py`, `block_propagation.py`:
  binary rollback and reusable transfer buffers. NativeEnsemble validates the
  thread setting, excludes it from identity, and rejects stale binaries.
- **共享注册点** `dynamics.py`: two private rollback methods. `engine.py`: RNG
  option plus the two constructor arguments needed to forward the existing
  Dynamics thread count. `input_tasks.py`: RNG preflight validation.
  `setup.py`: Philox header dependency. `docs/conf.py`: native guide inclusion.
- `src/pyrpmd/potentials/_builtin_native.py`: previously user-authorized source
  parameter cache; this review repair does not change it further.
- `benchmarks/efficiency/thermal_throughput.py`, `thermal_contract.py`,
  `thermal_statistics.py`, `thermal_kernels.cpp`, `thermal_processes.py`,
  `summarize_thermal.py`, `validate_thermal_builds.sh`: committed reproduction
  scripts, explicit thread selection, actual multi-process thread counts,
  default-build measurements and compilation/test logs.
- `tests/test_native_thermal_performance.py`, new
  `tests/test_native_thermal_threads.py`, `tests/test_builtin_native_pes.py`,
  `tests/test_integrators.py`: stochastic/restart/rollback coverage, real
  workflow thread checks and unchanged validation/identity behavior.
- `docs/native-ensembles.md`, `docs/validation/thermal-kernel-performance-2026-11-04/`
  and `docs/validation/thermal-kernel-performance-2026-11-05/`: guide, historical
  tuned-build record, current default-build record and raw validation evidence.

## 验证命令与原始 pytest 汇总行

```text
python -m pytest -q -m "not campaign"
9470 passed, 236 skipped, 1 deselected, 47 warnings, 410 subtests passed in 1758.83s (0:29:18)
```

The full suite uses `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, CPU affinity
32–47, and runs outside the socket-restricting sandbox. It starts after the
unbound performance measurements. Optional Torch/ASE/PySCF dependencies are
absent in this isolated environment; the raw log retains skip reasons.

- `PYRPMD_THERMAL_PYTHON=python PYRPMD_THERMAL_VALIDATION=<record> bash benchmarks/efficiency/validate_thermal_builds.sh`:
  OpenMP `480 passed, 1 skipped in 49.28s`; no OpenMP
  `477 passed, 4 skipped in 46.68s`; FMA
  `480 passed, 1 skipped in 46.37s`. Full compiler logs:
  [OpenMP](https://github.com/chen-yu-hao/PyRPMD/blob/agent/t49-thermal-kernel-performance/docs/validation/thermal-kernel-performance-2026-11-05/build-openmp.log),
  [no OpenMP](https://github.com/chen-yu-hao/PyRPMD/blob/agent/t49-thermal-kernel-performance/docs/validation/thermal-kernel-performance-2026-11-05/build-no-openmp.log),
  [FMA](https://github.com/chen-yu-hao/PyRPMD/blob/agent/t49-thermal-kernel-performance/docs/validation/thermal-kernel-performance-2026-11-05/build-fma.log).
- Pure Python wheel without native shared libraries:
  `40 passed, 13 skipped, 18 subtests passed in 9.21s`.
- `thermal_contract.py`: freshly generated main default-build checkpoints
  replay 118/118 bitwise at both requested 1 and 4 threads.
- Explicit 1/2/4/8-thread MT and Philox state comparisons pass in OpenMP/FMA
  builds even when the environment requests one thread. The four-process
  regression was reproduced before the repair and passes afterward.
- `thermal_throughput.py`, `thermal_processes.py`, `summarize_thermal.py`:
  33 default-build throughput measurements plus four independent process-pool
  runs; full repeats, hardware, flags, affinity and source identities are
  committed. The six selected gates pass, with the additional limits above.
- `python -m sphinx -W --keep-going -b html docs docs/_build/html`: passed.

## 未完成事项与限制

The maintainer must decide whether the reported optional-stream/direct-engine
gains meet the requested performance acceptance while default MT throughput
and public four-thread scaling remain below the stated thresholds. The
reviewer's numbers and this host's numbers are not compared across machines.

Actual OS-thread-count regressions use Linux `/proc`; portable stochastic and
validation tests remain available elsewhere. The native PIMD workflow keeps
its existing one-thread-per-trajectory limit. No GPU speedup, external-PES
thread control, cross-build/libm bitwise replay, or real-material physical
convergence is claimed. The original synthetic distribution/FMA checkpoint
evidence remains explicitly labeled as the earlier tuned-build record.

## 需要协调

The user explicitly authorized the original G2 scope extension for
`src/pyrpmd/potentials/_builtin_native.py`; no further edits to it are part of
the thread repair. The review request requires the two minimal thread
forwarding changes in `engine.py`. No additional scope authorization is pending.

T27 remains unstarted until the user explicitly confirms T49 is merged. T43
also waits for confirmation that the T16b continuation merged and, if reached,
uses only the authorized cn63 GPU 8/9.

## 需集成者处理

Add the native guide to the appropriate manual toctree, document the optional
RNG and rollback improvement in release notes, and add Salmon et al., SC'11,
DOI 10.1145/2063384.2063405 to the central bibliography. Decide any future
default-stream switch separately. The current default remains unchanged.

## AI 协助披露

OpenAI Codex implemented the change and the PR-review repair and ran the
reported validation. The repair responds to the maintainer's posted review;
final human approval and scientific acceptance have not been asserted. This
follows `docs/publication-readiness.md`.
