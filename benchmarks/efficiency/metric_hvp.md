# Matrix-free rate flux metrics (T02)

## Scope and implementation

The implementation replaces dense collective-variable Hessians in the Python
flux correction and both native propagation implementations with analytic
Hessian-vector products (HVPs). The comparison baseline is
`8cc402fb12391e4fb644d3e186e5e58d1699759a` (main after PR #77); the work branch
is `perf/t02-metric-hvp`.

For centroid coordinate $\xi(\bar q)$, let $g=\nabla\xi$,
$H=\nabla^2\xi$, and $z=g^\mathsf{T}M^{-1}g$. The existing correction is

$$
 U_{\mathrm{flux}}=-\frac{P}{\beta}
 \log\sqrt{\frac{z}{2\pi\beta}},\qquad
 F_b=\frac{H(M^{-1}g)}{\beta z},\quad b=1,\ldots,P.
$$

The force is the negative derivative of the **total ring** correction energy
with respect to one bead's coordinates. Its expression has no extra bead
factor. Positions, masses, energies and derivatives retain atomic units.

For a weighted norm $r=\|\sum_a w_a q_a\|$, define
$u=(\sum_a w_aq_a)/r$ and $t=\sum_a w_av_a$. Its HVP is

$$
 (Hv)_a=\frac{w_a}{r}\left[t-u(u^\mathsf{T}t)\right].
$$

Distances, COM distances and bond differences use these contractions. For the
rate quotient $\xi=a/b$, with $b=s_0-s_1$, the contraction is

$$
 H_\xi v=\frac{H_av}{b}
 -\frac{g_a(g_b^\mathsf{T}v)+g_b(g_a^\mathsf{T}v)+aH_bv}{b^2}
 +\frac{2a g_b(g_b^\mathsf{T}v)}{b^3}.
$$

Rate surfaces use the same linear combination as their retained Hessian API.
Min/max branch selection, first-branch tie handling and singularity thresholds
are preserved. Periodic distance, difference and angle derivatives apply within
the selected minimum-image branches; the derivative is not defined across an
image-switch seam. Affine projection coordinates have zero HVPs.

For a fixed number of CV terms and equivalent reaction paths, storage and work
are O(N). Both native headers remove their dense CV work arrays and reuse
prepared norm geometry within a metric evaluation. The Python rate/surface
fast path also prepares geometry once per evaluation. Prepared derivatives are
rebuilt for each centroid evaluation and are not checkpointed; no additional
dynamical state is introduced.

## Public API and custom CV compatibility

```python
value, gradient, hvp = cv.centroid_value_gradient_hvp(q, v)
# q and v: real, finite centroid arrays [N, 3]
# gradient and hvp: [N, 3], with hvp = H(q) @ v
```

Built-in distance, periodic distance, bond difference, periodic bond difference,
COM distance, Cartesian, surface projection, linear/fractional projection,
rate coordinate, dividing surface, angle and periodic angle support this API.
The direction is a centroid Cartesian direction, without ring scaling.
`FluxMetricBias` supplies `v = gradient / masses[:, None]`.

The dense `centroid_value_gradient_hessian(q)` APIs remain available for
diagnostics and legacy plugins. A Hessian-only custom CV explicitly uses the
old dense contraction. Existing Hessian overrides, including nested rate
surface/COM overrides, remain authoritative. A custom HVP failure propagates;
it does not silently select a different derivative path. Malformed, nonfinite
or incorrectly shaped plugin gradient/HVP arrays are rejected before NumPy
broadcasting. Newly supported custom angle/difference gradient overrides must
provide an explicit HVP or Hessian.

Native dispatch checks the new method among the supported built-in method
identities, so a custom HVP cannot be bypassed by native lowering. Descriptors,
scientific input options, run identity payloads, checkpoint schemas and RNG
state formats are unchanged. `native/cayley.cpp` is unchanged. This work does
not merge the two native CV implementations.

## Benchmark semantics

The production constrained parent does **not** use `FluxMetricBias`; the block
API rejects a simultaneous constraint and bias. The benchmark therefore labels
the following four cases separately rather than changing production physics:

| Mode | Hamiltonian and execution path | Purpose |
| --- | --- | --- |
| `parent_native` | Existing Cayley-RATTLE parent, C ABI harmonic PES, no flux bias | Production parent regression control |
| `parent_metric_python` | Benchmark-only ring Hamiltonian augmented with `FluxMetricBias`; unchanged Python `constrained_step`, native Cayley map, C ABI harmonic PES | Requested metric plus constrained-chain workload; synthetic, not the production parent measure |
| `flux_native` | Unconstrained rate-flux propagation through `native/propagation.h` | First native Hessian-to-HVP hotspot |
| `flux_fused` | Unconstrained rate-flux propagation through `native/fused.hpp` | Second native Hessian-to-HVP hotspot |

The generated C ABI provider evaluates $V=0.005\sum q^2$,
$F=-0.01q$, in atomic units. Cases use a three-atom reaction coordinate
embedded in N atoms; the COM fragment includes the remaining atoms. All
comparisons use identical initial conditions and operators within each case.

The harness launches a fresh subprocess for every sample and alternates old/new
order. An explicit source path and module-location checks select each checkout.
Wall time covers public propagation calls and their descriptor construction;
imports, compilation, initialization, warmup and state copies are excluded.
Peak RSS includes the interpreter, imported libraries and all work buffers.
It is not a count of CV allocations alone. Each sample records source hashes,
native-extension hashes, environment information and raw elapsed time.

The final result uses median seconds per step. N=3 slowdown above 5% produces
exit status 2. Failed or incomplete measurements produce exit status 1 and
remain visible in JSON; no missing value is extrapolated. A two-step smoke run
only checks execution and is too short to assess the performance gate.

## Environment and reproduction

Recorded environment: AMD EPYC 9V74; 8 CPU quota; 8 GiB memory limit; Linux;
one OpenMP/BLAS thread; four beads for the timing study. Affinity, interpreter,
NumPy and compiler details are recorded by each benchmark run. Run old and new
timings without concurrent test/build workloads.

From the candidate checkout, one way to build both revisions with the same
interpreter and dependencies is:

```bash
git worktree add --detach ../PyRPMD-hvp-baseline 8cc402fb12391e4fb644d3e186e5e58d1699759a
python -m venv ../metric-hvp-venv
../metric-hvp-venv/bin/python -m pip install -e '../PyRPMD-hvp-baseline[test]'
../metric-hvp-venv/bin/python -m pip install -e '.[test]'
```

Each checkout must retain its compiled native extension. The harness sets the
worker's `PYTHONPATH` to that checkout's `src`, and preserves the venv executable
path instead of resolving its symlink to the base interpreter.

Example complete measurement command (replace `CPU_ID` with an allowed CPU):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
../metric-hvp-venv/bin/python benchmarks/efficiency/metric_hvp.py \
  --baseline-source ../PyRPMD-hvp-baseline \
  --sizes 3 30 300 3000 --steps 2048 1024 64 2 \
  --repeats 7 --min-seconds .3 --warmup-steps 1 \
  --beads 4 --threads 1 --cpu CPU_ID --output metric-hvp.json
```

`--timeout` and `--memory-limit-mib` can bound each worker. Such limits are
recorded, and a limit failure is reported as a failed measurement. An omitted
`--cpu` leaves the worker's existing affinity unchanged.

## Numerical and regression validation

The new tests compare every HVP against retained dense Hessians where available
and independently against finite differences of analytic gradients. Metric
energies and every force component match dense references at approximately
1e-12 relative tolerance, with small absolute tolerances for near-zero values.
Coverage includes unequal masses, one/multiple beads, triclinic periodic images,
all four selected rate branches, branch ties, singularities, invalid directions,
legacy overrides and HVP-only plugins. Both native propagation paths are checked
against dense-force trajectories. Separate N=3000 subprocess checks complete
both native paths below a 1 GiB virtual-address limit.

Recorded command and literal final output after the expanded HVP cases:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_metric_hvp.py
```

```text
252 passed in 4.51s
```

The final measured source also passed the ten-file related regression set:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/workspace/scratch/9878348b1f70/venv/bin/python -m pytest -q \
  tests/test_cv_fes.py tests/test_cv_efficiency.py tests/test_materials_input_cv.py \
  tests/test_native_propagation.py tests/test_fused_propagation.py \
  tests/test_rate_workflows.py tests/test_general_coordinate_rates.py \
  tests/test_batched_recrossing.py tests/test_checkpoint.py tests/test_metric_hvp.py
```

```text
SKIPPED [1] tests/test_fused_propagation.py:120: optional real BKMP2 not built
522 passed, 1 skipped, 10 subtests passed in 57.26s
```

### Old-checkpoint experiment

The baseline checkout generated actual complete and deliberately interrupted
workflow checkpoints. The candidate then consumed copies of the old files;
the original baseline artifacts were retained. The small experiment used three
atoms, four beads, harmonic forces, Python scheduling and the native Cayley map.
The PMF case used the `rate_flux` measure and therefore exercised the changed
Python flux force during continuation.

| Workflow | Old committed interruption | Candidate final step | Comparison with uninterrupted baseline |
| --- | --- | --- | --- |
| Constrained parent | 3 | 6 | q, p, snapshots bitwise identical; maximum absolute error 0 |
| Rate-flux PMF window | 4 | 12 | q, p, values, energies bitwise identical; maximum absolute error 0 |

RNG state, step, time, elapsed count and fingerprints were exactly equal. The
candidate parent executed three additional steps. Replaying completed old PMF
and recrossing-child caches left their file bytes unchanged.

```text
Parent fingerprint:
0783a154aa2f55f4a96e23d980f54db6be55cb4bcba1f166f41ab7f976bf6bc8
Rate-flux PMF fingerprint:
88b9755ed1d50f86c5c35ae4941589f0023935954f88a3cc001c97b26a20e66e
Completed PMF SHA256 before and after cached replay:
8a4c7ba2ff50c970bee65a2640668edc6679c8a2417eccf32e5a60850277cf3b
Completed child SHA256 before and after cached replay:
2da59ed6d1f7aa9d300e95296488a2a7bb44f4b9ac34a1ad7b5ce7cc54f14e5e
```

The final-source rerun evidence is `t02-cross-version-final.log`, with
baseline/candidate JSON reports and original/copied NPZ artifacts in
`t02-cross-version-final/`. It repeated the bitwise comparisons and unchanged
hashes above after the prepared-geometry optimization.
The one-off harness `t02_cross_version_checkpoints.py` is reproduced in the final
appendix; it is not an installed PyRPMD module.
This short experiment checks serialization and continuation, not rate convergence.
The successful session command (exit status 0) was:

```bash
PYRPMD_CHECKPOINT_EXPERIMENT_OUTPUT=/workspace/scratch/9878348b1f70/t02-cross-version-final \
/workspace/scratch/9878348b1f70/venv/bin/python \
  /workspace/scratch/9878348b1f70/t02_cross_version_checkpoints.py
```

## Full-suite regression results

The same interpreter and dependencies were used for both checkouts. The full
runs used one OpenMP and BLAS thread and the normal workspace sandbox. Their
elapsed wall times include the planned suspension while isolated benchmarks
were running, so these durations are not a test-suite performance comparison.

Baseline command, from the baseline checkout:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/workspace/scratch/9878348b1f70/pyrpmd-t02-baseline/src \
/workspace/scratch/9878348b1f70/venv/bin/python -m pytest -q
```

Literal baseline summary (exit status 1):

```text
15 failed, 7979 passed, 295 skipped, 47 warnings, 486 subtests passed in 2548.82s (0:42:28)
```

Candidate command, from the candidate checkout:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/workspace/scratch/9878348b1f70/venv/bin/python -m pytest -q
```

Literal candidate summary (exit status 1):

```text
15 failed, 8231 passed, 295 skipped, 47 warnings, 486 subtests passed in 2824.98s (0:47:04)
```

The fifteen failure node IDs are exactly equal between revisions: no new or
removed failures. The candidate adds 252 passing tests; skipped, warning and
subtest counts are unchanged. This establishes the no-new-failures regression
gate in the tested environment, while the full suite remains non-green for
the environment/dependency reasons below.

### Environment-related failures and focused rechecks

The baseline full run contains thirteen `PermissionError: [Errno 1] Operation
not permitted` failures at local `AF_INET` socket creation, before the protocol
assertions run. A fourteenth socket test,
`test_existing_unix_address_is_not_removed_on_bind_failure`, instead failed with
`OSError: AF_UNIX path too long` in the deeply nested default pytest directory.
These are distinct environment issues. All fourteen exact failing node IDs
passed on both revisions with local socket permission and a short task-owned
pytest temporary directory. No source or dependencies were changed for this
recheck. The fifteenth failure is the known T01 test
`tests/test_water_preparation_plot.py::test_paired_rdf_bands_cancel_shared_chain_offsets_and_add_no_gate`,
which imports the absent optional `matplotlib` dependency. Matplotlib was absent
in both environments, and this test is unchanged.

Exact socket-test selection, usable with either checkout:

```bash
socket_nodes=(
  'tests/test_ipi_clients.py::test_multiple_clients_are_concurrent_ordered_and_persistent'
  'tests/test_ipi_clients.py::test_fast_client_takes_more_work_than_slow_client'
  'tests/test_ipi_clients.py::test_bad_client_closes_entire_pool_and_cannot_return_stale_forces[header]'
  'tests/test_ipi_clients.py::test_bad_client_closes_entire_pool_and_cannot_return_stale_forces[disconnect]'
  'tests/test_ipi_clients.py::test_bad_client_closes_entire_pool_and_cannot_return_stale_forces[truncate]'
  'tests/test_ipi_clients.py::test_missing_clients_timeout_and_close_listener'
  'tests/test_ipi_clients.py::test_existing_unix_address_is_not_removed_on_bind_failure'
  'tests/test_ipi_clients.py::test_close_while_accept_completes_does_not_leak_the_new_connection'
  'tests/test_ipi_clients.py::test_rebind_cannot_mix_cells_inside_one_batch_and_rejects_closed[bind_system]'
  'tests/test_ipi_clients.py::test_rebind_cannot_mix_cells_inside_one_batch_and_rejects_closed[set_cell]'
  'tests/test_ipi_clients.py::test_evaluate_rechecks_close_after_coordinate_preparation'
  'tests/test_ipi_clients.py::test_accept_oserror_closes_provider_and_uses_potential_error'
  'tests/test_periodic_materials.py::PeriodicPotentialTests::test_ipi_real_cell_inverse_and_virial_wire'
  'tests/test_potentials.py::SocketTests::test_ipi_protocol'
 )
# From the baseline checkout, with permission to create local sockets:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/workspace/scratch/9878348b1f70/pyrpmd-t02-baseline/src \
/workspace/scratch/9878348b1f70/venv/bin/python -m pytest -q \
  --basetemp /workspace/scratch/9878348b1f70/so "${socket_nodes[@]}"
# From the candidate checkout, with the same socket permission:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/workspace/scratch/9878348b1f70/pyrpmd-t02/src \
/workspace/scratch/9878348b1f70/venv/bin/python -m pytest -q \
  --basetemp /workspace/scratch/9878348b1f70/sn "${socket_nodes[@]}"
```

Literal baseline socket recheck output (exit status 0):

```text
14 passed in 4.57s
```

Literal candidate socket recheck output (exit status 0):

```text
14 passed in 3.99s
```

## Measured performance and peak memory

Both formal timing runs pinned workers to CPU 8, used five repeats, four beads,
one OpenMP/BLAS thread, one warmup step and a 0.3-second minimum timed duration.
The task's pytest processes were suspended during measurement. Per-pass step
counts were 2048, 1024, 64 and 2 for N=3, 30, 300 and 3000 respectively; short
passes were repeated from the same initialized state to reach the duration.
The large-system run imposed a 6500 MiB virtual-address limit per worker,
separate from the shared container's 8 GiB resident-memory limit.

The measured candidate commit was
`ea2242da08597d0ee5f6611f1898e7417b88eb69`, tree
`d72ef68bddf935aa41eda2091bd8973af1802ea8`. The published GitHub commit
`881a84d516e22ff09700a83117f8cde5f01c34bc` has the identical source tree.
Complete per-file and native-extension hashes are recorded in the appendix;
the source/extension hashes were identical across all successful samples of
each revision. Subsequent report-only commits do not change the measured code.

Exact commands, from the candidate checkout:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/workspace/scratch/9878348b1f70/venv/bin/python benchmarks/efficiency/metric_hvp.py \
  --baseline-source /workspace/scratch/9878348b1f70/pyrpmd-t02-baseline \
  --sizes 3 --repeats 5 --min-seconds .3 --cpu 8 \
  --work-dir /workspace/scratch/9878348b1f70 \
  --output /workspace/scratch/9878348b1f70/t02-benchmark-n3-quiet.json
# Exit status: 0

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/workspace/scratch/9878348b1f70/venv/bin/python benchmarks/efficiency/metric_hvp.py \
  --baseline-source /workspace/scratch/9878348b1f70/pyrpmd-t02-baseline \
  --sizes 30 300 3000 --repeats 5 --min-seconds .3 --cpu 8 \
  --memory-limit-mib 6500 --work-dir /workspace/scratch/9878348b1f70 \
  --output /workspace/scratch/9878348b1f70/t02-benchmark-large-quiet.json
# Exit status: 1 (incomplete old-version measurements under memory pressure)
```

Times below are median microseconds per step; peak RSS is the maximum among
successful workers. Successful sample counts are shown as old/new. A dagger
marks a summary of surviving samples only, not a complete benchmark result.

| N | Mode | Old us/step | New us/step | Speedup | Old peak MiB | New peak MiB | Samples old/new |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | `parent_native` | 26.012 | 27.139 | 0.958x | 46.641 | 46.637 | 5/5 : 5/5 |
| 3 | `parent_metric_python` | 2,837.684 | 1,758.697 | 1.614x | 47.105 | 46.520 | 5/5 : 5/5 |
| 3 | `flux_native` | 11.146 | 8.744 | 1.275x | 45.859 | 46.008 | 5/5 : 5/5 |
| 3 | `flux_fused` | 12.494 | 7.817 | 1.598x | 45.582 | 45.379 | 5/5 : 5/5 |
| 30 | `parent_native` | 87.043 | 80.611 | 1.080x | 46.543 | 46.145 | 5/5 : 5/5 |
| 30 | `parent_metric_python` | 8,706.154 | 5,873.988 | 1.482x | 47.234 | 46.996 | 5/5 : 5/5 |
| 30 | `flux_native` | 1,049.888 | 70.745 | 14.840x | 44.328 | 45.949 | 5/5 : 5/5 |
| 30 | `flux_fused` | 485.721 | 50.071 | 9.701x | 45.895 | 45.492 | 5/5 : 5/5 |
| 300 | `parent_native` | 479.821 | 482.545 | 0.994x | 47.000 | 47.133 | 5/5 : 5/5 |
| 300 | `parent_metric_python` | 229,077.346 | 6,600.659 | 34.705x | 85.676 | 46.699 | 5/5 : 5/5 |
| 300 | `flux_native` | 177,128.582 | 403.334 | 439.161x | 71.062 | 46.488 | 5/5 : 5/5 |
| 300 | `flux_fused` | 57,024.123 | 460.227 | 123.904x | 64.781 | 46.418 | 5/5 : 5/5 |
| 3000 | `parent_native` | 5,601.160 | 5,893.136 | 0.950x | 51.824 | 52.543 | 5/5 : 5/5 |
| 3000 | `parent_metric_python` | unknown (SIGKILL) | 32,743.164 | not assigned | unknown | 46.812 | 0/5 : 5/5 |
| 3000 | `flux_native` | 16,814,516.926 † | 8,149.318 | not assigned | 2519.418 † | 47.789 | 4/5 : 5/5 |
| 3000 | `flux_fused` | 5,163,055.069 | 9,104.793 | 567.070x | 1901.395 | 47.984 | 5/5 : 5/5 |

All four N=3 median gates passed. The production parent control slowed by
4.33%; the augmented Python parent, native block flux and fused flux improved
by 38.02%, 21.55% and 37.44% respectively. Timing spread was substantial on this
shared host: the production parent control ranged from 14.698 to 41.981 us/step
before, and 12.461 to 57.610 us/step after. These medians pass the requested gate
in this run; they are not a statistical upper confidence bound on slowdown.

The N=3000 old Python metric parent returned no measurements: all five workers
were terminated by SIGKILL. The old native block path completed four samples
and had one further SIGKILL. These failures coincided with cgroup OOM evidence:
`memory.events` showed `oom_kill=5` after the Python attempts and `oom_kill=6`
after the later native block failure. These are aggregate snapshots, not
counter deltas or unique per-process attribution; no reliable pre-run zero
snapshot was recorded. Killed workers returned neither timings nor peak RSS; those values
are unknown and no speedup is assigned to either incomplete case. The reported
old native block median and RSS describe only its four survivors. The large
run therefore remains **incomplete (exit 1)** despite all 60 candidate samples
finishing successfully. Memory pressure in a shared cgroup is not a measurement
of a killed worker's intrinsic memory requirement.

The N=3000 fused case completed all five old and new measurements: its median
fell from 5.163 seconds to 9.105 milliseconds per step (567.07x), and peak process
RSS fell from 1901.395 to 47.984 MiB. The unchanged-Hamiltonian production parent
is a regression control; this work does not claim an HVP speedup for that case.

As a supplemental check, 74 corresponding successful old/new sample pairs had
maximum endpoint differences of 1.7763568394002505e-15 bohr in q and
5.81756864903582e-14 atomic units in p. These stored endpoint diagnostics cover
**only the first bead and first three reactive atoms**, not the entire N-atom
state. All production-parent endpoint diagnostics were bitwise equal. The unit
tests above supply the independent full-component checks. The appendix retains
the exact timing/RSS samples and failures used for these summaries.

No physical convergence is inferred from unit tests, continuation checks or
timings. No new bibliography entries are required: the formulas preserve the
existing RPMDrate/Caracal attribution in `cv.py`, `bias.py` and
`THIRD_PARTY_NOTICES.md`.

### Suggested documentation integration

A later README update can link this report from the rate capability overview:
"Built-in rate-flux mass-metric forces use analytic Hessian-vector products with
linear atom-count storage and work; legacy custom Hessians remain supported."
Suggested Chinese wording: "内置 CV 的 rate-flux 质量度规力采用解析 Hessian–向量积，
其计算量与工作内存随原子数线性增长；自定义 CV 仍兼容旧 Hessian 接口。"
The corresponding implementation sentence in `docs/science.md` can change from
constructing the analytic centroid Hessian to applying its analytic HVP, while
retaining the existing metric formula and references. Neither update changes
the ensemble or any scientific parameter.

## Audit appendix: provenance and raw samples

The following JSON de-duplicates metadata shared by all workers. It preserves
the original numeric timing/RSS values for every requested sample, including
failed workers. Raw report hashes identify the original detailed JSON files.
The time stamps below are the recorder's UTC time stamps, copied unchanged.

```json
{
  "format": "PyRPMD.metric-hvp-audit.v1",
  "units": {
    "time": "seconds",
    "rss": "MiB",
    "position": "bohr",
    "momentum": "atomic units"
  },
  "hardware": {
    "platform": "Linux-6.18.57-x86_64-with-glibc2.39",
    "cpu_model": "AMD EPYC 9V74 80-Core Processor",
    "logical_cpus": 9,
    "affinity": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8
    ]
  },
  "container_limits": {
    "cpu_quota": 8,
    "memory_bytes": 8589934592
  },
  "compiler": "cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0",
  "beads": 4,
  "threads": 1,
  "cpu": 8,
  "repeats": 5,
  "warmup_steps": 1,
  "min_seconds": 0.3,
  "timeout_seconds": 300.0,
  "sources": {
    "baseline": {
      "source": {
        "root": "/workspace/scratch/9878348b1f70/pyrpmd-t02-baseline",
        "commit": "8cc402fb12391e4fb644d3e186e5e58d1699759a",
        "sha256": {
          "native/fused.hpp": "50b35e0eb186cdfc523c793984cc038a53cf618897f7f05627484c6df7c3e6cd",
          "native/propagation.h": "278e83ea3704c846ba8afc49940448d8afc2963b1973de1ceb656a1c755bd639",
          "src/pyrpmd/cv.py": "640d9ee61d76f847e0260acb2456723ff76c1b9fd11dd61dcdf16de3fc58a114",
          "src/pyrpmd/cv_periodic.py": "6f43a669405102325e4f2bfad9da8000b2e59813669f9428eb99f737cdc678d2",
          "src/pyrpmd/bias.py": "0a0ab4a074da0be7626c1519142ab5216999f54cc21f91a50b55e58145a38cb1",
          "src/pyrpmd/propagation.py": "0111120950a6b3defb6c199e799c32bb5e062310bde5f1c2b16248aa41d7131e"
        }
      },
      "python": "3.12.14 (main, Aug 25 2026, 14:00:49) [Clang 22.1.3 ]",
      "numpy": "2.5.4",
      "native_extension": "/workspace/scratch/9878348b1f70/pyrpmd-t02-baseline/src/pyrpmd/_native.cpython-312-x86_64-linux-gnu.so",
      "native_extension_sha256": "fe4361842cca57bfd71a8b0fb2437b5076aff2b48cdc6f524d141aac0e4b58d4",
      "affinity": [
        8
      ],
      "tree": "b57babbfd426d4c78cbb9b2f38b758478568fcb9"
    },
    "candidate": {
      "source": {
        "root": "/workspace/scratch/9878348b1f70/pyrpmd-t02",
        "commit": "ea2242da08597d0ee5f6611f1898e7417b88eb69",
        "sha256": {
          "native/fused.hpp": "6dc5e14bcb04d2f694599285a33f3d542d922b1b1f0b77b691dca6b290c9d506",
          "native/propagation.h": "8fd1d08e0198482939d4092189194b51a94b70d6551d309bf568f6cf55e6a946",
          "src/pyrpmd/cv.py": "756963253456901cb640e53aa00d2e90b127104c4964d8b83ea3d9e6bf733c08",
          "src/pyrpmd/cv_periodic.py": "7a1e8e321690957bb0f8b3f9438993514f3ab7ffa6fb4ca0e71804cc3e3e44b0",
          "src/pyrpmd/bias.py": "469420691c535c1fc3bb38cc89973e514b61300d678e565b648c2e4e39a32777",
          "src/pyrpmd/propagation.py": "a49a97d812cf9621f762ed8c49630424b1c8833a044f72045c719f6ec2141579"
        }
      },
      "python": "3.12.14 (main, Aug 25 2026, 14:00:49) [Clang 22.1.3 ]",
      "numpy": "2.5.4",
      "native_extension": "/workspace/scratch/9878348b1f70/pyrpmd-t02/src/pyrpmd/_native.cpython-312-x86_64-linux-gnu.so",
      "native_extension_sha256": "6a8656f291f99d66651afbbb5f7768d13b9ecebdd6ef4f07447d9bc230fd9c2d",
      "affinity": [
        8
      ],
      "equivalent_github_commit": "881a84d516e22ff09700a83117f8cde5f01c34bc",
      "tree": "d72ef68bddf935aa41eda2091bd8973af1802ea8"
    }
  },
  "raw_report_sha256": {
    "t02-benchmark-n3-quiet.json": "d2f50f83e3878e44f9781e0e2a428b43b6c5660e51958ea1c078f5751cdda3d7",
    "t02-benchmark-large-quiet.json": "39439f3f683ee5e25fb65e5c85accaf69787b93b8fdc1c73a6f624897c1420da"
  },
  "runs": [
    {
      "created_at_utc": "2026-11-02T16:03:02.125534+00:00",
      "command": [
        "/workspace/scratch/9878348b1f70/venv/bin/python",
        "benchmarks/efficiency/metric_hvp.py",
        "--baseline-source",
        "/workspace/scratch/9878348b1f70/pyrpmd-t02-baseline",
        "--sizes",
        "3",
        "--repeats",
        "5",
        "--min-seconds",
        ".3",
        "--cpu",
        "8",
        "--work-dir",
        "/workspace/scratch/9878348b1f70",
        "--output",
        "/workspace/scratch/9878348b1f70/t02-benchmark-n3-quiet.json"
      ],
      "memory_limit_mib": 0,
      "exit_status": 0
    },
    {
      "created_at_utc": "2026-11-02T16:06:58.566792+00:00",
      "command": [
        "/workspace/scratch/9878348b1f70/venv/bin/python",
        "benchmarks/efficiency/metric_hvp.py",
        "--baseline-source",
        "/workspace/scratch/9878348b1f70/pyrpmd-t02-baseline",
        "--sizes",
        "30",
        "300",
        "3000",
        "--repeats",
        "5",
        "--min-seconds",
        ".3",
        "--cpu",
        "8",
        "--memory-limit-mib",
        "6500",
        "--work-dir",
        "/workspace/scratch/9878348b1f70",
        "--output",
        "/workspace/scratch/9878348b1f70/t02-benchmark-large-quiet.json"
      ],
      "memory_limit_mib": 6500,
      "exit_status": 1
    }
  ],
  "oom_snapshots": {
    "after_python_n3000_failures": {
      "memory_current_bytes": 8586473472,
      "oom": 16,
      "oom_kill": 5
    },
    "after_later_native_block_failure": {
      "oom_kill": 6
    },
    "qualification": "Aggregate cgroup snapshots, not counter deltas or per-process attribution; no reliable pre-run zero snapshot."
  },
  "completed_final_checkpoint_rerun_log_sha256": "ec7572e37c535d211e0cd598e8d3fec8348ef76f683ade080cbbe899d2761479",
  "physical_converged": false
}
```

The compact sample arrays use the column order below. An array represents a
successful worker; a failure is the original failure object. Time is in seconds,
RSS in MiB. `peak_rss_before_propagation_mib` is a process high-water mark after
initialization; its difference from the later high-water mark is an incremental
diagnostic, not an allocator-level memory measurement. Every successful sample
used one warmup step; `steps_per_pass` is listed for each case. Samples appear in
repeat order, with old/new execution order alternating on successive repeats.
The endpoint summary is omitted when no corresponding pair succeeded.

```json
{
  "sample_columns": ["seconds", "timed_steps", "seconds_per_step", "passes", "peak_rss_mib", "peak_rss_before_propagation_mib", "peak_rss_increase_mib", "constraint_residual"],
  "cases": [
    {
      "natoms": 3, "mode": "parent_native", "steps_per_pass": 2048,
      "first_bead_first_three_atoms_endpoint_comparison": {"matched_successful_repeats":5,"max_abs_q":0.0,"max_abs_p":0.0},
      "small_system_slowdown_fraction": 0.043331740001730346, "small_system_5_percent_gate_passed": true,
      "baseline": [
        [0.3439063199912198,8192,4.19807519520532e-05,4,46.4453125,46.05859375,0.38671875,1.3877787807814457e-16],
        [0.3240793599979952,8192,3.956046874975527e-05,4,44.22265625,43.84765625,0.375,1.3877787807814457e-16],
        [0.3196357680062647,12288,2.6012025391134824e-05,6,46.0234375,45.6484375,0.375,1.3877787807814457e-16],
        [0.3332215500122402,16384,2.033822937086427e-05,8,46.640625,45.9921875,0.6484375,1.3877787807814457e-16],
        [0.301017696983763,20480,1.4698129735535304e-05,10,44.1953125,43.9453125,0.25,1.3877787807814457e-16]
      ],
      "candidate": [
        [0.3539565390019561,6144,5.7610113769849626e-05,3,46.5078125,45.984375,0.5234375,1.3877787807814457e-16],
        [0.3417386180080939,10240,3.337291191485292e-05,5,44.80859375,44.296875,0.51171875,1.3877787807814457e-16],
        [0.33348614200076554,12288,2.7139171712301884e-05,6,43.91796875,43.66796875,0.25,1.3877787807814457e-16],
        [0.31428001098538516,12288,2.5576172768992933e-05,6,44.88671875,44.5,0.38671875,1.3877787807814457e-16],
        [0.3062388830148848,24576,1.2460892049759309e-05,12,46.63671875,46.22265625,0.4140625,1.3877787807814457e-16]
      ]
    },
    {
      "natoms": 3, "mode": "parent_metric_python", "steps_per_pass": 2048,
      "first_bead_first_three_atoms_endpoint_comparison": {"matched_successful_repeats":5,"max_abs_q":1.7763568394002505e-15,"max_abs_p":5.81756864903582e-14},
      "small_system_slowdown_fraction": -0.3802349582253173, "small_system_5_percent_gate_passed": true,
      "baseline": [
        [9.144432069006143,2048,0.004465054721194406,1,47.03515625,46.26171875,0.7734375,1.5959455978986625e-16],
        [6.252215307998995,2048,0.003052839505858884,1,46.81640625,46.03125,0.78515625,1.5959455978986625e-16],
        [5.811577011001646,2048,0.0028376840874031473,1,46.9453125,46.1484375,0.796875,1.5959455978986625e-16],
        [2.349233896995429,2048,0.0011470868637672993,1,46.94921875,46.14453125,0.8046875,1.5959455978986625e-16],
        [2.862841327005299,2048,0.001397871741701806,1,47.10546875,46.33203125,0.7734375,1.5959455978986625e-16]
      ],
      "candidate": [
        [9.395491776005656,2048,0.004587642468752762,1,46.21875,45.71875,0.5,6.071532165918825e-16],
        [5.742472603000351,2048,0.0028039417006837652,1,46.51953125,45.89453125,0.625,6.071532165918825e-16],
        [3.60181226900022,2048,0.0017586973969727637,1,46.0546875,45.6796875,0.375,6.071532165918825e-16],
        [2.7939555660050246,2048,0.001364236116213391,1,44.3515625,43.9765625,0.375,6.071532165918825e-16],
        [2.645251735004422,2048,0.0012916268237326278,1,46.36328125,45.86328125,0.5,6.071532165918825e-16]
      ]
    },
    {
      "natoms": 3, "mode": "flux_native", "steps_per_pass": 2048,
      "first_bead_first_three_atoms_endpoint_comparison": {"matched_successful_repeats":5,"max_abs_q":1.3877787807814457e-17,"max_abs_p":5.551115123125783e-16},
      "small_system_slowdown_fraction": -0.21551469030434245, "small_system_5_percent_gate_passed": true,
      "baseline": [
        [0.30680127999949036,51200,5.9922124999900465e-06,25,45.640625,45.25390625,0.38671875,null],
        [0.30471202499757055,43008,7.085008021706904e-06,21,43.79296875,43.26953125,0.5234375,null],
        [0.3195796830113977,28672,1.1146054792529217e-05,14,45.859375,45.42578125,0.43359375,null],
        [0.3275743959893589,10240,3.1989687108335826e-05,5,45.53125,44.8828125,0.6484375,null],
        [0.31847995000862284,14336,2.2215398298592553e-05,7,43.671875,43.1484375,0.5234375,null]
      ],
      "candidate": [
        [0.30366007999691647,67584,4.493076467757405e-06,33,43.8828125,43.34375,0.5390625,null],
        [0.30161265702190576,73728,4.09088347740215e-06,36,45.72265625,45.18359375,0.5390625,null],
        [0.3044281880138442,34816,8.743916245802051e-06,17,43.74609375,43.234375,0.51171875,null],
        [0.32104747199628036,22528,1.425104190324398e-05,11,46.0078125,45.6328125,0.375,null],
        [0.3230624520147103,24576,1.3145444824817313e-05,12,45.91015625,45.48046875,0.4296875,null]
      ]
    },
    {
      "natoms": 3, "mode": "flux_fused", "steps_per_pass": 2048,
      "first_bead_first_three_atoms_endpoint_comparison": {"matched_successful_repeats":5,"max_abs_q":1.3877787807814457e-17,"max_abs_p":5.551115123125783e-16},
      "small_system_slowdown_fraction": -0.3743709198308397, "small_system_5_percent_gate_passed": true,
      "baseline": [
        [0.3098625669881585,20480,1.5130008153718676e-05,10,43.5,43.0,0.5,null],
        [0.30765788800636074,26624,1.155565985600814e-05,13,44.0390625,43.6640625,0.375,null],
        [0.3248406190032256,28672,1.1329541678404911e-05,14,45.58203125,45.0234375,0.55859375,null],
        [0.30705877798754955,24576,1.2494253661602764e-05,12,45.38671875,45.00390625,0.3828125,null],
        [0.30132204000256024,18432,1.6347766927222234e-05,9,44.09765625,43.72265625,0.375,null]
      ],
      "candidate": [
        [0.3015769839985296,38912,7.750230879896422e-06,19,43.63671875,42.984375,0.65234375,null],
        [0.30749653399107046,36864,8.341377332657077e-06,18,45.37890625,45.00390625,0.375,null],
        [0.32371096400311217,55296,5.854147931190541e-06,27,45.296875,44.921875,0.375,null],
        [0.3065797310046037,36864,8.316507459977313e-06,18,44.2265625,43.84375,0.3828125,null],
        [0.3041660929811769,38912,7.8167684257087e-06,19,44.14453125,43.76953125,0.375,null]
      ]
    },
    {
      "natoms": 30, "mode": "parent_native", "steps_per_pass": 1024,
      "first_bead_first_three_atoms_endpoint_comparison": {"matched_successful_repeats":5,"max_abs_q":0.0,"max_abs_p":0.0},
      "baseline": [
        [0.31413772700034315,4096,7.669378100594315e-05,4,46.54296875,46.15625,0.38671875,2.6020852139652106e-17],
        [0.3866663500011782,4096,9.44009643557564e-05,4,44.140625,43.890625,0.25,2.6020852139652106e-17],
        [0.3962478200046462,4096,9.674019043082183e-05,4,44.671875,44.15234375,0.51953125,2.6020852139652106e-17],
        [0.34048185900610406,4096,8.312545385891212e-05,4,44.24609375,43.7265625,0.51953125,2.6020852139652106e-17],
        [0.35652828300226247,4096,8.704303784234924e-05,4,46.484375,46.09765625,0.38671875,2.6020852139652106e-17]
      ],
      "candidate": [
        [0.34943946800922276,4096,8.531237011943915e-05,4,46.14453125,45.734375,0.41015625,2.6020852139652106e-17],
        [0.38433197799895424,5120,7.506483945292075e-05,5,44.2265625,43.84375,0.3828125,2.6020852139652106e-17],
        [0.3396016100014094,4096,8.291054931675035e-05,4,44.390625,43.8828125,0.5078125,2.6020852139652106e-17],
        [0.3005412200000137,4096,7.337432128906585e-05,4,44.63671875,44.25,0.38671875,2.6020852139652106e-17],
        [0.3301844040033757,4096,8.061142675863664e-05,4,44.7890625,44.27734375,0.51171875,2.6020852139652106e-17]
      ]
    },
    {
      "natoms": 30, "mode": "parent_metric_python", "steps_per_pass": 1024,
      "first_bead_first_three_atoms_endpoint_comparison": {"matched_successful_repeats":5,"max_abs_q":8.881784197001252e-16,"max_abs_p":2.3037127760972e-15},
      "baseline": [
        [8.60882994900021,1024,0.008407060497070518,1,47.234375,46.203125,1.03125,-1.682681771697503e-16],
        [8.287235381998471,1024,0.008093003302732882,1,46.66796875,45.79296875,0.875,-1.682681771697503e-16],
        [9.200519261998124,1024,0.008984882091795043,1,45.43359375,44.28515625,1.1484375,-1.682681771697503e-16],
        [8.915101409998897,1024,0.008706153720702048,1,45.5625,44.4140625,1.1484375,-1.682681771697503e-16],
        [11.738323721001507,1024,0.011463206758790534,1,44.93359375,43.93359375,1.0,-1.682681771697503e-16]
      ],
      "candidate": [
        [2.99102085299819,1024,0.002920918801756045,1,46.99609375,46.34765625,0.6484375,4.978656376053436e-16],
        [3.468489500999567,1024,0.0033871967783198897,1,44.48828125,43.98828125,0.5,4.978656376053436e-16],
        [6.01496323000174,1024,0.005873987529298574,1,44.33203125,43.95703125,0.375,4.978656376053436e-16],
        [6.480555597998318,1024,0.006328667576170233,1,46.87109375,46.22265625,0.6484375,4.978656376053436e-16],
        [8.705855856998824,1024,0.008501812360350414,1,44.34375,43.96875,0.375,4.978656376053436e-16]
      ]
    },
    {
      "natoms": 30, "mode": "flux_native", "steps_per_pass": 1024,
      "first_bead_first_three_atoms_endpoint_comparison": {"matched_successful_repeats":5,"max_abs_q":1.0408340855860843e-17,"max_abs_p":5.238864897449957e-16},
      "baseline": [
        [1.0979373709997162,1024,0.0010722044638669104,1,44.171875,43.671875,0.5,null],
        [1.2637509880005382,1024,0.0012341318242192756,1,43.734375,43.078125,0.65625,null],
        [1.075085640004545,1024,0.0010498883203169385,1,44.00390625,43.62890625,0.375,null],
        [0.8397668569959933,1024,0.0008200848212851497,1,43.90234375,43.2578125,0.64453125,null],
        [0.8400808820006205,1024,0.0008203914863287309,1,44.328125,43.828125,0.5,null]
      ],
      "candidate": [
        [0.41849424300016835,4096,0.00010217144604496298,4,45.875,45.4921875,0.3828125,null],
        [0.36117562399886083,4096,8.817764257784688e-05,4,45.83203125,45.58203125,0.25,null],
        [0.36221347500395495,5120,7.074481933670995e-05,5,43.984375,43.6015625,0.3828125,null],
        [0.31420934998459416,6144,5.1140844724055036e-05,6,45.94921875,45.57421875,0.375,null],
        [0.35075333200074965,6144,5.708875846366368e-05,6,43.96875,43.71875,0.25,null]
      ]
    },
    {
      "natoms": 30, "mode": "flux_fused", "steps_per_pass": 1024,
      "first_bead_first_three_atoms_endpoint_comparison": {"matched_successful_repeats":5,"max_abs_q":1.0408340855860843e-17,"max_abs_p":5.238864897449957e-16},
      "baseline": [
        [0.46699152899964247,1024,0.00045604641503871335,1,45.49609375,44.859375,0.63671875,null],
        [0.5558929460021318,1024,0.0005428642050802068,1,44.2109375,43.5859375,0.625,null],
        [0.5184263910050504,1024,0.0005062757724658695,1,45.89453125,45.51953125,0.375,null],
        [0.4181386510026641,1024,0.0004083385263697892,1,43.796875,43.296875,0.5,null],
        [0.4973785619949922,1024,0.00048572125194823457,1,45.65625,44.984375,0.671875,null]
      ],
      "candidate": [
        [0.3486498569982359,6144,5.674639599580663e-05,6,44.0390625,43.7890625,0.25,null],
        [0.36263080399658065,5120,7.082632890558216e-05,5,44.05859375,43.80859375,0.25,null],
        [0.3332092370110331,8192,4.067495568982338e-05,8,45.4609375,44.90625,0.5546875,null],
        [0.34426429799350444,7168,4.8027943358468815e-05,7,45.4921875,44.984375,0.5078125,null],
        [0.3589070699890726,7168,5.007074079088625e-05,7,44.046875,43.671875,0.375,null]
      ]
    },
    {
      "natoms": 300, "mode": "parent_native", "steps_per_pass": 64,
      "first_bead_first_three_atoms_endpoint_comparison": {"matched_successful_repeats":5,"max_abs_q":0.0,"max_abs_p":0.0},
      "baseline": [
        [0.31158413502271287,448,0.0006955003013899841,7,46.84375,46.05078125,0.79296875,-7.45931094670027e-17],
        [0.3020721950015286,448,0.0006742682924141263,7,44.578125,43.953125,0.625,-7.45931094670027e-17],
        [0.31345256300846813,704,0.0004452451179097559,11,45.06640625,44.41015625,0.65625,-7.45931094670027e-17],
        [0.3070854990000953,640,0.0004798210921876489,10,45.2421875,44.60546875,0.63671875,-7.45931094670027e-17],
        [0.31512525599100627,704,0.00044762110225995207,11,47.0,46.33984375,0.66015625,-7.45931094670027e-17]
      ],
      "candidate": [
        [0.3367928660081816,512,0.0006577985664222297,8,46.578125,45.953125,0.625,-7.45931094670027e-17],
        [0.31840043100964976,320,0.0009950013469051556,5,46.34375,45.84375,0.5,-7.45931094670027e-17],
        [0.3364405739994254,768,0.0004380736640617518,12,46.890625,46.10546875,0.78515625,-7.45931094670027e-17],
        [0.30882868200569646,640,0.0004825448156339007,10,44.6640625,44.0390625,0.625,-7.45931094670027e-17],
        [0.3000052789793699,896,0.00033482732028947533,14,47.1328125,46.21875,0.9140625,-7.45931094670027e-17]
      ]
    },
    {
      "natoms": 300, "mode": "parent_metric_python", "steps_per_pass": 64,
      "first_bead_first_three_atoms_endpoint_comparison": {"matched_successful_repeats":5,"max_abs_q":0.0,"max_abs_p":2.220446049250313e-16},
      "baseline": [
        [15.75403349300177,64,0.24615677332815267,1,85.67578125,47.96484375,37.7109375,-1.7520707107365752e-16],
        [14.931834300994524,64,0.23330991095303943,1,81.91796875,44.0625,37.85546875,-1.7520707107365752e-16],
        [14.660950169993157,64,0.22907734640614308,1,83.92578125,46.046875,37.87890625,-1.7520707107365752e-16],
        [9.602416556001117,64,0.15003775868751745,1,82.265625,44.29296875,37.97265625,-1.7520707107365752e-16],
        [3.935035747002985,64,0.06148493354692164,1,83.7265625,45.78515625,37.94140625,-1.7520707107365752e-16]
      ],
      "candidate": [
        [0.38791058200149564,64,0.006061102843773369,1,46.69921875,46.02734375,0.671875,-1.7520707107365752e-16],
        [0.4340369909987203,64,0.0067818279843550044,1,46.2109375,45.8359375,0.375,-1.7520707107365752e-16],
        [0.42244214600214036,64,0.006600658531283443,1,44.39453125,43.89453125,0.5,-1.7520707107365752e-16],
        [0.6705788170002052,64,0.010477794015628206,1,46.1484375,45.7734375,0.375,-1.7520707107365752e-16],
        [0.4600746479991358,192,0.0023962221249954987,3,45.1640625,44.3828125,0.78125,-1.7520707107365752e-16]
      ]
    },
    {
      "natoms": 300, "mode": "flux_native", "steps_per_pass": 64,
      "first_bead_first_three_atoms_endpoint_comparison": {"matched_successful_repeats":5,"max_abs_q":8.673617379884035e-19,"max_abs_p":4.440892098500626e-16},
      "baseline": [
        [7.207152857998153,64,0.11261176340622114,1,70.67578125,45.23828125,25.4375,null],
        [11.042916445003357,64,0.17254556945317745,1,71.0390625,45.7734375,25.265625,null],
        [11.336229241998808,64,0.17712858190623137,1,70.95703125,45.60546875,25.3515625,null],
        [13.336309452002752,64,0.208379835187543,1,71.0625,45.52734375,25.53515625,null],
        [11.450930359002086,64,0.1789207868594076,1,70.328125,44.875,25.453125,null]
      ],
      "candidate": [
        [0.31249733398726676,896,0.0003487693459679316,14,44.33984375,43.4453125,0.89453125,null],
        [0.301062609993096,960,0.000313606885409475,15,46.37890625,45.62109375,0.7578125,null],
        [0.3133563629889977,704,0.0004451084701548263,11,44.48046875,43.69140625,0.7890625,null],
        [0.3421493100031512,832,0.0004112371514460952,13,44.34765625,43.72265625,0.625,null],
        [0.3355737820020295,832,0.0004033338725985931,13,46.48828125,45.73828125,0.75,null]
      ]
    },
    {
      "natoms": 300, "mode": "flux_fused", "steps_per_pass": 64,
      "first_bead_first_three_atoms_endpoint_comparison": {"matched_successful_repeats":5,"max_abs_q":8.673617379884035e-19,"max_abs_p":4.440892098500626e-16},
      "baseline": [
        [4.2797729469966725,64,0.06687145229682301,1,64.66796875,45.41796875,19.25,null],
        [3.7498407780003618,64,0.05859126215625565,1,62.98046875,43.82421875,19.15625,null],
        [3.487600685999496,64,0.054493760718742124,1,63.1640625,43.9140625,19.25,null],
        [3.649543897998228,64,0.057024123406222316,1,62.328125,43.0390625,19.2890625,null],
        [3.2396246520002023,64,0.05061913518750316,1,64.78125,45.53125,19.25,null]
      ],
      "candidate": [
        [0.3254814950050786,704,0.00046233166904130485,11,46.41796875,45.79296875,0.625,null],
        [0.3064555770106381,768,0.00039903069923260165,12,45.77734375,45.02734375,0.75,null],
        [0.3310048920029658,704,0.0004701774034133037,11,45.92578125,45.140625,0.78515625,null],
        [0.32399977101158584,704,0.00046022694745963895,11,45.6328125,44.86328125,0.76953125,null],
        [0.3006547979966854,1024,0.00029360820116863806,16,44.44140625,43.69140625,0.75,null]
      ]
    },
    {
      "natoms": 3000, "mode": "parent_native", "steps_per_pass": 2,
      "first_bead_first_three_atoms_endpoint_comparison": {"matched_successful_repeats":5,"max_abs_q":0.0,"max_abs_p":0.0},
      "baseline": [
        [0.3026311429857742,50,0.006052622859715484,25,51.76953125,47.875,3.89453125,3.434752482434078e-16],
        [0.30464667600608664,58,0.005252528896656666,29,51.82421875,47.9296875,3.89453125,3.434752482434078e-16],
        [0.31107299900759244,54,0.005760611092733193,27,49.94140625,46.04296875,3.8984375,3.434752482434078e-16],
        [0.3089851769836969,72,0.004291460791440234,36,49.890625,46.09765625,3.79296875,3.434752482434078e-16],
        [0.3024626580008771,54,0.005601160333349576,27,50.9375,47.0546875,3.8828125,3.434752482434078e-16]
      ],
      "candidate": [
        [0.30823101499845507,56,0.005504125267829555,28,52.54296875,48.37890625,4.1640625,3.434752482434078e-16],
        [0.30563768698630156,54,0.005659957166412992,27,51.703125,47.6875,4.015625,3.434752482434078e-16],
        [0.31156089501746465,44,0.007080929432215105,22,52.0859375,48.19921875,3.88671875,3.434752482434078e-16],
        [0.3064430870072101,52,0.005893136288600194,26,50.93359375,47.0703125,3.86328125,3.434752482434078e-16],
        [0.30271541899855947,42,0.0072075099761561776,21,49.828125,45.77734375,4.05078125,3.434752482434078e-16]
      ]
    },
    {
      "natoms": 3000, "mode": "parent_metric_python", "steps_per_pass": 2,
      "baseline": [
        {"status":"failed","returncode":-9,"stderr":"","stdout":""},
        {"status":"failed","returncode":-9,"stderr":"","stdout":""},
        {"status":"failed","returncode":-9,"stderr":"","stdout":""},
        {"status":"failed","returncode":-9,"stderr":"","stdout":""},
        {"status":"failed","returncode":-9,"stderr":"","stdout":""}
      ],
      "candidate": [
        [0.32598051599779865,16,0.020373782249862415,8,46.4921875,43.49609375,2.99609375,1.3877787807814457e-16],
        [0.3057930739960284,8,0.03822413424950355,4,46.4921875,43.4765625,3.015625,1.3877787807814457e-16],
        [0.32743163599661784,10,0.032743163599661784,5,46.8125,43.79296875,3.01953125,1.3877787807814457e-16],
        [0.3003838650038233,8,0.03754798312547791,4,46.7265625,43.73046875,2.99609375,1.3877787807814457e-16],
        [0.30898063700442435,16,0.019311289812776522,8,46.76953125,43.7734375,2.99609375,1.3877787807814457e-16]
      ]
    },
    {
      "natoms": 3000, "mode": "flux_native", "steps_per_pass": 2,
      "first_bead_first_three_atoms_endpoint_comparison": {"matched_successful_repeats":4,"max_abs_q":0.0,"max_abs_p":2.7755575615628914e-17},
      "baseline": [
        [28.143924536001578,2,14.071962268000789,1,2519.41796875,43.359375,2476.05859375,null],
        [38.71966711099958,2,19.35983355549979,1,2517.24609375,43.46484375,2473.78125,null],
        {"status":"failed","returncode":-9,"stderr":"","stdout":""},
        [28.53840059399954,2,14.26920029699977,1,2517.20703125,43.453125,2473.75390625,null],
        [39.79308453599515,2,19.896542267997575,1,2517.32421875,43.43359375,2473.890625,null]
      ],
      "candidate": [
        [0.3026231289986754,22,0.013755596772667062,11,47.52734375,43.1015625,4.42578125,null],
        [0.30294677302299533,30,0.010098225767433178,15,47.6171875,43.265625,4.3515625,null],
        [0.30610859699663706,52,0.005886703788396867,26,47.64453125,43.1796875,4.46484375,null],
        [0.30339051999908406,54,0.005618342962946001,27,47.7890625,43.48828125,4.30078125,null],
        [0.3096740979963215,38,0.00814931836832425,19,47.74609375,43.2890625,4.45703125,null]
      ]
    },
    {
      "natoms": 3000, "mode": "flux_fused", "steps_per_pass": 2,
      "first_bead_first_three_atoms_endpoint_comparison": {"matched_successful_repeats":5,"max_abs_q":0.0,"max_abs_p":2.7755575615628914e-17},
      "baseline": [
        [13.31916071100568,2,6.65958035550284,1,1901.39453125,43.37890625,1858.015625,null],
        [10.326110137997603,2,5.163055068998801,1,1895.73046875,43.46875,1852.26171875,null],
        [7.671999227000924,2,3.835999613500462,1,1899.09765625,43.09765625,1856.0,null],
        [12.99720433299808,2,6.49860216649904,1,1901.359375,43.484375,1857.875,null],
        [10.167735631999676,2,5.083867815999838,1,1901.3359375,43.29296875,1858.04296875,null]
      ],
      "candidate": [
        [0.3148402420047205,34,0.009260007117785896,17,47.984375,43.578125,4.40625,null],
        [0.3156396059785038,28,0.011272843070660852,14,47.625,43.375,4.25,null],
        [0.31283784701372497,68,0.00460055657373125,34,47.875,43.61328125,4.26171875,null],
        [0.30956296301155817,34,0.009104793029751711,17,47.875,43.44921875,4.42578125,null],
        [0.3188441630190937,36,0.008856782306085935,18,47.5546875,43.14453125,4.41015625,null]
      ]
    }
  ]
}
```


## Reproduction appendix: cross-version checkpoint experiment

Save the following script as `t02_cross_version_checkpoints.py` next to checkouts
named `pyrpmd-t02-baseline` and `pyrpmd-t02`, with their native extensions built.
Use the baseline revision and candidate source revision identified above. Run it
with the same virtual-environment interpreter; `PYRPMD_CHECKPOINT_EXPERIMENT_OUTPUT`
may choose a fresh output directory. The script refuses to overwrite an existing
experiment. It asserts each worker's imported source location before generating
or resuming artifacts. This is the exact harness used for the recorded experiment.

```python
"""Small real baseline->candidate checkpoint compatibility experiment."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

SCRATCH = Path(__file__).resolve().parent
OLD = SCRATCH / "pyrpmd-t02-baseline"
NEW = SCRATCH / "pyrpmd-t02"
OUT = Path(os.environ.get("PYRPMD_CHECKPOINT_EXPERIMENT_OUTPUT",
                         str(SCRATCH / "t02-cross-version-checkpoints")))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(stage):
    import numpy as np
    import pyrpmd
    from pyrpmd.checkpoint import load_checkpoint
    from pyrpmd.system import System
    from pyrpmd.workflows import fes, rate_recrossing
    from pyrpmd.workflows.rate import _recrossing_options

    expected_root = OLD if stage == "baseline" else NEW
    assert Path(pyrpmd.__file__).resolve().is_relative_to(expected_root / "src")

    def arrays(path):
        with load_checkpoint(path) as saved:
            return {name: saved[name].copy() for name in saved.files}

    geometry = np.array([[-1.5, .2, 0.], [0., 0., 0.], [1.5, .2, .1]])
    system = System(("H", "H", "H"), np.full(3, 1800.), geometry, ((0,), (1, 2)))
    parent_config = {
        "task": "rate", "beta": 220., "beads": 4, "dt_au": .01,
        "seed": 17, "native": True, "potential": {"provider": "harmonic", "params": {"k": .01}},
        "sampling": {"andersen_interval": 2},
        "dynamics": {"loop_backend": "python", "block_steps": 3},
        "reaction": {"fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
                     "breaking_bonds": [[1, 2]], "rinf": 10.},
        "output": {"checkpoint_steps": 3},
        "recrossing": {"dividing_surface": 1., "parent_replicas": 1, "parent_samples": 2,
                       "equilibration_steps": 2, "parent_interval_steps": 2,
                       "child_pairs": 1, "child_steps": 5, "bootstrap": 0},
    }
    options = _recrossing_options(parent_config)

    def parent_task(directory):
        return {"config": parent_config, "system": system, "options": options,
                "chain": 0, "xi_star": 1.,
                "initial_q": np.repeat(system.geometry[None], parent_config["beads"], axis=0),
                "output": str(directory)}

    specs = [{"name": "xi", "kind": "rpmdrate_xi", "estimator": "centroid",
              "fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
              "breaking_bonds": [[1, 2]], "rinf": 10., "equivalent_paths": [[0, 2, 1]]}]
    pmf_config = {
        "task": "fes", "beta": 220., "beads": 4, "dt_au": .01, "seed": 41, "native": True,
        "potential": {"provider": "harmonic", "params": {"k": .01}},
        "dynamics": {"loop_backend": "python", "block_steps": 3},
        "sampling": {"equilibration_steps": 2, "production_steps": 10,
                     "sample_interval": 1, "andersen_interval": 3},
        "analysis": {"measure": "rate_flux"}, "cvs": specs,
        "output": {"checkpoint_steps": 4},
    }

    def pmf_task(directory):
        return {"config": pmf_config, "system": system, "specs": specs, "center": [1.],
                "stiffness": np.array([[.1]]), "window": 0, "replica": 0,
                "output": str(directory)}

    def child_task(directory, snapshots):
        return {"config": parent_config, "system": system, "chain": 0, "parent": 0,
                "pair": 0, "q": snapshots[0], "xi_star": 1., "child_steps": 5,
                "output": str(directory)}

    class Interrupted(Exception):
        pass

    def committed_interrupt(module, callback):
        original = module._atomic_npz

        def stop(path, **payload):
            original(path, **payload)
            raise Interrupted(f"stopped after committed step {payload['step']}")

        module._atomic_npz = stop
        try:
            callback()
        except Interrupted:
            pass
        else:
            raise AssertionError("expected a checkpoint interruption")
        finally:
            module._atomic_npz = original

    full = OUT / "baseline-complete"
    partial = OUT / "baseline-partial"
    parent_name = "parent/checkpoints/parent_0000.npz"
    pmf_name = "pmf/checkpoints/window_00000_replica_0000.npz"
    child_name = "child/checkpoints/child_0000_00000_0000.npz"
    if stage == "baseline":
        OUT.mkdir(parents=True)
        parent = rate_recrossing._parent_task(parent_task(full / "parent"))
        fes._sample_window(pmf_task(full / "pmf"))
        rate_recrossing._child_task(child_task(full / "child", parent["snapshots"]))
        committed_interrupt(rate_recrossing, lambda: rate_recrossing._parent_task(parent_task(partial / "parent")))
        committed_interrupt(fes, lambda: fes._sample_window(pmf_task(partial / "pmf")))
        p, f = arrays(partial / parent_name), arrays(partial / pmf_name)
        assert int(p["elapsed"]) == 3 and int(f["elapsed"]) == 4
        report = {"source": str(Path(pyrpmd.__file__).resolve()),
                  "parent_partial_step": int(p["step"]), "pmf_partial_step": int(f["step"]),
                  "parent_fingerprint": str(p["fingerprint"]), "pmf_fingerprint": str(f["fingerprint"]),
                  "baseline_artifact_sha256": {name: sha(full / name) for name in (parent_name, pmf_name, child_name)},
                  "partial_artifact_sha256": {name: sha(partial / name) for name in (parent_name, pmf_name)}}
        (OUT / "baseline-report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
        return

    resumed = OUT / "candidate-resumed"
    cached = OUT / "candidate-cached"
    shutil.copytree(partial, resumed, dirs_exist_ok=True)
    shutil.copytree(full, cached, dirs_exist_ok=True)
    parent = rate_recrossing._parent_task(parent_task(resumed / "parent"))
    fes._sample_window(pmf_task(resumed / "pmf"))
    comparisons = {}
    for label, name, floats in (("parent", parent_name, ("q", "p", "snapshots")),
                                ("rate_flux_pmf", pmf_name, ("q", "p", "values", "energies"))):
        old, new = arrays(full / name), arrays(resumed / name)
        exact = {}
        differences = {}
        for field in floats:
            np.testing.assert_allclose(new[field], old[field], rtol=1e-12, atol=1e-13)
            exact[field] = np.array_equal(new[field], old[field])
            differences[field] = float(np.max(np.abs(new[field] - old[field])))
        for field in ("rng", "fingerprint", "step", "time", "elapsed"):
            np.testing.assert_array_equal(new[field], old[field])
        comparisons[label] = {"bitwise_equal": exact, "maximum_absolute_error": differences,
                              "rng_exact": True, "fingerprint_exact": True,
                              "step": int(new["step"]), "fingerprint": str(new["fingerprint"])}
    assert parent["propagation_this_call"]["python_steps"] == 3
    before = {name: sha(cached / name) for name in (pmf_name, child_name)}
    replay_pmf = fes._sample_window(pmf_task(cached / "pmf"))
    full_parent = arrays(full / parent_name)
    replay_child = rate_recrossing._child_task(child_task(cached / "child", full_parent["snapshots"]))
    after = {name: sha(cached / name) for name in before}
    assert before == after
    np.testing.assert_array_equal(replay_pmf["samples"], arrays(full / pmf_name)["values"])
    np.testing.assert_array_equal(replay_pmf["last_q"], arrays(full / pmf_name)["q"])
    old_child = arrays(full / child_name)
    # This molecular-rate checkpoint already stores the aggregated pair.
    np.testing.assert_array_equal(replay_child["numerator"], old_child["numerator"])
    assert replay_child["denominator"] == float(old_child["denominator"])
    report = {"source": str(Path(pyrpmd.__file__).resolve()), "comparisons": comparisons,
              "completed_old_cache_sha256_before": before, "completed_old_cache_sha256_after": after,
              "completed_old_pmf_and_child_reused_without_byte_changes": True,
              "parent_additional_steps": parent["propagation_this_call"]["python_steps"]}
    (OUT / "candidate-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def main():
    if len(sys.argv) == 2:
        worker(sys.argv[1])
        return
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing experiment: {OUT}")
    for stage, root in (("baseline", OLD), ("candidate", NEW)):
        env = dict(os.environ, PYTHONPATH=str(root / "src"), OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), stage],
                                cwd=root, env=env, text=True, capture_output=True)
        print(stage + ": " + result.stdout, end="")
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
```
