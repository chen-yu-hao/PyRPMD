---
orphan: true
---

# Shared propagation stack: validation record

Date: 2026-11-01. Base revision:
`2f907ba96d22cb134198715e398b1d9f8d83d929`.

The complete repository suite passed **4,690 tests and 357 subtests**, with
**116 skips, no failures and no errors**, in 285.46 seconds. A fresh, final
focused invocation passed **184 tests** in 3.04 seconds after the small API
guards described below were added. These counts overlap and must not be added.
The retained [full-suite log](pytest.log), [final focused log](final-focused.log)
and [source/runtime manifest](source-hashes.json) identify what was exercised.

## Source identity and execution

The full suite started at 09:26:32 UTC with 354 source, native and test files
hashed. Its source digest was
`c9532eecdf13d0b0b77cbf4ef64a99ab1be58080dc62b3e45b76839c5bc3d4eb`.
Both compiled extension binaries remained byte-for-byte unchanged during the
run. The environment used Python 3.12.14, NumPy 2.5.4 and SciPy 1.18.1 on Linux,
with both native extensions built and one OpenMP/BLAS thread:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python \
  -m pytest -q --basetemp=../pz --junitxml=../propagation-final-suite.xml
```

The manifest retains the original absolute command, each file's SHA256, both
extension hashes and the JUnit totals. Local socket access was permitted so the
unmodified i-PI and periodic socket tests ran in this invocation.

After collection, four files changed: `dynamics.py`, `block_propagation.py`,
`propagation.py` and `test_propagation_composition.py`. The changes preserve an
explicit zero TRPMD friction parameter, reject contradictory friction inputs,
make a configured strict-native zero-step block a no-op, and cap temporary
dense native observations at 256 MiB independently of sparse public outputs.
Six regression cases were added. The full suite was not restarted; the fresh
184-test gate exercised the final source, including those new cases:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python \
  -m pytest -q tests/test_propagation_composition.py \
  tests/test_trpmd_spectroscopy.py tests/test_trpmd_rates.py \
  tests/test_composed_native_observations.py \
  tests/test_native_operator_switching.py tests/test_native_block_compatibility.py
```

The source digest for the final focused test run is
`d3ce7725211003fc04c286ec51669e32c42f1fe7abad4a7406175d86da235d91`.
Digests hash the sorted path-to-SHA256 mapping using the JSON serialization
specified in the manifest. They describe source content independently of the
eventual commit identifier. Publication removes one redundant blank line at
the end of `test_composed_native_observations.py`; the manifest records that
formatting-only hash change separately.

The 922 warnings comprise ASE array-shape deprecations with NumPy 2.5 and
Matplotlib colormap deprecations in the existing report renderer. The 116 skips
cover optional unbuilt or unconfigured BrHCl, BKMP2, EAM and ONNX potentials;
optional JaxPIP, PySCF, tblite, Caracal, PLUMED and PyTorch environments; two
original-optimizer MPI checks; and one independent reference solver. The exact
reasons are retained in the log and manifest. No real MPI, trained ML model or
GPU execution is established by this run.

## Behavioral checks

The new composition tests compare canonical scalar, native and batched
execution, including deterministic operator ordering, fixed coordinates and
momenta, independent bath and random-generator state, NHC, stochastic baths,
GLE/PIGLET controls and force splitting. The MTS velocity-Verlet regression
uses an outer step that would fail the unsplit spring stability criterion but
stable inner steps, and compares the batch result with canonical scalar
dynamics. Both native and NumPy free-ring paths are exercised.

Selective native-observation tests cover sparse public records, observer/stop
prefixes, terminal state and full auxiliary-state restart. Native bath
switching, constrained propagation and strict-native rejection of unsupported
Python force scheduling have explicit regressions. A pure-Python NHC path
remains available when the native extension is absent.

The task scheduler tests exercise fixed-window umbrella sampling, adiabatic
dRPMD and site-rate parent/paired-child propagation. Scalar and grouped task
runs retain the original per-task seeds, clocks, fixed atoms, sampled arrays
and checkpoint state. Checks include completed resumes, interrupted umbrella
resumes, actual local process transport, tasks finishing before their first
force call, per-frame provider failures, thread-start failures and coordinator
interruption. Provider result arrays are copied before another call can reuse
their storage. RPC contraction and lifting stay within each trajectory while
compatible physical force-provider calls are grouped.

These checks are concentrated in
[`test_batched_composition.py`](../../../tests/test_batched_composition.py),
[`test_trajectory_batch.py`](../../../tests/test_trajectory_batch.py),
[`test_input_propagation_stack.py`](../../../tests/test_input_propagation_stack.py),
[`test_propagation_composition.py`](../../../tests/test_propagation_composition.py)
and the native operator/observation regression modules. The site-rate public
workflow fixture supplies a sampled PMF while running the actual constrained
parent and paired child propagation; it is a scheduling-equivalence check,
not a new converged rate calculation.

## Measured force-provider batching

The retained
[`propagation_stack_batching.py`](../../../benchmarks/efficiency/propagation_stack_batching.py)
control uses eight independent trajectories, four beads, 16 steps and identical
scalar operator/energy-observation schedules. Reproduce it with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python \
  benchmarks/efficiency/propagation_stack_batching.py \
  --output benchmarks/efficiency/propagation_stack_batching.json
```

| Configuration | Provider | Scalar calls | Grouped calls | Frames evaluated in each run |
| --- | --- | ---: | ---: | ---: |
| NVE, Andersen, PILE-L, NHC, SVR and GLE, each separately | Physical PES | 392 | 49 | 1,568 |
| GLE, three inner MTS steps, two contracted beads | Fast PES | 648 | 81 | 2,592 |
| Same split configuration | Slow PES | 392 | 49 | 784 |

Every case has exactly equal terminal positions, momenta and energy traces,
with equal thermostat and RNG snapshots. The
[machine-readable result](../../../benchmarks/efficiency/propagation_stack_batching.json)
records maximum array differences of zero. The measured improvement is an
**eightfold reduction in physical PES dispatch calls** at unchanged physical
frame count. It is not a measured wall-clock speedup; a cheap harmonic PES may
not benefit from the scheduler's thread and synchronization overhead.

## Compatibility and distributions

The retained [legacy-wrapper reproduction](native_legacy_compat.py) compares the
base revision's and current `NativeEnsemble` Python wrappers for 12 pre-existing
thermostat modes and four barostats, each with mode fusion enabled and disabled.
All **32 cases** have identical constructor fingerprints, initial complete
checkpoints, observations after nine steps with stride two and moments, and
terminal complete checkpoint digests. The
[result record](native_legacy_compat_results.json) includes the current wrapper,
header, binding and extension hashes.

Both wrappers deliberately load the **same current `_thermal` binary**. This
checks wrapper schema, fingerprint, checkpoint and observation compatibility;
it does not compare independently built old and new native numerical engines.
The PIGLET case uses an identity-covariance interface control rather than a
physically calibrated PIGLET parameter set.

The final source distribution and CPython 3.12 Linux native wheel build passed.
The [package inspection](package-inspection.json) records distribution hashes
and confirms that all 110 wheel Python modules exactly match the final source,
both extensions are present, and the new modules, native header, tests and
benchmark are included in the source distribution. The
[isolated extracted-wheel smoke](wheel-smoke.json) passes import, native NHC
selective-prefix observations, common Dynamics NHC checkpoint restore,
MTS/RPC/NHC composition, batched NHC and explicit zero TRPMD friction.

## Boundaries

`parallel.trajectory_batch_size` is opt-in, defaults to one and is bounded at
256 concurrent task lanes. Each lane retains its own integrator, thermostat,
random stream, checkpoint and stop rule. Provider batching changes physical
force-call grouping; it does not flatten independent thermostats into one
thermodynamic system.

Task batching currently accepts fixed independent umbrella windows and rejects
adaptive/exchange window batching, mutable-cell/NPT batching, matrix electronic
potentials and stateful or coupled physical providers without an independent
frame contract. Known split/difference providers preserve their ring-local
operations. Native execution remains capability-selected and unsupported
strict-native combinations fail explicitly. Public compatibility classes
remain adapters rather than disappearing from the API.

Canonical umbrella sampling intentionally excludes PIGLET and TRPMD because
their sampling measures do not satisfy the workflow's canonical MBAR contract.
This is a scientific restriction, not an inability of the common propagation
stack to execute those thermostats. The tests and count benchmark establish
software behavior and numerical equivalence for their fixtures; they do not
establish new physical convergence, reaction-rate accuracy, GPU acceleration
or distributed scaling results.
