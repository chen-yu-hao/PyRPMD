# H + CH₄ native propagation: verified interface, retained energy gate failure

This record follows the [initial electronic PES check](../README.md). The only
adapter change is a guarded `native_binding()` exposing the existing compiled
`pyrpmd_cbe2008_batch` through PyRPMD's public `fortran-batch-v1` interface.
No Fortran wrapper, PES source, energy constant, gradient coefficient or binary
was changed. The original adapter is preserved byte for byte in
[source-before-native/pes.py](source-before-native/pes.py), with its hash in
[source-before-native.json](source-before-native.json).

**Native/Python propagation parity passes. The aggregate short-NVE verification
is deliberately recorded as failed at P=128, dt=0.1 fs. No equilibrium
distribution or rate is validated.**

## Actual checks

The [verification report](verification.json) contains twelve comparisons:
P=1 and 128; one and 64 steps; unbiased dynamics, umbrella plus flux metric,
and constrained-parent propagation. The time step was 0.1 fs. Every native call
completed its requested steps without fallback. Maximum absolute errors over
the comparisons were:

| Quantity | Native vs Python maximum |
|---|---:|
| Position | 4.09 × 10⁻¹⁴ bohr |
| Momentum | 5.01 × 10⁻¹¹ atomic units |
| Energy divided by beads | 7.44 × 10⁻¹⁵ hartree |
| Observed coordinate/surface | 8.03 × 10⁻¹⁵ |

Static checks verify the independent four-channel scalar CV, its finite-
difference gradient, ξ=1 at every equivalent reference, ξ=0 after rigid
reactant separation, explicit isotope masses and native provider construction.
The ordinary generator also wrote and reloaded
[generated-pilot-P1.py](generated-pilot-P1.py) and
[generated-pilot-P128.py](generated-pilot-P128.py). Those are **not-run inputs**;
their external library/output paths describe this workspace and can be
regenerated elsewhere.

A separate [three-step workflow check](workflow/report.json) runs the actual
rate preparation and umbrella worker at ξ=0 and ξ=1 for P=1 and 128. It verifies
strict native execution with zero fallback, the matched Python worker, and an
exact reload of each completed checkpoint, including unchanged checkpoint
bytes. This adds 0.12 CPU seconds and remains an interface check with no PMF
analysis. Its small checkpoints are retained for replay.

[pytest.log](pytest.log) records **11 passed**: the prior four PES/receipt tests
and seven new binding/mapping/reference tests. The new checks cover shared
canonical-library locks, descriptor ownership and rejection of instance,
subclass or class-method overrides that would otherwise bypass Python
semantics. The [post-binding PES recheck](pes-verification.json) again passes
all seven original electronic/force gates with unchanged source constants.

## The retained time-step result

Before execution, the [protocol](../../../../../../benchmarks/hch4/reproduction_protocol.json)
specified an absolute energy-span limit of 5×10⁻⁵ hartree per bead over a short
20 fs deterministic trajectory. Its initial state is intentionally reproducible
and non-equilibrium: nearly collapsed rings, thermal-width random momenta,
and tetrahedral separated reactants. This is a stress check for discretization,
not an estimate of thermodynamic energy or canonical sampling quality.

| Beads | Time step (fs) | Energy span / P (hartree) | Absolute gate |
|---|---:|---:|---|
| 1 | 0.1 | 1.6778840 × 10⁻⁶ | Pass |
| 1 | 0.05 | 4.2377160 × 10⁻⁷ | Pass |
| 128 | 0.1 | 1.8773814 × 10⁻⁴ | **Fail** |
| 128 | 0.05 | 4.6039519 × 10⁻⁵ | Pass |
| 128 | 0.025 | 1.1429660 × 10⁻⁵ | Diagnostic follow-up passes the same absolute bound |

The first four rows are in the original report. The last is a separate
[time-step follow-up](timestep-control.json) using exactly the same P128 state.
Successive energy-span ratios 0.2452 and 0.2483 are consistent with second-order
time-step behavior. They support a discretization diagnosis; they do not
establish that a rate computed at 0.05 fs is converged. The original gate and its
nonzero command exit status remain unchanged. The source's known 2.06 ppm
energy/gradient conversion discrepancy also remains unchanged.

The original probe used 4.98 CPU seconds (13.04 seconds wall time on the shared
machine); the smaller-step follow-up used 0.79 CPU seconds. This is well within the
authorized one-CPU-minute verification scope. No umbrella ensemble, long
sampling or thermal-rate task was launched.

## Throughput, with its limits

Timing includes real PES calls, normal modes, all four CV branches, the flux
metric Hessian, umbrella force, energy checking and 2 fs observations. At fixed
ξ=0,0.9,1.0, native costs were 17.5–24.5 µs/step for P1 and 0.957–1.061 ms/step for
P128. Matched Python-loop timings were 1.34–1.69 ms/step and 3.09–3.52 ms/step,
respectively. The native block repeats start from the same state and are
performance probes, not independent physical samples.

The [reproduction plan](../reproduction-plan.md) applies these measurements to
the paper's exact step count and identifies the dense-MBAR memory limitation.
The extrapolation excludes thermostat boundaries, checkpoint I/O, task
scheduling, long-time geometry variation, additional convergence controls and
analysis. CPU time and wall time are both saved; the shared-machine wall-time
slowdown is not attributed to PyRPMD.

## Replay

From the repository root, with an externally rebuilt byte-pinned library:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.hch4.verify_native \
  --library /path/to/libcbe2008.so --output /path/to/native-verification.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.hch4.timestep_control \
  --library /path/to/libcbe2008.so --probe /path/to/native-verification.json \
  --output /path/to/timestep-control.json

PYRPMD_CBE2008_LIBRARY=/path/to/libcbe2008.so \
  python -m pytest benchmarks/hch4/test_native.py tests/test_hch4_pes.py -q

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.hch4.verify_workflow \
  --library /path/to/libcbe2008.so --output /path/to/new-workflow-smoke
```

The first command intentionally returns a failure status while that fixed P128
energy gate fails. It still writes the full structured report. All scripts,
parameters, seeds, library/native-binary hashes, source hashes and output
digests are recorded in [execution.json](execution.json). Only project code,
small numerical results and provenance are committed; no external PES source
or compiled library is included here.
