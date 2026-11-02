# BCOCB in the RPMDrate normal-mode calculation flow

## What is compatible

Ring-polymer normal modes are a coordinate representation, not a competing
integrator. Both the exact free-ring A map and the Cayley C map are diagonal
in this representation. BCOCB preserves the bead -> mode -> bead workflow:
physical half-kick B in bead space, forward transforms, square-root C / full
OU / square-root C **without leaving mode space**, inverse transforms, and
another physical half-kick B. Nonlinear physical forces still act on beads.

Reference inspected: [GreenGroup/RPMDrate at
8c1828d2302aacf09a15faf5c7fc50a18d4c113e](https://github.com/GreenGroup/RPMDrate/tree/8c1828d2302aacf09a15faf5c7fc50a18d4c113e).
`rpmdrate/_math.f90:rfft,irfft` uses FFTW R2HC/HC2R divided by sqrt(P).
`rpmdrate/_main.f90:free_ring_polymer_step` uses exact cosine/sine rotations
and real masses, and `verlet_step` wraps that map in physical-force kicks.
The inspected version offers Andersen/GLE thermostatting. **It is not the
source of a BCOCB implementation**; we extend its normal-mode conventions
using Korol et al., JCP 152, 104102 (2020), DOI: 10.1063/1.5134810, Eq. 17.

No upstream runtime, input parser or checkpoint format is emulated. The
compatibility claim is the transform convention, free-ring map, real-mass
state/force conventions, and the explicitly tested sampling/production flow.
This does not claim reproduction of RPMDrate's constrained recrossing or GLE.

## Opt-in Torch API

```python
from pyrpmd.torch_backend import TorchDynamics, TorchState
from pyrpmd.torch_normal_modes import RPMDrateNormalModes

sampler = TorchDynamics(
    pes, masses, beta, P, dt,
    integrator="cayley-bcocb", thermostat="pile-l",
    normal_mode_layout="rpmdrate", seed=17,
)
sampler.propagate(state, sampling_steps)

# To inspect/export packed modes, without copying to the host:
nm = RPMDrateNormalModes(P, device=state.q.device)
q_nm = nm.forward(state.q, bead_axis=1)      # [trajectories,P,atoms,3]
q_beads = nm.inverse(q_nm, bead_axis=1)

# Separate physical production engine; endpoint BCOCB momenta are NOT Maxwell.
production = TorchDynamics(
    pes, masses, beta, P, dt,
    integrator="normal-mode-bab", thermostat=None,
    normal_mode_layout="rpmdrate", seed=23,
)
launch = TorchState(q_beads, production.sample_momenta(len(state.q)))
production.propagate(launch, production_steps)
```

The default `normal_mode_layout="rfft"` retains the previous arithmetic and
seeded trajectories. Layout is independent of `normal-mode-bab`, `cayley-bcb`
or `cayley-bcocb`; changing it on a live engine is rejected. Layouts use the
same bead Gaussian field and consume the same RNG sequence, but different
floating-point operation ordering means equality is numerical, not bitwise.
The new option is **not** a workflow `.inp` key. All state and transform work
remains on the selected CPU/CUDA device; CUDA tests require actual hardware.

The standalone transform defaults to the last bead axis, accepting RPMDrate's
`[3,N,P]` arrays. For bead-space atomic-unit arrays from such a calculation:

```python
# Transpose only axes, not units or masses. This is an explicit upload boundary.
state = TorchState.from_arrays(q_rpmdrate.transpose(2, 1, 0),
                               p_rpmdrate.transpose(2, 1, 0), device="cuda")
```

Do not transpose a packed-mode array into a bead State: call `nm.inverse`
first. No automatic checkpoint or trajectory-format inference is performed.
A constrained rate launch additionally requires the appropriate constrained
Maxwell draw; the unconstrained example does not implement that operation.
A runnable oscillator example is `examples/ensembles/torch_normal_modes.py`.

## Normalization, metric and thermostat

For `z = rfft(q, norm='ortho')`, the real packing is

```
Re(z0), Re(z1), ..., Re(z_floor(P/2)),
Im(z_floor((P-1)/2)), ..., Im(z1).
```

The imaginary slots have the **negative sine** sign. The centroid slot is
`sqrt(P)*mean(q)`, not the arithmetic centroid. An even-P Nyquist mode occurs
once, at packed index P/2. The frequencies are `2P/beta*sin(pi*k/P)`, repeated
for the matching real/imaginary pair. The P=1 limit has only a free centroid.
See [FFTW's halfcomplex convention](https://www.fftw.org/doc/The-Halfcomplex_002dformat-DFT.html).

This real packing is invertible but not orthonormal. Its metric has weight
one for centroid/Nyquist and two for all other slots:

```
sum_beads(q**2) = sum_modes(metric * q_nm**2).
```

Consequently a real bead Gaussian field with variance `m*P/beta` transforms
to variance `m*P/(2*beta)` in each paired real/imaginary component, and
`m*P/beta` in centroid/Nyquist. Sampling every packed slot with the same
variance is incorrect. The new path transforms the same real bead noise as
the default path, obtaining these factors without hand-built complex noise.
The CPU sampler's real **orthonormal** cos/sin ordering uses different factors
and a sine sign; its noise is mapped explicitly in cross-backend tests.

The BCOCB half-map is the principal square root of the full Cayley map:

```
z = omega*dt/2
(q_k, p_k) <- (q_k + dt*p_k/(2*m), p_k - dt*m*omega**2*q_k/2) / sqrt(1+z*z)
```

Both right-hand sides use the old q_k,p_k. This is not `cay(dt*A/2)`.
For exact normal-mode propagation RPMDrate instead uses `exp(dt*A)`.
Retaining that exact rotation around a middle OU would give BAOAB, not BCOCB.

## Existing CPU workflow clarification

PyRPMD's CPU `Dynamics.sample_step(method="bcocb")` **already** combines
normal-mode Cayley half maps with central PILE-L. Its native `sample_free`
route does likewise. BCOCB was not intrinsically limited to Torch; the new
`cayley-bcocb` name belongs to the separate Torch API. This change leaves those
CPU algorithms and input/checkpoint identities unchanged and tests agreement.

For existing CPU PIMD/FES configurations, the matching settings remain:

```toml
[dynamics]
integrator = "cayley-bcb"

[sampling]
thermostat = "pile-l"
integrator = "bcocb"
```

The Cayley label selects the free-ring map **within** the normal-mode flow.
`normal-mode-bab` with `sampling.integrator="baoab"` is the exact-rotation
alternative; simply relabeling BAOAB as BCOCB would be wrong. CPU rate/dRPMD
routing restrictions and Torch workflow/checkpoint limitations still apply.

## Validation and reproducibility

New tests are `tests/test_rpmdrate_normal_modes.py`. Independent references
in `benchmarks/rpmdrate_normal_modes.py` use direct trigonometric sums, SciPy's
general matrix square root, and actual FFTW R2HC/HC2R through ctypes. They do
not read production transform matrices or cached propagator coefficients.
No upstream source is vendored. The complete RPMDrate package was not built:
the actual FFTW ABI and a separately implemented pinned-source free-ring
formula are compared, not an unmodified RPMDrate executable.

Checks include odd/even/P=1 modes, inverse and Parseval metric, sine signs,
real-mass free-ring rotation, BCOCB noise covariance/placement, coupled atoms
of masses 1 and 5, same-noise agreement with the default Torch path and the
NumPy/native CPU sampler, sampling-to-exact-normal-mode production, RPC,
existing MTS, model-call counts, atomic/RNG rollback and host-copy guards.

The packed BCOCB middle uses **3 forward + 2 inverse batched transforms per
step**, versus **6 forward + 5 inverse** for the default unfused middle. The
noise transform is included. No potential calls are added: one initial plus
one endpoint call per step per additive component. This is an operation-count
measurement, not a demonstrated GPU/chemistry wall-clock speedup.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python -m pytest -q tests/test_rpmdrate_normal_modes.py
PYTHONPATH=src python benchmarks/rpmdrate_normal_modes_validation.py \
  --output normal-modes.json
PYTHONPATH=src python examples/ensembles/torch_normal_modes.py --device cpu
```

The experiment repeats the previous BCOCB protocol without changing physical
parameters: harmonic P=8/32/128/256, 1024 cold-start trajectories, beta=8,
dt=0.5, 1600 steps; quartic P=16, beta=1, dt=0.02/0.1, 256 trajectories,
equilibration time 20 and production time 100; thermostat-free quartic
correlations at P=54 with 2048 paired initial conditions to time 10. The
correlation check here measures layout equivalence, not a new exact-quantum
reference or digitized reproduction of a paper figure.

## Completed CPU results (2026-11-01)

Environment: Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, SciPy 1.17.0,
float64 and one numerical thread. Full source and native extensions were
available. No CUDA hardware was available.

| Test set | Result |
|---|---|
| New normal-mode file | 124 passed, 1 CUDA skip |
| Backend + BCOCB + new normal-mode tests | 345 passed, 5 CUDA skips |
| Extended potential/sampling/native integration suite | 887 passed, 8 skipped, 3 subtests passed |
| Complete project suite | 4911 passed, 1 failed, 143 skipped, 357 subtests passed |
| Five Torch test files from rebuilt/extracted source archive | 551 passed, 11 skipped |

The full-suite failure is the previously recorded GUI-backend isolation test.
A fresh subprocess importing only Matplotlib (not PyRPMD) selects Agg and loads
pyplot despite MPLBACKEND=TkAgg. The failing test was neither removed nor
marked expected-failure. The full suite is **not reported green**. Skips include
missing optional packages/PESs and CUDA. The source-archive tests additionally
skip six native comparisons because that freshly extracted archive has not
built the native extension; those same comparisons passed in the compiled
full source integration run. The archive retains both new reference scripts.

A separate parent-versus-new-default comparison exercised P=1/5/8/31, all
three integrators and None/PILE-L/TRPMD: all 36 cases retained bitwise q, p
and RNG state after 20 steps. A CPU periodic PIMD smoke run with explicit
BCOCB and a second with exact-normal-mode BAOAB both completed and returned
identical summaries on completed resume. These are routing/restart checks,
not evidence of statistical convergence or new Torch workflow integration.

Across eleven bead counts through P=256, maximum absolute transform errors:

| Comparison | Error |
|---|---:|
| Actual FFTW R2HC | 6.67e-16 |
| Independent trigonometric basis | 7.35e-14 |
| Forward/inverse round trip | 1.06e-15 |
| Weighted Parseval identity | 5.69e-14 |

### Repeated harmonic and nonharmonic sampling

The packed layout repeats the previous BCOCB harmonic result, including
finite-step endpoint momentum bias. Below is the primitive quantum kinetic
estimator, not mechanical kinetic energy; CIs are pointwise 95% Student-t
intervals over independent trajectories.

| P | Exact finite-P | Packed BCOCB mean | 95% CI |
|---:|---:|---:|---|
| 8 | 0.223810 | 0.222546 | [0.213242, 0.231851] |
| 32 | 0.248239 | 0.251175 | [0.224035, 0.278316] |
| 128 | 0.250046 | 0.286188 | [0.225505, 0.346872] |
| 256 | 0.250137 | 0.249970 | [0.165458, 0.334483] |

All harmonic intervals include the corresponding finite-P reference. Across
these four cases, paired final active-coordinate q differs by at most
2.67e-15 and p by
3.46e-13 from the default layout; all engine RNG states are identical.
Large-P primitive-estimator variance is still present; no new physical
accuracy improvement over the already validated BCOCB method is claimed.

For the quartic model the independent finite-P transfer-operator reference
is 0.5810778176. Errors use independent trajectory means, not beads or
correlated time samples as independent observations.

| dt | Packed primitive estimate (95% CI) | Packed centroid-virial estimate (95% CI) |
|---:|---|---|
| 0.02 | 0.577178 [0.560977, 0.593379] | 0.580717 [0.579694, 0.581740] |
| 0.1 | 0.590194 [0.574162, 0.606226] | 0.580252 [0.579319, 0.581186] |

The largest paired quartic final-state discrepancies are 1.36e-13 in q and
4.67e-13 in p. Nonlinear timestep bias, RPC bias, and estimator uncertainties
have not been waived: matching two layouts is not proof of exact sampling.

### Repeated unthermostatted correlations

At P=54, n=2048 and dt=0.01/0.05/0.1, the packed and default full Cayley maps
produce bitwise-identical final q,p and zero measured differences in the
sample mean correlation at all observation times. This verifies representation
compatibility. It is not a fresh claim of exact quantum dynamics, an original
RPMDrate trajectory reproduction, or a new paper-curve accuracy comparison.

## Evidence and remaining boundaries

The [machine-readable summary](../benchmarks/torch_literature/rpmdrate_normal_modes_summary.json)
records environment, source hashes, estimator means/intervals, comparison
errors and the SHA-256 of the complete numerical output. The review bundle
contains the full JSON, test logs/XML, source patch and additional default-
compatibility, example and CPU workflow smoke records. Regenerate the full
numerics using the command above; observations deliberately transfer to the
CPU and are not GPU timing measurements.

CUDA execution, trained chemistry models, BCOCB with MTS, new Torch workflow
input/checkpoint routing and constrained recrossing remain outside this
validation. BCOCB endpoint momentum bias and nonlinear/RPC convergence
requirements from the earlier report are unchanged. CI status is specific
to a commit and must be reviewed separately; no merge or green latest CI is
implied by these local results.
