# Torch observables: independent references and literature comparison

Validation date: 2026-11-01. Continuation of PR #61 at parent
`dae39b12c8b7e3d181661f2c1bff111d856f22ab`.
This report supplements [the earlier stability/RPC/MTS report](torch_validation_report.md)
and supersedes its historical statements about checkout availability and CI.
**No production numerical algorithm was changed.**

## Verdict

The backend passes the new independent map and observable regressions, including
published finite-step covariance formulas. It does **not** provide unbiased
large-bead sampling merely because Cayley propagation stays stable. In particular,
its thermostat splitting is **OBCBO, not BCOCB**. We directly observe the
large-bead primitive-kinetic-energy pathology predicted for OBCBO in [2]. Keep
this backend experimental; do not infer validated production chemistry or GPU
performance from these tests.

The previously reported harmonic failure-rate disagreement with [1] remains:
**85.53% measured versus approximately 25% in the paper** under the stated
criterion. This continuation does not resolve that discrepancy or relabel the
experiment as a successful complete paper reproduction.

## 1. Provenance, test coverage and a packaging fix

Backend Git blob: `973cc9a8005b85506033c009ba456862fa8a0e47`.
Backend SHA-256:
`c1bbeaf840f48d476ab41a335fd7bfdf7de89b4b21d580e7de69d3fcc7ed5946`.
Harness SHA-256:
`47052865e5f48aedb08e9984b8e5fd1c9876d50db274c193aefdd5d57e181d1c`.
Environment: Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0+cpu, float64,
one Torch thread. CUDA is unavailable.

A complete source distribution was obtained from the parent commit's successful
GitHub Actions package run `41310983273`, artifact `13085510537`. The downloaded
ZIP SHA-256 is
`1668150b9dbc2dead0c4f053f6cd5dc47c864ed95d669d61882f06cb4e275e1d`.
Its provenance records PR merge-build commit
`abdd9c3243983c88d78b2f614b6a40370dad72b8`. The tested production backend and
recovered existing benchmark were verified byte-for-byte against their Git blobs;
the native extensions were compiled locally. This is no longer an isolated
backend-only test tree.

| Local run | Result |
|---|---|
| Complete-source potential/periodic/Torch/paper/observable tests | **302 passed, 4 skipped, 3 subtests passed** |
| Complete source: `python -m pytest -q` | **4639 passed, 1 failed, 141 skipped, 357 subtests passed** |
| Rebuilt source archive: backend + paper + observable tests | **285 passed, 3 skipped** |

There are **85 new regression cases**. They cover independent transfer-operator
references, classical gamma integrals, quantum grid and bead convergence, exact
quartic rejection sampling, 48 parameterized published-covariance checks, 16
production-map checks including unequal masses, finite-time OU sampling,
large-bead bias, and an MTS resonance counterexample. They supplement the 127
paper-protocol cases already on the branch and the original backend tests.

The complete-suite failure is
`test_render_never_initializes_a_requested_gui_backend`, outside the numerical
backend. Its prerequisite is that `MPLBACKEND=TkAgg` remains TkAgg without loading
pyplot. A fresh subprocess importing **only Matplotlib, not PyRPMD**, instead
reports Matplotlib 3.10.8, backend `agg`, and pyplot already imported, both with
and without the host's Matplotlib integration environment flag. This isolates an
environment incompatibility with the test's premise, but **the full suite is not
reported green**. The test was neither removed nor marked expected-failure.
Optional dependency/PES tests and three CUDA tests account for the local skips.

The original source archive contained `test_torch_paper_validation.py` but omitted
its imported benchmark script. With Torch installed, that archive cannot collect
the test successfully. `MANIFEST.in` now includes the Torch benchmark harnesses
and reference-result directory. CI now runs the three backend/reference test
files from a newly built and extracted source archive, in addition to checkout
tests. The archive test passed locally; the CI modification itself needs its new
commit's run, not a borrowed status from the parent.

Separately, **all 11 parent-commit workflows completed successfully**. Run
`41310983169` has successful Python 3.11/3.12 CPU jobs, Torch CPU (217 passed,
4 skipped), and electronic-structure adapters. All eight MPI commands passed on
both Python versions with explicit `--oversubscribe`. The successful H3, Ni-H,
and other existing workflows do not validate the new Torch backend's chemistry
path: those workflows still use the existing engines.

## 2. Independent equilibrium references

We use `V(x)=x^4/4`, beta=m=hbar=1, one active coordinate. Two references are
constructed without using the integrator:

* A finite-P primitive transfer-operator quadrature on [-6,6], checked with 256
  and 512 grid points. Traces in its eigenbasis give q2, potential energy, and
  primitive/centroid-virial kinetic energies.
* A quantum Schrodinger finite-difference eigensolver on [-8,8], using 80 levels
  and 1024/2048/4096 grid points. This is the infinite-bead comparison, not a
  replacement for the finite-P target.

The finite-P primitive and centroid-virial kinetic energies agree to below
2e-9, as do 256/512-grid results. The classical P=1 values reproduce the gamma
integrals. The quantum 2048-to-4096 kinetic-energy change is 5.62e-6, and the
highest retained Boltzmann weight is 2.22e-129. These are numerical reference
checks, not certified interval-arithmetic error bounds.

| Reference | q2 | Quantum kinetic-energy estimator |
|---|---:|---:|
| P=1, transfer operator | 0.675978240 | 0.500000000 |
| P=8, transfer operator | 0.696478651 | 0.579376021 |
| P=16, transfer operator | 0.696665053 | 0.581077818 |
| P=54, transfer operator | 0.696720393 | 0.581605085 |
| Quantum eigensolver, 4096 grid points | 0.696725514 | 0.581657953 |

The P=8-to-16 kinetic-energy error decreases by approximately four, consistent
with the primitive path integral's second-order bead convergence. Independent
quartic rejection samples (16384 rings per P; seeds 11000+P) agree with the
transfer reference: the largest nonconstant-observable discrepancy is 1.31
estimated standard errors. The rejection weight is
`exp[-beta*mean((q^2-1)^2)/4]` relative to the exactly sampled harmonic ring.

## 3. Quartic centroid correlation versus timestep

This extends the type of observable shown in [1], Eq. (12) and Fig. 4(c), but
**does not digitize or reproduce the paper's numerical curve point by point**.
It compares independent implementations/references on the same model.

Protocol: P=54, beta=m=hbar=1, 2048 IID finite-P canonical initial rings,
seed 20261101, no thermostat, duration 10, observations every 0.1. Every method
uses the same positions and momenta. The observable is
`C(t)=mean[x_centroid(0)*x_centroid(t)]`.
The reference is normal-mode BAB with dt=0.0025, checked by the dt=0.005 run;
it is a fine-step **RPMD** reference, not exact quantum real-time dynamics.

| Scheme / dt | Maximum absolute mean paired difference / C(0) | Approx. simultaneous 95% upper bound |
|---|---:|---:|
| BAB / 0.005 | 0.001041% | 0.001348% |
| BAB / 0.010 | 0.005211% | 0.006745% |
| Cayley / 0.010 | 0.018664% | 0.077294% |
| Cayley / 0.050 | 0.146041% | 0.255324% |
| Cayley / 0.100 | **0.562156%** | **0.803050%** |

C(0)=0.5990230063. Normalizing by C(0), rather than by C(t), avoids exploding
relative errors near zero crossings. Paired differences retain common-initial-
condition cancellation. The upper bound uses Student-t standard errors with a
Bonferroni correction over all 505 non-reference method/time comparisons; it is
an approximate large-sample uncertainty statement, not an exact distribution-
free guarantee. Reference-discretization error is assessed by refinement, not
included in that statistical interval.

The larger Cayley timestep retains a close correlation function in this test,
consistent with the qualitative claim in [1]. This does not demonstrate exact
quantum dynamics, universal timestep safety, chemistry accuracy, or speedup.
Energy errors in this experiment are recorded at observation times only; the
earlier stability experiment checks every integration step.

## 4. OBCBO sampling: reproducing a published limitation

[2], Eqs. (20)-(23), gives analytic normal-mode stationary covariances. The
independent Lyapunov solution and production map agree with those formulas over
the tested frequencies, masses, stiffnesses, and timesteps. For V=x2/2 and m=1,
OBCBO inflates each positional mode variance by

`Var_dt(q_k) / Var_exact(q_k) = 1 / (1 - dt^2/4)`.

At dt=0.5 this is 16/15, or **+6.6667%**. Stability therefore does not imply the
correct canonical distribution. The primitive path-integral estimator

`K_prim = P/(2 beta) - (P/beta)^2 mean[(q_j-q_(j-1))^2]/2`

is particularly sensitive to the cumulative high-mode bias [2, Eqs. (25)-(30)].
We use beta=8, P=8/32/128/256, dt=0.5, 1024 independent cold-start trajectories,
1600 steps, seeds 5000+P. Internal PILE friction is 2*omega and centroid friction
is 1. Finite-time covariance is also computed independently to distinguish
incomplete equilibration from stationary discretization bias.

| P | Exact finite-P K | OBCBO stationary prediction | Measured primitive K | Pointwise 95% interval |
|---:|---:|---:|---:|---|
| 8 | 0.223810 | 0.205397 | 0.201274 | [0.191190, 0.211357] |
| 32 | 0.248239 | 0.131455 | 0.112123 | [0.082576, 0.141670] |
| 128 | 0.250046 | -0.266618 | -0.240903 | [-0.303374, -0.178432] |
| 256 | **0.250137** | **-0.799854** | **-0.752416** | **[-0.845089, -0.659743]** |

The sampled observables are within 2.21 estimated standard errors of the
independent finite-time predictions across the 12 comparisons. The negative
primitive estimate is not negative mechanical kinetic energy: it is a failure
of a quantum estimator when fed a biased distribution. The centroid-virial
estimator is less pathological but still biased; at P=256 its predicted value
is 0.262646 versus the exact 0.250137.

This is a comparison with the **analytic formulas and mechanism** of [2], not
its Fig. 1 parameters (which use stiffness 256 and beta=1). The dimension-free
BCOCB scheme in that paper has a different operator ordering and uses the square
root of the full Cayley map for each free-ring half step. It is not implemented
by the present backend, and substituting an ordinary Cayley(dt/2) is not the same
algorithm. No such production change was made during validation.

## 5. Anharmonic sampling is observable- and timestep-dependent

Protocol: quartic potential, P=16, beta=1, 256 independent cold-start PILE
trajectories, seed 717, equilibration time 20, production time 100, observations
every 0.2. Standard errors use independent **trajectory means**, not the 500
correlated observations within each trajectory. The finite-P transfer reference
for primitive kinetic energy is **0.5810778176**.

| Scheme | dt | Measured primitive K | Pointwise 95% interval |
|---|---:|---:|---|
| OBABO / BAB | 0.02 | 0.579548 | [0.562611, 0.596486] |
| OBCBO / Cayley | 0.02 | 0.580298 | [0.563308, 0.597288] |
| OBABO / BAB | 0.10 | **1.604347** | [1.553490, 1.655203] |
| OBCBO / Cayley | 0.10 | **0.556180** | [0.539093, 0.573267] |

The dt=0.1 BAB primitive estimate is 39.62 estimated standard errors from the
finite-P reference. Cayley is substantially closer, but its primitive estimate
is still 2.87 standard errors low. The latter is a pointwise diagnostic, not a
multiple-testing-corrected proof; the harmonic analytic bias is the stronger
controlled result. Agreement of a centroid or potential-energy observable alone
would miss important high-frequency sampling errors.

An additional deterministic regression constructs a classical fast/slow split
with fast stiffness 1, slow stiffness 0.1, outer dt=3, and 32 inner steps. Its
outer MTS map has spectral radius >1.1 even though the inner Verlet step is
stable. Thus a Cayley free-ring map does **not** remove all fast/slow MTS
resonance bands. Outer timestep and contraction levels still require convergence.

## 6. Reproduce and interpret

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q \
  tests/test_potentials.py tests/test_torch_periodic.py tests/test_torch_backend.py \
  tests/test_torch_paper_validation.py tests/test_torch_observable_validation.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python benchmarks/torch_observable_validation.py --section all --output observables.json
```

Each section can be run separately with `--section references`, `correlation`,
`dimension`, or `sampling`. `--device cuda` requests GPU execution explicitly;
no CUDA results are included in this report. Outputs record parameters, seeds,
confidence intervals, reference values, source hashes, and environment. The
harness intentionally transfers observations to the CPU and is **not a device
performance benchmark**. A compact checked result record is
[`observable_summary.json`](../benchmarks/torch_literature/observable_summary.json).

Before production readiness: implement and independently test a suitable sampling
splitting such as BCOCB; converge timesteps and contraction for each real system;
resolve or precisely explain the 2019 stability-percentage mismatch; run CUDA
residency/correctness tests; and validate a trained potential and actual
rate/dRPMD/PIMD workflow. Those workflows and ASE-native graph batching remain
outside this backend's current scope. Existing chemistry CI success cannot be
used as evidence that these missing paths were tested.

## References

[1] R. Korol, N. Bou-Rabee, T. F. Miller III, *Cayley modification for strongly
stable path-integral and ring-polymer molecular dynamics*, J. Chem. Phys. 151,
124103 (2019), DOI: 10.1063/1.5120282; arXiv:1907.07941v4.
https://arxiv.org/abs/1907.07941

[2] R. Korol, J. L. Rosa-Raices, N. Bou-Rabee, T. F. Miller III,
*Dimension-free path-integral molecular dynamics without preconditioning*,
J. Chem. Phys. 152, 104102 (2020), DOI: 10.1063/1.5134810;
arXiv:1911.00931v3. https://arxiv.org/abs/1911.00931

[3] M. Ceriotti, M. Parrinello, T. E. Markland, D. E. Manolopoulos,
*Efficient stochastic thermostatting of path integral molecular dynamics*,
J. Chem. Phys. 133, 124104 (2010), DOI: 10.1063/1.3489925.
https://arxiv.org/abs/1009.1045
