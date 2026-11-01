# BCOCB: implementation and independent observable validation

Date: 2026-11-01. Parent: `a88e5352b559e09ab5d68171b2cff52b7a1f3d25` (PR #61).
Tested backend Git blob: `f0ea6c570d859d7a827d95e77e6625540dc56934`.
Backend SHA-256: `eabdc36cdfc5004e1273dce8ae0b2fc56a5dd84ceb99e561c3088e106de0ac79`.

## Verdict

The explicit `integrator="cayley-bcocb"` option passes independent map, noise,
covariance, cold-start, reversibility-limit and failure-rollback tests. The
previous high-bead harmonic kinetic-estimator pathology is removed in both
the independently derived invariant measure and actual CPU sampling. Quartic
sampling at both repeated timesteps is consistent with independently converged
finite-P transfer-operator references at the reported statistical resolution.

This validates a sampling algorithm, **not the complete backend for production
chemistry**. CUDA hardware, trained chemical potentials, workflow inputs,
checkpoint integration and BCOCB with multiple time steps remain unvalidated
or unsupported. The PR remains experimental/draft; no merge is implied.

## Implementation and public contract

Select the new splitting explicitly:

```python
sampling = TorchDynamics(
    pes, masses, beta, P, dt,
    integrator="cayley-bcocb", thermostat="pile-l", seed=17,
)
```

The map is `B(dt/2) sqrt(cay(dt*A)) O(dt) sqrt(cay(dt*A)) B(dt/2)`.
The square root is the principal square root of the **full-step** Cayley map,
Eq. (17) of [1], not `cay(dt*A/2)`. The middle O uses the full timestep and one
real Gaussian noise field with bath momentum variance `m*P/beta`. Coordinates,
momenta, transforms, noise and force accumulation stay on the selected device.
The block-boundary finite check, explicit snapshots and RNG capture/rollback
retain their existing contracts. Models can still introduce their own transfers.

Old defaults and both old integrator names retain their previous behavior.
`thermostat=None` collapses the two roots to the original full BCB map, with
no extra transforms or RNG use. TRPMD leaves the centroid unthermostatted;
PILE-L uses the specified centroid friction.

RPC and additive fast/slow groups are supported with `inner_steps=1`, with
both force groups included in B. They still define the explicitly contracted
Hamiltonian, not the uncontracted one. `inner_steps>1` is rejected for BCOCB;
we do not label an unvalidated MTS composition as the published method.
Existing MTS behavior under the old integrators is unchanged.

## Independent verification

`benchmarks/torch_bcocb_validation.py` constructs the continuous free-ring
matrix, solves the Cayley linear system, and uses SciPy's general `sqrtm`.
It does not read production coefficients. For one mode, it assembles the
conditional mean `L` and noise map `G` from the full B-S-O-S-B composition.
The stationary covariance solves `C = L C L.T + G G.T` independently.

The 90 algebra cases span masses 1/7, stiffnesses 0.5/2, frequencies 0 through
100, three timesteps and two frictions. Maximum scaled covariance discrepancy
against [1], Eqs. (22)-(23), is **5.34e-14**. Production tests inject coordinate,
momentum and independent noise basis vectors to verify both L and G, not just
a plausible sampled temperature. Tests also cover the full coupled two-atom
matrix with unequal masses, odd/even/one-bead limits, real Nyquist noise,
finite-time cold-start covariance, RPC force placement, exact no-thermostat
BCB equivalence, block partitioning, force-call counts, RNG rollback after
central noise, hot-loop host-copy guards and second-order linear weak accuracy.

For `V(q)=k*q^2/2`, [1] predicts (real-mass momenta convention):

```
Var(q_k) = P / [beta * (k + m*omega_k^2)]
Var(p_k) = m*P/beta * [1 - dt^2*k/(4*m)].
```

The position marginal is exact for this harmonic model, but endpoint momenta
are **not** Maxwellian. The physical-force stability requirement
`dt*sqrt(k/m) < 2` remains; a regression exhibits instability above that bound.
No general nonlinear step-size bound is inferred from the harmonic result.

## Repeated high-bead harmonic observable test

Identical protocol to the previous report: `V(q)=q^2/2`, `beta=8`, `m=hbar=1`,
`dt=0.5`, 1024 independent trajectories, all starting at `q=p=0`, 1600 steps
(total model time 800). Only one final observation per independent trajectory
is used. These are pointwise Student-t 95% confidence intervals, not simultaneous
bands. Reference values are exact at each **finite** bead count.

| P | Exact finite-P kinetic energy | OBCBO primitive estimate | BCOCB primitive estimate | BCOCB 95% interval |
|---:|---:|---:|---:|---|
| 8 | 0.223810 | 0.201274 | 0.222546 | [0.213242, 0.231851] |
| 32 | 0.248239 | 0.112123 | 0.251175 | [0.224035, 0.278316] |
| 128 | 0.250046 | -0.240903 | 0.286188 | [0.225505, 0.346872] |
| 256 | 0.250137 | -0.752416 | 0.249970 | [0.165458, 0.334483] |

All old OBCBO estimates reproduce the previous run. All four BCOCB primitive
intervals include the reference. Primitive estimator variance grows with bead
count: this is why the P=256 interval remains wide despite removal of the bias.
The lower-variance centroid-virial estimate at P=256 is **0.246826**, interval
**[0.241441, 0.252211]**, also compatible with 0.250137.

Across BCOCB q2, primitive kinetic energy, centroid-virial kinetic energy and
p2 at all four P values, deviations from the independent *finite-time*
covariance are at most **1.654 estimated standard errors**. At P=256 the exact
finite-time primitive expectation is 0.250137309764334, whereas the stationary
expectation is 0.250137309763872; residual cold-start error is negligible here.
That statement is checked for this protocol, not assumed for every system.

The P=256 sampled endpoint p2 is **30.0841**, interval **[29.9281, 30.2401]**,
matching the BCOCB prediction **30**, not the Maxwell value **32**. For thermal
quantum kinetic energy use position-based primitive/centroid-virial estimators;
`energy()` remains a mechanical Hamiltonian. Before launching unthermostatted
physical RPMD trajectories from equilibrated positions, refresh Maxwell momenta
using `sample_momenta()` and a separately seeded production engine.

## Repeated quartic sampling test

`V(q)=q^4/4`, P=16, beta=m=hbar=1, 256 independent trajectories, seed 717,
equilibration time 20, production time 100, observation interval 0.2. Means
are averaged **within each trajectory first**; standard errors use the 256
independent trajectory means, not correlated times or beads.

The independently discretized transfer operator gives kinetic energy
**0.581077817578**, with 256- and 512-point quadrature agreeing to 1.2e-13
for this observable. The original Schrodinger/finite-P reference tests were
also rerun. The finite-P reference is not replaced by the infinite-P value.

| Splitting | dt | Primitive kinetic estimate | 95% interval |
|---|---:|---:|---|
| OBABO / normal modes | 0.02 | 0.579548 | [0.562611, 0.596486] |
| OBABO / normal modes | 0.10 | 1.604347 | [1.553490, 1.655203] |
| OBCBO / Cayley | 0.02 | 0.580298 | [0.563308, 0.597288] |
| OBCBO / Cayley | 0.10 | 0.556180 | [0.539093, 0.573267] |
| **BCOCB** | **0.02** | **0.577178** | **[0.560977, 0.593379]** |
| **BCOCB** | **0.10** | **0.590194** | **[0.574162, 0.606226]** |

At dt=0.1 the BCOCB centroid-virial estimate is **0.580252**, interval
**[0.579319, 0.581186]**. Both BCOCB step sizes are within 1.741 estimated
standard errors of the reference for each of q2, potential and both kinetic
estimators. This supports improved sampling at this resolution; it is **not**
a proof of zero timestep bias for nonlinear potentials. Use further step-size,
bead-count and statistical convergence checks for the intended chemistry.

## Repeated dynamical correlation check

The same quartic RPMD correlation protocol was rerun: P=54, beta=m=hbar=1,
2048 paired exact finite-P canonical initial trajectories, seed 20261101,
total time 10, observations every 0.1. **No thermostat is applied.** This is
an important compatibility check, not a thermostatted PILE trajectory passed
off as physical RPMD. The reference is BAB dt=0.0025; dt=0.005 checks refinement.

| Cayley dt | Maximum mean difference / C(0) | Approximate simultaneous 95% upper bound |
|---:|---:|---:|
| 0.01 | 0.018664% | 0.078582% |
| 0.05 | 0.146041% | 0.257764% |
| 0.10 | 0.562156% | 0.808071% |

For every tested dt, BCOCB with the thermostat disabled matches the old BCB
**bit for bit** in final q,p and every correlation product. The change from the
previous reported upper bound (0.8031% to 0.8081% at dt=0.1) comes only from
Bonferroni correction over seven rather than five non-reference method entries;
the measured mean difference and matching trajectories are unchanged.
BAB reference refinement changes the mean correlation by at most 0.001042% of
C(0). These are RPMD-reference comparisons, not exact quantum dynamics or
numerical digitizations of paper figures.

## Regression and packaging results

CPU environment: Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, SciPy 1.17.0,
float64, one numerical thread. Native extensions were built locally from the
CI source archive plus the reviewed PR validation files.

| Test set | Result |
|---|---|
| New BCOCB file | **148 passed, 1 skipped** |
| Potential/periodic/backend/paper/observable/BCOCB focused suite | **450 passed, 5 skipped, 3 subtests passed** |
| Complete source suite | **4787 passed, 1 failed, 142 skipped, 357 subtests passed** |
| Four Torch files from newly built/extracted source archive | **433 passed, 4 skipped** |

The complete-suite failure is the same GUI-backend isolation test as in the
parent validation. A fresh child process importing **only Matplotlib**, with
no PyRPMD loaded, reports `MPLBACKEND=TkAgg` but backend `agg` and pyplot already
loaded. This independently reproduces the environmental mismatch. The test
was neither removed nor marked expected-failure; the complete suite is **not**
reported as green. Skips cover CUDA, ASE and other unavailable optional providers.

The new benchmark is explicitly included in MANIFEST.in, and the new tests run
both in the Torch CI job and from its rebuilt/extracted source archive. CI
status for the eventual commit must be checked separately from local results.

## Reproduce / evidence

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python -m pytest -q tests/test_torch_backend.py tests/test_torch_paper_validation.py \
  tests/test_torch_observable_validation.py tests/test_torch_bcocb.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python benchmarks/torch_bcocb_validation.py --section all --output bcocb.json
```

Sections `algebra`, `dimension`, `sampling` and `correlation` can be run separately.
`--device cuda` requests CUDA explicitly without CPU fallback. The observation
harness intentionally copies statistics to the host and is not a timing test.
Output records environment, seeds, confidence intervals, backend and harness
SHA-256 values. [Machine-readable summary](../benchmarks/torch_literature/bcocb_summary.json)
retains the results and SHA-256 of the full numerical JSON supplied with the
review bundle. No unstable trajectory, random seed or observable was removed
to make the comparison look better.

The old 2019 paper harmonic failure-percentage mismatch remains unresolved;
this change neither tunes that statistic nor claims to settle it. No water,
H-in-metal, trained-ML, GPU speedup or rate/dRPMD/PIMD workflow claim follows
from these oscillator tests. The original PR acceptance gates still apply.

## Reference

[1] R. Korol, J. L. Rosa-Raices, N. Bou-Rabee and T. F. Miller III,
*Dimension-free path-integral molecular dynamics without preconditioning*,
J. Chem. Phys. **152**, 104102 (2020), DOI: **10.1063/1.5134810**.
Principal square root: Eq. (17); invariant position/momentum variances:
Eqs. (22)-(23); primitive kinetic estimator: Eqs. (25)-(28).
Primary preprint: https://arxiv.org/abs/1911.00931 .
