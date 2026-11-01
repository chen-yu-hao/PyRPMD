# Torch backend: paper protocols and independent numerical validation

Validation date: 2026-11-01. Tested production code: PR #61,
`fdad63de446ca51aed4245823fd1e4a5d2eba4c4`.
Backend Git blob: `973cc9a8005b85506033c009ba456862fa8a0e47`.
Backend SHA-256: `c1bbeaf840f48d476ab41a335fd7bfdf7de89b4b21d580e7de69d3fcc7ed5946`.
No production numerical algorithm was changed during this validation.

## Verdict and boundaries

The independent matrix, contraction, sampling and MTS checks pass. However,
**this is not a complete reproduction of the cited papers**: the published
harmonic-instability percentage does not match this experiment. CUDA, trained
chemistry models, water/H-in-metal observables and reaction rates have not been
validated. Existing rate/dRPMD/PIMD workflows still do not use this backend.
Keep the PR experimental/draft.

Local regression command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python -m pytest -q -ra tests/test_torch_backend.py tests/test_torch_paper_validation.py
```

Result: **199 passed, 4 skipped**. The 127 additional cases cover independent
real-mode projections, long-time matrix propagation, unequal-mass symplecticity
and reversibility, cold-start covariance, resonance and combined RPC/MTS/PILE.
Three skips require CUDA; one needs the full upstream checkout. Environment:
Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, one Torch thread, float64.
The local checkout contains the reviewed backend files, not the full repository.

The existing full-checkout GitHub Actions run `41308665435` for this production
commit provides separate evidence:

| Job / step | Observed result |
|---|---|
| Torch CPU: potential, periodic and backend tests | 90 passed, 4 skipped, 3 subtests passed |
| CPU Python 3.11: full pytest step | 4363 passed, 118 skipped, 357 subtests passed |
| Electronic-structure adapters | Job succeeded |
| CPU Python 3.11: first MPI command | Failed in launcher with `prte-rmaps-base:alloc-error` |
| CPU Python 3.12 | Cancelled while pytest was running |

The MPI failure happened before the Python MPI script ran; its diagnostic help
files were also missing. The runner installed Open MPI 5.0.10 on Ubuntu 26.04.
This validation update makes oversubscription explicit on each MPI command
(`--oversubscribe`, supported by Open MPI 5 [6]) and disables matrix fail-fast so
one failed environment does not erase the other's evidence. This does not skip
any MPI tests or change simulation code. **It is a launcher remediation, not a
claim that MPI has subsequently passed.** Review the new commit's CI separately.

## Reproduction

From a complete checkout with the GPU extra (a CPU PyTorch wheel suffices):

```bash
PYTHONPATH=src python benchmarks/torch_paper_validation.py --section stability --output stability.json
PYTHONPATH=src python benchmarks/torch_paper_validation.py --section sampling --output sampling.json
PYTHONPATH=src python benchmarks/torch_paper_validation.py --section rpc-mts --output rpc-mts.json
PYTHONPATH=src python benchmarks/torch_paper_validation.py --section resonance --output resonance.json
PYTHONPATH=src python benchmarks/torch_paper_validation.py --section rpc-sampling --output rpc-sampling.json
```

`--section all` runs every section; `--device cuda` explicitly requests CUDA
without fallback. JSON contains seeds, sample counts, parameters, environment,
backend hash, uncertainty and reference values. Observation transfers and
per-step energy checks in this harness are intentional: **this is a correctness
experiment, not a GPU timing benchmark**. The separate synthetic timing script
is `benchmarks/torch_device_backend.py`.

## 1. Cayley stability: direct comparison to the 2019 protocol

Korol, Bou-Rabee and Miller [1], Sec. III and Eqs. (27), (29), specify
`V(q)=q^2/2` and `V(q)=q^4/4`, mass 1, beta 1. A trajectory fails when the
fractional change of its full ring Hamiltonian exceeds 10% within 100 time
units. We check every integration step, with hbar=1 and one active Cartesian
coordinate. Multiplying our sum-beads Hamiltonian by 1/P gives the paper's
normalization and does not change fractional errors.

Harmonic initial conditions are IID exact finite-P Gibbs samples:
`Var(q_k)=P/[beta*(1+omega_k^2)]`, `Var(p_k)=P/beta`.
Quartic positions use rejection sampling from that Gaussian: the acceptance
probability is `exp[-beta*mean((q^2-1)^2)/4]`. This follows from
`q^4/4-q^2/2+1/4=(q^2-1)^2/4`; the spring action cancels. No short, potentially
biased equilibration trajectory is used. Failed rows are removed only after the
first crossing of the threshold, before overflow; rows are independent.

| Potential | P | dt | Independent trajectories | Normal-mode BAB failures | Cayley BCB failures |
|---|---:|---:|---:|---:|---:|
| Harmonic | 16 | 0.05 | 1000 | 0 | 0 |
| Harmonic | 16 | 0.10 | 3000, three seeds | 2566 (85.53%) | 0 |
| Quartic | 54 | 0.01 | 1000 | 0 | 0 |
| Quartic | 54 | 0.029 | 1000 | 1 | 0 |
| Quartic | 54 | 0.05 | 1000 | 7 | 0 |
| Quartic | 54 | 0.10 | 1000 | 31 | 0 |

The dt=0.029 case runs 3448 steps, ending at t=99.992; other rows end at t=100.
Zero observed failures is not a proof of universal stability: the two-sided
Wilson 95% upper bound is 0.383% for 0/1000 and 0.128% for 0/3000.
The P=54 maximum-safe-step expression in [1], Eq. (30), is pi/(2P)=0.02909;
this is a resonance condition, not a guarantee of less than 10% energy error
for every nonlinear trajectory.

### Unresolved quantitative discrepancy — do not hide it

[1] states that approximately **25%** of harmonic trajectories fail at
P=16, beta=1, dt=0.1. Our three seeds give 86.1%, 84.2%, and 86.3%, pooled
Wilson 95% interval **84.23%–86.75%**. Sampling error cannot explain the gap.
Thus this percentage is **not reproduced**.

An independent NumPy/SciPy calculation builds each full harmonic step as
`B @ expm(dt*A) @ B` or `B @ solve(I-dt*A/2,I+dt*A/2) @ B`, without using the
backend's cached coefficients. For seed 1001 it gives the same 861/1000 versus
0/1000 failures. Final q,p over 1000 steps match the backend for 32 trajectories
to maximum absolute errors **1.90e-11** (BAB) and **1.30e-12** (BCB). BAB's
largest one-step mode spectral radius is 1.000572773; BCB's is 1 to roundoff.

This independently supports the implemented maps and the resonance mechanism,
but does not settle the paper percentage. The paper's original generated
ensemble/code and exact implementation of the percentage statistic were not
available in this comparison. We neither fit an initialization to obtain 25%
nor conclude that the paper is wrong. No heatmap/curve was digitized and no
claim of reproducing its full figures is made.

## 2. PILE and resonant ergodicity

Ceriotti et al. [2], Eqs. (26)–(36), specify the OU half-step, variance m*P/beta
and internal friction gamma_k=2*omega_k. The backend implements OBABO/OBCBO,
not BAOAB/BCOCB. We start **q=p=0**, not a preloaded canonical ensemble, and run
1000 independent trajectories per case, beta=8, mass=1, V=q^2/2 per Cartesian
direction, total time 32. Coordinates are averaged within a trajectory; standard
errors use trajectories as independent units, not correlated beads or times.

The reference is a separately assembled 2x2 mode map `L` with noise covariance
`Q`. Its stationary covariance solves `C=L C L.T+Q`; the exact cold-start
covariance at N steps is `C-L^N C (L^N).T`. This separates finite-time error,
finite-step bias, finite-P error and Monte Carlo uncertainty.

Representative Cayley results at dt=0.02 (intervals are pointwise 95%):

| P | Exact finite-P q2 | Measured q2 | 95% interval |
|---:|---:|---:|---|
| 1 | 0.125000 | 0.125548 | [0.119389, 0.131706] |
| 8 | 0.447619 | 0.448619 | [0.439050, 0.458187] |
| 16 | 0.485424 | 0.486300 | [0.477279, 0.495320] |
| 32 | 0.496479 | 0.503411 | [0.493984, 0.512837] |

All 12 P/dt/scheme cases are within 1.92 estimated standard errors of their
exact finite-time discrete covariance. The infinite-P quantum value is
0.500335575; using it as the target for P=1 would incorrectly label a correct
classical-limit implementation as broken.

**Finite-step bias remains.** At P=16 the exact stationary q2 biases are:

| dt | OBABO / normal modes | OBCBO / Cayley |
|---:|---:|---:|
| 0.02 | +0.01156% | +0.01000% |
| 0.10 | +0.29043% | +0.25063% |
| 0.20 | +1.17965% | +1.01010% |

For this unit-frequency harmonic model, OBCBO's covariance is inflated by
`1/(1-dt^2/4)`. Stability alone is not sampling accuracy, and this result must
not be advertised as the zero-configurational-bias BCOCB algorithm.

A separate free-ring resonance experiment tests the mechanism in [1], Sec. IV:
P=8, beta=8, dt=pi/2, Nyquist omega=2, 100 steps, 4096 cold-start trajectories.
Normal-mode propagation gives q2=8.92e-31 instead of equilibrium 0.25; Cayley
gives **0.248828**, 95% interval **[0.242602, 0.255054]**. This is a mechanism
check, **not a replication of Figures 6–7**: those examples use gamma=omega,
whereas the public backend uses gamma=2*omega internally.

## 3. RPC: projection, forces and approximation bias

The energy prefactor P/Pc and adjoint force lift `(P/Pc)*T.T` follow the RPC
construction [3,4], explicitly Eqs. (14)–(17) of [4]. We independently build
real orthonormal cosine/sine bases and compare all Pc from 1 to P for
P in {1,2,3,4,5,8,16,31,32}: **102 cases**, maximum projection discrepancy
**4.61e-15**. Identity and centroid preservation are asserted. Even Pc uses the
backend's documented cosine-member convention; [4] describes odd Pc, so the
even-Pc tests validate that extension, not a separate published result.
Existing force finite-difference and regional-adjoint tests also pass.

For a transparent harmonic split Vfast=0.9*q2/2, Vslow=0.1*q2/2, P=32,
beta=8, inversion of the exact contracted Hamiltonian gives:

| Pc | Relative equilibrium q2 bias against full P |
|---:|---:|
| 1 | +2.7293% |
| 2 | +1.7001% |
| 3 | +0.6709% |
| 4 | +0.4513% |
| 8 | +0.0828% |
| 16 | +0.0162% |
| 32 | 0% |

These are **analytic biases of this test Hamiltonian**, not water-paper data.
They demonstrate why contracting the slow term is an approximation that needs
convergence checks. Atom selection must include every coordinate on which a
component's energy depends; selecting H and metal force entries from an
undecomposed ML potential is not this Hamiltonian.

The joint RPC+MTS+PILE experiment additionally uses P=8, Pc=1/3/8, beta=8,
dt=0.1, four inner steps, 1000 cold-start trajectories, both schemes. All six
sampled q2 values are within **0.063 standard errors** of independently derived
finite-time covariance for the *contracted, time-discretized* map. This is not
an assertion that the map samples the uncontracted Hamiltonian without bias.

## 4. Reversible MTS and actual force-work reduction

The symmetric fast/slow splitting is the r-RESPA construction [5]. Against a
full 2P-dimensional matrix exponential of a fixed contracted harmonic
Hamiltonian, halving dt gives error ratios **3.930–4.017**, approaching 4,
across both schemes and 1/2/4 inner steps. Six additional tests use coupled
atoms with masses 1 and 5, P=1/5/8, and assert symplecticity and momentum-reversal
reversibility over five outer steps at absolute tolerances 4e-14 and 5e-14.

A cost comparison keeps **physical duration 4 and inner timestep 0.01 fixed**:
4 trajectories, P=8, one propagation block with endpoint-force reuse.

| Method | Outer dt | Inner steps | Pc | Slow calls | Slow configurations | Fast calls |
|---|---:|---:|---:|---:|---:|---:|
| Small-step full ring | 0.01 | 1 | 8 | 401 | 12832 | 401 |
| MTS | 0.04 | 4 | 8 | 101 | 3232 | 401 |
| MTS + RPC | 0.04 | 4 | 2 | 101 | 808 | 401 |

This is approximately 3.97x fewer slow calls for MTS and 15.88x fewer slow
configurations for MTS+RPC. It is **not a wall-clock/GPU speedup measurement**,
and the RPC row has a different Hamiltonian. Calls include initial endpoints;
no hidden thermostat is applied to the conservative cost/accuracy trajectories.

## Remaining acceptance gates

- Run CUDA tests on real hardware and benchmark a trained chemistry model with
  device-native neighbor graphs, warmup, synchronization and memory measurements.
- Resolve/document the harmonic 25% discrepancy using the original protocol or
  author-provided initial conditions before claiming numerical figure replication.
- Reproduce actual water or H-in-metal observables with the same published PES,
  cell, estimator and uncertainties; analytic oscillator checks do not replace it.
- Integrate workflow inputs/checkpointing/biases/constraints and thermostat-free
  rate children before making rate/dRPMD/PIMD end-to-end claims.
- Review all CI jobs, including MPI, on the validation commit. No merge is implied.

## References

1. R. Korol, N. Bou-Rabee, T. F. Miller III, *Cayley modification for strongly
   stable path-integral and ring-polymer molecular dynamics*, J. Chem. Phys.
   **151**, 124103 (2019). DOI: 10.1063/1.5120282; arXiv:1907.07941.
2. M. Ceriotti, M. Parrinello, T. E. Markland, D. E. Manolopoulos, *Efficient
   stochastic thermostatting of path integral molecular dynamics*, J. Chem.
   Phys. **133**, 124104 (2010). DOI: 10.1063/1.3489925; arXiv:1009.1045.
3. T. E. Markland, D. E. Manolopoulos, *An efficient ring polymer contraction
   scheme for imaginary time path integral simulations*, J. Chem. Phys.
   **129**, 024105 (2008). DOI: 10.1063/1.2953308.
4. C. John, T. Spura, S. Habershon, T. D. Kuhne, *Quantum ring-polymer contraction
   method*, Phys. Rev. E **93**, 043305 (2016). DOI: 10.1103/PhysRevE.93.043305;
   arXiv:1512.08206 (especially the explicit contraction/force equations).
5. M. Tuckerman, B. J. Berne, G. J. Martyna, *Reversible multiple time scale
   molecular dynamics*, J. Chem. Phys. **97**, 1990–2001 (1992).
   DOI: 10.1063/1.463137.
6. Open MPI 5.0.10, `mpirun / mpiexec` manual, `--oversubscribe` option:
   https://docs.open-mpi.org/en/v5.0.10/man-openmpi/man1/mpirun.1.html
