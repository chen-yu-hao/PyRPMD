# Main-engine path-integral validation

The scripts in `benchmarks/tier2_literature/path_integral*.py` separate an
independent action reference, the finite-step integrator, and finite-trajectory
sampling error. All energies and lengths below use reduced `m=hbar=omega=1`.
The numerical CSV/JSON results are calculations performed for this PR. They
are not points digitized from published figures.

## Suzuki–Chin: fourth-order action and force-bearing dynamics

[Kapil, Behler and Ceriotti, JCP **145**, 234103 (2016),
DOI 10.1063/1.4971438](https://doi.org/10.1063/1.4971438)
describes the force-based high-order action. We use its alpha=0 alternating
weights and the even-bead position observable. The article's molecular-water
and ice calculations require their particular potential and long trajectories;
those systems are not reproduced here.

For a harmonic oscillator, we independently assemble and invert the precision
matrix of the finite-bead Gaussian action. Its variance is compared with
`coth(beta/2)/2`, for beta=1,4,10 and P=4,8,16,32,64,128. Fits on P=16–128 give:

| beta | Primitive fitted order | Suzuki–Chin fitted order |
| --- | ---: | ---: |
| 1 | 1.99967 | 4.00296 |
| 4 | 1.99467 | 3.99053 |
| 10 | 1.96817 | 3.94279 |

These are finite-range convergence fits, so the coldest/coarsest data have not
fully reached the asymptotic regime. `suzuki_chin_convergence.csv` retains
every point rather than only the fitted slopes.

For the anharmonic potential `V=q²/2+0.1q⁴`, analytic Hessian-vector products
provide an independent force reference. Reducing the symmetric displacement
from 0.1 to 0.0001 decreases the maximum force error from `7.32e-4` to
`7.32e-10`, consistent with a second-order finite difference. The minimum
observed error is about `6.26e-12` at `1e-5`; smaller displacements expose
floating-point cancellation. The driver covers two unequal physical masses.

Actual unthermostatted `Dynamics.step` trajectories run for time 20 at
dt=0.04,0.02,0.01. Maximum relative energy excursions are approximately
`2.97e-4`, `7.48e-5`, `1.88e-5`. Reversing all momenta and propagating back
returns q,p within `5.5e-13`. This checks the complete modified-force path and
the real-mass spring splitting, not just the static action formula.

## Bosons: the published two-dimensional harmonic trap

[Feldman and Hirshberg, JCP **159**, 154107 (2023),
DOI 10.1063/5.0173749](https://doi.org/10.1063/5.0173749), Appendix B2/Fig. 6
of the [available preprint](https://arxiv.org/html/2305.18025v5), uses P=36,
beta*omega=6 for a particle-number scan, and N=16 for a temperature scan.
The paper reports a maximum energy discrepancy of 0.5%, after one million
steps per run and 20% equilibration, with three independent runs.

The [authors' input repository](https://github.com/Hirshberg-Lab/quadratic-bosonic-pimd-jcp23-data)
is pinned to `c3939a9d718dcdd8bd2f3ff718c60b22d3fa6609`. It supplies original
LAMMPS input files and figure mappings, **not numerical curve arrays**. The
noninteracting P=36 inputs identify the two-dimensional harmonic model. We
use the paper's dimensionless beta values rather than pretending the rounded
physical temperature/mass constants in the inputs are exact reduced units.

### Independent canonical reference

For a cycle of k particles, the continuum single-cycle partition function is
`z_k=[2 sinh(k beta/2)]^(-2)`. With primitive P-bead discretization it is
`z_k(P)=[2 sinh(k P asinh(beta/(2P)))]^(-2)`. The canonical recursion
`Z_N=(1/N) sum_k z_k Z_(N-k)` and its beta derivative give the total energy.
This reference does not call the exchange potential, connection probabilities,
force routine or trajectory integrator. The reference itself is tested against
full N! permutation Gaussian integrals through N=4.

At beta=6, P=36, N=2–64, the exact finite-action discretization bias is about
**-0.344% to -0.345%**. That already occupies most of the paper's reported
0.5% error envelope. At N=16,beta=2, the bias is -0.0374%. This is an
explanation of the expected discretization error, not a claim that an MD
trajectory has attained the paper's statistical accuracy.

### Actual main-engine sampling

The public `Dynamics.sample_step` propagates the production
`BosonicExchangePotential`, with P=36 and N=2,4,16 at beta=6, plus N=16 at
beta=2. Three independent seeds use 12,000 steps at dt=0.02, discarding 20%.
An N=4 paired-seed refinement uses dt=0.01 and 24,000 steps, preserving total
sampling time. PILE-L/Cayley is used instead of the paper's NHC thermostat;
these shorter runs are explicitly **not full-budget reproductions**.

A driver RNG wrapper sets inactive z noise to zero *before* every thermostat
substep; initial z coordinates and momenta are zero. Both stay identically
zero, including force evaluations. Projecting only the completed positions
would instead allow spurious 3D distances to affect exchange probabilities and
is not used. There is no change to the production force or integrator code.

For this harmonic model only, the scaling/virial identity gives total energy
`E=2<V>`, including at finite P. It is a benchmark estimator, not a generic
new total-energy API. Seed means, all sampled observables and 16 time-block
means per seed are retained. `trajectory_summary.csv` reports seed SEM and
Student-t 95% intervals with two degrees of freedom. These are pointwise
sampling intervals; they do not remove discretization or ergodicity error.
A confidence interval containing the reference is a consistency check, not
proof of sub-percent convergence. The report preserves discrepancies instead
of filtering seeds or adding trajectories until a test passes.

### Scaling

`--mode timing` times only `bosonic_exchange` energy and force evaluation,
with P=32 and N=16–2048, one process/one BLAS thread, seven repeats after a
warmup. This matches the timing scope in Appendix B1, where the authors
report an i-PI exponent near 1.9. Python overhead, memory allocation and
different hardware preclude comparing absolute timings or claiming their
speedup. The N=256–2048 log-log fit and the full timings are retained.

On this machine the measured finite-range exponent is **1.2514**, with
median times 0.0394 s at N=256 and 0.5466 s at N=2048. This does **not**
reproduce the paper's 1.9 timing exponent or establish the asymptotic scaling
experimentally. The quadratic arithmetic is mixed with Python loop overhead
and allocation costs over this range. The algorithm's O(N²) operation count
and the timing fit are different claims; no near-linear asymptotic claim is
made from these measurements. Maximum connection-probability row/column
normalization errors remain below 1.1e-12 through N=2048.

## PIGS: projection and imaginary-time discretization

[Sarsa, Schmidt and Magro, JCP **113**, 1366 (2000),
DOI 10.1063/1.481926](https://doi.org/10.1063/1.481926) is the methodological
source for open-chain projection. This check uses an exactly solvable harmonic
oscillator, not a reproduction of that paper's interacting application.

For trial amplitude `exp(-alpha*x²/2)`, the projected midpoint has variance
`1/[2 alpha(Theta/2)]`, with
`alpha(t)=(alpha+tanh(t))/(1+alpha*tanh(t))`. An independent open-chain
precision matrix includes half-weight endpoint potentials and both trial
amplitudes. We vary alpha=0.25,2; projection time Theta=2,4,8,12; and
P=17,33,65,129. The midpoint variance converges toward its finite-projection
value at second order in tau; the finite-projection value separately
approaches the ground-state variance 0.5.

For example, alpha=0.25 and Theta=8 gives continuum-projected variance
0.5002013181, while P=33 gives 0.4963437199. Increasing P to 129 gives
0.4999575725: its closeness to 0.5 partly reflects cancellation of projection
and discretization errors. Reporting only its ground-state error would hide
that distinction. The P=33 actual main-engine trajectory is compared to
0.4963437199, with the same independent-seed analysis as above.

## Stationary integrator check

For harmonic forces, the actual thermostatted step is a linear Gaussian map
`x'=M x+B eta`. `path_integral_analysis.py` extracts M and B by propagating
basis vectors through `Dynamics.sample_step`, then solves
`C=M C M^T+B B^T`. The resulting coordinate covariance is compared with the
independently integrated finite action.

For both SC (P=8,beta=4) and PIGS (P=33,Theta=8), dt=0.08 down to 0.01
reproduces the action variance to relative error below `8e-14`, with Lyapunov
residual below `3e-14`. This establishes the harmonic stationary-distribution
limit of the implemented split map. It does not imply zero time-step bias
for anharmonic or exchange-coupled sampling.

## Reproduce

The committed sampling evidence was generated by the exact driver archived as
[`driver_at_run.py`](driver_at_run.py), SHA256
`5c07eb4d26a9ddd9a3161bde9610da454e210705801852ddf03ff08a54d347e2`.
The original manifest's driver identity is preserved. After all 24 jobs
completed, Python/native core hashes were verified unchanged against the
committed core and added to the manifest and individual job metadata. The
annotation and original job-JSON hashes are retained in
[`original_run_provenance.json`](original_run_provenance.json). These annotations
do not claim that the later driver produced the old trajectories.

A subsequent audit hardened the current driver's cache handling without
changing numerical equations, settings or trajectory operations. New completed
jobs and partial checkpoints carry full case, seed and source identities.
Incompatible or unidentified caches are rejected before restoring state or
overwriting an existing manifest; source changes during a new run are also
rejected before writing another checkpoint. The archived driver is a historical
source snapshot, not the supported entry point from its archive location.

Use the current driver with a **fresh output directory** to recompute the
evidence below. Trying to sample into this committed original-run directory
with the hardened driver deliberately fails its provenance check. Analysis,
summary and plotting modes can still read the original CSV/JSON evidence.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
python benchmarks/tier2_literature/path_integral.py --mode deterministic \
  --output runs/tier2-literature-path-integral

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
python benchmarks/tier2_literature/path_integral.py --mode sample --workers 2 \
  --output runs/tier2-literature-path-integral

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
python benchmarks/tier2_literature/path_integral_analysis.py \
  --output runs/tier2-literature-path-integral

# Run timing after other compute jobs finish; then render the figure.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
python benchmarks/tier2_literature/path_integral.py --mode timing \
  --output runs/tier2-literature-path-integral
PYTHONPATH=src python benchmarks/tier2_literature/path_integral.py --mode plots \
  --output runs/tier2-literature-path-integral
```

The result directory contains CSV/JSON source data and a PNG/SVG figure.
Local `.npz` trajectory restart checkpoints are rebuildable caches and are
not part of the committed evidence. Neither periodic winding sectors,
interacting-boson thermodynamics nor fermionic PIGS are validated here.
