# Non-harmonic revalidation: Morse, Fermi, and curved Eckart

This is a new validation of the attached adaptive-tunnelling prototype, on top
of PR #133 at `2800af5d128b8db270c8ad06c32bc42b2f1af2b6`. It does not replace
concurrent online-propagation work. All quantities use `hbar=m=kB=1`.

**Do not reuse the older harmonic report's 366%/33% as method scores.** That
report mixed classical and RPMD nonlinear correlations and different full/sub
observables. Its near-degenerate harmonic test was not a Fermi resonance test.
These tests use actual finite-P RPMD, the same observable across references,
and separate full quantum audits. No full quantum answer selects the subspace
or fits a correction parameter.

## Implementation

Import `pyrpmd.analysis.adaptive_quantum` for:

- `select_active_subspace`: affine force-residual directions plus harmonic
  partner loading; explicit dimension-budget and nonpositive-curvature failure;
- `passive_embed`: statically relaxed active response plus a harmonic-reservoir
  Schur-complement kernel, with no spectral clipping or renormalization;
- `matched_rate_defect`: experimental matched normalized plateau-rate transfer,
  returning negative results unchanged and `exact_general_rate_claim=False`.

The old attachment's array-level module is hardened and integrated here. The
new validation harness uses independent sampling/reference implementations;
it does not change PyRPMD's default integrator. The loading threshold `0.05`
and nonlinear retained fraction `0.999` are heuristics, not error bounds.
Pilot samples are separate from production. A random orthogonal rotation is
applied before selection, and the recovered physical span is checked.
For these model families that span is the Morse full space or the two-coordinate
Fermi/Eckart core. A generic rotated-PES quantum-grid builder is not implemented.

## 1. Three-dimensional anharmonic Morse

`V=sum_i D_i [1-exp(-a_i q_i)]^2`, with `D=(8,6,5)`, `a=(.5,.55,.6)` and
`beta=8`. This is a separable three-dimensional anharmonic control, not a
coupled Morse model. The observable is `(q1+q2+q3)/sqrt(3)` in all calculations.

The finite sampling box is `[-3,10]^3`; a larger quantum box is checked. An
unconfined dissociating Morse canonical partition function is not assumed.
Above-dissociation population is negligible at this temperature in the stated
box. The independent quantum reference is a sine-DVR product calculation;
it is not a harmonic approximation and does not require a dense 300^3 matrix.

| Quantity | Measured value |
| --- | ---: |
| Quantum thermal energy (essentially ZPE) | 2.7872615321 |
| PIMD energy, P=32 | 2.7021077054 +/- 0.0134119040 (block SE) |
| PIMD energy, P=64 | 2.7561176497 +/- 0.0121425004 (block SE) |
| Larger quantum box/grid ground-energy change | < 6.1e-13 |
| Automatically selected dimension | 3 |

PIMD already contains ZPE. No extra ZPE correction is added to it. All three
nonlinear directions matter; returning three dimensions is **no reduction**.
Replacing all separable response blocks by quantum blocks is an algebraic
full-system limit, not a successful low-dimensional embedding result.

## 2. Fermi resonance with xy^2 coupling

`V=2x^2+y^2/2+0.35*x*y^2+(0.35^2/8+0.005)y^4
   +1.3^2/2*(b-0.15x)^2`, with `beta=6`.
Completing the square establishes stability. Independent conditional Gaussian
rejection sampling targets the exact finite-P Boltzmann distribution. Production
RPMD is unthermostatted and uses physical bead masses. The observable is `x`.

The selector recovers the `(x,y)` plane, leaving one harmonic environment mode.
The quantum reference uses 16x22x10 oscillator functions and 90 eigenstates;
20x28x14 functions and 140 states change the response by < 9e-13 relatively.

Absorption-spectrum relative L2 errors on `[1.65,2.4]`, at fixed `eta=0.04`:

| Method | P=32, 2048 paths | P=64, 1024 paths |
| --- | ---: | ---: |
| Full RPMD | 74.70% | 74.03% |
| 2D inverse-response defect | 2.10% | 2.72% |
| 2D quantum + analytic harmonic-reservoir kernel | 0.0492% | 0.0492% |
| Forced 1D harmonic core + reservoir | 123.94% | 123.94% |

The identical last-two columns for the analytic kernel are **not a bead
convergence result**: that estimator has no sampled bath dependence. Its
remaining bath response equals the analytic harmonic RPMD response. It is a
special closure for this harmonic residual bath, not a general bath solver.
The full quantum peaks at approximately 1.9185 and 2.1615 are recovered. The
inverse-defect bootstrap intervals are [1.90%,3.85%] and [2.22%,6.22%].
Time-window and especially narrower-line/broadening convergence remain open;
eta is numerical broadening, not a physical lifetime.

## 3. Curved Eckart barrier with strong bath and spectator

`V=sech(q)^2 + 1.5^2/2*[x-0.65*tanh(2q)]^2
               + 2.3^2/2*[z-0.10*x]^2`.

The q-x core is strongly, nonlinearly coupled. The spectator is weakly but
**not zero** coupled. The pilot chooses two coordinates, including the nonlinear
partner. The exact audit is open-boundary multichannel scattering in q with
oscillator transverse channels, not a finite-box bound-spectrum surrogate.
At collision energy 0.5 (below the bare unit barrier), full 3D ground-channel
transmission is about 0.0437.

Quantum thermal rates use
`k=sqrt(beta/(2*pi))/Z_bath * integral exp(-beta*E) N(E) dE`, with collision
energy measured from the lowest asymptotic vibrational threshold and consistent
reactant normalization. The reported rates have units length/time.

The actual RPMD baseline uses centroid-constrained HMC, thermodynamic integration
of mean force over `[-6,0]`, and a positive-flux-sampled flux-side transmission
plateau. Real-time momenta are freshly sampled. The main plateau is `[10,16]`;
`[8,12]` and `[12,16]` are also recorded. HMC draws within a chain are not called
independent samples; mean-force errors are formed from chain means.

The **rate** experiment uses the candidate transfer
`k_mix=k_RP,3D + k_Q,2D - k_RP,2D`. This is distinct from the bound-spectrum
passive kernel and is not claimed as a general exact rate identity.

| Setting | Full quantum rate | Full RPMD rate | 2D-corrected rate | RPMD bias | Corrected bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| beta=4, P=32 | 0.00595969049 | 0.00407855062 | 0.00556804205 | -31.56% | -6.57% |
| beta=8, P=32 | 0.000696901753 | 0.000328070117 | 0.000653302495 | -52.92% | -6.26% |
| beta=8, P=64 | 0.000696901753 | 0.000349650409 | 0.000706597041 | -49.83% | +1.39% |

The conservative delta-method sampling SE of the corrected rates is about
5.4-5.8% of the quantum rate; it is **not** a complete error bound. Common seed
numbers across differing-dimensional samplers do not establish independent
full/sub errors: both root-sum-square and sum-of-SE estimates are saved.
Forcing a 1D core at beta=8,P=32 instead overestimates by about 59.5%.

The reference refinement doubles q resolution and energy resolution, increases
bath bases 10x4 -> 14x6, and extends the lead matching region. The beta=8 full
quantum rate changes by about 0.22%. Maximum ground-channel flux-conservation
defect is 1.32e-10; the independent analytic 1D Eckart probability check has
maximum error 7.09e-5. The P=64 sampling check also changes warmup, seeds and
quadrature nodes; it does not isolate every individual source of error.

## Reproduce

Requirements for these examples: Python >=3.11, NumPy >=2, SciPy >=1.11,
Numba >=0.61, pytest >=8, matplotlib >=3.8. No optional font is distributed.

```sh
python -m pytest -q tests/test_adaptive_anharmonic.py
python examples/anharmonic_quantum/run_benchmarks.py --stage all
python examples/anharmonic_quantum/check_convergence.py
python examples/anharmonic_quantum/run_benchmarks.py --stage morse --P 64 --tag beadcheck
python examples/anharmonic_quantum/run_benchmarks.py --stage fermi --P 64 --n 1024 --tag beadcheck
python examples/anharmonic_quantum/run_benchmarks.py --stage eckart_rpmd --beta 4 --chains 128
python examples/anharmonic_quantum/run_benchmarks.py --stage eckart_rpmd --P 64 --dimension 2 --chains 128 --warmup 1000 --seed-offset 10000 --quadrature-nodes 20 --tag convergence
python examples/anharmonic_quantum/run_benchmarks.py --stage eckart_rpmd --P 64 --dimension 3 --chains 128 --warmup 1000 --seed-offset 10000 --quadrature-nodes 20 --tag convergence
python examples/anharmonic_quantum/report.py
```

Raw output goes to `results/` (the heavy trajectories are not committed). The
small checked-in `summary.json` records measured outputs; the report generator
writes `anharmonic_validation_report.html`, PNGs and a CSV from computed data.
24 new local implementation checks passed. This is not a claim that all PyRPMD
CI jobs or all physics-convergence gates have passed. All results retain
`physical_converged=False`.

Theory references: Morse, Phys. Rev. 34, 57 (1929), DOI 10.1103/PhysRev.34.57;
Craig and Manolopoulos, JCP 121, 3368 (2004), DOI 10.1063/1.1777575; Craig and
Manolopoulos, JCP 122, 084106 (2005). The active-space and defect-transfer
constructions above are experimental additions, not theorems from these papers.
