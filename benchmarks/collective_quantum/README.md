# Goal-oriented collective/nonlinear quantum-coordinate candidates

Stacked research follow-up to PR #137 (`d035f8f279d3603bf0af54c4a32b591748960173`).
No default integrator or existing selector is overwritten. All candidate and
propagation results keep `physical_converged=False`.

## What changed

`pyrpmd.analysis.goal_quantum_coordinates` separates three questions:

1. Which force nonlinearities communicate with the declared target? A
   mass-weighted polynomial-kernel screen learns collective directions rather
   than collecting nonlinear force residuals from every coordinate. Square and
   product features detect even-symmetry partners such as `x*y**2`. Independent
   chain/path groups are held out together; beads are never split across groups.
   A nested-group RBF refit can test a nonlinear core plus affine environment.
2. Can the coordinates form a nonsingular chart? Learned graph coordinates,
   mass-weighted arc length, and the triatomic log-bond-transfer chart have
   explicit full-coordinate maps and kinetic metrics. Noninvertible features
   such as `y**2` are NOT treated as global quantum coordinates.
3. Does promotion of a remaining mode/group measurably change the SAME target?
   `assess_promotion` and `choose_core_from_probes` consume independently blocked
   matched-prediction differences. Missing tests, mismatched targets, incomplete
   kinetic couplings and uncertain changes withhold approval. Joint promotions
   must be explicitly listed. The pointwise-normal envelope is a diagnostic,
   not a simultaneous confidence bound or an a posteriori quantum error bound.

The force screen alone does not establish that a mode needs exact quantum
propagation. A high-frequency harmonic spectator can have large zero-point
fluctuations while contributing no RPMD error to a linear harmonic response.
Conversely a low-variance mode can control a tunnelling path or a resonance.

## Kinetic operator and quantum reference

`pyrpmd.analysis.curvilinear_quantum` supplies full-chart transforms and small
1--3D tensor-product Q1 finite-element references for

    H = -hbar**2/(2*rho) d_a[rho G**ab d_b] + V,
    norm = integral rho * abs(psi)**2.

All mixed derivatives are retained. The mass matrix defines the Hilbert-space
inner product. The measure `rho` is an independent required input: after
eliminating global molecular rotations it is not generally sqrt(det(g)).
The triatomic J=0 bond chart uses `rho = r1**2*r2**2*sin(theta)` and the exact
Wilson inverse-mass matrix. The alternative full chart is
`(log(r1/r2), log((r1+r2)/length_unit), theta)`; implementation distances use
atomic units (length_unit = 1 bohr). Collinear theta endpoints are rejected.
Finite-J rotational/Coriolis operators are not implemented here.

`propagate_curvilinear` evolves a wavefunction in the retained finite eigenbasis,
using the FE mass norm. It rejects excessive eigenbasis projection loss rather
than silently renormalizing it. `hbar` is mandatory and must match the reference.
This is a closed full-chart propagator, NOT an online quantum/RPMD bath closure.

A coordinate transformation is not a dimensional reduction. In particular,
`G[active,active]` is not generally the inverse of the induced covariant metric
on a frozen-environment surface. Neither operation establishes a reduced
quantum Hamiltonian; conditional quantization introduces additional couplings.

## Actual measured results

Reanalysis of preserved PR #134 pilots (NOT newly generated trajectories):

| Target/model | Candidate dimension | Held-out target-force error |
|---|---:|---:|
| One Morse coordinate response | 1 | 4.69% |
| All three Morse coordinate responses | 3 | same target scope retained |
| Fermi x response with xy^2 coupling | 2 | <3e-15 |
| Curved Eckart coordinate response | 2 | 8.82% after nested-group flexible fit |

The old all-coordinate Morse energy test and the new one-coordinate response
are different targets; their dimensions are NOT directly comparable accuracy
claims. Coordinate-rotation projector differences are below 7e-15. The source
pilots remain subject to their original sampling convergence limitations.

New restrained PIMD pilot runs on the historical BrHCl_LEPS1 PES:
256 independently initialized chains/temperature, P=32, 800 HMC cycles, seed
900001+T, target `log(r_BrH/r_HCl)`. Original pilot equations from PR #137 are
used. Only centroid geometries and bare-PES generalized forces enter screening.
No collision outcomes or full quantum answer enter selection.

| T / K | Default force-screen dimension | Held-out force error | 32 group-bootstrap dimensions |
|---|---:|---:|---|
|150|1|0.596%|32 x 1D|
|300|1|0.484%|32 x 1D|
|600|2|1.440%|29 x 2D, 3 x 1D|

Raising the retained nonlinear fraction from 0.99 to 0.995 changes the two
low-temperature candidates to 2D; lowering it to 0.95 gives 1D at all three
T. No threshold was fitted to a full quantum rate. Reference-metric variations
are large (2.80, 4.11, 6.06 in the recorded normalized diagnostic). Thus none
of these candidates is approved for a molecular quantum rate. Restrained-pilot
equilibrium has not been established. This is not a replacement for the previous
J=0 convergence failure and missing finite-J quantum evidence.

Full-coordinate nonlinear kinetic verification: harmonic oscillator under the
shear `x=s, y=b+0.6*s**2`; analytic ground energy is 1 in hbar=m=omega=1 units.
At 31/51/81/121 nodes per axis the correct-metric ground errors are
2.414%, 0.903%, 0.362%, 0.165%; the wrong-flat-mass errors are
6.215%, 5.489%, 5.238%, 5.148%. This tests coordinate consistency, not rate error.
A learned curved path needs one geometric progress variable, while its
normalized core/environment metric cross term reaches 0.899. That is an
explicit warning against inferring dynamical decoupling from geometric rank.

## Reproduce

From a checkout of this PR with numpy/scipy/numba/pytest; matplotlib for plots:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m pytest -q tests/test_goal_coordinates.py tests/test_curvilinear_propagation.py

# New independent molecular pilots; output defaults to results/.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/collective_quantum/molecular_pilot.py \
  --n 256 --cycles 800 --temperatures 150 300 600 --output results/molecular_main

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/collective_quantum/check_selection_stability.py

# Geometry/reference checks; archived pilots are optional, not fabricated.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/collective_quantum/run.py --output results

# To also re-evaluate the three archived PR134 pilots:
# .../run.py --pilot-dir /path/to/pilot_npz_directory --output results
python benchmarks/collective_quantum/plot_report.py
```

The downloadable archive includes the preserved pilot files with hashes, fresh
molecular pilot outputs, compact summaries and local extraction helpers. In
that archive use `--upstream-dir inputs/upstream`; in the actual PR checkout the
existing `benchmarks/noncollinear_quantum` sources are used by default.

34 added tests passed locally. The full repository suite was not run. No new
selective-quantum corrected gas-phase rate was computed. A force screen, graph
chart, and finite-grid kinetic implementation are not a proof of a universal
low-dimensional quantum/RPMD theory.

## Background references

- Craig and Manolopoulos, JCP 121, 3368 (2004), doi:10.1063/1.1777575:
  distinction between exact equilibrium and approximate real-time dynamics.
- Dynamics of complex molecular systems with numerical kinetic energy operators
  in generalized coordinates (2006), ScienceDirect S0301010406001534.
- Numeric kinetic energy operators for molecules in polyspherical coordinates
  (2012), PubMed 22779586.

The new screening/probe workflow is experimental; these papers do not prove
its transferability or the sufficiency of the reported molecular candidates.
