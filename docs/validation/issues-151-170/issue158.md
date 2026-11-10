# Issue 158: Cartesian transverse-channel projection

The optional channel calculation compresses the quantum calculation for an existing
two-dimensional Cartesian `SubspaceModel`. Its longitudinal coordinate is a fixed
axis of that plane. Curved valleys remain in `V(x,y)`; this implementation does not
identify the longitudinal coordinate with Frenet arc length.

At every longitudinal grid point, the transverse sinc-DVR Hamiltonian provides a
local basis. If the block-diagonal basis embedding is `U`, the channel Hamiltonian
is `U† H_2D U`. Complete inter-slice overlaps preserve the kinetic contribution of
the changing basis, including the Born–Huang term missed by using only the retained
connection matrix. Phase alignment changes phases only, so it cannot mix distinct
energies without transforming their Hamiltonian.

The RPMD denominator uses the original Cartesian model and its constant-mass
propagator, sampling measure, constrained mean force and flux normalization.
`channels_from_model(model, n_channels=N).correction(...)` accepts a finer
`rpmd_model` after checking masses, surface, reference, normalization, shared grid
nodes and potential values. Interpolation convergence remains a separate check.

## Matrix and rate oracle

A small nonseparable potential was evaluated on an 11 by 9 Cartesian grid:

```text
V(x,y) = 0.08 / cosh(x/0.8)^2
         + 0.5 * 0.22^2 * (y - 0.35*tanh(x/1.2))^2
x in [-3,3], y in [-5,5], masses = [1.7,1], beta = 5
dividing_surface = 0, reactant_reference = -1.5
```

The comparison uses a two-dimensional absorbing potential,
`0.15*max(abs(x)-1,0)^2*(1+0.04*y^2)`, reference-slice normalization and
`energy_floor=0.1`. The absorber is projected as `U† epsilon U`; it is not reduced
to a diagonal approximation. Slice normalization uses the same thermal line density
as the full DVR solver.

| Channels per slice | Rate relative to full 99-point DVR, minus one | Rate relative to the energy-truncated 63-point DVR, minus one |
| ---: | ---: | ---: |
| 1 | 1.2211305e-2 | 2.7237119e-2 |
| 2 | 5.6111262e-3 | 3.3584901e-3 |
| 4 | 3.2603396e-4 | 4.7268710e-4 |
| 6 | 1.7519396e-6 | 3.6415315e-14 |
| 9 | 2.0650148e-14 | 3.6415315e-14 |

The truncated calculation retains points with `V<=0.35`. Its local transverse
bases have different sizes; six requested channels cover every retained state.
The complete basis reproduces the full Hamiltonian, flux, reactant partition and
rate to numerical precision. Finite channel count is a quantum discretization
error and is not included in the RPMD statistical error.

These are fixed-grid numerical comparisons, not physical convergence results.
Energy-truncation walls, spatial grids, absorbers, bead count and RPMD sampling need
their own convergence checks. The curvature-family transfer error and H+H2 at
200 K have not been validated by this small test.

## Correctness and unsupported cases

The regressions cover direct Galerkin matrix projection at 1/2/4/6/full channels,
the translated-oscillator Born–Huang contribution, nondegenerate phase alignment,
full-basis flux/partition/rate equivalence, reference-slice normalization,
projected absorbers and energy truncation.

An explicitly constant inverse-metric grid reduces exactly to longitudinal mass
`m/g`; the quantum and RPMD routes use that same mass, including normalization.
The constant-metric trajectory map is reversible and preserves volume. Variable
metric grids and metric callables are rejected before rate computation: the old
explicit variable-mass Verlet/HMC path did not provide the required reversible
proposal or consistent constrained sampling and flux.

For a Frenet embedding `r(s)+Q*N(s)` with inward normal `N=T'/kappa`, the geometric
Jacobian is `1-kappa*Q`. A sign change anywhere across the normal interval is
rejected, including a pole between grid points. Nonzero-curvature Frenet rate
calculations remain unsupported; the usable alternative is the original Cartesian
plane. Disabling inter-channel coupling produces a diagnostic matrix that cannot
be paired with the coupled Cartesian RPMD denominator.

## Pipeline and resource checks

The input option `[subspace] channels = N` is used by the dimension-2 pipeline.
The default `None` keeps the original full-DVR route. Tests verify that the channel
count, quantum options and fine-grid RPMD model reach the channel correction API,
invalid channel counts fail before the rate solver, and changing the channel count
invalidates a previously cached factor.

`dense_limit` accepts a positive integer and is a strict bound on the active
channel matrix after energy truncation. A 99-state matrix is rejected at limit 98;
the same model truncated to 63 states runs at limit 63 and is rejected at limit 62.
The legacy full-DVR limit policy is unchanged by this channel check.

Run the targeted regressions with the existing test environment:

```bash
python -m pytest -q tests/test_subspace_channels.py tests/test_subspace_pipeline_channels.py
```

The targeted result is 38 passed. The existing subspace correction and transfer
tests also passed in the isolated implementation checkout (31 passed). Neither
result establishes the requested full scientific acceptance of issue 158.
