# Subspace quantum correction of RPMD rates

RPMD treats the environment of a reaction well (zero-point energy, adiabatic following,
recrossing) but misses deep tunnelling, resonances and coherence along the reaction
coordinate. The subspace correction keeps the full-dimensional RPMD rate and fixes only
that defect with an exact quantum calculation in one to three coordinates:

```latex
k \approx k_{\mathrm{RPMD}}(\text{full system}) \times C_S, \qquad
C_S = \frac{k_{\mathrm{QM}}(H_S)}{k_{\mathrm{RPMD}}(H_S)}
```

Numerator and denominator use the same subspace Hamiltonian `H_S` and the same reactant
normalization, so `C_S` contains only the RPMD method error inside `S`. The transfer is
multiplicative: the absolute rate of `H_S` may differ from the full system's by orders of
magnitude, but a shared prefactor cancels in the ratio. `C_S` is exact when `S` is the
whole system and an approximation otherwise. Nothing in this workflow marks a rate as
physically converged.

## One command from a rate run

```bash
pyrpmd subspace-correct correct.toml
```

`correct.toml` names a finished rate run (`[source] rate_run = "..."`) and optional
`[path]`, `[subspace]`, `[quantum]`, `[rpmd]` and `[output]` settings
(`benchmarks/clhbr_subspace/correct.toml` is a complete example). The rate run must have
been made with

```toml
[output]
centroid_path = true
```

which makes every umbrella window of the rate PMF stage accumulate, for each recorded ring
polymer aligned onto the transition-state geometry (mass-weighted Kabsch, after assigning
the nearest equivalent reaction path): the mean centroid, the centroid covariance, the
bead-about-centroid covariance `Sigma(xi)` and the mean interatomic distances. The moments
are additive, so replicas and checkpoint resumes combine exactly; they are written to
`centroid_path.npz` and read back with `mean_path_from_rate_run(run_dir)`. The switch is
analysis-only (it is not part of the physics fingerprint and does not change the
trajectories) but disables one fused native fast path, so it is off by default.
Extend the umbrella windows past the dividing surface (e.g. `stop = 1.45`) if the
reaction path bends after the transition state.

The command then runs:

1. **Path source.** `[path] source = "centroid"` is the historical mean-centroid path.
   `"bead_weighted"` shifts each locked-window centroid along the transverse principal
   eigenvector of the bead covariance `Sigma(xi)` by its bead RMS `sqrt(lambda_max)`, with
   the sign chosen from the local curvature. `"instanton"` uses the folded half-ring from
   `pyrpmd.instanton.optimize_instanton` below the crossover temperature and records a
   centroid fallback at or above `T_c` (or when the orbit fails to converge). `"least_action"`
   scans a one-parameter interpolation between the sampled MEP and its endpoint chord and
   selects the smallest one-dimensional WKB action. All sources retain their selection and
   fallback diagnostics in `subspace_pipeline.json`.
2. **Mean path and path to both asymptotes.** For the centroid source, windows where the
   aligned centroids stay compact (spread at most twice that near the transition state) are
   kept; near the reactant asymptote the fragments rotate freely and a mean geometry has no
   meaning. `xbar(xi)` is fitted by local quadratic regression in `xi` (window means carry
   transverse noise). For a linear reference the mean path is built from mean bond lengths
   along the axis: the Kabsch rotation about the axis is undetermined, and axial projections
   of librating light atoms are shorter than the bonds. Every selected core is then continued
   by steepest-descent path following (minimum on a sphere around each point, forward hemisphere, rigid-body modes
   excluded; uphill segments use the hyperplane minimum), with step halving at sharp turns.
3. **Curvature × width.** From `Sigma(xi)` along the sampled windows
   (`rho = sqrt(2 n^T Sigma n)` along the curvature normal) and, along the whole path, with
   the harmonic bead width of the mode along the normal. Above the threshold (default 0.5)
   a 2D subspace is added.
4. **1D model** along the path (normal-cut vibrationally adiabatic, rigid-body modes
   removed) and its `C_S`. With large curvature the normal cuts cross inside the bend and
   the adiabatic potential develops a cusp; `C_S(1D)` is then not trustworthy, which is
   what the curvature test says.
5. **2D model**: the plane through the transition state spanned by the rate dividing-surface
   normal (mass-weighted `grad xi`, so the 2D dividing line is the rate's `xi = 1` surface)
   and the principal direction of the path within the complement; for a collinear atom
   transfer this is exactly the collinear plane. The curvature-normal candidate is reported
   with its overlap with that plane. Absorbing regions are keyed to arc length along the
   path, so hairpin channels that leave the grid on the same side are both absorbed; the
   rate is normalized by the quantum line density on the reference slice, matching RPMD
   constrained TI along axis 0; an energy-truncated DVR keeps the dense exact route
   affordable. Both halves of `C_S` share one `V_S`: it is projected on a grid
   `refinement` (3) times finer than the DVR; the DVR uses every third node and RPMD the
   spline of the fine grid (a spline through DVR nodes alone rings across narrow valleys).
   The box only extends to a reference slice where the reactant channel is within
   `reference_kt` (0.3 kT) of its asymptote — `C_S` needs both rates on the same slice,
   not at infinity — and products are absorbed once the path is `product_drop_kt` (15 kT)
   below its maximum. The default grid is anisotropic: a third of the narrowest
   in-plane vibrational width along the transfer axis, a quarter of the shortest de
   Broglie wavelength along the path axis.
6. **Nested sensitivity** `ln(C_2D/C_1D)` selects the dimension, and
   `rate_quantum_corrected.json` is written next to the run. Every intermediate result is
   in `subspace_pipeline.json`.

**Exo- or endothermic reactions.** The box holds deep states of the lower channel whose net
flux vanishes but whose Boltzmann weight is enormous (`exp(25)` for Cl + HBr at 300 K); any
absorber reflection then swamps the reactive flux. Both models therefore restrict the
thermal flux operator to energies above the higher channel threshold minus one `kT`
(`exact_quantum_rate(..., energy_floor=...)`), which is exact in the continuum limit because
the cumulative reaction probability vanishes below that threshold. Without it the Cl + HBr
1D rate changes by 10% and the 2D rate by orders of magnitude when the absorber changes.

**Cl + HBr (BrHCl_LEPS1, 300 K)** is worked through in
[docs/validation/2026-11-08-clhbr-subspace](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-08-clhbr-subspace/README.md):
curvature × width is 1–12 at the H-transfer corner, the 1D model develops a cusp there
with the first, kinked path (normal cuts cross; with the corrected path C_S(1D) = 0.99), and
in the collinear plane RPMD matches the exact rate within its statistical error
(C_S(2D) = 1.02 ± 0.29 in the pipeline run, 0.9 ± 0.2 with a second RPMD run), with the 2D
quantum rate converged to 2.5% in the grid. The corrected rate is RPMD × C_S(2D).

## Workflow

1. **Build `H_S`** (`pyrpmd.analysis.subspace_model`). Coordinates are mass weighted and
   axis 0 is the reaction coordinate; the dividing surface is a value on axis 0.
   - `model_from_function(V, axes, masses=..., dividing_surface=...)` — the Hamiltonian
     already has 1–3 coordinates.
   - `project_linear_subspace(V, grad, reference, masses, basis, axes, environment=...)`
     — a linear subspace of a larger system. The complement is `"frozen"`,
     `"relaxed"` (minimized) or `"adiabatic"` (relaxed plus its harmonic zero-point
     energy, the default).
   - `project_path_subspace(V, grad, path, masses, axis)` — a curved 1D path, for
     example the ring-polymer mean path from constrained-centroid sampling
     (`ring_polymer_mean_path`), with a normal-cut adiabatic bath and arc-length
     coordinate.  `zero_point="harmonic"` retains the historical projected-Hessian
     frequencies; `zero_point="rph"` uses the same quadratic reaction-path
     potential Hessian `N.T @ H @ N` on the linear normal cut (no additional
     gradient--curvature subtraction); and `zero_point="dvr"` solves
     independent sinc-DVR cuts along the local normal modes.  For
     `environment="free_energy"`, the DVR option uses the thermal cut free energy
     instead of the ground-state energy.  `dvr_points`, `dvr_half_width`, and
     `dvr_spacing` are recorded in the model provenance so the normal-cut
     convergence can be reported.
2. **Exact quantum rate** `exact_quantum_rate(model, beta)`
   (`pyrpmd.analysis.subspace_quantum`). Thermal flux–flux formula with an absorbing
   potential on a sinc-DVR grid:
   - `method="sylvester"` (default up to 1600 grid points) solves
     `H†X − XH = iF` once; there is no time cutoff, so long-lived resonances are exact.
   - `method="flux_eigenstates"` (larger 2D/3D grids) finds the significant thermal flux
     eigenstates by Lanczos with a Chebyshev Boltzmann operator and propagates them with a
     split operator. Norm still trapped at `max_time` is assigned by the absorption
     branching ratio of the last half of the run; the estimates at T/2, 3T/4 and T are
     reported as `diagnostics["extrapolation"]`.
3. **RPMD rate of the same `H_S`** `ring_polymer_rate(model, beta, nbeads)`
   (`pyrpmd.analysis.subspace_rpmd`): centroid thermodynamic integration (exact mean
   force, HMC with exact free ring-polymer propagation and Metropolis test) plus
   recrossing trajectories. The reported error combines walker statistics, a TI
   quadrature estimate and the transmission bootstrap.
4. **Correction factor** `subspace_correction(model, beta, nbeads)` returns a
   `SubspaceCorrection` (`factor`, `log_standard_error`, both sub-results) that can be
   saved with `.write(path)`. Use the bead number of the full RPMD run.
5. **Apply it.** `apply_subspace_correction(rate, correction, standard_error=..., beta=...)`
   for a rate in hand, or `correct_rate_run(run_dir, correction)` for a saved PyRPMD
   `rate`/`surface_rate` run. The latter checks `beta` against `input/resolved.json`,
   never touches the source files, records their SHA-256 and writes
   `rate_quantum_corrected.json`. A run whose rate is not publishable keeps
   `corrected_rate_*` empty and reports only the corrected estimate.

```python
import numpy as np
from pyrpmd.analysis import (project_linear_subspace, subspace_correction,
                             correct_rate_run)

sub = project_linear_subspace(V, grad, x_ts, masses, basis_mw, [np.linspace(-150, 150, 301)],
                              environment="adiabatic", dividing_surface=0.0)
corr = subspace_correction(sub, beta, nbeads=32)
corr.write("C_S_300K.json")
correct_rate_run("runs/rate-300K", corr)
```

## Choosing the subspace

`S` must contain the RPMD defect, not merely the strongly coupled modes. The research behind
this module found that static coupling strength is the wrong criterion: a mode that follows a
smooth path adiabatically is handled by RPMD itself, so adding it to `S` changes `C_S` by only
a few percent. Two checks decide:

- `curvature_width_criterion(curvature, width)` — the path curvature in mass-weighted
  coordinates times the transverse quantum width (√2 × bead RMS). Values ≪ 1 let a 1D path
  carry the coupling; values near 1 mean the vibration straddles the bend (collinear H + H₂:
  0.80–1.01) and the transverse coordinate belongs in `S`. The 0.5 threshold is provisional.
- `nested_subspace_sensitivity([C_1D, C_2D, ...])` — the change of `ln C_S` when a coordinate
  is added. It is the final verdict and costs one larger quantum calculation.

[Subspace transfer theory](subspace-transfer-theory.md) derives why: static environment effects
(zero-point energy, imaginary-time corner cutting), which curvature × width measures, cancel from
the transfer error; what remains is the environment's influence on RPMD's dynamical error. It
also gives the first-order-exact `environment="free_energy"` projection and the finite-bead
factor `harmonic_bead_factor` for environment modes.

## Numerical parameters to converge

| Parameter | Where | Symptom when too small |
| --- | --- | --- |
| Box length and absorbing-potential width/strength | `exact_quantum_rate(cap_width=, cap_strength=)` | Reflection from the absorber; at 200 K a 16 bohr Eckart box is 0.7% high, 32 bohr 0.06% |
| Grid spacing | model axes | Rate drifts when refined |
| Truncation energy (energy-truncated DVR) | `exact_quantum_rate(truncation_energy=)`, pipeline `truncation_kt` and `truncation_min` (Eh) | The removed points are hard walls. The rate converges only once the margin above the local valley floor exceeds about six quanta of the stiffest transverse mode (walls at ~5 ground-state widths). A separable 2D Eckart model (transverse 0.010 Eh) at 200 K is 12% low with 0.016 Eh above barrier + ZPE, 1% low at 0.03 Eh, and exact (1 − 4×10⁻⁶) at 0.07 Eh. The diagnostic `truncation_margin` reports the smallest margin. The 30 kT default is too small for stiff modes at low temperature: set `truncation_min` to about six quanta |
| Propagation time and step | `max_time`, `dt` (iterative route) | `relative_extrapolation_spread` large |
| TI windows | `ring_polymer_rate(windows=)` | `ti_quadrature_log_error` comparable to the statistical error |
| Beads | `nbeads` | `C_S` should correct the same finite-`P` estimator as the full run |
| Sampling | `walkers`, `sweeps`, `trajectories` | `log_standard_error` |

Ring polymers that leave the model grid sample an extrapolated spline; this is reported as
`spline_outside_fraction` and must be zero for a gridded (projected) model.

## Validation

`examples/subspace_correction/eckart_bath_2d.py`: symmetric Eckart barrier (V₀ = 0.425 eV,
a = 0.66 bohr, m = 1060) with a vibration that softens by 30% and shifts by 0.2 bohr at the
barrier. The 1D subspace is projected from the 2D potential (relaxed vibration plus its
zero-point energy). Results and settings:
[docs/validation/2026-11-07-subspace-correction](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-07-subspace-correction/README.md).

Unit tests (`tests/test_subspace_correction.py`) check the exact solver against the analytic
Eckart transmission, the iterative route against the dense route, a separable 2D rate against
its 1D rate, single-bead RPMD against classical TST, the projections against analytic
adiabatic potentials, and the provenance and temperature checks of `correct_rate_run`.

## Relation to the other quantum-correction tools

[Quantum correction](quantum-correction.md) covers response-level (spectral) corrections, the
online Ehrenfest propagator and flux-side curve transfers. For thermal rate constants use this
module. `hybrid_flux_side_correction(..., mode="ratio")` and
`adaptive_quantum.matched_rate_defect(..., mode="ratio")` apply the same multiplicative
transfer to curves or plateau values computed elsewhere; their additive default is not
invariant to the subspace model's absolute rate.
