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
     coordinate.
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

## Numerical parameters to converge

| Parameter | Where | Symptom when too small |
| --- | --- | --- |
| Box length and absorbing-potential width/strength | `exact_quantum_rate(cap_width=, cap_strength=)` | Reflection from the absorber; at 200 K a 16 bohr Eckart box is 0.7% high, 32 bohr 0.06% |
| Grid spacing | model axes | Rate drifts when refined |
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
[docs/validation/2026-11-07-subspace-correction](validation/2026-11-07-subspace-correction/README.md).

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
