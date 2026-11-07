# Subspace rate correction: 2D Eckart barrier with a coupled vibration

Date: 2026-11-07. Code: `pyrpmd.analysis.subspace_{model,quantum,rpmd,correction}`.

## What was tested

`k_RPMD(2D) × C_S(1D)` against the exact 2D quantum rate. The 1D subspace is projected from
the 2D potential with `project_linear_subspace(..., environment="adiabatic")`: the vibration is
relaxed at each reaction-coordinate point and its harmonic zero-point energy added. `C_S` is
the exact-over-RPMD ratio for that 1D Hamiltonian.

Model (atomic units, both coordinates mass 1060):

```latex
V(s, y) = V_0\,\mathrm{sech}^2(s/a) + \tfrac12 m\,\omega(s)^2\,[y - c\,\mathrm{sech}^2(s/a)]^2,
\qquad \omega(s) = \omega_0\,[1 - 0.3\,\mathrm{sech}^2(s/a)]
```

V₀ = 0.425 eV, a = 0.66 bohr, ω₀ = 0.010 Eh, c = 0.2 bohr. The vibration softens by 30% at the
barrier (zero-point energy falls by 1.5 mEh) and its equilibrium shifts by 0.2 bohr (path
curvature).

## Result

| T (K) | P | RPMD(2D) / exact | C_S (1D) | required C = exact / RPMD | corrected / exact |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 16 | 0.710 ± 0.005 | 1.369 ± 0.027 | 1.408 | 0.972 ± 0.021 |
| 200 | 24 | 0.544 ± 0.006 | 1.753 ± 0.039 | 1.839 | 0.953 ± 0.025 |

Errors are one standard error from RPMD sampling and TI quadrature (`C_S` error = `C_S` ×
`log_standard_error`). The 1D correction removes 90% of the RPMD error at both
temperatures. The residual, −3% and −5%, is 1.3σ and 1.9σ and has a physical origin: the 1D
adiabatic model has no corner cutting through the curved valley. The two selection checks
agree that a 1D subspace is only just adequate here:

- curvature × width: κ_max = 0.028 (mass-weighted) times ρ = √2 × bead spread of the
  vibration about its centroid gives κρ = 0.29 (300 K) and 0.31 (200 K), below the provisional
  0.5 threshold but not ≪ 1;
- nested sensitivity: ln(C_2D / C_1D) = 0.028 and 0.048, against a 0.05 tolerance.

## Settings

| Quantity | Value |
| --- | --- |
| 2D quantum grid | s ∈ [−8, 8] bohr, 201 points; y ∈ [−1.4, 1.6] bohr, 31 points (6231 points, `flux_eigenstates` route) |
| Absorbing potential | default: quadratic, 20% of the s range at each end, strength 30 v_th / width |
| Trapped-norm extrapolation spread | 3×10⁻⁴ (300 K), 3×10⁻³ (200 K) |
| 1D subspace grid | s√m on the same 201 points (dense Sylvester route) |
| RPMD (2D and 1D) | 48 TI windows × 32 walkers, 200 + 1200 HMC sweeps, 1024 paired recrossing trajectories |
| Wall time (4 cores) | 300 K: 14 s exact 2D, 89 s RPMD 2D, 82 s `C_S`; 200 K: 127 s, 185 s, 127 s |

The exact 2D rates were recomputed after the flux eigensolver was changed from ARPACK to block
subspace iteration: identical to 10⁻¹⁰, and 9–11 s instead of 14–127 s.

## Reproduce

```bash
PYTHONPATH=src python examples/subspace_correction/eckart_bath_2d.py --temperatures 300 200 --beads 16 24 --sweeps 1200 --trajectories 1024 --out eckart_bath_2d.json
```

The run recorded in `eckart_bath_2d.json` used seed 7 (2D RPMD) and 8 (1D RPMD). A shorter pilot
at 300 K (400 sweeps, 512 trajectories) gave corrected/exact = 1.002 ± 0.032.

## Exact solver checks (unit tests)

| Check | Result |
| --- | --- |
| Dense route vs analytic symmetric Eckart, 1000 K / 300 K / 200 K, 16 bohr box | 1.0001 / 1.0009 / 1.0070 |
| Same at 200 K with a 32 bohr box | 1.0006 (the 16 bohr error is absorber reflection) |
| Iterative vs analytic, 1000 K / 300 K / 200 K | 1.0000 / 0.9997 / 1.0057; reported spread 1×10⁻⁴ / 5×10⁻⁴ / 7×10⁻³ |
| Separable 2D (dense) and 3D (iterative) vs 1D on the same s grid | 1 ± 10⁻⁸ and 1 ± 3×10⁻³ |
| Single-bead RPMD vs classical TST, 1000 K | 0.9993, κ = 1 |

## Not covered

No `physical_converged` flag is set. Bead-number convergence of `C_S`, absorber and grid
convergence beyond the checks above, and systems where κρ ≈ 1 (collinear H + H₂, where the
research note found a 1D correction recovers only 16–62% of the defect) need their own runs; for
those, the 2D subspace (q, s) is the intended use.
