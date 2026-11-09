# Subspace transfer error on 2D Eckart model families

Date: 2026-11-09. Theory: [docs/subspace-transfer-theory.md](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/subspace-transfer-theory.md).
Scripts: `benchmarks/subspace_theory/` (`transfer.py` one point, `plan.py` the matrix,
`aggregate.py` the table, `elasticity.py` the 1D elasticity). Computed on GitHub Actions:
`.github/workflows/subspace-transfer.yml` (run 42573202493) and
`.github/workflows/h3-subspace-truncation.yml`; every point is an artifact of those runs.

## What is measured

For a full 2D Hamiltonian H and its 1D projection H_S along the valley floor
(`project_path_subspace`, zero-point `"adiabatic"` and `"free_energy"` environments), the
transfer error of the corrected rate is

    Delta = ln C(H) - ln C(H_S),   C = k_QM / k_RPMD,

which is `ln(k_QM(H) / (k_RPMD(H) C_S))`, the error of the subspace-corrected rate when H is
taken as the full system. G_X = ln k_X(H) − ln k_X(H_S) is the environment response of
method X (rates in the same units), and Delta = G_QM − G_RPMD.

Models (mass 1060 au in s and y; V₀ = 0.425 eV, a = 0.66 bohr, T_c = 413 K for the symmetric
barrier; asymmetric: products 0.2 eV lower):

| Family | Coupling | Parameter |
| --- | --- | --- |
| curvature | V_b(s) + ½ m ω_y² (y − h sech²(s/a))², ω_y = 0.010 Eh | h = 0, 0.3, 0.6, 0.9 bohr |
| frequency | V_b(s) + ½ m ω(s)² y², ω(s) = ω_B (1 − 0.3 sech²(s/a)) | ω_B = 0.0035, 0.007, 0.014 Eh |

Temperatures 600, 300 and 200 K (1.45, 0.73 and 0.48 T_c). Bead number
P = max(16, 2βħω_max rounded up to 8): 16, 24, 32 for ω = 0.010 Eh. RPMD: 48 TI windows,
64 walkers × 1000 HMC sweeps, 2048 trajectory pairs. Both 1D halves of C_S use the same V_S;
the 2D RPMD uses the analytic potential.

## Exact 2D rates: convergence

The energy-truncated DVR needed far larger truncation energies than the 25–30 kT used before.
The removed points are hard walls, and a transverse vibration squeezed between them gains
confinement energy:

| Model, 200 K | Walls above barrier + ZPE (Eh) | k_QM(2D) relative to the reference |
| --- | ---: | ---: |
| separable (h = 0) | 0.016 / 0.03 / 0.05 / 0.07 | 0.881 / 0.990 / 0.99984 / 0.999996 (reference: exact 1D rate) |
| curvature h = 0.9 | 0.016 / 0.025 / 0.03 / 0.05 / 0.07 / 0.10 | 0.566 / 0.814 / 0.869 / 0.995 / 1 / 1.001 (reference: 0.07 Eh) |

Convergence needs walls about six quanta of the stiffest transverse mode above the valley
floor (about five ground-state widths). All points below use 0.07 Eh (seven quanta of
ω_y = 0.010 Eh). They agree between DVR spacings 1.0 and 0.8 (0.1 and 0.08 bohr in s) to
≤ 1.2 × 10⁻³, and with the 1D Eckart rate to 4 × 10⁻⁶ for h = 0.

## Curvature family

| Barrier | h | κρ (200 K) | Δ 600 K | Δ 300 K | Δ 200 K |
| --- | ---: | ---: | ---: | ---: | ---: |
| symmetric | 0 | 0 | +0.008 ± 0.007 | −0.015 ± 0.012 | +0.004 ± 0.019 |
| symmetric | 0.3 | 0.40 | +0.022 ± 0.006 | +0.024 ± 0.011 | +0.028 ± 0.020 |
| symmetric | 0.6 | 0.79 | +0.078 ± 0.007 | +0.111 ± 0.013 | +0.068 ± 0.020 |
| symmetric | 0.9 | 1.19 | +0.163 ± 0.011 | +0.156 ± 0.017 | +0.162 ± 0.024 |
| asymmetric | 0 | 0 | −0.004 ± 0.008 | +0.004 ± 0.018 | +0.062 ± 0.043 |
| asymmetric | 0.3 | 0.40 | +0.030 ± 0.009 | +0.015 ± 0.017 | +0.065 ± 0.043 |
| asymmetric | 0.6 | 0.79 | +0.062 ± 0.008 | +0.079 ± 0.016 | +0.061 ± 0.037 |
| asymmetric | 0.9 | 1.19 | +0.129 ± 0.013 | +0.124 ± 0.018 | +0.066 ± 0.031 |

(Zero-point projection; the free-energy projection agrees within the errors, as expected for
βħω_y ≥ 5.) The six separable controls are within 1.5σ of zero. With walls at 0.10 Eh instead
of 0.07 Eh the two checked points change by +0.002 and 0.000 in Δ. The curvature points fit

    symmetric:   Delta = (0.210 ± 0.009) h² = (0.148 ± 0.006) (κρ)²,   χ²/dof = 1.2 (9 points)
    asymmetric:  Delta = (0.161 ± 0.010) h² = (0.112 ± 0.007) (κρ)²,   χ²/dof = 1.4 (9 points)

There is no linear term and no temperature dependence for the symmetric barrier (h = 0.9:
+0.163, +0.156, +0.162 at 600, 300, 200 K). The asymmetric 200 K point at h = 0.9 is 2σ below
the other two temperatures. Meanwhile the environment response itself grows steeply as the
temperature falls (h = 0.9 symmetric: G_QM = 0.37, 1.09,
2.27 at 600, 300, 200 K). RPMD reproduces 50–94% of the corner-cutting gain G_QM (G_RPMD/G_QM), and the
missing part is a temperature-independent h² term.

Where it comes from: at 600 K, ln C_S of the 1D path model falls by 0.12 between h = 0 and
0.9, while ln C(H) rises by 0.04. Most of Δ is a change of the 1D model's RPMD error caused by
the normal-cut zero-point energy. On the tilted flanks the normal-mode frequency is
ω_y √(1 + c'²), which reshapes V_S. A projection that carries the curvature coupling of the
reaction-path Hamiltonian is the obvious next step and was not tried.

## Elasticity of C (1D Eckart, barrier height scaled 0.8 / 1.0 / 1.25)

| Barrier | 600 K | 300 K | 200 K |
| --- | ---: | ---: | ---: |
| symmetric | −0.004 ± 0.004 | −0.017 ± 0.004 | −0.037 ± 0.005 |
| asymmetric | −0.010 ± 0.005 | −0.013 ± 0.005 | −0.027 ± 0.009 |

C changes slowly with the barrier height, so errors of the projected potential that act like a
barrier change are suppressed by a factor of 25–250 in C_S. The estimate Δ ≈ εG does **not**
describe curvature: it predicts −0.08 at h = 0.9, 200 K, against the measured +0.16.
Corner cutting is not a barrier-height change.

## Frequency family: the first-order theorem

Without curvature, the transfer is accurate (|Δ| ≤ 0.044 at all nine points, βħω_reactant
from 1.84 to 22). The theorem says the free-energy projection is first-order exact, while the
zero-point projection misses the thermal average of the coupling. The exact quantum response
G_QM (no statistical error) tests this directly:

| ω_B (Eh) | T (K) | βħω_reactant | G_QM, zero point | G_QM, free energy | first-order estimate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0035 | 600 | 1.84 | +0.1287 | −0.0001 | +0.150 |
| 0.0035 | 300 | 3.68 | +0.0269 | −0.0056 | +0.053 |
| 0.007 | 600 | 3.68 | +0.0434 | −0.0013 | +0.053 |

The free-energy model reproduces the exact 2D quantum rate to 6 × 10⁻⁵ in ln k at
βħω = 1.84. What remains is second order. The "first-order estimate" is the harmonic thermal
shift of the barrier, β[F − ZPE]‡ − β[F − ZPE]_r. In C_S the two projections agree within the
RPMD errors (Δ_zpe − Δ_fe ≤ 0.023 ± 0.02), as the elasticity |ε| ≤ 0.04 predicts: the
projection error changes the 1D rate by up to 14% but its RPMD error by less than 0.1%.

## Finite bead number

An adiabatic or free-energy H_S carries the converged environment zero-point energy, so C_S
at matched P does not contain the environment's bead error. The remaining Δ of the frequency
family is that error. Adding ln B_P (`harmonic_bead_factor`, environment frequency at the
barrier and at the reactants) removes it:

| ω_B = 0.014 Eh, 200 K | P = 8 | 16 | 32 | 48 |
| --- | ---: | ---: | ---: | ---: |
| Δ | +0.650 ± 0.021 | +0.294 ± 0.016 | +0.093 ± 0.016 | +0.070 ± 0.020 |
| −ln B_P (harmonic) | 1.162 | 0.457 | 0.136 | 0.062 |
| Δ + ln B_P | −0.51 | −0.16 | −0.04 | +0.008 |

Over the nine frequency points at the default P, χ²/n of Δ falls from 5.1 to 0.77 when ln B_P
is added. The harmonic factor is quantitative once |ln B_P| ≲ 0.15. At low P it overestimates
the bead error by 40–60%, because the ring polymer then samples the environment frequency over
a wide range of s. Use B_P as a correction near convergence and as a warning far from it.

In collinear H + H₂ (docs/validation/2026-11-08-h3-collinear-lowT), dividing by B_P collapses
the RPMD-TST rates at P = 32, 64 and 128 to within 5%, with B_P as low as 0.59.

## Truncation of the earlier exact rates

The confinement artifact also affects exact rates reported earlier in this repository.

**Collinear H + H₂** (docs/validation/2026-11-08-h3-collinear-lowT). The report used walls
0.04 Eh above the collinear H₂ ground level. Run 42575336118 extends this (spacing 2.0,
`h3_truncation_run1.json`):

| T (K) | k_QM, 0.04 Eh (report) | 0.07 | 0.10 | 0.13 | 0.16 Eh | change 0.04 → 0.16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 600 | 6.873e−5 | 6.802e−5 | 6.694e−5 | 6.664e−5 | 6.656e−5 | −3.1% |
| 400 | 5.832e−6 | 5.537e−6 | 5.420e−6 | 5.386e−6 | 5.377e−6 | −7.8% |
| 300 | 5.846e−7 | 5.268e−7 | 5.140e−7 | 5.102e−7 | 5.092e−7 | −12.9% |
| 200 | 8.765e−9 | 7.869e−9 | 7.670e−9 | 7.606e−9 | 7.590e−9 | −13.4% |
| 150 | 2.942e−10 | 2.860e−10 | 2.815e−10 | 2.799e−10 | 2.796e−10 | −5.0% |

The last step changes the rate by at most 0.2%. With the high-energy points retained, however,
the spacing-2.0 grid is no longer converged: spacing 1.75 at 0.13 Eh gives 7.848e−9 at 200 K
(+3.2%) and 2.978e−10 at 150 K (+6.4%). The earlier spacing check was made at 0.04 Eh, where
those points were missing. The spacing ladder at 0.10 Eh (run 42599202874) converges with oscillations: at 200 K the rate is 7.670, 7.914 and 7.766e−9 for spacings 2.0, 1.75 and 1.5. Converged estimates take k(spacing 1.5, 0.10 Eh) × k(2.0, 0.16)/k(2.0, 0.10); the grid uncertainty is half of the 1.75 → 1.5 change (`h3_converged.json`):

| T (K) | k_QM converged | grid ± | vs report | k_RPMD / k_QM | ÷ B_P |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 600 | 6.681e−5 | 0.3% | −2.8% | 0.651 | 0.665 |
| 400 | 5.408e−6 | 0.4% | −7.3% | 0.698 | 0.711 |
| 300 | 5.131e−7 | 0.6% | −12.2% | 0.693 | 0.723 |
| 200 | 7.685e−9 | 1.0% | −12.3% | 0.649 | 0.672 |
| 150 | 2.870e−10 | 1.9% | −2.5% | 0.530 | 0.577 |

The collinear RPMD deficit is therefore 0.65–0.72 from 600 to 200 K and 0.58 at 150 K (bead-corrected), not 0.52–0.65 as first reported.

**Cl + HBr** (docs/validation/2026-11-08-clhbr-subspace). The 2D plane model was rebuilt from
the saved rate run on a finer grid (101 × 243). With walls 0.0285 Eh above the energy floor,
as in the report, k_QM = 4.595e−4 (report: 4.686e−4 on the coarser grid). With 0.04 Eh it is
4.521e−4 (−1.6%). At 0.055 Eh the grid keeps 12 043 points, beyond the dense route on 16 GB,
and was not computed. The change is far below the ±29% RPMD error of C_S(2D); the Cl + HBr
conclusion is unchanged, and convergence beyond the 2% level is not shown.

**The dense route** keeps four instead of about eight complex N × N matrices (identical rates
to 10⁻¹⁴), which is what made the 10 263-point Cl + HBr solve fit.

## Limits

- Two model families, one mass, one transverse frequency for the curvature family.
- The fitted h² law is empirical; the theory predicts the absence of a linear term, not the
  coefficient.
- RPMD statistical errors are 0.6–4%; no point is marked physically converged.
