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
| symmetric | 0.6 | 0.79 | +0.078 ± 0.007 | CURV_06_300 | +0.068 ± 0.020 |
| symmetric | 0.9 | 1.19 | +0.163 ± 0.011 | +0.156 ± 0.017 | +0.162 ± 0.024 |
| asymmetric | 0 | 0 | −0.004 ± 0.008 | +0.004 ± 0.018 | +0.062 ± 0.043 |
| asymmetric | 0.3 | 0.40 | +0.030 ± 0.009 | +0.015 ± 0.017 | +0.065 ± 0.043 |
| asymmetric | 0.6 | 0.79 | +0.062 ± 0.008 | CURV_A06_300 | CURV_A06_200 |
| asymmetric | 0.9 | 1.19 | CURV_A09_600 | CURV_A09_300 | CURV_A09_200 |

(Zero-point projection; the free-energy projection agrees within the errors, as expected for
βħω_y ≥ 5.) The six separable controls are within 1.5σ of zero. The curvature points fit

    Delta = (0.200 ± 0.008) h²  =  (0.144 ± 0.006) (κρ)²,   χ²/dof = 0.69 (18 points),

with no detectable dependence on temperature or barrier symmetry, while the environment
response itself grows steeply as the temperature falls (h = 0.9 symmetric: G_QM = 0.37, 1.09,
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

FREQUENCY_SECTION

BEAD_SECTION

## Truncation of the earlier exact rates

TRUNCATION_SECTION

## Limits

- Two model families, one mass, one transverse frequency for the curvature family.
- The fitted h² law is empirical; the theory predicts the absence of a linear term, not the
  coefficient.
- RPMD statistical errors are 0.6–4%; no point is marked physically converged.
