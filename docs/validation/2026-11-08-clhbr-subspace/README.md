# Cl + HBr → HCl + Br: automatic ring-polymer path and subspace correction

Date: 2026-11-08. Surface: BrHCl_LEPS1 (historical LEPS, `benchmarks/brhcl`, atom order
Br, H, Cl; build with `python benchmarks/brhcl/build_pes.py`). Temperature 300 K, 16 beads.

## What was run

1. **Full RPMD rate with path accumulation.**
   `benchmarks/clhbr_subspace/rate_production.inp`. This is a standard gas-phase rate run
   with 61 umbrella windows on the RPMDrate ξ from −0.05 to 1.45 (past the dividing surface,
   because the H-transfer corner follows the early TS) and `output.centroid_path = true`.
2. **One command:**
   `pyrpmd subspace-correct benchmarks/clhbr_subspace/correct.toml`. It runs
   run directory → mean path → 1D model → C_S(1D) → curvature × width → 2D plane →
   C_S(2D) → nested sensitivity → `rate_quantum_corrected.json`.
3. **Independent checks of each stage** on the same 2D model (below).

## Surface and transition state

| Quantity | Value |
| --- | --- |
| Collinear saddle | r(Br–H) = 2.720, r(H–Cl) = 3.642 bohr |
| Barrier above Cl + HBr | 1.22 kcal/mol (classical) |
| Reaction energy | −16.2 kcal/mol |
| Saddle frequencies | 93i (reaction), 269 × 2 (bend), 2153 cm⁻¹ |
| HBr harmonic frequency | 0.01233 Eh (2707 cm⁻¹) |

## Full RPMD rate

**Historical input note:** The results below used the original
`rate_production.inp` with `dividing_surface = 1.0`. The current example omits
that override so the PMF selects `xi*`. These historical numbers have not been
recomputed with the corrected input.

`rate_production.inp` (4 ps per window, 64 parents × 16 child pairs × 1 ps, native kernels,
2 min 9 s on 4 cores): k_RPMD = 5.75 × 10⁻¹³ ± 3.32 × 10⁻¹³ cm³ molecule⁻¹ s⁻¹,
κ = 0.201 ± 0.020, free-energy barrier 3.16 kcal/mol. The run is labelled
`pilot_requires_convergence` because the endpoint energy error (0.15 kT) exceeds the
0.1 kT gate. With and without `centroid_path` the rate is bit-identical (the accumulator
does not touch the dynamics).

## Ring-polymer mean path

- All 61 windows are complete (400 samples each). The aligned-centroid spread falls from
  0.75–1.05 bohr at the reactant asymptote to 0.26–0.36 bohr near the TS. 23 windows
  (ξ = 0.80–1.52) are orientation-locked.
- The mean path (mean bond lengths along the axis, local quadratic fit in ξ) runs
  r(Br–H)/r(H–Cl) = 2.71/4.33 → 2.96/2.97 → 3.48/2.54 bohr. It passes inside the classical
  corner (corner cutting) and, in the exit channel, keeps the HCl bond ~0.14 bohr longer
  than the classical valley floor.
- Curvature × width from Σ(ξ) peaks at 11.9 at ξ = 1.32, the H-transfer corner. With
  harmonic widths along the whole assembled path it peaks at 1.3. Both are far above 0.5,
  so the pipeline adds a 2D subspace. The curvature normal at the peak lies entirely in the
  collinear plane (overlap 1.00).

## 1D model

Built from the first assembled path, the normal-cut vibrationally adiabatic potential had a
cusp at the corner (0.068 Eh): the window means were averaged as Cartesian positions
(librating H shortens the axial bonds), ordered by noisy ξ and joined to the continuation
with sharp kinks, and normal planes crossed inside the bend. With the path built from
mean bond lengths and fitted in ξ, the 1D barrier is smooth (2.5 mEh above the reactant
threshold) and C_S(1D) = 0.99 ± 0.09: along this path the effective mass is dominated by
the Cl approach and tunnelling is small. Because curvature × width is far above 0.5, the
1D model is not a trustworthy carrier of the RPMD defect here, whatever its value; the 2D
result is the one applied.

## 2D model (collinear plane)

The plane is spanned by the rate dividing-surface normal (mass-weighted ∇ξ, so its
dividing line is the rate's ξ = 1 surface) and the principal direction of the path. For
this collinear reaction that is the collinear plane: the path lies in it to 4 × 10⁻¹³.
V_S = collinear energy + bend zero-point energy (adiabatic). The default box has its
reference slice where the reactant channel is within 0.3 kT of its asymptote; products are
absorbed once the path is 15 kT below its maximum. The grid is anisotropic, 2.1 × 5.1
mass-weighted units for the DVR. RPMD uses the cubic spline of the same V_S projected 3×
finer.

### Exact quantum rate (dense route, energy-truncated DVR, flux restricted to E ≥ reactant threshold − kT)

| DVR spacing (transfer × path axis) | Retained points | k_QM (model units) | Wall time |
| --- | ---: | ---: | ---: |
| 2.8 × 6.8 | 3993 | 4.573 × 10⁻⁴ | 5 min |
| 2.1 × 5.1 | 7099 | 4.686 × 10⁻⁴ | 28 min |

The 2.5% change is the grid error at the finer spacing. On the earlier isotropic grids
(spacing 8.0 and 6.5, a different box) the rate was 1.6 × 10⁻⁷ and 2.2 × 10⁻⁶: an
isotropic grid at the dense limit does not resolve the H-stretch valley or the corner. On
the 6.5 grid the following invariances held (they test the flux machinery, not the grid):

- dividing line at a = −20, 0, +15: 2.1563, 2.1565, 2.1572 × 10⁻⁶;
- absorber ×0.5, ×1, ×2: 2.1573, 2.1565, 2.1568 × 10⁻⁶;
- energy-floor margin 1 kT vs 3 kT: 2.1565 vs 2.1563 × 10⁻⁶.

Without the energy floor the 116 (1D) and ~200 (2D) Br + HCl states below the Cl + HBr
threshold, weighted by e²⁵, made the rate depend on the absorber by orders of magnitude.

### RPMD rate of the same V_S

| Check | Value |
| --- | --- |
| One-bead RPMD QTST vs classical TST by quadrature | 1.44 × 10⁻³ vs 1.18 × 10⁻³ (17% statistical error) |
| P = 16, MD step 2.03 au (seed 11) | k = 4.58 × 10⁻⁴ ± 28%, κ = 0.092 ± 0.024, QTST = 4.96 × 10⁻³ |
| P = 16, MD step 1.02 au (seed 12) | k = 6.38 × 10⁻⁴ ± 22%, κ = 0.128 ± 0.024, QTST = 4.99 × 10⁻³ |
| Classical TST (quadrature, new box) | 1.60 × 10⁻³ |

The maximum energy error of the recrossing trajectories (0.6 in units of 1/β_P) does not
change with the time step. It comes from products leaving the RPMD grid, where the
potential is continued as a constant (force kink); those trajectories are already
products and cannot recross.

### Correction factor

C_S(2D) = 4.69 × 10⁻⁴ / (5.4 ± 0.9) × 10⁻⁴ ≈ 0.9 ± 0.2 (RPMD runs combined). In the
collinear H-transfer plane at 300 K, RPMD reproduces the exact rate of the same Hamiltonian
within its statistical error, so the correction leaves the full RPMD rate unchanged within
errors. The corner-cutting tunnelling that a 1D model misses is captured by the ring
polymer itself once the transverse coordinate is included.

## One-command pipeline result

`pyrpmd subspace-correct benchmarks/clhbr_subspace/correct.toml` (about 45 min on 4 cores, 39 of
them for the 2D stage, mostly the 7099-point dense solve), output in `runs/clhbr-300K-production/subspace/`:

| | 1D path model | 2D collinear plane |
| --- | ---: | ---: |
| exact quantum rate (model units) | 8.83 × 10⁻⁴ | 4.686 × 10⁻⁴ |
| RPMD rate, same V_S (P = 16) | 8.94 × 10⁻⁴ (κ = 1.000) | 4.575 × 10⁻⁴ (κ = 0.092 ± 0.024) |
| C_S | 0.99 ± 0.09 | 1.02 ± 0.29 |
| DVR points / energy-floor states | 2137 / 116 | 7099 / 50 |

Nested sensitivity: ln(C_2D/C_1D) = 0.036 ± 0.30, i.e. undetermined, so the pipeline applies
the dimension chosen by curvature × width (2D). The first run of the pipeline counted this
step as "settled" (a large uncertainty passed the old `max(tolerance, 2σ)` test) and applied
1D; the selection rule was corrected and the corrections re-applied from the saved C_S records
(`output.reuse = true`).

| Rate (cm³ molecule⁻¹ s⁻¹) | Value |
| --- | ---: |
| RPMD (full system) | 5.75 × 10⁻¹³ ± 3.32 × 10⁻¹³ |
| RPMD × C_S(2D) | 5.89 × 10⁻¹³ ± 3.79 × 10⁻¹³ |

The corrected rate is not marked publishable because the source RPMD run is not (energy
gate); `rate_quantum_corrected.json` keeps `corrected_rate_cm3_s` empty and reports the
corrected estimate.

## Limits

No result is marked physically converged. The 2D quantum rate is converged to ~3% in
the grid; RPMD C_S(2D) carries 20–30% statistical error per run. Bead-number convergence
of C_S was not tested. Non-collinear (bend) quantum effects enter only through the
adiabatic bend zero-point energy; a 3D subspace (collinear plane + bend) was not computed.
The curvilinear (transfer-chart) quantum route was attempted and is not shipped: on this
system the chart needs ~0.025 bohr spacing along the separation to resolve heavy-fragment
wavelengths, so it is no cheaper than the flat collinear plane, and it could not be
validated within this work.
