# Collinear H + H₂ below the crossover temperature: RPMD vs exact quantum rate

Date: 2026-11-08. Surface: BKMP2 (`benchmarks/h3`, build with `python benchmarks/h3/build_pes.py`).
Scripts: `benchmarks/h3_subspace/` (`collinear.py` model, `run.py` driver, `campaign.txt` the runs
below). Result records: `results/`.

> **Correction (2026-11-09).** The exact rates below used DVR walls 0.04 Eh above the H₂
> ground level. That is too low: the removed points confine the H₂ vibration (about six quanta,
> 0.12 Eh, are needed). Walls at 0.16 Eh lower the exact rates by 3.1, 7.8, 12.9, 13.4 and 5.0% at
> 600, 400, 300, 200 and 150 K. With those points retained, spacing 2.0 is 3–6% low at 200–150 K;
> a spacing ladder is in
> [docs/validation/2026-11-09-subspace-transfer](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-09-subspace-transfer/README.md).
> The RPMD rates are unaffected; the ratios k_RPMD/k_QM below are therefore low by up to ~15%.

## Why this test

The Cl + HBr test (docs/validation/2026-11-08-clhbr-subspace) ran at 300 K, above that
system's crossover temperature, where RPMD is expected to be accurate. Here the model is
the whole collinear H + H₂ system: no projection and no bath. The ratio
C = k_QM / k_RPMD is therefore the RPMD method error itself, at temperatures down to 0.43 T_c.

## Model

| Quantity | Value |
| --- | --- |
| Coordinates | s = (r₂ − r₁)√(m/6), t = (r₁ + r₂)√(m/2): orthonormal, mass weighted, centre of mass removed |
| Saddle | r₁ = r₂ = 1.757 bohr, V‡ = 15.35 mEh (9.63 kcal/mol) |
| Saddle frequencies | 1511i cm⁻¹ (s), 2055 cm⁻¹ (symmetric stretch) |
| Collinear H₂ ground level | 9.959 mEh above the classical asymptote |
| Crossover temperature | T_c = ħω‡/(2πk_B) = 346 K |
| Box | s ∈ [−110, 110], t ∈ [70, 290]; quadratic absorber for \|s\| > 75; reactant slice at s = −63 (r₁ − r₂ = 3.6 bohr, V within 2 × 10⁻⁵ Eh of the asymptote) |

Both rates use the same V(s, t): the DVR at its grid nodes, RPMD the BKMP2 call itself with
its analytic gradient. Rates are in model units (s length per time); only ratios matter.
`exact_quantum_rate(..., method="sylvester", normalization="slice")` with
`energy_floor` = H₂ ground level − kT and `truncation_energy` = H₂ ground level + max(0.04, V‡ + 25 kT);
`ring_polymer_rate` with 48 TI windows, 32 walkers × 400 sweeps, 512 trajectory pairs.

## Exact quantum rate: convergence

| T (K) | spacing 3.0 | 2.5 | 2.0 | absorber ×0.5 / ×2 (spacing 2.5) |
| ---: | ---: | ---: | ---: | --- |
| 300 | 5.984e−7 | 5.958e−7 | 5.846e−7 | 5.957e−7 / 5.960e−7 |
| 200 | | | 8.765e−9 | 8.962e−9 / 8.912e−9 |
| 150 | 3.69e−10 | 2.938e−10 | 2.942e−10 | 3.097e−10 / 2.677e−10 |
| 100 | | | −8.1e−13 | +3.4e−12 / −1.1e−11 |

The quantum rate is converged to ~2% at 300 K (grid), ≤ 0.6% in the absorber at 200–300 K
and ~7% (absorber) at 150 K; 400 and 600 K use the same grid and absorber without separate
checks. **At 100 K it is not converged**: the sign changes with the absorber. The thermal flux
there is a small difference between near-threshold box states of both channels, and absorber
reflection of the slow (λ ≈ 100 mass-weighted units) near-threshold waves dominates. A
cumulative-reaction-probability route, N(E) = 4 Tr[ε_r G† ε_p G], was tried. It agrees with the
flux route at 300 K to 10⁻⁵, but its N(E) near threshold carries an absorber-dependent background
of 10⁻⁸–10⁻⁷, which dominates the Boltzmann integral at ≤ 150 K. A transmission-free absorber
(Manolopoulos 2002) did not remove it. 100 K needs a scattering method with exact asymptotic
boundary conditions, for example a log-derivative or R-matrix propagation in the H₂ channel
basis. That route was not shipped.

## RPMD: bead convergence

| T (K) | T/T_c | βħω(H₂) | P | k_RPMD | rel. error | QTST | κ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 600 | 1.73 | 10 | 16 | 4.41e−5 | 4% | 5.78e−5 | 0.764 ± 0.022 |
| 600 | | | 32 | 4.35e−5 | 5% | 5.93e−5 | 0.734 ± 0.023 |
| 400 | 1.16 | 15 | 32 | 2.97e−6 | 5% | 4.46e−6 | 0.666 ± 0.027 |
| 400 | | | 64 | 3.78e−6 | 6% | 5.01e−6 | 0.754 ± 0.022 |
| 300 | 0.87 | 20 | 32 | 2.71e−7 | 6% | 4.08e−7 | 0.663 ± 0.027 |
| 300 | | | 64 | 3.56e−7 | 6% | 4.77e−7 | 0.746 ± 0.024 |
| 200 | 0.58 | 30 | 32 | 2.92e−9 | 6% | 4.52e−9 | 0.645 ± 0.028 |
| 200 | | | 64 | 4.80e−9 | 6% | 6.92e−9 | 0.694 ± 0.026 |
| 200 | | | 128 | 4.99e−9 | 9% | 7.20e−9 | 0.692 ± 0.028 |
| 150 | 0.43 | 40 | 64 | 1.30e−10 | 8% | 2.13e−10 | 0.612 ± 0.029 |
| 150 | | | 128 | 1.52e−10 | 9% | 2.48e−10 | 0.612 ± 0.032 |
| 100 | 0.29 | 60 | 64 | 2.14e−12 | 10% | 3.92e−12 | 0.546 ± 0.031 |

P ≈ βħω is not enough. For harmonic environment modes the finite-bead factor
B_P = k_RPMD(P)/k_RPMD(∞) has a closed form (`pyrpmd.analysis.harmonic_bead_factor`,
derivation in [subspace transfer theory](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/subspace-transfer-theory.md)).
With the BKMP2 harmonic frequencies, 4404 cm⁻¹ (H₂) and 2055 cm⁻¹ (TS symmetric stretch),
B_P is 0.59 / 0.87 / 0.97 at 200 K for P = 32 / 64 / 128, 0.72 at 150 K for P = 64, and 0.35 at
100 K for P = 64. Dividing by B_P collapses the bead series:

| T (K) | P | B_P | QTST raw → corrected | k_RPMD raw → corrected |
| ---: | ---: | ---: | --- | --- |
| 600 | 16 / 32 | 0.922 / 0.979 | 5.78 / 5.93 → 6.27 / 6.05 e−5 | 4.41 / 4.35 → 4.79 / 4.45 e−5 |
| 400 | 32 / 64 | 0.932 / 0.982 | 4.46 / 5.01 → 4.78 / 5.10 e−6 | 2.97 / 3.78 → 3.19 / 3.85 e−6 |
| 300 | 32 / 64 | 0.849 / 0.958 | 4.08 / 4.77 → 4.81 / 4.98 e−7 | 2.71 / 3.56 → 3.19 / 3.71 e−7 |
| 200 | 32 / 64 / 128 | 0.594 / 0.868 / 0.965 | 4.52 / 6.92 / 7.20 → 7.60 / 7.97 / 7.47 e−9 | 2.92 / 4.80 / 4.99 → 4.90 / 5.53 / 5.17 e−9 |
| 150 | 64 / 128 | 0.721 / 0.919 | 2.13 / 2.48 → 2.96 / 2.71 e−10 | 1.30 / 1.52 → 1.81 / 1.66 e−10 |

The corrected QTST values agree within their 4–5% errors. The corrected rates at 300 and
400 K still differ by 14–20%: P = 32 has a lower κ (0.66 vs 0.75), which the static factor
cannot remove. The required bead number grows as (βħω)^{3/2}:
P ≳ [(Σ_r u³ − Σ_‡ u³)/(48δ)]^{1/2} for |ln B_P| < δ, so P ≈ 110 at 200 K and P ≈ 300 at 100 K
for 5%. An earlier version of this report stated P ≥ 2βħω; that is not sufficient (B_P = 0.87 at
200 K, P = 64).

## Result

| T (K) | T/T_c | P | k_QM | k_RPMD | k_RPMD / k_QM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 600 | 1.73 | 32 | 6.87e−5 | 4.35e−5 ± 5% | 0.63 ± 0.03 |
| 400 | 1.16 | 64 | 5.83e−6 | 3.78e−6 ± 6% | 0.65 ± 0.04 |
| 300 | 0.87 | 64 | 5.85e−7 (±2%) | 3.56e−7 ± 6% | 0.61 ± 0.04 |
| 200 | 0.58 | 128 | 8.77e−9 (±2%) | 4.99e−9 ± 9% | 0.57 ± 0.05 |
| 150 | 0.43 | 128 | 2.94e−10 (±7%) | 1.52e−10 ± 9% | 0.52 ± 0.06 |
| 100 | 0.29 | — | not converged | (2.1e−12 at P = 64, bead-unconverged) | — |

Divided by B_P of the bead number used, the ratios are 0.65, 0.66, 0.63, 0.59 and 0.56 at 600,
400, 300, 200 and 150 K.

RPMD underestimates the exact collinear rate at every temperature. The ratio is flat
(0.61–0.65) from 600 K to 300 K, above and just below T_c, and then falls to 0.57 at
0.58 T_c and 0.52 at 0.43 T_c. The deficit is already present above T_c. In collinear
H + H₂ it comes from RPMD's dynamics (κ ≈ 0.6–0.75, against strong quantum recrossing and
resonance effects that the exact rate has) more than from the deep-tunnelling regime. Below
T_c the deficit grows slowly: at 0.43 T_c RPMD is within a factor of 2 of the exact rate.

For the subspace correction this means C = k_QM/k_RPMD ≈ 1.6 at 300–600 K and 1.9 at
150 K for collinear H + H₂. Unlike Cl + HBr at 300 K (C = 1.0 ± 0.3), the correction is
significant here.

Checks that support the comparison:

- One-bead RPMD at 600 K gives QTST 5.80 × 10⁻⁶ against classical TST by quadrature,
  5.84 × 10⁻⁶. The RPMD normalization matches the slice normalization of the quantum rate.
- The quantum flux route is invariant to the absorber at ≥ 200 K (≤ 0.6%) and grid converged to ~2%.
- The maximum energy error of the RPMD trajectories is ≤ 0.04 / β_P; HMC acceptance is 0.71–0.77.

## Limits

- Collinear only: there is no bend, so no comparison with 3D H + H₂ rates. Collinear systems
  recross more strongly than 3D ones.
- 100 K is not covered by a converged exact rate or by a bead-converged RPMD rate.
- The κ values at P = 32 lie systematically ~3σ below those at P = 64 (300 and 400 K). Part of
  the P = 32 deficit is in κ, not in QTST; this was not investigated further.
- No result is marked physically converged.
