# Instanton anchor of the subspace correction: tests on valley models (2026-11-14)

This report tests the statements of
[the instanton-anchor theory page](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/subspace-instanton-anchor.md)
against exact quantum rates and RPMD rates of two-dimensional model potentials:

1. the RPMD defect `eps = k_QM / k_RPMD` below the crossover temperature is, to leading
   order, the steepest-descent factor `eps_SD = 2 pi / (beta hbar eta_1)` of the instanton;
2. the transfer error of a subspace correction `C_S` is predicted by the difference of the
   steepest-descent factors of the full system and of the *skeleton* of the reduction
   (the "skeleton anchor"), not of the reduction itself (the "naive anchor");
3. a path-free estimate from the instanton and a symmetric Eckart barrier reproduces `eps`.

The tables are generated from `benchmarks/subspace_instanton/results/*.json` by
`benchmarks/subspace_instanton/report.py`; `tests/test_subspace_instanton.py` checks that
this file is the generated one.

## Models and reductions

`V(s, Q) = V0 sech^2(s/a) + omega(s)^2 (Q - g(s))^2 / 2`, `hbar = m = 1`, `V0 = 2.1`,
bare barrier frequency 1 (`benchmarks/subspace_instanton/models.py`).

| model | transverse frequency | valley floor | saddle: unstable / stable frequency | what it isolates |
|---|---|---|---|---|
| M1 | 1.8 → 1.1 at the barrier | straight | 1.000 / 1.100 | zero-point change only |
| M2 | 1.5 | `g = -sech^2(s/a)` | 1.000 / 1.500 | curved path (corner cutting) |
| M3 | 1.8 → 1.1 | `g = -sech^2(s/a)` | 1.000 / 1.100 | both |
| M4 | 0.5 | `g = 2 sqrt(2) tanh(s/2)` | 0.800 / 0.625 | slow bath displaced between reactants and products |

One-dimensional reductions along the minimum-energy path (`reductions.py`):

| reduction | coordinate | potential | skeleton |
|---|---|---|---|
| `mep` | arc length | `V` on the path | itself |
| `fe` | arc length | plus the harmonic quantum free energy of the transverse mode | `mep` |
| `fe+sct` | small-curvature coordinate `d tau = sqrt(mu_eff) d sigma` | as `fe` | `mep+sct` |
| `bare` (M4 only) | `s`, bare mass | `V0 sech^2(s/a)` | itself |

## Method

- **Exact quantum rates** (`greens.py`): cumulative reaction probability from a recursive
  Green's function with exact lead self-energies, no absorbing potential. The 2D rates
  change by less than 1e-3 between two oscillator basis sizes and by less than 3e-3 when
  the grid spacing is halved (`basis_convergence`, `grid_convergence` in `qm.json`).
- **Instantons** (`instanton.py`): Newton search on the ring-polymer potential with 256
  beads; `eta_1` changes by less than 1e-3 from 128 to 256 beads.
- **RPMD rates** (`run_rpmd.py`): `pyrpmd.analysis.subspace_rpmd.ring_polymer_rate`, the
  solver of the subspace pipeline. The full models use their analytic potential, the
  reductions a tabulated one. 48 thermodynamic-integration windows from `s = -14` to the
  barrier and 2048 recrossing trajectories per run. Full models: two independent runs of
  32 walkers and 600 sweeps, averaged in the logarithm with inverse-variance weights.
  Reductions and Eckart barriers: one run of 32 walkers and 1600 sweeps. The full RPMD rate
  is divided by the harmonic finite-bead factor `B_P` of the transverse mode
  (`harmonic_bead_factor`).
- **Beads**: 32 at `1.25 T_c`, 64 at `0.8 T_c`, 96 (M2, M4) or 128 (M1, M3) at `0.55 T_c`;
  the reductions use the power of two at or above the number of the full model. The bead
  error of the instanton action, the leading factor of the rate, is below 0.003 at these
  numbers (`tests/test_subspace_instanton.py`), and the Eckart defect at `u = 11.4` agrees
  within errors between 64 and 128 beads (table below).
- **Errors** are one standard error of the RPMD rates; the quantum rates and the instantons
  are deterministic. `chi²` is the mean of `(value / error)^2`: about one when a quantity
  vanishes within the statistical error.

Temperatures are in units of the crossover temperature `T_c = hbar lambda / 2 pi k_B` of the
full saddle.

## RPMD defect of the full models

| model | T/T_c | P | runs | kappa | B_P | eps(H) | ln eps(H) | A_H | R_H = ln eps − A_H |
|---|---|---|---|---|---|---|---|---|---|
| M1 (zero-point change) | 1.25 | 32 | 2 | 0.997 | 0.9886 | 1.127 ± 0.003 | +0.119 ± 0.003 | 0.000 | +0.119 ± 0.003 |
| M1 (zero-point change) | 0.8 | 64 | 2 | 0.990 | 0.9890 | 1.265 ± 0.006 | +0.235 ± 0.005 | 0.211 | +0.024 ± 0.005 |
| M1 (zero-point change) | 0.55 | 128 | 2 | 0.964 | 0.9915 | 1.613 ± 0.021 | +0.478 ± 0.013 | 0.518 | -0.040 ± 0.013 |
| M2 (curved path) | 1.25 | 32 | 2 | 0.988 | 1.0000 | 1.132 ± 0.005 | +0.124 ± 0.004 | 0.000 | +0.124 ± 0.004 |
| M2 (curved path) | 0.8 | 64 | 2 | 0.972 | 1.0000 | 1.266 ± 0.009 | +0.236 ± 0.007 | 0.212 | +0.024 ± 0.007 |
| M2 (curved path) | 0.55 | 96 | 2 | 0.929 | 1.0000 | 1.692 ± 0.025 | +0.526 ± 0.015 | 0.492 | +0.034 ± 0.015 |
| M3 (both) | 1.25 | 32 | 2 | 0.984 | 0.9886 | 1.133 ± 0.005 | +0.125 ± 0.004 | 0.000 | +0.125 ± 0.004 |
| M3 (both) | 0.8 | 64 | 2 | 0.966 | 0.9890 | 1.238 ± 0.008 | +0.214 ± 0.006 | 0.219 | -0.005 ± 0.006 |
| M3 (both) | 0.55 | 128 | 2 | 0.939 | 0.9915 | 1.586 ± 0.023 | +0.461 ± 0.015 | 0.496 | -0.035 ± 0.015 |
| M4 (slow bath) | 1.25 | 32 | 2 | 0.803 | 1.0006 | 1.117 ± 0.011 | +0.111 ± 0.010 | 0.000 | +0.111 ± 0.010 |
| M4 (slow bath) | 0.8 | 64 | 2 | 0.798 | 1.0006 | 1.270 ± 0.015 | +0.239 ± 0.012 | 0.200 | +0.039 ± 0.012 |
| M4 (slow bath) | 0.55 | 96 | 2 | 0.792 | 1.0008 | 1.648 ± 0.025 | +0.499 ± 0.015 | 0.481 | +0.018 ± 0.015 |

## Transfer errors below the crossover temperature

`Δ` is the error of the corrected rate `k_RPMD(H) C_S` in the logarithm; `G` is the error of
using the quantum rate of the reduction on its own. An anchored correction replaces `C_S` by
`C_S exp(anchor)`, so its error is `Δ − anchor`. The naive anchor is not given where the
instanton of the reduction has a second unstable mode.

| model | T/T_c | reduction | G (QM alone) | ln C_S | Δ = ln eps(H) − ln C_S | skeleton anchor | Δ − anchor | naive anchor | Δ − anchor |
|---|---|---|---|---|---|---|---|---|---|
| M1 | 1.25 | `mep` | +1.552 | +0.129 ± 0.004 | -0.010 ± 0.005 | +0.000 | -0.010 ± 0.005 | +0.000 | -0.010 ± 0.005 |
| M1 | 1.25 | `fe` | -0.006 | +0.114 ± 0.004 | +0.006 ± 0.005 | +0.000 | +0.006 ± 0.005 | +0.000 | +0.006 ± 0.005 |
| M1 | 0.8 | `mep` | +2.086 | +0.268 ± 0.008 | -0.033 ± 0.010 | -0.000 | -0.033 ± 0.010 | -0.000 | -0.033 ± 0.010 |
| M1 | 0.8 | `fe` | -0.013 | +0.236 ± 0.006 | -0.001 ± 0.008 | -0.000 | -0.001 ± 0.008 | +0.083 | -0.085 ± 0.008 |
| M1 | 0.55 | `mep` | +2.282 | +0.544 ± 0.012 | -0.066 ± 0.018 | +0.000 | -0.066 ± 0.018 | +0.000 | -0.066 ± 0.018 |
| M1 | 0.55 | `fe` | -0.019 | +0.474 ± 0.011 | +0.004 ± 0.017 | +0.000 | +0.004 ± 0.017 | +0.070 | -0.066 ± 0.017 |
| M2 | 1.25 | `mep` | -0.025 | +0.121 ± 0.004 | +0.003 ± 0.006 | +0.000 | +0.003 ± 0.006 | +0.000 | +0.003 ± 0.006 |
| M2 | 1.25 | `fe` | +0.094 | +0.100 ± 0.003 | +0.025 ± 0.005 | +0.000 | +0.025 ± 0.005 | +0.000 | +0.025 ± 0.005 |
| M2 | 1.25 | `fe+sct` | -0.390 | +0.177 ± 0.004 | -0.052 ± 0.006 | -0.004 | -0.048 ± 0.006 | +0.000 | -0.052 ± 0.006 |
| M2 | 0.8 | `mep` | -0.063 | +0.256 ± 0.008 | -0.020 ± 0.011 | +0.021 | -0.041 ± 0.011 | +0.021 | -0.041 ± 0.011 |
| M2 | 0.8 | `fe` | +0.184 | +0.216 ± 0.007 | +0.020 ± 0.010 | +0.021 | -0.001 ± 0.010 | +0.077 | -0.056 ± 0.010 |
| M2 | 0.8 | `fe+sct` | -0.870 | +0.324 ± 0.008 | -0.088 ± 0.011 | -0.150 | +0.063 ± 0.011 | -0.109 | +0.022 ± 0.011 |
| M2 | 0.55 | `mep` | -0.048 | +0.506 ± 0.013 | +0.020 ± 0.019 | +0.024 | -0.003 ± 0.019 | +0.024 | -0.003 ± 0.019 |
| M2 | 0.55 | `fe` | +0.301 | +0.493 ± 0.011 | +0.033 ± 0.019 | +0.024 | +0.009 ± 0.019 | +0.033 | -0.000 ± 0.019 |
| M2 | 0.55 | `fe+sct` | -1.475 | +0.590 ± 0.011 | -0.064 ± 0.018 | -0.016 | -0.048 ± 0.018 | -0.036 | -0.028 ± 0.018 |
| M3 | 1.25 | `mep` | +1.546 | +0.119 ± 0.004 | +0.006 ± 0.006 | +0.000 | +0.006 ± 0.006 | +0.000 | +0.006 ± 0.006 |
| M3 | 1.25 | `fe` | +0.092 | +0.089 ± 0.003 | +0.036 ± 0.005 | +0.000 | +0.036 ± 0.005 | +0.000 | +0.036 ± 0.005 |
| M3 | 1.25 | `fe+sct` | -0.311 | +0.155 ± 0.004 | -0.031 ± 0.006 | +0.000 | -0.031 ± 0.006 | +0.000 | -0.031 ± 0.006 |
| M3 | 0.8 | `mep` | +2.086 | +0.254 ± 0.007 | -0.040 ± 0.010 | +0.025 | -0.065 ± 0.010 | +0.025 | -0.065 ± 0.010 |
| M3 | 0.8 | `fe` | +0.195 | +0.180 ± 0.005 | +0.034 ± 0.008 | +0.025 | +0.009 ± 0.008 | +0.134 | -0.100 ± 0.008 |
| M3 | 0.8 | `fe+sct` | -0.731 | +0.298 ± 0.007 | -0.084 ± 0.009 | -0.197 | +0.113 ± 0.009 | -0.054 | -0.030 ± 0.009 |
| M3 | 0.55 | `mep` | +2.329 | +0.498 ± 0.011 | -0.037 ± 0.018 | +0.033 | -0.070 ± 0.018 | +0.033 | -0.070 ± 0.018 |
| M3 | 0.55 | `fe` | +0.347 | +0.407 ± 0.010 | +0.054 ± 0.017 | +0.033 | +0.021 ± 0.017 | +0.143 | -0.089 ± 0.017 |
| M3 | 0.55 | `fe+sct` | -1.351 | +0.536 ± 0.011 | -0.075 ± 0.018 | -0.053 | -0.022 ± 0.018 | +0.029 | -0.104 ± 0.018 |
| M4 | 1.25 | `mep` | -0.416 | +0.101 ± 0.005 | +0.010 ± 0.011 | +0.000 | +0.010 ± 0.011 | +0.000 | +0.010 ± 0.011 |
| M4 | 1.25 | `fe` | +0.013 | +0.097 ± 0.005 | +0.014 ± 0.011 | +0.000 | +0.014 ± 0.011 | +0.000 | +0.014 ± 0.011 |
| M4 | 1.25 | `fe+sct` | -0.146 | +0.104 ± 0.004 | +0.006 ± 0.010 | +0.000 | +0.006 ± 0.010 | +0.000 | +0.006 ± 0.010 |
| M4 | 1.25 | `bare` | -1.102 | +0.178 ± 0.005 | -0.067 ± 0.011 | +0.000 | -0.067 ± 0.011 | +0.000 | -0.067 ± 0.011 |
| M4 | 0.8 | `mep` | -0.585 | +0.247 ± 0.010 | -0.008 ± 0.015 | -0.000 | -0.008 ± 0.015 | -0.000 | -0.008 ± 0.015 |
| M4 | 0.8 | `fe` | +0.037 | +0.235 ± 0.009 | +0.004 ± 0.015 | -0.000 | +0.004 ± 0.015 | +0.008 | -0.004 ± 0.015 |
| M4 | 0.8 | `fe+sct` | -0.652 | +0.304 ± 0.008 | -0.065 ± 0.015 | -0.089 | +0.024 ± 0.015 | -0.079 | +0.014 ± 0.015 |
| M4 | 0.8 | `bare` | -2.476 | +0.424 ± 0.010 | -0.185 ± 0.016 | -0.199 | +0.014 ± 0.016 | -0.199 | +0.014 ± 0.016 |
| M4 | 0.55 | `mep` | -0.655 | +0.506 ± 0.015 | -0.007 ± 0.021 | +0.008 | -0.015 ± 0.021 | +0.008 | -0.015 ± 0.021 |
| M4 | 0.55 | `fe` | +0.059 | +0.507 ± 0.014 | -0.007 ± 0.021 | +0.008 | -0.015 ± 0.021 | +0.022 | -0.029 ± 0.021 |
| M4 | 0.55 | `fe+sct` | -1.489 | +0.624 ± 0.012 | -0.124 ± 0.019 | -0.136 | +0.012 ± 0.019 | -0.129 | +0.004 ± 0.019 |
| M4 | 0.55 | `bare` | -3.959 | +0.718 ± 0.014 | -0.219 ± 0.020 | -0.198 | -0.021 ± 0.020 | -0.198 | -0.021 ± 0.020 |

Root-mean-square errors over the temperatures below `T_c` (0.8 and 0.55):

| reduction | points | rms G | rms Δ: `C_S` | chi² | skeleton-anchored | chi² | naive-anchored | chi² |
|---|---|---|---|---|---|---|---|---|
| `mep` | 8 | 1.585 | 0.034 | 6.5 | 0.046 | 12.7 | 0.046 | 12.7 |
| `fe` | 8 | 0.190 | 0.027 | 4.2 | 0.010 | 0.4 | 0.065 | 43.0 |
| `fe+sct` | 6 | 1.150 | 0.086 | 39.5 | 0.058 | 31.2 | 0.047 | 8.6 |
| `bare` | 2 | 3.302 | 0.203 | 127.0 | 0.018 | 0.9 | 0.018 | 0.9 |

## Transfer errors above the crossover temperature

At `1.25 T_c` the full models have no instanton and the anchors are zero, except −0.004 for
the SCT skeleton of M2, which is just below its own crossover.

| model | reduction | G (QM alone) | Δ |
|---|---|---|---|
| M1 | `mep` | +1.552 | -0.010 ± 0.005 |
| M1 | `fe` | -0.006 | +0.006 ± 0.005 |
| M2 | `mep` | -0.025 | +0.003 ± 0.006 |
| M2 | `fe` | +0.094 | +0.025 ± 0.005 |
| M2 | `fe+sct` | -0.390 | -0.052 ± 0.006 |
| M3 | `mep` | +1.546 | +0.006 ± 0.006 |
| M3 | `fe` | +0.092 | +0.036 ± 0.005 |
| M3 | `fe+sct` | -0.311 | -0.031 ± 0.006 |
| M4 | `mep` | -0.416 | +0.010 ± 0.011 |
| M4 | `fe` | +0.013 | +0.014 ± 0.011 |
| M4 | `fe+sct` | -0.146 | +0.006 ± 0.010 |
| M4 | `bare` | -1.102 | -0.067 ± 0.011 |

## Symmetric Eckart barrier

`run_eckart.py`: units `hbar = m = omega_b = 1`; `A(u)` is the steepest-descent factor and
`R` the remainder. Rows with two bead numbers are bead-convergence checks.

| u = βħω_b | v = V_0/ħω_b | P | kappa | eps | ln eps | A(u) | R = ln eps − A |
|---|---|---|---|---|---|---|---|
| 9.000 | 1.200 | 64 | 0.986 | 1.514 ± 0.011 | +0.415 | 0.329 | +0.086 ± 0.007 |
| 5.027 | 1.750 | 32 | 0.999 | 1.157 ± 0.004 | +0.146 | 0.000 | +0.146 ± 0.003 |
| 7.854 | 1.750 | 64 | 0.988 | 1.357 ± 0.009 | +0.305 | 0.211 | +0.094 ± 0.007 |
| 11.424 | 1.750 | 128 | 0.965 | 1.740 ± 0.018 | +0.554 | 0.518 | +0.036 ± 0.011 |
| 5.027 | 2.100 | 32 | 0.998 | 1.134 ± 0.005 | +0.125 | 0.000 | +0.125 ± 0.004 |
| 6.000 | 2.100 | 32 | 0.997 | 1.192 ± 0.006 | +0.175 | 0.000 | +0.175 ± 0.005 |
| 7.854 | 2.100 | 64 | 0.992 | 1.333 ± 0.009 | +0.287 | 0.211 | +0.076 ± 0.007 |
| 10.500 | 2.100 | 128 | 0.970 | 1.618 ± 0.017 | +0.481 | 0.453 | +0.028 ± 0.010 |
| 11.424 | 2.100 | 64 | 0.956 | 1.717 ± 0.021 | +0.541 | 0.518 | +0.023 ± 0.012 |
| 11.424 | 2.100 | 128 | 0.955 | 1.722 ± 0.021 | +0.543 | 0.518 | +0.026 ± 0.012 |
| 12.000 | 2.100 | 128 | 0.949 | 1.835 ± 0.022 | +0.607 | 0.555 | +0.052 ± 0.012 |
| 13.500 | 2.100 | 128 | 0.930 | 1.931 ± 0.027 | +0.658 | 0.640 | +0.018 ± 0.014 |
| 5.027 | 2.702 | 32 | 0.999 | 1.113 ± 0.005 | +0.107 | 0.000 | +0.107 ± 0.004 |
| 7.854 | 2.702 | 64 | 0.992 | 1.285 ± 0.011 | +0.251 | 0.211 | +0.040 ± 0.008 |
| 11.424 | 2.702 | 128 | 0.956 | 1.719 ± 0.023 | +0.542 | 0.518 | +0.024 ± 0.013 |

## Path-free estimate

`ln C_Eck = A_H + R_Eck(u, v)` with `u = beta hbar lambda` and
`v = V_ad / hbar lambda`, where `V_ad` is the barrier height plus the change of the
transverse zero-point energy. It uses the saddle, its frequencies, the full-dimensional
instanton and one 1D calculation, and no reaction path.

| model | T/T_c | u | v | ln eps(H) | A_H | R_Eck(u, v) | ln C_Eck = A_H + R_Eck | ln eps(H) − ln C_Eck |
|---|---|---|---|---|---|---|---|---|
| M1 | 1.25 | 5.027 | 1.750 | +0.119 ± 0.003 | 0.000 | +0.146 ± 0.003 | +0.146 | -0.027 ± 0.004 |
| M1 | 0.8 | 7.854 | 1.750 | +0.235 ± 0.005 | 0.211 | +0.094 ± 0.007 | +0.305 | -0.070 ± 0.008 |
| M1 | 0.55 | 11.424 | 1.750 | +0.478 ± 0.013 | 0.518 | +0.036 ± 0.011 | +0.554 | -0.076 ± 0.017 |
| M2 | 1.25 | 5.027 | 2.100 | +0.124 ± 0.004 | 0.000 | +0.125 ± 0.004 | +0.125 | -0.001 ± 0.006 |
| M2 | 0.8 | 7.854 | 2.100 | +0.236 ± 0.007 | 0.212 | +0.076 ± 0.007 | +0.288 | -0.052 ± 0.010 |
| M2 | 0.55 | 11.424 | 2.100 | +0.526 ± 0.015 | 0.492 | +0.026 ± 0.012 | +0.518 | +0.008 ± 0.019 |
| M3 | 1.25 | 5.027 | 1.750 | +0.125 ± 0.004 | 0.000 | +0.146 ± 0.003 | +0.146 | -0.021 ± 0.005 |
| M3 | 0.8 | 7.854 | 1.750 | +0.214 ± 0.006 | 0.219 | +0.094 ± 0.007 | +0.314 | -0.100 ± 0.009 |
| M3 | 0.55 | 11.424 | 1.750 | +0.461 ± 0.015 | 0.496 | +0.036 ± 0.011 | +0.532 | -0.071 ± 0.018 |
| M4 | 1.25 | 5.027 | 2.702 | +0.111 ± 0.010 | 0.000 | +0.107 ± 0.004 | +0.107 | +0.003 ± 0.011 |
| M4 | 0.8 | 7.854 | 2.702 | +0.239 ± 0.012 | 0.200 | +0.040 ± 0.008 | +0.240 | -0.001 ± 0.015 |
| M4 | 0.55 | 11.424 | 2.702 | +0.499 ± 0.015 | 0.481 | +0.024 ± 0.013 | +0.505 | -0.006 ± 0.020 |

Difference `ln eps(H) − ln C_Eck`: rms 0.050, largest 0.100, chi² per point 25.3 (12 points).

## Findings

1. **The steepest-descent factor is the leading part of the RPMD defect below `T_c`.** On the
   four valley models `ln eps` is 0.21 to 0.53 at 0.8 and 0.55 `T_c`, and the remainder
   `R_H = ln eps − A_H` lies between −0.04 and +0.04. For the Eckart barrier the remainder
   is 0.02 to 0.05 for `u` of 10.5 to 13.5, where `A` is 0.45 to 0.64. Near the crossover it
   is not small: 0.04 to 0.09 at `0.8 T_c` and 0.11 to 0.18 just above `T_c`, where `A` is
   zero. `A` is the leading term well below `T_c` only.
2. **The naive anchor is wrong; the skeleton anchor holds for the free-energy reduction.**
   With a zero-point change along the path (M1, M3) the naive anchor of `fe` is +0.07 to
   +0.14, and the naive-anchored correction is off by −0.066 ± 0.017 to −0.100 ± 0.008 in all
   four cases. The skeleton-anchored correction of `fe` has no detectable error below `T_c`
   on any model (rms 0.010 over eight points, `chi²` 0.4).
3. **For the free-energy reduction the anchor is a small correction.** The plain `C_S` of
   `fe` is already within 0.054 of the exact defect at all twelve points (rms 0.025), where
   the quantum rate of `fe` on its own is off by up to 0.35 (`G`). The transfer error is
   +0.02 to +0.05 on the curved paths (M2, M3) and zero within errors on M1 and M4. Below
   `T_c` the skeleton anchor (+0.02 to +0.03) accounts for it. Above `T_c` there is no
   anchor and errors of the same size remain: +0.025 ± 0.005 (M2) and +0.036 ± 0.005 (M3).
   The agreement below `T_c` therefore does not show that the anchor is the whole transfer
   error of a curved path.
4. **A skeleton with a different crossover temperature is not anchored reliably near `T_c`.**
   The SCT mass raises the barrier frequency of the skeleton by 25% (M2) and 20% (M3). At
   `0.8 T_c` the skeleton anchor of `fe+sct` (−0.15, −0.20) is twice the measured transfer
   error (−0.088 ± 0.011, −0.084 ± 0.009); at `0.55 T_c` and on M4, where the SCT skeleton
   keeps the crossover temperature, it agrees within 0.05. The bare-mass model of M4, with a
   barrier frequency 25% above that of the saddle, is anchored within errors at both
   temperatures (transfer errors −0.185 ± 0.016 and −0.219 ± 0.020, anchors −0.199 and −0.198)
   but has an unanchored error of −0.067 ± 0.011 at `1.25 T_c`, which is its own crossover.
5. **`C_S` is far less sensitive to the reduction than the quantum rate of the reduction.**
   The SCT mass overshoots the quantum rate by factors of 2 to 4 (`G` of −0.65 to −1.49
   below `T_c`), but `C_S` of `fe+sct` is off by 0.06 to 0.12. The bare potential on the
   path (`mep`) leaves out the zero-point change altogether (`G` up to 2.3 on M1 and M3) and
   its `C_S` is within 0.07. The SCT mass is still the worst of the
   path-based reductions: on every curved model `C_S` of `fe+sct` is too large below `T_c`,
   and `fe` without the SCT mass is closer to the exact defect at all six of those points.
6. **The path-free Eckart estimate is not better than `C_S`.** It reproduces the defect of M2
   (at two of three temperatures) and of M4 within 0.01, but is 0.02 to 0.10 too large
   wherever the zero-point energy changes along the path (M1, M3): rms 0.050 over twelve
   points, against 0.025 for `C_S` of `fe`. The remainder of those models is not that of an
   Eckart barrier with the adiabatic barrier height.
7. **Checks of the method.** The `mep` reduction of M1 is the Eckart barrier with `v = 2.1`;
   `run_rpmd.py` and `run_eckart.py` agree on its defect within 1.8 combined standard errors
   at the three temperatures. The two independent runs of each full model differ by less
   than one combined standard error in ten of twelve cases and by 2.1 at most.

## Limits

- Two-dimensional models with one symmetric Eckart barrier, harmonic transverse mode, no
  wells, no rotations. Three temperatures; the lowest is `0.55 T_c`.
- Statistical errors of the transfer errors are 0.005 to 0.02. Differences below about 0.02
  between corrections are not resolved at `0.55 T_c`.
- The exact rates use a finite-difference grid and a truncated oscillator basis; their
  convergence (below 3e-3) is recorded in `qm.json` but not folded into the errors.
- Only the instanton action was checked for bead convergence at every bead number; the
  RPMD defect itself at one point (`u = 11.4`).
- The anchors are not used by any correction in PyRPMD. This report does not mark any rate
  as physically converged.

## Reproduce

```bash
cd benchmarks/subspace_instanton
python run_qm.py results/qm.json                       # exact rates and instantons, about 45 min
python run_skeleton.py results/qm.json results/skeleton.json
for m in M1 M2 M3 M4; do for t in 1.25 0.8 0.55; do
  python run_rpmd.py $m $t results/qm.json results/rpmd_${m}_${t}.json        # full model and reductions
  python run_rpmd.py $m $t results/qm.json results/rpmd_${m}_${t}.json full   # second run of the full model
done; done
for u in 2pi/1.25 2pi/0.8 2pi/0.55; do for v in 1.75 2.1 2.702; do
  python run_eckart.py $u $v results/eckart_u<u>_v<v>.json                    # file names as in results/
done; done
python run_eckart.py 2pi/0.55 2.1 results/eckart_u11.424_v2.100_P64.json --nbeads 64
python report.py
```

The remaining Eckart points are `(u, v)` = (6, 2.1), (9, 1.2), (10.5, 2.1), (12, 2.1) and
(13.5, 2.1). With `OMP_NUM_THREADS=1`, one run of a full model takes 3 minutes with 32
beads, 10 with 64 and 30 to 55 with 96 or 128 on one core; a reduction or an Eckart point
takes 2 to 7 minutes.
