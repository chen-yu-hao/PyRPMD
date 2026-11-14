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
  reductions a tabulated one. 47 thermodynamic-integration windows from `s = -14` to the
  barrier, 32 walkers (16 for the reductions), 2048 recrossing trajectories. The full RPMD
  rate is divided by the harmonic finite-bead factor `B_P` of the transverse mode
  (`harmonic_bead_factor`).
- **Errors** are one standard error of the RPMD rates; the quantum rates and the instantons
  are deterministic. `chi²` is the mean of `(value / error)^2`: about one when a quantity
  vanishes within the statistical error.

Temperatures are in units of the crossover temperature `T_c = hbar lambda / 2 pi k_B` of the
full saddle.

## RPMD defect of the full models

{{DEFECT}}

## Transfer errors below the crossover temperature

`Δ` is the error of the corrected rate `k_RPMD(H) C_S` in the logarithm; `G` is the error of
using the quantum rate of the reduction on its own. An anchored correction replaces `C_S` by
`C_S exp(anchor)`, so its error is `Δ − anchor`. The naive anchor is not given where the
instanton of the reduction has a second unstable mode.

{{TRANSFER}}

Root-mean-square errors over the temperatures below `T_c` (0.8 and 0.55):

{{SUMMARY}}

## Transfer errors above the crossover temperature

At `1.25 T_c` there is no instanton and every anchor is zero.

{{ABOVE}}

## Symmetric Eckart barrier

`run_eckart.py`: units `hbar = m = omega_b = 1`; `A(u)` is the steepest-descent factor and
`R` the remainder. Rows with two bead numbers are bead-convergence checks.

{{ECKART}}

## Path-free estimate

`ln C_Eck = A_H + R_Eck(u, v)` with `u = beta hbar lambda` and
`v = V_ad / hbar lambda`, where `V_ad` is the barrier height plus the change of the
transverse zero-point energy. It uses the saddle, its frequencies, the full-dimensional
instanton and one 1D calculation, and no reaction path.

{{PATH_FREE}}

Difference `ln eps(H) − ln C_Eck`: {{PATH_FREE_SUMMARY}}.

## Findings

FINDINGS

## Limits

LIMITS

## Reproduce

```bash
cd benchmarks/subspace_instanton
python run_qm.py results/qm.json                       # exact rates and instantons, about 45 min
python run_skeleton.py results/qm.json results/skeleton.json
for m in M1 M2 M3 M4; do for t in 1.25 0.8 0.55; do
  python run_rpmd.py $m $t results/qm.json results/rpmd_${m}_${t}.json
done; done
python run_eckart.py 2pi/0.8 1.75 results/eckart_u7.854_v1.750.json     # one per (u, v) of the table
python run_eckart.py 2pi/0.55 2.1 results/eckart_u11.424_v2.100_P128.json --nbeads 128
python report.py
```

`OMP_NUM_THREADS=1`; the RPMD jobs take 4 minutes (32 beads) to about TIMING on one core.
