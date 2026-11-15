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

At `1.25 T_c` the full models have no instanton and the anchors are zero, except −0.004 for
the SCT skeleton of M2, which is just below its own crossover.

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
