# RPH 2D subspace model: validation against exact rates (2026-11-13)

This report checks the subspace correction against exact full-dimensional
quantum and RPMD rates on model potentials. It covers the defaults introduced
together: the minimum-energy path (`path.source = "mep"`), the reaction-path
Hamiltonian 2D model (`subspace.model_2d = "rph"`) and the reference-slice
criterion (`quantum.reference_density_tolerance`).

Reproduce with `benchmarks/subspace_rph/run.py`; the commands are listed below.
The tables are generated from `results/*.json` by `report.py`.

## Models

All coordinates are mass-weighted with unit mass (hbar = 1, hartree). A
symmetric Eckart barrier (16 mEh, barrier frequency 1500 cm^-1; the crossover
temperature is about 345 K) lies along `x`, and the transverse valleys bend.

| model | dimension | transverse coordinates | what it tests |
|---|---|---|---|
| `curved_2d` | 2 | `y`, 2000 cm^-1, valley `y = 10 exp(-x^2/400)` | planar bend: the RPH development is the plane itself, so the RPH 2D model must be exact |
| `curved_3d` | 3 | plus `z`, 1200 cm^-1 softened by 30% at the barrier, valley `z = 0.25 x exp(-x^2/625)` | non-planar path: the RPH normal carries only part of the curvature; ZPE changes along the path |
| `embedded_8d` | 8 | `curved_3d` plus five uncoupled modes (800–3600 cm^-1), all eight coordinates rotated by a random orthogonal matrix | uncoupled constant-frequency modes cancel from both rates, so every `C_S` must equal its `curved_3d` value |

## Method

For every model and temperature, `run.py` uses the pipeline's own builders
(`path_model`, `plane_model`, `rph_model`) on the default MEP path through a
rate-run stand-in with `rigid_body = False`, and solves:

- `C_S = k_QM(H_S) / k_RPMD(H_S)` for the 1D model, the 1D model with the SCT
  effective mass (`curvature = "small"`), the 2D plane model and the 2D RPH
  model, with the pipeline's quantum and RPMD solvers;
- the exact correction `C = k_QM(full) / k_RPMD(full)` of the whole model: a
  dense (2D) or flux-eigenstate (3D) exact quantum rate on the full grid and an
  RPMD rate with the analytic potential, both normalized far out in the
  reactant channel, where the quantum density correction is below 1e-3.

`C_S / C - 1` (in parentheses) is the relative error of the corrected rate
`k_RPMD(full) * C_S` against the exact quantum rate. Errors are one standard
error of the RPMD rate (the quantum rates are deterministic). RPMD: 48 windows,
32 walkers, 800 sweeps, 1024 trajectories. Quantum: 30 kT and at least 0.055
Eh (six quanta of the stiffest mode) of energy truncation. Default grids
unless stated.

## Corrections

{{CORRECTIONS}}

## Component rates

{{RATES}}

## Checks

### RPH reproduces a two-dimensional system

On `curved_2d` at 300 K the RPH model's quantum rate equals the exact quantum
rate of the whole system normalized on the same slice:

{{SAME_SLICE}}

### Invariance under embedding and rotation

{{EMBEDDING}}

### Convergence checks

{{CONVERGENCE}}

## Findings

1. **Both 2D models are within a few percent of the exact correction; the 1D
   models are not.** The plane model (with its absorber fixed, finding 5e) is
   within 1% at 300 K and 2% at 200 K, the RPH model within 1–2% at 300 K and
   3.3% at 200 K. Combining the statistical errors of the subspace and the exact
   RPMD rates (about 1.3% at 300 K, 2% at 200 K), every 2D result lies within
   1.7 standard errors of the exact value. The RPH residual comes mostly from
   the reference-slice tolerance: tightening `reference_density_tolerance`
   from 1e-2 to 1e-3 moves the 300 K value from -1.9% to -1.3% (Convergence
   checks). On `curved_2d` the RPH model is the system itself, and its quantum
   rate equals the exact rate on the same slice to 0.4%.
2. **1D underestimates the correction** (-4 to -6% at 300 K, -14% at 200 K): it
   cannot cut the corner. **1D + SCT overestimates at 300 K (+4 to +5%) and
   underestimates at 200 K (-5%)**; the SCT effective mass is a one-dimensional
   approximation of exactly what the 2D models treat.
3. **These models do not separate plane and RPH.** Their paths stay close to
   one plane, which is where the plane model is exact. RPH needs no plane: its
   development follows the path, and `captured_curvature` reports how much of
   the curvature its single normal carries (1.000 for `curved_2d`, 0.998 for
   `curved_3d`). A path that leaves every plane (a hairpin, or curvature in
   two directions at once) would distinguish them; no exact reference for
   one is included here.
4. **Embedding invariance holds** for 1D, plane and RPH within statistical
   error (RPH: identical to three digits). 1D + SCT differs by 1.8 standard
   errors (1.255 vs 1.284), which may be noise or the SCT effective mass's
   sensitivity to the path sampling; it is not resolved here.
5. **Defects found and fixed during this validation** (each has a unit test):
   a. the RPH model walled off the grid beyond its box: the quantum rate
      absorbed first and never saw it, but ring polymers bounced back
      (kappa 0.84 instead of 0.98), `C_S` +20%;
   b. `exact_quantum_rate` checked a supplied absorber against end caps it
      does not use, refusing the RPH model at 200 K;
   c. the quantum rate is normalized on the DVR line nearest the reference
      and the RPMD TI exactly at it: 2D references are now snapped to a node;
   d. the reference slice was chosen by a 0.3 kT energy rule only; on a slope
      the quantum and the centroid reactant densities differ at O(hbar^2)
      (`beta^2 V''/12` etc.), ~5% at the 300 K slice of `curved_2d`. Slices
      now also need `beta^2|V''|/12 + beta^3 V'^2/24 <= 1e-2`;
   e. the plane model's reactant absorber stopped at the path start, so its
      ramp never reached full strength: k_QM 22% low (4% with the former
      reference slice).

## Limits of this validation

- The models are abstract mass-weighted potentials with a single barrier;
  there are no rotations, rigid-body modes, complexes or anharmonic baths.
  The atomistic path code is tested separately (`test_subspace_path_cores.py`,
  `test_subspace_reactant_climb.py`).
- `curved_3d` bends only weakly in `z` (the RPH normal captures 99.8% of the
  weighted curvature); it mainly tests the adiabatic `z` mode, whose
  frequency softens at the barrier. A strongly twisting path, where one normal
  cannot carry the curvature, is not covered by an exact reference here.
- The bead number is fixed (P = 32 at 300 K, 48 at 200 K) and enters the
  subspace and the full RPMD rates alike; bead convergence was not checked
  separately.
- The exact 3D quantum rate uses the flux-eigenstate route on a 3 x 3 x 3.5
  grid; its grid convergence was not checked separately.

## Reproduce

```bash
cd benchmarks/subspace_rph
python run.py --model curved_2d --temperature 300 --nbeads 32 --full --full-spacing 2 2 --full-reference 145 --out results/curved_2d_300K.json
python run.py --model curved_2d --temperature 200 --nbeads 48 --full --full-spacing 2 2 --full-reference 155 --out results/curved_2d_200K.json
python run.py --model curved_3d --temperature 300 --nbeads 32 --full --full-spacing 3 3 3.5 --full-reference 145 --out results/curved_3d_300K.json
python run.py --model embedded_8d --temperature 300 --nbeads 32 --out results/embedded_8d_300K.json
python run.py --model curved_2d --temperature 300 --nbeads 32 --models rph --reference-density-tolerance 1e-3 --out results/curved_2d_300K_tol1e-3.json
python run.py --model curved_3d --temperature 300 --nbeads 32 --models plane --spacing-2d 2.47 2.47 --out results/curved_3d_300K_plane_fine.json
python report.py README.template.md ../../docs/validation/2026-11-13-subspace-rph-2d/README.md
```

A dense 2D solve needs up to about 5 GB; run the memory-heavy cases one at a
time on a 16 GB machine.
