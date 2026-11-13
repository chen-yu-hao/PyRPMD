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

{{NOTES}}
