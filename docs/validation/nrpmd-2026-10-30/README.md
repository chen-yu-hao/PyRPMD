# NRPMD implementation validation — 2026-10-30

Base: `4115580fcd6f54aa5f74613772dc052f32e5481a` (PyRPMD 0.2.0).
This change implements experimental state-selected MMST NRPMD scattering.
These are implementation and numerical checks, not physical convergence of a
real chemical reaction or validation of electronic thermal-equilibrium rates.

## Local environment and results

Python 3.12, NumPy 2.3.5, SciPy 1.17.0, h5py 3.16.0, pytest 9.2.0;
Linux x86_64, compiled C++17/OpenMP free-ring extension. No CUDA, MPI execution,
real BKMP2 library, Torch or ASE was used for the new method.

- Full repository suite: **379 passed, 13 skipped, 96 subtests passed**.
  The 13 skips are 10 optional BKMP2 checks, two Torch checks and one ASE check.
- New NRPMD core, input/provider and workflow checks: **86 passed**.
- Independently unpacked sdist, rebuilt native extension, and reran those
  same three test modules: **86 passed**, no skips.
- `git diff --check`: clean.

The executable checks include constant two/three-state electronic propagation,
MInt momentum impulse against independent DOP853 integration, exact and near
energy degeneracies, symplectic Jacobian, time reversal, second-order full
Hamiltonian convergence, single-state and distinct uncoupled diagonal limits,
norm/energy diagnostics and transactional failure rollback.

Workflow checks propagate nonzero coupling, interrupt inside its support and
resume with a changed block size. Saved nuclear/electronic endpoints and traces,
parent thermal clocks and joint statistics equal uninterrupted results.
Terminal checkpoint recovery, missing electronic data, classifier errors,
all-bead incoming coupling, continuing through a coupled geometric exit,
signed weighted thermal/Maxwell/fixed-energy statistics and shared-bootstrap
covariance are covered. Serial and two-process trajectories match bitwise.
Unsupported MPI/GPU/bead execution and electronic equilibrium tasks are rejected.

## Artificial exchange example

Commands (fresh output directory):

```sh
pyrpmd validate examples/nonadiabatic/exchange.inp --check-pes
pyrpmd run examples/nonadiabatic/exchange.inp --workers 2 --output runs/nrpmd-check
pyrpmd resume runs/nrpmd-check --block-steps 32
pyrpmd analyze runs/nrpmd-check
```

The included artificial H + HD matrix model, P=4, two processes and eight
fixed-energy collisions gave eight nonreactive, zero product, zero unresolved,
and zero failed outcomes. Electronic population changed after coupling;
negative individual estimates were retained. Maximum population-sum error was
`3.40e-13`; maximum energy drift per bead was `1.14e-6` hartree (rounded upward).
The matrix finite-difference gradient diagnostic was `1.49e-11` hartree/bohr.
The completed-run resume/analyze summaries were exactly identical.

This example is an integration demonstration, not a fitted H3 surface or a
converged reactive cross section. It does not validate H + HO2, spin-orbit
transitions, an adiabatic derivative-coupling interface, or electronic thermal
initialization. See [method and restrictions](../../nonadiabatic.md).
