# Ecosystem and trust validation — 2026-11-01

This record accompanies the trajectory exporters, OpenMM/LAMMPS providers,
bond constraints, distribution automation, reference translations and
literature-evidence work. It separates implementation verification from
physical convergence and publication. The base revision is
`2f907ba96d22cb134198715e398b1d9f8d83d929`; exact runtime/test source hashes
are recorded in the integration evidence rather than inventing a future
commit identifier.

**Final local result:** 4,475 tests and 357 subtests passed; 117 optional-runtime
tests were skipped. The suite finished in 270.68 seconds on Python 3.12.14,
and every recorded source hash remained unchanged. The final native Linux
wheel also passed an independent fresh-install smoke test outside the checkout.

## Evidence map

| Record | What it establishes | What it does not establish |
| --- | --- | --- |
| `integration/` | Full regression result, source stability, constrained public run/restart/export and real reader agreement | A converged rigid-water quantum ensemble, GUI testing or GPU execution |
| `distribution/` | Real local native wheels and conda builds, installed-package checks, strict documentation and workflow syntax checks | Completed remote portable-wheel matrix or publication to a package registry |
| `water-pilot/` | Real q-TIP4P/F model, PME/virtual-site force checks, short RPMD trajectory and replayable diagnostics | Liquid diffusion, resolved IR bands, statistical convergence or literature reproduction |
| `h3-archived-comparison/` | Fresh arithmetic comparisons of explicitly identified existing pilot records | A fresh PES build, fresh H3 dynamics or a successful rate reproduction |

The final full-suite log and JSON record are in `integration/`. The JSON
includes the command, exit status, source-file hashes and an end-of-run
comparison to detect concurrent code changes. The optional-dependency skips
remain visible in the log. A passing suite does not imply those skipped
backends were exercised.

## Concrete integration checks

The self-contained constrained-water example executes 160 Python RATTLE
steps over eight beads, resumes its saved state and exports 17 centroid
frames to extxyz, H5MD and ASE `.traj`. The final bead bond-length error is
below `1e-12` relative, and the final fractional bond-extension rate is below
`1e-12` in inverse atomic time. ASE reads both of its formats and MDAnalysis
reads the H5MD trajectory with matching positions. Full numeric results,
dependency versions and output checksums are in `integration/rigid-water.json`.
This zero-potential example establishes kinematics and interoperability.

The actual OpenMM and LAMMPS provider tests cover analytic energies/forces,
periodic translations, virtual-site force transfer, reciprocal-space
electrostatics and physical LAMMPS virials against cell derivatives. OpenMM
models that declare constraints must have matching PyRPMD constraints.
Direct unsupported workflow calls reject declared constraints before creating
outputs or evaluating a potential. The LAMMPS dependency-file test confirms
that a changed declared model asset invalidates restart identity.

The two-rank LAMMPS smoke uses a separate `MPI.COMM_SELF` engine per worker
and verifies host MPI remains alive after closing the engines. The dedicated
ecosystem CI also requires working external engines/readers, executes the
public provider examples, resumes them and exports their trajectories.

## Scientific scope

SHAKE/RATTLE currently supports the fixed-cell velocity-Verlet Cartesian
ring Hamiltonian with projected Maxwell/Andersen sampling. Native constrained
propagation, other thermostats/integrators, pressure control and constrained
reaction/FES workflows are rejected. Unconstrained quantum kinetic and
internal-energy estimators are not reported for constrained materials runs.
The implementation is not a specialized rigid-rotor quantum propagator.

The flexible q-TIP4P/F pilot uses eight molecules, four beads and 12.8 fs of
production. Its energy/force and replay checks pass, but diffusion is still
classified `ballistic_or_drifting`. Its saved RPMD IR diagnostic is not a
reproduction of the original paper's PA-CMD liquid spectrum. All physical
validation flags remain false. No requested literature family is presented
as reproduced by this work.

The requested benchmark inventory and model/data/convergence blockers are
maintained in [Literature validation](../../literature-validation.md), with
the publication requirements in [Publication readiness](../../publication-readiness.md).
No release, conda-forge submission, journal submission or GPU performance
claim is part of this evidence record.
