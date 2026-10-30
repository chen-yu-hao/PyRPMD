# Surface implementation validation — 2026-10-30

Base: `a53c49f01e8024b9b6fa1e3c4ab5e059f8172533` (merged native propagation).
The initial implementation is local commit `77c6e1bc5e1cace445c865fad4726327d619ccc8`.
The initial record below covers that implementation and its exact
source checksums in [results.json](results.json). It establishes software and
numerical contracts, not converged catalytic rate predictions.

## Integration with current PyRPMD 0.2 main

Before publication, upstream advanced to
`4115580fcd6f54aa5f74613772dc052f32e5481a`. The surface branch was merged with
that commit, retaining the new integrators, checkpoint schema, adaptive
umbrella workflow and gas-phase reaction networks. The native extension was
rebuilt and advertises OpenMP plus `cayley-bcb`, `normal-mode-bab` and
`velocity-verlet`.

Post-merge checks, with matching source hashes and machine-readable evidence
in [integration-results.json](integration-results.json):

- Full regression: **413 passed, 12 skipped, 98 subtests passed**
  ([log](integration-pytest.txt)). Ten skips need the optional real BKMP2
  build, and two need PyTorch.
- Three newly added adaptive-surface smoke cases were collected separately
  after that suite: **3 passed** ([log](integration-adaptive.txt)). Each runs
  real periodic/fixed-atom pilot and production sampling with a different
  integrator, validates v2 checkpoints, and completes a resume without new
  force evaluations or propagation.
- Actual four-rank `G=2 × B=2` surface workflow repeated successfully:
  **27 array/checkpoint files exactly match serial execution** and both
  completed resumes pass ([log](integration-mpi.txt)).
- The original completed `diffusion.inp` run resumed under version 0.2.0,
  migrated its manifest to schema 2, and retained exactly the same rate
  estimate (`1.3880235179088665e10 s^-1`).

Integration tests also cover fixed layers across all three real-time
integrators, BCOCB/BAOAB sampling with RNG rollback, both product directions,
and the checkpoint integrity envelope. Periodic/fixed-atom inputs are
explicitly rejected by gas-phase networks and by the currently unsupported
metadynamics combination. Surface FES supports adaptive umbrellas;
`surface_rate` still requires a fixed, declared set of windows.

The complete-example numbers and original build evidence below belong to
the initial implementation; they are preserved as historical measurements,
not relabeled as new full runs after integration.

## Local suite

Python 3.12, NumPy/SciPy, ASE, h5py and the compiled C++/OpenMP kernel:

```text
257 passed, 6 skipped, 87 subtests passed
```

[Full pytest summary](pytest.txt). Four skips require the optional real BKMP2
build; two require PyTorch. The existing i-PI loopback test passed when run
with local socket access. This environment initially rejected socket creation;
that infrastructure restriction was resolved for the final full-suite run.

New tests cover skew/partial-PBC nearest images against exhaustive searches,
CV finite-difference gradients, fixed-layer dynamics against a reduced
Hamiltonian, reversibility, Andersen and PILE-L, both block failure contracts,
steering with fixed atoms, interrupted/complete resumes, analytic basin
integrals, invariance under coordinate scaling and PMF offsets, correlated
bootstrap support, and paired positive/negative initial flux. The ASE examples
also check real EMT propagation and preserve the pre-existing formula-string API.

## Actual MPI validation

Four local MPICH ranks ran the new surface-rate workflow with **G=2 × B=2**.
A separate serial process ran the same scientific input. The test compares
all 27 umbrella/parent/child and PMF/recrossing array files exactly, including
q/p, RNG states, sample streams, flux numerators/denominators and uncertainty.
Completed resumes of both runs perform zero additional propagation.

[MPI result](mpi.txt):

```text
Surface rate serial versus G2 B2 MPI: PASS; 27 array/checkpoint files match exactly, fixed substrate remains exact, and both completed resumes pass.
```

The host lacks `process_vm_readv`, which the default MPICH CMA optimization
attempted to use. The successful run used the supported transport settings:

```bash
UCX_TLS=tcp,self MPIR_CVAR_CH4_CMA_ENABLE=0 \
  .venv/bin/mpiexec -n 4 .venv/bin/python \
  tests/mpi_surface_smoke.py validation/surface-mpi
```

This is a compatibility test, not a multi-node, distributed-native-loop or
GPU performance claim. The same script is added to the ordinary CPU CI jobs.

## Complete examples

| Example | Executed evidence | Physical status |
| --- | --- | --- |
| `diffusion.inp` | 15 windows, 30 streams, 15,000 samples, 16 constrained parents, 64 children; all 20 PMF bootstrap draws supported | `converged=false` |
| `fes2d.inp` | 27 windows, 8,100 samples, connected overlap | `converged=false` |
| `ase_emt.py` | Real ASE Cu(111)/Cu-adatom calculator; skew cell, four fixed bottom atoms, mobile upper layers; 3 windows, 300 samples, connected overlap | `converged=false` |

The periodic teaching example gives a finite-time estimate of
`1.3880235179e10 s^-1`, kappa 1, and maximum child endpoint energy error
`2.0975e-4 kBT` per bead. **Its relative sampling standard error is about 198%**;
it is deliberately not reported as a converged physical result. The potential
is separable and its substrate is rigid, so it also does not validate thermal
site-to-site diffusion or a real hydrogen/copper chemical model. Its value is
the end-to-end execution, normalization, diagnostics and exact completed resume.

The source archive and wheel build succeeded. The sdist was checked for the
new geometry, analysis and workflow modules, documentation, runnable examples
and surface integration tests. Native ABI/GPU support and a package release
are not added by this change.

## Reproduction

```bash
python -m pip install -e '.[test,ase,mpi]'
python -m pytest -q
mpiexec -n 4 python tests/mpi_surface_smoke.py runs/surface-mpi
pyrpmd run examples/surface/diffusion.inp --output runs/surface-diffusion
pyrpmd resume runs/surface-diffusion
pyrpmd run examples/surface/fes2d.inp --output runs/surface-fes2d
pyrpmd run examples/surface/ase_emt.py --output runs/surface-ase
```

For real catalysis, replace the teaching potential with a validated surface
PES and converge slab dimensions/layers, fixed-layer choice, beads, timestep,
basin support/truncation, independent sampling, histogram resolution and the
finite-time transmission plateau. See [the surface documentation](../../surface-catalysis.md).
