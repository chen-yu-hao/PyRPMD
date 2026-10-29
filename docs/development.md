# Development and verification

Install from the checkout with `python -m pip install -e '.[test]'`, then run
`python -m pytest -q`. Native tests are enabled when the compiled extension is
available; OpenMP-specific tests are skipped on a serial-only build. The test
suite deliberately compares the Cayley map to an independent dense midpoint
solve and compares analytic CV/PES derivatives with numerical derivatives.

Optional ASE and Torch tests skip if those libraries are not installed.
Install them to exercise their actual CPU runtime paths. Socket integration
requires permission to create a local loopback socket. For a NumPy-only install,
set `PYRPMD_NO_NATIVE=1` before building in a clean environment; requesting native
execution with `native='required'` must fail explicitly if it is unavailable.

## MPI

```bash
mpiexec -n 4 python tests/mpi_integration.py
```

This checks two trajectory groups, two bead ranks per group, force ordering,
initialization failure propagation, continuous parent streams, restart without
duplicate trajectories, and mocked GPU placement. The mock does not perform
CUDA computation. Run the additional FES/rate MPI smoke script if present in
`tests/mpi_workflow_smoke.py /fresh/shared/output`; these multi-stage workflows use root broadcasts between
stages and subgroup collectives inside each PES call.

The MPI library loaded by mpi4py must match the launcher. Cluster runs need a
shared input/model/output filesystem and the same software installation on all
nodes. Site-specific transport restrictions belong in the job environment, not
in the scientific configuration. Avoid oversubscribing BLAS/OpenMP pools.

## Package construction

```bash
python -m pip install build
python -m build
python -m pip install dist/pyrpmd-0.1.0-*.whl
pyrpmd info
```

The source distribution includes the native source/header, C example, tests,
scientific docs, working input examples and third-party notices. The wheel
contains the Python package, compiled extension when enabled, public ABI header
under `include/pyrpmd`, and the C example under `share/pyrpmd/examples`.

Native wheels are platform- and Python-version-specific; compile against each
cluster's supported toolchain. Set `PYRPMD_NO_OPENMP=1` when the compiler lacks
OpenMP, or `PYRPMD_NO_NATIVE=1` for the reference path. No prebuilt proprietary or
third-party chemical PES is distributed.

## Contribution boundaries

- Keep PES physical force units/sign independent of dynamics and biases.
- Keep external models process local; use serializable factories in jobs.
- Add a numerical test for changes to sampling measures, forces or constraints.
- Preserve random stream/task identities and include changes to sampling
  Hamiltonians in checkpoint identity.
- An equilibrium FES, a finite-time rate estimate and a converged chemical rate
  are distinct outputs. Tests on teaching potentials do not validate a named
  real reaction.
- Retain MIT notices and pinned source provenance when adapting RPMDrate or
  Caracal code; bundled third-party engines/models have separate licensing.
