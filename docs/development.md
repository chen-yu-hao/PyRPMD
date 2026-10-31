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

## Shared utilities and module boundaries

Import shared primitives from their owning module when adding code:

- `validation.py` owns scalar integer, finiteness and sign/bounds checks. Integer
  checks reject booleans and fractional values. Callers select the exception
  type and diagnostic wording; unit conversion and task-specific constraints
  stay with their domain resolver.
- `io.py` owns portable JSON conversion, atomic JSON writes and fingerprint
  encoding. `jsonable`/`write_json` reject nonfinite metadata by default;
  `report_jsonable`/`write_report_json` represent unavailable numerical estimates
  as JSON `null`. The callable, dataclass and `to_dict` policies preserve legacy
  schemas; they are explicit choices, not interchangeable serializers.
- `checkpoint.atomic_savez` owns atomic compressed NPZ writes, the integrity
  envelope and per-array checksums. `save_metadata_npz` packages dRPMD metadata
  and delegates to that writer. Workflow modules define payload contents and
  restart checks, and use `load_checkpoint` to validate the container.

Scientific identities retain their existing schema and encoding:

| Entry point | Identity contract |
| --- | --- |
| `io.fingerprint(config, system)` | Manifest configuration/system identity; retains analysis settings and excludes output destination, execution topology and transient input metadata. |
| `io.fingerprint(config, system, extras)` | Sampling-stream identity; excludes analysis-only and execution settings, while retaining the sampling measure, flux axis, CV geometry and stream-specific values. FES and OPES use this path. |
| `io.fingerprint_payload(payload, ...)` | Shared encoding for an explicitly defined payload. dRPMD retains its versioned physics payload, parent-chain policy and local dependency digests in `drpmd_persistence.py`. |

Do not replace these schemas with one new digest during a structural refactor:
that would reject existing manifests and checkpoints even when the physics is
unchanged. Changes to a scientific identity need an explicit compatibility or
migration decision and regression evidence.

The main implementation boundaries, relative to `src/pyrpmd/`, are:

| Family | Responsibility |
| --- | --- |
| `input.py`, `input_*.py` | `JobSpec` loading, overrides and resolution order remain in `input.py`; schema, scientific options, paths, runtime settings, systems/providers and task contracts have separate modules. |
| `workflows/drpmd.py`, `drpmd_*.py` | The entry point dispatches work; sampling, trajectory propagation, outcome classification, analysis and restart conventions are separate modules. |
| `workflows/rate.py`, `rate_*.py` | The entry point coordinates PMF and recrossing stages; constraints, options, PMF preparation, parent/child tasks and aggregation have separate modules. |
| `analysis/fes.py` | FES reconstruction and established analysis imports. |
| `analysis/mbar.py`, `umbrella.py`, `fes_statistics.py`, `rate_pmf.py` | MBAR solving/reusable weights, umbrella energies/histograms/UI, correlation/bootstrap statistics, and rate-barrier estimation respectively. |

Compatibility modules keep established imports and deserialization paths with
explicit re-exports or small adapters. dRPMD worker entry points remain
importable for process/MPI dispatch; factory and persistence callbacks are passed
into kernels explicitly. FES adapters pass the solver, block-resampling and UI
callbacks used by existing diagnostics. New implementation modules should depend
on the owning helpers rather than import the compatibility module back again.

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
python -m pip install dist/pyrpmd-0.2.0-*.whl
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
