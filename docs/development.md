# Development and verification

Install from the checkout with `python -m pip install -e '.[test]'`, then run
the complete suite, including validation campaigns:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
```

On PowerShell, set `$env:OMP_NUM_THREADS = "1"` and
`$env:OPENBLAS_NUM_THREADS = "1"` before running the Python command.
Native tests are enabled when the compiled extension is
available; OpenMP-specific tests are skipped on a serial-only build. The test
suite deliberately compares the Cayley map to an independent dense midpoint
solve and compares analytic CV/PES derivatives with numerical derivatives.

Optional ASE and Torch tests skip if those libraries are not installed.
Install them to exercise their actual CPU runtime paths. Socket integration
requires permission to create a local loopback socket. For a NumPy-only install,
set `PYRPMD_NO_NATIVE=1` before building in a clean environment; requesting native
execution with `native='required'` must fail explicitly if it is unavailable.

## Library tests and validation campaigns

For the faster library test entry point, use:

```bash
python -m pytest -q -m "not campaign"
```

The registered `campaign` marker describes a test module's dependencies, not
its runtime or scientific convergence. `tests/conftest.py` inspects Python's
syntax tree without importing benchmark code. It marks the entire module when
it has an `import benchmarks...` / `from benchmarks...` statement (including
inside a function), a literal `import_module("benchmarks...")` / `__import__`
call, or a static benchmark path in an assignment, call argument or `/` path
expression. Both `ROOT / "benchmarks" / "family" / "driver.py"` and
`ROOT / "benchmarks/family/driver.py"` are recognized. For the repository's flat
test layout, local `.py` test/helper imports are followed transitively, with
cycles handled. A comment, an embedded Python
source string, a test filename prefix, or the installed optional packages alone
do not determine the marker. A mixed module containing library and campaign
tests is marked as a whole; split such modules to move independent tests into
the fast suite.

Default pytest selection is unchanged and runs both groups. Use
`python -m pytest -q -m campaign` to select campaigns explicitly. The exact
expression `-m "not campaign"` also omits campaign modules **before import**,
including explicitly supplied file paths, so POSIX-only campaign dependencies
cannot break Windows test collection. The summary reports the number of modules
omitted; these modules are not reported as collected or skipped tests. Other
marker expressions use ordinary pytest selection after collection and may
therefore require campaign dependencies. Computed paths/import names are not
statically resolved; use a literal dependency or an explicit `pytestmark =
pytest.mark.campaign` for selection after import when adding such a module.

Linux CI continues to run the full suite and the existing MPI integration
commands. Separate `macos-15` and `windows-2022` jobs install `.[test]` and run
the fast library suite with one OpenMP/BLAS thread. The normal native build is
retained: its non-Linux configuration is serial, and OpenMP-only tests already
have capability guards. Windows checkout preserves repository line endings so
scientific/source digests are not changed by Git's automatic CRLF conversion.

Optional dependencies must be guarded at the narrowest useful scope. For
example, use `pytest.importorskip("matplotlib")` inside a plotting test while
continuing to run adjacent data-contract checks without plotting extras.
POSIX-specific Caracal/JaxPIP writers require `fcntl` and are skipped with an
explicit reason where unavailable; their portable readers and validation
checks remain covered. Only the directory-fsync fault-injection parameter is
skipped on non-POSIX systems, matching the implementation's durability steps.

### Observed Windows runtime limitations

The first Windows library run exposed runtime gaps in addition to test-only
path assumptions. `tests/conftest.py` records a Windows-only registry of exact
affected test names (and parameter names where only part of a test is affected),
each with a diagnostic skip reason. The
[initial Actions log](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41476577723/job/125383616891)
provides the failure evidence. These cases still execute on Linux and macOS.

The drive-path repair in [f9e85b4](https://github.com/chen-yu-hao/PyRPMD/commit/f9e85b43152cc768939cd7281649142cb30da8ed)
is on `main`. T18b removes that group's 15 registrations, restoring 16
parametrized source-provenance and resume-preflight cases on Windows. The
registry shrinks from 37 to 22 names, or from 61 to 45 parametrized skips in
the [main baseline job](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41518522419/job/125528228876).
The three remaining groups depend on [PR #83](https://github.com/chen-yu-hao/PyRPMD/pull/83),
which was still unmerged when this cleanup was prepared against `main@f877051`.
Remove them after that repair is integrated and its Windows library job passes.

| Limitation | Affected checks | Status / removal requirement |
| --- | --- | --- |
| Native trajectory publication calls `os.fsync()` on a read-only descriptor and can unlink a mapped file on Windows. | 19 registrations / 40 parametrized native streaming, journal, publication and restart cases. In-memory native dynamics and pre-publication failure checks remain enabled. | Retained: integrate PR #83's portable publication and rollback repair, then validate on Windows. |
| File/source walkers omitted drive-letter paths or split a `.py:factory` at the drive colon. | 15 registrations / 16 source-provenance, model-change and early resume-preflight cases. | Removed: f9e85b4 repairs parsing and hashing; the legacy-provenance refusal below supplies an explicit compatibility policy. |
| Archived rate generations write relative mapping keys using `\\`, while the report reader looks them up using `/`. | Two registrations / two archived-generation report/reanalysis cases. | Retained: integrate PR #83's writer/reader compatibility for authenticated historical archives, then validate on Windows. |
| Simultaneous `os.replace()` calls targeting one JSON destination can raise `WinError 5`. | One registration / three concurrent writer-policy variants. | Retained: integrate PR #83's bounded Windows replacement retries, then validate on Windows. |

This registry does not certify the skipped Windows capabilities. Keep its
entries narrow and remove them only with regression evidence for the relevant
runtime fix. Do not replace it with blanket native, resume, or Windows skips.
Additional capability guards cover Python builds without `socket.AF_UNIX` and
the existing POSIX-only writers. The C ABI tests still execute on Windows when a
compiler is present; cleanup of temporary loaded DLLs is best-effort only on
Windows because those files remain locked until the process releases them.

### Windows runs with missing historical source hashes

Before f9e85b4, Windows inputs such as `C:\models\pes.py:build` and
`C:\models\weights.bin` could be omitted from a run's `manifest.json` file
hashes. Correctly including these dependencies changes the identities of
workflows that bind those hashes into their checkpoints. Other workflows could
previously pass the manifest's one-way file check despite the missing evidence.

T18b uses **explicit refusal**, not identity migration. After validating the
scientific settings, system and previously recorded file hashes, resume checks
the saved Windows platform and saved drive-path references. If a referenced
file is now hashed but has no historical entry, it raises `InputError` beginning
`Cannot resume this Windows run`, lists the affected files and explains the
missing provenance. Explicit plugin/model file references are also checked if
the file has since disappeared; output/base directories are excluded. Public
entry points retain the existing `Root stage failed: InputError:` wrapper.
This check also runs before public rate/dRPMD extensions,
including pending extensions reached through `pyrpmd resume`, acquire their
lock or change a ledger. Refusal does not migrate or rewrite the manifest,
resolved input, checkpoints or RNG state.

The diagnostic identifies an evidence gap, rather than inferring a release
from `version`: both affected and repaired code can report version `0.2.0`.
Hashing a file today cannot prove what its contents were when the trajectory
was generated. There is no ignore-identity option and no automatic insertion
of current hashes into an old manifest.

Recovery is a fresh run:

1. Keep the original run directory and its checkpoints for inspection.
2. Locate the original input and verify the intended plugin/model files in its
   original input environment. Do not edit the old manifest's `files` mapping
   or checkpoint fingerprints to make resume pass.
3. Run the original input into a **new, empty output directory**, for example
   in PowerShell:

   ```powershell
   pyrpmd run .\original.inp --output .\runs\fresh-after-windows-hash-fix
   ```

4. Treat this as a new trajectory with newly recorded provenance. Do not copy
   old checkpoints into it or use `--resume` for the fresh run.

Runs whose Windows hashes were recorded completely continue through the
existing checks. A changed or missing recorded file still reports
`Input/model file changed or is missing`; a change to scientific settings keeps
its existing diagnostic. POSIX identities, manifest/checkpoint schemas and
authenticated schema-1 migration are unchanged. The new regression module
`tests/test_windows_resume_provenance.py` emulates drive paths and
`os.path.isabs` on Linux and uses actual drive paths in the Windows CI job.

The macOS fixes canonicalize temporary paths and keep Unix socket names short.
The symmetric teaching-saddle tests accept either physically equivalent
entrance orientation; the short seeded MBAR fixture specifies its intended
reactant fragments. Both Hessian eigenvector signs are checked without relaxing
PMF support or convergence criteria.

For the read-only file/hash inventory and the staged validation-data archive
proposal, see [Data archive plan](data-archive-plan.md). No data is moved or
removed by the inventory tool.

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
  `write_json` delegates publication to `atomic_write_text`; `ensure_ascii` and
  `trailing_newline` preserve each workflow's existing text format.
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
