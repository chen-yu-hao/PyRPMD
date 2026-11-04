# T48: MetaD / OPES implementation consolidation

This is a refactor of `801953dc9e2ca620f36069eb4551c41c529c432a`.
The maintainer explicitly selected that base after `origin/main` advanced to
`cb28829`, which reintroduced the rejected T26 module patching. No T26 or T31
changes are included. No numerical formulas, random streams, checkpoint schemas,
run fingerprints, or input defaults are changed.

## Implementation inventory

| Old implementation | Destination / disposition |
| --- | --- |
| `src/pyrpmd/_opes_serial.py` | Functions/classes moved into `opes.py`; six-line deprecated import alias retained |
| `src/pyrpmd/_shared_metadynamics_serial.py` | Class moved into `shared_metadynamics.py`; private module deleted |
| `src/pyrpmd/workflows/_metadynamics_base.py` | Moved into `workflows/metadynamics.py`; private module deleted |
| `src/pyrpmd/workflows/_opes_serial.py` | Moved into `workflows/opes.py`; private module deleted |
| `src/pyrpmd/workflows/_shared_metadynamics_serial.py` | Moved into `workflows/shared_metadynamics.py`; private module deleted |

The retained alias is referenced by the historical module-name entry in
`block_propagation._PURE_BIASES`. It only imports the three original names and
issues `DeprecationWarning`; it contains no implementation or module mutation.
The other four names have no remaining source/test importers or references in
the generated checkpoint payloads. The public exports are unchanged.

The seven `_distributed_*` helpers use explicit ordinary imports. Public
workflow entry points import a distributed implementation inside the selected
branch, avoiding cyclic initialization. The OPES Andersen wrapper is called
explicitly; its refresh clock is unchanged. The distributed validators and
`walkers_mode` whitelist entries now live directly in the public resolvers.
There are no changes to shared registration points or reserved files.

`definition-comparison.json` records an AST comparison against the old private
implementations. All OPES mathematical definitions and all original shared
MetaD class methods are unchanged. Workflow changes are limited to the two
resolvers and serial/distributed run/reanalysis dispatch; propagation,
checkpoint readers/writers, analysis, and RNG code are unchanged.

## Existing monkeypatch locations

No existing test patch target needed renaming: the public locations previously
forwarded by the shims now contain the real implementations. In particular:

- `tests/test_opes.py`: `workflows.opes._new_bias` and
  `workflows.opes.make_task_potential`.
- `tests/test_sampling_public_api.py`, `tests/mpi_opes_smoke.py`:
  `workflows.opes.make_task_potential`.
- `tests/test_shared_metadynamics.py`: `workflows.shared_metadynamics.make_potential`,
  `workflows.metadynamics.make_task_potential`, and
  `workflows.metadynamics._sample_replica`.
- Bias class methods and `Dynamics` methods used for fault injection continue
  to patch those real class objects.

The new import-order test launches fresh interpreters and runs the public
`run` / `resume_run` API through actual input parsing and FES dispatch. It
compares all checkpoint bytes and manifest physics fingerprints across import
orders, and checks the deprecated alias. It uses no mock implementation.

The AST test scans every Python file under `src/pyrpmd`, including top-level
loops and conditional imports. It rejects module replacement, namespace copies,
and cross-module attribute assignments / `setattr`. The two explicit exceptions
for user-file plugin loading are `input.py` and `potentials/python.py`; even
these may not replace `sys.modules[__name__]`.

## Cross-version checkpoint evidence

**32 / 32 cases passed, with 1,333 array comparisons and 206 output files.**
`checkpoint-evidence.json` contains each input, physics fingerprint, prefix and
uninterrupted-reference SHA256 inventories, resumed SHA256 inventory, and
comparison counts. The generation/comparison script is
`checkpoint_compatibility.py`; its SHA256 is in the evidence JSON. Raw NPZ/log
artifacts are retained outside Git at
`/data1/lyl/PyRPMD/t48-evidence/accepted-checkpoints`.

The initial references were generated before implementation changes. A second
fixed `801953d` worktree supplied the additional restart-boundary verification
described below. The script observes returns from the real atomic writer with
`sys.setprofile`, copies the durable checkpoint, and lets the reference
trajectory continue without interruption. It does not replace any runtime
function or module. Schema-1 OPES uses the existing worker's
`initialization_schema=1` task flag before the first MD step, retaining its
historical zero empty bias. All inputs pass through `JobSpec.resolve`; ordinary
runs use the real FES dispatcher.

| Coverage | Cases |
| --- | ---: |
| Independent standard / well-tempered MetaD, Andersen / PILE-L | 4 |
| Synchronous shared standard / well-tempered MetaD, Andersen / PILE-L | 4 |
| Independent OPES metad / explore, Andersen / PILE-L | 4 |
| OPES schema 1 / 2, fixed metad plus adaptive metad / explore | 6 |
| Async-files MetaD / OPES, Andersen / PILE-L | 4 |
| Real two-rank mpi-sync MetaD / OPES, Andersen / PILE-L | 4 |
| All original `examples/metadynamics/*.inp` and `examples/opes/*.inp` | 6 |

The six example files use their complete original budgets and input values,
including the grid/fixed slab, block multiple-walker, two-dimensional,
adaptive/explore, and frozen-bias cases. No example was shortened.

Comparison requires identical dtype, shape, and `tobytes()` for all numeric
arrays, including NaN payloads, q, p, clocks, hills/kernels, RNG JSON, operator
state and output arrays. Only absolute output-directory strings in textual
metadata are normalized. NPZ integrity envelopes are checked by the loader and
excluded from content comparison; container metadata is not a scientific array.
Filenames/inventories must match, and CSV / walker log contents are compared.
Async runs use an identical reproducible publication order: walker 0 completes,
then walker 1 runs and is snapshotted/resumed. This does not claim deterministic
agreement between different asynchronous schedules.

### Pre-existing distributed clock limitation

A restart at step 10 with `dt = 0.001 fs` fails the stronger arbitrary-boundary
claim **in both the original baseline and T48**. Distributed readers reconstruct
`State.time` using `elapsed * dt`, whereas uninterrupted propagation accumulates
`time += dt`. At total step 26 the observed MPI times are:

| Path | Exact hexadecimal time |
| --- | --- |
| Baseline uninterrupted | `0x1.132b0de79a749p+0` |
| Baseline resumed at step 10 | `0x1.132b0de79a748p+0` |
| T48 resumed at step 10 | `0x1.132b0de79a748p+0` |

q, p and RNG agree for this MPI counterexample, but timestamps do not. Async
MetaD also rejects the inconsistent hill-clock order with
`shared metadynamics hills must increase by (step, walker) with consistent times`.
`clock-limitation.json` retains the compared clock bits and state evidence.
The diagnostic tracebacks are in the external evidence directory as
`baseline-async-clock-error.log` and `baseline-mpi-clock-error.log`.

The 32 passing cases use step 8 for the eight distributed runs, at the same
step size, after three deposition events per walker. At that boundary the saved
and reconstructed clocks agree exactly. Independent runs retain step 10 and
examples use their original checkpoint schedules. This limitation is reported,
not fixed by this pure refactor; a separate numerical-contract fix is needed
before promising bitwise restart at arbitrary distributed boundaries.

## Reproduction

Use the same environment for both checkouts. `environment.json` records the
actual versions, CPU, thread limits and compiler. The run used Python 3.11,
NumPy 1.26.4, SciPy 1.17.1, pytest 9.2.0 and MPICH 4.1.1 / mpi4py 4.1.2 on the
local Intel Xeon Platinum 8176. Native extensions were built with the ordinary
OpenMP build; no native sources were modified. No GPU or performance claim is
made. Scientific convergence is not assessed by these small regression cases.

```bash
python -m venv /tmp/pyrpmd-t48-env
. /tmp/pyrpmd-t48-env/bin/activate
python -m pip install -e '.[test,docs,mpi]'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Run from a clean baseline checkout after building its unchanged extensions.
export PYTHONPATH=/path/to/801953d/src
python /path/to/T48/docs/validation/compat-shim-removal-2026-11-04/checkpoint_compatibility.py generate \
  --root /tmp/t48-replay --examples /path/to/801953d/examples
mpiexec -n 2 python /path/to/T48/docs/validation/compat-shim-removal-2026-11-04/checkpoint_compatibility.py generate \
  --root /tmp/t48-replay --examples /path/to/801953d/examples --mpi
export PYTHONPATH=/path/to/T48/src
python /path/to/T48/docs/validation/compat-shim-removal-2026-11-04/checkpoint_compatibility.py resume \
  --root /tmp/t48-replay --examples /path/to/801953d/examples
mpiexec -n 2 python /path/to/T48/docs/validation/compat-shim-removal-2026-11-04/checkpoint_compatibility.py resume \
  --root /tmp/t48-replay --examples /path/to/801953d/examples --mpi
```

Use a filesystem supporting advisory locks for async-files. The workspace NFS
returned `ENOLCK`; the accepted executions used local `/tmp` storage and the
ordinary, unmodified locking implementation. To reproduce the known clock
limitation separately, add `--case metadynamics-async-files-andersen
--checkpoint-steps 10` (or the corresponding `mpi-sync` case with two ranks) to
both commands and choose a fresh output root. Both baseline and T48 fail that
comparison. Keep these diagnostic failures separate from the passing cases.

## Test gate

The original system Python had pytest 7.4 and NumPy 1.24 (below project
requirements), an incomplete PySCF namespace, and sandbox socket restrictions.
Its test run is excluded from acceptance. The shared source checkout was also
being edited by another lane. All acceptance runs use isolated source worktrees,
the dedicated environment, local test temporary files, and real local socket /
MPI communication. Missing optional packages are skipped by the existing tests.

Commands and unedited summary lines are recorded in `test-results.txt`.
