---
orphan: true
---
# PIMD engine convergence and checkpoint compatibility

The standard `pimd` and fixed-cell `diffusion` workflows now select their
execution contract in one place: `pyrpmd.workflows.pimd.select_engine`.
Validation, execution, diffusion reanalysis and numerical preparation use this
same selection. Native checkpoints also use the atomic storage implementation
already used by the legacy and Torch workflows. An offline tool inspects all
three checkpoint families and converts legacy trajectory storage in either
direction.

This is a compatibility-preserving convergence of workflow ownership and
storage. The historical NumPy, C++ and Torch numerical/RNG contracts remain
distinct execution backends. Replacing one stochastic map with another would
not preserve an existing trajectory, even when both sample the same equilibrium
distribution.

## Stable input and execution contracts

| Input | Selected backend | Compatibility contract |
| --- | --- | --- |
| `sampling.engine` omitted | `legacy` | The key remains absent; historical defaults, fingerprint, NumPy RNG, sample clock and checkpoint payload remain unchanged. |
| `sampling.engine="legacy"` | `legacy` | The explicit selector retains its existing fingerprint, which can differ from the omitted-selector fingerprint. |
| `sampling.engine="native"` | `native` | Retains `NativeEnsemble`, its C++ RNG and thermostat/barostat splitting, and the native journal commit protocol. |
| `sampling.engine="torch"` | `torch` | Retains the explicit tensor device, float64 state, two owned generators, batch identity and atomic batch checkpoint. |

There is no `auto` engine selector and no fallback from an explicitly selected
backend. Unknown selectors, reused routes whose configuration now selects a
different backend, and unsupported combinations fail explicitly. Direct
`run_resolved` calls perform materials preflight before creating an output
directory. Selecting a route alone does not import Torch or load the native
thermal extension.

The existing `materials.resolve_materials`, `materials.run_materials` and
`materials.reanalyze_diffusion` entry points remain available as dispatch
wrappers. The backend adapters retain their different transaction and recording
rules. In particular:

- Legacy sampling records the production boundary frame; native journal
  sampling retains its historical first recorded production step.
- Legacy and native Andersen sequences, NumPy and native bath noise, and the
  supported pressure-control splittings are not interchangeable restart data.
- Native PIMD still rejects diffusion. Torch still rejects unsupported devices,
  constraints, pressure control and external trajectory executors.
- Optional libraries are required only by the execution paths that use them.

The existing `PropagationPlan`/`Dynamics` composition is still the owner of the
mathematical operators. This change does not introduce another propagator or
another set of CV, force, thermostat or barostat equations.

## Read a checkpoint without starting its engine

```bash
python -m pyrpmd.pimd_checkpoint inspect RUN/checkpoints/materials_0000.npz
python -m pyrpmd.pimd_checkpoint inspect PATH/TO/NATIVE/state.npz
python -m pyrpmd.pimd_checkpoint inspect RUN/checkpoints/materials_torch.npz
```

The Python interface is:

```python
from pyrpmd.pimd_checkpoint import inspect_pimd_checkpoint

report = inspect_pimd_checkpoint("RUN/checkpoints/materials_0000.npz")
print(report["engine"], report["step"], report["container_integrity"])
```

Inspection checks the local NPZ members, the recognized state layout and clock
fields, the legacy PCG64 state, and the native state digest where applicable.
It reports scientific fingerprints, array shapes/dtypes and the RNG family;
it does not print coordinates or serialized generator state. It constructs no
PES and does not initialize a native or Torch execution runtime.

The report deliberately distinguishes storage integrity from resumability:

| Field | Meaning |
| --- | --- |
| `container_integrity="verified"` | Every member was checked against the common integrity envelope. |
| `container_integrity="unverified_legacy"` | The file predates that envelope; it is not silently described as verified. |
| `state_digest_verified=true` | The native payload's original digest was checked, independently of the outer container. |
| `resume_validated=false` | A single-file inspection has not authenticated the full run, model provenance or runtime compatibility. |
| `external_trajectory_verified=false` | A single-file inspection has not walked the referenced trajectory chunks or native journal. |

An old native file can therefore have an unverified outer container and a
verified native state digest. Normal workflow resume continues to perform its
full identity, runtime and committed-history checks.

## Migrate legacy trajectory storage without propagation

The offline converter supports both complete runs and interrupted runs,
including checkpoints with no saved production frames. Stop the simulation
before migration: the checkpoint protocol assumes one writer per replica.

```bash
python -m pyrpmd.pimd_checkpoint migrate RUN --storage stream --replica 0
python -m pyrpmd.pimd_checkpoint migrate RUN --storage memory --replica 0
```

The command authenticates the saved run manifest, input/system identity and
referenced model files. It then validates the checkpoint and every committed
trajectory frame before atomically replacing that replica's checkpoint.
Memory-to-stream conversion writes immutable chunks first; the checkpoint
replacement is the commit point. Failed publication leaves the previous
checkpoint usable, and unreferenced chunks never advance the accepted history.

The conversion preserves the dtype, shape and raw bytes of every non-trajectory
payload array. This includes `q`, `p`, clocks, fingerprints, formatted RNG JSON,
barostat state and any opaque auxiliary arrays. A memory-to-stream-to-memory
round trip also restores every trajectory payload array exactly. The NPZ ZIP
file itself can have different bytes; same-storage no-ops leave the entire
file unchanged.

```python
from pyrpmd.pimd_checkpoint import migrate_legacy_checkpoint

# config and system are the resolved inputs used for this run.
saved = migrate_legacy_checkpoint(
    config, system, "RUN", replica=0, storage="stream",
)
assert saved["storage"] == "stream"
```

Saved input and manifest files are never rewritten. The command reports
`configured_resume_storage`: ordinary `pyrpmd resume RUN` still follows that
original preference and can convert the storage back. To continue with a new
storage preference, pass an explicit execution configuration to the existing
resolved runner:

```python
import json
from pathlib import Path

from pyrpmd.runner import run_resolved
from pyrpmd.system import System

directory = Path("RUN")
config = json.loads((directory / "input/resolved.json").read_text())
system = System.from_dict(
    json.loads((directory / "input/system.json").read_text())
)
config.setdefault("output", {})["trajectory_storage"] = "stream"
run_resolved(config, system, output=directory, resume=True)
```

The runner still authenticates the original scientific identity. Trajectory
storage is an execution choice, not a change to the sampled Hamiltonian or RNG.

### Migration limits

- This converts legacy **storage**, not NumPy, C++ or Torch numerical state
  into another engine's restart format. Cross-engine exact continuation is
  unsupported and is never inferred from matching coordinate shapes.
- Truly unenveloped legacy materials records can be inspected, but migration
  requires the integrity envelope already required by the materials resume
  reader. The utility does not manufacture provenance for unverified data.
- Stream-to-memory conversion materializes the complete committed history.
  Memory-to-stream conversion necessarily reads the existing all-in-one file.
- The current implementation reads the source checkpoint again through the
  established workflow reader, preserving opaque fields while reusing all
  identity/history validation. This is an offline I/O cost.
- Source chunks remain on disk. The utility does not delete history or
  coordinate with a concurrently running simulation.

## Common atomic native checkpoint storage

`NativeEnsemble.save` now calls `atomic_savez_uncompressed`; both common writers
share the same no-pickle validation, owned payload, SHA256 envelope, temporary
file, `fsync`, atomic replace and POSIX directory `fsync` implementation.
The existing `atomic_savez` API continues to use compression.

Native files keep the `pyrpmd.native-ensemble.v1` payload, original array names,
sorted metadata JSON, state digest, fingerprint and serialized RNG. A new outer
envelope adds storage integrity without changing those scientific fields.
`NativeEnsemble.load` accepts both new containers and untouched old native NPZ
files. An old native workflow journal remains valid: its referenced old state
file is read without rewriting it, and only subsequent committed blocks use
the new writer. Newly saved native containers need the new reader; older readers
that require the exact bare-NPZ member inventory do not understand the added
envelope.

Native array data remain uncompressed. Compression of high-entropy float64
state saves little space while adding substantial CPU cost. The reproducible
storage benchmark is:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
  examples/ensembles/checkpoint_storage_benchmark.py \
  --beads 32 --atoms 128 4096 --repeats 7 --warmup 1 --seed 1 \
  --output checkpoint-storage.json
```

It compares the old native writer, the common uncompressed writer and the common
compressed writer using synthetic arrays. It records every timing, file size,
platform and thread setting, and verifies the payloads after writing. The new
integrity copy/hash and directory sync have a cost; this is not a claim of
checkpoint or propagation speedup. The numerical propagation loop is unchanged.

The recorded Linux x86_64/Python 3.12.14/NumPy 2.3.5 run exposed 9 logical CPUs
(CPU model unavailable), with `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`.
Two regression suites were running concurrently, so these are illustrative I/O
costs, not isolated throughput measurements. Medians of seven interleaved writes
after one warmup were:

| Beads | Atoms | Old native, uncompressed | Common, uncompressed | Common, compressed |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 128 | 2.104 ms | 6.853 ms | 31.782 ms |
| 32 | 4096 | 24.074 ms | 51.083 ms | 915.194 ms |

For 4096 atoms, the respective files were 9,445,566, 9,450,842 and 9,071,886
bytes. The new uncompressed writer pays for additional integrity verification;
compression would save about 4% of the new file while substantially increasing
the measured write cost. The complete timing samples and environment are in
`docs/validation/t16b-pimd/checkpoint-storage.json`.

## Validation and remaining convergence route

The regression tests cover literal pre-change scientific fingerprints,
pre-envelope native restart, PILE-L/Andersen/NHC/GLE/BZP continuation,
complete/partial/zero-frame legacy round trips, NVT/NPT RNG and piston state,
publication failure, damaged state/history, optional-runtime-free inspection,
and public CLI migration followed by resume. They compare exact state and
trajectory arrays with independently executed uninterrupted references.

The repository also includes a cross-version harness under
`docs/validation/t16b-pimd`: one checkout creates the real old checkpoint and an
uninterrupted reference, and the changed checkout resumes that checkpoint.
Local full-suite, documentation-build and storage-benchmark evidence is recorded
alongside the harness. These are compatibility/engineering checks; they do not
certify physical convergence.

The untouched main baseline was `ceee59cd5f702165b56c5e541a503d8810fb9111`.
Both complete suites used the same Python 3.12.14 / NumPy 2.3.5 / Torch
2.15.0+cpu environment, with OpenMP and OpenBLAS limited to one thread:

| Suite | Passed | Failed | Skipped | Warnings | Passed subtests |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untouched main | 11,469 | 1 | 356 | 68 | 493 |
| T16b implementation | 11,530 | 1 | 356 | 68 | 493 |

All 61 added test cases pass. The failure set is identical. The raw summaries
are `1 failed, 11469 passed, 356 skipped, 68 warnings, 493 subtests passed in
2791.17s (0:46:31)` and `1 failed, 11530 passed, 356 skipped, 68 warnings, 493
subtests passed in 2853.64s (0:47:33)`. The suites ran concurrently; these wall
times are not a performance comparison.

All seven actual old-checkpoint resumes match uninterrupted baseline payloads
by dtype, shape and bytes: legacy PILE/Andersen NVT and BZP NPT, native PILE/NHC
NVT and BZP NPT, and a two-replica Torch CPU PILE NVT run. The three legacy cases
also complete a memory-to-stream-to-memory round trip before resuming.

The exact commands, environment, source hashes, raw pytest output, strict
Sphinx build output and source-package inventory are in
`docs/validation/t16b-pimd/validation.json` and its referenced files.

The cloud executor used for this validation denies creation of `AF_UNIX`
sockets. The unchanged baseline independently reproduces
`tests/test_ipi_clients.py::test_existing_unix_address_is_not_removed_on_bind_failure`
with `PermissionError: [Errno 1] Operation not permitted` at `socket.socket`,
before binding a path. The validation records retain that environment failure;
this change does not modify the socket code or add a skip for it.

The implemented route is:

1. **One workflow selection point:** all supported standard materials operations
   use the same explicit engine registry.
2. **One atomic NPZ lifecycle:** legacy, native and Torch checkpoint writers
   share the storage implementation while retaining their payload contracts.
3. **Read-only legacy access and reversible storage migration:** old runs can be
   audited and moved to streamed recording without resampling or changing RNG.
4. **Historical numerical adapters remain:** their noise order, sample clocks,
   pressure maps and supported combinations are compatibility obligations.
   Future operator work should extend the common propagation plan and its
   backends. Removing a historical adapter requires either a demonstrated
   bitwise-compatible implementation of every supported old map/state or an
   explicitly versioned new numerical contract with continued old-format access.

This release completes the first three steps. It does not delete the historical
NumPy propagation path or claim one universal numerical kernel.
