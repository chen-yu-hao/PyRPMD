# Checkpoint and resume contract

PyRPMD 0.2.0 separates three versioned formats. The user input schema remains
version **1**; the public run manifest is version **2**; the portable NPZ
checkpoint container is version **2**. Individual workflow payload identifiers
remain unchanged where their numerical meaning has not changed.

## Atomic NPZ container

`pyrpmd.checkpoint.atomic_savez(path, kind="...", **arrays)` owns a copy of the
payload, writes a temporary file in the destination directory, flushes and
fsyncs it, then commits it with `os.replace`. On POSIX it also fsyncs the
directory. An interruption before the replace preserves the previous complete
file. Guarantees still depend on the filesystem honoring atomic rename/fsync;
the implementation does not turn a remote filesystem into a transactional
database.

The reserved scalar UTF-8/Unicode JSON entry `_pyrpmd_checkpoint` records:

| Field | Contract |
| --- | --- |
| `schema` | Exactly `pyrpmd.checkpoint` |
| `schema_version` | Integer `2` |
| `kind` | Payload type identifier supplied by the workflow |
| `writer_version` | PyRPMD package version that wrote the container |
| `arrays` | Complete payload inventory: each array's dtype, shape and SHA256 over C-order bytes |

All payloads use `allow_pickle=False`. Object arrays are rejected on both write
and read. `load_checkpoint()` verifies the entire inventory and every array
before exposing any state to the caller. The reserved envelope is hidden from
the normal payload view. Corrupt, truncated, unsupported-schema or mismatched
payloads raise `CheckpointError`; callers do not continue from a partly trusted
q/p/RNG state. SHA256 is an accidental-corruption check, not a digital signature
or proof of the origin of an untrusted file.

```python
from pyrpmd.checkpoint import atomic_savez, load_checkpoint, inspect_checkpoint

atomic_savez("state.npz", kind="my.state.v1", q=q, p=p, step=step)
with load_checkpoint("state.npz", expected_kind="my.state.v1",
                     require_integrity=True) as data:
    q = data["q"].copy()
print(inspect_checkpoint("state.npz"))
```

The CLI validates the same envelope and reports array shapes/dtypes without
printing coordinates, momenta or RNG values:

```bash
pyrpmd checkpoint runs/meta/checkpoints/metadynamics_0000.npz --require-integrity
```

## Legacy files

Unenveloped v0.1 NPZ files remain readable with `allow_pickle=False` and ZIP CRC
validation. They report `integrity="unverified_legacy"`. Use
`require_integrity=True` to reject them. Recognized version-1 legacy payloads
keep their original interpretation; unknown numbered payload versions are
rejected. Legacy payloads are upgraded when the owning workflow next writes
them; merely inspecting an old file does not rewrite it or assert that its
historical contents were checksummed.

## Public run identity

`manifest.json` has `schema="pyrpmd.run"`, `schema_version=2`, input/checkpoint
schema versions, package/environment/native capabilities, referenced file
hashes and a scientific configuration fingerprint. Scientific identity includes
the system, isotopes, temperature, bead count, timestep, integrator, potential,
CV/bias specification and workflow sampling settings.

Output location, worker layout and the execution-only
`dynamics.loop_backend`/`dynamics.block_steps` fields are excluded. Therefore:

```bash
pyrpmd resume runs/example --workers 4 --loop-backend auto --block-steps 128
```

does not request a different Hamiltonian or integrator. Existing workflow
restrictions still apply: dRPMD parent allocation must have been explicitly
fixed if changing worker counts would otherwise change it; strict native mode
requires every requested feature; metadynamics cannot use a frozen native
history. Backend changes can produce floating-point differences and are not a
promise of bitwise equality across architectures.

An explicit integrator change is a new scientific run. The resume API accepts
only block/backend execution overrides and does not expose a way to force an
incompatible state into a changed calculation. Changing referenced input/PES
file contents is also rejected. The file list tracks explicit dependencies
and plugin sources; an environment lock and separately pinned external servers
are needed to capture indirect Python dependencies and remote PES state.

A v1 run manifest is migrated only after its saved input/system reproduce its
old fingerprint and its referenced files still match. The old fingerprint is
retained as `legacy_physics_fingerprint`. Unknown manifest versions are errors.
Migration changes the manifest identity rule; it does not fabricate an integrity
envelope for unchanged legacy trajectory files.

## Workflow state and failure boundaries

| Workflow | Authoritative restart state |
| --- | --- |
| dRPMD | Continuous parent-chain q/p/RNG and emission clock; launch records; accepted child prefix, classifier/history state and completed results |
| Fixed umbrella / rate PMF | Per-window q/p/RNG, elapsed clock and observations; completed samples support reanalysis without propagation |
| Rate recrossing | Constrained parent state and snapshots; paired child state/results, selected surface and PMF identity |
| Adaptive umbrella | Checksummed `adaptive-plan.json`, committed pilot round and its checkpoints; frozen production uses separate seeds and ordinary window checkpoints |
| Metadynamics | One NPZ commits q/p/RNG, deposition and sample clocks, hills, observations and propagation counters; `HILLS.csv` is a regenerable view |
| Reaction network | Stable logical path identities and dependency hashes; each path's standard rate checkpoints plus atomically saved completion/progress records |

The two native propagation interfaces retain different failure semantics:
`Dynamics.propagate()` commits the whole requested block only on success;
`Dynamics.propagate_block()` commits each accepted step and reports the valid
prefix if a later step fails. Workflow event boundaries preserve thermostat,
observation, hill-deposition and checkpoint ordering. See the
[block API contract](block-propagation.md) for detailed exception fields.

JSON manifests and derived CSV/HDF5 result views are separate from the NPZ
container. Adaptive decisions have their own content checksum; network path
records use scientific/dependency fingerprints. This is not a claim that every
file in an output directory is covered by one global cryptographic archive.
