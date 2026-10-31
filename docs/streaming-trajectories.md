# Streaming materials trajectories

Materials runs can write saved frames in bounded chunks while keeping a small
atomic checkpoint for the current ring-polymer state, RNG state and committed
trajectory prefix. This removes the need to retain the full coordinate and
velocity histories in each propagation worker.

```toml
[output]
trajectory_storage = "stream"
chunk_frames = 256
checkpoint_steps = 1000
```

`trajectory_storage="stream"` is the default for new materials runs. The
`"memory"` option preserves the earlier single-checkpoint trajectory layout for
compatibility. `chunk_frames` limits the number of staged frames per worker;
`checkpoint_steps` controls how often the simulation state and its trajectory
prefix are committed together. Both are storage choices and may change on
restart without changing the Hamiltonian, random stream or sampling clock.

The writer stores centroid positions and velocities, sample steps, physical
energy estimators and radii of gyration. Constant-pressure runs add the sampled
cell, pressure and volume. Atom and bead arrays in the restart state remain
separate from these saved observables.

## Commit and recovery

Each trajectory chunk is an immutable NPZ container with the usual PyRPMD
per-array integrity envelope. Its filename is the SHA256 digest of the complete
file. Metadata records the stream identity, schema digest, frame range and the
digest of the preceding chunk. The simulation checkpoint stores one head
digest, total frame and chunk counts, and the field schema. Its trajectory
descriptor has constant size apart from the digits of these counts.

The workflow commits in this order:

1. Write, flush and synchronize any pending frame chunk.
2. Publish its immutable content-addressed filename and synchronize the directory.
3. Atomically replace the simulation checkpoint containing the state, RNG, phase
   clock and descriptor for that prefix.

A crash before step 3 can leave an extra chunk. Restart follows only the head
in the last valid checkpoint, so extra chunks do not add frames or advance the
simulation. Existing committed chunks are never overwritten. They should be
kept with the checkpoint when moving or archiving a run. Deleting an old chunk
breaks the committed chain, even if the newest chunk remains present.

Resume validates every referenced chunk, the file digests, the array checksums,
the schema and the contiguous frame ranges. Descriptor paths are not accepted;
the only filenames constructed from a descriptor are validated digest names
inside the selected trajectory directory. Symbolic-link chunk files are
rejected. Recovery assumes one writer per replica and a filesystem supporting
hard links, atomic file replacement and synchronization.

Earlier materials checkpoints containing their full arrays remain readable.
When a run is resumed with streaming enabled, those already validated arrays
are copied into bounded chunks, then the new state/descriptor checkpoint is
committed. If that migration is interrupted before checkpoint replacement, the
old checkpoint remains authoritative. Loading a legacy checkpoint necessarily
requires memory for its old embedded history once.

## Memory and analysis

Propagation stages at most `chunk_frames` frames. NPZ compression, checksumming
and validation require temporary copies of a chunk, so the peak is a small
multiple of the chunk payload rather than exactly one buffer. Worker results
can carry only the state and trajectory descriptor, avoiding full histories in
process or MPI result messages.

The low-level iterator can read selected fields one chunk at a time. It still
validates unselected fields. Chronological reading builds a temporary disk
index of chunk digests; reverse chronological reading follows the chain
directly. Both have memory bounded by chunk size. Resume validation reads the
committed history from disk, so restart time grows with saved data even though
restart memory does not.

Offline correlation and diffusion analysis may materialize selected histories.
Streaming propagation does not make those existing analysis algorithms use
constant memory. Use the iterator below for custom block analysis, and budget
analysis memory separately for long runs or many replicas.

## Python storage API

Read an exported materials trajectory without loading its whole history:

```python
import json
from pathlib import Path
from pyrpmd.trajectory import iter_chunks

path = Path("runs/materials-pimd/samples/materials_0000/manifest.json")
manifest = json.loads(path.read_text())
metadata = manifest["metadata"]
for chunk in iter_chunks(path.parent, manifest["trajectory"],
                         fields=["sample_steps", "positions"]):
    time_au = ((chunk["sample_steps"] - metadata["equilibration_steps"])
               * metadata["production_timestep_au"])
    # Analyze or export this bounded block of positions and production times.
```

The manifest is rebuilt from the checkpoint when a completed run is resumed
or reanalyzed. For application-specific frame schemas, use the writer directly:

```python
import json
import numpy as np

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.trajectory import TrajectoryStream, iter_chunks, materialize_stream

schema = {
    "sample_steps": {"dtype": "<i8", "shape": []},
    "positions": {"dtype": "<f8", "shape": [2, 3]},
}
writer = TrajectoryStream("run/frames", schema=schema, chunk_frames=256)
writer.append({"sample_steps": 10, "positions": np.zeros((2, 3))})
descriptor = writer.prepare_commit()
atomic_savez("run/state.npz", kind="example.state.v1",
             trajectory=json.dumps(descriptor), step=10)

with load_checkpoint("run/state.npz", require_integrity=True) as saved:
    descriptor = json.loads(str(saved["trajectory"]))
writer = TrajectoryStream("run/frames", descriptor=descriptor, chunk_frames=512)

for chunk in iter_chunks("run/frames", descriptor, fields=["positions"]):
    consume_positions(chunk["positions"])

# Explicitly allocate a full selected history for offline analysis.
positions = materialize_stream(
    "run/frames", descriptor, fields=["positions"]
)["positions"]
```

`append` owns its input buffers and rejects nonfinite values, mismatched shapes
and lossy integer-clock conversion. A validation or storage failure while
appending leaves that frame unaccepted. The caller must restore its trial
simulation state before committing a checkpoint after such a failure.
`prepare_commit` flushes pending data but does not commit a simulation state by
itself. Use its returned descriptor only with the state and RNG belonging to the
same accepted sample prefix.

The tests in `tests/test_trajectory_stream.py` exercise interrupted publication,
orphan chunks, exact prefix recovery with a changed chunk size, corruption in
older chunks, schema and path checks, legacy-array migration and memory use
while writing a much larger logical history than the in-memory buffer.
