# Standard trajectory export

Export a saved PIMD or diffusion replica to **extended XYZ**, **H5MD**, or
**ASE `.traj`** without rerunning its potential:

```bash
pyrpmd export runs/water water.extxyz
pyrpmd export runs/water water.h5md --replica 0
pyrpmd export runs/water water.traj
```

The source may also be a `samples/materials_0000/manifest.json` or a legacy
`samples/materials_0000.npz` archive. The suffix selects the format;
`--format extxyz|h5md|traj` overrides it. Install `PyRPMD[ase]` for `.traj`.
Extended XYZ needs no optional dependency; H5MD uses the core h5py dependency.

## What is exported

Materials trajectories store **unwrapped ring-polymer centroids** and centroid
velocities. These exports retain that representation explicitly. They do not
reconstruct individual bead coordinates, wrap beads before averaging, infer
bonds or reinterpret imaginary-time bead indices as physical time.

| Property | Extended XYZ | H5MD | ASE `.traj` |
| --- | --- | --- | --- |
| Position/cell | Å | Å | Å |
| Time | `Time` in fs, with `time_unit=fs` | `position/time` in ps | `atoms.info["time_fs"]` |
| Velocity | `velocities` column in Å/fs | `velocity/value` in Å/ps | ASE momenta/velocity units |
| Step | `Step` | `position/step`, int64 | `atoms.info["step"]` |
| Isotopes | `masses` in u plus `pyrpmd_symbol` | `mass` in u plus `pyrpmd_symbols` | Explicit masses and original labels in `info` |
| Cell/PBC | `Lattice` and three `pbc` flags | Per-frame `box/edges`, boundary attributes | Per-frame cell and PBC |

D and T are represented as chemical H with their actual masses, preserving
compatibility with element-based readers. The original label is also retained.
All formats preserve nonperiodic, partially periodic and fully periodic cells.
Per-frame cells are retained for existing variable-cell PIMD trajectories.

Production time is `(sample_step - equilibration_steps) * production_timestep`.
The integration step includes the equilibration prefix; production time starts
at zero. PIMD sampling times remain distinct from physical RPMD production
times. Mean bead energies are intentionally not written as the energy of a
centroid configuration: those are different observables.

## Open the results

### OVITO and VMD

Open `.extxyz` in OVITO. It recognizes the cell and per-atom fields described
by `Properties`; see the [OVITO extended XYZ reader](https://www.ovito.org/manual/reference/file_formats/input/xyz.html).

In VMD, load the same text file with file type **XYZ**:

```tcl
mol new water.extxyz type xyz waitfor all
```

The ordinary XYZ-compatible element/coordinate columns provide the basic
trajectory. Do not assume that VMD's XYZ reader imports extended cell or
velocity metadata; use the H5MD plugin where installed, or set cell/topology
information separately. See the [VMD XYZ plugin](https://www.ks.uiuc.edu/Research/vmd/plugins/molfile/xyzplugin.html).
No GUI-level VMD/OVITO validation is claimed by the automated tests.

### ASE

```python
from ase.io import read
from ase import units

frames = read("water.traj", index=":")
velocity_angstrom_per_fs = frames[0].get_velocities() * units.fs

xyz_frames = read("water.extxyz", index=":", format="extxyz")
xyz_velocity_angstrom_per_fs = xyz_frames[0].arrays["velocities"]
```

Extended XYZ cells follow the [ASE format convention](https://ase.gitlab.io/ase/ase/io/formatoptions.html#extxyz):
three row lattice vectors are serialized consecutively. `.traj` uses ASE's
official trajectory writer and its mass/momentum unit conventions.

### MDAnalysis

```python
import MDAnalysis as mda

u = mda.Universe("water.extxyz", "water.h5md", topology_format="XYZ")
for ts in u.trajectory:
    positions_angstrom = ts.positions
    velocities_angstrom_per_ps = ts.velocities
    time_ps = ts.time
```

H5MD stores no guessed bond topology. Use an appropriate topology file when
your analysis requires bonds, residues, atom types or charges. XYZ topology
also does not preserve isotope masses automatically; set masses explicitly
from your model if a mass-weighted analysis requires them.

The [MDAnalysis H5MD reader](https://docs.mdanalysis.org/stable/documentation_pages/coordinates/H5MD.html)
supports all-periodic or all-nonperiodic boxes, not partial PBC. Partial flags
are preserved in the H5MD file, but that reader will reject them. ASE/extxyz is
a suitable alternative for a slab or wire. H5MD units follow the
[H5MD specification](https://www.nongnu.org/h5md/h5md.html); synchronized position,
velocity and box clocks are HDF5 hard links.

## Integrity and memory behavior

Stream exports follow the **selected manifest's committed prefix**. Every
chunk's complete-file hash, array checksums, schema and linkage are checked.
Orphan chunks from interrupted runs are ignored. The writer holds one frame
and one source chunk at a time; chronological chunk indexing uses temporary
disk space rather than a trajectory-sized in-memory list. A legacy single-NPZ
archive still requires loading its coordinate/velocity arrays.

The manifest is an analysis export, not the authoritative restart checkpoint.
Run `pyrpmd analyze RUN --recompute` after a partial/interrupted run if you need
to regenerate its trajectory export from saved checkpoints. If both legacy
and stream exports exist after a storage migration, select the intended source
file explicitly instead of relying on automatic selection.

New output is written to a temporary file in the destination directory and
published atomically after the entire source passes validation. Existing files
are refused unless `--overwrite` is explicit. A late invalid frame, bad clock,
nonconvergent reader or missing chunk leaves the old export unchanged. Exports
are portable analysis files; use the original PyRPMD checkpoints and chunks
for exact restart.

## Export other workflows through Python

The common writer accepts any iterable of `TrajectoryFrame` objects. Supply
one configuration per frame in atomic units and a validated `System`:

```python
from pyrpmd import frames_from_states, write_trajectory

# states is an iterable of State objects observed during your simulation.
write_trajectory("centroids.h5md", frames_from_states(states, system),
                 system, representation="centroid")

# With a separate iterable, export an actual bead rather than a centroid.
write_trajectory("bead0.extxyz", frames_from_states(bead_states, system, bead=0),
                 system, representation="bead")
```

For custom frame data, import `TrajectoryFrame(positions, step, time,
velocities=None, cell=None)`. Times and steps must increase strictly. Every
frame must have the same atom order and consistent velocity/cell availability.
Use one output per replica or bead so independent trajectories are never
mistaken for consecutive physical frames.

Roundtrip tests read actual files with ASE and MDAnalysis, including velocities,
isotopes, triclinic/variable cells, production time, unwrapped centroids, source
corruption, orphan chunks and atomic failure behavior.
