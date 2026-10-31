---
orphan: true
---
# Native PIMD through the standard workflow

`task="pimd"` now accepts `sampling.engine="native"`. This explicitly selects
`NativeEnsemble`; it is not a silent rerouting of historical inputs. Old PIMD
inputs retain their existing integrator, RNG and checkpoint behavior. Selecting
this engine changes the scientific checkpoint identity.

```bash
pyrpmd validate examples/ensembles/pimd_native.inp --check-pes
pyrpmd run examples/ensembles/pimd_native.inp --output runs/pimd-native
pyrpmd resume runs/pimd-native
pyrpmd run examples/ensembles/pimd_native_npt.inp --output runs/pimd-native-npt
```

Both examples are numerical controls (harmonic oscillators and an ideal gas),
not validated material simulations. Their short durations do not establish
sampling convergence.

## Controls

```toml
[dynamics]
integrator = "cayley-bcb"
loop_backend = "native"
timestep = "0.1 fs"

[sampling]
engine = "native"
thermostat = "pile-g"
thermostat_params = {tau = "100 fs"}
equilibration = "1 ps"
production = "10 ps"
interval = "1 fs"

[output]
checkpoint_interval = "1 ps"
```

Supported sampling thermostats are Andersen, Langevin, PILE-L/G, SVR,
Cartesian/normal-mode GLE, normal-mode GLE with centroid SVR, parameterized
PIGLET, and FFL-rescale. `none` and `trpmd` are rejected in this equilibrium
workflow. They remain available in the separate native dynamics API.

`thermostat_params` uses the controls documented in
[native ensembles](native-ensembles.md), including explicit `gamma`, `tau`,
`pile_scale`, `interval` and `gle`. Frequencies and times accept explicit units;
GLE drift matrices accept `A_unit`. For PILE-L, top-level
`sampling.gamma_centroid` retains its unit-aware convention and defaults to
`0.01 fs^-1` when neither location specifies it. Do not specify it in both
locations. Andersen uses its explicit `interval` or `sampling.andersen_interval`.
The native Andersen kernel uses the middle thermostat splitting, not the
historical workflow's boundary-refresh sequence; trajectories are not expected
to reproduce historical RNG streams or finite-step samples bitwise.

For NPT add `[barostat]` with `mode="bzp"`, `"mttk"`, `"rgb"` or `"scr"`,
using explicit pressure/time/friction settings. Pressure accepts `bar`, `GPa`
and `hartree/bohr^3`. NPT requires 3D PBC and all atoms mobile; SCR requires
one bead. Flexible-cell row lattice vectors must be lower triangular with
positive diagonal. RGB requires a symmetric external `stress` and a
`reference_cell`; `reference_cell_unit` defaults to bohr for resolved systems.
PIGLET/NPT and Suzuki-Chin NPT are explicitly rejected. Frozen atoms work in
native NVT, not NPT. No calibrated PIGLET parameter set is bundled.

The existing Python BZP route remains available with the default/`legacy`
sampling engine and its `method`, `time_constant`, and `friction` controls.
The native route requires an explicit barostat `mode` and uses `tau`/`gamma`.
Controls for the two engines cannot be mixed; selecting native PIMD does not
reinterpret an existing Python BZP table.

## Potentials and parallel execution

Exact built-in mappings exist for free particles, scalar zero-centered
harmonic potentials, and the periodic cosine model with unit repeats, all
atoms, and no confinement. Unsupported parameterizations raise an error rather
than changing the potential. Existing compiled molecular C ABI/Fortran bindings
are supported in nonperiodic NVT.

A compiled periodic evaluator can be selected through the standard potential
factory:

```toml
[potential]
provider = "native_periodic"
library = "./libmy_pes.so"
function = "evaluate"
```

The signature is `native/include/pyrpmd/thermal_pes.h`. It must fill physical
energy, forces and configurational virial for every bead, and honor the current
row-vector cell supplied on each call. The Python `evaluate()` method is only a
preflight/reference adapter. The trajectory invokes the compiled function
pointer directly without reacquiring the GIL. There is no Python/ASE/Torch/socket
fallback, and opaque mutable contexts are not configured through this input.

Independent replicas use the existing serial/process/MPI task executor. Each
trajectory must use `mode="serial"`, one bead worker and one thread/rank. This
change does not implement native bead-MPI, GPU or socket force dispatch.

## Checkpoint and observation cost

One C++ call advances to the next checkpoint or equilibration boundary.
Observations are computed and streamed in C++; a save interval of one step
still does not cause one Python call per step. Increasing the checkpoint
interval reduces boundary overhead but increases the amount of work rolled
back after failure. `dynamics.block_steps` does not override this transactional
checkpoint cadence.

Each replica writes `samples/native_pimd_NNNN/journal.json` and immutable
transaction directories containing a state checkpoint and optional `.npy`
records. A checksummed, atomically replaced journal is the sole commit point.
Restart validates file hashes and sample clocks before loading the last accepted
state. Uncommitted directories left by a process crash are ignored. Completed
resume reconstructs summaries without loading/evaluating the potential.
Only one run/resume process should own a replica output directory at a time.
Historical checkpoint files are retained for audit; disk usage still grows
with the number of checkpoints and recorded frames.

The production sample clock is `equilibration_steps + k*save_interval`, for
`k>=1`. Unlike the legacy materials format, the equilibration endpoint itself
is not an additional production frame. Chunk sizes can change on restart without
changing saved trajectories or RNG order. Native and legacy checkpoints are
not interchangeable.

Each record contains the 18 scalar columns from `NativeEnsemble.run()`, followed
by centroid positions `[N,3]`, centroid velocities `[N,3]`, and squared radii of
gyration `[N]`, all in atomic units. Read a chunk with
`numpy.load(path, mmap_mode="r", allow_pickle=False)`. For NPT the cell and volume
are saved at each sample, not assumed to equal the starting cell.
Propagation uses O(P*N) state memory, independent of trajectory length.
Offline correlation diagnostics still retain scalar time series, but do not
assemble full coordinate histories or transfer them through MPI.

`H_P` is the particle ring-polymer Hamiltonian, **not** the NPT conserved
quantity: piston energy and thermostat heat are not included. Statistical and
real-system convergence remain separate from implementation tests.

## Normal-mode fusion

For fixed-cell normal-mode baths, both free-ring half maps and the thermostat
now share one normal-mode visit: four transforms instead of ten per step.
The real orthonormal basis, Cayley square root/exact half rotations and random
draw order are unchanged. Removing transform roundoff produces tiny numerical
differences; equivalence to the retained unfused path is tolerance-based, not
bitwise. Same-path chunk/restart remains bitwise identical.

The Python API accepts `fuse_modes=False` for diagnostics. This setting is
fingerprinted; checkpoints from different rounding paths are not silently mixed.
See `benchmarks/efficiency/native_mode_fusion.py` for a reproducible microbenchmark.
The measured gain on a cheap harmonic PES is not a prediction of speedup for
an expensive electronic-structure or machine-learning potential.

## Scope remaining

This follow-up integrates PIMD only. Diffusion, rate, FES and metadynamics retain
their previous workflow dispatch. Surface/frozen-atom NPT, calibrated PIGLET
datasets, Suzuki-Chin action, native socket/GPU/bead-MPI backends and real-material
accuracy validation remain outside this change.
