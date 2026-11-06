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
PIGLET, FFL-rescale and NHC (`nhc`). `none` and `trpmd` are rejected in this equilibrium
workflow. They remain available in the separate native dynamics API.

For production PIMD with more than one bead, use `pile-l` or `pile-g`
(Langevin, FFL, Andersen and calibrated GLE/PIGLET are also local baths). `svr`
and `nhc` are accepted but act as **one global bath** on all beads; see
[thermostat choice](#thermostat-choice).

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

For NHC, `sampling.integrator` resolves to `obcbo` with the fixed-cell Cayley
map and `obabo` with normal-mode integration or pressure control. Its parameters
include `tau` (or `tdamp`), `chain_length` and `substeps`. Output
`sampling_splitting` reports the complete map executed by `NativeEnsemble`:
NHC fixed-cell sampling uses OBCBO/OBABO, other fixed-cell baths use
BCOCB/BAOAB, and pressure-controlled trajectories use the OBABO wrapper.
Historical non-NHC native inputs retain their existing particle-integrator
labels in the resolved input.

For NPT add `[barostat]` with `mode="bzp"`, `"mttk"`, `"rgb"` or `"scr"`,
using explicit pressure/time/friction settings. Pressure accepts `bar`, `GPa`
and `hartree/bohr^3`. NPT requires 3D PBC and all atoms mobile; SCR requires
one bead. Flexible-cell row lattice vectors must be lower triangular with
positive diagonal. RGB requires a symmetric external `stress` and a
`reference_cell`; `reference_cell_unit` defaults to bohr for resolved systems.
PIGLET/NPT supports isotropic BZP when the supplied calibrated matrices have a
canonical centroid momentum and zero covariance with its auxiliary variables.
The general native input driver does not infer a Suzuki–Chin action: use
the legacy `[path_integral]` SC workflow for SC-NPT. Frozen atoms work in
native NVT, not NPT. The [condensed-phase benchmark](condensed-phase-nqe.md)
contains an explicitly sourced PIGLET parameter set; it is not an automatic
preset or a guarantee of transferability to another system.

The existing Python BZP route remains available with the default/`legacy`
sampling engine and its `method`, `time_constant`, and `friction` controls.
The native route requires an explicit barostat `mode` and uses `tau`/`gamma`.
Controls for the two engines cannot be mixed; selecting native PIMD does not
reinterpret an existing Python BZP table.

## Thermostat choice

`svr` and `nhc` couple one bath to the total kinetic energy of the ring
polymer. With `beads > 1`, the stiff internal ring modes are nearly harmonic
and exchange energy only weakly. One global bath then cannot thermalize each of
them, so the trajectory samples a non-canonical distribution. A
canonical-ensemble estimator can be biased far beyond its reported error bar,
and the replica scatter does not reveal this. Validation and run summaries
therefore add an explicit warning for these selections when `beads > 1`. This
is a warning, not a rejection: existing inputs, fingerprints and checkpoints are
unchanged, and with `beads = 1` (classical sampling) both thermostats are
unflagged. `pile-g` and `nm-gle-g` are not affected. They apply SVR to the
centroid only and use a local bath for every internal mode.

An analytic control (one particle, three dimensions, `NativeEnsemble`,
`dt = 0.25` a.u., 20 block averages) shows how large the effect is. The
reference is the exact primitive-action value for the same P:

| Model | Thermostat | ⟨V⟩ deviation from exact discrete-P value |
| --- | --- | --- |
| harmonic, βħω = 10, P = 32 | `pile-g`, `andersen`, `langevin` | within 2σ |
| harmonic, βħω = 10, P = 32 | `svr` | −7σ |
| harmonic, βħω = 10, P = 32 | `nhc` | −27σ |
| quartic `x²/2 + x⁴/8`, β = 8, P = 16 | `pile-l`, `pile-g`, `ffl` | within 1σ |
| quartic `x²/2 + x⁴/8`, β = 8, P = 16 | `svr` | −1.6σ, with a 7× wider error bar |
| quartic `x²/2 + x⁴/8`, β = 8, P = 16 | `nhc` | ⟨V⟩ = 0.486 vs 0.734 (−100σ) |

The NHC operator is not at fault. With P = 1 on the same quartic model (8
atoms) it reproduces the classical average (0.6323 ± 0.0030 vs 0.6338). These
are numerical controls, not a parameter study. Longer `tau` or more NHC chain
links do not provide a guarantee either.

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

`pressure_mean_au` retains the mean of the recorded `pressure` column. The
`pressure_estimator` metadata identifies it as an
`instantaneous_centroid_momentum_virial`, using the sampled centroid momenta;
for NPT it is the piston feedback pressure. It is not the pressure with the
kinetic term fixed to the configured temperature. For the all-mobile, fully
periodic case, the latter can be reconstructed from each saved centroid
velocity `vbar_i` and its **same-frame** volume:

$$
p_{\mathrm{CV},T}=p_{\mathrm{recorded}}+
\frac{3N/\beta-\sum_i m_i|\bar v_i|^2}{3V}.
$$

Apply this correction before averaging. The usual equality of ensemble means
follows from canonical conditional centroid kinetic moments.
PIGLET's noncanonical internal baths mean this
equivalence cannot be assumed from its centroid bath parameters alone.
Assess the configured-temperature pressure separately from piston feedback;
see [the pressure definitions](native-ensembles.md#meaning-of-the-reported-pressure).

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

The native route still does not cover rate, FES or metadynamics workflows.
Surface/frozen-atom NPT, inferred Suzuki–Chin actions, and native
socket/GPU/bead-MPI backends remain outside this workflow. The separately
documented condensed-phase benchmark reports the evidence and limits of the
SC, PIGLET-NPT and RPC additions.

## Native diffusion opt-in

`task="diffusion"` may opt into the native route by setting
`sampling.engine="native"`.  The route is deliberately separate from the
legacy materials engine because `NativeEnsemble` has its own thermostat
splitting, random-number stream, and checkpoint fingerprint.  Equilibration
uses native PILE-L; after the equilibration boundary the accepted state is
handed to a second native engine with `thermostat="none"` for fixed-cell NVE
RPMD production.  The handoff uses `set_state`, so the production engine uses
the same Cayley map as native unthermostatted propagation and does not redraw
momenta at the boundary.

Native diffusion writes the same materials trajectory fields as the legacy
analysis (`sample_steps`, `times`, centroid `positions` and `velocities`,
`observables`, and `radii_gyration_squared`) under each replica's
`trajectory.npz`.  The journal remains the commit point, and `resume` checks
the checkpoint and journal clocks before replaying an incomplete transaction.
A completed resume reanalyzes those journals without constructing the PES or
consuming new random numbers.  The summary records `python_steps = 0` and the
number of native blocks for each replica.

The native engine records only steps after the equilibration boundary, so a
replica saves `production_steps // sample_interval` frames and has no frame at
production time zero; the legacy engine saves one more (the handoff frame).
`diffusion.max_lag`, `diffusion.block_length` and `diffusion.fit_range` are
checked against the native frame count before any output is created.

This opt-in currently supports fixed-cell PILE-L equilibration and one
trajectory worker per native replica.  NPT, trajectory batching, bond
constraints, and other equilibrium thermostats are rejected before creating
an output directory.  For a native pressure-controlled calculation use
`task="pimd"` with the native barostat controls described above; native NPT is
not part of the diffusion route.  The short
[`diffusion_native.inp`](../examples/ensembles/diffusion_native.inp) input is a
numerical control, not a materials validation or a claim of physical
long-time diffusion convergence.
