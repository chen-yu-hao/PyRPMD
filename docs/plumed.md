# Optional PLUMED bridge

`pyrpmd.plumed.PlumedBias` couples a separately installed PLUMED kernel to
PyRPMD's existing dynamics. It sends one set of centroid positions to PLUMED
and applies the resulting coordinate-dependent bias to the complete ring
polymer. Native PyRPMD sampling does not require PLUMED.

The bridge is available through a Python API and `task="plumed"` input files.
Use it for PLUMED coordinate CVs, restraints, metadynamics, and compatible
coordinate-based enhanced-sampling actions.
It does not add a second potential evaluator or a separate integrator.

## Installation

Install PLUMED and its Python wrapper using the
[official installation instructions](https://www.plumed.org/doc-v2.9/user-doc/html/_installation.html).
Both the `plumed` Python module and a loadable `libplumedKernel` are needed;
installing a Python wrapper alone may leave the kernel unavailable. The bridge
requires PLUMED API version 8 or later, checked at construction. `import pyrpmd`
and `import pyrpmd.plumed` do not import the optional wrapper.

The wrapper discovers its kernel normally, including through `PLUMED_KERNEL`.
Alternatively pass `kernel_path="/path/to/libplumedKernel.so"` explicitly.
PyRPMD leaves PLUMED installation and module selection to the user. Its test
suite includes an optional real-kernel check and independent protocol checks
that do not require PLUMED.

## Input-file workflow

A `task="plumed"` job uses the standard system, potential, dynamics and sampling
sections. The following small example is a numerical smoke check on a harmonic
model; it is not a molecular free-energy benchmark:

```toml
task = "plumed"
temperature = "300 K"
beads = 8

[system]
symbols = ["H", "H"]
geometry = [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]
geometry_unit = "bohr"

[potential]
provider = "harmonic"
params = { k = 0.01 }

[dynamics]
timestep = "0.05 fs"
integrator = "cayley-bcb"
loop_backend = "python"

[sampling]
thermostat = "pile-l"
integrator = "bcocb"
equilibration_steps = 100
production_steps = 1000
sample_interval = 10

[plumed]
file = "plumed.dat"
```

Place this restraint in `plumed.dat` beside the job input:

```text
UNITS LENGTH=Bohr ENERGY=Ha TIME=atomic
d: DISTANCE ATOMS=1,2 NOPBC
restraint: RESTRAINT ARG=d AT=1.4 KAPPA=0.05
PRINT ARG=d,restraint.bias STRIDE=10 FILE=COLVAR
```

```bash
pyrpmd validate job.inp --check-pes
pyrpmd run job.inp --output runs/plumed-input
```

Use exactly one of `plumed.file` and `plumed.input`; the latter is a complete
inline PLUMED string. Optional `plumed.kernel` selects a library path and
`plumed.charges` supplies one static elementary-charge value per atom. Input
and explicit kernel paths resolve against the job file's directory. Paths
inside PLUMED directives, including relative `FILE` and included input paths,
resolve from `RUN/plumed`. The log is `RUN/logs/plumed.log`.

The workflow supports one serial, all-mobile trajectory, fixed cell, all or
no periodic axes, and PILE-L or Andersen sampling. Sampling and dynamics must
use the same timestep so every PLUMED update has a consistent clock. PLUMED
updates begin at the first step and also occur during the declared
equilibration period; PyRPMD's saved production snapshots begin afterward.
Unsupported controls are rejected instead of silently combining another bias,
barostat, native sampling engine or electronic dynamics with the bridge.

`validate --check-pes` validates the PyRPMD PES without initializing a PLUMED
kernel or writing its files. Kernel availability, action compatibility and
PLUMED parsing are checked when the run starts. The saved
`checkpoints/plumed-md-state.npz` contains the accepted MD state, RNG and sampled
geometry/bias prefix, explicitly marked `kernel_state_saved=false`. On a kernel
failure, `results/plumed-failure.json` records the last accepted MD step and
error. By default these files do not capture PLUMED action history and
`pyrpmd resume` is rejected. Opt into the verified file-based profile below
when automatic continuation is needed. The deliberate Python restart contract
also remains available when the user supplies mutually consistent action files.

## Verified file-based continuation

Add `restart = "files"` to `[plumed]` **before the initial run**:

```toml
[plumed]
file = "plumed.dat"
restart = "files"
```

For a complete standalone input, run:

```bash
pyrpmd run examples/plumed/restart_metadynamics.inp --output runs/plumed-restart
# After an interruption, use the same run directory:
pyrpmd resume runs/plumed-restart
```

This profile uses PLUMED's own
[RESTART](https://www.plumed.org/doc-v2.10/user-doc/html/_r_e_s_t_a_r_t.html)
mechanism: METAD reconstructs its bias from HILLS and PRINT appends to COLVAR.
It does not attempt to pickle an opaque kernel. A checkpoint contains, in one
integrity-checked atomic NPZ replacement:

- The last checkpointed coordinates, momenta, absolute step/time and NumPy RNG.
- All sampled geometry/bias records up to that step.
- The complete bytes and path inventory of every declared METAD/PRINT output.
- A scientific fingerprint including the resolved input, system, PLUMED program,
  explicit input/kernel file hashes, and the versioned restart profile.

Resume validates the entire checkpoint before restoring files or initializing a
kernel. It restores the saved HILLS/COLVAR prefixes, including removal of partial
or additional writes made after the checkpoint. It then enables `setRestart`
and continues at the saved absolute clock. A restore-only update at the saved
endpoint initializes PLUMED's first-update flag (required by METAD in PLUMED 2.9).
The bridge verifies that this replay leaves the bias energy/forces/virial
unchanged, verifies the saved output prefixes, and truncates the duplicate PRINT
row before propagation resumes. The next accepted MD endpoint is updated exactly
once. This replay is restricted to the audited actions below; it is not used by
the unrestricted Python API.
Missing live history files can be recovered from the checkpoint. The checkpoint
itself must exist and pass integrity verification. Use the same PLUMED build and
numerical environment for continuation; the API version is also checked.

The profile appends `FLUSH STRIDE=1`, as documented by
[FLUSH](https://www.plumed.org/doc-v2.10/user-doc/html/_f_l_u_s_h.html), so no
buffered HILLS/COLVAR prefix is omitted. It sets `FMT=%.17g` on METAD and PRINT
unless `FMT=%.17g` or `FMT=%.17e` is already specified. Lower precision is
rejected because rounded hill centers or heights would change the reconstructed
bias. These explicit profile semantics are recorded in the checkpoint identity.
Old inputs receive no new defaults and retain their previous identity.

### Supported restart profile

The allowed coordinate actions are `DISTANCE`, `ANGLE`, `TORSION`, `POSITION`,
`COM`, `CENTER`, `COORDINATION`, `GROUP`, `FIXEDATOM`, `COMBINE`, `CUSTOM`,
`MATHEVAL`, `CONSTANT`, `RESTRAINT`, `UPPER_WALLS` and `LOWER_WALLS`, with `UNITS`
and `FLUSH` setup. Expressions and keyword values must be whitespace-free;
standard `ACTION ...` / `... ACTION` multiline input is supported. All `FILE`
paths must be distinct, normalized relative paths below `RUN/plumed`, without
symlinks. External reference/index files and included input are rejected.

The only history-dependent action in this profile is ordinary or well-tempered
`METAD` with fixed widths and a fixed deposition pace. Its accepted keywords are
`ARG`, `SIGMA`, `HEIGHT`, `PACE`, `FILE`, `BIASFACTOR`, `TEMP`, `FMT`, `LABEL`, and
`RESTART=AUTO`. `PRINT` additionally supports `STRIDE`, `UPDATE_FROM` and
`UPDATE_UNTIL`. Other actions and METAD options fail during input preflight.
In particular, adaptive widths or pace, grids, neighbor lists, multiple walkers,
acceleration/work accumulators, OPES and custom plugin actions are not certified
by this profile. These have additional state or numerical conventions beyond a
fixed HILLS prefix; see the
[METAD manual](https://www.plumed.org/doc-v2.10/user-doc/html/_m_e_t_a_d.html).
Standalone `RESTART` and per-action `RESTART=YES/NO` are rejected because the
workflow owns the restart flag for both fresh runs and continuations.

If a kernel call fails, the last complete atomic checkpoint is retained. The
failure report distinguishes `last_accepted_step` from `checkpoint_step`;
continuation replays any accepted but uncheckpointed steps. The output keeps
`kernel_state_saved=false` and adds `plumed_file_state_saved=true` to distinguish
file reconstruction from serialization of arbitrary internal kernel state.
Every checkpoint copies the complete declared history, so checkpoint I/O and
storage grow with HILLS/COLVAR length. Select `output.checkpoint_steps` with that
cost in mind; only the most recent checkpoint is needed for recovery.

The independent mock-kernel tests require bitwise agreement of coordinates,
momenta, RNG, samples, and file bytes between uninterrupted and interrupted runs,
including failures after a hill was written. The optional real-kernel tests
check numerical agreement of HILLS/COLVAR and MD state for ordinary and WT-METAD.
Decimal round trips and WT height rescaling can differ at the last floating-point
bit, so the live checks use an explicit tolerance. A skipped live test is not a
claim that the installed PLUMED runtime was verified. None of these restart
checks establishes a physically converged free-energy surface.

## Run a centroid bias

The runnable example uses a toy Morse bond, eight beads, and well-tempered
metadynamics on the centroid bond length:

```bash
python examples/plumed/centroid_metadynamics.py --steps 200 --output runs/plumed-centroid
```

It produces PLUMED's `HILLS` and `COLVAR`, a log, and a small JSON smoke-test
summary. This short example does not establish a converged free-energy
surface or a physical rate.

For an existing `Dynamics` and `State`, the essential API is:

```python
from pyrpmd.plumed import PlumedBias

lines = [
    "UNITS LENGTH=Bohr ENERGY=Ha TIME=atomic",
    "d: DISTANCE ATOMS=1,2 NOPBC",
    "restraint: RESTRAINT ARG=d AT=1.4 KAPPA=0.05",
]
with PlumedBias(
    lines, dynamics.masses, dynamics.beta, dynamics.nbeads, dynamics.dt,
    log_file="plumed.log",
) as bias:
    for _ in range(100):
        bias.step(dynamics, state, thermostat="pile-l", gamma_centroid=0.01)
        if bias.stop_requested:
            break
    result = bias.evaluate(state.q)
    total_ring_energy = dynamics.energy(state) + result.energy
```

Replace `lines` with a complete PLUMED input string, or use
`PlumedBias.from_file("plumed.dat", ...)`. Paths inside the PLUMED input are
resolved from the process working directory, as in PLUMED itself. Atom
indices in PLUMED inputs start at **1**. There are `N` centroid atoms, rather
than `P*N` separately numbered beads.

`thermostat=None` invokes `Dynamics.step`, `thermostat="andersen"` uses its
momentum sampler with an explicit `andersen_interval`, and
`thermostat="pile-l"` invokes `Dynamics.sample_step`. The existing PILE-L
integrator restrictions still apply. Biased time and thermostatted sampling
time do not become unbiased RPMD kinetic time.

## Units and ring-polymer normalization

All arrays and constructor parameters passed through the Python API retain
PyRPMD atomic units. The bridge declares their conversion factors through
PLUMED's MD interface:

| Quantity supplied to the bridge | Python/MD unit | PLUMED default unit |
|---|---|---|
| Positions and cell vectors | bohr | nm |
| Energy and bias | hartree | kJ/mol |
| Force | hartree/bohr | kJ/mol/nm |
| Time step | atomic time | ps |
| Mass | electron mass | amu |
| Optional charges | elementary charge | elementary charge |

These declarations do **not** change the units of text in `plumed.dat` or
PLUMED output files. PLUMED input and output use its defaults unless the
input begins with an explicit
[`UNITS` directive](https://www.plumed.org/doc-v2.9/user-doc/html/_u_n_i_t_s.html).
For example, `UNITS LENGTH=Bohr ENERGY=Ha TIME=atomic` sets text inputs to
atomic length, energy, and time units. PLUMED's named-unit constants can
differ slightly from the CODATA values installed with SciPy; the declared
MD conversion factors keep arrays and returned forces consistent.

For a physical centroid bias $W(\bar q)$, PyRPMD uses

$$
H'_P = H_P + P W(\bar q), \qquad
-\frac{\partial [P W(\bar q)]}{\partial q_j}
= -\frac{\partial W}{\partial \bar q}.
$$

The bridge therefore multiplies PLUMED's energy by `P` and broadcasts the
centroid force to **every** bead without division by `P`. The physical
temperature supplied as `setKbT` is `1/beta`, with no bead factor.
`PlumedBiasResult.physical_energy` is $W$;
`PlumedBiasResult.energy` is $PW$; `forces` has shape `[P,N,3]`;
and `centroid_forces` has shape `[N,3]`.

## Event order and failure handling

Call `bias.step(dynamics, state)` for every biased step. Passing a
`PlumedBias` directly as `Dynamics.step(..., bias=...)` raises an explanatory
error, because the ordinary force-only bias protocol cannot supply PLUMED's
event clock.

For a transition from step `n` to `n+1`, the wrapper:

1. Evaluates the initial force at `n` with `prepareCalc` and
   `performCalcNoUpdate`.
2. Lets the existing Dynamics integrator propagate and evaluate the endpoint
   force at `n+1`, again without updating history.
3. Calls PLUMED `update` exactly once, at the endpoint, after the full
   Dynamics map succeeds.
4. Commits the new MD state and clock.

The history is fixed during both force calls. The step label is accurate at
each endpoint for time-dependent actions. No update/deposition is performed
at the initial step zero; update events start at step one. This follows
PLUMED's documented
[split calculation/update interface](https://www.plumed.org/doc-v2.9/developer-doc/html/_how_to_plumed_your_m_d.html).

`evaluate(q)` recalculates the current bias and returns owned diagnostic
arrays without an `update` call. Repeated diagnostics cannot add hills or
advance PLUMED output strides. Keep each bridge bound to one trajectory;
its step clock must equal `state.step`, and `state.time` must equal
`state.step * dt`. The wrapper detects coordinate changes made outside it
after the first step. Construct a separate bridge and separate output paths
for each independent walker.

A failing PES/force call leaves the caller's state and Dynamics RNG
unchanged. PLUMED does not expose a generic transaction or rollback for
arbitrary internal actions and files. If a PLUMED command fails, the bridge
is marked unusable and requires a new kernel; even a failed `update` may
already have written part of an external file. This is not advertised as an
atomic MD-plus-PLUMED checkpoint for the unrestricted Python API. The restricted
input-file profile above instead retains a complete earlier MD/file checkpoint
and reconstructs a new kernel from it.

For a deliberate restart, restore mutually consistent MD state, Dynamics
RNG, PLUMED history/state files, and input first. Then construct a new
bridge with `restart=True` and `start_step=state.step`. The bridge forwards
PLUMED's restart flag but does not infer or validate the contents of all
action-specific restart files. The generic `pyrpmd resume` workflow requires the
explicit `restart="files"` profile described above. For PLUMED metadynamics, consult the
[METAD restart documentation](https://www.plumed.org/doc-v2.9/user-doc/html/_m_e_t_a_d.html).
Use a context manager or `close()` to flush/finalize resources.

## Geometry, supported actions, and limits

Supply a fixed full-rank `cell` with row lattice vectors in bohr and
`pbc=[True, True, True]` for a periodic system. A system without PBC supplies
a zero box. Mixed-axis periodicity is rejected rather than silently treating
a slab as fully periodic. The ring coordinates must be consistently
unwrapped before their centroid is formed; independently wrapping beads
changes the sampled centroid.

The supported action class is a coordinate-dependent bias whose Cartesian
forces are derived from the passed centroid geometry. Physical masses and
optional static charges are available for its CV definitions. Actions that
require `ENERGY` are detected after dependency preparation and rejected:
the derivative of a ring-averaged potential energy is not a centroid force.
Force/velocity-dependent CVs, adaptive biasing force methods that need the
unbiased atomic forces, dynamic charges, atom reordering, domain-decomposed
PLUMED MPI, and PLUMED replica-exchange orchestration are outside this
bridge's contract. The bridge is not a blanket claim that every action in
every optional PLUMED module is supported.

`plumed_virial` retains PLUMED's raw, unscaled bias virial in hartree as a
diagnostic. It is neither a complete ring-polymer virial nor a pressure
estimator and is not connected to a barostat. Use this bridge only with a
fixed cell. Changing cell, masses, bead count, temperature, or time step
requires a new bridge and compatible dynamics.

The local protocol tests check force finite differences for `P=1,4,16`,
unit directions, endpoint/deposition clocks, independent output ownership,
PES failure rollback, and failed-kernel rejection. The optional live test
checks a real PLUMED distance restraint and repeated METAD diagnostics.
Skipping that optional test means live PLUMED compatibility has not been
verified in that environment.

The dedicated `Sampling / real PLUMED` GitHub Actions job builds both the
kernel and Python wrapper from the official PLUMED 2.9.4 source at commit
[`52da2bb76d37dbad0d19f59c6fe0b2ab939e3ded`](https://github.com/plumed/plumed2/tree/52da2bb76d37dbad0d19f59c6fe0b2ab939e3ded).
It sets `PYRPMD_REQUIRE_PLUMED=1`, making missing dependencies a failure, and
runs the live tests and the executable METAD example. Its runtime provenance,
test results, HILLS/COLVAR files, and build logs are retained as an artifact.
A configured job is not a successful runtime check until that commit's
Actions result has passed.
