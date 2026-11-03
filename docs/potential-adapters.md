# Optional potential adapters

Named providers load their model or calculator once per provider context and
reuse it across evaluations. Optional packages are imported only when the
provider is requested. All PyRPMD-facing coordinates and cell vectors are in
bohr, energies and configurational virials in hartree, and forces in
hartree/bohr. Cell vectors are rows; trajectories remain unwrapped.

## Provider overview

| `potential.provider` | Optional installation | Boundary conditions | Evaluation route |
|---|---|---|---|
| `mace` | `pip install 'pyrpmd[mace]'` | Full, partial, or no PBC, subject to model | Reused MACE ASE calculator; frames evaluated in order |
| `nequip` | `pip install 'pyrpmd[nequip]'` | Full, partial, or no PBC, subject to model | Reused compiled-model NequIP ASE calculator |
| `deepmd` | `pip install 'pyrpmd[deepmd]'` | Full PBC or isolated; partial PBC rejected | One native `DeepPot.eval` call per batch |
| `fairchem` | `pip install 'pyrpmd[fairchem]'` | Subject to the selected model and task | Reused FAIRChem ASE calculator |
| `tblite` / `xtb` | `pip install 'pyrpmd[tblite]'` | Full, partial, or no PBC | Reused tblite xTB calculator; sequential electronic calculations |
| `pyscf` | `pip install 'pyrpmd[pyscf]'` | Isolated molecules | Reused HF/DFT object with fresh geometry and analytic gradients |
| `torch` | `pip install 'pyrpmd[gpu]'` | Explicit periodic model contract below | Native tensor batch |
| `openmm` | `pip install 'pyrpmd[openmm]'` | All or no PBC; reduced lower-triangular cells | Reused Context or explicit concurrent Context pool; virtual-site forces redistributed |
| `lammps` | Matching LAMMPS Python module and shared library, or `pip install 'pyrpmd[lammps]'` | Full, partial, or no PBC; restricted-triclinic cells | Reused library instances, optional bead affinity and neighbor reuse; private MPI subgroups |

ASE-backed models accept a batch at the PyRPMD interface but call the calculator
once per frame. They do not advertise native GPU batching. This choice reuses
the established neighbor-list, species, units and stress interfaces without
silently assuming a particular graph-model format. Models retain their own
training domain, reference electronic state and numerical precision. Converting
outputs to NumPy float64 does not increase a model's inference precision.

The fixed System supplies atom identities, cell, PBC, charge and multiplicity.
`D` and `T` map to the chemical element `H` in potential engines; isotope masses
remain in PyRPMD dynamics. Do not repeat System fields inside `potential.params`.

## Independent trajectory batching

The `ase`, `mace`, `nequip` and `fairchem` adapters accept an explicit
**caller-audited** declaration:

```toml
[potential]
provider = "ase"
factory = "ase.calculators.eam:EAM"
trajectory_batch_safe = true
params = {potential = "/models/NiAlH_jea.eam.alloy"}

[parallel]
trajectory_batch_size = 4
bead_workers = 1
```

The declaration promises independent, history-independent energies, forces and
virials: changing the frame order or count, interleaving replicas, or using a
new calculator instance must not change any frame's physical result. Audit the
actual calculator/model, including its convergence and warm-start behavior.
The [Ni–H example](../benchmarks/materials/batch_acceptance.md) checks these
properties for a checksum-pinned EAM table and compares against ASE directly.

Without an explicit declaration in the input or calculator metadata, ASE is
not assumed safe for cross-trajectory batching. A calculator's explicit
`trajectory_batch_safe=false`, `batch_independent=false`, `stateful=true`,
`capabilities.stateful=true`, `coupled_frames=true` or
`ring_polymer_coupled=true` veto cannot be overridden by the input. The adapter
preserves these declarations through named models and transparent Python
wrappers. The flag must be a boolean and is accepted only for these four
ASE-backed provider kinds; custom Python/Torch models use their existing
provider metadata contract.

This execution declaration is excluded from the scientific checkpoint
identity, so an otherwise compatible scalar checkpoint can acquire the flag
and resume batched. Model parameters and files remain part of that identity;
the separate ASE force-consistent energy convention is unchanged. ASE still
loops through frames with `native_batch=false`. A larger PyRPMD request reduces
dispatch calls but neither creates a fused GPU forward pass nor guarantees
better throughput. Benchmark the intended provider, workload and device.

## OpenMM

An OpenMM `System` XML file defines the force field, particle order, charges,
nonbonded method and virtual sites. A Python factory returning an
`openmm.System` is an alternative. The adapter makes an owned copy and never
advances an OpenMM integrator. PyRPMD remains responsible for the ring springs,
nuclear masses, timestep, thermostats and bond constraints.

```toml
[potential]
provider = "openmm"
model = "water-system.xml"
device = "cpu"
params = {platform = "CPU", properties = {Threads = "1"}}
```

For a factory, replace `model` with `factory = "water.py:build_system"` and put
its keyword arguments in `params.factory_kwargs`. The
[harmonic teaching input](../examples/potentials/openmm_harmonic.inp) is runnable
without a separately supplied force field. `model` and `factory` are exclusive.

PyRPMD atom arrays contain **only non-virtual particles**, in increasing OpenMM
particle-index order. After setting each frame, the adapter updates virtual
sites and reads the forces already redistributed to their parent atoms. This
supports four-site water layouts without inventing masses for their charge
sites. The caller must preserve atom identity/order when building the model;
an OpenMM System does not contain element names to verify against coordinates.

Periodic forces must already be configured in the model, including Ewald/PME,
cutoff, error tolerance and dispersion correction choices. Model periodicity
must agree with System PBC. Use the OpenMM reduced lower-triangular row-cell
convention; partial PBC and arbitrary cell rotations are rejected. The adapter
converts nm and kJ/mol to PyRPMD atomic units. It does not currently provide a
physical virial, so it is a fixed-cell force route rather than an NPT provider.

`device="cuda:0"` selects CUDA device 0 and defaults to double precision;
`device="opencl:0"` selects OpenCL. `params.properties` accepts the installed
platform's string properties. Conflicting platform/device choices fail.
PyRPMD CUDA worker routing supports this provider. The default evaluates frames
sequentially in one reused Context. GPU availability and force accuracy must be
checked on the intended hardware.

OpenMM constraints must also appear with the same pairs and lengths in
`system.bond_constraints`. `bind_system` checks them; the force provider never
silently projects coordinates. Use a compatible PyRPMD constrained propagator.
Remove OpenMM barostats, Andersen thermostats and center-of-mass removers from
the force-only model. XML/source files and dependency versions enter the run
provenance; the adapter also reports its serialized-model SHA256.

### Persistent OpenMM copies

For many beads or a combined bead/trajectory batch, explicitly select a pool:

```toml
[potential]
provider = "openmm"
model = "water-system.xml"
device = "cpu"
params = {platform = "CPU", properties = {Threads = "1"}, batch_mode = "contexts", max_workers = 4}

[parallel]
backend = "serial"
bead_workers = 1
```

Every input row gets its own persistent Context, including positions, virtual
sites, force kernels and neighbor state. Contexts are allocated lazily up to
the largest batch seen and retained until `close()`. For P beads and T
independent trajectories this can require P×T Contexts; `max_workers` bounds
simultaneous evaluations, **not** the number of resident copies. A smaller
later batch reuses the leading copies. Input order is preserved and no Context
is accessed by two workers at once. Cell changes reach every copy and roll back
on failure; a failed evaluation returns no partial batch.

Start with `Threads="1"` and `max_workers` no larger than the allocated CPU
cores. Benchmark the complete batch before selecting a worker count: Context
memory and cache pressure can outweigh concurrency for small models. The
adapter does not change OpenMM thread counts, precision or device properties.
On CUDA/OpenCL all copies use the explicitly selected device and precision;
ensure enough device memory for the largest expected batch. The pool never
calls an OpenMM integration step and does not introduce springs, a thermostat
or constraint projection.

This is concurrent evaluation through the public Context API. It still uploads
each changed configuration and retrieves each copy's state; it does not claim
a fused native batch or a single host/device transfer (`native_batch=false`).
The legacy `batch_mode="serial"` remains the default, with unchanged metadata.
Its `max_workers` must be 1. New mode/worker parameters stay in the ordinary
potential input and therefore in the run identity; existing inputs acquire
no new scientific options.

The `RPMDIntegrator` copy API was also evaluated. Its force-only `getState`
path copies a replica into a Context and evaluates that Context, so it does
not remove the per-copy state requests. It also cannot preserve the adapter's
general constraint/virtual-site contract without extra handling. OpenMM's
force-group contraction is not enabled: PyRPMD's
[`SplitPotential`](../src/pyrpmd/potentials/contraction.py) requires one complete
ring and a specific coordinate projection, lifted force scaling and energy
bookkeeping, while this adapter accepts arbitrary independent rows, including
multiple trajectories. No ring topology is inferred from the batch dimension.
Use PyRPMD's explicit contraction wrapper with suitable force components when
that approximation is intended.

Upstream contracts: [Context evaluation and virtual sites](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.Context.html),
[periodic boxes, serialization and virtual sites](https://docs.openmm.org/latest/userguide/theory/05_other_features.html),
[platform properties](https://docs.openmm.org/latest/userguide/library/04_platform_specifics.html),
and [RPMD copy evaluation implementation](https://github.com/openmm/openmm/blob/8.6.1/plugins/rpmd/openmmapi/src/RPMDIntegrator.cpp).

## LAMMPS as a library

A LAMMPS setup script supplies atoms, IDs, types, charges, topology and force
field. It must create a complete three-dimensional system and contain no fixes
or dumps. The default evaluates `run 0 pre yes post no`, without a LAMMPS integration
step. The explicit persistent mode below refreshes forces without advancing
physical coordinates. Standard static force fields can use the installed KSPACE package for
Ewald/PPPM and the accelerator packages available in that LAMMPS build.
The initial adapter accepts standalone LJ, Buckingham/Born, Coulomb, Morse,
soft/table, EAM/MEAM, Tersoff/SW/EDIP, AIREBO/REBO/BOP and selected static ML
pair families, including accelerator suffixes. Hybrid and velocity-dependent
styles (including DPD), engine r-RESPA and shrink-wrap boundaries are rejected.

```toml
[potential]
provider = "lammps"
model = "potential.in"
params = {type_map = ["O", "H"], asset_files = ["water.data", "forcefield.inc"]}
```

Atom IDs must be consecutive `1..N` in PyRPMD's atom order. `type_map[0]` names
LAMMPS type 1, and so on; the adapter verifies the resulting element order,
including H aliases for D/T. LAMMPS may reorder its local arrays during neighbor
rebuilds: forces are reordered by atom ID before returning them to PyRPMD.
Use `units real` or `units metal`; the adapter reads the actual style from the
engine and rejects reduced LJ units and `potential.units` overrides.

The script's box/PBC must match `system.cell` and `system.pbc`. Full and partial
PBC are supported with LAMMPS restricted-triclinic row vectors. Isolated systems
still need a sufficiently large fixed nonperiodic LAMMPS box. Coordinates outside
its nonperiodic bounds raise an error. Periodic wrapping affects only an engine
copy, preserving the unwrapped PyRPMD trajectory used for diffusion.

Physical virials come from `compute pressure NULL virial`, including the
reciprocal-space contribution and excluding kinetic and ring-spring terms.
They are converted using the engine's pressure conversion, not reconstructed
from wrapped positions and forces. Full-periodic cell updates use `change_box`
without remapping the caller's positions.

Alternatively, `factory="potential.py:build_engine"` calls
`build_engine(comm=..., **factory_kwargs)`. It must return a fully configured
`lammps.lammps` instance and honor the supplied communicator. By default PyRPMD
supplies `MPI.COMM_SELF` inside a multi-rank job. The explicit MPI subgroup
configuration below allows several ranks per provider without passing
`MPI.COMM_WORLD` to LAMMPS.
An MPI-enabled LAMMPS wheel requires its matching runtime, such as MPICH; a
successful Python import alone does not establish that the shared library loads.
Do not call `lammps.finalize()` in a factory: provider cleanup uses `close()` and
preserves the host application's MPI state.

`params.cmdargs` and `params.library_name` apply to script loading. Accelerator
selection belongs to LAMMPS; PyRPMD does not claim a GPU device from a generic
`device` string, and its GPU worker scheduler rejects this provider. LAMMPS
potential/setup data use their own licensing and validity domains. **Every
external dependency must be listed in `params.asset_files`**, including data,
EAM/table files, include scripts and files loaded by a factory. Paths resolve
relative to the PyRPMD input. Their SHA256 values enter both provider metadata
and the run manifest; resume rejects changed or missing listed assets. An empty
list is appropriate for a self-contained setup. PyRPMD does not attempt to infer
dependencies from LAMMPS variables, nested includes or arbitrary factory code;
omitting an external file leaves that dependency outside the restart guarantee.
Script asset paths can use `${pyrpmd_model_dir}/filename`; the adapter never
changes the process working directory.

The [Lennard-Jones teaching input](../examples/potentials/lammps_lj.inp) exercises
the public workflow. These are coordinate-only force providers: no LAMMPS
thermostat, SHAKE, charge-equilibration fix, driven or velocity-dependent force
can be substituted for a PyRPMD potential.

### Bead affinity and safe neighbor reuse

```toml
[potential]
provider = "lammps"
model = "potential.in"
# neighbor_skin is in bohr; 3.77945225 bohr is approximately 2 angstrom.
params = {type_map = ["Ar"], batch_mode = "persistent", neighbor_skin = 3.77945225}
```

`batch_mode="persistent"` keeps one configured instance for each input row,
up to the largest batch seen. The first frame for a slot uses full setup;
later evaluations retain that slot's neighbor list while every atom remains
strictly within half the skin distance of its last rebuild position. The
comparison is against the last rebuild, not just the preceding evaluation.
At the threshold, after a cell change, or after cache invalidation, full setup
is required. A wrapped coordinate crossing a periodic face also forces a
conservative rebuild, even when its true unwrapped displacement is small: the
existing ghost images must remain consistent with the coordinates written to
the engine. Periodic wrapping is confined to the engine; PyRPMD input arrays
remain unwrapped. Changing frame order can cost a rebuild but does not change
the potential or its results.

The adapter sets `neighbor <skin> bin` for this mode. The default skin is
2 angstrom, expressed internally in bohr; an explicit `neighbor_skin` must be
finite and positive. The legacy serial mode retains the setup script's
neighbor settings. It does not silently switch to persistent mode.

**`run 0 pre no` is not a force refresh:** it skips the initial force evaluation
as well as setup and can return stale forces. On a valid retained list the
adapter instead executes `run 1 pre no post no` with no integration or force
modifying fixes. This invokes communication and force/virial kernels, while
positions and velocities stay unchanged. Only LAMMPS' private evaluation
counter advances; it is not PyRPMD's physical time and is not checkpoint state.
The force engine remains restricted to static, coordinate-only force fields.
Every restarted provider reconstructs its cache and begins with full setup.
Do not mutate an exposed engine's force parameters, neighbor configuration,
computes or fixes between evaluations; create a new provider for a changed
model. `set_cell` is the supported cell-update entry point.

`batch_stats` reports resident instances, full rebuilds and reused-list
evaluations. This cache uses memory proportional to the largest batch; include
model, neighbor and accelerator memory when choosing bead/trajectory batch
sizes. Failed batches are never returned partially; provider failures close
the affected provider instead of reusing uncertain engine state.

### Several MPI ranks per bead worker

The existing G×B hierarchy can give each bead worker a private group of R
LAMMPS ranks through `parallel.pes_ranks`:

```toml
[parallel]
backend = "mpi"
mode = "bead"
trajectory_groups = 2
bead_workers = 4
pes_ranks = 2
threads_per_rank = 1
scheduling = "static"
```

Launch this example with 16 ranks: `mpiexec -n 16 pyrpmd run simulation.inp`.
Static execution needs `G * B * R` ranks; dynamic scheduling adds one dedicated
coordinator, for `1 + G * B * R`. Members of a private bead subgroup evaluate
the same chunk collectively, LAMMPS decomposes atoms within that subgroup, and
only its leader contributes the result to the bead assembly. Unrelated bead
groups never share a LAMMPS communicator. `pes_ranks=1` keeps the previous
topology and behavior. This option is restricted to the MPI LAMMPS path;
process workers and PyRPMD's GPU scheduler reject multi-rank engine groups.

For direct Python use, `LAMMPSPotential(..., communicator=private_comm)` accepts
an explicit non-null intracommunicator. The provider duplicates and owns it;
the caller retains ownership of the original. Pass a communicator obtained by
`Split` or `Dup`, never `MPI.COMM_WORLD` itself, and call construction,
evaluation, cell updates and close collectively on its members. MPI objects
are runtime resources and are not placed in the serializable input or run
identity. The factory must honor the supplied communicator; a mismatching
engine fails validation.

Upstream contracts: [Python library](https://docs.lammps.org/Python_module.html),
[units](https://docs.lammps.org/units.html),
[configurational pressure](https://docs.lammps.org/compute_pressure.html),
[populated-box changes](https://docs.lammps.org/change_box.html),
[run setup semantics](https://docs.lammps.org/run.html),
and [neighbor skin/rebuild conditions](https://docs.lammps.org/neighbor.html).

### Choosing the direct adapter or an i-PI socket client

| Workload | Suggested route | Starting configuration |
|---|---|---|
| Small systems, few frames, or limited memory | Direct serial OpenMM/LAMMPS | One reused engine; measure before adding copies |
| Many water beads on a multicore CPU | OpenMM Context pool | `batch_mode="contexts"`, `Threads="1"`, explicit worker count |
| Static LJ/EAM/ML forces with small per-step displacements | Direct persistent LAMMPS | One instance per stable batch slot; measure rebuild/reuse counts |
| Larger systems needing atom decomposition inside each bead worker | Direct MPI LAMMPS | `pes_ranks>1` with explicit G×B×R rank budget |
| Separately managed MPI/GPU LAMMPS executables, incompatible Python/MPI libraries, or engines on other nodes | i-PI socket provider | Multiple independent clients, each with its own MPI partition/job |

The existing socket provider serves one geometry per protocol request and
dispatches a batch over persistent clients. It avoids linking LAMMPS into the
PyRPMD Python process and lets each client use its own MPI/runtime and
accelerator configuration. Network copies and latency remain; it is most
useful when force calculations are sufficiently expensive to amortize them.
Do not add LAMMPS `fix pimd/langevin` to either route: PyRPMD already propagates
the ring polymer.

A server configuration for four independent MPI clients is:

```toml
[potential]
provider = "ipi"
params = {host = "127.0.0.1", port = 31415, clients = 4, timeout = 300.0}

[parallel]
backend = "serial"
mode = "serial"
bead_workers = 1
```

Run PyRPMD first so the listening socket exists. Configure a LAMMPS client
with the same atom order, force field and boundary conventions, then add
`fix force_client all ipi 127.0.0.1 31415` and a sufficiently long `run` command.
A single `mpiexec -n 16 lmp -partition 4x4 -in force-client.in` can provide four
clients with four ranks each; four separate four-rank jobs can connect to the
same server as well. This is a socket-client script, not a script for the
direct adapter (which refuses fixes). Use a Unix socket on one host when
supported, or an explicitly reachable TCP address across nodes.
LAMMPS' `fix force_client all ipi name 31415 unix` connects to `/tmp/ipi_name`; set
PyRPMD's `params.unix` to that full path, rather than just `name`.

For long-range forces under substantial cell changes, follow the LAMMPS
[`fix ipi reset` contract](https://docs.lammps.org/fix_ipi.html) to reinitialize
box-dependent electrostatics. Select and record this choice consistently with
the intended Hamiltonian. PyRPMD expects atomic units on the i-PI wire; LAMMPS
handles conversion from its supported unit style. List and checksum client
models/configuration in the run's external provenance: a socket connection
alone cannot authenticate a remote force field.

### Reproducing the force-engine benchmark

The source-distributed
[`force_engine_adapters.py`](../benchmarks/efficiency/force_engine_adapters.py)
compares the actual pre-change adapter source against the new modes:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENMM_CPU_THREADS=1 \
  python benchmarks/efficiency/force_engine_adapters.py \
  --legacy-ref 8cc402fb12391e4fb644d3e186e5e58d1699759a \
  --rpmd-copy --warmup 2 --repeats 7 --output force-engine-timings.json
```

The historical commit must be available in the local checkout. Install matching
OpenMM/LAMMPS/MPI runtimes first; `--engine openmm` or `--engine lammps` selects
one engine. The default has P=32, q-TIP4P/F water with 192 and 2187 physical
atoms, and FCC Lennard-Jones systems with 256 and 2048 atoms. The water model
comes from the existing `benchmarks/water` factory, with PME tolerance `1e-6`;
the cutoff is `min(9 angstrom, 0.45 * box side)` so the small box is valid too.
These are perturbed initial grids used for evaluator timing, not equilibrated
liquid configurations or evidence of physical convergence.

Each timed call returns the complete batch's energies and forces, plus LAMMPS
virials. Coordinates change between calls and match exactly across modes.
Construction and the first evaluation are reported separately from warmup and
steady-state samples. The JSON includes every sample, medians, the 95th
percentile, maximum absolute and relative-infinity-norm differences, source
hashes, dependency versions, CPU quota and thread settings. CPU comparisons
include serial OpenMM with 1/4/8 internal threads and resident Context pools
with 1/4/8 workers at one internal thread each. This exposes both concurrency
gains and the overhead of retaining many copies. LAMMPS records neighbor
rebuild/reuse counts and rejects results exceeding `1e-10` relative error.
The optional `--rpmd-copy` control loads all copies through `RPMDIntegrator`
and queries them without ever calling `step()`. Its water-only probe explicitly
fills the model's three-parent average virtual sites; it is not a production
adapter for arbitrary Systems and does not apply ring-polymer contraction.

On a machine with a CUDA platform, repeat with `--engine openmm --device cuda:0`.
The adapter retains its default double precision; it never switches to CPU or
a lower precision to make the benchmark run. CPU PME agreement is assessed at
OpenMM CPU precision. Exact restart tests use controlled Reference/LJ cases;
they do not certify every external model and accelerator as bitwise reproducible.

### Measured CPU results (2026-11-02)

The [complete baseline comparison](validation/t05-force-engines/benchmark-cpu.json)
ran on an AMD EPYC 9V74 host with an **8-CPU cgroup quota**, nine affinity-visible
CPUs and 8 GiB memory. OpenMM was 8.6.1 (CPU platform); LAMMPS was wheel
2025.7.22.4.0 with MPICH 5.0.2. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`OPENMM_CPU_THREADS` were 1. No CUDA platform or NVIDIA device was available.
These are shared-host wall times, not dedicated-node scaling measurements.
Each entry is the median of seven complete P=32 batch evaluations after two
warmups; construction and the cold first evaluation are excluded and retained
in the raw JSON. All modes use the same changing coordinate sequence.

| OpenMM mode | Internal CPU threads per Context | Workers | Water, 192 atoms (ms) | Water, 2187 atoms (ms) |
|---|---:|---:|---:|---:|
| Original serial adapter | 1 | 1 | 443.50 | 2283.54 |
| Original serial adapter | 4 | 1 | 521.75 | 1330.58 |
| Original serial adapter | 8 | 1 | 855.62 | 1390.62 |
| Persistent Contexts | 1 | 1 | 896.25 | 1773.62 |
| Persistent Contexts | 1 | 4 | 236.45 | 647.61 |
| Persistent Contexts | 1 | 8 | **145.37** | **451.60** |
| Force-only RPMD copy probe | 1 | 1 | 560.11 | 1141.07 |

Eight Context workers improve on the original one-thread adapter by 3.05x and
5.06x. Compared with the best measured original thread configuration at each
size, the gains are **3.05x and 2.95x**. Retaining many Contexts without
concurrency is not a general speedup. Across the new Context modes, maximum
relative-infinity-norm differences were `9.28e-7` for energies and `1.61e-6`
for forces, consistent with the tested CPU precision contract. The copy probe
matched the serial energies and forces exactly for these water inputs.

The original FCC benchmark puts many atoms exactly on periodic faces. Tiny
perturbations then trigger the safe wrapped-image rebuild path. To measure
ordinary neighbor reuse separately, a second comparison translates the entire
periodic crystal by 0.25 lattice spacings (1.315 angstrom) along each axis.
This translation leaves the periodic LJ model unchanged. Both layouts and
their outcomes are retained; the translated case is not a replacement for the
boundary-crossing result.

| LJ layout | Atoms | Original serial (ms) | Current serial (ms) | Persistent (ms) | Original / persistent | Rebuilds / reuses |
|---|---:|---:|---:|---:|---:|---:|
| FCC planes on boundaries | 256 | 52.56 | 51.46 | 41.83 | 1.26x | 280 / 40 |
| FCC planes on boundaries | 2048 | 261.80 | 276.00 | 285.02 | **0.92x** | 288 / 32 |
| Uniformly translated FCC | 256 | 37.19 | 23.84 | 32.43 | 1.15x | 32 / 288 |
| Uniformly translated FCC | 2048 | 123.65 | 130.54 | 116.74 | 1.06x | 32 / 288 |

Counts include cold, warmup and timed evaluations: 320 individual frames in
each row. The translated layouts retain all lists after the initial setup.
Their energies, forces and virials match the old adapter exactly; across both
layouts the largest relative error is `6.50e-14`, below `1e-10`. Removing setup
does not guarantee a wall-time gain: the boundary layout is slower at 2048
atoms, and the current serial path is faster for the translated 256-atom case.
Keep serial as the default and benchmark the intended model, layout, batch
size and MPI rank budget before selecting persistent mode. These measurements
do not establish EAM/ML throughput or multi-node MPI scaling.

The [translated-grid report](validation/t05-force-engines/benchmark-lammps-offset.json)
is reproduced with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENMM_CPU_THREADS=1 \
  python benchmarks/efficiency/force_engine_adapters.py --engine lammps \
  --legacy-ref 8cc402fb12391e4fb644d3e186e5e58d1699759a \
  --lj-grid-offset 0.25 --warmup 2 --repeats 7 --output lammps-offset-timings.json
```

The first report's exact benchmark source is preserved in commit
`a928171521c94e886f9e638278a3e05e3cf5d8d8`; the subsequent script only adds
the explicit grid-offset control, whose default remains zero. Source hashes,
commands and raw test/build logs are indexed in
[validation.json](validation/t05-force-engines/validation.json).

### T19 measurements and RPMD copy evaluation (2026-11-03)

The T19 [validation record](t19-performance.md) adds reproducible large-system
profiling and an explicit `batch_mode="rpmd"` route. CUDA remains unmeasured
in this environment; the CPU results are not GPU speedup claims.

The new explicit OpenMM mode keeps every batch row in one `RPMDIntegrator`
copy bank and one Context:

```toml
[potential]
provider = "openmm"
model = "system.xml"
device = "cuda:0"
params = {batch_mode = "rpmd", properties = {Precision = "double"}}
```

Only physical System energies and forces are requested. No integration,
thermostat, ring springs or force-group contraction is performed. The copy
bank grows when needed and retains capacity for smaller batches; the public
copy API still uploads and retrieves each row separately. Virtual sites use
OpenMM's kernels and parent-force redistribution, with a per-row physical
position check guarding the copy-to-Context implementation. Those additional
position downloads are included in measured time.

This mode requires a **fixed cell and a System without constraints**. A
changed cell is rejected before mutation, preventing automatic PME grids
from depending on a previous cell or maximum batch size. The existing
serial and concurrent-Context modes retain their current contracts and
metadata; serial remains the default. `max_workers>1` remains specific to
the concurrent-Context mode.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENMM_CPU_THREADS=1 \
  python benchmarks/efficiency/t19_openmm.py --device cuda:0 \
  --model lj --atoms 64 512 4096 --beads 8 32 64 \
  --workers 4 --rounds 3 --output t19-openmm-cuda.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/efficiency/t19_lammps.py \
  --baseline-ref f87705115c7496821b283a0966c991abd541231c \
  --atoms 256 2048 4096 --beads 32 --offsets 0 .173 \
  --rounds 3 --output t19-lammps-cpu.json
```

OpenMM compares every measured configuration to the float64 Reference
platform, records separate construction/cold costs and alternates mode order
across rounds. Missing CUDA produces explicit unmeasured rows and exit code
2. Use `--device cpu --platform CPU` only for an explicitly labeled CPU
measurement.

The LAMMPS profiler compares the actual pinned old adapter with the new
linear-time atom validator, in both serial and persistent modes. Boundary-
aligned and translated FCC grids separate frequent image-triggered neighbor
rebuilds from ordinary reuse. Uninstrumented wall/CPU medians and separate
phase profiles identify where time is spent. Periodic wrapping, neighbor
rebuild rules, force refresh and scientific metadata are unchanged; no
history-dependent periodic image cache is introduced.

### ASE energy and force consistency

ASE-backed adapters request `atoms.get_potential_energy(force_consistent=True)` so that a
calculator with electronic smearing returns the electronic free energy differentiated by
its forces and stress. This is the calculator's electronic thermodynamic potential, not
a nuclear free energy or a PyRPMD PMF. Electronic smearing temperature and nuclear thermostat
temperature remain separate model inputs. See the
[ASE energy contract](https://docs.ase-lib.org/ase/atoms.html#ase.Atoms.get_potential_energy).

If a calculator does not advertise `free_energy` and raises ASE's
`PropertyNotImplementedError`, the adapter falls back to its ordinary energy. A calculator
that advertises `free_energy` but fails to provide it raises an error; other calculator
failures are not swallowed. Classical calculators whose two energies coincide retain their
energy convention. The ASE energy convention is part of the run identity: older ASE-backed
checkpoints cannot be resumed or recomputed with the new adapter, including classical runs
where the two energies happen to coincide. Their existing summary can still be read. Start
a new output directory; new-convention runs resume normally. In particular, do not splice
old zero-temperature extrapolated energies or free-energy weights into a trajectory using
the corrected convention.

The identity guard recognizes the explicit `ase`, `mace`, `nequip` and `fairchem` providers;
it cannot discover an ASE calculator hidden inside arbitrary Python factory code. Keep the
source and energy convention of such plugins in the model provenance as well.

## MACE and NequIP

Python input dictionaries:

```python
mace = {
    "provider": "mace",
    "model": "/models/material.model",
    "device": "cuda:0",
    "params": {"default_dtype": "float64"},
}

nequip = {
    "provider": "nequip",
    "model": "/models/material.nequip.pt2",
    "device": "cuda:0",
    "params": {
        "chemical_species_to_atom_type_map": {"H": "H", "O": "O"},
    },
}
```

MACE uses `MACECalculator(model_paths=..., device=...)`. Its adapter defaults to
`default_dtype="float64"`; other calculator options can be passed in `params`.
NequIP uses `NequIPCalculator.from_compiled_model(compile_path=..., device=...)`.
Compile the NequIP checkpoint for the target integration and hardware; AOT
artifacts require the ASE compile target. The default species mapping is an
explicit identity mapping. Supply a mapping when trained type names differ
from chemical symbols. NequIP's `energy_units_to_eV` and `length_units_to_A`
options handle models trained in other units before the ASE-to-AU conversion.

These routes use the maintained upstream interfaces described in the
[MACE calculator source](https://github.com/ACEsuit/mace/blob/develop/mace/calculators/mace.py)
and [NequIP ASE guide](https://nequip.readthedocs.io/en/latest/integrations/ase.html).

## DeePMD

```python
deepmd = {
    "provider": "deepmd",
    "model": "/models/graph.pb",
    "params": {
        "model_kwargs": {"auto_batch_size": True},
        "eval_kwargs": {},
    },
}
```

The adapter obtains species indices from `DeepPot.get_type_map()`. An explicit
`params.type_map` can override that mapping for checkpoints whose type labels
are not chemical symbols. The map must contain each system element exactly
once. Full cells are passed as `[B,9]`, with row vectors flattened in C order;
isolated calculations pass `cells=None`. Partial PBC is rejected because this
API has no independent boundary flags.

Inputs default to the usual DeePMD dataset units of angstrom and eV. A model
trained in different consistent units requires an explicit `potential.units`
mapping. `eval_kwargs` can contain trained frame or atom parameters such as
`fparam` or `aparam`; they cannot replace geometry, type mapping or the expected
energy/force/virial output layout. The System's charge and multiplicity do not
automatically become a model input: electronic-state-dependent models must
receive the appropriate trained parameters.

`DeepPot` chooses its execution device through its backend. The adapter reports
`execution_device="backend"`; it does not claim that a generic `device` argument
selects a particular GPU. Configure the DeePMD runtime before starting the
process and use PyRPMD serial or bead mode. Explicit PyRPMD GPU scheduling for
this adapter is rejected. Native model batching remains active in serial mode.

See the [official inference API](https://docs.deepmodeling.com/projects/deepmd/en/v3.2.0/autoapi/deepmd/infer/deep_pot/index.html).

## fairchem

```python
fairchem = {
    "provider": "fairchem",
    "model": "/models/uma-checkpoint.pt",
    "device": "cuda:0",
    "params": {
        "task_name": "omat",
        "inference_settings": "high-precision",
    },
}
```

The adapter uses `FAIRChemCalculator.from_model_checkpoint`. A local checkpoint
or an upstream registered model name is accepted. Registered models may require
an authorized upstream download; PyRPMD supplies no weights or model access.
`task_name` is required so the electronic-structure reference is an explicit
choice. The calculator receives integer total `charge` and `spin` (multiplicity)
through `Atoms.info` for `task_name="omol"`. Other UMA tasks use the upstream
`charge=0, spin=0` model-input sentinels. Keep System's neutral/default values
(`charge=0, multiplicity=1`) for these tasks: they cannot select an electronic
charge or magnetic state, and other System values are rejected before model
loading. The zero model spin is distinct from a molecular singlet constraint.

Select inference settings supported by the installed fairchem version.
Current fairchem's `high-precision` mode is full FP32, not FP64; defaults and
available acceleration modes can change between releases. The adapter preserves
upstream choices and does not enable untrained stress predictions. A task without
stress produces no virial through the ASE route. Check a model's energy drift
and gradient accuracy before using its trajectories as production evidence.

References: [FAIRChem calculator API](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/calculate/ase_calculator.py),
[inference guide](https://fair-chem.github.io/core/common_tasks/ase_calculator.html),
the [UMA task and electronic-state conventions](https://github.com/facebookresearch/fairchem/blob/main/docs/core/uma.md),
and [UMA precision changelog](https://facebookresearch.github.io/fairchem/uma-changelog/).

## On-the-fly xTB through tblite

```python
xtb = {
    "provider": "tblite",
    "params": {
        "method": "GFN2-xTB",
        "accuracy": 0.1,
        "max_iterations": 250,
        "electronic_temperature": 300.0,  # kelvin, distinct from nuclear T
        "warm_start": False,
    },
}
```

Available methods are `GFN2-xTB`, `GFN1-xTB` and `IPEA1-xTB`. System multiplicity
maps to `uhf=multiplicity-1`. Settings use tblite's native `Calculator.set`
interface; `params.settings` allows additional supported native options. The
electronic temperature is converted from kelvin to hartree. Convergence or
engine failures raise `PotentialError` and discard cached electronic state.

The calculator object survives geometry and cell updates. `warm_start=False`
starts each electronic solve without a previous result; this is the default.
With `warm_start=True`, result objects are cached by batch index. Preserve bead
ordering when using this option and assess electronic-state convergence.

tblite's result named `virial` is the tensile energy-strain derivative. PyRPMD
negates it, matching the upstream ASE relation `stress=tblite_virial/volume` and
PyRPMD's `W=-volume*stress` convention. No coordinate wrapping, kinetic pressure,
spring pressure or bead normalization is added.

References: [tblite Python API](https://tblite.readthedocs.io/en/latest/api/python.html)
and [upstream ASE conversion](https://tblite.readthedocs.io/en/latest/_modules/tblite/ase.html).

## On-the-fly molecular PySCF

```python
pyscf = {
    "provider": "pyscf",
    "params": {
        "method": "DFT",
        "xc": "pbe0",
        "basis": "def2-svp",
        "conv_tol": 1e-10,
        "conv_tol_grad": 1e-6,
        "grid_level": 4,
        "grid_response": True,
        "density_fit": True,
        "warm_start": False,
    },
}
```

`HF` and `DFT` choose RHF/RKS for singlets and UHF/UKS otherwise. Explicit
`RHF`, `UHF`, `RKS` and `UKS` are also accepted. DFT requires an explicit `xc`;
restricted methods require a singlet. Basis, optional ECP, density fitting and
auxiliary basis are delegated to PySCF. The default basis is `def2-svp`; the
appropriate scientific basis/grid/convergence study remains system dependent.

One SCF object is reused, while geometry-dependent integrals and DFT grids are
reset on each geometry change. Analytic nuclear gradients are negated to obtain
forces. DFT grid response is included by default. The adapter checks
`engine.converged` before requesting gradients; unconverged energies cannot be
silently accepted. CPU thread limits belong to the execution configuration,
not to global settings modified inside this adapter.

With `warm_start=True`, converged density matrices are cached by batch index;
the default disables this history dependence. SCF failure clears the cache.
The reported isolated-molecule virial is `sum_i F_i outer r_i`. Periodic PySCF,
post-HF gradients, excited-state dynamics and GPU4PySCF are outside this adapter.

References: [PySCF gradient example](https://github.com/pyscf/pyscf/blob/master/examples/grad/16-scan_force.py)
and [gradient API](https://pyscf.org/pyscf_api_docs/pyscf.grad.html).

## Periodic Torch model contract

The existing nonperiodic `forward(q)` interface remains unchanged. A periodic
model requires `potential.periodic=true` and a nonsingular System cell:

```python
potential = {
    "provider": "torch",
    "factory": "my_models:make_model",
    "device": "cpu",
    "periodic": True,
    "virial": "auto",
    "units": {"positions": "angstrom", "energy": "eV"},
}
```

The model is called positionally as `model(q, cell, pbc)`. Inputs are
float64 `q[B,N,3]`, float64 `cell[B,3,3]` and boolean `pbc[B,3]`, on the selected
device. Cell and coordinates use the declared provider length unit. Models
must honor partial boundary flags, construct any required neighbor lists and
keep frames independent. No cutoff or minimum-image approximation is imposed
by the adapter.

The energy output has shape `[B]` or `[B,1]`. A dictionary may also contain
`forces[B,N,3]` and either `virial[B,3,3]` or tensile-positive
`stress[B,3,3]`. All numerical tensor outputs must be float64. Forces can be
derived from a differentiable energy graph; detached energy requires explicit
forces. Stress is converted to virial in the provider's consistent units.

Virial modes:

| Mode | Behavior |
|---|---|
| `auto` (default) | Use returned virial/stress; otherwise differentiate periodic energy with respect to strain when its graph is available |
| `autograd` | Require a differentiable energy and obtain virial from strain, even if the model also provides one |
| `provided` | Require returned virial or stress |
| `none` | Return no virial |

For autograd, every frame has an independent strain tensor. The adapter uses
`q'=q@(I+strain).T`, `cell'=cell@(I+strain).T` and
`W[a,b]=-dE/dstrain[a,b]`. It includes the cell contribution, which cannot in
general be reconstructed from `sum F outer r` for a periodic system. Energy
must remain differentiable through both coordinates and cell. Automatic mode
returns no virial when a direct-force model's energy is detached and no stress
or virial is supplied; this does not establish pressure support.

A factory returning a Python provider continues through `PythonPotential` and
must declare its own periodic binding or keyword contract. It is not treated as
a Torch tensor model.

## Verification and scope

`tests/test_torch_periodic.py` uses actual Torch CPU differentiation, partial
PBC, a triclinic cell, periodic translations, force finite differences,
all-nine-component strain derivatives, a deliberately nonsymmetric tensor,
stress units and detached-energy failures.

`tests/test_potential_adapters.py` checks named ML loader arguments and model
reuse with dependency doubles, including a real ASE conversion path. These
tests do not validate pretrained MACE, NequIP, DeePMD or fairchem weights.
Optional actual-runtime checks exercise tblite H2 forces, periodic strained-cell
virial and calculator reuse, and PySCF H2/H HF/DFT forces, convergence failure
and object reuse. No GPU speedup or production-model accuracy is established
by these software tests.
