# Input, systems and the command line

## One task, two notations

`.inp` / `.toml` files are TOML; a `.py` file is an ordinary Python module that defines `build()`
and returns a `JobSpec` or a dictionary. Both go through the same parser and become the same
canonical configuration. Python inputs suit parameter generation, batches and importable
plugins; TOML suits a clear record of one specific calculation.

```python
from pathlib import Path
from pyrpmd import JobSpec


def build():
    return JobSpec.from_mapping(
        {
            "task": "pimd",
            "temperature": "300 K",
            "beads": 8,
            "system": {
                "symbols": ["H"],
                "geometry": [[0.0, 0.0, 0.0]],
                "geometry_unit": "bohr",
            },
            "potential": {"provider": "harmonic", "params": {"k": 0.002}},
            "dynamics": {"timestep": "0.1 fs"},
            "sampling": {
                "thermostat": "andersen",
                "equilibration_steps": 100,
                "production_steps": 500,
                "sample_interval": 5,
                "replicas": 2,
            },
        },
        base_directory=Path(__file__).resolve().parent,
    )
```

Save this as `job.py` and run `pyrpmd validate job.py`. The numbers are teaching settings; a
production PIMD run must check equilibration, estimators, time step and bead convergence as
described in the [materials chapter](periodic.md). A Python input executes ordinary Python code;
it is not a declarative data format.

## What relative paths are relative to

| Source of the path | Base for relative paths |
| --- | --- |
| XYZ files, models, `file.py:factory` and the output directory inside an input | The directory of the input file |
| `JobSpec.from_mapping(..., base_directory=...)` | The given directory; the current directory if omitted |
| CLI `--pes`, `--pes-manifest`, `--output` | The directory where the command is started |
| Sources/dependencies in a `.o` link manifest | The directory of the manifest |

For example, `output.directory="runs/manual-quickstart"` in `examples/manual/quickstart.inp`
lives next to that input. The Quickstart passes `--output runs/manual-first` explicitly, which
places the results under `runs/` in the repository root. When you move an input, move the files
it references with it and keep the atom order.

## Common fields

| Field | Convention and effect |
| --- | --- |
| `schema_version` | The current user input version is the integer 1 |
| `task` | See the [task table](../index.md); different tasks allow different sections |
| `preset` | Defaults to `pilot` for ordinary dynamics tasks; `production` requires more explicit fields and still does not guarantee convergence |
| `temperature` | Positive temperature, preferably `"300 K"`; a bare number is in K |
| `beads` | Positive integer; `auto` is the pilot convention P=32 for trial runs |
| `seed` | Non-negative integer; independent tasks derive random streams from stable logical IDs |
| `dynamics.timestep` | Positive step; `"0.1 fs"` carries a unit, a bare number is atomic time |
| `dynamics.integrator` | `cayley-bcb`, `normal-mode-bab`, `velocity-verlet` |
| `sampling` | Equilibration, sampling interval, production length, thermostat, number of replicas or parents |
| `parallel` | Process/MPI topology, bead splitting and thread settings |
| `output.checkpoint_steps` | Positive integer checkpoint interval; the time-based field is `checkpoint_interval` |

Only sections relevant to the task should appear. PCET uses its own `[pcet]` model and scan
settings; do not construct a molecular `system` or RPMD beads for it, see [PCET](pcet.md).
Unknown fields and invalid combinations are errors; a typo is never silently ignored as a comment.

Time-based fields must be integer multiples of the time step. For exact counts use the step
fields supported by the workflow, such as `equilibration_steps`, `production_steps` and
`interval_steps`; when both a time and a step count are given, the step count wins. Do not copy
an interval field from one task to another without checking its meaning: the parent launch
interval in dRPMD and the saved-sample interval in FES control different events.

### Defaults for omitted fields

The `pilot` preset lets you omit the fields below; the program uses these defaults and records
some notes in `input_warnings`. The defaults only let a trial run start; they are not converged
settings. `preset="production"` requires explicit `temperature`, `beads`, `[dynamics]` and
`[sampling]`.

| Field | Default | Note |
| --- | --- | --- |
| `temperature` | `300 K` | The pilot records a note that the default was used |
| `beads` | 32 | `"auto"` also gives 32 with an unconverged warning; production rejects `auto` |
| `seed` | 0 | Independent streams are derived from stable task IDs |
| `system.charge` / `multiplicity` | 0 / 1 | Must match the selected PES |
| `dynamics.timestep` | `0.1 fs` | |
| `dynamics.integrator` | `cayley-bcb` | |
| `dynamics.native` / `loop_backend` / `block_steps` | `true` / `auto` / 256 | Execution settings, see [Execution and parallelism](execution.md) |
| `sampling.thermostat` | `andersen` | `pimd` / `diffusion` default to `pile-l` (Andersen with velocity-Verlet) |
| `sampling.equilibration` / `production` / `interval` | `100 fs` / `200 fs` / `1 fs` | All must be multiples of the step |
| `sampling.andersen_interval` | 10 steps | Steps between Andersen momentum resamplings |
| `sampling.replicas` | 1 | `diffusion` defaults to 4 |
| `output.directory` | `runs/<task>` | Relative to the input file |
| `output.checkpoint_interval` | `10 fs` | |
| `analysis.bins` | 40 for FES; 100 for the rate PMF | |
| `analysis.bootstrap` | 20 | Fixed to 0 for metadynamics; 1 is rejected |
| `analysis.min_count` | 1 | Bins below this count are masked |
| `collision.trajectories` / `max_time` | 4 / `100 fs` | `drpmd` only |
| `collision.sampling` / `bmax` | `thermal_flux` / `auto` = R0/4 | With `auto` no thermal rate is reported |
| `outcomes.exit_distance` / `persistence` | R0 / `1 fs` | `drpmd` only |

`pyrpmd validate INPUT --write-resolved resolved.json` writes the canonical configuration with all
defaults resolved, which is convenient for review and archiving.

### Which sections apply to which tasks

| Section | Used by | Main fields |
| --- | --- | --- |
| `[system]` | All except `pcet` | `geometry`, `transition_state`, `reactants`, `separation`, `symbols`, `masses`, `masses_amu`, `isotopes`, `geometry_unit`, `fragments`, `charge`, `multiplicity`, `cell`, `cell_unit`, `pbc`, `fixed_atoms` |
| `[potential]` | All except `pcet` | `provider`, `factory`, `library`, `manifest`, `params`, `units`, `derivative`, `device`, …; see [Potentials](potentials.md) |
| `[dynamics]`, `[sampling]`, `[parallel]` | All except `pcet` | See the table above and [Execution and parallelism](execution.md) |
| `[[cvs]]`, `[umbrella]`, `[analysis]` | `fes`, `rate`, `network`, `site_rate`, `surface_rate`; `metadynamics` uses `cvs` and `analysis` | See below |
| `[reaction]` | `rate` (`drpmd` may use `electronic_weight`) | `fragments`, `forming_bonds`, `breaking_bonds`, `rinf`, `symmetry_factor`, `equivalent_paths` |
| `[rate]` | `rate` | `pmf_method`, `dividing_surface`, `pmf_search_range`, `compare_ui` |
| `[recrossing]` | `rate`, `network`, `site_rate`, `surface_rate` | Parent count, parent interval, child pairs, child time, bootstrap |
| `[collision]`, `[outcomes]` | `drpmd` | Launch distribution, `bmax`, total trajectories, maximum time; product classification |
| `[nonadiabatic]` | `drpmd` (experimental NRPMD) | See [Nonadiabatic NRPMD](nonadiabatic.md) |
| `[adaptive_umbrella]` | `fes`, `rate` | See [Gas phase and free energy](workflows.md) |
| `[metadynamics]` | `metadynamics` | |
| `[diffusion]`, `[site_rate]`, `[surface]` | The task of the same name (`[surface]` belongs to `surface_rate`) | See [Materials and surfaces](periodic.md) |
| `[network]` | `network` | Entrances, channels, list of TSs |
| `[pcet]` | `pcet` | PCET accepts only `schema_version`, `task`, `preset`, `description`, `temperature`, `pcet`, `output` |
| `[output]` | All | `directory`, `checkpoint_interval` / `checkpoint_steps`, `trajectory_stride` |

The complete fields and combination rules of every section are in the
[input specification](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/input.md) *(Chinese)*.

## Atoms, isotopes and geometry

A non-PCET system is usually declared in one of these ways:

```toml
[system]
geometry = "molecule.xyz"
geometry_unit = "angstrom"
charge = 0
multiplicity = 1
isotopes = [{atom = 0, mass = "2.01410177812 amu"}]
```

Alternatively use inline `symbols` and `geometry=[[x,y,z], ...]`; inline lengths default to bohr
and XYZ files default to angstrom. `geometry_unit` overrides either. Numerical `masses` are in
electron masses and `masses_amu` in atomic mass units. The isotope labels H, D and T keep
different nuclear masses but still map to the nuclear charge of hydrogen when passed to an
electronic potential that needs chemical elements. `charge` and `multiplicity` must match the
selected potential model.

Gas-phase direct collisions use `reactants=["A.xyz", "B.xyz"]` with an explicit `separation`;
a TS entry can use `transition_state="ts.xyz"`. A system cannot obtain all reaction channels or
statistical degeneracies from an XYZ file alone. All atom indices start at **0**, and fragments
must form a complete, non-overlapping partition of the atoms.

Periodic cells, PBC, fixed atoms and unwrapped coordinates are described in
[Materials and surfaces](periodic.md). For new inputs, always state `geometry_unit`, `cell_unit`
and `pbc` explicitly instead of relying on defaults that differ between file sources.

## Collective variables and bias

`[[cvs]]` names the CVs and `umbrella.cvs` selects the coordinates used in the reconstruction.
`centroid` evaluates the coordinate at the centroid; `bead_average` evaluates it on each bead and
averages. The two differ for a nonlinear CV. Custom CVs must return the complete Jacobian; see
[Potentials and plugins](potentials.md).

Numerical umbrella centers use the declared input unit of the CV; strings with units carry their
own unit. Numerical stiffness is in atomic units. Ordinary `distance`, `angle` and
`bond_difference` use unwrapped geometry; minimum images require `periodic_distance`,
`periodic_angle` or `periodic_bond_difference`. Continuous fractional or surface-lateral
coordinates also mean something different from coordinates wrapped into the home cell; they are
not interchangeable.

### CV kinds

`kind` defaults to `distance` and `estimator` defaults to `centroid`. `unit` (or per-component
`units`) only sets the input unit of **numerical** umbrella centers and analysis ranges; CVs work
internally in atomic units.

| `kind` | Required fields | Canonical unit | Notes |
| --- | --- | --- | --- |
| `distance` | `atoms=[i,j]` | bohr | Interatomic distance on unwrapped geometry |
| `angle` | `atoms=[i,j,k]` (j is the vertex) | radian | Three distinct atoms; undefined derivative at a linear geometry |
| `bond_difference` (alias `difference`) | `atoms=[i,j,k]` or `bonds=[[i,j],[k,l]]` | bohr | $r_{ij}-r_{jk}$ or $r_{ij}-r_{kl}$, e.g. a proton-transfer coordinate |
| `cartesian` | `atom`, `axis` as `x`/`y`/`z` or 0/1/2 | bohr | Model systems and restrained sampling |
| `com_distance` | Optional `fragments`, defaults to the system fragments | bohr | Distance between fragment centers of mass; rejected for periodic systems |
| `rpmdrate` (alias `rpmdrate_xi`) | `forming_bonds`, `breaking_bonds`, `rinf` | dimensionless | RPMDrate reaction coordinate used by the rate workflow; rejected for periodic systems |
| `periodic_distance`, `periodic_angle`, `periodic_bond_difference` | As for the non-periodic versions | As above | Minimum image; periodic systems only |
| `projection` | `atom`, `direction`; optional `origin`, `reference_atoms` | bohr | Projection of a continuous centroid on a direction |
| `fractional` | `atom`, `axis`; optional `origin`, `reference_atoms` | dimensionless | Unwrapped fractional coordinate |
| `surface_height` | `atoms`; optional `normal`, `origin`, `reference_atoms` | bohr | Adsorbate-group COM along the surface normal |
| `surface_lateral` | `atoms`; optional `direction`, `origin`, `reference_atoms` | bohr | Continuous in-plane COM projection |
| `python` | `factory`; optional `params` | Declared by the plugin | Custom CV returning a full Jacobian, see [Potentials and plugins](potentials.md) |

With exactly two periodic directions, `surface_height` / `surface_lateral` default to the cross
product of the two lattice vectors as the normal and to the first periodic lattice vector as the
in-plane direction; otherwise they default to z and x.

### Writing umbrella windows

Give the window centers in `[umbrella]` in one of the following ways; `cvs` selects the biased
CVs by name:

```toml
[umbrella]
cvs = ["x"]
centers = [-0.5, 0.0, 0.5]              # explicit list; in 2D use [[x, y], ...]
# start = -1.0, stop = 1.0, count = 9   # evenly spaced, 1D
# grid = [{start = -1.0, stop = 1.0, count = 3},
#         {start = -1.0, stop = 1.0, count = 3}]  # 2D tensor grid; an axis may also be a list
stiffness = "0.002 hartree/bohr^2"      # kappa is accepted as well
```

`centers_au` gives the centers in atomic units and ignores the CV `unit`. `stiffness` may be a
scalar, a diagonal list with one entry per CV, or a full symmetric positive-definite matrix; when
CVs of different dimensions are mixed, write each entry as a string with units.

## Command-line reference

| Command | Purpose | Common options |
| --- | --- | --- |
| `pyrpmd validate INPUT` | Parse the configuration and check the task | `--check-pes`, `--write-resolved FILE` |
| `pyrpmd prepare INPUT` | Show preparation information | `--pilot` allows numerical preparation; `--output FILE` writes JSON |
| `pyrpmd run INPUT` | New run, or restart from the input | `--output DIR`, `--resume` |
| `pyrpmd resume DIR` | Continue from the saved configuration | `--workers N`, `--loop-backend auto`, `--block-steps N` |
| `pyrpmd analyze DIR` | Read results or reanalyze offline | `--recompute`, `--bins`, `--bootstrap`; diffusion options below |
| `pyrpmd checkpoint FILE` | Verify a checkpoint container | `--require-integrity` rejects legacy containers without checksums |
| `pyrpmd info` | Show version and native capabilities | No input file |
| `pyrpmd pes build/inspect/check` | Build and diagnose C ABI potentials | [Potentials](potentials.md) |

`validate`, `prepare` and `run` all accept `--pes`, `--pes-manifest`, `--temperature`,
`--beads` and `--workers`; an override must apply to the task. PCET accepts only
`--temperature` and `--output`; other overrides are errors. Use `pyrpmd COMMAND --help` for the
exact options of the installed version.

Behaviors that are easy to miss:

- `validate --check-pes` and `prepare --pilot` perform the same numerical preparation
  (constructing the PES, TS preparation, PCET eigenstates); without the option only the input is
  checked.
- `--workers N` sets `parallel.trajectory_workers`; for a non-MPI input, N>1 switches to the
  `process` backend.
- `run` refuses to write into an existing non-empty output directory. Continue an interrupted
  task with `pyrpmd resume DIR` or with `pyrpmd run INPUT --resume` on the original input.
- `--bins` and `--bootstrap` of `analyze` imply `--recompute`; offline reanalysis supports
  `fes`, `metadynamics` and `diffusion` only. For other tasks, omit these options to read the
  saved summary.

Diffusion reanalysis accepts `--fit-range '10 fs' '100 fs'`, `--axes 0 1`, `--atoms 0` and
`--max-lag N`, where the maximum lag counts saved frames. These values must fit the saved
duration and the physical fitting window; see [Output and analysis](results.md) for how to read
the results.

The more detailed section specification is in
[docs/input.md](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/input.md) *(Chinese)*.
