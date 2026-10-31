# Input, systems and the command line

## TOML or Python

An input is either a TOML file (`.inp` or `.toml`) or a Python module (`.py`) that defines `build()`
and returns a `JobSpec` or a dictionary. Both go through the same parser. TOML gives a compact record
of one calculation; Python suits parameter scans, batches and importable plugins.

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

Save this as `job.py` and run `pyrpmd validate job.py`. Python inputs run as ordinary, trusted
Python code. Production PIMD settings are covered in [Materials and surfaces](periodic.md).

## Relative paths

| Path given in | Relative to |
| --- | --- |
| An input file (XYZ files, models, `file.py:factory`, output directory) | The directory of the input file |
| `JobSpec.from_mapping(..., base_directory=...)` | `base_directory`, or the current directory if omitted |
| CLI `--pes`, `--pes-manifest`, `--output` | The directory where the command runs |
| A `.o` link manifest | The directory of the manifest |

For example, `output.directory="run"` in the starter `my-first-run/quickstart.inp` resolves to
`my-first-run/run`, while CLI `--output runs/custom` resolves from the current directory.
`validate INPUT --output runs/custom` shows the resolved input and output paths without creating
the run directory. Move referenced files together with the input and keep the atom order.

## Common fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Input schema version, currently the integer 1 |
| `task` | One of the tasks in the [task table](../index.md); each task accepts its own sections |
| `preset` | `pilot` (default) or `production`; `production` requires explicit settings |
| `temperature` | Positive temperature, e.g. `"300 K"`; a bare number is in K |
| `beads` | Positive integer; `auto` gives P=32 in a pilot |
| `seed` | Non-negative integer; independent streams derive from stable task IDs |
| `dynamics.timestep` | Positive step, e.g. `"0.1 fs"`; a bare number is in atomic time |
| `dynamics.integrator` | `cayley-bcb`, `normal-mode-bab` or `velocity-verlet` |
| `sampling` | Equilibration, sampling interval, production length, thermostat, replicas or parents |
| `parallel` | Process/MPI topology, bead splitting and threads |
| `output.checkpoint_steps` | Checkpoint interval in steps; `checkpoint_interval` takes a time |
| `output.trajectory_storage` / `chunk_frames` | Materials trajectory storage: `"stream"` (default) or `"memory"`; default chunk buffer 256 frames |
| `barostat` | Optional BZP NPT settings for fully periodic, all-mobile `pimd`: pressure, time constant and friction |

Each task accepts only its own sections. PCET uses a separate `[pcet]` model with no molecular
`system` or beads; see [PCET](pcet.md). Unknown fields and invalid combinations raise an error.

Times must be integer multiples of the time step. For exact counts use the step fields, such as
`equilibration_steps`, `production_steps` and `interval_steps`; a step count takes precedence over
the matching time. Interval fields belong to their task: in dRPMD the interval is the time between
two launches from a parent chain, in FES it is the time between saved samples.

### Defaults

A `pilot` input may omit the fields below; PyRPMD fills in these defaults and records notes in
`input_warnings`. `preset="production"` requires explicit `temperature`, `beads`, `[dynamics]`
and `[sampling]`. It checks input completeness, not scientific convergence, and does not select
a higher-accuracy parameter set. Omitted entries inside those tables still receive defaults.
Specify the physical model and sampling budgets explicitly for a production study.

| Field | Default | Note |
| --- | --- | --- |
| `temperature` | `300 K` | Noted in `input_warnings` |
| `beads` | 32 | `"auto"` also gives 32, with a warning; production rejects `auto` |
| `seed` | 0 | Independent streams derive from stable task IDs |
| `system.charge` / `multiplicity` | 0 / 1 | Must match the selected PES |
| `dynamics.timestep` | `0.1 fs` | Check the reported observable again at half the step |
| `dynamics.integrator` | `cayley-bcb` | |
| `dynamics.native` / `loop_backend` / `block_steps` | `true` / `auto` / 256 | See [Execution and parallelism](execution.md) |
| `sampling.thermostat` | `andersen` | `pile-l` for `pimd` / `diffusion` (Andersen with velocity-Verlet) |
| `sampling.equilibration` / `production` / `interval` | `100 fs` / `200 fs` / `1 fs` | Enabled adaptive umbrellas and metadynamics default to `1000 fs` production; explicit lengths override this |
| `sampling.andersen_interval` | 10 steps | Steps between Andersen momentum resamplings |
| `sampling.replicas` | 1 | 4 for `diffusion` |
| `output.directory` | `runs/<task>` | Relative to the input file |
| `output.checkpoint_interval` | `10 fs` | |
| `analysis.bins` | 40 for FES; 100 for the rate PMF | |
| `analysis.bootstrap` | 20 | A short analysis budget, not a precision target; 0 for metadynamics and OPES; 1 is rejected |
| `analysis.min_count` | 1 | Bins with fewer samples are masked |
| `collision.trajectories` / `max_time` | 4 / `100 fs` | `drpmd` only |
| `collision.sampling` / `bmax` | `thermal_flux` / `auto` = R0/4 | With `auto`, no thermal rate is reported |
| `outcomes.exit_distance` / `persistence` | R0 / `1 fs` | `drpmd` only |

`pyrpmd validate INPUT --write-resolved resolved.json` writes the resolved front-end input.
Some workflow settings, including automatically prepared rate windows and adaptive plans, are
resolved at runtime and recorded in their sampling outputs.

The gas-phase `rate` workflow supplies these additional starting values when omitted:

| Field | Default | Meaning |
| --- | --- | --- |
| `rate.pmf_method` | `ui` | Gaussian umbrella integration; `mbar` is an explicit alternative |
| Initial umbrella centers | 23 evenly spaced points from -0.05 to 1.05 | Step 0.05 in the dimensionless reaction coordinate |
| Umbrella stiffness | `50 / beta` | Physical harmonic stiffness, i.e. 50 kBT for the dimensionless coordinate |
| `rate.pmf_points` | 401 | UI integration grid, not a number of sampled windows |
| `recrossing.parent_replicas` / `parent_samples` | 2 / 4 | `sampling.parents`, if supplied, replaces the default parent-sample count |
| `recrossing.equilibration` / `parent_interval` / `child_time` | `20 fs` / `5 fs` / `50 fs` | Must be converged for the reaction |
| `recrossing.child_pairs` / `bootstrap` | 2 / 100 | Per parent / number of resampling draws |

These are not the RPMDrate manual's production examples. The initial window spacing and stiffness
need system-specific checks, especially with the default Gaussian UI model. Adaptive placement
can add centers but keeps the supplied stiffness fixed.

For enabled `[adaptive_umbrella]`, the following are **pilot selection criteria and resource
limits**, not guarantees about the production PMF or rate:

| Field | Default | Meaning |
| --- | --- | --- |
| `pilot_steps` | 1000 | Per window and replica; 100 fs at the default time step |
| `pilot_equilibration_steps` | `sampling.equilibration_steps` | 1000 steps / 100 fs with otherwise default input |
| `max_pilot_steps` | `16 * pilot_steps` | Per window and replica; 1.6 ps at the default time step |
| `max_rounds` | 8 | Maximum pilot rounds |
| `max_windows` | 128 | Insertion cap; longer pilots may still satisfy ESS/coverage at this count |
| `target_overlap` | 0.03 | Normalized overlap score in new schema-3 plans; legacy plans retain their metric |
| `min_effective_samples` | 20 | Per-window correlation-corrected ESS, summed over independent replicas |
| `min_spacing` | `1e-6` | Distance in the dimensionless `sqrt(beta * delta.T @ K @ delta)` metric |
| `coverage_bins` / `coverage_ranges` | Omitted | Spatial coverage is not checked unless both are supplied |
| `min_bin_effective_samples` / `target_coverage` | 5 / 0.95 | Apply only to an explicitly enabled coverage grid |

At the default 1 fs save interval, the first 100 fs pilot stores only about 100 frames per
window/replica, before accounting for correlation. Passing ESS 20 is a screening result. Use
independent production diagnostics and the [convergence procedure](science.md#assessing-adaptive-umbrella-results)
to choose lengths, replicas, bead number and uncertainty precision for the actual system.

### Sections by task

| Section | Used by | Main fields |
| --- | --- | --- |
| `[system]` | All except `pcet` | `geometry`, `transition_state`, `reactants`, `separation`, `symbols`, `masses`, `masses_amu`, `isotopes`, `geometry_unit`, `fragments`, `charge`, `multiplicity`, `cell`, `cell_unit`, `pbc`, `fixed_atoms` |
| `[potential]` | All except `pcet` | `provider`, `factory`, `library`, `manifest`, `params`, `units`, `derivative`, `device`, `abi_version`, `periodic`, `virial`, …; see [Potentials](potentials.md) |
| `[dynamics]`, `[sampling]`, `[parallel]` | All except `pcet` | See above and [Execution and parallelism](execution.md) |
| `[[cvs]]`, `[umbrella]`, `[analysis]` | `fes`, `rate`, `network`, `site_rate`, `surface_rate`; `metadynamics` and `opes` use `cvs` and `analysis` | See below |
| `[reaction]` | `rate` (`drpmd` may use `electronic_weight`) | `fragments`, `forming_bonds`, `breaking_bonds`, `rinf`, `symmetry_factor`, `equivalent_paths` |
| `[rate]` | `rate` | `pmf_method`, `dividing_surface`, `pmf_search_range`, `compare_ui` |
| `[recrossing]` | `rate`, `network`, `site_rate`, `surface_rate` | Parent count, parent interval, child pairs, child time, bootstrap |
| `[collision]`, `[outcomes]` | `drpmd` | Launch distribution, `bmax`, total trajectories, maximum time; product classification |
| `[nonadiabatic]` | `drpmd` (experimental NRPMD) | See [Nonadiabatic NRPMD](nonadiabatic.md) |
| `[electronic_friction]` | `diffusion` | `method = "centroid"`, `provider`, `gamma` or `tensor`, or Python `factory` and `params`; see [electronic friction](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/electronic-friction.md) |
| `[initial_ensemble]` | `drpmd` (experimental nuclear state selection) | `method = "ne_rpmd"` or `"qct"`, explicit NE `beta`, fragment Hessians/modes and `vibrational_quanta`; see [Direct collisions](workflows.md) |
| `[adaptive_umbrella]` | `fes`, `rate` | See [Gas phase and free energy](workflows.md) |
| `[metadynamics]` | `metadynamics` | `grid.ranges` / `grid.bins` optionally select a bias grid |
| `[opes]` | `opes` | Probability-bias widths, barrier, update pace and optional freeze/relaxation steps; [OPES guide](../opes.md) |
| `[replica_exchange]` | `fes` | Fixed-window same-temperature exchange interval and optional graph; [REUS guide](../replica-exchange.md) |
| `[barostat]` | `pimd` | `method="bzp"`, `pressure`, `time_constant`, `friction` |
| `[diffusion]`, `[site_rate]`, `[surface]` | The task of the same name (`[surface]` belongs to `surface_rate`) | See [Materials and surfaces](periodic.md) |
| `[network]` | `network` | Entrances, channels, transition states |
| `[pcet]` | `pcet` | PCET accepts only `schema_version`, `task`, `preset`, `description`, `temperature`, `pcet`, `output` |
| `[output]` | All | `directory`, `checkpoint_interval` / `checkpoint_steps`, `trajectory_stride` |

Every field and combination rule is in the
[input specification](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/input.md) *(Chinese)*.

## Atoms, isotopes and geometry

A typical `[system]`:

```toml
[system]
geometry = "molecule.xyz"
geometry_unit = "angstrom"
charge = 0
multiplicity = 1
isotopes = [{atom = 0, mass = "2.01410177812 amu"}]
```

You can also give inline `symbols` and `geometry=[[x,y,z], ...]`. Inline lengths default to bohr
and XYZ files to angstrom; `geometry_unit` sets either explicitly. `masses` are in electron masses,
`masses_amu` in atomic mass units. H, D and T carry their own nuclear masses and map to hydrogen for
electronic potentials. `charge` and `multiplicity` must match the potential model.

Direct collisions take `reactants=["A.xyz", "B.xyz"]` with an explicit `separation`; a rate
calculation can start from `transition_state="ts.xyz"`. Reaction channels and statistical
degeneracies are defined in the input, not inferred from the XYZ files. Atom indices start at **0**,
and fragments must partition all atoms without overlap.

Periodic cells, PBC, fixed atoms and unwrapped coordinates are covered in
[Materials and surfaces](periodic.md). Always state `geometry_unit`, `cell_unit` and `pbc`
explicitly in new inputs.

## Collective variables and bias

`[[cvs]]` defines named CVs and `umbrella.cvs` selects the ones to bias. `centroid` evaluates the
coordinate at the ring centroid; `bead_average` evaluates it on each bead and averages, which
differs for nonlinear CVs. Custom CVs return the complete Jacobian; see
[Potentials and plugins](potentials.md).

Numerical umbrella centers use the CV's declared input unit; strings with units carry their own.
Numerical stiffness is in atomic units. `distance`, `angle` and `bond_difference` act on unwrapped
geometry; for minimum-image geometry use `periodic_distance`, `periodic_angle` or
`periodic_bond_difference`. Continuous `fractional` and `surface_lateral` coordinates are unwrapped
and distinguish neighboring cells.

### CV kinds

`kind` defaults to `distance` and `estimator` to `centroid`. `unit` (or per-component `units`) sets
the unit of **numerical** umbrella centers and analysis ranges; CVs compute in atomic units.

| `kind` | Required fields | Canonical unit | Notes |
| --- | --- | --- | --- |
| `distance` | `atoms=[i,j]` | bohr | Interatomic distance on unwrapped geometry |
| `angle` | `atoms=[i,j,k]` (j is the vertex) | radian | Three distinct atoms; the derivative is undefined at a linear geometry |
| `bond_difference` (alias `difference`) | `atoms=[i,j,k]` or `bonds=[[i,j],[k,l]]` | bohr | $r_{ij}-r_{jk}$ or $r_{ij}-r_{kl}$, e.g. a proton-transfer coordinate |
| `cartesian` | `atom`, `axis` as `x`/`y`/`z` or 0/1/2 | bohr | Model systems and restrained sampling |
| `com_distance` | Optional `fragments`, defaults to the system fragments | bohr | Distance between fragment centers of mass; molecular systems only |
| `rpmdrate` (alias `rpmdrate_xi`) | `forming_bonds`, `breaking_bonds`, `rinf` | dimensionless | RPMDrate reaction coordinate of the rate workflow; molecular systems only |
| `periodic_distance`, `periodic_angle`, `periodic_bond_difference` | As the non-periodic versions | As above | Minimum image; periodic systems only |
| `projection` | `atom`, `direction`; optional `origin`, `reference_atoms` | bohr | Continuous centroid projected on a direction |
| `fractional` | `atom`, `axis`; optional `origin`, `reference_atoms` | dimensionless | Unwrapped fractional coordinate |
| `surface_height` | `atoms`; optional `normal`, `origin`, `reference_atoms` | bohr | Adsorbate-group COM along the surface normal |
| `surface_lateral` | `atoms`; optional `direction`, `origin`, `reference_atoms` | bohr | Continuous in-plane COM projection |
| `python` | `factory`; optional `params` | Declared by the plugin | Custom CV with a full Jacobian; see [Potentials and plugins](potentials.md) |

With exactly two periodic directions, `surface_height` uses the cross product of the two lattice
vectors as the default normal and `surface_lateral` the first periodic lattice vector as the default
direction; otherwise they default to z and x.

### Umbrella windows

Give the window centers in one of these forms; `cvs` names the biased CVs:

```toml
[umbrella]
cvs = ["x"]
centers = [-0.5, 0.0, 0.5]              # explicit list; in 2D use [[x, y], ...]
# start = -1.0, stop = 1.0, count = 9   # evenly spaced, 1D
# grid = [{start = -1.0, stop = 1.0, count = 3},
#         {start = -1.0, stop = 1.0, count = 3}]  # 2D tensor grid; an axis may also be a list
stiffness = "0.002 hartree/bohr^2"      # kappa is accepted as well
```

`centers_au` gives centers in atomic units and ignores the CV `unit`. `stiffness` is a scalar, a
diagonal list with one entry per CV, or a full symmetric positive-definite matrix; for CVs of
different dimensions, write each entry as a string with units.

## Command-line reference

| Command | Purpose | Common options |
| --- | --- | --- |
| `pyrpmd init DIR` | Write the bundled harmonic FES starter in a new or empty directory | Prints the input path and next commands |
| `pyrpmd validate INPUT` | Parse the configuration and check the task | `--check-pes`, `--output DIR`, `--write-resolved FILE` |
| `pyrpmd prepare INPUT` | Show preparation information | `--pilot` runs the numerical preparation; `--output FILE` writes JSON |
| `pyrpmd run INPUT` | Start a run, or continue one from its input | `--output DIR`, `--resume` |
| `pyrpmd resume DIR` | Continue from the saved configuration | `--workers N`, `--loop-backend auto`, `--block-steps N` |
| `pyrpmd extend DIR` | Raise an existing dRPMD ensemble's total time budget for unresolved trajectories | `--max-time '20 ps'` (required; total time from launch) |
| `pyrpmd state-check INPUT` | Evolve prepared nuclear states without a collision kick and report harmonic-mode stability | `--time '100 fs'`, `--samples 8`, `--stride 10`, `--output report.json` |
| `pyrpmd analyze DIR` | Read results or reanalyze offline | `--recompute`, `--bins`, `--bootstrap`; diffusion options below |
| `pyrpmd report RUN [RUN ...]` | Generate figures and an HTML report from saved results | `--output DIR`, `--style`, `--formats`, `--overwrite`; see [Reports](analysis-reports.md) |
| `pyrpmd project DIR` | Marginalize a saved FES over omitted CV axes | Required `--axes` and `--output`; optional `--selection` |
| `pyrpmd rate-postprocess DIR` | Recompute rates from saved PMF/FES and signed fluxes | `--spec FILE`, `--output FILE`, `--bootstrap`; see [Rate postprocessing](../rate-postprocessing.md) |
| `pyrpmd study INPUT` | Plan or run temperature, isotope and convergence scans | `--plan`, `--output DIR`, `--resume`; see [Rate studies](../rate-studies.md) |
| `pyrpmd checkpoint FILE` | Verify a checkpoint container | `--require-integrity` rejects legacy containers without checksums |
| `pyrpmd info` | Show version and native capabilities | |
| `pyrpmd pes build/inspect/check` | Build and test C ABI potentials | See [Potentials](potentials.md) |

`validate`, `prepare` and `run` accept the overrides `--pes`, `--pes-manifest`, `--temperature`,
`--beads` and `--workers`, where they apply to the task; PCET accepts only `--temperature` and
`--output`. `pyrpmd COMMAND --help` lists the options of the installed version.

- `validate --check-pes` and `prepare --pilot` run the same numerical preparation: constructing
  the PES, TS preparation, or PCET eigenstates. Without the option, only the input is checked.
- `validate --output DIR` selects the destination to show in its report; it does not create a run.
  `prepare --output FILE` instead writes a preparation JSON file. `--write-resolved FILE` explicitly
  writes the resolved configuration from `validate`.
- `--workers N` sets `parallel.trajectory_workers`; for non-MPI inputs, N>1 selects the `process`
  backend.
- `run` refuses an existing non-empty output directory. Continue a run with `pyrpmd resume DIR`
  or with `pyrpmd run INPUT --resume`.
- `--bins` and `--bootstrap` imply `--recompute`. Offline reanalysis is available for `fes`,
  `metadynamics`, `opes` and `diffusion`; for other tasks, `analyze` reads the saved summary.

Diffusion reanalysis accepts `--fit-range '10 fs' '100 fs'`, `--axes 0 1`, `--atoms 0` and
`--max-lag N`, where the lag counts saved frames. The fit range must lie within the saved duration;
see [Output and analysis](results.md).

### Terminal output and debugging

`python -m pyrpmd` accepts the same commands as the `pyrpmd` executable. Use `pyrpmd --help` for
the command list and `pyrpmd COMMAND --help` for a command's options. Command results are JSON
on standard output; expected input/runtime errors go to standard error and return exit code 2.
A successful command returns 0. This is an execution status: still inspect `ready`, `warnings`,
`converged` and task-specific diagnostics before interpreting a scientific result.

For an actionable error message, run the normal command. When reporting an unexpected failure,
add `--debug` to include its traceback, for example:

```bash
pyrpmd validate my-first-run/quickstart.inp --check-pes --debug
pyrpmd info > environment.json
```

`--debug` is accepted before or after the subcommand. Save the command, input and environment
report with the error; the [troubleshooting guide](troubleshooting.md) explains recovery.

The full section reference is in
[docs/input.md](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/input.md) *(Chinese)*.
