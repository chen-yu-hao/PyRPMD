# PyRPMD User Manual

[中文](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/index.md) · **English**

**Start from a runnable input and work through ring-polymer sampling, dynamics and analysis.**

PyRPMD is an extensible Python package for scientific computing. A single Python/TOML input
format connects potential energy surfaces, ring-polymer integrators and restartable workflows.
This manual describes the **0.2.0 integrated release** in the repository: gas-phase reactions,
free energies, periodic materials, fixed-substrate surfaces and local PCET models, plus an
experimental NRPMD extension for scattering from a selected incoming electronic state.

If you are new, read [Installation](manual/installation.md), the [Quickstart](manual/quickstart.md)
and [Input and command line](manual/input.md) in that order. If you already have a system and a
potential, jump to the task you need:

| What do you want to compute? | `task` | Where to read |
| --- | --- | --- |
| Gas-phase bimolecular RPMD rate | `rate` | [Gas phase and free energy](manual/workflows.md) |
| Independent direct collisions and product events | `drpmd` | [Gas-phase workflows](manual/workflows.md); [experimental NRPMD](manual/nonadiabatic.md) |
| 1D/2D equilibrium free-energy surface | `fes` | [Quickstart](manual/quickstart.md), [workflows](manual/workflows.md) |
| Separate rate matrix for several entrances and TSs | `network` | [Gas phase and free energy](manual/workflows.md) |
| Standard / well-tempered metadynamics | `metadynamics` | [Gas phase and free energy](manual/workflows.md) |
| Equilibrium nuclear quantum statistics and energy estimators | `pimd` | [Materials and surfaces](manual/periodic.md) |
| RPMD self-diffusion, MSD, VACF | `diffusion` | [Materials and surfaces](manual/periodic.md) |
| One directed site-to-site hop in a material | `site_rate` | [Materials and surfaces](manual/periodic.md) |
| One directed escape on a fixed substrate | `surface_rate` | [Materials and surfaces](manual/periodic.md) |
| Local electrochemical PCET in the weak vibronic-coupling limit | `pcet` | [PCET](manual/pcet.md) |

The analytic-potential examples are for learning the input, testing algorithms and reproducing
program behavior. Running successfully, passing a numerical check, a physical model being
appropriate, and statistical and discretization convergence are separate questions. Every chapter
explains the units, normalization and limits of its results; read
[Scientific conventions and convergence](manual/science.md) before production work.

## The basic cycle of a calculation

Every task uses the same commands; give each new calculation a new output directory:

```bash
pyrpmd validate job.inp --check-pes     # check input and units, construct the PES
pyrpmd run job.inp --output runs/job    # run; results in runs/job/results/summary.json
pyrpmd analyze runs/job                 # read the summary; FES, metadynamics, diffusion accept --recompute
pyrpmd resume runs/job                  # continue from checkpoints without changing the science
```

See [Input and command line](manual/input.md#command-line-reference) for all options. The same
cycle in Python is `prepare`, `run`, `analyze_run` and `resume_run` (see the
[Quickstart](manual/quickstart.md)).

## Learning paths

The chapters can also be read directly on GitHub:

| Path | Chapters |
| --- | --- |
| Getting started | [Installation](manual/installation.md) · [Quickstart](manual/quickstart.md) · [Input and CLI](manual/input.md) · [Scientific conventions](manual/science.md) |
| Choosing a task | [Gas phase and free energy](manual/workflows.md) · [Materials and surfaces](manual/periodic.md) · [PCET](manual/pcet.md) · [Nonadiabatic NRPMD](manual/nonadiabatic.md) · [Example index](manual/examples.md) |
| Connecting and running models | [Potentials and plugins](manual/potentials.md) · [Execution, parallelism and restart](manual/execution.md) · [Output and analysis](manual/results.md) |
| Reference and development | [Python API](manual/api.md) · [Troubleshooting](manual/troubleshooting.md) · [Developer guide](manual/development.md) |

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

Installation <manual/installation>
Quickstart <manual/quickstart>
Input <manual/input>
Science <manual/science>
```

```{toctree}
:maxdepth: 2
:caption: User guide
:hidden:

Gas phase <manual/workflows>
Materials and surfaces <manual/periodic>
PCET <manual/pcet>
Nonadiabatic NRPMD <manual/nonadiabatic>
Potentials and plugins <manual/potentials>
Execution <manual/execution>
Output and analysis <manual/results>
Examples <manual/examples>
```

```{toctree}
:maxdepth: 2
:caption: Reference and development
:hidden:

Python API <manual/api>
Troubleshooting <manual/troubleshooting>
Developer guide <manual/development>
```

## How this manual is organized

The manual follows the structure of the [PySCF User Guide](https://pyscf.org/user/index.html)
and [Quickstart](https://pyscf.org/quickstart.html): installation and a minimal example,
task-oriented tutorials, the public API and a developer guide. It is built with Sphinx, MyST and
the PyData Sphinx Theme into searchable HTML. The text, examples and interface descriptions are
written from the PyRPMD source code. PySCF is a model for the documentation layout only; it is not
the name of a potential adapter implemented by this project.

The manual exists in Chinese and English with one-to-one chapters. In the HTML build, the
language button at the top right of the navigation bar opens the same page in the other language.

The older [input specification](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/input.md) *(Chinese)*,
[public interface conventions](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/interfaces.md)
and [historical validation records](https://github.com/chen-yu-hao/PyRPMD/tree/main/docs/validation)
remain in the repository. Several of these reference documents are written in Chinese; links to
them from this edition are marked *(Chinese)*. The commit hashes, sample sizes and hardware in a
historical report belong to the run that report describes; they do not automatically validate a
later version or your own system.
