# PyRPMD User Manual

**English** · [中文](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/zh/index.md)

PyRPMD runs ring-polymer molecular dynamics from a single Python or TOML input. It connects
potential energy surfaces to ring-polymer integrators and restartable workflows for gas-phase
reaction rates, direct collisions, free-energy surfaces, enhanced sampling, periodic materials,
surfaces with fixed layers and local electrochemical PCET. An experimental NRPMD extension
handles scattering from a selected incoming electronic state. This manual covers release 0.2.0.

New users: read [Installation](manual/installation.md), the [Quickstart](manual/quickstart.md)
and [Input and command line](manual/input.md). If you already have a system and a potential,
go straight to your task:

| Quantity | `task` | Chapter |
| --- | --- | --- |
| Gas-phase bimolecular RPMD rate | `rate` | [Gas phase and free energy](manual/workflows.md) |
| Direct collisions and product channels | `drpmd` | [Gas phase](manual/workflows.md); [NRPMD](manual/nonadiabatic.md) |
| 1D/2D free-energy surface | `fes` | [Quickstart](manual/quickstart.md), [Gas phase and free energy](manual/workflows.md) |
| Rate matrix for several entrances and transition states | `network` | [Gas phase and free energy](manual/workflows.md) |
| Standard or well-tempered metadynamics | `metadynamics` | [Gas phase and free energy](manual/workflows.md) |
| Equilibrium nuclear quantum statistics and energy estimators | `pimd` | [Materials and surfaces](manual/periodic.md) |
| RPMD self-diffusion, MSD, VACF | `diffusion` | [Materials and surfaces](manual/periodic.md) |
| Directed site-to-site hop in a material | `site_rate` | [Materials and surfaces](manual/periodic.md) |
| Directed escape on a fixed substrate | `surface_rate` | [Materials and surfaces](manual/periodic.md) |
| Electrochemical PCET in the weak vibronic-coupling limit | `pcet` | [PCET](manual/pcet.md) |

The examples use analytic model potentials so they run in seconds. Before a production study,
work through [Scientific conventions and convergence](manual/science.md).

## The basic cycle

Every task uses the same four commands. Give each new calculation its own output directory.

```bash
pyrpmd validate job.inp --check-pes     # check input and units, construct the PES
pyrpmd run job.inp --output runs/job    # run; results in runs/job/results/summary.json
pyrpmd analyze runs/job                 # read the summary; FES, metadynamics, diffusion accept --recompute
pyrpmd resume runs/job                  # continue from checkpoints without changing the science
```

All options are listed in [Input and command line](manual/input.md#command-line-reference). The
Python equivalents are `prepare`, `run`, `analyze_run` and `resume_run`; see the
[Quickstart](manual/quickstart.md).

## Chapters

| Topic | Chapters |
| --- | --- |
| Getting started | [Installation](manual/installation.md) · [Quickstart](manual/quickstart.md) · [Input and CLI](manual/input.md) · [Scientific conventions](manual/science.md) |
| Tasks | [Gas phase and free energy](manual/workflows.md) · [Materials and surfaces](manual/periodic.md) · [PCET](manual/pcet.md) · [Nonadiabatic NRPMD](manual/nonadiabatic.md) · [Examples](manual/examples.md) |
| Models and runs | [Potentials and plugins](manual/potentials.md) · [Execution, parallelism and restart](manual/execution.md) · [Output and analysis](manual/results.md) |
| Dynamics extensions | [Expensive potentials](performance-dynamics.md) · [Trajectory batching](batched-dynamics.md) · [RPC/MTS](rpc-mts.md) · [TRPMD and spectra](trpmd-spectroscopy.md) |
| Reference | [Python API](manual/api.md) · [Troubleshooting](manual/troubleshooting.md) · [Developer guide](manual/development.md) |

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
Expensive potentials <performance-dynamics>
Trajectory batching <batched-dynamics>
RPC/MTS <rpc-mts>
TRPMD and spectra <trpmd-spectroscopy>
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
Dynamics validation <validation/tier1-2026-10-31/README>
PMF diagnostic replay <validation/tier1-2026-10-31/pmf-replay>
```

## About this manual

The manual is available in English and [Chinese](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/zh/index.md)
with matching core chapters. In the HTML build, the language button opens the
corresponding translated page. Extension guides and validation reports currently
available only in English link to the Chinese manual contents instead. Detailed reference specifications, such as the
[input specification](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/input.md) *(Chinese)* and
[public interface conventions](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/interfaces.md),
and the [validation records](https://github.com/chen-yu-hao/PyRPMD/tree/main/docs/validation)
live alongside the manual in the repository.
