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
| Automatic AUS sampling and recrossing | `rate` with `auto = true` | [Automatic rate](automatic-rate.html) |
| Direct collisions and product channels | `drpmd` | [Gas phase](manual/workflows.md); [NRPMD](manual/nonadiabatic.md) |
| Model electronic populations and coherences | `nonadiabatic` | [Population workflow](nonadiabatic-workflow.md) |
| Collisions from selected incoming vibrational states (NE-RPMD, QCT) | `drpmd` + `[initial_ensemble]`; `state-check`, `extend` commands | [Gas phase](manual/workflows.md#selected-vibrational-initial-states) |
| 1D–3D umbrella free-energy surface | `fes` | [Quickstart](manual/quickstart.md), [Unified enhanced sampling](enhanced-sampling.md) |
| Postprocess saved PMF/FES rates and branching | `rate-postprocess` command | [2D rate postprocessing](rate-postprocessing.md) |
| Marginalize a saved FES onto fewer CVs | `project` command | [Output and analysis](manual/results.md) |
| Automatic analysis, publication figures and HTML reports | `report` command, or `run`/`analyze --report` | [Automatic analysis and figures](manual/analysis-reports.md) |
| Rate matrix for several entrances and transition states | `network` | [Gas phase and free energy](manual/workflows.md) |
| Standard or well-tempered metadynamics, shared walkers | `fes` + `method="metadynamics"` | [Unified enhanced sampling](enhanced-sampling.md) |
| OPES probability-based sampling | `fes` + `method="opes"` | [Unified enhanced sampling](enhanced-sampling.md), [OPES](opes.md) |
| Replica-exchange umbrellas | `fes` + `[replica_exchange]` | [Replica exchange](replica-exchange.md) |
| Optional PLUMED centroid biases | `plumed` / Python API | [PLUMED bridge](plumed.md) |
| Transition-interface path ensembles | Python API | [Transition-interface sampling](transition-interface-sampling.md) |
| Equilibrium nuclear quantum statistics and energy estimators | `pimd` | [Materials and surfaces](manual/periodic.md) |
| C++ PIMD driver: SVR, GLE, PIGLET, FFL thermostats; BZP/MTTK/RGB/SCR NPT | `pimd` + `sampling.engine="native"`; `pyrpmd-native` | [Materials and surfaces](manual/periodic.md#native-pimd-engine) |
| RPMD self-diffusion, MSD, VACF | `diffusion` | [Materials and surfaces](manual/periodic.md) |
| Directed site-to-site hop in a material | `site_rate` | [Materials and surfaces](manual/periodic.md) |
| Directed escape on a fixed substrate | `surface_rate` | [Materials and surfaces](manual/periodic.md) |
| Electrochemical PCET in the weak vibronic-coupling limit | `pcet` | [PCET](manual/pcet.md) |
| Constant-potential effective PES, electron-number sampling and HER current | `fes` / Python API | [Electrode models](manual/electrochemistry.md) |
| Adaptive restrained paths with constant-potential thermal NRPMD | Python API | [Adaptive NRPMD paths](manual/adaptive-nrpmd.md) |
| Smooth nonlinear centroid escape coordinate | `site_rate` | [General-coordinate rates](general-coordinate-rates.md) |
| Temperature, isotope and numerical convergence scans | `pyrpmd study` | [Rate studies](rate-studies.md) |
| Instanton paths and one-dimensional low-temperature rates | Python API | [Instantons](instanton.md) |
| Multidimensional instanton rates, splittings and TST/Wigner/Eckart | Python API / cached-Hessian CLI | [Low-temperature cross-checks](low-temperature.md) |
| Thermal NRPMD, surface hopping, SOC, KC-/MV-RPMD, electronic friction | Python API; `diffusion` + `[electronic_friction]` | [Nonadiabatic dynamics](manual/nonadiabatic.md) |
| Ab initio reference data, fitted PES, JaxPIP and native ONNX export | `python -m pyrpmd.pesgen`, Python API | [PES generation](manual/pes-generation.md), [JaxPIP](manual/jaxpip-generation.md) |
| Higher-order equilibrium sampling and quantum estimators | Python API | [Quantum statistics](quantum-statistics.md) |
| Exact quantum references and controlled RPMD response correction | Python API | [Quantum correction](quantum-correction.md) |
| RPMD rate × exact 1–3D subspace quantum correction | Python API | [Subspace rate correction](subspace-correction.md) |
| PA-CMD and relative IR/Raman/SFG spectra | `spectroscopy` | [Spectroscopy](tier2-spectroscopy.md) |
| Suzuki–Chin, bosonic exchange and PIGS | `pimd` + `[path_integral]` | [Path-integral actions](tier2-path-integral.md) |
| Condensed-phase SC energy/Cv/Cp/pressure, SC-NPT, PIGLET-NPT and RPC virial | SC workflow, native PIGLET or Python RPC | [Condensed-phase NQE validation](condensed-phase-nqe.md) |
| Thermal nonadiabatic population-flux calculation | `nonadiabatic-rate` command | [Direct data and rates](tier2-nonadiabatic.md) |
| Method coverage, ML lifecycle and remaining gaps | Input / Python APIs | [Tier 2 overview](tier2-methods.md) |

The examples use analytic model potentials so they run in seconds. Before a production study,
work through [Scientific conventions and convergence](manual/science.md).

For method attribution and research planning, use the [original-paper catalog](manual/references.md)
and [reproduction checklist](manual/reproduction.md), including source systems, target observables and implementation gaps.

## The basic cycle

Most input workflows use the same four commands. PLUMED cannot automatically restore its
opaque kernel state; the thermal nonadiabatic rate command uses a separate input schema.
Give each new calculation its own output directory.
To create a ready input first, use `pyrpmd init my-first-run` and follow the
[Quickstart](manual/quickstart.md). In the template below, `job.inp` is your own input file.

```bash
pyrpmd validate job.inp --check-pes     # check input and units, construct the PES
pyrpmd run job.inp --output runs/job    # run; results in runs/job/results/summary.json
pyrpmd analyze runs/job                 # read the summary; FES methods, diffusion and nonadiabatic populations accept --recompute
pyrpmd resume runs/job                  # continue from checkpoints without changing the science
pyrpmd report runs/job                  # optional: figures, index.html and report.json from saved data
```

The command index and common options are in [Input and command line](manual/input.md#command-line-reference);
`pyrpmd COMMAND --help` lists all options of the installed version. The
Python equivalents are `prepare`, `run`, `analyze_run` and `resume_run`; see the
[Quickstart](manual/quickstart.md).

## Chapters

| Topic | Chapters |
| --- | --- |
| Getting started | [Installation](manual/installation.md) · [Quickstart](manual/quickstart.md) · [Input and CLI](manual/input.md) · [Scientific conventions](manual/science.md) |
| Tasks | [Gas phase and free energy](manual/workflows.md) · [Materials and surfaces](manual/periodic.md) · [PCET](manual/pcet.md) · [Electrode models](manual/electrochemistry.md) · [Nonadiabatic NRPMD](manual/nonadiabatic.md) · [Examples](manual/examples.md) |
| Models and runs | [Potentials and plugins](manual/potentials.md) · [PES generation](manual/pes-generation.md) · [JaxPIP fitting](manual/jaxpip-generation.md) · [Execution, parallelism and restart](manual/execution.md) · [Output and analysis](manual/results.md) · [Automatic analysis and figures](manual/analysis-reports.md) |
| Dynamics extensions | [Common propagation stack](propagation-stack.md) · [Expensive potentials](performance-dynamics.md) · [Trajectory batching](batched-dynamics.md) · [RPC/MTS](rpc-mts.md) · [Mixed sampling](mixed-sampling.md) · [TRPMD and spectra](trpmd-spectroscopy.md) |
| Reference | [Original papers](manual/references.md) · [Reproduction checklist](manual/reproduction.md) · [Python API](manual/api.md) · [Troubleshooting](manual/troubleshooting.md) · [Developer guide](manual/development.md) |

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
Rate PMF selection <rate-pmf-selection>
Explicit rate input and regional umbrellas <rate-input-v2>
Automatic AUS rate calculations <automatic-rate>
Transition-state search <transition-state-search>
Materials and surfaces <manual/periodic>
PCET <manual/pcet>
Electrode models <manual/electrochemistry>
Nonadiabatic dynamics <manual/nonadiabatic>
Nonadiabatic population workflow <nonadiabatic-workflow>
Nonadiabatic models <nonadiabatic-models>
Nonadiabatic ensembles <nonadiabatic-ensembles>
Nonadiabatic comparison and gaps <nonadiabatic-comparison>
SM-NRPMD and RP-MASH <nonadiabatic-spin-methods>
BA/DM hopping, DBOC and SOC <nonadiabatic-hopping-extensions>
Molecular multistate potentials <molecular-nonadiabatic>
Electronic propagation performance <nonadiabatic-performance>
Potentials and plugins <manual/potentials>
PES generation <manual/pes-generation>
JaxPIP fitting and native execution <manual/jaxpip-generation>
Execution <manual/execution>
Propagation stack <propagation-stack>
Expensive potentials <performance-dynamics>
Trajectory batching <batched-dynamics>
RPC/MTS <rpc-mts>
Mixed sampling <mixed-sampling>
Condensed-phase nuclear quantum effects <condensed-phase-nqe>
TRPMD and spectra <trpmd-spectroscopy>
Output and analysis <manual/results>
Standard trajectory formats <trajectory-export>
Bond constraints <bond-constraints>
rate-postprocessing
Automatic analysis and figures <manual/analysis-reports>
Examples <manual/examples>
General-coordinate rates <general-coordinate-rates>
Rate studies <rate-studies>
Instantons <instanton>
Low-temperature cross-checks <low-temperature>
Quantum statistics <quantum-statistics>
Quantum correction <quantum-correction>
Subspace rate correction <subspace-correction>
Subspace transfer theory <subspace-transfer-theory>
Tier 2 coverage <tier2-methods>
PA-CMD and property spectra <tier2-spectroscopy>
Main-engine path-integral actions <tier2-path-integral>
Direct nonadiabatic data and rates <tier2-nonadiabatic>
ML model lifecycle <tier2-ml-lifecycle>
```

```{toctree}
:maxdepth: 1
:caption: Enhanced sampling
:hidden:

Andersen collision timing <andersen-sampling>
Unified FES and enhanced sampling <enhanced-sampling>
Replica exchange <replica-exchange>
OPES <opes>
PLUMED bridge <plumed>
Transition-interface sampling <transition-interface-sampling>
Fixed-ring committor analysis <committor>
Adaptive constant-potential NRPMD paths <manual/adaptive-nrpmd>
Extended CVs and sampling methods <tier2-sampling>
```

```{toctree}
:maxdepth: 2
:caption: Reference and development
:hidden:

Original papers and implementation <manual/references>
Paper systems and reproduction <manual/reproduction>
Python API <manual/api>
Troubleshooting <manual/troubleshooting>
Developer guide <manual/development>
Distribution and releases <distribution>
Literature validation <literature-validation>
Publication readiness <publication-readiness>
Dynamics validation <validation/tier1-2026-10-31/README>
PMF diagnostic replay <validation/tier1-2026-10-31/pmf-replay>
```

## About this manual

The manual is available in English and [Chinese](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/zh/index.md)
with matching core chapters. In the HTML build, the language button opens the
corresponding translated page. Extension guides and validation reports currently
available only in English link to the Chinese manual contents instead. Detailed reference specifications, such as the
[input specification](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/input.md) and
[public interface conventions](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/interfaces.md),
and the [validation records](https://github.com/chen-yu-hao/PyRPMD/tree/main/docs/validation)
live alongside the manual in the repository.
