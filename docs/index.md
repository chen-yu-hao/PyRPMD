# PyRPMD User Manual

**English** · [中文](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/zh/index.md)

PyRPMD connects potential energy surfaces to ring-polymer molecular dynamics,
reaction-rate calculations, free-energy sampling, quantum statistics and
spectroscopy. Start with [Installation](manual/installation.md), the
[Quickstart](manual/quickstart.md), and [Input and command line](manual/input.md).

This is the manual for the source checkout. The package version is still 0.2.0;
[the next-release notes](releases/unreleased.md) distinguish newly integrated
features from the [0.2.0 release](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/releases/0.2.0.md).
The [capabilities and limits](manual/capabilities.md) page explains supported
combinations and the evidence behind them. Short examples exercise an interface;
production calculations also need [scientific convergence](manual/science.md).

## Find your workflow

| Goal | Entry point | Guide |
| --- | --- | --- |
| Thermal gas-phase RPMD rates and reaction networks | `rate`, `network` | [Gas phase and free energy](manual/workflows.md) · [Explicit rate input](rate-input-v2.md) · [PMF selection](rate-pmf-selection.md) |
| Budgeted rates with declared statistical and numerical targets | `pyrpmd precision plan/run/status` | [Precision-driven rate control](rate_precision.md) |
| Instanton rates, tunnelling splittings and normal-regime golden-rule rates | `pyrpmd instanton`, Python API | [Instanton workflow](instanton.md) · [Low-temperature normalization](low-temperature.md) |
| Thermal or selected-vibrational-state collisions, NE-RPMD and QCT | `drpmd`, `state-check`, `extend` | [Direct collisions](manual/workflows.md#selected-vibrational-initial-states) |
| Site hopping and escape from a fixed substrate | `site_rate`, `surface_rate` | [Materials and surfaces](manual/periodic.md) · [General-coordinate rates](general-coordinate-rates.md) |
| Umbrella FES, metadynamics, OPES and replica exchange | `fes` | [Enhanced sampling](enhanced-sampling.md) · [Replica exchange](replica-exchange.md) · [OPES](opes.md) |
| ABF, eABF and steered free-energy differences | `fes` with `abf`, `eabf`, `steered` | [Enhanced sampling](enhanced-sampling.md) |
| Finite-temperature strings and learned CVs | Python API / exported CV | [String sampling](finite-temperature-string.md) · [mlcolvar](mlcolvar.md) |
| PLUMED biases with declared file-based restart | `plumed` | [PLUMED bridge](plumed.md) |
| TIS, RETIS minus-ensemble flux, REPPTIS and direct FFS | Python API | [Path sampling](transition-interface-sampling.md) |
| Equilibrium PIMD, native thermostats and barostats | `pimd`, `sampling.engine="native"`, `pyrpmd-native` | [Native workflow](native-pimd-workflow.md) · [Native ensembles](native-ensembles.md) |
| GLE/PIGLET parameter import, generation and estimator checks | Python API / thermostat parameters | [Condensed-phase quantum statistics](condensed-phase-nqe.md) |
| Device-resident PIMD and RPMD transport | `pimd` / `diffusion`, `sampling.engine="torch"` | [Tensor workflows](torch-materials-workflows.md) |
| Suzuki–Chin, bosonic/fermionic exchange, periodic winding and exchange NPT, PIGS | `pimd` + `[path_integral]` | [Path-integral actions](tier2-path-integral.md) · [Quantum estimators](quantum-statistics.md) |
| Self-diffusion, MSD and VACF | `diffusion` | [Materials](manual/periodic.md) · [Common propagation](propagation-stack.md) |
| TRPMD, PA-CMD, quadratic f-CMD and experimental molecular f-QCMD spectra | `spectroscopy` | [Property spectra](tier2-spectroscopy.md) · [f-QCMD](qcmd.md) · [TRPMD](trpmd-spectroscopy.md) |
| Mapping dynamics, surface hopping, thermal rates and electronic friction | `nonadiabatic`, `nonadiabatic-rate`, Python API | [Population workflow](nonadiabatic-workflow.md) · [Method comparison](nonadiabatic-comparison.md) |
| Classical IESH and multistate MASH | `nonadiabatic`, P=1 | [IESH and multistate MASH](nonadiabatic-iesh-multistate.md) |
| PySCF SA-CASSCF electronic data and spin–boson HEOM references | Python API | [Electronic structure and HEOM](tier2-nonadiabatic.md) |
| PCET, constant-potential models and restrained nonadiabatic paths | `pcet`, `fes`, Python API | [PCET](manual/pcet.md) · [Electrochemistry](manual/electrochemistry.md) · [Adaptive paths](manual/adaptive-nrpmd.md) |
| Batched ML potentials, OpenMM and LAMMPS | Potential providers | [Batched ML](batched-ml-potentials.md) · [Provider interfaces](manual/potentials.md) · [Measured performance and limits](t19-performance.md) |
| Fast analytic potentials, trajectory batches and RPC/MTS | Native bindings / Python API | [Built-in native potentials](builtin-native-potentials.md) · [Batching](batched-dynamics.md) · [RPC/MTS](rpc-mts.md) |
| PES fitting and resumable sampling–labelling–training loops | `python -m pyrpmd.pesgen`, Python API | [PES generation](manual/pes-generation.md) · [JaxPIP](manual/jaxpip-generation.md) · [Active learning](tier2-ml-lifecycle.md) |
| Reanalyse FES, postprocess rates, export trajectories and make reports | `analyze`, `project`, `rate-postprocess`, `report` | [Results](manual/results.md) · [Rate postprocessing](rate-postprocessing.md) · [Trajectory export](trajectory-export.md) · [Reports](manual/analysis-reports.md) |

For attribution, use the [original-paper catalog](manual/references.md).
The [reproduction checklist](manual/reproduction.md) retains its source-pinned
research plans; current supported combinations are in the linked guides.

## The basic cycle

Most input workflows use the same commands. PLUMED automatic continuation requires
`restart="files"` before the first run; precision campaigns and thermal nonadiabatic
rates use their own commands and schemas.
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
| Getting started | [Installation](manual/installation.md) · [Quickstart](manual/quickstart.md) · [Input and CLI](manual/input.md) · [Scientific conventions](manual/science.md) · [Capabilities and limits](manual/capabilities.md) |
| Tasks | [Gas phase and free energy](manual/workflows.md) · [Materials and surfaces](manual/periodic.md) · [PCET](manual/pcet.md) · [Electrode models](manual/electrochemistry.md) · [Nonadiabatic NRPMD](manual/nonadiabatic.md) · [Examples](manual/examples.md) |
| Models and runs | [Potentials and plugins](manual/potentials.md) · [PES generation](manual/pes-generation.md) · [JaxPIP fitting](manual/jaxpip-generation.md) · [Execution, parallelism and restart](manual/execution.md) · [Output and analysis](manual/results.md) · [Automatic analysis and figures](manual/analysis-reports.md) |
| Dynamics extensions | [Common propagation stack](propagation-stack.md) · [Expensive potentials](performance-dynamics.md) · [Trajectory batching](batched-dynamics.md) · [RPC/MTS](rpc-mts.md) · [Mixed sampling](mixed-sampling.md) · [TRPMD and spectra](trpmd-spectroscopy.md) |
| Reference | [Original papers](manual/references.md) · [Reproduction checklist](manual/reproduction.md) · [Python API](manual/api.md) · [Troubleshooting](manual/troubleshooting.md) · [Developer guide](manual/development.md) · [Next release](releases/unreleased.md) |

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

Installation <manual/installation>
Quickstart <manual/quickstart>
Input <manual/input>
Science <manual/science>
Capabilities and limits <manual/capabilities>
```

```{toctree}
:maxdepth: 2
:caption: User guide
:hidden:

Gas phase <manual/workflows>
Rate PMF selection <rate-pmf-selection>
Explicit rate input and regional umbrellas <rate-input-v2>
Transition-state search <transition-state-search>
Materials and surfaces <manual/periodic>
Native PIMD workflow <native-pimd-workflow>
Native thermostats and barostats <native-ensembles>
Tensor PIMD and transport <torch-materials-workflows>
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
Batched machine-learning potentials <batched-ml-potentials>
Built-in native potentials <builtin-native-potentials>
External force engines and adapters <potential-adapters>
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
Precision-driven rate control <rate_precision>
Instantons <instanton>
Low-temperature cross-checks <low-temperature>
Tunnelling phase projection <tunnelling-phase-projection>
Quantum statistics <quantum-statistics>
Method coverage and boundaries <tier2-methods>
PA-CMD and property spectra <tier2-spectroscopy>
Molecular quasi-centroids and f-QCMD <qcmd>
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
Next release <releases/unreleased>
Portable publication and archives <portable-persistence>
Performance evidence and CUDA work <t19-performance>
Data archive plan <data-archive-plan>
Historical implementation review <implementation-review-2026-11-03>
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
