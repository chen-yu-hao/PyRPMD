# PyRPMD

**[User manual](docs/index.md)** · [Quickstart](docs/manual/quickstart.md) ·
[Examples](docs/manual/examples.md) · [Python API](docs/manual/api.md) ·
[中文](README.zh-CN.md)

PyRPMD provides path-integral sampling and ring-polymer molecular dynamics for
reaction rates, scattering, free energies, materials, transport and vibrational
spectroscopy. A Python or TOML input connects a potential to a workflow, with
strict unit checks, saved diagnostics and restartable calculations. C++17/OpenMP,
NumPy and explicitly selected Torch paths serve different supported workloads.

For a first calculation, use the bundled harmonic example below. For a research
calculation, choose a method in the capability table and follow its guide's
model, sampling and convergence requirements.

<a id="installation"></a>

## Installation

Use **Python 3.11+** and a **C++17 compiler** for the native build. OpenMP is enabled
by default on Linux. NumPy, SciPy, h5py and threadpoolctl are installed as dependencies.

```bash
git clone https://github.com/chen-yu-hao/PyRPMD.git
cd PyRPMD
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pyrpmd info
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
The [installation guide](docs/manual/installation.md) covers compiler selection,
an explicit NumPy-only build, MPI and optional engines. Install only the extras
needed by your calculation, for example `'.[analysis]'` for analysis tools,
`'.[openmm]'` or `'.[lammps]'` for force engines, and `'.[gpu]'` with a suitable
PyTorch distribution for Torch execution. Framework-specific ML extras and model
requirements are in the [batched-potential guide](docs/batched-ml-potentials.md).

<a id="first-run"></a>

## Your first run

This short one-dimensional umbrella calculation uses a built-in harmonic
potential. After installation, run it from any working directory:

```bash
python -m pyrpmd init my-first-run
python -m pyrpmd validate my-first-run/quickstart.inp --check-pes --output my-first-run/run
python -m pyrpmd run my-first-run/quickstart.inp --output my-first-run/run
python -m pyrpmd analyze my-first-run/run
python -m pyrpmd resume my-first-run/run
```

Read `my-first-run/run/results/summary.json`: the starter has 3 umbrella windows,
6 sampling streams and 120 saved samples. It checks the workflow;
`converged` remains false. `resume` continues an interrupted calculation, while
an already finished run stays complete. Use a new output directory when changing
scientific settings. The [quickstart](docs/manual/quickstart.md) explains the
input, saved files and reanalysis, and the [CLI reference](docs/manual/input.md)
lists commands and their distinct input schemas.

<a id="capabilities"></a>

## Capabilities

The table distinguishes input workflows from Python APIs. Availability of two
features does not imply that every combination is supported; each linked guide
states its restrictions. The [capability and validation guide](docs/manual/capabilities.md)
collects method boundaries, checkpoint rules and the evidence behind the results.

| Goal | Entry point and scope | Guide |
| --- | --- | --- |
| Reaction rates and scattering | `rate`, `drpmd`, `network`, `site_rate`, `surface_rate`; PMF and recrossing, direct collisions, state-selected NE-RPMD/QCT | [Reaction workflows](docs/manual/workflows.md) · [State-selected scattering](docs/state-selected.md) · [Materials and surfaces](docs/manual/periodic.md) |
| Rate precision and parameter studies | `precision plan/run/status` for budgeted, resumable thermal-rate production; `study` for temperature, isotope and numerical scans | [Precision control](docs/rate_precision.md) · [Rate studies](docs/rate-studies.md) |
| Free energy and enhanced sampling | `fes`: umbrella, metadynamics, OPES, ABF/eABF and steered MD; REUS, learned CVs and finite-temperature strings have separate controls/APIs | [Enhanced sampling](docs/enhanced-sampling.md) · [CVs and sampling](docs/tier2-sampling.md) · [Strings](docs/finite-temperature-string.md) |
| External biases and rare-event paths | PLUMED centroid bias with opt-in file-based restart; Python TIS/RETIS with a minus ensemble, REPPTIS and direct FFS | [PLUMED](docs/plumed.md) · [Path sampling](docs/transition-interface-sampling.md) |
| Equilibrium PIMD and transport | `pimd`, `diffusion`; native thermostats and supported NPT barostats, energy estimators, MSD/VACF and streamed trajectories | [Materials](docs/manual/periodic.md) · [Native PIMD](docs/native-pimd-workflow.md) |
| Device-resident trajectory batches | `sampling.engine="torch"` for fixed-cell PIMD and diffusion, float64 CPU/CUDA and whole-batch checkpoints; explicit model/device requirements | [Torch workflows](docs/torch-materials-workflows.md) · [Performance evidence](docs/t19-performance.md) |
| Quantum statistics and exchange | Suzuki–Chin, PIGS, bosonic exchange, fermionic sign reweighting, periodic winding sums and all-bead isotropic exchange NPT; GLE/PIGLET matrix import and generation | [Path-integral actions](docs/tier2-path-integral.md) · [Quantum estimators](docs/quantum-statistics.md) · [Condensed-phase NQE](docs/condensed-phase-nqe.md) |
| Vibrational spectra | `spectroscopy`: PA-CMD, quadratic f-CMD, experimental bent-triatomic f-QCMD and relative IR/Raman/SFG; TRPMD correlation APIs | [Spectroscopy](docs/tier2-spectroscopy.md) · [f-QCMD](docs/qcmd.md) · [TRPMD](docs/trpmd-spectroscopy.md) |
| Low-temperature rates and splittings | `instanton` and Python APIs: molecular paths, Hessian providers, convergence scans, fluctuation prefactors and normal-regime golden-rule instantons | [Instantons](docs/instanton.md) · [Low-temperature methods](docs/low-temperature.md) |
| Nonadiabatic dynamics | `nonadiabatic`, selected scattering and separate thermal-rate workflows; NRPMD, spin mapping and hopping methods; IESH and multistate MASH currently use P=1 | [Nonadiabatic workflows](docs/nonadiabatic-workflow.md) · [IESH and multistate MASH](docs/nonadiabatic-iesh-multistate.md) |
| Electronic-structure and quantum references | Python SA-CASSCF bridge with PySCF; spin–boson HEOM; model-specific PCET and constant-potential calculations | [Electronic data and HEOM](docs/tier2-nonadiabatic.md) · [PCET](docs/manual/pcet.md) · [Electrode models](docs/manual/electrochemistry.md) |
| Potentials and scalable evaluation | Native built-in analytic potentials; Python, C ABI, RPMDrate/f2py, ASE, Torch, i-PI, OpenMM and LAMMPS; batched MACE/NequIP/fairchem/DeePMD/metatomic | [Potential interfaces](docs/manual/potentials.md) · [Native built-ins](docs/builtin-native-potentials.md) · [Adapters](docs/potential-adapters.md) · [Batched ML](docs/batched-ml-potentials.md) |
| Model construction and active learning | PES generation, committees, delta corrections and restartable sample–label–train loops with user callbacks or the CPU MACE reference trainer | [PES generation](docs/manual/pes-generation.md) · [ML lifecycle](docs/tier2-ml-lifecycle.md) |
| Execution, output and analysis | Supported process/MPI parallelism, atomic checkpoints, trajectory export, offline analysis and reports; RPC/MTS and bond-constraint APIs | [Execution](docs/manual/execution.md) · [Outputs](docs/manual/results.md) · [Propagation](docs/propagation-stack.md) |

Real-time QCMD and h-CMD remain unimplemented; f-QCMD does not yet have the
requested converged spectral validation. The current performance reports contain
CPU measurements and CUDA benchmark drivers, **not measured GPU speedups**.
Numerical tests and short model comparisons do not establish physical convergence.

<a id="documentation"></a>

## Documentation

| Start here | Follow up |
| --- | --- |
| [English manual](docs/index.md) · [中文手册](docs/zh/index.md) | Matching core chapters, searchable HTML and topical guides |
| [Examples](docs/manual/examples.md) · [Input and CLI](docs/manual/input.md) | Runnable inputs, scientific settings and task-specific commands |
| [Scientific conventions](docs/manual/science.md) · [Capabilities and validation](docs/manual/capabilities.md) | Atomic units, convergence, limitations, published-model checks and audit records |
| [Python API](docs/manual/api.md) · [Developer guide](docs/manual/development.md) | Extending a PES/CV, tests, native builds and parallel execution |
| [Method references](docs/manual/references.md) · [Reproduction checklist](docs/manual/reproduction.md) | Original papers, implementation scope and reproducible research planning |
| [Next-release notes](docs/releases/unreleased.md) · [Development roadmap](docs/roadmap.md) | Merged additions, integration status and remaining work |

Build both language editions locally:

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -W --keep-going -b html docs/zh docs/_build/html/zh
```

The Documentation workflow builds the manuals and runs the quickstart. Cite the
actual method, PES, parameters, upstream software and PyRPMD commit used in a
study; [BibTeX](docs/references.bib) and the [structured reference catalog](docs/method-references.json)
are available. Source attribution, pinned upstream commits and third-party
licenses are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
PyRPMD's own license is [MIT](LICENSE); bundled third-party components retain
their stated licenses.
