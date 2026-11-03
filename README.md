# PyRPMD

**Ring-polymer molecular dynamics and reproducible molecular simulation workflows.** PyRPMD combines C++17/OpenMP numerical kernels, a NumPy reference implementation, strict potential-energy interfaces, and checkpointed Python/TOML workflows for reaction rates, quantum equilibrium sampling, spectroscopy, materials, and nonadiabatic model dynamics. Numerical checks are not a claim of physical convergence.

[中文](README.zh-CN.md) · [User manual](docs/index.md) · [Installation](docs/manual/installation.md) · [Quickstart](docs/manual/quickstart.md) · [Examples](docs/manual/examples.md) · [API](docs/manual/api.md)

## Install

Python and a C++17 compiler are required for the default native build.

```bash
python -m pip install -e '.[test]'
```

For the documentation environment use `python -m pip install -e '.[docs]'`. GPU, electronic-structure, MPI, and external-engine adapters have additional optional requirements; consult [installation](docs/manual/installation.md) and [potential adapters](docs/manual/potentials.md). Unsupported combinations fail explicitly rather than silently changing device, precision, or engine.

## First run

```bash
python -m pyrpmd init my-first-run
python -m pyrpmd validate my-first-run/quickstart.inp --check-pes --output my-first-run/run
python -m pyrpmd run my-first-run/quickstart.inp --output my-first-run/run
python -m pyrpmd analyze my-first-run/run
python -m pyrpmd resume my-first-run/run
```

The template uses an analytic potential. Read [the quickstart](docs/manual/quickstart.md) before supplying a real PES; every scientific change requires a new run identity. Restart and convergence limits are described in [execution](docs/manual/execution.md) and [scientific conventions](docs/manual/science.md).

## Capabilities

| Goal | Entry point | Guide |
| --- | --- | --- |
| Thermal RPMD rates, recrossing, PMFs, networks | `rate`, `network`, `pyrpmd study` | [Rates and workflows](docs/manual/workflows.md) · [Precision-controlled rates](docs/rate_precision.md) |
| Direct scattering and state-selected collisions | `drpmd` | [Gas-phase workflows](docs/manual/workflows.md) |
| Enhanced sampling, free-energy surfaces, OPES, ABF and replica exchange | `fes` | [Enhanced sampling](docs/enhanced-sampling.md) · [Replica exchange](docs/replica-exchange.md) |
| Equilibrium PIMD, quantum statistics and exchange | `pimd`, Python API | [Materials](docs/manual/periodic.md) · [Path-integral actions](docs/tier2-path-integral.md) |
| Native and Torch batched PIMD/diffusion | `sampling.engine` | [Native](docs/native-pimd-workflow.md) · [Torch](docs/torch-materials-workflows.md) |
| Diffusion, sites, surfaces and electrochemical PCET | `diffusion`, `site_rate`, `surface_rate`, `pcet` | [Materials](docs/manual/periodic.md) · [PCET](docs/manual/pcet.md) |
| Instanton paths, tunnelling rates and splitting | `pyrpmd instanton`, Python API | [Instantons](docs/instanton.md) · [Low-temperature cross-checks](docs/low-temperature.md) |
| Spectroscopy and centroid-based methods | `spectroscopy`, Python API | [Spectroscopy](docs/tier2-spectroscopy.md) · [f-QCMD](docs/qcmd.md) |
| Nonadiabatic models, IESH, MASH and thermal NRPMD | `nonadiabatic`, Python API | [Nonadiabatic guide](docs/manual/nonadiabatic.md) · [IESH/MASH limitations](docs/nonadiabatic-iesh-multistate.md) |
| RETIS, REPPTIS, FFS and committor | Python API | [Path sampling](docs/transition-interface-sampling.md) · [Committor](docs/committor.md) |
| ML/ASE/OpenMM/LAMMPS/PyTorch potential adapters | Python API and model providers | [Potentials](docs/manual/potentials.md) · [Batched ML](docs/batched-ml-potentials.md) |

The table distinguishes input workflows from Python-only APIs; individual guides document unsupported combinations and reference comparisons.

## Documentation and reproducibility

The [English manual](docs/index.md) and [中文手册](docs/zh/index.md) share the same core chapter set. Use the [method and primary-paper catalog](docs/manual/references.md), [reproduction checklist](docs/manual/reproduction.md), [validation and limitations](docs/literature-validation.md), and [release history](docs/releases/0.2.0.md) before making scientific claims. [Development and contributing](docs/manual/development.md) provides testing, CI and extension guidance.

> A passing software test, a finite statistical uncertainty or a completed checkpoint is not evidence that a material property, rate or spectrum has physically converged.
