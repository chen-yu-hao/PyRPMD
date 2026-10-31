# PyRPMD

**[User manual](docs/index.md)** · [Installation](docs/manual/installation.md) ·
[Quickstart](docs/manual/quickstart.md) · [Examples](docs/manual/examples.md) ·
[Python API](docs/manual/api.md) · [中文](README.zh-CN.md)

**Ring-polymer molecular dynamics in Python.** PyRPMD 0.2.0 computes bimolecular RPMD rates,
direct-collision dRPMD, rate matrices over several entrances and transition states, 1D/2D free-energy
surfaces with adaptive umbrella sampling and metadynamics, fixed-cell PIMD, RPMD diffusion, site and
surface hop rates, and local electrochemical PCET rates. Inputs are plain Python or TOML `.inp`
files. C++17/OpenMP kernels implement the Cayley, exact normal-mode BAB and velocity-Verlet
integrators, with a NumPy reference implementation alongside.

Every run reports finite-sample estimates, failed or unresolved trajectories and diagnostics
separately. The [convergence chapter](docs/manual/science.md#converging-a-result) of the manual
describes how to take a result from a pilot run to production.

## Features

| Area | What PyRPMD provides |
| --- | --- |
| Python inputs | `build()` in a `.py` file returns a `JobSpec` or dict; custom PES, CVs and product classifiers come from importable factories |
| TOML inputs | Strict checking of fields, units, atom indices and relative paths; command-line overrides for PES, temperature, beads and workers |
| RPMD rates from a TS | A TS and a PES start a numerical Hessian and two-sided descent; the identified bimolecular channel runs through PMF, constrained parents and paired recrossing children |
| dRPMD | Two reactant XYZ files, an initial COM distance and a PES; persistent parent chains launch collisions at a set interval, with remainder-aware trajectory allocation |
| Potentials | Python, RPMDrate/f2py, versioned C ABI `.so`, manifest-driven `.o` linking, ASE, Torch, i-PI socket |
| Parallelism | Trajectory/window processes, persistent bead PES pools, MPI `G × B` groups, OpenMP, multiple CUDA devices with several trajectories per card |
| Free-energy surfaces | 1D/2D umbrella sampling, full CV Jacobians, MBAR, overlap graphs, correlated-block bootstrap, masks for unsampled regions, offline reanalysis |
| Rate PMF | Gaussian UI or histogram MBAR; fixed dividing surfaces, paired UI/MBAR comparison on the same draws, per-draw bootstrap diagnostics |
| Equivalent exchange channels | Equivalent TSs from atom permutations, combined as RPMDrate's `min_bond(max_TS(s1))`; the two H + H₂ exchange exits are distinguished |
| Multiple entrances and TSs | Separate entrance partitions, TS structures and PESs for one atom and isotope inventory; each path runs PMF and recrossing and fills a rate matrix |
| Integrators | `cayley-bcb`, `normal-mode-bab`, `velocity-verlet`, each through both native block APIs, Andersen parents and centroid RATTLE |
| Adaptive umbrella sampling | 1D/2D pilots insert windows and extend sampling from MBAR overlap and effective sample size, then freeze for independent production; works for FES and the rate PMF |
| Metadynamics | Standard and well-tempered, 1D/2D and periodic CVs, full Jacobians, independent replicas, HILLS and finite-time bias FES |
| Replica-exchange umbrellas | Same-temperature fixed-window REUS, PILE-L or Andersen, synchronized ensemble restart and joint window bootstrap ([guide](docs/replica-exchange.md)) |
| OPES | Fixed-bandwidth weighted probability estimation in 1D/2D, explicit bias updates, optional frozen-bias histogram reweighting and independent restartable histories ([guide](docs/opes.md)) |
| PLUMED bridge | Optional Python/kernel integration for coordinate-dependent centroid biases, physical unit conversion and one update per completed step ([guide](docs/plumed.md)) |
| Transition-interface sampling | Unbiased two-way shooting path ensembles, rejection-aware crossing statistics, separate effective-flux normalization and streaming callbacks ([guide](docs/transition-interface-sampling.md)) |
| Electrochemical PCET | Golden-rule forward/reverse rates (s⁻¹) for a local complex with one proton, a wide-band metal and a classical Gaussian environment; potential/temperature scans, H/D KIE, vibronic contributions, numerical diagnostics |
| Fixed cells and PBC | Orthogonal and skewed cells, partially periodic slabs, continuous unwrapped coordinates; periodic contracts for ASE, Python and i-PI potentials |
| PIMD | PILE-L or Andersen sampling matched to the integrator; primitive and centroid-virial energy estimators, radii of gyration |
| Diffusion | Thermostat-free RPMD after thermalization; centroid MSD, diffusion tensor, VACF, replica errors, offline reanalysis with regime checks |
| Site and surface rates | Linear centroid coordinates, basin-normalized QTST, constrained parents and paired recrossing; first-order rates in s⁻¹; tilted cells, partial PBC and fixed layers for surfaces ([details](docs/surface-catalysis.md)) |
| Restart | Full q/p, RNG, bias history, stage and physics fingerprint; v2 atomic checkpoints with per-array SHA256, v1 data still readable; block size and backend adjustable on restart |

Multi-TS results are reported per entrance, channel and TS (`aggregation="separate"`), since fluxes
through different dividing surfaces can overlap. Metadynamics gives finite-time free-energy
estimates. PCET solves a 1D proton eigenproblem with an electrode energy integral; its conventions
are in the [PCET notes](docs/pcet.md). New interfaces and compatibility are listed in the
[0.2.0 release notes](docs/releases/0.2.0.md).

MPI runs are tested with four ranks on one machine, and Torch CPU execution and multi-GPU placement
have their own tests. MPI groups are static with `pes_ranks=1`.

## Installation

```bash
git clone https://github.com/chen-yu-hao/PyRPMD.git PyRPMD
cd PyRPMD
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pyrpmd info
```

PyRPMD needs Python 3.11+, NumPy, SciPy, h5py, threadpoolctl and a C++17 compiler; OpenMP is built
on Linux. Set `CC`/`CXX` to choose the compiler. Without a compiler, install the NumPy path:

```bash
PYRPMD_NO_NATIVE=1 python -m pip install -e .
```

For MPI, install the site MPI with a matching mpi4py and the `.[mpi]` extra. ASE uses `.[ase]`. For
Torch, install the PyTorch build that matches the node's CUDA, then `.[gpu]`. `native='required'`
demands the compiled kernels, and requesting several dynamics threads without OpenMP raises an error.

## Quick start

```bash
# Two persistent parent chains share 5 collisions as 3 + 2.
pyrpmd validate examples/drpmd/free_scattering.inp
pyrpmd run examples/drpmd/free_scattering.inp --output runs/scattering
pyrpmd resume runs/scattering

# Free-energy density surface on two CVs.
pyrpmd run examples/fes/harmonic2d.inp --output runs/fes2d
pyrpmd analyze runs/fes2d --recompute --bins 20 20 --bootstrap 50

# Analytic reaction potential: automatic TS preparation, then the full Bennett–Chandler pipeline.
pyrpmd prepare examples/rate/pilot.inp --pilot
pyrpmd run examples/rate/pilot.inp --output runs/rate

# Two entrances, four TS/PES paths.
pyrpmd run examples/network/two_entrances.py --output runs/network

# Adaptive windows and well-tempered metadynamics.
pyrpmd run examples/fes/adaptive_harmonic.inp --output runs/adaptive
pyrpmd run examples/metadynamics/well_tempered_2d.inp --output runs/meta
pyrpmd analyze runs/meta --recompute --bins 30 30

# Replica-exchange umbrellas and OPES with frozen-bias production.
pyrpmd run examples/fes/replica_exchange_2d.inp --output runs/reus
pyrpmd run examples/opes/harmonic_2d.inp --output runs/opes
pyrpmd analyze runs/opes --recompute --bins 10 10 --bootstrap 0

# Interface paths on a confined model; PLUMED is a separate optional runtime.
python examples/sampling/transition_interface.py
python examples/plumed/centroid_metadynamics.py

# PCET model: temperature and potential scan for H/D, local first-order rates.
pyrpmd run examples/pcet/harmonic_scan.inp --output runs/pcet-harmonic

# Restart with a different block size; scientific settings keep the run identity.
pyrpmd resume runs/meta --loop-backend auto --block-steps 128
pyrpmd checkpoint runs/meta/checkpoints/metadynamics_0000.npz --require-integrity

# 2 trajectory groups × 2 bead PES ranks.
mpiexec -n 4 pyrpmd run examples/drpmd/mpi.py --output runs/mpi

# First-order hop rate on a fixed periodic surface, in s^-1.
pyrpmd run examples/surface/diffusion.inp --output runs/surface-diffusion
```

`validate` checks any input; `validate --check-pes` also constructs and evaluates the potential, and
for a TS-only rate input it runs the numerical preparation. For PCET, `--check-pes` solves the proton
eigenstates; `run` adds the rates and discretization checks. Use a new directory for each run, and
continue a run with `resume` or with `run --resume` on the original input.

## Hydrogen and protons in materials

```bash
# Fixed-cell PIMD on an analytic model.
pyrpmd run examples/materials/pimd.inp --output runs/materials-pimd

# Free particle: the analysis detects ballistic motion and reports no D.
pyrpmd run examples/materials/diffusion.inp --output runs/materials-free
pyrpmd analyze runs/materials-free --recompute --axes 0 1 \
  --fit-range '10 fs' '20 fs' --max-lag 40

# Rate for one directed exit from a finite basin.
pyrpmd run examples/materials/site_rate.inp --output runs/materials-site
```

The [Ni–H EAM pilot](benchmarks/materials/README.md) runs H and D on a Ni(100) slab with 1 and 4
beads on a source-pinned EAM model. Materials tasks use a fixed cell with all atoms mobile; proton
conductors need a periodic potential suited to their composition, charge and mechanism. Inputs,
units, periodic CVs and statistics are described in the [materials notes](docs/materials.md), and
the test results in the [materials validation record](docs/validation/materials-2026-10-30.md).

## Validation on H + H₂ / BKMP2

The [2026-10-30 PMF accuracy validation](docs/validation/pmf-accuracy-2026-10-30/README.md) checks
1D/2D PMFs against analytic integrals and PyMBAR across bead numbers, integrators, periodic CVs and
correlated errors, and runs the BKMP2 workflow. The [H + HO₂ 2D scan](benchmarks/hho2/README.md)
provides multiple starting geometries, a shared finite-distance wall, six-bond channel
classification and connectivity analysis on supported grids, ready for a full-dimensional reactive
H + HO₂ PES.

[`benchmarks/h3`](benchmarks/h3/README.md) runs dRPMD, a 2D centroid FES, an umbrella PMF and
recrossing on the published BKMP2 surface. The build downloads the potential source from a pinned
commit and checks its SHA256, then verifies the authors' energy/gradient reference values, the H₂
equilibrium structure and the first-order saddle before propagating. The benchmark runs at 1500 K
with 16 beads.

```bash
# Needs GNU Fortran; figures need matplotlib.
python -m pip install matplotlib
python benchmarks/h3/build_pes.py
python benchmarks/h3/pes.py --check --output runs/h3/pes-check.json
python benchmarks/h3/numerics.py --output runs/h3/numerics.json
python benchmarks/h3/drpmd.py --output runs/h3/drpmd --workers 2
python benchmarks/h3/rpmd.py --stage fes2d --output runs/h3/fes2d --workers 2
python benchmarks/h3/rpmd.py --stage rate --output runs/h3/rate --workers 2
```

The **Real H3 BKMP2 validation** GitHub Actions workflow runs this system and stores the inputs,
samples, checkpoints, PMF, κ(t), diagnostics and figures. The
[2026-10-29 record](docs/validation/h3-bkmp2-2026-10-29/README.md) keeps the numbers, figures and
commit of an actual run, and the [UI / MBAR estimator audit](docs/h3-pmf-estimator-audit.md)
explains the PMF difference between the two estimators on the same dividing surface.

An independent-seed study uses MBAR for the rate PMF, with stiffer and denser umbrella windows, and
recomputes the constrained parents and recrossing for every case:

```bash
python -m pip install -e '.[analysis]' matplotlib
python benchmarks/h3/convergence.py plan
python benchmarks/h3/convergence.py run --case k400-a --output runs/h3-pmf/k400-a
python benchmarks/h3/solver_check.py --run runs/h3-pmf/k400-a/run \
  --stride 10 --reference-pymbar --output runs/h3-pmf/k400-a/solver-check.json
```

The four cases are `k400-a`, `k400-b`, `k800-a` and `k800-b`. The **H3 PMF independent-seed study**
workflow runs them in parallel and stores all samples, per-draw diagnostics, an independent PyMBAR
check and a cross-seed summary; `--smoke` runs a short check only. Method and plan are in
[MBAR rate PMF](docs/mbar-rate-pmf.md).

The [results of the four independent repeats](docs/validation/h3-pmf-refinement-2026-10-29/README.md)
cover 848,000 samples and 1,024 new recrossing trajectories. All four cases pass, the four paired
intervals for the UI/MBAR barrier difference on the same surface include zero, and the joint relative
standard error of the rate is 20–23%.

## Minimal inputs

A transition state and a potential:

```toml
task = "rate"

[system]
transition_state = "ts.xyz"

[potential]
provider = "cabi"
library = "libpes.so"
```

The `[potential]` section can also come from the command line as `--pes ./libpes.so`. The `.so`
implements the PyRPMD PES v1 ABI; f2py modules use `provider='rpmdrate'`, and a plain `.o` adds
`--pes-manifest` naming the compiler, wrapper and link dependencies.

A minimal dRPMD input:

```toml
task = "drpmd"

[system]
reactants = ["A.xyz", "B.xyz"]
separation = "12 angstrom"

[potential]
provider = "cabi"
library = "libpes.so"
```

Both use the `pilot` preset: 300 K, 32 beads and a 0.1 fs step. Without `bmax`, dRPMD starts at
`R0/4` and reports no thermal rate; give a `bmax` checked for truncation to get one. Without
classification rules, PyRPMD infers connectivity from covalent radii, labels outcomes as candidates
and withholds chemical rates and cross sections; isomerization and exchange channels take a custom
classifier.

## Python extensions

Python inputs can compute parameters, load models and build any static, differentiable CV. Put
factories in importable modules; each process or MPI rank constructs its own potential context:

```python
# job.py
from pyrpmd import JobSpec

def build():
    return JobSpec.from_mapping({
        "task": "drpmd",
        "temperature": "500 K",
        "beads": 32,
        "system": {
            "reactants": ["A.xyz", "B.xyz"],
            "separation": "12 angstrom",
        },
        "potential": {
            "provider": "python",
            "factory": "my_pes.py:build_pes",
            "params": {"model": "model.dat"},
        },
        "collision": {"bmax": "4 angstrom", "trajectories": 1000},
        "parallel": {"backend": "process", "mode": "serial", "trajectory_workers": 8},
    })
```

A PES receives `float64[B,N,3]` coordinates in atomic units and returns `[B]` energies and `[B,N,3]`
forces (negative gradients). A custom CV returns `value[D]` and the full `jacobian[D,P,N,3]`. See the
runnable [Python PES](examples/drpmd/python_pes.py) and [custom 2D CV](examples/fes/custom_input.py)
examples. Python inputs run as ordinary, trusted code.

## Documentation

The user manual goes from installation and a quickstart through task guides to the API and
developer notes, following the PySCF manual layout. It covers all 10 tasks across gas-phase
reactions, free energies, periodic materials, fixed surfaces and PCET, with units, outputs and
restart. It is available in [English](docs/index.md) and [Chinese](docs/zh/index.md) with matching
chapters.

Build the HTML manual with navigation and search (the Chinese edition goes to `zh/`, and the navbar
switches languages):

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -W --keep-going -b html docs/zh docs/_build/html/zh
```

Open `docs/_build/html/index.html` (English) or `docs/_build/html/zh/index.html` (Chinese). The
GitHub **Documentation** workflow builds both editions, runs the getting-started examples and
uploads the `pyrpmd-manual-html` artifact.

| Path | Responsibility |
| --- | --- |
| `src/pyrpmd/input.py`, `units.py`, `system.py`, `geometry.py` | Input, units, atom/isotope systems and fixed-cell geometry |
| `src/pyrpmd/dynamics.py`, `native/` | Three ring integrators, momentum sampling, thermostats, C++/OpenMP |
| `src/pyrpmd/potentials/` | The potential interface and its adapters |
| `src/pyrpmd/cv.py`, `bias.py` | CVs, full Jacobians, umbrellas, rate flux metric, history-dependent Gaussian bias |
| `src/pyrpmd/workflows/`, `reactions.py` | rate, dRPMD, PIMD, diffusion, site/surface rates, PCET, FES, adaptive windows, metadynamics, multi-TS organization and preparation |
| `src/pyrpmd/analysis/` | MBAR, umbrella integration, metadynamics bias FES, MSD/VACF/diffusion tensor, statistical diagnostics |
| `src/pyrpmd/checkpoint.py`, `runner.py` | Atomic state containers, integrity checks, run identity and public entry points |
| `src/pyrpmd/parallel/` | Trajectory pools, bead pools, MPI, GPU placement |
| `examples/`, `tests/`, `docs/` | Runnable examples, numerical/integration tests, specifications |

Reference specifications (some in Chinese):

- [Input specification](docs/input.md)
- [Connecting potentials](docs/potentials.md) and the [C ABI](native/ABI.md)
- [Parallel and multi-node runs](docs/parallel.md)
- [Scientific conventions](docs/science.md)
- [Development and validation](docs/development.md), [validation record](docs/validation.md)
- [Native propagation](docs/native-propagation.md), [block API and failure handling](docs/block-propagation.md)
- [Integrators](docs/integrators.md), [multiple entrances and TSs](docs/reaction-networks.md)
- [Experimental NRPMD and state-resolved collisions](docs/nonadiabatic.md)
- [Adaptive umbrella sampling and metadynamics](docs/enhanced-sampling.md)
- [Electrochemical PCET model](docs/pcet.md), [runnable PCET example](examples/pcet/README.md)
- [Fixed cells, PIMD, materials diffusion and site rates](docs/materials.md)
- [Surface catalysis and periodic slabs](docs/surface-catalysis.md)
- [Public interfaces](docs/interfaces.md), [checkpoints and restart](docs/checkpoints.md)

A run directory holds `manifest.json`, `input/`, `checkpoints/`, workflow data and
`results/summary.json`. FES adds `fes.h5`/`fes.npz`/`fes.csv`; dRPMD stores each initial state,
collision trajectory and classification record. MPI ranks never share one HDF5 file.

## Native C++17 propagation

`Dynamics.propagate()` propagates continuously and commits whole blocks atomically.
`Dynamics.propagate_block()` commits step by step with observation masks, full q/p snapshots and
native centroid RATTLE, keeping the successful prefix on a numerical failure. Both APIs support all
three integrators. dRPMD collisions and constrained parents use the block interface; native FES and
the PMF workflows that reuse its sampler collect several samples per block with per-step energy
checks, while Andersen thermalization and recrossing children use continuous propagation.

Custom Python PES and CVs use the compatible path, and custom classifiers run in Python at their
event boundaries. The built-in connectivity channels and the H3 exchange classifier run inside the
C++ blocks at their original frequency, accumulate persistence and stop early; borderline geometries
fall back to the Python test. Native dRPMD reuses endpoint forces within a block and keeps a single
rollback snapshot instead of a per-step q/p history. Defaults are `loop_backend="auto"` and
`block_steps=256`.

PILE-L in `auto` mode uses a fused C++ free-ring half-step–OU–half-step kernel for power-of-two and
small bead numbers, with NumPy random numbers and the Python potential interface; fixed atoms stay
exactly fixed. Normal-mode BAB uses a native FFT for power-of-two `P>=32`, and the default Cayley
algorithm is unchanged. Benchmarks and regression coverage are in the
[optimization validation record](docs/validation/audit-optimization-2026-10-30/README.md); method
selection, numerical comparisons and restart rules are in the [integrator notes](docs/integrators.md).

## Tests and provenance

```bash
python -m pytest -q
mpiexec -n 4 python tests/mpi_integration.py
```

The Cayley integrator, reaction coordinate and rate algorithms are adapted from
[RPMDrate](https://github.com/GreenGroup/RPMDrate) and the author's
[Cayley/thermal-dRPMD branch](https://github.com/chen-yu-hao/RPMDrate/tree/9a44d6b378d8bf835d67e1b3b7f78b27bdc84a78);
the rate workflow also follows the umbrella/recrossing organization of
[Caracal](https://github.com/Trebonius91/Caracal/tree/7f95ce9818b7f6d81d7cb3b634f360373c090820).
Adapters, the input specification and scheduling are new layers. Source commits, authors, MIT
notices and paper DOIs are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Caracal's
QMDFF, xTB and machine-learning models are not bundled.

License: [MIT](LICENSE).
