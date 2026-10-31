# PyRPMD

**[User manual](docs/index.md)** · [Installation](docs/manual/installation.md) ·
[Quickstart](docs/manual/quickstart.md) · [Examples](docs/manual/examples.md) ·
[Python API](docs/manual/api.md) · [中文](README.zh-CN.md)

**Ring-polymer molecular dynamics from a single input file — from a transition state to a
recrossing-corrected rate, with fingerprinted checkpoint restart.**

Give PyRPMD a transition state and a potential, and it builds the whole Bennett–Chandler
calculation for you: a numerical Hessian and two-sided descent identify the reaction channel,
umbrella sampling gives the PMF, and constrained parent chains launch paired recrossing
trajectories for the transmission coefficient. The same Python or TOML input drives eleven tasks,
from gas-phase rates and direct collisions to free-energy surfaces, metadynamics, PIMD, diffusion,
surface hops and electrochemical PCET, on C++17/OpenMP integrator kernels with a NumPy reference
implementation alongside.

## Why PyRPMD

- **From TS to rate in one command.** No hand-staged windows or separate recrossing scripts: one
  input carries a calculation from structure to rate, with uncertainties.
- **Results you can defend.** Every run reports estimates, failed or unresolved trajectories and
  diagnostics separately. When overlap, the recrossing plateau or energy conservation fails a
  check, PyRPMD withholds the number and tells you why.
- **Restart with confidence.** Checkpoints hold full q/p, RNG state, bias history and a physics
  fingerprint, written atomically with per-array SHA256. Change the block size or loop backend
  on restart; changes to the science are refused.
- **Bring your own potential.** Python, RPMDrate/f2py, a versioned C ABI `.so`, a plain `.o`
  linked from a manifest, ASE, Torch or an i-PI socket.
- **Laptop to cluster.** Trajectory and window process pools, persistent bead-PES pools, MPI
  `G × B` groups, OpenMP and several trajectories per GPU across multiple cards.
- **Checked against published references.** H + H₂ on the BKMP2 surface, PCET rates within
  0.062% of pyPCET, and PMFs against analytic integrals and PyMBAR ([Validation](#validation)).

Expensive-potential workflows can use [provider reuse and trajectory batching](docs/performance-dynamics.md),
[explicit RPC/MTS Python APIs](docs/rpc-mts.md), and
[TRPMD, correlation functions and relative IR spectra](docs/trpmd-spectroscopy.md).
The [published-rate comparison tools](benchmarks/published_rates/README.md) check model identity,
statistical conventions and convergence evidence.


## Additional scientific methods

- [General-coordinate rates](docs/general-coordinate-rates.md): smooth nonlinear centroid coordinates with constrained-parent mass-metric and correlated-bootstrap corrections.
- [Temperature, isotope and convergence studies](docs/rate-studies.md): `pyrpmd study` with Arrhenius fits, KIEs and explicit numerical equivalence checks.
- [Ring-polymer instantons](docs/instanton.md): multidimensional paths and a one-dimensional low-temperature rate with explicit reactant normalization; a full multidimensional rate prefactor is not implemented.
- [Quantum statistics](docs/quantum-statistics.md): primitive/Takahashi–Imada equilibrium sampling, heat capacities, kinetic energies, fractionation and single-component open-path momentum distributions. Fourth-order TI accuracy applies to thermodynamic traces.

These model examples test the numerical implementation; physical and sampling convergence require a separate system-specific study.

## Installation

```bash
git clone https://github.com/chen-yu-hao/PyRPMD.git PyRPMD
cd PyRPMD
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
pyrpmd --version
pyrpmd info
```

PyRPMD needs Python 3.11+, NumPy, SciPy, h5py, threadpoolctl and a C++17 compiler; OpenMP is built
on Linux. Set `CC`/`CXX` to choose the compiler. Without a compiler, install the NumPy path:

```bash
PYRPMD_NO_NATIVE=1 python -m pip install -e .
```

| Extra | Install |
| --- | --- |
| MPI | Site MPI with a matching mpi4py, then `.[mpi]` |
| ASE | `.[ase]` |
| Torch / GPU | The PyTorch build matching the node's CUDA, then `.[gpu]` |

`native='required'` demands the compiled kernels, and requesting several dynamics threads without
OpenMP raises an error. MPI supports static groups and an opt-in dynamic trajectory queue with
one coordinator rank. Tests exercise four-rank static and five-rank dynamic execution on one
host, Torch on CPU and multi-GPU placement. See [Execution](docs/manual/execution.md).

## Your first run

A restartable 1D free-energy calculation on a built-in harmonic potential takes only a short
pilot run. After installation, start in any working directory; `init` supplies the input without
needing a source checkout or an external model:

```bash
pyrpmd init my-first-run
pyrpmd validate my-first-run/quickstart.inp --check-pes --output my-first-run/run
pyrpmd run my-first-run/quickstart.inp --output my-first-run/run
pyrpmd analyze my-first-run/run
pyrpmd resume my-first-run/run
```

Every task uses this same `validate` → `run` → `analyze` → `resume` cycle. `--check-pes` also
evaluates the potential and, for a TS-only rate input, runs the numerical preparation. Give each
new calculation its own output directory; `resume` (or `run --resume` on the original input)
continues an existing one. The starter produces 3 windows, 6 streams and 120 saved samples in
`my-first-run/run/results/summary.json`; its small sample is a workflow check, not a converged
physical prediction. `init` refuses a nonempty destination. If `pyrpmd` is not on your shell's
path, use `python -m pyrpmd` with the same arguments. The [Quickstart](docs/manual/quickstart.md)
explains the input, output and offline reanalysis.

## What you can compute

All examples use analytic model potentials and run on one core. Times are wall time with
`OMP_NUM_THREADS=1`.

| Task | `task` | Example | Time |
| --- | --- | --- | --- |
| Bimolecular RPMD rate from a TS | `rate` | `pyrpmd prepare examples/rate/pilot.inp --pilot` then `pyrpmd run examples/rate/pilot.inp` | ~15 s |
| Direct collisions (dRPMD) | `drpmd` | `examples/drpmd/free_scattering.inp` | ~2 s |
| Rate matrix over several entrances and TSs | `network` | `examples/network/two_entrances.py` | ~25 s |
| 1D/2D free-energy surface | `fes` | `examples/fes/harmonic2d.inp` | ~6 s |
| Adaptive umbrella windows | `fes` | `examples/fes/adaptive_harmonic.inp` | ~4 s |
| Standard or well-tempered metadynamics | `metadynamics` | `examples/metadynamics/well_tempered_2d.inp` | ~2 s |
| Probability-based enhanced sampling | `opes` | `examples/opes/harmonic_2d.inp` | — |
| Fixed-cell PIMD and energy estimators | `pimd` | `examples/materials/pimd.inp` | ~2 s |
| RPMD self-diffusion, MSD, VACF | `diffusion` | `examples/materials/diffusion.inp` | ~3 s |
| Directed site-to-site hop in a material | `site_rate` | `examples/materials/site_rate.inp` | ~25 s |
| Directed escape on a fixed substrate | `surface_rate` | `examples/surface/diffusion.inp` | ~1.5 min |
| Electrochemical PCET, H/D, potential and temperature scans | `pcet` | `examples/pcet/harmonic_scan.inp` | ~2 s |
| State-selected nonadiabatic scattering (NRPMD, experimental) | `drpmd` | `examples/nonadiabatic/exchange.inp` | ~15 s |

Run any of them with `pyrpmd run INPUT --output runs/NAME`. Two more patterns worth knowing:

```bash
# Restart with a different block size; the scientific settings keep the run identity.
pyrpmd resume runs/meta --loop-backend auto --block-steps 128
pyrpmd checkpoint runs/meta/checkpoints/metadynamics_0000.npz --require-integrity

# 2 trajectory groups × 2 bead-PES ranks.
mpiexec -n 4 pyrpmd run examples/drpmd/mpi.py --output runs/mpi
```

The [example index](docs/manual/examples.md) gives a learning path and the steps from an example
to a research calculation.

## Capabilities

| Area | What PyRPMD provides |
| --- | --- |
| Inputs | Python `build()` returning a `JobSpec` or dict, or TOML with strict checks on fields, units, atom indices and paths; command-line overrides for PES, temperature, beads and workers |
| RPMD rates | TS + PES → numerical Hessian and two-sided descent → channel identification → PMF (Gaussian UI or histogram MBAR) → constrained parents → paired recrossing children; paired UI/MBAR comparison on the same draws |
| Equivalent channels | Equivalent TSs from atom permutations, combined as RPMDrate's `min_bond(max_TS(s1))`; the two H + H₂ exchange exits are distinguished |
| Reaction networks | Separate entrances, TSs and PESs for one atom and isotope inventory; each path fills a rate matrix, reported per entrance, channel and TS |
| dRPMD | Two reactant XYZ files and an initial COM distance; persistent parent chains launch collisions at a set interval |
| Selected nuclear initial states | Approximate vibrational NE-RPMD and P=1 QCT at fixed collision energy, independent initial-state draws, precollision checks and unresolved-trajectory extension ([guide](docs/state-selected.md), [examples](examples/state_selected/README.md)) |
| Free energy | 1D–3D umbrella sampling with full CV Jacobians, MBAR, overlap graphs, correlated-block bootstrap, masks for unsampled regions and offline reanalysis; adaptive 1D/2D window insertion from MBAR overlap, effective sample size and optional coverage targets |
| Metadynamics | Standard and well-tempered on 1D–3D CVs, periodic/fixed-layer systems, NHC/PILE-L/Andersen sampling, soft walls and optional 1D/2D bias grids; independent histories and finite-time bias FES |
| Replica-exchange umbrellas | Same-temperature fixed-window REUS, PILE-L or Andersen, synchronized ensemble restart and joint window bootstrap ([guide](docs/replica-exchange.md)) |
| OPES | Fixed-bandwidth weighted probability estimation, optional frozen-bias reweighting and restartable independent histories ([guide](docs/opes.md)) |
| PLUMED and path sampling | Optional centroid-bias PLUMED bridge; unbiased two-way shooting transition-interface ensembles with separate flux normalization ([PLUMED](docs/plumed.md), [TIS](docs/transition-interface-sampling.md)) |
| Rate postprocessing | Offline 2D dividing-surface flux integration and signed channel contributions with explicit normalization and strict plateau checks ([guide](docs/rate-postprocessing.md)) |
| Materials and surfaces | Fixed-cell or isotropic BZP NPT PIMD, fixed layers, primitive and centroid-virial estimators, streamed trajectories; ordinary or centroid electronic-friction diffusion; basin-normalized site/surface rates and source-FES projection ([details](docs/surface-catalysis.md)) |
| Electrochemical PCET | Golden-rule forward/reverse rates in s⁻¹ for a local complex with one proton, a wide-band metal and a classical Gaussian environment; H/D KIE and vibronic contributions ([method](docs/pcet.md)) |
| Integrators | `cayley-bcb`, `normal-mode-bab`, `velocity-verlet`, each with native block propagation, Andersen parents and centroid RATTLE |
| Restart | v2 atomic checkpoints with per-array SHA256 (v1 still readable); block size and backend adjustable on restart |

The [native ensemble driver](docs/native-ensembles.md) provides opt-in C++ thermostat/barostat
trajectories with persistent state and compiled PES callbacks. The [potential adapters](docs/potential-adapters.md)
cover C ABI v2, periodic Torch, named ML calculators, tblite, PySCF and multi-client i-PI.
The [offline PES-generation workflow](docs/manual/pes-generation.md) connects PySCF references
to source-pinned Caracal QMDFF/EVB/TREQ, with explicit pilot and validation requirements.

PIMD and fixed-cell diffusion support frozen substrate atoms. Isotropic BZP NPT PIMD requires
fully periodic all-mobile cells and physical PES virials; materials trajectories stream to
checked chunks by default. A proton conductor needs a periodic potential suited to its
composition, charge and mechanism. Inputs, units, periodic CVs and
statistics are in the [materials notes](docs/materials.md). What's new in this release is in the
[0.2.0 release notes](docs/releases/0.2.0.md).

## Minimal inputs

A transition state and a potential are a complete rate input:

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

Two reactants and a separation are a complete dRPMD input:

```toml
task = "drpmd"

[system]
reactants = ["A.xyz", "B.xyz"]
separation = "12 angstrom"

[potential]
provider = "cabi"
library = "libpes.so"
```

Both use the `pilot` preset: 300 K, 32 beads and a 0.1 fs step. From there, each setting you add
unlocks a result:

- add a `bmax` checked for truncation to get a thermal dRPMD rate (without it, collisions start at
  `R0/4` and no thermal rate is reported);
- add classification rules to get chemical rates and cross sections (without them, PyRPMD infers
  connectivity from covalent radii and labels outcomes as candidates);
- add a custom classifier for isomerization and exchange channels.

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

## Validation

**H + H₂ on BKMP2.** [`benchmarks/h3`](benchmarks/h3/README.md) runs dRPMD, a 2D centroid FES, an
umbrella PMF and recrossing on the published BKMP2 surface at 1500 K with 16 beads. The build pins
the potential source by commit and SHA256, and verifies the authors' energy/gradient reference
values, the H₂ equilibrium structure and the first-order saddle before any propagation. Four
independent-seed repeats with an MBAR rate PMF cover 848,000 samples and 1,024 new recrossing
trajectories: all four pass, every paired interval for the UI/MBAR barrier difference on the same
dividing surface includes zero, and the joint relative standard error of the rate is 20–23%
([record](docs/validation/h3-pmf-refinement-2026-10-29/README.md),
[method](docs/mbar-rate-pmf.md), [estimator audit](docs/h3-pmf-estimator-audit.md)).

**State-selected scattering.** The [32-bead NE-RPMD / QCT reproduction](benchmarks/h3/STATE_SELECTED_REPRODUCTION.md)
compares H + H₂ cross sections with the literature. The subsequent
[initial-state preparation study](benchmarks/h3/STATE_PREPARATION_STUDY.md) separates centroid
conditioning, full-PES action QCT and experimental smooth preparation. Its
[paired results](benchmarks/h3/preparation_results/README.md) retain unresolved outcomes and the
remaining low-energy discrepancy, alongside isolated-diatom stability and energy diagnostics.

**PCET.** On the harmonic model, PyRPMD matches a pinned pyPCET commit to within 0.062% across 12
forward and reverse rates ([record](docs/validation/pcet-model-2026-10-30/README.md)).

**PMF numerics.** 1D/2D PMFs agree with analytic integrals and PyMBAR across bead numbers,
integrators, periodic CVs and correlated errors
([record](docs/validation/pmf-accuracy-2026-10-30/README.md)).

**Multichannel and sampling controls.** The [H + BrCl benchmark](benchmarks/brhcl/README.md) compares a historical LEPS-model PMF with an independent classical reference and records hidden-coordinate mixing. The [sampling-method study](docs/validation/fes-methods-2026-10-30/README.md) and [coverage follow-up](docs/validation/fes-optimization-2026-10-30/README.md) retain finite-sampling limitations. Projected PMF connectivity alone does not establish rates or branching fractions.

**Materials.** The [Ni–H EAM pilot](benchmarks/materials/README.md) runs H and D on a Ni(100) slab
with 1 and 4 beads on a source-pinned EAM model
([record](docs/validation/materials-2026-10-30.md)).

The **Real H3 BKMP2 validation** and **H3 PMF independent-seed study** GitHub Actions workflows
rerun these studies and store inputs, samples, checkpoints, PMF, κ(t), diagnostics and figures.

<details>
<summary>Reproduce the BKMP2 benchmark</summary>

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

The independent-seed study (cases `k400-a`, `k400-b`, `k800-a`, `k800-b`; `--smoke` runs a short
check):

```bash
python -m pip install -e '.[analysis]' matplotlib
python benchmarks/h3/convergence.py plan
python benchmarks/h3/convergence.py run --case k400-a --output runs/h3-pmf/k400-a
python benchmarks/h3/solver_check.py --run runs/h3-pmf/k400-a/run \
  --stride 10 --reference-pymbar --output runs/h3-pmf/k400-a/solver-check.json
```

</details>

The [H + HO₂ 2D scan](benchmarks/hho2/README.md) is ready for a full-dimensional reactive H + HO₂
PES, with multiple starting geometries, a shared finite-distance wall, six-bond channel
classification and connectivity analysis.

Before a production study, read [Converging a result](docs/manual/science.md#converging-a-result),
which takes a calculation from pilot to production.

The [adaptive umbrella reliability audit](docs/validation/umbrella-reliability-2026-10-31/README.md)
records reproducible overlap and uncertainty fixes, actual default budgets, and the remaining
system-specific convergence checks. Passing adaptive pilot criteria is not a precision guarantee.

## Native propagation

`Dynamics.propagate()` commits whole blocks atomically; `Dynamics.propagate_block()` commits step
by step with observation masks, full q/p snapshots and native centroid RATTLE, and keeps the
successful prefix on a numerical failure. Both support all three integrators. Built-in
connectivity and H3 exchange classifiers run inside the C++ blocks and stop collisions early; a
fused C++ PILE-L kernel and a native FFT for normal-mode BAB at power-of-two `P>=32` accelerate
thermostatted and large-bead runs. Defaults are `loop_backend="auto"` and `block_steps=256`.
Details: [native propagation](docs/native-propagation.md),
[block API and failure handling](docs/block-propagation.md), [integrators](docs/integrators.md),
[optimization record](docs/validation/audit-optimization-2026-10-30/README.md).

State-selected collisions use the same C++ production path. Optional native observations
record potential, centroid kinetic, internal-bead kinetic and spring energy per bead, plus
per-atom ring radii, without additional PES calls. `pyrpmd state-check` projects sparse saved
states onto the molecular reference outside the production block. See the
[state-selected guide](docs/state-selected.md) for the harmonic-preparation approximation,
the auxiliary beta parameter and fixed-energy statistical eligibility rules.

## Documentation

The user manual covers installation, a quickstart, guides for all eleven tasks, the API and developer
notes, in [English](docs/index.md) and [Chinese](docs/zh/index.md) with matching chapters. Build
the searchable HTML manual (the Chinese edition goes to `zh/`):

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -W --keep-going -b html docs/zh docs/_build/html/zh
```

The GitHub **Documentation** workflow builds both editions, runs the getting-started examples and
uploads the `pyrpmd-manual-html` artifact.

See the [user-experience review and to-do list](docs/ux-review-2026-10-31.md) for the current
onboarding audit and follow-up work.

A run directory holds `manifest.json`, `input/`, `checkpoints/`, workflow data and
`results/summary.json`. FES adds `fes.h5`/`fes.npz`/`fes.csv`; dRPMD stores each initial state,
collision trajectory and classification record. MPI ranks never share one HDF5 file.

Reference specifications (some in Chinese):

- [Input specification](docs/input.md)
- [Connecting potentials](docs/potentials.md) and the [C ABI](native/ABI.md)
- [Parallel and multi-node runs](docs/parallel.md)
- [Scientific conventions](docs/science.md)
- [Development and validation](docs/development.md), [validation record](docs/validation.md)
- [Multiple entrances and TSs](docs/reaction-networks.md)
- [Experimental NRPMD and state-resolved collisions](docs/nonadiabatic.md)
- [Electronic friction](docs/electronic-friction.md), [thermal NRPMD](docs/thermal-nrpmd.md), [KC/MV/isomorphic methods](docs/nonadiabatic-methods.md), [SOC and RPSH](docs/nonadiabatic-soc-rpsh.md)
- [BZP constant-pressure PIMD](docs/materials-npt.md), [grid metadynamics](docs/grid-metadynamics.md), [streaming trajectories](docs/streaming-trajectories.md)
- [Constant-potential electrode models and HER postprocessing](docs/electrochemistry.md)
- [Adaptive umbrella sampling and metadynamics](docs/enhanced-sampling.md)
- [Electrochemical PCET model](docs/pcet.md), [runnable PCET example](examples/pcet/README.md)
- [Fixed cells, PIMD, materials diffusion and site rates](docs/materials.md)
- [Surface catalysis and periodic slabs](docs/surface-catalysis.md)
- [Public interfaces](docs/interfaces.md), [checkpoints and restart](docs/checkpoints.md)

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
