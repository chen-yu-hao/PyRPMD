# Example index and learning path

The short examples below are smoke tests: they exercise a workflow before you connect your
own potential. The analytic tutorial tier uses longer sampling and explicit reference checks.
A tutorial is accepted only when its comparison script passes on the saved run. Run every command from the repository root: `pyrpmd validate INPUT` first (add `--check-pes` to
construct the PES), then `pyrpmd run` with a new `--output` directory. External models, MPI and GPU
need the dependencies listed in [Installation](installation.md).

For a first run without locating repository files, use `pyrpmd init my-first-run` and follow the
[Quickstart](quickstart.md). The paths below refer to the source checkout.

## Analytic tutorials with measured acceptance

The times below are measured `run` wall times from the 2026-11-10 one-core local
validation batch (Python 3.11.18, OMP/OpenBLAS/MKL threads set to one). Comparison
time is separate. `pending` means no completed record is available; it is not
an estimate or an accepted result. Each tutorial's `compare.py` reports the
measured deviation, SEM and applicable discretization bound without changing
its production summary.

| Tutorial input | Independent reference | Measured run time | Comparison |
| --- | --- | --- | --- |
| [harmonic PIMD](../../examples/tutorials/harmonic_pimd/README.md) | Finite-P harmonic energy and ring radius | 10.20 s | passed; potential-energy deviation 1.565 SEM |
| [periodic site rate](../../examples/tutorials/periodic_site_rate/README.md) | Directed classical TST | 176.80 s | passed |
| [double-well WT-MetaD](../../examples/tutorials/double_well_metad/README.md) | Exact bin-integrated Boltzmann density | 93.95 s | passed |
| [free diffusion](../../examples/tutorials/free_diffusion/README.md) | kBT/(m gamma) | 31.07 s | passed |
| [harmonic spectroscopy](../../examples/tutorials/harmonic_spectroscopy/README.md) | sqrt(k/m) centroid frequency | 95.36 s | passed; 0.225 cm⁻¹ difference within 2.545 cm⁻¹ half-bin bound |

Use a fresh output directory and run the tutorial's comparison command. These
fixed-model checks do not certify arbitrary production systems; see
[How to converge a result](../convergence.md).

## Smoke examples without external models

The approximate smoke-test times below are historical reference timings, not
measurements from the tutorial batch above. Unmeasured entries are `pending`.

| Input/program | Content | Time | Chapter |
| --- | --- | --- | --- |
| `examples/manual/quickstart.inp` | 1D harmonic FES with three umbrella windows and 6 streams | ~2 s | [Quickstart](quickstart.md) |
| `examples/manual/low_level.py` | 100-step program with `Dynamics` / `State` | <1 s | [API](api.md) |
| `examples/fes/harmonic2d.inp` | Two centroid CVs, 2D MBAR and masks | ~6 s | [Gas phase and free energy](workflows.md) |
| `examples/fes/replica_exchange_2d.inp` | PILE-L, replica-exchange umbrellas and synchronized bootstrap | ~8 s | [Replica exchange](../replica-exchange.md) |
| `examples/fes/custom_input.py` | Custom 2D CV with a full Jacobian | ~7 s | [Potentials and plugins](potentials.md) |
| `examples/fes/adaptive_harmonic.inp` | Pilot window insertion, then frozen production | ~4 s | [Gas phase and free energy](workflows.md) |
| `examples/drpmd/free_scattering.inp` | Two parent chains sharing 5 collisions on V=0 | ~2 s | [Gas phase and free energy](workflows.md) |
| `examples/state_selected/ne_rpmd.inp`, `qct.inp` | Selected-vibrational-state NE-RPMD and its one-bead QCT control on a Morse model; `state-check`, `extend` | ~1 s each | [Gas phase and free energy](workflows.md#selected-vibrational-initial-states) |
| `examples/nonadiabatic/exchange.inp` | NRPMD scattering from a selected state on a coupled two-state model | ~15 s | [Nonadiabatic NRPMD](nonadiabatic.md) |
| `examples/nonadiabatic/population.inp` | NRPMD populations and coherences after a harmonic spin-boson preparation | pending | [Nonadiabatic populations](nonadiabatic.md) |
| `examples/nonadiabatic/rpsh_population.inp` | Centroid RPSH on Tully I with optional energy-based decoherence | pending | [Nonadiabatic populations](nonadiabatic.md) |
| `examples/nonadiabatic/extensions/` | SM-NRPMD, CA-RP-MASH, BA/DM RPSH, DBOC and complex SOC inputs | pending | [Extended methods](../nonadiabatic-workflow.md) |
| `examples/nonadiabatic/extensions/nah2_sm_nrpmd.inp` | Published NaH2 molecular two-state PES with Cartesian gradients | pending | [Molecular potentials](../molecular-nonadiabatic.md) |
| `examples/rate/research.inp` | Explicit reaction, independent TS checks, regional windows and separate parent/child stages | pending | [Rate input guide](../rate-input-v2.md) |
| `examples/rate/pilot.inp` | Legacy schema-1 TS preparation, PMF and recrossing | ~15 s | [Gas phase and free energy](workflows.md) |
| `examples/rate/general_coordinate.inp` | `site_rate` along a smooth nonlinear centroid coordinate | pending | [General-coordinate rates](../general-coordinate-rates.md) |
| `examples/studies/site_rate.json`, `drpmd.json` | `pyrpmd study` temperature/isotope/convergence scans; inspect with `--plan` | pending | [Rate studies](../rate-studies.md) |
| `examples/network/two_entrances.py` | Two entrances, four TS/PES paths | ~25 s | [Gas phase and free energy](workflows.md) |
| `examples/metadynamics/well_tempered_2d.inp` | 2D well-tempered metadynamics | ~2 s | [Gas phase and free energy](workflows.md) |
| `examples/metadynamics/standard.inp` | Standard constant-height metadynamics on a quartic model | ~1 s | [Gas phase and free energy](workflows.md) |
| `examples/opes/harmonic_2d.inp` | OPES adaptation, frozen-bias relaxation and histogram reweighting | ~3 s | [OPES](../opes.md) |
| `examples/plumed/centroid_metadynamics.py` | Optional PLUMED centroid MetaD with explicit kernel updates | pending | [PLUMED](../plumed.md) |
| `examples/sampling/transition_interface.py` | Four interface ensembles on a confined double-well model | ~7 s | [TIS](../transition-interface-sampling.md) |
| `examples/instanton/eckart.py` | Ring-polymer instanton and quantum cross-check on an Eckart barrier | ~1 s | [Instantons](../instanton.md) |
| `examples/quantum_statistics/harmonic_validation.py` | Higher-order PIMD estimators against analytic harmonic results | ~13 s | [Quantum statistics](../quantum-statistics.md) |
| `examples/nonadiabatic/method_models.py`, `soc_and_rpsh.py`, `thermal_rates.py` | KC-/MV-/isomorphic RPMD, SOC-NRPMD and RPSH, thermal NRPMD rate curves | 1–3 s | [Nonadiabatic](nonadiabatic.md) |

```bash
pyrpmd validate examples/fes/harmonic2d.inp
pyrpmd run examples/fes/harmonic2d.inp --output runs/tutorial-fes2d
pyrpmd analyze runs/tutorial-fes2d --recompute --bins 20 20 --bootstrap 50
```

`examples/rate/minimal.inp` explicitly selects TS-based channel inference as a convenience
mode; research inputs should declare the fragments and bond changes as in `research.inp`.
`examples/drpmd/minimal.inp` uses two reactants and a separation. Other settings use pilot defaults.

`examples/plumed/centroid_metadynamics.py` demonstrates optional PLUMED centroid metadynamics
with explicit kernel updates. It requires a working PLUMED installation; follow the
[PLUMED guide](../plumed.md) before running it.

## Materials, surfaces and PCET

| Input | `task` | Content | Time |
| --- | --- | --- | --- |
| `examples/materials/pimd.inp` | `pimd` | Equilibrium energies and ring size in a periodic model | ~2 s |
| `examples/materials/fixed_slab.inp` | `pimd` | Periodic slab with fixed layers | ~1 s |
| `examples/materials/npt_free.inp` | `pimd` | Python BZP NPT on an ideal gas with an analytic volume check | ~4 s |
| `examples/ensembles/pimd_native.inp`, `pimd_native_npt.inp` | `pimd` | Native C++ NVT/BZP NPT engine with separate thermostat/barostat and checkpoint contracts | ~1 s each |
| `examples/metadynamics/grid_fixed_slab.inp` | `fes` | Grid-accelerated bias on a fixed slab | ~1 s |
| `examples/nonadiabatic/friction_surface.inp` | `diffusion` | Centroid electronic friction on a fixed slab | ~3 s |
| `examples/materials/diffusion.inp` | `diffusion` | Free particle; the analysis flags ballistic motion | ~3 s |
| `examples/materials/site_rate.inp` | `site_rate` | Periodic model, explicit basin and one directed exit | ~25 s |
| `examples/surface/fes2d.inp` | `fes` | 2D surface CVs on a fixed substrate | ~1 min |
| `examples/surface/diffusion.inp` | `surface_rate` | One directed surface hop rate | ~1.5 min |
| `examples/pcet/harmonic_scan.inp` | `pcet` | 18-condition potential/temperature/isotope scan | ~2 s |
| `examples/electrochemistry/fes2d.inp`, `sample_grand_canonical.py`, `her_scan.py` | `fes` / API | Constant-potential effective PES, electron-number sampling, HER current | ≤5 s |

```bash
pyrpmd validate examples/materials/pimd.inp --check-pes
pyrpmd run examples/materials/pimd.inp --output runs/tutorial-pimd
pyrpmd resume runs/tutorial-pimd

pyrpmd validate examples/pcet/harmonic_scan.inp
pyrpmd run examples/pcet/harmonic_scan.inp --output runs/tutorial-pcet
pyrpmd resume runs/tutorial-pcet
```

Further inputs: `examples/pesgen/` ([PES generation](pes-generation.md), [JaxPIP](jaxpip-generation.md)),
`examples/potentials/` (periodic Torch model, on-the-fly tblite/PySCF), `examples/drpmd/*.py`
(Python, Torch, bead, MPI and GPU layouts), `examples/slurm/` (cluster job scripts) and
`examples/analysis/reference-rates.csv` (the `pyrpmd report --reference` format).

The surface and site-rate examples run the full umbrella sampling, constrained-parent and
recrossing pipeline, hence the longer times. Their output is described in [Materials and surfaces](periodic.md) and
[PCET](pcet.md).

## Benchmarks with real models

The [H + H₂ / BKMP2 benchmark](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/h3/README.md) *(partly Chinese)*
runs dRPMD, FES and PMF/recrossing on the published BKMP2 surface, with a pinned source, SHA256 and
energy/gradient reference checks. Building it needs GNU Fortran; the build downloads the third-party
source, whose license and provenance are listed in the benchmark README.

The [Ni–H materials pilot](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/materials/README.md)
runs H and D in nickel with P=1 and 4 on a pinned, checksummed EAM model, which is downloaded
separately.

The [ASE surface example](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/surface/README.md)
connects ASE EMT for a Cu adatom on Cu(111).

The [PCET comparison](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/pcet-model-2026-10-30/README.md)
matches a pinned pyPCET commit to within 0.062% on the harmonic model.

## From an example to a research calculation

1. Copy the closest input into your project and set the task, coordinates and normalization.
2. Replace the model potential with yours; check units, atom order, forces and the domain of
   validity.
3. Run a short pilot and read the diagnostics: failures, masks, overlap, ballistic regimes and the
   recrossing plateau.
4. Increase the bead number, reduce the time step, extend equilibration and production, add
   independent replicas and check the domain, one change at a time.
5. Fix the final protocol, run an independent production and keep the full provenance and
   uncertainties.

`preset="production"` enforces required fields and task restrictions; omitted parameters can
still use defaults. Inspect `pyrpmd validate INPUT --write-resolved resolved.json` before fixing the
protocol. Steps 3 and 4 establish convergence; see [Scientific conventions](science.md).
