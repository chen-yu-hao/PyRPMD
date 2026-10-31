# Example index and learning path

Every task has a runnable example on an analytic model that finishes in seconds to about a minute
on one core, so you can see a complete workflow and its output before connecting your own
potential. Run every command from the repository root: `pyrpmd validate INPUT` first (add `--check-pes` to
construct the PES), then `pyrpmd run` with a new `--output` directory. External models, MPI and GPU
need the dependencies listed in [Installation](installation.md).

## Examples without external models

"Time" is the wall time of `run` on one CPU core with `OMP_NUM_THREADS=1`.

| Input/program | Content | Time | Chapter |
| --- | --- | --- | --- |
| `examples/manual/quickstart.inp` | 1D harmonic FES with three umbrella windows and 6 streams | ~2 s | [Quickstart](quickstart.md) |
| `examples/manual/low_level.py` | 100-step program with `Dynamics` / `State` | <1 s | [API](api.md) |
| `examples/fes/harmonic2d.inp` | Two centroid CVs, 2D MBAR and masks | ~6 s | [Gas phase and free energy](workflows.md) |
| `examples/fes/replica_exchange_2d.inp` | PILE-L, replica-exchange umbrellas and synchronized bootstrap | — | [Replica exchange](../replica-exchange.md) |
| `examples/fes/custom_input.py` | Custom 2D CV with a full Jacobian | ~7 s | [Potentials and plugins](potentials.md) |
| `examples/fes/adaptive_harmonic.inp` | Pilot window insertion, then frozen production | ~4 s | [Gas phase and free energy](workflows.md) |
| `examples/drpmd/free_scattering.inp` | Two parent chains sharing 5 collisions on V=0 | ~2 s | [Gas phase and free energy](workflows.md) |
| `examples/nonadiabatic/exchange.inp` | NRPMD scattering from a selected state on a coupled two-state model | ~15 s | [Nonadiabatic NRPMD](nonadiabatic.md) |
| `examples/rate/pilot.inp` | TS preparation, PMF and recrossing on an analytic reaction potential | ~15 s | [Gas phase and free energy](workflows.md) |
| `examples/network/two_entrances.py` | Two entrances, four TS/PES paths | ~25 s | [Gas phase and free energy](workflows.md) |
| `examples/metadynamics/well_tempered_2d.inp` | 2D well-tempered metadynamics | ~2 s | [Gas phase and free energy](workflows.md) |
| `examples/opes/harmonic_2d.inp` | OPES adaptation, frozen-bias relaxation and histogram reweighting | — | [OPES](../opes.md) |
| `examples/plumed/centroid_metadynamics.py` | Optional PLUMED centroid MetaD with explicit kernel updates | — | [PLUMED](../plumed.md) |
| `examples/sampling/transition_interface.py` | Four interface ensembles on a confined double-well model | — | [TIS](../transition-interface-sampling.md) |

```bash
pyrpmd validate examples/fes/harmonic2d.inp
pyrpmd run examples/fes/harmonic2d.inp --output runs/tutorial-fes2d
pyrpmd analyze runs/tutorial-fes2d --recompute --bins 20 20 --bootstrap 50
```

`examples/rate/minimal.inp` (TS only) and `examples/drpmd/minimal.inp` (two reactants and a
separation) show the smallest valid inputs, with every other setting at its pilot default.

## Materials, surfaces and PCET

| Input | `task` | Content | Time |
| --- | --- | --- | --- |
| `examples/materials/pimd.inp` | `pimd` | Equilibrium energies and ring size in a periodic model | ~2 s |
| `examples/ensembles/pimd_native.inp`, `pimd_native_npt.inp` | `pimd` | Native ensemble engine with separate thermostat/barostat and checkpoint contracts | — |
| `examples/materials/diffusion.inp` | `diffusion` | Free particle; the analysis flags ballistic motion | ~3 s |
| `examples/materials/site_rate.inp` | `site_rate` | Periodic model, explicit basin and one directed exit | ~25 s |
| `examples/surface/fes2d.inp` | `fes` | 2D surface CVs on a fixed substrate | ~1 min |
| `examples/surface/diffusion.inp` | `surface_rate` | One directed surface hop rate | ~1.5 min |
| `examples/pcet/harmonic_scan.inp` | `pcet` | 18-condition potential/temperature/isotope scan | ~2 s |

```bash
pyrpmd validate examples/materials/pimd.inp --check-pes
pyrpmd run examples/materials/pimd.inp --output runs/tutorial-pimd
pyrpmd resume runs/tutorial-pimd

pyrpmd validate examples/pcet/harmonic_scan.inp
pyrpmd run examples/pcet/harmonic_scan.inp --output runs/tutorial-pcet
pyrpmd resume runs/tutorial-pcet
```

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
