# Example index and learning path

Run every command from the root of the source repository. Run `pyrpmd validate INPUT` first, then
run with a new `--output` directory; add `--check-pes` to exercise the actual PES. Environment
requirements for external models, MPI and GPU are in [Installation](installation.md). The inputs
below are demonstrations or limited validations, not recipes that converge automatically for
production.

## Starting points without external models

"Time" is the measured order of magnitude of `run` on one CPU core with `OMP_NUM_THREADS=1`, meant
only for planning practice sessions; actual times depend on hardware and parallel settings.

| Input/program | Content | Time | Recommended reading |
| --- | --- | --- | --- |
| `examples/manual/quickstart.inp` | 1D harmonic FES with three umbrella windows and 6 independent streams | ~2 s | [Quickstart](quickstart.md) |
| `examples/manual/low_level.py` | Complete 100-step program with `Dynamics` / `State` | <1 s | [API](api.md) |
| `examples/fes/harmonic2d.inp` | Two centroid CVs, 2D MBAR and masks | ~6 s | [Gas phase and free energy](workflows.md) |
| `examples/fes/custom_input.py` | Custom 2D CV with a full Jacobian | ~7 s | [Potentials and plugins](potentials.md) |
| `examples/fes/adaptive_harmonic.inp` | Pilot window insertion, then frozen independent production | ~4 s | [Gas phase and free energy](workflows.md) |
| `examples/drpmd/free_scattering.inp` | Two persistent parents sharing 5 direct collisions | ~2 s | [Gas phase and free energy](workflows.md) |
| `examples/nonadiabatic/exchange.inp` | Experimental selected-incoming-state scattering on a coupled artificial matrix potential | ~15 s | [Nonadiabatic NRPMD](nonadiabatic.md) |
| `examples/rate/pilot.inp` | TS preparation, PMF and recrossing on a teaching reaction potential | ~15 s | [Gas phase and free energy](workflows.md) |
| `examples/network/two_entrances.py` | Two entrances, four independent TS/PES paths | ~25 s | [Gas phase and free energy](workflows.md) |
| `examples/metadynamics/well_tempered_2d.inp` | 2D well-tempered history-dependent bias | ~2 s | [Gas phase and free energy](workflows.md) |

```bash
pyrpmd validate examples/fes/harmonic2d.inp
pyrpmd run examples/fes/harmonic2d.inp --output runs/tutorial-fes2d
pyrpmd analyze runs/tutorial-fes2d --recompute --bins 20 20 --bootstrap 50
```

The free-scattering example uses a zero potential; it tests launching, classification, output and
restart, not the chemistry of H + H₂. The external potential library path in `minimal.inp` is a
dependency you have to provide; to get started, prefer the self-contained examples in the table.

## Materials, surfaces and PCET

| Input | Actual `task` | Model and purpose of the result | Time |
| --- | --- | --- | --- |
| `examples/materials/pimd.inp` | `pimd` | Finite-P harmonic equilibrium energies and ring size | ~2 s |
| `examples/materials/diffusion.inp` | `diffusion` | Ballistic free-particle counterexample; no converged diffusion coefficient is accepted | ~3 s |
| `examples/materials/site_rate.inp` | `site_rate` | Periodic teaching potential, explicit basin and one directed exit | ~25 s |
| `examples/surface/fes2d.inp` | `fes` | 2D surface CVs on a fixed substrate | ~1 min |
| `examples/surface/diffusion.inp` | `surface_rate` | One directed surface hop rate; not the MSD diffusion workflow | ~1.5 min |
| `examples/pcet/harmonic_scan.inp` | `pcet` | 18-condition potential/temperature/isotope scan of local harmonic wells | ~2 s |

```bash
pyrpmd validate examples/materials/pimd.inp --check-pes
pyrpmd run examples/materials/pimd.inp --output runs/tutorial-pimd
pyrpmd resume runs/tutorial-pimd

pyrpmd validate examples/pcet/harmonic_scan.inp
pyrpmd run examples/pcet/harmonic_scan.inp --output runs/tutorial-pcet
pyrpmd resume runs/tutorial-pcet
```

The surface and site-rate examples include umbrella sampling, constrained parents and recrossing, so
they take noticeably longer than the Quickstart. Interpret their output with the
[periodic task guide](periodic.md) and the [PCET guide](pcet.md).

## External models and real-system benchmarks

The [H + H₂ / BKMP2 benchmark](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/h3/README.md) *(partly Chinese)*
pins the potential source and SHA256 and includes energy/gradient reference checks, dRPMD, FES and
PMF/recrossing. Building it needs GNU Fortran; read the license and source notes after downloading
the third-party code.

The [Ni–H materials pilot](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/materials/README.md)
pins the source and checksum of a real EAM model and compares short H/D trajectories with P=1/4. The
model table is not distributed with the repository and must be downloaded explicitly. A short NVE
energy check does not mean long-time diffusion has developed.

The [ASE surface example](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/surface/README.md) *(partly Chinese)*
describes the EMT connection for Cu(111)/a Cu adatom. It validates one specific calculator and a short
workflow; it does not extend to every ASE machine-learning model.

The [independent PCET comparison](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/pcet-model-2026-10-30/README.md)
compares the same local model with a pinned external pyPCET commit; it does not reproduce a real
electrode experiment.

## Turning an example into a research calculation

1. Copy the input closest to your problem into your own project directory and make the task,
   coordinates and normalization explicit.
2. Replace the teaching model with the real potential; check units, atom order, forces and the domain
   of validity first.
3. Run a small pilot and look at the diagnostics: failures, masks, overlap, ballistic regimes or the
   recrossing plateau.
4. Increase the bead number, reduce the time step, extend equilibration/production, add independent
   replicas and check the domain, one at a time.
5. Freeze the final protocol, run an independent production and keep complete provenance and
   uncertainty information.

`preset="production"` cannot replace the physical judgment at each step. See the
[scientific conventions](science.md).
