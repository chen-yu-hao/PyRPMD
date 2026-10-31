# dRPMD examples

These jobs show the full direct-collision workflow in seconds: thermal parent chains, sampled
collisions, outcome classification, restart, bead-parallel potentials, GPU placement and MPI.

> **Scope.** The free and Gaussian repulsive potentials here describe no chemical reaction, and
> the bead count, time step, sampling duration and `bmax` are illustrative pilot settings. Use
> these inputs to learn the workflow; a zero reaction count is not a chemical prediction.

Run from the repository root after installing PyRPMD:

```bash
python -m pyrpmd validate examples/drpmd/free_scattering.inp
python -m pyrpmd run examples/drpmd/minimal.inp --output runs/drpmd-minimal
python -m pyrpmd run examples/drpmd/free_scattering.inp --output runs/drpmd-free
python -m pyrpmd resume runs/drpmd-free
python -m pyrpmd run examples/drpmd/python_pes.py --output runs/drpmd-python
python -m pyrpmd run examples/drpmd/bead.py --output runs/drpmd-bead
mpiexec -n 4 python -m pyrpmd run examples/drpmd/mpi.py --output runs/drpmd-mpi
mpiexec -n 5 python -m pyrpmd run examples/drpmd/mpi_dynamic.py --output runs/drpmd-mpi-dynamic
```

## Persistent parent chains

`free_scattering.inp` needs only two XYZ files and a radial COM distance; no transition state.
Its two independent parents equilibrate once, keep their own full ring-polymer coordinates,
momenta and RNG, and advance five thermal steps before each sampled collision. Five requested
collisions are allocated **3 + 2**, so the parent thermal clocks end at **35 and 30** — one
equilibration per parent instead of five separate ones. Production trajectories run without a
thermostat or a fixed COM constraint. The serial PES mode still runs two trajectories at once.

## Collision sampling

| `collision.sampling` | Relative energy | Result |
| --- | --- | --- |
| `thermal_flux` | Gamma(shape=2, scale=kBT) | Thermal rate |
| `maxwell` | Maxwell, with a velocity-weighted estimator | Thermal rate |
| `fixed_energy` | Explicit `collision.energy` | Cross section |

`b = bmax sqrt(U)` samples a uniform impact area, and `z = sqrt(R0²-b²)` keeps the specified radial
distance at launch. Internal thermal bead momenta and coordinates are retained, and fragment
orientations are isotropic.

## Outcome classification

`minimal.inp` gives only the task, two reactant structures, their radial COM distance and the PES;
the pilot numerical defaults are recorded in the resolved input. Without an explicit classifier,
a covalent-radius rule for supported elements labels outcomes geometrically:

- bonds use 1.25 times the radius sum, and only stable, separated, outward-moving components
  resolve;
- new molecular compositions get names such as `candidate:H2+O` — composition candidates, not
  chemical identities;
- the summary records the inferred cutoffs, radii and assumptions with
  `classification_unvalidated=true`, and reports event counts and fractions while chemical rates,
  cross sections and chemical-outcome bounds stay null.

To get chemical rates, supply and validate an explicit classifier; `preset="production"` requires
one. Exchange and isomer reactions, where atom memberships or bond graphs change at fixed
composition, need an identity-aware Python classifier. Unknown radii, or an inferred graph that
contradicts the supplied reactants, stop with an error asking for explicit cutoffs or a factory.
The minimal pilot also uses an automatic impact cutoff and a short propagation limit, so expect
some unresolved events.

## Custom potentials and bead parallelism

`python_pes.py` uses an ordinary importable Python factory. The callable takes coordinates
`[bead, atom, xyz]` in bohr and returns energies in hartree and forces in hartree/bohr; its source
file is part of the dRPMD checkpoint identity.

`bead.py` adds two persistent PES processes to each of two trajectory processes. Bead parallelism
pays off for expensive potentials; on this tiny analytic PES, communication dominates.

## GPU placement

Install a CUDA-capable PyTorch environment, then run:

```bash
python -m pyrpmd run examples/drpmd/gpu.py --output runs/drpmd-gpu
```

The example requests two visible GPUs and two independent trajectory processes per GPU; set
`TRAJECTORIES_PER_GPU = 1` for one process per card. Each process loads its own model and batches
all its beads in one PES call, so budget model, activation and trajectory memory per process.
Requested CUDA devices must exist; there is no CPU fallback.

Device IDs follow each process's `CUDA_VISIBLE_DEVICES`. For MPI GPU jobs, each node exposes the
same device list to all its local ranks and runs `len(devices) * trajectories_per_gpu` local ranks;
placement uses the node-local MPI rank. The two-node Slurm GPU template requests two GPUs and four
ranks per node. CUDA correctness and performance depend on the hardware and are not covered by the
CPU and MPI tests.

## MPI and Slurm

MPI requires a matching MPI runtime and mpi4py installation. Static scheduling
uses `world_size = trajectory_groups * bead_workers`, with
`pes_ranks = 1` and no dedicated coordinator. Group leaders advance trajectories;
the other group ranks persistently evaluate their bead subsets. Only a group's
ranks synchronize at a force call. All nodes need the same importable plugins,
PES files and a shared output directory. `../slurm/` contains CPU and GPU templates
whose module/environment and Slurm MPI plugin must match the actual cluster.

`mpi_dynamic.py` adds `scheduling="dynamic"` and `coordinator_ranks=1`, requiring
`world_size = 1 + trajectory_groups * bead_workers`. Its default is five ranks:
one coordinator and two groups of two bead ranks. Parent sampling completes
first; the coordinator then gives individual saved collisions to the first
available group, so a short collision does not leave its group idle while a
different group has queued work. The two thermal parents and five global
collision IDs are unchanged. Set `PYRPMD_TRAJECTORY_GROUPS` and
`PYRPMD_BEAD_WORKERS` for the execution topology, and keep `PYRPMD_PARENTS`
(default 2) fixed when changing resources on resume. The coordinator consumes
one rank but no GPU slot. See `docs/parallel.md` for error handling and boundaries.

The opt-in integration driver is `mpiexec -n 4 python tests/mpi_integration.py`.
It checks communication, ordered tasks, collective error propagation, persistent
parents and restart. Running four ranks on one host does not measure scaling
across physical nodes. On restricted hosts where MPICH reports an unsupported
cross-process-memory syscall, its TCP transport can be selected with
`MPIR_CVAR_CH4_CMA_ENABLE=0 UCX_TLS=tcp,self`; this workaround is runtime-specific.
`mpiexec -n 5 python tests/mpi_dynamic_integration.py` checks the dynamic queue,
group cleanup on failure, and exact static/dynamic free-PES states and restart.

## Results and restart

`results/summary.json` reports product, nonreactive, unresolved and failed counts. A trajectory
resolves once its centroid classification is stable, separated and moving outward; one that runs
out of propagation time stays unresolved. When outcomes are unknown, PyRPMD reports observed and
possible outcome bounds (not statistical confidence limits) in place of a point rate.

| File | Contents |
| --- | --- |
| `states/initial-*.npz` | Full thermal and launched q/p |
| Trajectory NPZ files | Centroid and energy traces, final full q/p |
| Parent and child checkpoints | RNG, progress and classification state, written atomically |

Use `resume` after an interruption; each completed trajectory is counted once. Keep
`sampling.parents` fixed when changing the CPU worker count on restart. Checkpoints verify the
scientific inputs and the declared local PES, model and factory files; keep transitive Python
dependencies and any remote i-PI engine unchanged between runs.
