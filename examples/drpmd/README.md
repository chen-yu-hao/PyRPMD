# dRPMD examples

These small jobs exercise execution, sampling and restart. The free and Gaussian
repulsive potentials describe no chemical reaction. Their bead count, time step,
sampling duration and `bmax` are illustrative and have not been converged. A zero
reaction count in these examples is not a prediction of a real chemical rate.

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

`minimal.inp` supplies only the task, two reactant structures, their radial COM
distance and the PES. Pilot numerical defaults are recorded in the resolved
input. With no explicit outcome classifier, a supported-element covalent-radius
rule provides **unvalidated geometric labels**: bonds use 1.25 times the radius
sum; only stable, separated and outward-moving components resolve. Newly
observed molecular compositions receive names such as `candidate:H2+O`. These
are composition candidates, not inferred chemical identities. Changed atom
memberships or internal bond graphs with the same composition remain unresolved;
exchange and isomer reactions require an identity-aware Python classifier.
Unknown radii or an inferred graph inconsistent with the supplied reactants
produce an explicit error requiring user-supplied cutoffs or a factory.

The summary records inferred cutoffs, radii, assumptions and
`classification_unvalidated=true`. Event counts and fractions are available,
but chemical rates, cross sections and chemical-outcome bounds remain null.
These defaults are unavailable for `preset="production"`; supply and validate
an explicit classifier. The minimal pilot also uses an unvalidated automatic
impact cutoff and a short propagation limit, so unresolved events are expected.

`free_scattering.inp` reads two XYZ files and a radial COM distance; no transition
state is needed. Its two independent parents equilibrate once, retain their own
full ring-polymer coordinates, momenta and RNG, and advance five thermal steps
before each sampled collision. Five requested collisions are allocated **3 + 2**.
The parent thermal clocks therefore end at **35 and 30**, not five separate
equilibration runs. Production trajectories have no thermostat or fixed COM
constraint. The serial PES mode still permits two simultaneous trajectories.

`thermal_flux` samples the relative collision energy from Gamma(shape=2,
scale=kBT); `b = bmax sqrt(U)` samples uniform impact area. `maxwell` is also
available, with a velocity-weighted estimator. `fixed_energy` requires an explicit
energy and produces a cross section rather than a thermal rate. At launch,
`z = sqrt(R0²-b²)` preserves the specified radial distance. Internal thermal bead
momenta and coordinates are retained, and fragment orientations are isotropic.

`python_pes.py` uses an ordinary importable Python factory. The callable accepts
coordinates `[bead, atom, xyz]` in bohr and returns energies in hartree and forces
in hartree/bohr. Its source file is included in the dRPMD checkpoint identity.
`bead.py` adds two persistent PES processes to each of two trajectory processes.
For this tiny analytic PES the communication overhead exceeds the useful work;
bead parallelism is intended for substantially more expensive potentials.

## GPU placement

Install a CUDA-capable PyTorch environment, then run:

```bash
python -m pyrpmd run examples/drpmd/gpu.py --output runs/drpmd-gpu
```

The example requests two visible GPUs and two independent trajectory processes
per GPU. Change `TRAJECTORIES_PER_GPU` to 1 for one process per card. Each process
loads a separate model and batches all its beads in one PES call, so multiply
model, activations and trajectory memory by the number of processes on the card.
This is process concurrency, not a shared model or an asynchronous stale-force
scheme. Requested CUDA devices must exist; a CPU fallback is not used.

Device IDs refer to the process's `CUDA_VISIBLE_DEVICES` mapping. For MPI GPU
jobs, each node must expose the same configured device list to all its local
ranks, and run `len(devices) * trajectories_per_gpu` local ranks. Placement uses
the node-local MPI rank. The accompanying two-node Slurm GPU template requests
two GPUs and four ranks per node. GPU execution is hardware-dependent; local
CPU and MPI tests do not establish CUDA correctness or performance.

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

`results/summary.json` reports product, nonreactive, unresolved and failed counts.
A trajectory needs a stable, separated, outward-moving centroid classification
before it resolves. Missing classification or exhausted propagation time remains
unresolved. Unknown outcomes suppress the point rate and produce observed and
possible outcome bounds, which are not statistical confidence limits. The
formula-based classifier cannot establish identity exchange or isomers; use
an importable `outcomes.factory` classifier for those reactions.

`states/initial-*.npz` retain full thermal and launched q/p. The trajectory NPZ
files retain centroid traces, energy traces and final full q/p. Atomic parent and
child checkpoints store RNG, progress and classification state. Use `resume`
after interruption; a completed trajectory is counted only once. Keep
`sampling.parents` fixed when changing CPU worker count on restart. Checkpoints
verify scientific inputs and declared local PES/model/factory file contents;
they do not fingerprint every transitive Python dependency or a remote i-PI
engine, which must remain consistent in the run environment.
