# Execution, parallelism and restart

## Integrator and execution path

The integrator defines the numerical dynamics. The native/Python and block settings choose how that
dynamics is executed; both paths use the same potential, masses, temperature, bias and random
streams.

```toml
[dynamics]
integrator = "cayley-bcb"
timestep = "0.1 fs"
native = true
loop_backend = "auto"
block_steps = 256
```

| Setting | Meaning |
| --- | --- |
| `native=true` | Use compiled integrator substeps when available, NumPy otherwise |
| `native=false` | Use the NumPy integrator |
| `native="required"` | Require compiled support for the chosen integrator |
| `loop_backend="auto"` | Fuse whole propagation blocks when the PES, CVs and bias allow it; otherwise use a Python loop |
| `loop_backend="python"` | Python outer loop with compiled substeps where available |
| `loop_backend="native"` | Require fused whole-block propagation; error if unavailable |
| `block_steps` | Execution block size, 1–65536 |

Periodic and fixed-atom systems, PILE-L sampling and metadynamics run in the Python outer loop with
compiled deterministic substeps. `pimd` with `sampling.engine="native"` is different: the C++
driver advances each replica from checkpoint to checkpoint without a Python step loop, and
`block_steps` does not change its checkpoint cadence; see
[Materials and surfaces](periodic.md#native-pimd-engine). The `pimd`, `diffusion` and `metadynamics` summaries report
native and Python step counts with fallback reasons under `propagation`; in the API,
`Dynamics.loop_stats` gives the same counters.

[NRPMD](nonadiabatic.md) propagates the electronic matrix with CPU NumPy, serially or with local
trajectory processes; its nuclear free-ring substeps use the compiled kernels.

## CPU trajectory and bead parallelism

Parallelize independent windows, parent chains or replicas first. Split the beads of one trajectory
over several processes when a single PES call is expensive.

```toml
[parallel]
backend = "process"
mode = "serial"
trajectory_workers = 4
threads_per_rank = 1
```

`mode="serial"` keeps each PES call in one process and runs four trajectory processes. With
`mode="bead"` and `bead_workers=2`, each concurrent trajectory gets a persistent pool of two bead PES
processes. `collision.trajectories` is always the total for the task.

For cheap analytic potentials, process communication dominates the cost. Choose trajectory
processes × bead processes × potential/BLAS threads to match the allocated CPUs, typically with:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## The G × B MPI topology

```toml
[parallel]
backend = "mpi"
mode = "bead"
trajectory_groups = 2
bead_workers = 2
threads_per_rank = 1
pes_ranks = 1
coordinator_ranks = 0
scheduling = "static"
```

This uses four ranks: two trajectory groups with two bead PES ranks each. The first rank of each
group advances the state while the others evaluate PES batches; world rank 0 collects the results.

```bash
mpiexec -n 4 pyrpmd run examples/drpmd/mpi.py --output runs/mpi-first
mpiexec -n 4 pyrpmd resume runs/mpi-first
```

The world size equals `trajectory_groups * bead_workers`. With static scheduling, `pes_ranks=1` and
`coordinator_ranks=0`; each bead PES partition runs on its assigned rank. All nodes need the same code,
environment and model paths and a shared output directory.

## Dynamic MPI work queue

For dRPMD trajectories with different stopping times, select dynamic scheduling:

```toml
[parallel]
backend = "mpi"
mode = "bead"
scheduling = "dynamic"
coordinator_ranks = 1
trajectory_groups = 2
bead_workers = 2
threads_per_rank = 1
```

This uses `1 + trajectory_groups * bead_workers` ranks: five in this example. World rank 0
coordinates the queue; each free group requests another task. The static layout remains the
default. `pes_ranks=1` still applies. Dynamic scheduling is an MPI option, not a new local
process backend. In GPU mode the coordinator consumes no GPU slot.

```bash
mpiexec -n 5 pyrpmd run examples/drpmd/mpi_dynamic.py --output runs/dynamic-mpi
```

For dRPMD, persistent parent chains first generate restartable launch samples. Individual
collision trajectories then enter the dynamic queue, so an early-stopping trajectory immediately
frees its group. Each task keeps its global index, random stream and checkpoint; returned results
retain task order. Set `sampling.parents` explicitly when comparing layouts or changing worker
counts. A task error stops new dispatch, drains active groups, closes provider contexts and
reports the error collectively. MPI process crashes and network failures remain launcher-level
failures.

## GPU placement

```toml
[parallel]
backend = "process"
mode = "gpu"
devices = [0, 1]
trajectories_per_gpu = 2
bead_workers = 1
```

This starts four trajectory processes, two per card. Device numbers are local indices after
`CUDA_VISIBLE_DEVICES`; multi-node MPI routes by node-local rank. A request for CUDA fails if the
device is unavailable or the model runs on the CPU. GPU mode uses one bead worker per trajectory.

## Saving and restarting

```bash
pyrpmd resume runs/example
pyrpmd resume runs/example --workers 4 --loop-backend auto --block-steps 128
pyrpmd resume runs/adaptive --adaptive-max-rounds 12 --adaptive-max-windows 192 --adaptive-max-pilot-steps 4000
pyrpmd extend runs/selected --max-time '20 ps'
```

The adaptive options raise the budget of an exhausted adaptive-umbrella pilot. `extend` raises the
total propagation time of a dRPMD ensemble and continues only unresolved trajectories from their
saved q/p.

A restart reads the saved resolved input, the system or PCET model, q/p, RNG state, stage and
history. A finished task stays finished, and a checkpoint that fails verification stops the restart
with an error. Worker layout, backend and block size can change between restarts. Worker processes reuse their
potential objects across tasks by default (`parallel.reuse_potentials=true`); see
[expensive potentials](../performance-dynamics.md). In dRPMD, the
number of parents follows the number of execution slots when `sampling.parents` is omitted, so set
`parents` explicitly if you plan to change the worker count.

Temperature, bead number, isotopes, potential, time step, integrator, CVs and sampling settings form
the run identity; changing any of them needs a new output directory. Hashes of the model and plugin
files are part of the identity; pin indirect Python dependencies, system libraries and remote i-PI
services through your environment.

To inspect one checkpoint, pick an `.npz` file from the run directory:

```bash
pyrpmd checkpoint runs/example/checkpoints/ACTUAL_FILE.npz --require-integrity
```

Replace `ACTUAL_FILE` with a real file name; names differ between workflows. NPZ containers are
verified with per-array SHA256, and old containers without checksums report `unverified_legacy`.
Random streams reproduce on the same platform; different hardware or reduction orders can change the
last bits.

## Failures

`Dynamics.propagate()` commits a block only when every step succeeds; `propagate_block()` keeps the
successful prefix. Workflows commit state at thermostat, observation, hill deposition and checkpoint
boundaries. After a PES failure, the exception and the saved stage information show where the run
stopped; fix the cause and resume from the last checkpoint.

Detailed contracts are in [parallelism](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/parallel.md) *(Chinese)*,
[native propagation](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/native-propagation.md),
the [block API](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/block-propagation.md) *(Chinese)* and
[checkpoints](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/checkpoints.md).
