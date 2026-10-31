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
compiled deterministic substeps. The `pimd`, `diffusion` and `metadynamics` summaries report
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

The world size equals `trajectory_groups * bead_workers`. Groups are static, with `pes_ranks=1` and
`coordinator_ranks=0`, so each PES call runs on a single rank. All nodes need the same code,
environment and model paths and a shared output directory.

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
```

A restart reads the saved resolved input, the system or PCET model, q/p, RNG state, stage and
history. A finished task stays finished, and a checkpoint that fails verification stops the restart
with an error. Worker layout, backend and block size can change between restarts. In dRPMD, the
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
