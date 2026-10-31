# Execution, parallelism and restart

## Integration method versus execution path

The integrator determines the numerical dynamics; native/Python and block settings determine how
that same calculation is executed. Choosing C++ must not change the potential, masses, temperature,
bias or random sampling semantics.

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
| `native=true` | Prefer compiled integrator substeps; fall back to the reference implementation |
| `native=false` | Use the NumPy integration path |
| `native="required"` | Compiled support for the chosen integrator is mandatory |
| `loop_backend="auto"` | Fuse whole propagation blocks when the PES/CV/bias qualify, otherwise use the compatible path |
| `loop_backend="python"` | Python outer loop; integrator substeps may still be compiled |
| `loop_backend="native"` | Strictly require fused whole-block propagation; error if not possible |
| `block_steps` | Execution block size; allowed range 1–65536 |

Periodic or fixed-atom systems currently take the compatible Python outer loop and may use compiled
deterministic integrator substeps. PILE-L sampling and dynamic metadynamics histories have their own
capability limits too. Look at the propagation statistics and fallback reasons of a run; a
successful installation does not imply that every path is fused.

The electronic matrix propagation of experimental [NRPMD](nonadiabatic.md) uses CPU/NumPy, serially
or with local trajectory processes; MPI, bead-worker splitting, GPU and fused native electronic
propagation are not supported. The nuclear free-ring substeps can use the existing compiled kernels.

## CPU trajectory and bead parallelism

Usually parallelize independent windows, parent chains or replicas first. Only when a single PES call
is expensive should bead batches be split over several processes.

```toml
[parallel]
backend = "process"
mode = "serial"
trajectory_workers = 4
threads_per_rank = 1
```

Here `mode="serial"` means one PES call does not split beads, while there are still four independent
trajectory processes. With `mode="bead"` and `bead_workers=2`, every concurrent trajectory owns
persistent bead PES processes. None of these settings reinterpret `collision.trajectories` as a
"per worker" count: that field is always the total for the whole task.

A very fast analytic potential can be limited by inter-process communication. Balance
trajectories × bead processes × external potential/BLAS threads so that the allocated CPUs are not
oversubscribed. A common environment setting is:

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

This corresponds to four ranks: two groups of independent trajectories, each with two bead potential
ranks. The first rank of each group advances the state and the others serve PES batches; world rank
0 collects the results.

```bash
mpiexec -n 4 pyrpmd run examples/drpmd/mpi.py --output runs/mpi-first
mpiexec -n 4 pyrpmd resume runs/mpi-first
```

The world size must equal `trajectory_groups * bead_workers`. The current implementation uses static
groups with `pes_ranks=1` and `coordinator_ranks=0`; an external PES that nests its own MPI
communicator is not supported. All nodes need the same code, environment and model paths, plus a
shared output directory. A four-rank test on one machine and a scaling measurement across physical
nodes are different kinds of evidence.

## GPU placement

```toml
[parallel]
backend = "process"
mode = "gpu"
devices = [0, 1]
trajectories_per_gpu = 2
bead_workers = 1
```

This creates four independent trajectory processes with two model instances per card. Device numbers
are local indices after the `CUDA_VISIBLE_DEVICES` mapping; multi-node MPI routes by node-local rank.
Requesting CUDA while the model actually runs on the CPU, or an unavailable device, is an error. GPU
mode does not split beads over processes at the same time. The task must still be within the
provider's supported scope.

## Saving and restarting

```bash
pyrpmd resume runs/example
pyrpmd resume runs/example --workers 4 --loop-backend auto --block-steps 128
```

A restart reads the saved resolved input, the system/PCET model, q/p, RNG, stage and history state.
A finished task never adds trajectories again, and a failed checkpoint verification is never skipped
silently. Changing only the worker layout, backend and block size is an execution-level adjustment,
but individual workflows still have constraints. For example, if dRPMD omits `sampling.parents`, the
number of parents may follow the number of execution slots; if changing workers then changes the
parent identities, the calculation cannot continue as the same task. For portable scheduling, set
parents explicitly.

Temperature, bead number, isotopes, potential, time step, integrator, CVs and the scientific sampling
settings are part of the run identity. Changing them needs a new output directory. Hashes of the
model and explicit plugin files are checked; indirect Python dependencies, system shared libraries
and the state of a remote i-PI service must be pinned by the environment.

To inspect a single file, pick an `.npz` that actually exists in the run directory:

```bash
pyrpmd checkpoint runs/example/checkpoints/ACTUAL_FILE.npz --require-integrity
```

`ACTUAL_FILE` is a placeholder; file names differ between workflows. NPZ containers are verified with
per-array SHA256; the `unverified_legacy` status of an old container means something different from a
passed verification. The same random stream is not guaranteed to be bitwise identical across hardware
or reduction orders either.

## State boundaries on failure

The low-level `Dynamics.propagate()` commits a block only when the whole block succeeds;
`propagate_block()` keeps the successful prefix of steps. Workflows commit their state at thermostat,
observation, hill deposition and checkpoint event boundaries. When the PES fails, read the exception
and the preserved stage information; do not treat the last frame as a complete production result.

The detailed contracts are in [parallelism](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/parallel.md) *(Chinese)*,
[native propagation](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/native-propagation.md),
the [block API](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/block-propagation.md) *(Chinese)* and
[checkpoints](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/checkpoints.md).
