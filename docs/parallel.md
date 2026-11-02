# Parallel execution and reproducibility

**English** · [中文](zh/parallel.md)

PyRPMD configures independent-task transport through `parallel.backend` and potential
evaluation within one trajectory through `parallel.mode`. The `serial`, `bead` and
`gpu` potential modes can all run several trajectories concurrently. `serial` means
only that one PES call does not split beads. Standard RPMD's independent window,
parent and child tasks use the same executor.

| Configuration | Execution | Typical use |
| --- | --- | --- |
| `backend="serial"`, `trajectory_workers=1` | Independent tasks run in the calling process | Debugging, lightweight tasks |
| `trajectory_workers=n > 1`, without MPI | `spawn` starts n trajectory processes with static task assignment | Parallel trajectories, windows or replicas on one host |
| `mode="bead"`, `bead_workers=B > 1`, without MPI | Each concurrent trajectory owns B persistent PES subprocesses | Expensive or non-reentrant PES implementations |
| `backend="mpi"` | G trajectory groups, each with B ranks | Multiple ranks on one host or MPI across nodes |
| `mode="gpu"` | Each trajectory process loads its own GPU model and batches all beads in one call | One or several GPUs, with one or more trajectories per GPU |

`parallel.trajectory_batch_size` controls a separate dimension: how many
independent workflow tasks share a PES request within each worker. Fixed-cell
legacy materials `pimd` / `diffusion` support it alongside the rate workflows
listed in [trajectory batching](batched-dynamics.md). For example, eight
replicas, two trajectory workers and batch size four schedule two groups of
four replicas. This keeps eight independent trajectories; adding workers or batch
slots does not increase that count. Cross-trajectory batching requires
`bead_workers=1` and a provider with an independent-frame contract.

`threads_per_rank` controls compiled OpenMP dynamics kernels and requires an actual
OpenMP-enabled build. Local task batches use `threadpoolctl` to limit loaded BLAS
thread pools to one thread. They also set OpenBLAS, MKL, BLIS and Accelerate thread
environment variables within the task scope so libraries imported during a task
normally use one thread. Both loaded-pool settings and environment variables are
restored when a task finishes or raises an exception. This applies even when there
is only one task executing directly in the calling process. Persistent bead workers
limit thread pools at initialization and retain those limits until exit, avoiding a
library scan on every PES call. This changes BLAS settings only; it does not modify
`OMP_NUM_THREADS` or the dynamics kernel's `threads_per_rank`.

BLAS settings are process-wide and cannot isolate unrelated numerical work in other
Python threads of the same process. Configure MPI and external engines' own thread
pools for the deployment. Avoid exceeding allocated CPUs with trajectory count ×
bead workers × engine threads. Before launching MPI, settings such as
`OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1` can help. For very fast analytical
potentials, process communication can cost more than evaluation; increasing the
number of independent trajectories is usually preferable.

## MPI topology

Static scheduling requires `MPI world size = G × B`, where
`G = trajectory_groups` and `B = bead_workers`. Each group leader advances the
integrator and trajectory state; the other ranks persistently service bead-PES
requests. Each force call uses the coordinates submitted for that call, preserves
bead order and never reuses stale asynchronous forces. Steps synchronize within
their own group without a world barrier.

```toml
[parallel]
backend = "mpi"
mode = "bead"
trajectory_groups = 4
bead_workers = 2
threads_per_rank = 1
pes_ranks = 1
coordinator_ranks = 0
scheduling = "static"
```

This static setup requires eight MPI ranks. `pes_ranks=1` is still the only supported
value; a nested Q level of MPI inside one PES evaluation is not implemented. Static
scheduling remains the default.

Across nodes, use compatible MPI/mpi4py, Python, PyRPMD and potential libraries on
every node. Python factories must be importable in every execution process, and data
files must occupy the same paths. The output directory must be shared storage visible
to all nodes. Only a group leader writes that group's trajectory files. World rank 0
performs final statistics, then broadcasts stage results. See `examples/slurm/` for
templates; the `srun --mpi` plugin, module environment and allocation must match the
actual cluster.

### Dynamic MPI scheduling

Set `scheduling="dynamic"` to give the next independent task to whichever
trajectory group finishes first. World rank 0 becomes a dedicated coordinator;
the remaining ranks form `G` fixed groups of `B` bead ranks. Consequently,
**world size must be `1 + G × B`**. For example:

```toml
[parallel]
backend = "mpi"
mode = "bead"
trajectory_groups = 4
bead_workers = 2
coordinator_ranks = 1
scheduling = "dynamic"

[sampling]
parents = 4
```

This needs nine ranks: `mpiexec -n 9 pyrpmd run input.inp`. Dynamic scheduling
defaults `coordinator_ranks` to 1; explicitly requesting 0 is an error. Static
scheduling requires 0. The coordinator does not integrate trajectories or
evaluate forces. Each worker group retains its own bead communicator, and an
integration step never waits at a world barrier.

Eligible potentials acquired through `task_potential` remain cached for the
worker group's complete dynamic queue, rather than being reconstructed for
each dispatched task. Opt-out and explicitly stateful providers are still
created and closed per task. Unclosed directly owned MPI bead providers
(`B>1`) are cleaned up at task boundaries; other directly constructed
`make_task_potential` providers remain caller-owned. Cache cleanup also runs on task,
serialization and final-close errors; a dynamic worker cannot keep a failed
lease alive while waiting for coordinator shutdown.

For dRPMD, dynamic mode first samples the persistent parent chains and writes
the usual `states/initial-*.npz` launch files. It then queues **individual
collisions**, rather than whole parent chains. A group whose collision stops
early can immediately take another launch. The two phases reuse the same
thermalization, launch, propagation, classification and checkpoint routines as
static execution. Sampling finishes before collision propagation starts; this
is a finite two-phase queue, not streaming thermalization and collision overlap.
Sampling memory does not grow with the number of nuclear frames: launches stay
on shared storage, and tasks carry their IDs and configuration.

Global trajectory IDs, parent IDs and their RNG seeds do not depend on which
group receives a collision. Results return in input order. With an explicit
`sampling.parents`, static and dynamic jobs can resume each other's checkpoints,
including a job stopped after sampling but before all collisions finish.
Completed trajectory files are replayed once; unsampled parent clocks are
advanced only as needed. Model state that is not a function of the saved
coordinates/configuration still requires its own reproducible checkpoint
contract; scheduling cannot make arbitrary stateful external engines portable.

A Python task/PES exception stops new dispatch. The coordinator drains tasks
already running, all groups close their live MPI PES contexts, and every rank
receives the collective error. A result that cannot be serialized is also
reported collectively. This does not recover from a process/node crash, a hung
external engine, or an MPI transport failure. Long-running tasks already in
flight must return before a controlled failure can finish.

MPI GPU placement counts only worker ranks in each node-local communicator;
the coordinator does not consume a GPU slot. A coordinator-only node is allowed.
Worker nodes still require `len(devices) × trajectories_per_gpu` worker ranks,
with `B=1`. Keep the coordinator CPU and the worker CPUs/GPUs in the allocated
resource budget.

The dispatcher uses mpi4py's documented
[`Comm.send`, `Comm.recv`, `Comm.Dup` and `Comm.Split`](https://mpi4py.readthedocs.io/en/stable/reference/mpi4py.MPI.Comm.html)
interfaces. Control messages use a private communicator; bead force traffic
stays within each group.

An i-PI server's several client connections belong to that single potential
instance. Distinct trajectory groups or bead-process providers still need
distinct TCP ports/Unix paths; the executor does not silently offset endpoints.
For a single shared i-PI server, use one trajectory slot with provider-side
client parallelism. A custom group-aware factory can configure separate servers
when several trajectory groups are required.

## GPU placement and multiple trajectories per card

```toml
[parallel]
backend = "process"
mode = "gpu"
devices = [0, 1]
trajectories_per_gpu = 2
bead_workers = 1
```

This starts four trajectory processes: two on GPU 0 and two on GPU 1.
`trajectories_per_gpu=1` assigns one process per card. Each process has its own model,
cache and trajectory state; GPU memory requirements scale with the number of
processes per card. Trajectories do not share a single model instance. In GPU mode,
the provider batches all beads; `bead_workers > 1` cannot be enabled simultaneously.
A GPU provider must declare and use the requested CUDA device. Missing devices do
not silently fall back to CPU.

For an underfilled local process queue, workers are spread across configured
devices before assigning a second process to a device. With `devices=[0,1]`,
`trajectories_per_gpu=2` and only two tasks (or two task batches), each card gets
one process. A full four-process queue retains two processes per card. This
does not alter MPI's required node-local worker layout.

`devices` contains local device indices after `CUDA_VISIBLE_DEVICES` remapping.
MPI GPU placement obtains node-local ranks through `MPI.COMM_TYPE_SHARED` and chooses
devices from those local ranks, not directly from global ranks. Each node must run
`len(devices) × trajectories_per_gpu` worker ranks, all of which must see the node's
configured device list. For two nodes with two GPUs each and two trajectory processes
per GPU, world size is 8, G=8 and B=1 under static scheduling. `[0,1]` names each
node's own two devices.

## Persistent parent chains and restart

dRPMD uses n independent thermal parents. Each parent equilibrates once, then draws
full q/p every `sampling.interval_steps` from that same persistent chain to assemble
a collision trajectory. Requested N is the total for the whole job: the first
`N % n` parents receive `N // n + 1` trajectories each, and the others receive
`N // n`. Production trajectory state is separate from parent state and the
production thermostat is disabled.

Parent RNG streams derive from the global seed and parent ID; collision sampling
streams derive from the global trajectory ID. Checkpoints atomically save parent
q/p, RNG, thermalization progress, pending child, child q/p, classification persistence
and completion records. Completed trajectories are counted once after interruption;
resuming a terminal checkpoint does not advance an extra step.

With explicit `sampling.parents`, CPU workers or MPI G/B layouts can change while
preserving parent ownership and physical settings. If omitted, `parents` defaults to
the number of trajectory slots. Changing slots then changes the parent count and
checkpoints reject the incompatible restart. Model files, PES libraries, manifests
and declared Python factory source files enter SHA256 validation. Arbitrary transitive
Python dependencies, system shared libraries and remote i-PI engine state are not
fully fingerprinted; preserve them by pinning the runtime environment. Different
devices or parallel reduction orders may change floating-point rounding. Identical
random streams do not imply bitwise identity on different hardware.

A minimal dRPMD pilot may omit the classifier. The available covalent-radius table
then determines centroid bond thresholds, and new compositions are recorded as
candidate channels. Inferred values and assumptions enter resolved input, summaries
and physical checkpoint identity. With `classification_unvalidated=true`, only
geometric event counts are reported; formal chemical rates, cross sections and
chemical reaction bounds are withheld. Same-composition exchange or changes in the
internal bond graph remain unresolved. Unknown elements or initial fragments that
disagree with the inferred graph require explicit classification settings. More
processes or MPI ranks do not validate these pilot rules.

## Validated scope

Single-host tests cover real `spawn` trajectory workers, persistent bead processes,
their combination, five trajectories assigned to two parents, bead batches that do
not divide evenly, model-file changes rejected on restart, and running/terminal
checkpoint recovery. A PES-domain error in the initial collision geometry counts as
a failed trajectory in the unknown-outcome bounds. Missing energy is recorded as
JSON null and an empty energy trajectory, not a numerical result.

`tests/mpi_integration.py` uses a real four-rank MPI run on one host to check G=2,
B=2 potential communication, ordered results, initialization error propagation,
persistent parents and restart. The CLI MPI example has also been run and resumed.
`tests/mpi_workflow_smoke.py` uses real G=2, B=2 execution for FES and the rate
workflow's PMF, parent and child stage broadcasts and restarts.

GPU placement tests use a mock provider to check one or several processes per GPU
and MPI node-local routing. They do not execute real CUDA calculations or measure
performance/scaling on multiple physical nodes. Slurm files are inspectable submission
templates, not evidence of jobs submitted to a particular cluster.

`tests/mpi_dynamic_integration.py` runs with five ranks (`C=1, G=2, B=2`). A
rendezvous verifies that a free group starts the third task before a slow first
task finishes; it is an ordering check rather than a wall-clock speed claim.
The script checks ordered bead forces, empty/underfilled queues, task and
initialization failure, unpicklable results, resource cleanup, mock GPU routing,
and exact free-PES dRPMD agreement with static scheduling and resumed jobs.
`tests/test_dynamic_scheduling.py` also tests interruption between the sampling
and collision phases and changing scheduling mode on resume. These checks do
not establish multi-node scaling or a speedup for a particular physical PES.

The [Ni–H materials acceptance](../benchmarks/materials/batch_acceptance.md)
uses a real ASE EAM calculator for fixed-cell PIMD and diffusion. It compares
serial/process execution, two-rank static MPI and three-rank dynamic MPI,
with and without cross-trajectory batching. Independent call-shape records
verify that batches reach the actual provider, while complete stream/state
comparisons and interruption/restart checks verify replica ownership.
