---
orphan: true
---

# Dynamic CPU trajectory scheduling

For single-host CPU ensembles with uneven trajectory lifetimes, select
`parallel.scheduling="dynamic"`. The default remains `"static"` for backward
compatibility. The dynamic CPU backend uses **spawn processes**, not a Python
thread pool, and returns results in the original task order.

## 20,000 trajectories, one CPU core per active trajectory

Add these fields to the corresponding tables of an otherwise validated input:

```toml
[collision]
trajectories = 20000

[parallel]
backend = "process"
mode = "serial"
scheduling = "dynamic"
trajectory_workers = 64
bead_workers = 1
threads_per_rank = 1
trajectory_batch_size = 1
coordinator_ranks = 0

[sampling]
parents = 64
```

Replace 64 with the number of allocated CPU cores, accounting for memory per
worker and the light coordinating process. The launch does not reserve CPUs or
set process affinity. Use the cluster scheduler/OS to restrict the allocation;
do not run 64 workers inside a job allocated fewer CPUs. For thermal dRPMD,
choose and explicitly preserve the parent-chain count when changing resources
on resume: `parents` is a scientific sampling setting, not a scheduling knob.
For QCT/NE-RPMD, `parents` only labels work groups; there is no thermal burn-in.

Also limit external libraries before starting Python:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
python -m pyrpmd run input.inp
```

Each worker initializes the existing BLAS limiter and execution context once.
The coordinator keeps at most one outstanding future per worker and refills
free capacity when any task finishes; it does not wait for an earlier slow
result to return. Thus 20,000 jobs do not allocate 20,000 outstanding futures.
Task descriptors and returned summaries are still retained in memory. The
number of simultaneous collisions is bounded by the workers, not the requested
trajectory count. `mode="serial"` describes the PES within one trajectory,
not the parallelism across trajectories.

## Workflow granularity and reproducibility

Thermal dRPMD first runs each persistent parent chain in its original sampling
order and saves the launch states. Only after sampling completes are individual
collisions dynamically dispatched. Parent RNG streams, sampling clocks, global
trajectory IDs and saved-launch replay are unchanged. There is no concurrent
mutation of a parent chain by multiple workers, and no sampling/production
overlap in this two-phase implementation.

QCT and NE-RPMD sample and propagate one independent trajectory per queued task,
even when `sampling.parents=1`. This per-trajectory split also applies to their
MPI dynamic path. Initialization seeds remain derived from the global trajectory
ID, not the process, task arrival order or worker count. Static and dynamic
execution can reuse compatible launch files/checkpoints. Preserve explicit
thermal parent counts and all scientific inputs when changing execution layout.
Custom providers/classifiers must not depend on hidden cross-trajectory history.

With `trajectory_batch_size=1`, native propagation remains available; changing
the task scheduler does not replace C++ propagation with a Python step loop.
Larger cooperative PES batches are a separate feature: they may use threads and
introduce their own within-batch waiting. Keep their size at one for the
one-process/one-active-trajectory model described here.

## Boundaries and trade-offs

The local dynamic mode currently supports `mode="serial"`, `bead_workers=1`
and `coordinator_ranks=0`. Local GPU placement and nested bead-worker pools retain
the static path; incompatible dynamic requests fail explicitly. MPI dynamic
scheduling still requires one coordinator and `world_size=1+G*B`, as described in
[parallel execution](parallel.md).

A task, serialization or worker initialization error stops new dispatch once
observed. Not-yet-started futures are cancelled; already running tasks finish
before the error is returned. Existing per-trajectory checkpoint/error handling
remains responsible for preserving scientific results. This is not recovery
from a hung PES, process/node crash or broken filesystem.

Dynamic scheduling reduces static assignment imbalance but cannot split the
last single long trajectory. Sampling can still have a slow parent chain.
Per-collision task setup, PES construction and checkpoint I/O may dominate a
very cheap trajectory; this patch does not promise a speedup for every PES and
does not add provider reuse to collision workers that did not already lease
providers. Benchmark representative collision durations on the target machine.

## Tests

`tests/test_process_queue.py` exercises real spawn workers with a rendezvous:
task 0 cannot finish until task 2 starts while task 1 has freed the other worker.
It also checks ordered results, bounded futures for 20,000 synthetic tasks,
empty/underfilled queues, initialization/serialization errors and worker cleanup.
These are correctness checks, not a physical 20,000-trajectory benchmark.

`tests/test_process_scheduling.py` covers configuration normalization, CPU/BLAS
context, QCT/NE-RPMD task granularity, actual process runs and static/dynamic
restart equivalence. Existing MPI queue tests remain in
`tests/test_dynamic_scheduling.py`.
