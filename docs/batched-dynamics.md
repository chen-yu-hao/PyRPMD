# Independent trajectory batching

`pyrpmd.batched` advances independent real-mass ring polymers through one
physical potential call per step. The public state shape is `[K,P,N,3]` for
`K` trajectories, `P` beads and `N` atoms. Only the potential input becomes
`[K*P,N,3]`. The existing free-ring integrator sees `[P,K*N,3]` with repeated
atom masses, preserving `P`-bead springs within every trajectory. The selected
Cayley, exact normal-mode, or velocity-Verlet map is unchanged.

## Batch existing workflow tasks

Fixed-cell `pimd` / `diffusion` replicas, independent umbrella windows, dRPMD
collisions and `site_rate` tasks can share PES calls without replacing their
sampling or trajectory algorithms:

```toml
[parallel]
trajectory_batch_size = 8
bead_workers = 1

[dynamics]
loop_backend = "auto"
```

`trajectory_batch_size` is a maximum number of independent workflow tasks in
one group, from 1 to 256. The upper bound limits the number of simultaneous
cooperative worker threads. The default, 1, retains the scalar task schedule. Each task keeps its
own state, clock, random generator, thermostat state, observations and
checkpoint files. When the live tasks request forces, their independent
configuration arrays are concatenated into one PES call and its results are
returned to the originating tasks. A task that finishes, stops or fails leaves
the group; completed checkpoints do not have to make a dummy force request.

For dRPMD, parent sampling first writes the existing launch checkpoints and
then collision tasks enter the batch groups. This two-phase schedule is
automatic when batching is requested; no separate launch-file option is
needed. `site_rate` groups both parent and child tasks. Independent umbrella
batching currently applies to fixed windows; adaptive window scheduling and
replica-exchange ladders reject this execution option explicitly.

This execution layer also batches the PES calls of thermostatted sampling. It
does not couple thermostats between windows, synchronize independent collision
clocks, or turn the individual trajectories into a larger ring. Group size and
task completion order do not change the task-specific RNG streams. Floating
point differences are still possible for providers whose arithmetic changes
with batch size.

The force provider must explicitly support independent, history-independent
frames as described below. All tasks in a group use the same potential and
system identity. Mutable-cell/NPT tasks, unknown ring-coupled potentials,
stateful providers and bead-process parallelism are rejected. Exact built-in
`SplitPotential` and `DifferencePotential` wrappers have a separate route:
contraction/lifting stay local to each ring, and only requests to their
independent physical component providers are batched. The scheduler runs in
Python, so `loop_backend="native"` cannot be combined with task batching;
compiled ring substeps remain available through `native=true` or
`native="required"`.

Task batching and `recrossing.child_batch_pairs` describe different groups:
the former groups existing workflow tasks, while the latter vectorizes the
opposite-momentum children within gas-phase/surface recrossing. Choose the
setting for the workflow being run; neither changes the number of requested
trajectories or their statistical weights. For `site_rate`,
`recrossing.child_batch_pairs` is an additional child-group size control; the
larger of it and `parallel.trajectory_batch_size` determines the child group.

### Materials replicas

For the default materials engine (`sampling.engine="legacy"`), set the number
of independent replicas and the execution packing separately:

```toml
[sampling]
replicas = 8

[parallel]
backend = "process"
trajectory_workers = 2
trajectory_batch_size = 4
bead_workers = 1
```

This schedules eight replicas in two groups of at most four, with two worker
processes. Each process combines the live replicas' requests into
`[live_replicas * beads, atoms, 3]` PES inputs (using the contracted bead count
for an RPC component). An incomplete final group is valid. `replicas`
controls the number of independent trajectories; neither worker count
nor batch size creates extra replicas. Saved frames remain correlated and
need a separate effective-sampling analysis. The same grouping works with static
MPI (`world size = G`, `B=1`) and dynamic MPI (`world size = 1+G`, `B=1`),
where the unit dispatched to a trajectory group is a batch of replicas.

The existing scalar step still owns each replica's thermostat, RNG, fixed
atoms, RATTLE constraints, electronic friction and transaction boundaries.
Known split/difference RPC potentials retain their per-ring contraction and
batch only the independent physical components. A finished lane borrows the
shared provider and cannot close it while other lanes are active. Per-frame
PES failures are attributed to the affected replicas; healthy lanes finish
their work before the aggregate failure is raised.

Completed checkpoints are validated and replayed before opening a batch PES.
Mixed complete, partially complete and new replicas can be resumed together.
A fully completed restart constructs no PES and consumes no random numbers.
Changing batch size, trajectory worker count or stream chunk size preserves
the scientific identity; changing the model, cell, temperature or physical
schedule still invalidates the restart. Bitwise equivalence is tested for the
same CPU engine; other hardware or model reduction orders can change rounding.

| Materials configuration | `trajectory_batch_size > 1` |
| --- | --- |
| Fixed-cell legacy `pimd` and `diffusion` | Supported, including process and MPI trajectory execution |
| Fixed atoms, bond constraints, electronic friction, known RPC wrappers | Preserve the established per-replica implementation |
| Mutable-cell/NPT | Rejected before sampling |
| `sampling.engine="native"` or forced `loop_backend="native"` | Rejected; compiled free-ring substeps remain available in the legacy engine |
| `bead_workers > 1` | Rejected; choose either cross-trajectory batching or bead processes |
| Materials active-learning callbacks | Rejected because their uncertainty/candidate lifecycle is not batch-adapted |

The [real Ni–H acceptance example](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/materials/batch_acceptance.md)
executes both materials workflows, counts actual PES input shapes, compares
scalar/process/static-MPI/dynamic-MPI results, and interrupts and resumes a
streamed run. Its separate convergence study checks scientific sampling
choices; passing a short execution test does not certify a diffusion constant.

## Recrossing input

Gas-phase `rate` and conditional `surface_rate` child propagation can opt in:

```python
recrossing = {
    # Existing parent sampling and child_pairs settings remain applicable.
    "child_batch_pairs": 8,  # 8 independent +/- pairs = at most 16 trajectories
}
dynamics = {"loop_backend": "auto", "block_steps": 256}
```

The default `child_batch_pairs=1` retains scalar workflow behavior. A larger
value groups pairs into worker tasks. Parent-chain sampling stays separate;
every pair keeps its original `SeedSequence([seed,307,chain,parent,pair])`,
unconstrained Maxwell momentum draw, signed flux weight, and checkpoint name.
Changing pair order, worker packing, or batch size does not change these RNG
streams. Existing scalar checkpoints can be resumed in a batch and batched
checkpoints can be read by the scalar path. Finite precision differences remain
possible between differently packed native/FFT kernels.

`dynamics.loop_backend="native"` requires a fully native propagation loop and
therefore rejects this path: batch orchestration and custom potential/CV calls
remain Python. `native="required"` can still require the compiled free-ring
kernel. TRPMD children currently require `child_batch_pairs=1`.

## Potential capability

Batching needs **independent, history-independent physical PES frames**. A
custom Python or ML provider must explicitly promise this:

```python
class MyPotential:
    metadata = {"trajectory_batch_safe": True}

    def evaluate(self, positions):
        # positions[B,N,3]; changing B or frame order must not alter a frame's
        # physical energy/force, and earlier calls must not alter the result.
        ...
```

Exact built-in analytic providers and C ABI providers declaring
`capabilities.stateful=false` have known batch semantics. Python unit/gradient
adapters preserve declarations; the Torch adapter now preserves explicit model
or delegate declarations. Merely accepting a batched tensor or declaring
`native_batch=true` is insufficient. An explicitly stateful, ring-coupled,
`batch_independent=false`, or `trajectory_batch_safe=false` provider is rejected.
Ring-polymer contraction wrappers use the separate per-ring path in the
composed API below; flattening complete rings through a coupled evaluator is
never allowed. Process/MPI bead proxies
without an independent-frame declaration are also rejected.

For an audited ASE calculator, `ase`, `mace`, `nequip` and `fairchem` inputs may
declare `potential.trajectory_batch_safe=true`. This is an explicit guarantee
by the caller that frame order, frame count and previous calls cannot affect
the physical result. It is disabled by default and cannot override a
calculator's explicit stateful/coupled/unsafe veto, including one visible
through a transparent Python adapter. See the
[ASE declaration contract](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/potential-adapters.md#independent-trajectory-batching).
ASE still evaluates each frame separately and reports `native_batch=false`;
fewer PyRPMD calls alone do not imply fused model inference or a speedup.

## Python API and stop masks

```python
from pyrpmd.batched import BatchedState, BatchedDynamics
from pyrpmd.cv import Cartesian

# q and p have shape [K,P,N,3], with atomic-unit coordinates and momenta.
state = BatchedState(q, p)
engine = BatchedDynamics(potential, masses, beta, P, dt,
                         integrator="normal-mode-bab")
result = engine.propagate_block(
    state, 500,
    observe=Cartesian(0, 0),
    sample_interval=5,
    active=active_mask,  # optional bool[K]
    stop=lambda index, snapshot: bool(snapshot.q[:, 0, 0].mean() > 20),
    raise_on_error=False,
)
```

Composed array batches with more than 256 active lanes are evaluated in bounded
groups; original lane indices and RNG identities are preserved. The vectorized
NVE path does not create one worker thread per lane and has no such lane limit.

`state.step[K]` and `state.time[K]` advance independently. A stopping step is
always recorded, even between sampling points. Stopped/inactive trajectories
are omitted from subsequent PES calls. The stop callback receives a scalar
`State` snapshot; modifying it cannot modify the live trajectory.

`result.values` has shape `[observations,K,dimension]`. `result.valid` marks
which trajectories completed each recorded step; missing observations are
NaN. `completed`, `stopped`, `failed`, and `errors` retain original trajectory
indices. `initial_energy` and sampled `energies` use the same sum-beads
Hamiltonian as `Dynamics.energy`, without another potential call.

Each full step commits only after its force evaluations, finite-energy check,
CV observation and stop callback succeed. A nonzero per-frame PES status
fails only its trajectory. A whole-batch exception or invalid batch result
fails every trajectory in that call; there is no retry through another
backend. Failed trajectories retain the previously committed q/p/clock. The
default raises `BatchedPropagationError`, whose `result` carries completion
details, after committing healthy trajectories in that iteration. With
`raise_on_error=False`, remaining healthy trajectories continue.

This low-level core can propagate preinitialized collision ensembles. A collision stop
callback can close over one classifier/persistence state per trajectory; it
must not use a shared persistence counter across trajectories. The existing
`drpmd` runner uses task batching above to retain its existing classifier,
thermalization and checkpoint behavior.

Static umbrella forces may be supplied as `bias=[window_bias_0, ...]`, one per
trajectory. Fixed atoms use the existing reference-position and zero-momentum
convention. The default low-level API is NVE propagation. A full canonical
umbrella sampler needs its thermostat, per-window RNG streams and statistical
bookkeeping; the workflow task scheduler supplies batching around that
established sampler. `BatchedDynamics` itself does not schedule adaptive
umbrellas or metadynamics depositions; time-dependent or depositing bias
objects are rejected.

## Compose batched thermostats and force splitting

`BatchedDynamics` also accepts the common `thermostat` and `force_splitting`
operators:

```python
from pyrpmd.engine import ForceSplitting

engine = BatchedDynamics(
    split_potential, masses, beta, P, dt,
    integrator="normal-mode-bab",
    force_splitting=ForceSplitting(inner_steps=4),
    thermostat={"mode": "nhc", "tau": 800.0},
    seed=1729,
)
result = engine.propagate_block(state, 100, active=active_mask)
operators = engine.operator_state_dict()
```

The composed path delegates each lane to the common scalar plan and combines
its physical PES requests. Each original trajectory index has its own random
stream and bath variables; temporarily inactive or completed lanes do not
consume noise. NHC is therefore one chain per independent trajectory, not
one chain across all concatenated atoms. Use `load_operator_state_dict()` with
the saved `BatchedState` to resume these auxiliary states.

Exact built-in split/difference wrappers keep each RPC projection and force
lift within the corresponding ring. Fast, slow and correction-provider calls
are grouped by their physical component, even when tasks are at different
force-splitting stages. Arbitrary ring-coupled providers still need an explicit
adapter. An observer failure restores that lane's physical and operator state
together; healthy lanes retain their accepted steps.

## Cost and validation

For `T` steps and no stopped trajectories, a block uses `T+1` PES calls by
reusing its final force at the next step. Calls process all active trajectory
frames; no cached force survives the block. A larger batch can reduce model
launch/dispatch overhead and improve accelerator utilization, but increases
working memory. The implementation still transfers NumPy coordinates through
the existing potential adapter; persistent GPU integrator state is future work.

Run the reproducible CPU reference benchmark with:

```bash
python benchmarks/batched_dynamics.py --trajectories 16 --beads 32 --steps 128
```

The script reports scalar/batch timings, potential call/frame counts, and final
q/p differences. Optional `--call-latency-us` adds explicitly labelled synthetic
dispatch latency. These results are not a GPU or real ML-potential benchmark.
Tests compare all three integrators against independent scalar trajectories,
exercise native and NumPy kernels, verify reversibility/no cross-trajectory
springs, fixed atoms, distinct umbrella biases, borrowed force buffers,
per-frame failures, stop masks, RNG order, and checkpoint interoperability.

### Real BKMP2 timing check

The [recorded BKMP2 comparison](validation/tier1-2026-10-31/recrossing-performance.json)
uses 8 child pairs, 16 beads, 128 steps at 0.1 fs, one trajectory worker, and
three repetitions. Timings include potential construction, process-pool
startup/shutdown, propagation and checkpoint writing. Every variant reproduces
the scalar paired flux numerator and denominator exactly. These near-TS starting
coordinates are a propagation regression, not an equilibrium rate calculation.

| Execution | Median time / s |
| --- | ---: |
| Fused scalar, reconstruct provider for every pair | 0.131282 |
| Fused scalar, reuse provider across pairs | 0.111223 |
| Two bead processes, rebuild pool for every pair | 10.355196 |
| Two bead processes, reuse pool across pairs | 6.187187 |
| Batch 4 pairs, reuse provider | 0.418252 |
| Batch 8 pairs, reuse provider | 0.360104 |

Provider reuse is 1.18× faster in the serial case; persistent two-process bead
execution is 1.67× faster than rebuilding its pool for each pair. Batching is
slower than the existing fused scalar loop on this inexpensive Fortran PES.
The default scalar path is retained, and batch size is opt-in. A costly batched
ML provider requires its own throughput/memory benchmark before choosing the
batch size; neither this result nor the harmonic timing establishes GPU gains.

Reproduce this comparison after building the checksum-verified BKMP2 library:

```bash
python benchmarks/tier1_recrossing.py \
  --library benchmarks/h3/build/libbkmp2.so \
  --output runs/tier1-recrossing.json
```
