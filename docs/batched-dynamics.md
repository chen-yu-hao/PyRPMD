# Independent trajectory batching

`pyrpmd.batched` advances independent real-mass ring polymers through one
physical potential call per step. The public state shape is `[K,P,N,3]` for
`K` trajectories, `P` beads and `N` atoms. Only the potential input becomes
`[K*P,N,3]`. The existing free-ring integrator sees `[P,K*N,3]` with repeated
atom masses, preserving `P`-bead springs within every trajectory. The selected
Cayley, exact normal-mode, or velocity-Verlet map is unchanged.

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
Ring-polymer contraction wrappers need a separate per-ring contraction path
and cannot flatten multiple rings through this API. Process/MPI bead proxies
without an independent-frame declaration are also rejected.

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

This core can propagate preinitialized collision ensembles. A collision stop
callback can close over one classifier/persistence state per trajectory; it
must not use a shared persistence counter across trajectories. The existing
`drpmd` runner's checkpoint/thermalization/task scheduling has not been replaced
by a batched runner.

Static umbrella forces may be supplied as `bias=[window_bias_0, ...]`, one per
trajectory. Fixed atoms use the existing reference-position and zero-momentum
convention. This API is NVE propagation: a full canonical umbrella sampler
still needs its established thermostat, per-window RNG streams and statistical
bookkeeping. It does not batch adaptive umbrella scheduling or metadynamics
depositions. Time-dependent/depositing bias objects are rejected.

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
