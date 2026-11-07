# Fused C++17 production propagation

PyRPMD can advance a block of the selected deterministic integrator inside the
C++17 extension, directly invoking a stateless native PES and calculating energy,
centroid and supported collective-variable observations. This is production
code used by the production workflows. Two public entrypoints share the same
selected integration map and explicit native PES bindings, with different recording and
failure contracts:

| API | Observations | Numerical failure | Production use |
| --- | --- | --- | --- |
| `Dynamics.propagate()` | Centroid CVs, sample interval, energy extrema | Entire block leaves the caller's state unchanged | Recrossing children and ordinary Andersen thermalization |
| `Dynamics.propagate_block()` | CVs, boolean mask, optional full q/p, centroid RATTLE | Last accepted step and valid result prefix are retained | dRPMD collision blocks, constrained recrossing parents, native-supported FES/PMF sampling and unthermostatted `diffusion` production |

The second entrypoint also supports bead-average geometric CVs. Its full
contract is documented in [block-propagation.md](block-propagation.md).

Native descriptors encode built-in mathematics without calling the corresponding
Python methods. Both APIs therefore check instance overrides and class-level
method replacements against identities saved by the defining modules. This
also applies to nested CVs/biases and the built-in native PES adapters, including
changes made before the propagation dispatcher is first imported. `auto` keeps
such objects on the compatible Python route; strict `native` rejects them before
advancing the state. Unmodified built-in objects retain native execution.

## Input and API

```toml
[dynamics]
timestep = "0.05 fs"
native = "required"         # require the compiled Cayley extension
loop_backend = "auto"       # auto | native | python
block_steps = 256            # maximum workflow block, 1..65536
```

`auto` selects the fused path only when the provider, bias and observations have
an explicit supported native representation. Otherwise it retains the existing
Python workflow. `python` is the numerical/control path; its individual Cayley
steps may still use the C++ extension. `native` requires a native block and
raises when its PES/CV/bias cannot be represented, rather than silently skipping
Python definitions. `native=False` disables both compiled Cayley and fusion.
The block options are execution settings, not a changed Hamiltonian, and are
removed from physical checkpoint identities. Other scientific settings and
existing PES fingerprints remain checked.

The input default remains 256 and its maximum remains 65536. A direct
`propagate_block()` call accepts 0..4096 steps; workflows that use it cap the
configured block length at 4096 and at their next event. This does not change
the user-requested observation cadence.

```python
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.cv import CVStack, Distance
from benchmarks.h3.pes import BKMP2Potential

# q,p have shape [P,N,3]; all low-level quantities use atomic units.
dynamics = Dynamics(BKMP2Potential(), masses, beta, len(q), dt,
                    loop_backend="auto", block_steps=256)
state = State(q, p)
observations = dynamics.propagate(
    state, 256,
    observe=CVStack([Distance([0, 1]), Distance([1, 2])]),
    sample_interval=1,
)
# observations.energies [samples] is H_P, not energy/P.
# observations.values [samples,D], centroids [samples,N,3].
# The final step is always sampled, including an incomplete final interval.
print(observations.backend, observations.fallback_reason)
print(dynamics.propagation_stats)
```

## Supported direct paths for `propagate()`

* The standard PyRPMD C ABI v1 adapter, when metadata explicitly declares
  `capabilities.stateful=false`. C, C++ and Fortran BIND(C) libraries can provide
  this existing ABI. The exact initialized context is reused.
* The pinned real BKMP2 Fortran batch wrapper. It uses the same canonical-library
  path lock and physical function as ordinary Python/ctypes evaluation, including
  when Python imports the provider under different module names.
* Centroid Distance, BondDifference, Cartesian, COMDistance, RPMDrateCoordinate
  and RateSurface, including the existing equivalent-path min/max rule.
* Full-matrix centroid umbrella bias (including two-dimensional off-diagonal
  stiffness), the RPMDrate flux-metric bias and their composition.
* Fixed-fragment centroid-COM projections in Andersen parent thermalization.

The native binding retains its library/context owner and acquires the provider's
ordinary lock for the whole block. No Fortran COMMON state is made reentrant by
this change. Providers are not called concurrently inside a process. The GIL is
released during propagation; no Python callback is invoked from that section.

Both entrypoints reuse each endpoint PES/bias force within a block of L
steps, reducing PES batches from 2L to L+1 with bit-identical results for a
deterministic provider. `propagate()` does so whenever the provider metadata
does not declare it stateful, history- or time-dependent and no fragment
projection is active. The segmented `propagate_block()` API exposes the choice
as `reuse_forces=None` (the same default); `reuse_forces=False` restores the
two-call reference schedule. The native loop reuses forces with umbrella/flux
biases, centroid RATTLE and fixed atoms; fragment projection moves q after the
force call and is re-evaluated. No force cache survives a block or restart,
so a Python classifier can still change the full state between blocks.
Energy observations reuse the last returned physical energy and add the same
kinetic, ring-spring and bias terms. Work arrays are allocated once per block,
and the existing cyclic Cayley factorization is preserved.

### Per-step data flow

The `propagate_block()` and PILE-L/TRPMD sampling loops keep their
valid-prefix contract without copying the ring every step:

* **Double-buffered state.** The caller's `q`/`p` arrays are one half of a step
  buffer and a private array the other. A step reads the accepted half and
  writes the other; acceptance swaps two pointers. A failed step therefore
  leaves the accepted state untouched. The block start no longer copies the
  input state, and the end copies once, only when the accepted state finished in
  the private half. The free-ring step and first half kick read the accepted
  half and write the trial half directly, so no separate `q` copy remains
  (fragment projection keeps the historical copy/project/propagate order).
* **One reaction-coordinate evaluation.** Centroid umbrella, flux-metric and
  observed CVs lowered from the same descriptor share one evaluation per force
  call: value, gradient and the metric Hessian-vector product, as RPMDrate
  reuses ξ, ∇ξ and ∇²ξ. The observation at a saved step reuses the endpoint
  value instead of evaluating again and materializing a `[P,N,3]` Jacobian. The
  centroid bias forces are added in one pass over the force rows, in bias
  order; other bias kinds keep their own evaluation.
* **Cheaper checks.** Finite checks are branch-free vectorized scans, and
  checks already implied by an earlier one (PES-validated endpoint
  coordinates, unbiased forces) are not repeated. The cyclic Cayley solve
  sweeps whole bead rows, so its Thomas recurrences vectorize across
  Cartesian coordinates and contain no integer modulo.

Every coordinate and force element performs the same floating-point
operations in the same order as before, so trajectories, energies, CV
observations and failure prefixes are bit-identical to the previous loop.

## Workflow/event semantics

**dRPMD:** collision blocks use `propagate_block()`. The exact, unmodified
`ConnectivityClassifier` with explicit channels and the benchmark's identity-aware
`H3ExchangeClassifier` can export `native_outcome_descriptor()`. With a compatible
native PES, C++ evaluates these classifiers at the original `check_interval`,
maintains the original candidate/persistence state and returns immediately when
an outcome resolves. A block can therefore span several classification checks,
while still ending at a checkpoint, the configured block length or 4096 steps.
Near a bond, exit-distance or outward-motion threshold, the native observer yields
the accepted state to the original Python predicate. This conservative roundoff
band changes only where a check executes, never its strict threshold or label.

Pilot candidate discovery, arbitrary Python classifiers, subclasses and overridden
classification hooks retain the Python callback path and its check boundaries.
They still receive the live full q/p. A version marker prevents an older compiled
extension from silently ignoring native event descriptors; the workflow then uses
the original callback path. Every accepted step contributes to the energy-deviation
diagnostic. Per-step energy/centroid records preserve the exact output stride;
only one pre-final q/p snapshot is retained for classifier-error rollback, rather
than a full bead-state history. PES or observation failures retain the valid prefix
without retrying through Python. Classification failures restore the pre-final
state, candidate state and accepted-step counters together.

The input resolver defaults to `check_interval=1` and `persistence="1 fs"`;
the low-level workflow defaults used when bypassing the resolver are 10 steps
and `3*check_interval`. These defaults are unchanged. Native event checks can
keep an every-step check inside a longer block; raising `block_steps` alone
cannot enlarge blocks on the Python-classifier path beyond its check interval.

Trajectory checkpoints retain cumulative accepted `native_steps` and
`python_steps` across restart. Rolled-back steps are excluded. Older checkpoints
without backend counters retain their completed steps as `untracked_steps`,
because their original backend cannot be recovered from positions and momenta.

**Umbrella/PMF/FES:** native-supported sampling uses `propagate_block()` with
multiple observations per block. Blocks stop at Andersen randomization,
checkpoint events, the remaining trajectory length, or the smaller of the
configured block size and 4096 steps. An observation mask retains the original
saved-sample times without returning to Python at every sample. Energy is
checked at the initial state and every accepted candidate step, including
unobserved steps. The NumPy RNG, draw order and continuous-chain state are
unchanged. The flux-metric correction is evaluated inside native forces. The
saved stream/checkpoint includes propagation counters; interruption and a
changed block size can be tested against an uninterrupted run. The compatible
Python sampling path retains its original successful PES-call schedule.

**Materials diffusion production:** the unthermostatted NVE production segment
of the legacy `diffusion` engine uses `propagate_block()` when its PES has a
supported native block and no barostat, electronic friction or active-learning
monitor is configured. A block stops at the next saved sample, checkpoint or the
end of the run, and never exceeds `block_steps` or 4096 steps; `block_steps` is
therefore an upper bound, and with `sample_interval=16` blocks are 16 steps long.
Energy is checked at the block's initial state and after every step, and forces
are reused within a block (L+1 PES batches per L steps instead of 2L). Results
are bit-identical to the former one-step schedule for a deterministic provider.
A failed step commits the valid prefix, and a failed observation discards only
the observed step, matching that schedule's checkpoint. PILE-L thermalization
keeps its per-step schedule, and `loop_backend="native"` raises instead of
falling back when the block path is unavailable.

**Materials Andersen sampling and recrossing children:** the legacy materials
engine also uses deterministic `propagate_block()` segments while an Andersen
sampling phase is active when the PES has a stateless native binding, there is
no barostat, active-learning monitor, electronic-friction provider, bond
constraint or periodic cell, and the configured dynamics object can be fused.
The NumPy momentum refresh remains at its original step boundary and remains
the only random-number draw. Blocks stop at the next Andersen refresh, saved
sample, checkpoint, equilibration/production handoff or the smaller of
`dynamics.block_steps` and 4096. Unsupported providers and those configurations
retain the old one-step path. Sampling, checkpoint RNG state, observations and
restart results are bit-identical to that path; failed blocks retain only their
valid prefix and failed final observations roll back that final step.

RPMD recrossing child trajectories use the same segmented entrypoint with the
dividing-surface CV observed at every step. Each block uses
`min(dynamics.block_steps, 4096, remaining_steps)` and reuses endpoint forces,
so a deterministic block of L steps needs at most L+1 PES batches. The returned
CV indicator is accumulated into the existing numerator without changing the
paired-child result. `dynamics.block_steps = 4096` is a useful execution-only
setting for dRPMD and state-selected runs when fewer Python returns are desired;
it is not part of the scientific fingerprint.

**Recrossing:** every child timepoint evaluates the actual dividing surface in
C++, including equivalent arrangements. Python accumulates the returned indicator
array into the existing paired-child numerator. Constrained parents use
`propagate_block()` with centroid Cayley-RATTLE when its PES and surface are
supported, and the original constrained Python algorithm otherwise. Their
Andersen and checkpoint boundaries are preserved; an observation mask retains
the exact parent-snapshot times within each block. The native
constraint implements the same map and momentum projection, rather than
unconstrained NVE.

**PILE-L/TRPMD sampling blocks:** `pyrpmd.sampling_block.propagate_sampling_block`
uses the same BCOCB/BAOAB coefficient cache and per-step NumPy noise order as
`Dynamics.sample_step`. For an eligible built-in `Dynamics`/`CayleyIntegrator`
with a native PES, PILE-L or TRPMD noise is drawn in Python and passed to
`advance_sampling`; the C++ loop executes `kick -> sample_free -> kick` without
Python callbacks. `P<=8` and power-of-two rings use the native real-mode
kernel; unsupported rings, custom providers, fragment or bond constraints, and
composed operators use the original one-step Python path. `observe` is a
binary per-step mask and `cv` uses the existing native centroid descriptors.

The pre-drawn noise buffer is bounded by `MAX_PREDRAWN_BYTES = 64 MiB`; a
larger request is split into consecutive native chunks, preserving the same
Generator stream and state clock. `record_step_energies` records every
accepted step independently of the observation mask. A native failure returns
the completed prefix and the Python wrapper restores the Generator to the
chunk start before replaying only accepted noise, so a failed suffix consumes
no random numbers. With no bias this is E0; umbrella and flux-metric bias
descriptors use the existing native force implementation and are E1.
When energy output is requested, the native loop retains accepted q/p snapshots
and the wrapper applies the same `Dynamics.energy` NumPy reduction order before
returning energies, preserving the E0 bitwise contract without putting Python
callbacks inside the native step loop.

## Python and distributed compatibility

Arbitrary Python PESs, custom/subclassed CVs or biases, Angle,
BCOCB/PILE, ASE, Torch/GPU, legacy f2py without an explicit native binding,
and process/MPI bead-service providers retain their existing paths. A Python PES
wrapper forwards a binding only for identity unit/sign conversion and explicit
native-provider opt-in. Overridden built-in types do not bypass their Python
mathematics. Independent trajectory processes/MPI groups with one rank can still
use the fused engine. This does not implement native distributed PES messaging.
Bead-average geometric CVs use Python in `propagate()` and have native support
in `propagate_block()`; the latter's rate-coordinate surfaces remain centroid
based. Periodic CVs use Python in `propagate()`; the segmented API applies
periodic umbrella displacement for otherwise supported built-in CVs.
`Dynamics.loop_stats` is a compatibility alias for `propagation_stats`.

A successful `propagate()` commits the entire block. A failure, invalid
observation, interrupt or nonfinite result leaves its caller's q/p/clock unchanged.
No statement is made about undoing arbitrary Python callback side effects; only
stateless native providers are eligible. Long user-specified blocks may delay
signal handling until the next block boundary. Per-step time accumulation is
retained, so changing block sizes does not alter the state clock by rounding.

## Native history-bias descriptors

`Dynamics.propagate_block()` accepts frozen native descriptors for `WallBias`,
`MetadynamicsBias` and `GridMetadynamicsBias`, including their flattened
`CompositeBias` components. A metadynamics descriptor copies the hill centers,
heights, widths and periodicity together with its `hill_count`; a grid
descriptor additionally copies the `tensor_hermite_v1` coefficient array and
history-bound grid metadata. The C++ `Bias<false>` evaluator uses the complete
ring Jacobian and applies the same physical-energy convention as the Python
implementation: the bias potential is multiplied by the bead count in the
Hamiltonian. Periodic hills use the same wrapped image sum and broad-Gaussian
Fourier branch as `MetadynamicsBias._kernel`; grid values outside a
nonperiodic range are rejected.

Descriptors are rebuilt for every block and are never cached across a hill
deposition. The Python endpoint checks the hill count (and the grid metadata)
after `advance` returns, before committing the state clock. Subclasses or
overridden `energy_force` methods retain the Python path. The extension exports
`NATIVE_HISTORY_BIAS_API = 1`; an older extension causes `auto` to fall back
and a strict native request to fail before mutating the state.

## Validation

```bash
python -m pytest -q tests/test_fused_propagation.py
python -m pytest -q tests/test_native_propagation.py
python benchmarks/h3/build_pes.py
python benchmarks/h3/fused_propagation.py --output runs/fused-native.json
python benchmarks/h3/native_propagation.py --output runs/fused-segmented/report.json
```

The real-system harness compares 1/2/8/16/32 beads, two-dimensional umbrellas,
rate-flux PMF windows, all recrossing numerator timepoints, early dRPMD termination
and interruption/resume. Timings include actual propagation and the same energy
and CV observations in each backend; they are not physical-rate convergence.
Checkpoint/file I/O is included in the short workflow measurements but not in the
block-throughput rows. Do not generalize those measurements to an entire
production calculation or a different PES/hardware setup.

The companion segmented harness adds native RATTLE, full multi-window FES/PMF,
three parent configurations and six children, eight real H3 collisions with the
identity-aware H3 classifier, and interrupted collision restart. It records
its exact source commit, source hashes, timing repeats and actual execution
counts. Its core microbenchmark uses the original `Dynamics.step()` reference;
its workflow comparisons include the same checkpoint/output operations. Those
two baselines must be distinguished when quoting timings.

The dedicated [dRPMD optimization validation](validation/drpmd-optimization-2026-10-30/README.md)
compares separate baseline/candidate checkouts with the same BKMP2 library,
build commands, classification cadence, checkpoint settings and observations.
`benchmarks/h3/drpmd_optimization.py` records all timing repeats, source/binary
hashes, exact output comparisons and interrupted-restart checks. It measures
both core propagation and complete dRPMD jobs; these are separate workloads.

## Pre-drawn Andersen momentum refreshes

`Dynamics.propagate_block()` accepts a `RefreshPlan` from
`pyrpmd.momentum_refresh.plan_refresh`.  The plan records strictly increasing
relative `offsets` (the step before which a refresh is installed), float64
`momenta` with shape `[R,P,N,3]`, and a `projection` of `none`,
`centroid_constraint`, or `fragments`.  The planner calls the historical
`Dynamics.sample_momenta()` and geometric `AndersenSchedule.refreshed()` one
refresh at a time while restoring the caller's RNG and collision clock.

Native `advance` validates the descriptor before releasing the GIL, installs a
refresh at the beginning of its offset step, and increments
`refreshes_completed` only after that step commits.  A failed step therefore
rolls the momentum and its RNG draw back together.  The Python path applies the
same transaction and projection rules.  On return, the planner replays exactly
the successful refresh prefix to commit the NumPy RNG and schedule state; a
changed block length or a failure after a refresh cannot consume an extra draw.

The default pre-drawn buffer is capped at `MAX_PREDRAWN_BYTES = 64 MiB` and a
plan shortens its block when that budget is reached.  This is an execution
option and does not enter a physical checkpoint fingerprint.  The native
extension exports `MOMENTUM_REFRESH_API = 1`; an older extension falls back in
`auto` and raises before state mutation in strict `native` mode.  Bond-constraint
momentum sampling remains on the existing Python path and is outside this
contract.

## Serial MetaD history blocks

Serial standard and well-tempered MetaD can keep a frozen wall and hill
history in the native loop between Python events. The NP7 scheduler ends a
block at the next pace deposition, equilibrium or freeze boundary,
checkpoint, 4096-step limit, or pre-drawn buffer limit. Its observation mask
contains both ordinary sample steps and the deposition endpoint. Python then
computes sample_bias from the frozen history and performs the single hill
transaction at the block tail; a failed native prefix is retained without a
hill deposit.

Andersen uses the NP1 pre-drawn RefreshPlan, while eligible PILE-L runs use
the NP2 sampling block. NP4 detached wall, direct MetaD and tensor-Hermite grid
descriptors are rebuilt at every block and checked for history expiry. New
serial checkpoints carry native_path_contract=1; checkpoints without that
execution marker continue through the historical per-step scheduler so restart
behavior is unchanged. Shared or custom-walker histories remain on their
existing paths.

## Materials and diffusion integration

The materials workflow uses the same transactional native block contract as the
other production paths.  When a legacy fixed-cell materials run selects an
eligible PILE-L sampler, `propagate_sampling_block()` advances the sampling
block and commits the sampler clock only after the state and RNG prefix have
succeeded.  Andersen sampling uses a pre-drawn `RefreshPlan` from
`plan_refresh()`, with block ends at refresh/observation/checkpoint/ensemble
handoff events.  This preserves the old engine's refresh schedule and makes
chunk and resume boundaries execution details rather than new random draws.

The native `diffusion` opt-in uses `NativeEnsemble` for both the PILE-L
equilibration blocks and the unthermostatted production blocks.  Production
is fixed-cell NVE and records the materials trajectory schema; it does not
invoke Python once per physical step.  The native path has a distinct
thermostat/RNG/checkpoint identity (E2), so it is not expected to reproduce
legacy arrays bit-for-bit.  `python_steps = 0` in the summary is the required
runtime check.  NPT remains the native `task="pimd"` route with a barostat, and
is intentionally outside this diffusion integration.
