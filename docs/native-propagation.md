# Fused C++17 production propagation

PyRPMD can now advance a block of BCB steps inside the existing C++17 Cayley
extension, directly invoking a stateless native PES and calculating energy,
centroid and supported collective-variable observations. This is production
code used by the production workflows. Two public entrypoints share the same
Cayley map and explicit native PES bindings, with different recording and
failure contracts:

| API | Observations | Numerical failure | Production use |
| --- | --- | --- | --- |
| `Dynamics.propagate()` | Centroid CVs, sample interval, energy extrema | Entire block leaves the caller's state unchanged | FES/PMF, recrossing children, Andersen thermalization |
| `Dynamics.propagate_block()` | CVs, boolean mask, optional full q/p, centroid RATTLE | Last accepted step and valid result prefix are retained | dRPMD collision blocks and constrained recrossing parents |

The second entrypoint also supports bead-average geometric CVs. Its full
contract is documented in [block-propagation.md](block-propagation.md).

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

The first version deliberately retains **two force evaluations per BCB step**.
It does not reuse forces across steps or assume that a new PES supports caching.
Energy observations reuse the last returned physical energy and add the same
kinetic, ring-spring and bias terms. It allocates work arrays once per block,
not per time step, and preserves the existing cyclic Cayley factorization.

## Workflow/event semantics

**dRPMD:** collision blocks use `propagate_block()` and stop at the next
classifier or checkpoint boundary, or the 4096-step limit. Per-step records
preserve the exact trajectory-output times even when an output stride falls
inside a block. The actual built-in or
custom Python classifier sees the accepted full q/p at its original check
interval; it is not replaced by a native classifier. Every accepted step
contributes to the energy-deviation diagnostic. Stop/persistence counts are
unchanged. PES or observation failure retains the valid prefix without retrying
PES calls through Python. Classifier exceptions use the workflow's established
rollback rule; physical state and accepted-step counters are restored together.

Trajectory checkpoints retain cumulative accepted `native_steps` and
`python_steps` across restart. Rolled-back steps are excluded. Older checkpoints
without backend counters retain their completed steps as `untracked_steps`,
because their original backend cannot be recovered from positions and momenta.

**Umbrella/PMF/FES:** blocks stop at Andersen randomization, saved-sample and
checkpoint events. The NumPy RNG, draw order and continuous-chain state are
unchanged. The flux-metric correction is evaluated inside native forces, not
omitted to obtain a speedup. The saved stream/checkpoint includes propagation
counters; interruption and a changed block size can be tested against an
uninterrupted run.

**Recrossing:** every child timepoint evaluates the actual dividing surface in
C++, including equivalent arrangements. Python accumulates the returned indicator
array into the existing paired-child numerator. Constrained parents use
`propagate_block()` with centroid Cayley-RATTLE when its PES and surface are
supported, and the original constrained Python algorithm otherwise. Their
Andersen and checkpoint boundaries are preserved; an observation mask retains
the exact parent-snapshot times within each block. The native
constraint implements the same map and momentum projection, rather than
unconstrained NVE.

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
identity-aware Python classifier, and interrupted collision restart. It records
its exact source commit, source hashes, timing repeats and actual execution
counts. Its core microbenchmark uses the original `Dynamics.step()` reference;
its workflow comparisons include the same checkpoint/output operations. Those
two baselines must be distinguished when quoting timings.
