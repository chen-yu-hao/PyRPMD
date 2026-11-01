# Composable propagation

`Dynamics` is the common entry point for a ring-polymer propagation plan. The
integrator, thermostat, pressure controller and explicit force splitting are
selected independently. Native execution and trajectory batching are execution
choices around that plan. Existing `NativeEnsemble`, `BatchedDynamics` and
`MTSDynamics` entry points remain available for compatibility with their state,
observation and failure-handling contracts.

## Select the mathematical operators

All quantities in the Python API are atomic units. A deterministic real-time
trajectory still uses the familiar constructor:

```python
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import HarmonicPotential
import numpy as np

potential = HarmonicPotential(k=0.4)
dynamics = Dynamics(potential, [1837.0], beta=1000.0, nbeads=16, dt=2.0,
                    integrator="normal-mode-bab", seed=1729)
state = State(np.zeros((16, 1, 3)), dynamics.sample_momenta())
dynamics.step(state)
```

No thermostat is applied by this constructor. To construct a sampling plan,
add an explicit thermostat:

```python
dynamics = Dynamics(
    potential, [1837.0], beta=1000.0, nbeads=16, dt=2.0,
    integrator="normal-mode-bab", seed=1729,
    thermostat={"mode": "nhc", "tau": 800.0,
                "chain_length": 3, "substeps": 2},
)
state = State(np.zeros((16, 1, 3)), dynamics.sample_momenta())
dynamics.step(state)
```

The shared thermostat specification accepts the native thermostat modes,
including PILE-L, PILE-G, stochastic velocity rescaling (`svr`), GLE,
normal-mode GLE and PIGLET. `nhc` and `nose-hoover-chain` select the same
Nosé–Hoover-chain operator; `tdamp` is an alias for `tau`. A GLE specification
must supply its drift/covariance matrices. PIGLET also requires explicit
temperature, bead-count and parameter-provenance information; no universal
PIGLET matrix set is inferred.

`Dynamics.sample_step(state, thermostat=spec)` selects the same operators for
a sampling call. Existing calls without an explicit composition retain their
PILE-L/NumPy noise order under `auto` and `python` execution. A configured
thermostat makes `step()` a sampling operation; use an unthermostatted plan for
ordinary real-time RPMD.

| Component | Responsibility |
| --- | --- |
| `integrator` | Cayley, exact normal-mode or velocity-Verlet free-ring map |
| `thermostat` | Momentum heat bath and its auxiliary variables/RNG |
| `barostat` | Cell/piston dynamics and pressure or stress coupling |
| `ForceSplitting` | Outer slow-force kicks and inner fast-force/ring substeps |
| Fixed-atom projection | Exact reference coordinates and zero frozen momenta |
| Native executor | Execute a supported plan without Python step callbacks |
| Trajectory task batching | Combine independent PES requests while preserving each task's plan |

The plan preserves the sum-beads Hamiltonian and physical nuclear masses.
Thermostat noise has the target temperature `P/beta`; fixed atoms do not count
as thermal degrees of freedom. An all-frozen system cannot use a thermostat
that needs positive thermal degrees of freedom.

## Combine sampling and force splitting

Declare the fast/slow Hamiltonian with `SplitPotential`, then attach the number
of inner steps to the same `Dynamics` object:

```python
from pyrpmd.engine import ForceSplitting
from pyrpmd.potentials.contraction import DifferencePotential, SplitPotential

reference = HarmonicPotential(k=0.4)
target = HarmonicPotential(k=0.45)
split = SplitPotential(reference, DifferencePotential(target, reference),
                       nbeads=16, contracted_beads=3)
dynamics = Dynamics(
    split, [1837.0], beta=1000.0, nbeads=16, dt=2.0,
    integrator="normal-mode-bab",
    force_splitting=ForceSplitting(inner_steps=4),
    thermostat={"mode": "nhc", "tau": 800.0},
    seed=1729,
)
```

`dt` and the state clock count outer steps. The inner ring/fast-force step is
`dt/inner_steps`. A thermostat is composed symmetrically around the outer
multiple-time-step map; Andersen collisions use the outer absolute clock.
The legacy `MTSDynamics` adapter retains its deterministic default, its
`State` return value and its two transaction boundaries.

RPC remains an explicitly approximate Hamiltonian when fewer than `P` beads
are retained. It does not export a contracted pressure estimator, so the
shared plan rejects RPC pressure control without an appropriate virial. A
force split also remains a complete-ring provider. Composed trajectory batching
keeps its RPC projection/lift local to each trajectory and combines only the
physical component force calls; it never flattens rings through the coupled
wrapper. See
[RPC/MTS](rpc-mts.md) for normalization and convergence checks.

## Native execution and fixed atoms

`native=true` allows compiled substeps. `loop_backend="native"` additionally
requires a supported complete native propagation loop. `auto` may compose
Python PES/bias callbacks with compiled substeps; a compiled thermostat alone
does not make that loop fully native. Unsupported strict-native plans fail
explicitly; a failed PES call is never retried through a different backend.

Fixed atoms no longer require the deterministic Python loop. Supported native
blocks project their positions and momenta at the same splitting boundaries
as the Python map. The initial/restarted state must already have identical
reference positions on every bead and zero fixed momenta. Invalid initial
states are rejected rather than silently repaired.

PILE-L has a native complete-loop route in the common stack. Legacy automatic
sampling continues to use its established NumPy stream. Native thermostat RNG
and auxiliary state belong to the saved operator configuration; restoring a
checkpoint does not replace that stream when the requested execution backend
changes.

Native capability checks still respect overridden Python integrator, force,
projection, bias and CV methods. An implementation cannot bypass a custom
method merely because it has a familiar base class.

After a composed native executor binds a provider, its model and binding must
remain unchanged. Replacing that provider requires constructing a new
`Dynamics`; the engine rejects rebinding instead of continuing with cached
forces from the previous model.

## Checkpoint the complete plan

Coordinates and momenta are only part of a thermostatted trajectory. Save the
state clock and `dynamics.operator_state_dict()` together with `q` and `p`.
Restore the operator record with `load_operator_state_dict()` before continuing.
The operator record is JSON serializable and includes the state needed by
thermostat auxiliary variables and stochastic operators.

For a full native trajectory that will restart in a new `Dynamics` instance,
pass an explicit `potential_identity` that identifies the model and its
immutable parameters. Without it, the native checkpoint identity is local to
the current `Dynamics`, which supports rollback but cannot establish that a
different object represents the same model. Bath-only operator checkpoints
do not bind a physical PES. An attached `BZPBarostat` also records its pressure,
piston parameters and reference cell, so changing those values invalidates a
restart.

The transaction boundary includes this state:

- `step()` restores the pre-step physical and operator states on failure.
- `propagate()` restores the entire requested block on failure.
- The valid-prefix block API retains its documented completed-step contract.
- Independent umbrella windows save one operator record per window/replica.
- Replica exchange saves all window operators in the same ensemble transaction
  as coordinates, samples, exchange RNG and acceptance history.
- Metadynamics and OPES save composed bath state with their hill/kernel history;
  a failed force call or history update restores the same accepted prefix.

An observation failure cannot leave a chain momentum or GLE auxiliary variable
advanced beyond the last saved physical step. Missing operator state in a new
thermostatted checkpoint is an error. Legacy Andersen/PILE-L checkpoints retain
their established payload when they do not use an explicit operator plan.

## Workflow sampling and batching

Independent umbrella sampling, same-temperature replica exchange, metadynamics
and OPES accept canonical thermostat operators through the existing sampling
table:

```toml
[sampling]
thermostat = "nhc"

[sampling.thermostat_params]
tau = "20 fs"
chain_length = 3
substeps = 2
```

Use `svr` with `tau`, `langevin` or `ffl` with `gamma`, or a GLE mode with its
explicit `gle` table. Quantities in this input table may have units; direct
`Dynamics` specifications use atomic values.

Historical metadynamics Andersen, PILE-L and `nose-hoover-chain` inputs retain
their original maps, random streams and checkpoint fields. The existing
`nose-hoover-chain` option also retains its explicit
`reset_after_equilibration` policy. Use `nhc` to select the common NHC operator;
its chain continues across the equilibration boundary. New modes and explicit
PILE-L operator parameters save the common operator record. OPES also supports
the established NumPy PILE-L map. History updates still run at their existing
step boundaries and require `loop_backend="auto"` or `"python"`.

Canonical umbrella reweighting still needs a canonical sampling measure.
Unthermostatted dynamics, TRPMD with its unthermostatted centroid, and PIGLET
are rejected as ordinary MBAR umbrella samplers. Canonical GLE requires unit
physical variance and zero physical/auxiliary covariance. A shared execution
stack does not establish a new statistical estimator for a different measure.
The same canonical-measure requirement applies to the existing MetaD and OPES
free-energy estimators.

For independent fixed-window umbrellas, dRPMD collisions and `site_rate`,
`parallel.trajectory_batch_size` batches force requests from the existing
task algorithms. Each task keeps its own RNG, thermostat, classifier and
checkpoint. See [Trajectory batching](batched-dynamics.md) for provider
capabilities, process/MPI grouping, and stop/failure handling.

## Physical compatibility remains explicit

Pressure control requires a fully periodic, all-mobile system and a provider
with the required physical virials/cell support. Fixed atoms with cell scaling
and unvalidated PIGLET NPT combinations remain errors. GLE/PIGLET matrices must
match their declared convention and calibration. Bath-free TRPMD limits do
not consume thermostat noise. These are constraints on the requested
mathematics, not reasons to substitute a different thermostat or integrator.

The existing detailed guides remain useful for individual operators:
[native ensembles](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/native-ensembles.md), [RPC/MTS](rpc-mts.md),
[trajectory batching](batched-dynamics.md) and
[TRPMD and spectroscopy](trpmd-spectroscopy.md).
