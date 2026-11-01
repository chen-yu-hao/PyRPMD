# Bond constraints and rigid molecules

PyRPMD supports fixed pair distances with per-bead SHAKE/RATTLE in the
`velocity-verlet` integrator. This permits constrained molecules, including a
rigid three-site water geometry defined by two O–H distances and one H–H
distance. A force-field provider supplies physical forces; PyRPMD owns the
constraint impulses, nuclear propagation and heat bath.

## Input and first run

```toml
task = "pimd"
temperature = "300 K"
beads = 8

[system]
geometry = "water.xyz"
bond_constraints = [
  { atoms = [0, 1], length = "0.9572 angstrom" },
  { atoms = [0, 2], length = "0.9572 angstrom" },
  { atoms = [1, 2], length = "1.5139 angstrom" },
]

[dynamics]
integrator = "velocity-verlet"
timestep = "0.05 fs"
loop_backend = "auto"

[sampling]
thermostat = "andersen"
andersen_interval = 10
equilibration_steps = 1000
production_steps = 10000
sample_interval = 10
```

Add a potential appropriate to the model. Each index is zero based. Numeric
lengths are **bohr**, independent of `geometry_unit`; quantity strings avoid
ambiguity. Pair order is canonicalized, repeated pairs and nonpositive lengths
are rejected, and isotope masses determine all projection weights. Constraints
are part of the system/checkpoint identity, so changing a pair or distance
requires a new run.

The self-contained [`rigid_water.inp`](../examples/materials/rigid_water.inp)
uses a zero physical potential to test constrained propagation and restart:

```bash
pyrpmd run examples/materials/rigid_water.inp --output runs/rigid-water
pyrpmd resume runs/rigid-water
pyrpmd export runs/rigid-water runs/rigid-water.extxyz
```

That example checks kinematics. It is not an interacting water model or a
liquid-water validation. The [q-TIP4P/F benchmark](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/water/README.md)
is a **flexible** water model; adding rigid bonds would change that model and
remove its O–H stretching spectrum.

## Algorithm and supported combinations

Each bead follows the full Cartesian ring-polymer Hamiltonian, with physical
masses and spring frequency `P / beta`. A step uses:

1. A half kick from physical and ring-spring forces.
2. A Cartesian drift followed by iterative SHAKE along the **old** bond
   gradients. The coordinate correction also changes momentum by
   `mass * displacement / timestep`.
3. A second physical/spring half kick, followed by RATTLE projection of the
   relative velocities onto every bond's tangent plane.

The second momentum projection and the drift impulse are both necessary.
Simply repairing the coordinates after an unconstrained step is insufficient.
The implementation follows the velocity-Verlet constraint construction in
[Andersen, J. Comput. Phys. 52, 24–34 (1983)](https://doi.org/10.1016/0021-9991(83)90014-1).
The [LAMMPS SHAKE/RATTLE documentation](https://docs.lammps.org/fix_shake.html)
also describes the distinction between position and velocity constraints.

| Choice | Supported behavior |
| --- | --- |
| Integrator | `velocity-verlet`, all beads, physical masses |
| Production | Fixed-cell, unthermostatted `Dynamics.step/propagate` and `task="diffusion"` |
| Sampling | Projected Maxwell momenta and Andersen, including `task="pimd"` |
| Execution | Python constraint loop; `auto` records the fallback reason |
| Boundaries | Nonperiodic, orthorhombic or triclinic fixed cells; partial PBC retained |
| Initial conditions | Explicit position projection followed by tangent momentum sampling |
| Restart | Constraint lengths and velocity tangency are checked; invalid states are rejected |

Cayley/normal-mode splitting, PILE/TRPMD, Nosé–Hoover, pressure control,
electronic friction, specialized dynamics subclasses, fragment-COM projections
and constrained rate/FES workflows are not implemented for these bond
constraints and are rejected. `loop_backend="native"` and `native="required"`
are rejected for constrained propagation. Fixed atoms may coexist only when
they are not endpoints of a constrained bond.

The Python API defaults to a relative length tolerance of `1e-12`, a fractional
extension-rate tolerance of `1e-12` per atomic time, and at most 1000 iterations.
A nonconvergent solve or physical-force failure does not commit the trial state.
Large rotations through 90 degrees in a drift are rejected with a request to
reduce the timestep. The ordinary velocity-Verlet ring stability restriction
still applies; constraints do not remove fast ring-spring modes.

Periodic bonds select a nearest image at the start of the drift and hold its
lattice shift fixed during SHAKE. Coordinates remain unwrapped. A constrained
distance must be less than half the smallest periodic cell height; this
conservative limit prevents ambiguous bond images in skew cells.

## Python API

```python
import numpy as np
from pyrpmd import Dynamics, State
from pyrpmd.potentials import FreePotential

dyn = Dynamics(
    FreePotential(), masses=[1837.0, 3671.0], beta=1000.0,
    nbeads=4, dt=1.0, integrator="velocity-verlet",
    bond_constraints=[(0, 1, 1.4)],  # bohr
    native=False,
)
q = np.repeat(np.array([[[0., 0., 0.], [1.4, 0., 0.]]]), 4, axis=0)
state = State(q, np.zeros_like(q))
dyn.initialize_constraints(state)
state.p[:] = dyn.sample_momenta(state.q)
dyn.propagate(state, 20)
position_error, tangent_error = dyn.constraints.residuals(state.q, state.p)
```

`initialize_constraints` is an explicit initialization operation. Do not use it
to conceal drift or repair a restart. Ordinary steps reject inconsistent input
states. `sample_momenta(q)` conditions the bead Maxwell distribution on the
mass-metric tangent space; constrained dynamics requires the current `q`.

## What a constrained run establishes

The supported P>1 model is a **holonomically constrained Cartesian ring
Hamiltonian**. This implementation does not provide the specialized rigid-rotor
imaginary-time propagator or a Fixman potential needed by other constrained
ensemble definitions. Applying a constraint on every bead also does not imply
that the centroid has the same bond length. These distinctions matter when
comparing with a rigid-water quantum reference.

The existing unconstrained primitive and centroid-virial kinetic-energy
formulas omit constraint/metric terms. Constrained materials results therefore
report mean physical potential energy and the diagnostic ring Hamiltonian,
and explicitly mark quantum kinetic/internal-energy estimators unavailable.
PIMD trajectory time remains sampling time. Diffusion requires separate
equilibration and unthermostatted production, with bead, timestep, size and
statistical convergence checked for the selected constrained model.

Tests cover independent mass-metric momentum projection, Maxwell degrees of
freedom, diatomic and triangular-water constraints, momentum/angular-momentum
conservation, time reversal, second-order energy error, periodic image bonds,
failure rollback, unchanged completed checkpoints and public input handling.
