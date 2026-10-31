# Ring-polymer contraction and multiple time stepping

PyRPMD provides two explicit Python APIs for reducing evaluations of an expensive
potential component. `SplitPotential` implements ring-polymer contraction (RPC),
and `MTSDynamics` integrates a supplied fast/slow force split with two-level
r-RESPA. They reuse the existing real-mass ring propagator and `State` arrays.
These APIs are opt-in; existing workflow defaults are unchanged.

## Define the Hamiltonian first

Let the physical potential be `V = Vfast + Vslow`. The fast part should be cheap
and contain the rapidly varying forces. The slow part must be smooth enough to
support the chosen contraction and outer step. A common choice is a cheap
reference potential and the correction `Vslow = Vtarget - Vreference`. Passing
the complete target potential as `slow` while also passing the reference as
`fast` would instead simulate their sum.

For `P` full beads and `Pc` contracted beads, the implemented potential energy is

```{math}
U_{\mathrm{RPC}}(q) = \sum_{j=1}^{P} V_{\mathrm{fast}}(q_j)
 + \frac{P}{P_c}\sum_{a=1}^{P_c} V_{\mathrm{slow}}((Tq)_a).
```

`T` retains low ring normal modes and preserves the centroid. The lifted force
is the derivative of this total energy:

```{math}
F_j = F_{\mathrm{fast},j}
 + \frac{P}{P_c}\sum_a T_{aj}F_{\mathrm{slow},a}.
```

`Pc=P` uses the identity map exactly. `Pc=1` evaluates the correction at the
centroid and applies that correction force equally to all beads. Intermediate
even sizes retain the cosine member of the final degenerate mode pair, matching
the conventional i-PI rescaling choice. The discarded sine partner makes this
cutoff convention explicit; it can introduce dependence on which bead is used
as the first bead. Odd contracted sizes retain complete sine/cosine pairs and
avoid this particular asymmetry, although a nonlinear contracted potential can
still have quadrature/aliasing error. Compare different contracted sizes and
converge toward `Pc=P`; neither parity choice establishes contraction convergence.

RPC at `Pc<P` changes the Hamiltonian. In particular, evaluating an arbitrary
full reactive PES only at the centroid is not an exact replacement for full-bead
RPMD. Validate the contraction against increasing `Pc`, separately from full
bead-count, time-step, sampling and reaction-rate convergence.

## Python example

All numerical inputs below use PyRPMD atomic units: bohr, hartree, electron
masses and atomic time. The artificial harmonic model illustrates the API; it
is not a molecular reaction benchmark.

```python
import numpy as np
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.potentials.contraction import DifferencePotential, SplitPotential
from pyrpmd.multitime import MTSDynamics

reference = HarmonicPotential(k=0.4)
target = HarmonicPotential(k=0.45)
split = SplitPotential(
    fast=reference,
    slow=DifferencePotential(target, reference),
    nbeads=32,
    contracted_beads=3,
)

dyn = MTSDynamics(
    split, masses=[1837.0], beta=1000.0, nbeads=32,
    dt=2.0, inner_steps=4, integrator="normal-mode-bab",
)
state = State(np.zeros((32, 1, 3)), dyn.sample_momenta())
dyn.propagate(state, 100)       # 100 outer steps; 400 inner ring/fast steps
print(state.time, dyn.energy(state), dyn.force_evaluations)

# Ordinary Dynamics can also use the contracted Hamiltonian without MTS.
sampling = Dynamics(split, [1837.0], 1000.0, 32, 0.5)
# sampling.thermalize(state, ...) or sampling.sample_step(state, ...) then
# uses the specified contracted Hamiltonian, not the original target one.

reference.close()              # wrappers borrow; callers own the providers
target.close()
```

## What the MTS step does

For outer time step `dt` and `inner_steps=M`, `MTSDynamics` applies a slow
half-kick, `M` fast half-kick / free-ring / fast half-kick steps of length
`dt/M`, and a final slow half-kick. The free-ring map is selected through the
existing `normal-mode-bab`, `cayley-bcb` or `velocity-verlet` kernel. The last
option retains its existing spring stability check at the inner time step.
No thermostat, fictitious mass change or automatic force splitting is applied.

For a block of `L` outer steps, force reuse gives `L+1` slow calls and
`L*M+1` fast calls. RPC reduces each slow call from `P` to `Pc` geometries.
For sufficiently long blocks, expensive-geometry count therefore falls by
approximately `M*P/Pc` compared with full-bead evaluation at every inner step.
This is a call-count estimate, not a guaranteed wall-time speedup: reference
forces, data movement, provider batching, and contraction overhead still cost
time. No endpoint cache survives a public propagation call.

The clock counts outer steps. `step()` and `propagate()` commit state only after
the requested step/block succeeds. `propagate_block()` commits each completed
outer step and preserves that valid prefix if a later step fails. None of these
methods returns an observation record; they return the same `State` object.
`force_evaluations` counts attempted propagation calls, including failed calls;
explicit `energy()` queries are separate.

## Supported scope and checks

- Components must return independent configuration energies and forces under
  the ordinary PES contract. Declared stateful or ring-coupled components are
  rejected. A custom provider is responsible for being a deterministic function
  of its coordinates.
- A `SplitPotential` acts on one complete ring. Its metadata rejects flattened
  cross-trajectory batching. Contract each trajectory separately before any
  future provider-side combination.
- For `Pc<P`, returned slow energies are distributed uniformly across the `P`
  output rows solely so their sum equals the contracted Hamiltonian energy.
  These rows are not physical bead energies. Use finite differences of the
  **total** energy to check the force; `check_potential` assumes independent
  frames and is not the appropriate checker for this coupled wrapper.
- The wrapper exports no contracted virial or pressure estimator. Periodic
  applications require consistent unwrapped ring coordinates and appropriately
  bound component providers; wrapping beads individually before contraction
  changes the projected geometry.
- `MTSDynamics` currently provides deterministic Python propagation with the
  existing native free-ring kernel when available. It does not yet provide the
  workflow CLI, bias/constraint callbacks, trajectory observation records or a
  dedicated checkpoint adapter. Existing workflows continue using their
  established integrators.
- MTS resonance and force-splitting errors are independent of RPC errors. A
  stable finite trajectory does not demonstrate converged equilibrium sampling
  or an accurate reaction rate. Compare both the outer time step and `Pc`
  independently before a production calculation.

`tests/test_rpc_mts.py` checks the force against total-energy finite differences,
centroid and identity limits, even-cutoff convention, exact component
normalization, output-buffer ownership, scalar single-step equivalence,
reversibility, second-order convergence, call counts, and both failure rollback
boundaries. The reproducible model benchmark is
`python benchmarks/rpc_mts.py --output /tmp/rpc-mts.json`.

## References

- T. E. Markland and D. E. Manolopoulos, *J. Chem. Phys.* **129**, 024105 (2008),
  [original RPC formulation](https://doi.org/10.1063/1.2953308).
- V. Kapil, J. VandeVondele and M. Ceriotti, *J. Chem. Phys.* **144**, 054111
  (2016), [combined real/imaginary-time splitting](https://doi.org/10.1063/1.4941091).
- O. Marsalek and T. E. Markland, *J. Chem. Phys.* **144**, 054112 (2016),
  [reference-potential RPC](https://arxiv.org/abs/1512.00473).
- [i-PI implementation overview](https://docs.ipi-code.org/features.html) and
  [MTS/RPC tutorial by its developers](https://atomistic-cookbook.org/examples/pi-mts-rpc/mts-rpc.html).
- [i-PI normal-mode rescaling implementation](https://github.com/i-pi/i-pi/blob/main/ipi/utils/nmtransform.py),
  for the documented convention when the contracted bead count is even. The
  PyRPMD projection above was independently implemented from its trigonometric
  normal-mode expression.
