---
orphan: true
---

# Native execution for analytic built-in potentials

The five scalar reference potentials in `pyrpmd.potentials.builtin` can provide
compiled C ABI v1 bindings to the existing fused propagation loops. With an
up-to-date `_native` extension, supported `Dynamics` blocks automatically use
those bindings under `loop_backend="auto"`. Calling `potential.evaluate(q)`
directly continues to use the Python/NumPy implementation.

The binding changes execution, with the same model parameters and physical PES
contract: float64 positions `[B,N,3]` in bohr, energy `[B]` in hartree, and
forces `[B,N,3]` equal to `-dV/dq` in hartree/bohr. There is no bead scaling,
precision change, device transfer, or additional scientific input option.
These analytic potentials remain reference models for examples and validation.

## Supported models and propagation plans

| Potential | Native model and parameter support |
| --- | --- |
| `FreePotential` | Zero energy and forces for any positive atom count. The Python evaluator retains its existing zero virial. |
| `HarmonicPotential` | Cartesian quadratic energy, including nonzero centers and finite nonnegative force constants broadcastable to `[N,3]`. |
| `MorsePairPotential` | One selected pair, with its original Euclidean distance and physical force convention. No periodic minimum-image transformation is introduced. |
| `DoubleWellPotential` | One selected Cartesian quartic coordinate, with the existing harmonic confinement on all remaining coordinates. |
| `PeriodicCosinePotential` | The model's fixed cell, periodic axes, repeats, selected atoms, and Cartesian confinement perpendicular to the periodic lattice vectors. |

Unthermostatted native blocks support `cayley-bcb`, `normal-mode-bab`, and
`velocity-verlet`. The existing composed native executor supports eligible
thermostat plans with Cayley and exact normal modes. Its velocity-Verlet and
cell limitations remain explicit. Biases, collective variables, constraints,
and other operators must also satisfy their existing native capability checks;
an available PES binding alone does not make every plan eligible. See the
[propagation stack](propagation-stack.md) for operator and RNG semantics.

For example, from an installation with the compiled extension:

```python
import numpy as np
from pyrpmd import Dynamics, State
from pyrpmd.potentials import HarmonicPotential

potential = HarmonicPotential(k=0.002)
dynamics = Dynamics(
    potential, [1836.0], beta=1000.0, nbeads=32, dt=0.1,
    native="required", loop_backend="auto", threads=1,
)
q = np.full((32, 1, 3), 0.2)
state = State(q, np.zeros_like(q))
dynamics.propagate(state, 100)
print(dynamics.loop_stats)
# {'native_steps': 100, 'python_steps': 0, 'fallback_reasons': {}}
```

`loop_backend="python"` selects Python scheduling for comparison.
`loop_backend="native"` requires complete native support and raises
`NativeUnavailableError` for an unsupported combination. Automatic capability
fallback is recorded in `loop_stats`; a runtime PES failure is never retried
using different mathematics or a different backend.

## Binding dimensions and lifetime

Ordinary `Dynamics` dispatch supplies the atom count before the first force
evaluation. Direct callers can request a binding explicitly:

```python
binding = potential.native_binding(natoms=1)
assert binding.kind == "cabi-v1"

potential.evaluate(q)
same_shape_binding = potential.native_binding()
```

The optional argument avoids inferring the system size from scalar parameters
or selected atom indices. The no-argument form uses the most recent evaluated
atom count; a fresh provider without a supplied count explicitly declines the
request. The compiled context validates the batch dimensions and buffer sizes
again at the ABI boundary.

Each binding owns an immutable C++ parameter snapshot. Repeated requests for
the same provider, dimensions, and parameters reuse that snapshot. Changing a
parameter and requesting another binding creates a new snapshot; an outstanding
older binding remains valid with its original parameters even after the Python
provider is released. Native capsules and locks are cached outside the provider
instance, so ordinary deepcopy and pickle retain the model data without
serializing process-local pointers. A persistent executor already bound to a
provider rejects a changed binding; create a new `Dynamics` when changing the
physical model.

Only exact built-in classes with their original `evaluate` method acquire
implicit bindings. Subclasses and instance- or class-level evaluator overrides
retain their Python semantics. A provider declaring coupled or stateful frames
also declines this stateless route. Harmonic parameters with more than two
dimensions decline native binding, including arrays with a frame axis; their
existing Python broadcasting remains available. Invalid or unsupported native
parameters fail explicitly.

A transparent, atomic-unit `PythonPotential` wrapper retains its existing
no-argument forwarding hook. If its built-in child has never been evaluated,
the child does not yet know its atom count, and the first automatic block can
use Python scheduling. Evaluating the wrapper once supplies those dimensions
for later blocks. Direct built-in objects do not need this initialization step.

## Fixed-cell periodic scope

`PeriodicCosinePotential` uses row lattice vectors, with fractional coordinates
`s = q @ inverse(cell)`. Its native context copies the already resolved inverse,
periodic heights, repeats, atom selection, and perpendicular projector. It
therefore preserves triclinic and partial-PBC behavior, including invariance
under translations by periodic lattice vectors and the existing confinement
outside their span. Coordinates are not wrapped or reassigned to different
images by the binding.

C ABI v1 has no cell argument or virial output. The fixed-cell cosine evaluator
can run in ordinary fused blocks because the model carries its fixed geometry
in the context. This does not add a variable-cell or pressure-control contract
to that potential. Its Python evaluator still does not return a virial, while
`FreePotential.evaluate()` still returns the established zero virial.

The composed thermal executor rejects molecular C ABI v1 providers whenever it
has a cell. Built-in dispatch declines that route before construction. An
eligible periodic thermostat plan uses its existing Python force scheduling
under `auto`; a forced fully native request fails explicitly. Free-particle
NPT retains the existing Python-force bridge, its cell setter, and zero-virial
contract. It does not become a fully native PES path through this binding.

## Checkpoints and scientific identity

Changing `loop_backend` remains an execution choice. Model parameters, atomic
units, integrator and thermostat definitions, run identities, and scientific
fingerprints keep their existing meanings. Native contexts are reconstructed;
their memory addresses are never checkpoint data.

For an exact built-in used without an explicit `potential_identity`, the
composed full native executor derives an internal restart identifier from the
model, atom count, and resolved parameter snapshot. This permits a fresh
`Dynamics` with the same built-in to restore its executor state while rejecting
changed parameters. The internal identifier adds no input option and does not
change the run-level scientific fingerprint. Explicit caller identities and
other provider identity policies remain in effect.

The existing checkpoint path retains the thermostat auxiliary state and RNG
stream. Backend switching reuses that state rather than starting a second
stream. Regression tests cover native/Python maps, operator combinations, and
checkpoint continuation; these software checks do not establish physical
sampling convergence.

## Reproduce the timing comparison

From an installed checkout, run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python examples/benchmark_builtin_native.py \
  --label current --backends auto python --output builtin-native-timing.json
```

The script measures `HarmonicPotential(k=0.002)` at `N=3,300,3000`, `P=32`,
with single-threaded Cayley propagation and no thermostat. Construction and
10 warm-up steps are excluded. Each timed call includes per-step energy
accounting and one final observation without centroid output. The default is
seven repeats with about 0.25 seconds per repeat, subject to bounded step
counts. It reports all samples and their median in microseconds per step,
the selected backend, loop counters, platform, CPU information, and thread
settings. Use an otherwise idle machine for comparisons.

The original `microseconds_per_step` fields measure elapsed wall time with
`perf_counter_ns`. Separate `process_cpu_microseconds_per_step` fields measure
CPU time consumed by this process with `process_time_ns`, with the same timed
propagation region and per-repeat statistics. On a shared host, wall time also
includes time waiting to be scheduled; report both when host load changes
between runs. Process CPU time helps distinguish that waiting from execution,
but CPU frequency and shared cache or memory contention can still affect it.

For a true before/after comparison, run the same script against both installed
revisions with `--backends auto` and distinct `--label` and `--output` values.
The explicit Python mode in the updated revision is an additional control.
Use `--steps` to fix the timed step count or `--target-seconds` to change the
default timing duration. These are propagation timings for an analytic model,
not estimates of molecular PES cost or physical convergence.

## Model families outside this binding

`potentials/diabatic.py` adapts arbitrary Python matrix PES callbacks.
`potentials/nonadiabatic_models.py` supplies Tully I–III, linear
vibronic-coupling, and discrete spin-boson models. They return Hermitian
`V[B,S,S]` and positive gradients `[B,S,S,N,3]`, which the scalar PES C ABI v1
cannot represent. Their model identities and implementations are unchanged.

`NaH2LEPS5G`, external `ChemPotPyDPEM` matrices, and state-selected wrappers
also retain their current paths. A future extension needs a matrix ABI or
explicit state-projection interface, together with tests for electronic
indices, gradient signs, complex values, propagation, and restart identity.

## Periodic Lennard-Jones C ABI v2

`LennardJonesPotential` is an analytic, fixed-cell condensed-phase reference.
It is selected with `provider = "lj"` (or `"lennard_jones"`) and requires a
full-rank cell with all three periodic axes. Parameters are `epsilon` (energy),
`sigma`, `cutoff`, and `skin` (lengths), plus the explicit execution choice
`backend = "native"` or `"numpy"`. Numeric values are atomic units; quantity
strings and `(value, unit)` pairs are parsed by the standard unit layer.

```toml
[system]
cell = [[18.0, 0.0, 0.0], [1.5, 17.0, 0.0], [0.5, 0.8, 16.0]]
pbc = [true, true, true]

[potential]
provider = "lj"

[potential.params]
epsilon = "0.0103 eV"
sigma = "3.40 angstrom"
cutoff = "8.50 angstrom"
skin = "0.40 angstrom"
backend = "native"
```

The pair energy is truncated and shifted so that it is zero at `cutoff`; the
force is the unshifted derivative inside the cutoff and is zero at and beyond
the cutoff. The cutoff may not exceed half the shortest cell face height. The
minimum image follows the row-lattice convention in `geometry.py` and performs
an exact bounded closest-lattice search for triclinic cells rather than simple
fractional rounding.

The native evaluator uses C ABI v2 because the call carries per-frame cells,
PBC flags, and physical virials. It caches a deterministic Verlet pair list for
each frame and rebuilds it after a cell change, atom-count change, or maximum
minimum-image displacement greater than half the skin. Pair iteration and each
frame's floating-point accumulation order are independent of the OpenMP thread
count. The cache is execution state only: it is reconstructible and is not
checkpoint or scientific-identity state.

`backend = "native"` fails explicitly when the compiled extension is missing
or stale. `backend = "numpy"` selects the independent all-pairs reference; the
provider never silently changes backend, precision, device, cutoff, or cell.
The v2 model is not advertised as a C ABI v1 fused molecular binding, so an
unsupported propagation combination declines native fusion rather than losing
its cell or virial contract.

The returned virial is the physical tensor `-dE/dstrain` under homogeneous
fixed-fractional-coordinate strain. Tests compare forces to coordinate finite
differences, all nine virial components to finite strain, the shifted cutoff,
an FCC nearest-neighbour lattice sum, native/NumPy frames, and—when installed—
ASE's unsmoothed Lennard-Jones calculator for triclinic cells. These software
checks validate implementation consistency; they do not establish a physically
converged material model.

The scaling campaign and raw timings are recorded in
`docs/validation/native-threading-2026-11-03/`. The reproducible public driver
is `benchmarks/efficiency/benchmark_native_threading.py`.
