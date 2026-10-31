# Connecting potentials and extension plugins

## The scalar PES interface

Every execution process constructs its own potential object. A call receives a batch of structures,
usually the beads that process handles, and returns the **physical energy and forces** of each
structure. PyRPMD adds ring springs, biases, constraints and bead scaling itself.

Experimental NRPMD uses a separate matrix interface that returns the full diabatic matrix and its
positive gradient; see the [nonadiabatic tutorial](nonadiabatic.md).

| Data | Shape | Unit / meaning |
| --- | --- | --- |
| Coordinates `q` | `float64[B,N,3]` | bohr, C-contiguous |
| Energy `energy` | `[B]` | Total potential energy of each structure, hartree |
| Forces `forces` | `[B,N,3]` | Negative energy gradient, hartree/bohr |
| Status `status` | `[B]` | 0 for success, non-zero for a failed structure |
| Messages `messages` | B strings | Optional failure descriptions |
| Virial | Optional `[B,3,3]` | hartree, when the provider implements it |

`B` is the number of structures in one call. Evaluate each structure independently.

## A complete Python provider

Save this harmonic potential as `my_pes.py`:

```python
import numpy as np
from pyrpmd.potentials import PESResult


class Harmonic:
    metadata = {"execution_device": "cpu", "thread_safety": "context_serial"}

    def __init__(self, k=0.002):
        self.k = float(k)

    def evaluate(self, q):
        energy = 0.5 * self.k * np.sum(q * q, axis=(1, 2))
        return PESResult(energy, -self.k * q)

    def close(self):
        pass


def build_pes(k=0.002):
    return Harmonic(k)
```

and point the input at it:

```toml
[potential]
provider = "python"
factory = "my_pes.py:build_pes"
params = {k = 0.002}
```

Keep the factory in its own importable module. Each worker process calls the factory and owns its
context, such as GPU handles, sockets or Fortran COMMON state, and releases it in `close()`.

## Units and gradient checks

Python, f2py and Torch adapters can declare the units of the model:

```toml
[potential]
provider = "python"
factory = "my_pes.py:build_pes"
units = {positions = "angstrom", energy = "eV"}
derivative = "force"
```

The model then works in Å and eV, and the adapter converts. A Python provider that returns positive
gradients declares `derivative="gradient"`. The legacy f2py interface always returns positive
gradients, which the adapter negates; Torch output is always interpreted as forces, or as the energy
whose autograd gives them. C ABI v1 uses atomic units.

Finite-difference a new model on a few representative configurations, and check the atom order,
energy zero, electronic state and domain of validity:

```python
import numpy as np
from pyrpmd.potentials import check_potential
from my_pes import build_pes

provider = build_pes()
try:
    q = np.array([[[0.1, -0.2, 0.3]]])  # bohr for this provider
    print(check_potential(provider, q, epsilon=1e-5))
finally:
    provider.close()
```

## Adapters

| Provider | Use |
| --- | --- |
| `builtin` / `free` / `harmonic`, … | Analytic model potentials |
| `python` | `module:factory` or `file.py:factory`; wraps your own model |
| `rpmdrate` / `f2py` | Legacy Fortran wrappers with `(3,N,B)` layout and positive gradients |
| `cabi` | Shared libraries implementing the versioned PyRPMD PES v1 ABI |
| `ase` | The factory returns an ASE Calculator; handles eV/Å, cell/PBC and stress |
| `torch` | Batched energies, or energies and forces, from a Torch model; molecular systems |
| `ipi` | Synchronous i-PI force protocol with an external driver |
| `diabatic_python` | Real symmetric matrix potential with gradients for experimental NRPMD |

For electronic calculations, D and T are passed as hydrogen and keep their nuclear masses.

The Torch adapter evaluates the model on its device and transfers coordinates and forces each call;
each trajectory process loads its own copy of the model.

## Periodic models

Periodic tasks need a provider that declares cell/PBC support. Python providers use one of two
protocols:

| Protocol | Declaration and call | Units |
| --- | --- | --- |
| Cell per call | `metadata["periodic"]=True`, `evaluate(q, *, cell, pbc)` | q and cell in the provider's declared length unit |
| Bound system | `metadata["supports_pbc"]=True`, `bind_system(system)` then `evaluate(q)` | The bound `system` in atomic units; q in the provider unit |

Implement one protocol explicitly. The materials examples also include the `periodic_cosine` model.
i-PI's POSDATA carries the fixed cell and its inverse but no PBC flags, so partially periodic
systems, and non-periodic systems with a cell, set `potential.params.init.pbc`, which the driver
reads from INIT. C ABI v1, f2py and Torch providers are rejected for periodic tasks.

Fixed layers are a dynamics constraint; the potential always receives every atom. See
[Materials and surfaces](periodic.md).

## Native shared libraries and Fortran

Use a `.so` that implements the ABI like this:

```bash
pyrpmd pes inspect ./libpes.so --geometry ts.xyz
pyrpmd pes check ./libpes.so --geometry ts.xyz --step 1e-5 --tolerance 1e-5
pyrpmd run job.inp --pes ./libpes.so --output runs/native-pes
```

`pes check` takes the step in bohr and the tolerance in hartree/bohr. An object file (`.o`) is
linked into a shared library with `pyrpmd pes build` and a manifest naming the wrapper, compiler,
PIC flags and link dependencies. f2py extensions use the `rpmdrate` / `f2py` provider and must match
the Python and NumPy ABI. The protocol and lifecycle are specified in the
[PES v1 ABI](https://github.com/chen-yu-hao/PyRPMD/blob/main/native/ABI.md).

## Custom CVs and classifiers

A custom CV factory returns an object whose metadata declares `dimension`, canonical `units` and
optional `periods`; `value_and_jacobian(state)` returns `value[D]` and `jacobian[D,P,N,3]`, with all
bead factors of its estimator included. The protocol covers static, differentiable coordinates.

`outcomes.factory` supplies a product classifier for dRPMD. Encode same-composition exchange and
isomerization channels explicitly. Complete examples and return values are in the
[plugin specification](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/potentials.md) *(partly Chinese)*
and the [2D custom CV](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/fes/custom_cv.py).
