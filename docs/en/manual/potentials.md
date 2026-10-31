# Connecting potentials and extension plugins

## Get the physical interface right first

High-level workflows construct the potential object separately in every execution process. One PES
call receives a batch of structures, usually several beads handled by that process. **Return only
the physical potential energy and forces**: do not mix ring springs, umbrellas, constraints,
temperature or `1/P` scaling into the model.

The scalar potential interface is described below. Experimental NRPMD uses a separate matrix PES
contract that returns the full diabatic matrix and its **positive gradient**; the scalar
negative-gradient/force interface does not apply there, see the
[nonadiabatic tutorial](nonadiabatic.md).

| Data | Shape | Canonical unit / meaning |
| --- | --- | --- |
| Coordinates `q` | `float64[B,N,3]` | bohr, C-contiguous |
| Energy `energy` | `[B]` | Total potential energy of each structure, hartree |
| Forces `forces` | `[B,N,3]` | Negative energy gradient, hartree/bohr |
| Status `status` | `[B]` | 0 means success; a failure must not be hidden behind zero forces |
| Messages `messages` | B strings | Optional failure descriptions |
| Physical virial | Optional `[B,3,3]` | hartree; only when the provider really implements it |

`B` is the size of one batch and is not guaranteed to equal the job's bead number P. A model must
evaluate different structures independently and must not mix energies across the batch dimension.

## A complete Python provider

Save the following non-periodic harmonic potential as `my_pes.py`:

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

Replace the potential section of an input in the same directory with:

```toml
[potential]
provider = "python"
factory = "my_pes.py:build_pes"
params = {k = 0.002}
```

Put the factory in a separate importable module. Do not pass non-serializable GPU contexts, sockets
or Fortran COMMON state from the parent process to all workers; every worker should create and close
its own context.

## Units and gradient checks

Python/f2py/Torch adapters can declare their own coordinate and energy units, for example:

```toml
[potential]
provider = "python"
factory = "my_pes.py:build_pes"
units = {positions = "angstrom", energy = "eV"}
derivative = "force"
```

Then the parameters and implementation of `my_pes.py` must really use Å and eV. If a Python provider
returns a positive gradient, declare `derivative="gradient"`; the adapter handles the sign and units.
The legacy f2py interface always receives a positive gradient and negates it; the direct output of
Torch is always interpreted as a force, or the negative gradient is obtained from energy autograd,
so this derivative switch does not apply to them. C ABI v1 always uses atomic units.

For any newly connected model, finite-difference the energy and forces on a few representative
configurations first, and check the atom order, the energy zero, the electronic state and the model
domain:

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

A force check diagnoses the program connection; it does not prove that the model is accurate on
every reaction path, temperature and periodic structure.

## Choosing an existing adapter

| Provider | Entry point and conditions |
| --- | --- |
| `builtin` / `free` / `harmonic`, … | Teaching analytic models; not real chemical PESs |
| `python` | `module:factory` or `file.py:factory`; suited to wrapping user models |
| `rpmdrate` / `f2py` | Legacy Fortran wrappers with `(3,N,B)` layout and positive gradients |
| `cabi` | A shared library implementing the versioned PyRPMD PES v1 ABI |
| `ase` | The factory returns an ASE Calculator; handles eV/Å, cell/PBC and available stress |
| `torch` | The factory/model provides batched energies or energies and forces; periodic systems unsupported |
| `ipi` | Synchronous i-PI force protocol; needs an external driver and connection |
| `diabatic_python` | Real symmetric matrix potential with gradients of every element for experimental NRPMD; selected-incoming-state dRPMD only |

Connecting a calculator through ASE does not guarantee the elements, periodic boundaries, spin,
download license or device support of a particular model. Periodic D/T nuclei are treated as the
element H in the electronic calculation while keeping their nuclear masses.

The PyRPMD Torch interface includes CPU/device transfers; it does not mean the whole integrator runs
on the GPU. Each trajectory process loads its own model, so memory needs grow accordingly.

## The explicit contract of periodic models

Periodic tasks must use a provider that declares cell/PBC support. The integrated release keeps two
existing Python protocols:

| Protocol | Declaration and call | Units |
| --- | --- | --- |
| Periodic parameters per call | `metadata["periodic"]=True`, `evaluate(q, *, cell, pbc)` | q and cell use the declared provider length unit |
| Bind the full system | `metadata["supports_pbc"]=True`, `bind_system(system)` then `evaluate(q)` | The bound `system` is in atomic units; q still follows the provider unit declaration |

A model should implement one of these protocols explicitly instead of guessing periodic support by
catching `TypeError`. Binding does not make a non-periodic potential periodic. The materials examples
also provide the `periodic_cosine` teaching model. i-PI's POSDATA carries the true fixed cell and its
inverse but no PBC flags; partially periodic systems, or non-periodic systems with a cell, also need
an explicit `potential.params.init.pbc`, and the driver must honor this INIT contract. The external
driver must implement the periodic problem correctly. The C ABI v1, legacy f2py and Torch periodic
combinations are currently rejected rather than ignoring the cell and continuing.

Fixed layers are handled by dynamics constraints; the potential must still see the structure of all
atoms. See [Materials and surfaces](periodic.md) for more boundaries.

## Native shared libraries and legacy Fortran

A `.so` implementing the ABI can be used like this:

```bash
pyrpmd pes inspect ./libpes.so --geometry ts.xyz
pyrpmd pes check ./libpes.so --geometry ts.xyz --step 1e-5 --tolerance 1e-5
pyrpmd run job.inp --pes ./libpes.so --output runs/native-pes
```

The step of `pes check` is in bohr and the tolerance in hartree/bohr. A plain `.o` is not a loadable
shared library; it needs a manifest stating the wrapper, compiler, PIC and link dependencies, and is
then built with `pyrpmd pes build`. An arbitrary `.so` is not a PyRPMD ABI library either: an f2py
extension needs the matching provider and must match the Python/NumPy ABI. The full protocol and
lifecycle are in the [PES v1 ABI](https://github.com/chen-yu-hao/PyRPMD/blob/main/native/ABI.md).

## Custom CVs and classifiers

A custom CV factory returns an object with metadata declaring `dimension`, canonical `units` and
optional `periods`; `value_and_jacobian(state)` returns `value[D]` and `jacobian[D,P,N,3]`. The
Jacobian must already contain all bead factors of its estimator. The current protocol covers static
differentiable coordinates; implicit finite differences or coordinates that depend on an external
clock are not treated as the same capability.

The dRPMD `outcomes.factory` defines a physically justified product classifier. Connectivity
inference is only for pilots; same-composition exchange or complex isomerization should be encoded
and validated explicitly. Complete examples and classifier return values are in the
[plugin specification](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/potentials.md) *(partly Chinese)*
and the [2D custom CV](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/fes/custom_cv.py).
