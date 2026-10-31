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
whose autograd gives them. Both C ABI versions use atomic units.

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
| `cabi` / `native` | Versioned C ABI v1/v2 libraries; v2 adds cells/PBC and physical virials |
| `ase` | The factory returns an ASE Calculator; handles eV/Å, cell/PBC and stress |
| `torch` | Batched molecular or explicitly cell-aware periodic Torch models |
| `ipi` | Persistent i-PI client pool, evaluating multiple beads concurrently |
| `mace`, `nequip`, `fairchem` | Ready-made reusable ASE calculators for explicit ML checkpoints |
| `deepmd` | DeePMD `DeepPot` with native structure batching |
| `tblite` / `xtb` | On-the-fly GFN1/GFN2/IPEA1-xTB through tblite |
| `pyscf` | Molecular HF/DFT with analytic nuclear gradients |
| `diabatic_python` | Real symmetric or complex Hermitian matrix potential with gradients for experimental NRPMD |

For electronic calculations, D and T are passed as hydrogen and keep their nuclear masses.

ASE-backed adapters request the energy consistent with the forces
(`get_potential_energy(force_consistent=True)`). With electronic smearing this is usually the
calculator's electronic free energy, not its zero-temperature extrapolated energy and not
a nuclear PMF. The adapter falls back only when the calculator does not advertise `free_energy`
and ASE raises `PropertyNotImplementedError`; a failure of an advertised property remains an
error. Keep electronic and nuclear temperatures explicit. The energy convention is part of
the run identity: older ASE-backed runs cannot be resumed or recomputed with this adapter,
even if a classical calculator's two energies happen to coincide. Their saved summary remains
readable. Use a new output directory; new-convention checkpoints resume normally. Do not combine
old extrapolated energies or free-energy weights with the corrected ones. See the
[ASE energy contract](https://docs.ase-lib.org/ase/atoms.html#ase.Atoms.get_potential_energy).

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
reads from INIT. C ABI v1 and f2py remain molecular interfaces. C ABI v2, periodic Torch,
ASE-backed models and tblite support cell-aware evaluation; DeePMD requires full or no PBC.
The PySCF adapter currently supports molecules.

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
[PES v1/v2 ABI](https://github.com/chen-yu-hao/PyRPMD/blob/main/native/ABI.md).

## C ABI v2 and periodic Torch

A compiled library exporting the v2 symbols is selected automatically; set `abi_version=2` to
require it. Existing v1 libraries keep their original contract. The v2 batch includes
`cells[B,3,3]` (row lattice vectors, bohr), `pbc[B,3]` (int32 booleans), and optional
`virials[B,3,3]` (hartree). A structure's virial is `W = -dE/dstrain`; tensile stress is `-W/V`.
It contains no kinetic, ring-spring, bias or bead factor. V2 calls the compiled PES for the whole
batch, while the outer integrator uses the compatible Python loop.

```toml
[potential]
provider = "cabi"
library = "./libperiodic_pes.so"
abi_version = 2
```

For Torch, select `periodic=true` explicitly and implement `forward(q, cell, pbc)` with tensor
shapes `[B,N,3]`, `[B,3,3]`, `[B,3]`. q and cell use the model's declared position unit. The
model owns periodic neighbor construction and minimum images. `virial="auto"` accepts model
`virial` or tensile `stress`, or differentiates a common deformation of coordinates and cell.
Use `virial="autograd"` to require that derivative, `"provided"` to require model output, or
`"none"` to omit it. The generic Torch interface requires float64; that requirement does not
change the trained precision of the named ML adapters.

```toml
[potential]
provider = "torch"
factory = "periodic_model.py:build"
periodic = true
virial = "autograd"
device = "cpu"
units = {positions = "angstrom", energy = "eV"}
```

## Ready-made ML and electronic-structure adapters

Install only the extra needed for the selected provider, for example `pip install '.[tblite]'`
or `pip install '.[pyscf]'`. All engines are imported lazily. Checkpoints are explicit: PyRPMD
does not choose a chemical model for your system. NequIP uses an ASE-targeted compiled model;
fairchem requires an explicit `task_name` and accepts a registered model name or local checkpoint.

```toml
[potential]
provider = "mace"  # nequip, deepmd and fairchem are also available
model = "models/material.model"
device = "cpu"
```

For molecular electronic structure, the System supplies elements, charge and spin multiplicity:

```toml
[potential]
provider = "pyscf"
params = {method = "DFT", xc = "PBE", basis = "def2-svp", grid_level = 3, conv_tol = 1e-10}
```

Or use `provider="tblite"` with `params={method="GFN2-xTB"}`. SCF failure raises an error;
unconverged energies are not accepted. Calculators persist across frames and evaluation calls.
Electronic warm starts are opt-in and indexed by batch slot; keep each slot attached to the same
bead when using them. ASE-backed adapters evaluate frames sequentially, whereas Torch and DeePMD
have native batches. DeePMD device selection belongs to its backend and is not assigned by
PyRPMD's GPU scheduler. Detailed options, upstream APIs and validation boundaries are in the
[adapter guide](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/potential-adapters.md).

## Several i-PI clients on one server

```toml
[potential]
provider = "ipi"
params = {host = "127.0.0.1", port = 31415, clients = 4, timeout = 60.0}

[parallel]
backend = "serial"
bead_workers = 1
```

Start four identical force clients connected to the same address. The first call waits for all
four clients within the connection timeout. Each client evaluates one structure at a time and
then receives the next pending bead. Energy, force and virial arrays retain input order. The
server waits for the complete batch before dynamics advances. Clients and communication threads
persist between calls. `clients=1` retains single-driver operation.

The wire format remains standard single-structure i-PI. INIT carries the frame index only when
the driver requests initialization; there is no strict bead-to-client affinity. Drivers must
return the same physical PES independently of assignment. Each geometry has a deadline covering
its entire exchange. Malformed replies, missing clients, failed frames or disconnects close the
whole pool and fail the batch; the caller must create a fresh provider to retry. Use one PyRPMD
trajectory slot per socket server, or a custom factory supplying distinct addresses for separate
slots; automatic port offsets are not provided.

## Custom CVs and classifiers

A custom CV factory returns an object whose metadata declares `dimension`, canonical `units` and
optional `periods`; `value_and_jacobian(state)` returns `value[D]` and `jacobian[D,P,N,3]`, with all
bead factors of its estimator included. The protocol covers static, differentiable coordinates.

`outcomes.factory` supplies a product classifier for dRPMD. Encode same-composition exchange and
isomerization channels explicitly. Complete examples and return values are in the
[plugin specification](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/potentials.md) *(partly Chinese)*
and the [2D custom CV](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/fes/custom_cv.py).
