# PyRPMD PES ABI v1 and v2

## Version 2: periodic cells and physical virials

Version 2 is implemented by `CABIPotential` and declared in
`include/pyrpmd/pes_api_v2.h`. It retains synchronous, contiguous host-memory
batch evaluation and adds an explicit cell and PBC flags for every frame,
plus optional physical configurational virials. Fixed composition is shared
by all frames; variable compositions and device pointers remain outside this
ABI. Batches can contain beads or several independent trajectories.

The loader selects v2 when `pyrpmd_pes_abi_version_v2` is exported, otherwise
it uses v1. An explicitly supplied `abi_version=1` or `abi_version=2` selects
that version without fallback. A malformed v2 implementation is an error;
it is never silently retried as v1. Existing v1 libraries and the legacy
`nrpmd_*_v1` symbol aliases continue to work unchanged.

Export all five symbols with the `_v2` suffix: `pyrpmd_pes_abi_version_v2`,
`pyrpmd_pes_create_v2`, `pyrpmd_pes_metadata_v2`,
`pyrpmd_pes_evaluate_v2`, and `pyrpmd_pes_destroy_v2`. The version function
returns 2. Create, metadata sizing, ownership, status/error handling and
process-local serialization follow the v1 lifecycle described below. The
v2 structures are **not** compatible with a v1 evaluation function pointer.

### Geometry and virial contract

| Field | Shape and type | Meaning |
| --- | --- | --- |
| `positions` | `[B,N,3]`, float64 | Unwrapped Cartesian positions in bohr |
| `cells` | `[B,3,3]`, float64 | Lattice vectors as **rows**, in bohr |
| `pbc` | `[B,3]`, int32 | Exactly 0 or 1 for each lattice direction |
| `energies` | `[B]`, float64 | Physical potential energy in hartree |
| `forces` | `[B,N,3]`, float64 | `-dE/dq` in hartree/bohr |
| `virials` | `[B,3,3]`, float64 | Physical configurational virial in hartree |

All arrays use C order, and all counts are numbers of elements, not bytes.
The cell count must equal `9*B`, except that `cells=NULL` and
`cells_count=0` represent a batch with no cell. In that case all PBC flags
must be zero. The PBC buffer is always supplied, including for isolated
molecules. Supplied cells must be finite and nonsingular even for partial
periodicity, and the provider must honor all three flags. Coordinates are
never independently wrapped by the loader.

Virial means

```text
W[a,b] = -dE / d epsilon[a,b]
q' = q @ (I + epsilon).T
H' = H @ (I + epsilon).T
```

The derivative deforms Cartesian positions and the row-vector cell together.
It therefore includes **explicit cell dependence at fixed fractional
coordinates**. In particular, using only `sum(q outer F)` is generally
incorrect for periodic potentials. For rotationally invariant systems the
physical tensor is symmetric; the loader never silently symmetrizes it.
Tensile-positive stress satisfies `stress = -W / abs(det(H))`. No kinetic
energy, ring-polymer springs, constraints, bias, or bead normalization belongs
in the returned tensor. It is not a complete quantum pressure estimator.

Metadata must declare `capabilities.virial` as a boolean. When it is true,
the loader supplies a virial output buffer and requires finite virials for
every successful frame. When false it supplies `virials=NULL` and
`virials_count=0`, and `PESResult.virial` is `None`. A zero tensor must never
be substituted for unavailable stress/virial. A provider that supports
virials must also allow a caller to omit that buffer.

For a failed frame, set nonzero `structure_status` and NaN energy, force,
and requested virial entries. A nonzero global return invalidates all
outputs. Providers must fill every status even when some frames fail.
The Python loader initializes output sentinels so missing successful outputs
or unwritten statuses cannot be mistaken for a valid result.

### V2 metadata

The existing composition, state, units, dtype and position layout fields
remain required. V2 adds these fields:

```json
{
  "schema": "pyrpmd.pes/2",
  "abi_version": 2,
  "boundary_conditions": "periodic",
  "cell_layout": "B,3,3:C:rows",
  "pbc_layout": "B,3:C",
  "virial_unit": "hartree",
  "virial_layout": "B,3,3:C",
  "virial_convention": "minus_dE_dstrain",
  "capabilities": {
    "native_batch": true,
    "virial": true,
    "stateful": false,
    "thread_safety": "context_serial",
    "execution_device": "cpu"
  }
}
```

`boundary_conditions` is either `"periodic"` (full and partial periodicity
are supported) or `"nonperiodic"` (any true PBC flag is rejected).
Virial unit/layout/convention fields are required when virial capability is
true. The host makes one compiled call for the whole batch; the provider
may implement that batch using vectorization, internal threads, or a loop.
`native_batch` describes its implementation and does not authorize the
host to call one context concurrently. The current loader serializes all
calls into the same library within a process, including geometry rebinding
and context destruction. Separate workers use separate contexts.

### Python use and example

```bash
cc -O3 -std=c11 -fPIC -shared -I native/include \
  native/examples/periodic_cosine_pes.c -lm -o periodic_cosine.so
```

```python
import numpy as np
from pyrpmd.potentials import CABIPotential

cell = np.array([[6., 0., 0.], [1., 5., 0.], [0.2, 0.3, 4.]])
system = {"symbols": ["H"], "atomic_numbers": [1],
          "cell": cell, "pbc": [True, True, False]}
with CABIPotential("./periodic_cosine.so", system=system) as pes:
    result = pes.evaluate(np.zeros((8, 1, 3)))
    assert result.virial.shape == (8, 3, 3)
    # Explicit per-frame cells and flags are available to batch consumers.
    result = pes.evaluate(np.zeros((2, 1, 3)),
                          cell=np.stack([cell, 1.01 * cell]),
                          pbc=[True, True, False])
    # Rebind a single system cell between integration steps if needed.
    pes.bind_system({**system, "cell": 1.01 * cell})
```

Per-call `cell` accepts `[3,3]` or `[B,3,3]`; `pbc` accepts three booleans
or `[B,3]` booleans. Omitted arguments use the bound System geometry. If
`cell` is explicitly provided and `pbc` omitted, a supplied cell defaults
to full periodicity, consistently with `System`; explicitly pass the flags
for a slab or wire. `cell=None` explicitly requests no cell. Per-call
overrides do not alter the bound geometry. `bind_system` validates composition
and electronic state and updates only the bound cell and PBC.

The example is a smooth analytic cosine lattice model, including arbitrary
triclinic and partial PBC and full cell derivatives. Its cell-free branch
is a Cartesian harmonic potential. It is an interface/finite-difference
test, **not a chemistry potential**. Fortran implementations can view
positions as `(3,N,B)` and cells/virials as `(3,3,B)` through `ISO_C_BINDING`,
while accounting explicitly for the transpose of matrix indices in C order.

The existing fused C++ propagation kernel accepts v1 descriptors only. V2
evaluation therefore uses the ordinary integrator with compiled batched PES
calls; exporting a v2 function as a v1 native binding is explicitly rejected.
This preserves periodic/virial semantics without claiming a fused v2 loop.

## Version 1: existing molecular ABI

The normative C declarations are in `include/pyrpmd/pes_api_v1.h`. This is a
synchronous host-memory ABI for fixed-composition, nonperiodic, single-surface
molecular potentials. It does not accept MPI communicators, device pointers,
periodic cells, variable atom counts or nonadiabatic couplings.

## Symbols and lifecycle

Export these unmangled C symbols:

1. `pyrpmd_pes_abi_version_v1() -> int32_t`, returning 1.
2. `pyrpmd_pes_create_v1(options, context**, error*)`.
3. `pyrpmd_pes_metadata_v1(context, json_buffer, capacity, required_capacity*, error*)`.
4. `pyrpmd_pes_evaluate_v1(context, input*, output*, error*)`.
5. `pyrpmd_pes_destroy_v1(context)`.

Create receives a UTF-8 JSON object containing user parameters and system
`natoms`, `symbols`, `atomic_numbers`, `charge`, `multiplicity` when available.
The library owns its opaque context until destroy. The caller owns every input
and output buffer. No pointers survive a synchronous evaluate call. No C++
exception or Fortran STOP may cross the ABI. Destroy accepts NULL.

The Python adapter currently serializes calls into a shared library within one
process, including create and destroy; parallel PES throughput uses separate
process-local contexts. The metadata may declare greater reentrancy, but this
version does not exploit it through concurrent calls on one library.

## Data, units and failure

- Positions are contiguous IEEE binary64 `[B,N,3]`, C order, in bohr.
- Energy is binary64 `[B]`, in hartree.
- Forces are binary64 `[B,N,3]`, in hartree/bohr, `F = -dV/dq`.
- Indices are zero based; sizes/capacities are signed int64, versions/enum-like
  codes are int32. `struct_size` is checked before interpreting members.
- Only physical PES energy/force belongs here. No spring, bias or P scaling.
- A nonzero global return invalidates the whole call. Global success means all
  frame statuses are defined. Failed frames have nonzero status and NaN data;
  they are never silently accepted for a dynamics step.
- Metadata sizing returns BUFFER_TOO_SMALL with required bytes including NUL.
  Insufficient metadata buffers never contain partial JSON. Diagnostic buffers
  may truncate messages but must retain valid NUL-terminated UTF-8.
- Check multiplication overflow and negative sizes before reading buffers.

## Required metadata checked by the loader

```json
{
  "schema": "pyrpmd.pes/1",
  "abi_version": 1,
  "provider_name": "my-potential",
  "provider_version": "1",
  "natoms": 3,
  "atomic_numbers": [1, 1, 1],
  "state": {"charge": 0, "multiplicity": 2},
  "layout": "B,N,3:C",
  "dtype": "float64",
  "position_unit": "bohr",
  "energy_unit": "hartree",
  "force_unit": "hartree/bohr",
  "boundary_conditions": "nonperiodic"
}
```

Atom order and electronic-state fields must match the requested System. Optional
capabilities may document batch evaluation, threading, statefulness, execution
device and bead affinity. The current native loader does not automatically use
arbitrary optional capability fields to schedule internal GPU/MPI work.

Fortran wrappers should use `ISO_C_BINDING`, `BIND(C)`, `integer(c_int32_t)`,
`integer(c_int64_t)` and `real(c_double)`. Explicitly account for the memory
layout: viewing the C coordinates as `(3,N,B)` is convenient in Fortran.
The working C example is `native/examples/harmonic_pes.c`.
