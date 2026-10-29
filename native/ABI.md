# PyRPMD PES ABI v1

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
