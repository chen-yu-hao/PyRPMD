# Python API reference

Signatures on this page are generated from the current source. The high-level API handles input
validation, run identity, parallel scheduling and restart. The low-level API gives direct control of
the integrator for method development; the caller manages sampling, results and object lifetimes.

## High-level entry points

```python
from pyrpmd import JobSpec, prepare, run, resume_run, analyze_run
```

| Object/function | Input and return value |
| --- | --- |
| `JobSpec.from_file(path)` | Read a TOML or Python input and record the base for relative paths |
| `JobSpec.from_mapping(mapping, base_directory=...)` | Build a configuration from a copy of a mapping |
| `JobSpec.with_overrides(...)` | Return a new configuration with overrides applied |
| `JobSpec.resolve()` | Return `(config, identity)`; the identity is a `System`, or the PCET model |
| `prepare(job, numerical=False)` | Return preparation information; `numerical=True` runs the model preparation and PES |
| `run(job, output=None, resume=False)` | Run the task and return the JSON-compatible summary on the root rank |
| `resume_run(directory, parallel=None, execution=None)` | Continue from the saved input and checkpoints; `execution` accepts block size and backend |
| `analyze_run(directory, ...)` | Read the summary, or reanalyze offline for supported tasks |

`job` is a `JobSpec`, a dictionary or an input file path. Non-root MPI ranks return `None`, so handle
the return value on the root. `resume_run` keeps the saved run identity; a new scientific setting
needs a new task.

```{autoclass} pyrpmd.input.JobSpec
:members: from_file, from_mapping, with_overrides, resolve
```

```{autofunction} pyrpmd.runner.prepare
```

```{autofunction} pyrpmd.runner.run
```

```{autofunction} pyrpmd.runner.resume_run
```

```{autofunction} pyrpmd.runner.analyze_run
```

## System and State

`System` holds the atoms, nuclear masses, initial geometry, fragments, electronic state and optional
cell, PBC and fixed atoms. Its constructor takes atomic units: geometry and cell in bohr, masses in
electron masses. `System.from_dict()` reads a saved atomic-unit system; inputs with units go through
`JobSpec`.

`State.q` and `State.p` have shape `[P,N,3]`. `p` is the momentum, so divide by the atomic mass for
velocities. `step` counts steps and `time` is in atomic time. The constructor copies its arrays, and
`copy()` returns an independent state.

```{autoclass} pyrpmd.system.System
:members: natoms, atomic_numbers, to_dict, from_dict
```

```{autoclass} pyrpmd.dynamics.State
:members: copy
```

## A low-level dynamics program

```{literalinclude} ../../examples/manual/low_level.py
:language: python
```

Run it with:

```bash
python examples/manual/low_level.py
```

The program propagates deterministically from zero momenta and prints the change of the ring
Hamiltonian; for thermal averages use the `pimd` task or `sample_step()`. `beta` is in `hartree^-1`
and `dt` in atomic time; in an input, `temperature="300 K"` and `timestep="0.1 fs"` are converted for
you. The last line prints `loop_stats`: the Python `HarmonicPotential` has no native PES binding, so
every step runs in Python and the fallback reason says so.

```{autoclass} pyrpmd.dynamics.Dynamics
:members: step, energy, sample_momenta, propagate, propagate_block, sample_step, loop_stats
```

```{autofunction} pyrpmd.dynamics.native_capabilities
```

`step()` takes one thermostat-free deterministic step. `sample_step()` takes one sampling step with
the matching thermostat. `energy()` returns the summed-bead ring Hamiltonian. `propagate()` and
`propagate_block()` differ in what they commit on failure; see [Execution and restart](execution.md).

## Potentials

```{autoclass} pyrpmd.potentials.base.PESResult
```

```{autofunction} pyrpmd.potentials.make_potential
```

```{autofunction} pyrpmd.potentials.base.validate_result
```

```{autofunction} pyrpmd.potentials.base.check_potential
```

Adapters own their models and release them in `close()`. Shapes, signs, units, periodic binding and
the optional virial are specified in the [potential chapter](potentials.md).

## Units and checkpoints

```{autofunction} pyrpmd.units.parse_quantity
```

```{autofunction} pyrpmd.units.unit_factor
```

```{autofunction} pyrpmd.checkpoint.atomic_savez
```

```{autofunction} pyrpmd.checkpoint.load_checkpoint
```

```{autofunction} pyrpmd.checkpoint.inspect_checkpoint
```

`load_checkpoint()` is a context manager that verifies the container before exposing the data, and
never reads pickles or object arrays. The `kind` and schema of a custom payload are defined by the
caller.

## PCET kernels

The usual way to run PCET is through `JobSpec` and `run`. The kernels below serve model development
and independent checks; grids, potentials, masses and spectra are in atomic units. The energy zero,
log rates and truncation diagnostics are described in the [PCET tutorial](pcet.md).

```{autofunction} pyrpmd.pcet.solve_proton
```

```{autofunction} pyrpmd.pcet.electrode_log_kernels
```

```{autofunction} pyrpmd.pcet.golden_rule_rates
```

## NRPMD objects (experimental)

These objects are also importable from `pyrpmd` and serve collisions from a selected incoming
diabatic state; see the [nonadiabatic guide](nonadiabatic.md).

`NRPMDState` stores `mapping_q/mapping_p[P,K]` next to the nuclear `q/p[P,N,3]`. `DiabaticResult`
takes the energy matrix `[B,K,K]` and its positive gradient `[B,K,K,N,3]`, both real, finite and
symmetric in the electronic indices.

```{autoclass} pyrpmd.nonadiabatic.DiabaticResult
```

```{autoclass} pyrpmd.nonadiabatic.NRPMDState
:members: copy, populations
```

```{autoclass} pyrpmd.nonadiabatic.NRPMDDynamics
:members: step, energy, populations, asymptotic_coupling
```

## Versioning

The package, input schema, run manifest, checkpoint container and PES ABI each carry their own
version; package 0.2.0 uses PES ABI v1. Names that start with an underscore are internal. Interface
and capability negotiation are described in the
[public interface conventions](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/interfaces.md).
