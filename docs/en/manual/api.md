# Python API reference

The function signatures on this page are generated from the current source. The high-level API
handles strict input, run identity, parallel scheduling and restart; the low-level API suits method
development, and the caller manages the sampling scheme, results and lifecycle.

## High-level workflow entry points

```python
from pyrpmd import JobSpec, prepare, run, resume_run, analyze_run
```

| Object/function | Input and return value |
| --- | --- |
| `JobSpec.from_file(path)` | Read a TOML/Python input and record the base for relative paths |
| `JobSpec.from_mapping(mapping, base_directory=...)` | Build an independent copy of a configuration |
| `JobSpec.with_overrides(...)` | Return a new configuration with overrides; the original is not modified |
| `JobSpec.resolve()` | Return `(config, identity)`; the identity is usually a `System`, for PCET its model identity |
| `prepare(job, numerical=False)` | Return preparation information; numerical mode calls the model preparation/PES |
| `run(job, output=None, resume=False)` | Run the task and return the JSON-compatible summary on the root |
| `resume_run(directory, parallel=None, execution=None)` | Read the saved input and checkpoints; execution accepts only block size and backend |
| `analyze_run(directory, ...)` | Read the summary by default; supported tasks can be reanalyzed offline |

`job` may be a `JobSpec`, a dictionary or an input file path. MPI workers may return `None`, so handle
the aggregated return value on the root only. To change scientific parameters, create a new task;
`resume_run` has no option to force past the identity check.

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

`System` records the atoms, nuclear masses, initial geometry, fragments, electronic state and optional
cell/PBC/fixed atoms. Direct construction uses atomic units: geometry/cell in bohr and masses in
electron masses. `System.from_dict()` reads a saved atomic-unit system; user configurations with units
should be resolved through `JobSpec`.

`State.q` and `State.p` have shape `[P,N,3]`. `p` is the momentum; divide by each atom's mass to get
velocities. `step` is a non-negative step count and `time` is atomic time. The constructor copies the
arrays and `copy()` creates an independent state.

```{autoclass} pyrpmd.system.System
:members: natoms, atomic_numbers, to_dict, from_dict
```

```{autoclass} pyrpmd.dynamics.State
:members: copy
```

## A complete low-level dynamics program

```{literalinclude} ../../../examples/manual/low_level.py
:language: python
```

Run it with:

```bash
python examples/manual/low_level.py
```

The example starts from zero momenta and demonstrates deterministic propagation and the change of the
ring Hamiltonian. There is no thermalization or production sampling, so it cannot estimate quantum
thermodynamic energies at 300 K. The variable `beta` is in `hartree^-1` and `dt` in atomic time; the
high-level input `temperature="300 K"` performs these conversions for you. Its last line prints
`loop_stats`: with the Python `HarmonicPotential` used here, all steps take the Python path and the
fallback reason names the missing native PES binding.

```{autoclass} pyrpmd.dynamics.Dynamics
:members: step, energy, sample_momenta, propagate, propagate_block, sample_step, loop_stats
```

```{autofunction} pyrpmd.dynamics.native_capabilities
```

`step()` is one thermostat-free deterministic step. `sample_step()` needs a matching integrator and
sampling method. `energy()` returns the summed-bead ring Hamiltonian, not a physical quantum energy
estimator. `propagate()` and `propagate_block()` commit differently on failure; see
[Execution and restart](execution.md).

## Potential interfaces

```{autoclass} pyrpmd.potentials.base.PESResult
```

```{autofunction} pyrpmd.potentials.make_potential
```

```{autofunction} pyrpmd.potentials.base.validate_result
```

```{autofunction} pyrpmd.potentials.base.check_potential
```

Adapters own their in-process models and the `close()` lifecycle. Follow the shapes, signs, units,
periodic binding and optional virial agreed in the [potential chapter](potentials.md); never connect
a provider without PBC support to a periodic task and keep ignoring the cell.

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

`load_checkpoint()` is used as a context manager and verifies the container before exposing data. It
never reads pickles or object arrays. The `kind` and business schema of a user payload are the
caller's responsibility; an identical NPZ container version does not make the physical meaning of a
payload irrelevant.

## PCET numerical kernels

PCET is a separate local quantum model, and the main user entry is still `JobSpec` and `run`. The
kernels below are for model development and independent validation; grids, potentials, masses and
spectral arrays are in atomic units. The electrode energy zero, log rates and truncation diagnostics
are described in the [PCET tutorial](pcet.md).

```{autofunction} pyrpmd.pcet.solve_proton
```

```{autofunction} pyrpmd.pcet.electrode_log_kernels
```

```{autofunction} pyrpmd.pcet.golden_rule_rates
```

## Experimental NRPMD objects

These objects can also be imported from `pyrpmd`. They serve direct collisions from a selected
incoming diabatic electronic state and do not build an electronic thermal-equilibrium PMF; the
experimental interface and physical scope are in the [nonadiabatic guide](nonadiabatic.md).

`NRPMDState` stores `mapping_q/mapping_p[P,K]` in addition to the nuclear `q/p[P,N,3]`.
`DiabaticResult` takes the energy matrix `[B,K,K]` and the positive gradient `[B,K,K,N,3]`. All matrix
elements and their derivatives must be real, finite and symmetric in the electronic indices.

```{autoclass} pyrpmd.nonadiabatic.DiabaticResult
```

```{autoclass} pyrpmd.nonadiabatic.NRPMDState
:members: copy, populations
```

```{autoclass} pyrpmd.nonadiabatic.NRPMDDynamics
:members: step, energy, populations, asymptotic_coupling
```

## Interface stability

The package version, input schema, run manifest, checkpoint container and PES ABI are versioned
independently. Package version 0.2.0 does not mean the PES ABI moved to v2. Helper functions whose
names start with an underscore are internal and should not be used as long-term extension interfaces.
More complete interface and capability-negotiation notes are in the
[public interface conventions](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/interfaces.md).
