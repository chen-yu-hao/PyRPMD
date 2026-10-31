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

## Additional scientific APIs

Temperature and isotope studies reuse the workflows and checkpoints. Instanton and quantum-statistics APIs use `Potential.evaluate` and leave real-time dynamics explicit. See [rate studies](../rate-studies.md), [instantons](../instanton.md) and [quantum statistics](../quantum-statistics.md) for normalization and scope.

```{autofunction} pyrpmd.studies.plan_rate_study
```

```{autofunction} pyrpmd.studies.run_rate_study
```

```{autofunction} pyrpmd.studies.arrhenius_fit
```

```{autofunction} pyrpmd.studies.kinetic_isotope_effect
```

```{autofunction} pyrpmd.instanton.optimize_instanton
```

```{autofunction} pyrpmd.instanton.instanton_rate_1d
```

```{autoclass} pyrpmd.path_integral.PathIntegralAction
:members: evaluate, evaluate_batch
```

```{autofunction} pyrpmd.path_integral.sample_path_integral
```

```{autoclass} pyrpmd.path_integral.OpenPathAction
```

```{autofunction} pyrpmd.path_integral.sample_open_path
```

```{autofunction} pyrpmd.quantum_statistics.energy_heat_capacity
```

```{autofunction} pyrpmd.quantum_statistics.quantum_kinetic_energy
```

```{autofunction} pyrpmd.quantum_statistics.isotope_substitution_free_energy
```

```{autofunction} pyrpmd.quantum_statistics.isotope_fractionation
```

```{autofunction} pyrpmd.quantum_statistics.momentum_distribution_from_open_paths
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
:members: step, energy, sample_momenta, propagate, propagate_block, sample_step, trpmd_step, loop_stats
```

```{autofunction} pyrpmd.dynamics.native_capabilities
```

`step()` takes one thermostat-free deterministic step. `sample_step()` takes one sampling step with
the matching thermostat. `energy()` returns the summed-bead ring Hamiltonian. `propagate()` and
`propagate_block()` differ in what they commit on failure; see [Execution and restart](execution.md).

## Trajectory batches, RPC/MTS and spectroscopy

```python
from pyrpmd import BatchedState, BatchedDynamics, MTSDynamics
from pyrpmd import RingPolymerContraction, SplitPotential, DifferencePotential
from pyrpmd.analysis.spectroscopy import (
    velocity_autocorrelation, dipole_autocorrelation, flux_correlation,
    power_spectrum, relative_ir_spectrum,
)
```

`BatchedState` stores `[K,P,N,3]` arrays and an independent clock for each trajectory.
`MTSDynamics` separates the outer timestep from the number of inner steps; RPC
requires an explicitly declared slow potential correction. `Dynamics.trpmd_step()`
retains physical nuclear masses and applies friction only to internal ring modes.
Examples and supported combinations are in [Trajectory batching](../batched-dynamics.md),
[RPC/MTS](../rpc-mts.md) and [TRPMD and spectra](../trpmd-spectroscopy.md).

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
takes the energy matrix `[B,K,K]` and its positive gradient `[B,K,K,N,3]`, both finite and
Hermitian in the electronic indices. Complex spin-orbit couplings retain their phases.

```{autoclass} pyrpmd.nonadiabatic.DiabaticResult
```

```{autoclass} pyrpmd.nonadiabatic.NRPMDState
:members: copy, populations
```

```{autoclass} pyrpmd.nonadiabatic.NRPMDDynamics
:members: step, energy, populations, asymptotic_coupling
```

## Enhanced sampling APIs

```python
from pyrpmd import (
    OPESBias, PlumedBias, TransitionInterfaceSampler, TISPath,
    estimate_effective_flux, combine_tis_rate,
)
```

`task="opes"` uses the standard run/prepare/resume/analyze entry points.
`OPESBias` also supports direct integration through explicit `update()` events;
force evaluations do not change the probability estimate.
See [OPES](../opes.md) for fixed-bandwidth and frozen-production conventions.

`PlumedBias.step(dynamics, state)` owns the PLUMED update clock and calls the
existing integrator. `evaluate()` returns read-only diagnostics without a
history update. The optional PLUMED runtime is loaded only when constructing
the bridge. See [PLUMED](../plumed.md) for units and its external restart files.

`TransitionInterfaceSampler` generates and samples unbiased path ensembles
using a scalar CV or callable. `TISPath` owns each q/p path; the optional sample
callback can stream retained paths, including rejected-move repeats.
`estimate_effective_flux` consumes a separate unbiased trace, and
`combine_tis_rate` returns a candidate first-order rate with its normalization
and convergence status explicit. See [TIS](../transition-interface-sampling.md).
## Electronic friction and thermal nonadiabatic models

These classes are also available from `pyrpmd`. The [nonadiabatic chapter](nonadiabatic.md)
distinguishes the initial distributions and method restrictions. Friction uses a scalar
mean-force PES; the other methods use a complete diabatic matrix and positive gradients.

```{autoclass} pyrpmd.electronic_friction.ElectronicFrictionDynamics
:members: step, propagate
```

```{autoclass} pyrpmd.electronic_friction.ConstantElectronicFriction
:members: from_rates
```

```{autoclass} pyrpmd.electronic_friction.FrictionResult
```

```{autoclass} pyrpmd.thermal_nrpmd.ThermalNRPMDSampler
:members: sample, snapshot, restore
```

```{autoclass} pyrpmd.thermal_nrpmd.ThermalNRPMDDynamics
:members: step, energy, population_flux
```

```{autofunction} pyrpmd.thermal_nrpmd.thermal_flux_side
```

```{autofunction} pyrpmd.thermal_nrpmd.thermal_population_rate
```

```{autoclass} pyrpmd.surface_hopping.RPSHDynamics
:members: initialize, step, energy, populations, attempt_hop
```

```{autoclass} pyrpmd.mv_rpmd.MVRPMDDynamics
:members: step, energy, populations
```

```{autoclass} pyrpmd.kc_rpmd.KCRPMDDynamics
:members: step, energy
```

```{autoclass} pyrpmd.isomorphic.IsomorphicPotential
:members: evaluate
```

```{autoclass} pyrpmd.isomorphic.IsomorphicEhrenfestDynamics
:members: step, energy
```

## Versioning

The package, input schema, run manifest, checkpoint container and PES ABI each carry their own
version; package 0.2.0 uses PES ABI v1. Names that start with an underscore are internal. Interface
and capability negotiation are described in the
[public interface conventions](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/interfaces.md).
