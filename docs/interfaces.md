# Public interface contract for 0.2

## Workflow entry points

```python
from pyrpmd import JobSpec, run, prepare, resume_run, analyze_run

report = prepare("job.inp", numerical=True)
result = run("job.inp", output="runs/new-job")
result = resume_run("runs/new-job", execution={"block_steps": 128})
summary = analyze_run("runs/new-job")
```

`run` and `prepare` accept a `JobSpec`, a mapping, or a TOML/Python input path.
Every fresh input goes through the same strict schema and unit resolution.
`run(..., resume=True)` validates the saved scientific identity. `resume_run`
reads the saved resolved input; `parallel=` changes the worker layout and
`execution=` accepts only `loop_backend`/`block_steps`.

`prepare(..., numerical=False)` resolves input and prepares declared reactions
without a general PES evaluation. `numerical=True` additionally evaluates the
PES and requested TS preparation. A reaction network checks each effective PES,
including TS overrides. `ready` means that the prerequisites for the requested
pilot calculation are present; it is not a physical-convergence certificate.

`analyze_run(..., recompute=True, bins=..., bootstrap=...)` reanalyzes saved
umbrella or metadynamics data. Metadynamics rejects unsupported nonzero
bootstrap requests. Without recomputation it reads any workflow's summary.
For a diffusion run, `analyze_run(..., recompute=True, diffusion={...})` reads
saved unwrapped centroid data and can change atom/axis/reference selections,
lag and fit windows without a PES evaluation. `analyze_diffusion` is also
exported as a standalone array API; see [materials.md](materials.md).
Command-line commands expose these same entry points, plus `validate`, `info`,
`checkpoint` and PES adapter tools.

| `task` | Output meaning |
| --- | --- |
| `rate` | Single entrance/TS Bennett--Chandler PMF and recrossing, with physical/finite-sampling diagnostics |
| `drpmd` | Direct collisions with explicit channel/outcome records and uncertainty rules |
| `fes` | Fixed-bias 1D/2D umbrella FES; optional pilot adaptation followed by independent production |
| `network` | Entrance → channel label → TS matrix of independent complete rate calculations; no implicit sum |
| `metadynamics` | Standard/WT bias history and finite-time 1D/2D bias FES estimates; no rate inferred from biased time |
| `pimd` | Fixed-cell or isotropic BZP NPT sampling, quantum energy estimators and ring spread; fixed atoms supported at fixed cell |
| `diffusion` | Equilibrate then unthermostatted RPMD; unwrapped centroid MSD/tensor, VACF, uncertainty and finite-record diagnostics |
| `site_rate` | Finite-basin QTST and paired recrossing for one constant-gradient centroid surface; rate in s⁻¹ |

Workflow return values are JSON-compatible dictionaries. MPI workers may
return `None`; the root result and `results/summary.json` carry software version,
output path, preset and input diagnostics. Workflow-specific schemas and
physical qualifiers remain visible rather than being reduced to a scalar.

## Dynamics and capability negotiation

`Dynamics(..., integrator=...)` accepts the three canonical choices documented
in [integrators.md](integrators.md). `integrator_name` reports the selected
method; `integrator` remains the existing integrator object. The older class
name `CayleyIntegrator` is retained, and its `scheme=` argument chooses the
actual free-ring method.

`native_capabilities()` reports the methods supported by the loaded extension.
The Python and compiled versions negotiate capabilities before execution. An
old Cayley-only extension cannot silently execute a normal-mode or Verlet
request as Cayley. `native="required"` and `loop_backend="native"` are strict;
automatic fallback records its reason.

`propagate()` is a whole-block transaction and `propagate_block()` retains a
successful prefix on failure. Neither interface changes its failure meaning
in 0.2. The [native API](native-propagation.md) and
[segmented API](block-propagation.md) documents remain authoritative for their
observations, result arrays and exception payloads.

## Potential and CV plugins

The existing **v1** C ABI and its header remain compatible. **v2** adds explicit
per-structure cells/PBC and optional physical virials while retaining synchronous
host-memory batching and the same create/metadata/evaluate/destroy lifecycle.
See [the versioned ABI specification](../native/ABI.md) and
[`pes_api_v2.h`](../native/include/pyrpmd/pes_api_v2.h). Potential
evaluation receives atomic-unit `float64[B,N,3]` coordinates and returns
`energy[B]`, physical negative-gradient `forces[B,N,3]`, and status. Shape,
finite-value and lifecycle checks remain enforced. Thread safety, per-context
state and native stateless capability must be declared honestly; Python
callbacks do not become native just because a C++ ring substep is available.

`System` adds optional fixed `cell[3,3]` (row lattice vectors, bohr) and
`pbc[3]`. No-cell molecular serialization and existing identities are preserved.
Periodic requests must reach a cell-aware provider: C ABI v2, ASE, explicitly
capable Python/Torch, named periodic adapters, or i-PI. The molecular C ABI v1,
f2py and PySCF interfaces reject periodic systems. Generic Torch requires
`periodic=true` and `forward(q, cell, pbc)`; DeePMD rejects partial periodicity. `PESResult` adds optional physical configurational
`virial[B,3,3]` in hartree; it is not a quantum pressure estimator. Dynamics
keeps ring positions continuous and unwrapped.

A CV provides its declared estimator's `value[D]` and complete
`jacobian[D,P,N,3]`, canonical units and optional periods. Centroid and
bead-average estimators keep distinct meanings. Bias forces apply the full
Jacobian and the sum-beads Hamiltonian scaling. Metadynamics additionally
supports one to three CV dimensions, fixed-cell slabs, static quadratic walls,
and a serializable hill/thermostat history. `NoseHooverChain` wraps a `Dynamics`
instance for sampling; its `state_dict()` must be restored together with q/p.

Python factories must be importable in process/MPI workers. Paths resolve
relative to the input file. For plugin examples and exact adapter protocols,
see [potentials.md](potentials.md), [input.md](input.md) and
[enhanced-sampling.md](enhanced-sampling.md).

## Versioning and compatibility

User input v1, run manifest v2, checkpoint container v2 and PES ABI v1/v2 have
independent version numbers. A new distribution version does not imply a
new PES binary ABI. Unknown checkpoint/manifest versions fail explicitly;
legacy reads and run-identity migration follow [checkpoints.md](checkpoints.md).
Existing inputs retain Cayley as their default. New features require an
explicit task or option and do not silently change old numerical methods.
