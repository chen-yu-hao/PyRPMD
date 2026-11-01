# Validation record: 2026-11-01

The comparison and additions start from PyRPMD
`2f907ba96d22cb134198715e398b1d9f8d83d929`. This record describes local
implementation checks, not converged physical predictions.

## Targeted regression suite

**616 passed in 15.04 s**, with Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0,
pytest 8.3.5, and PyRPMD's compiled C++/OpenMP extensions. Both thread-count
environment variables below were one. The exact selected test modules were:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q \
  tests/test_nonadiabatic.py tests/test_complex_nonadiabatic.py \
  tests/test_thermal_nrpmd.py tests/test_surface_hopping.py \
  tests/test_rpsh_decoherence.py tests/test_nonadiabatic_models.py \
  tests/test_nonadiabatic_ensemble.py tests/test_nonadiabatic_sampling.py \
  tests/test_isomorphic.py tests/test_kc_rpmd.py tests/test_mv_rpmd.py \
  tests/test_nonadiabatic_reference.py tests/test_nonadiabatic_task_input.py \
  tests/test_nonadiabatic_task_workflow.py \
  tests/test_nonadiabatic_checkpoint_validation.py \
  tests/test_nonadiabatic_output.py tests/test_nrpmd_input.py \
  tests/test_nrpmd_integration.py tests/test_nonadiabatic_public_extensions.py \
  tests/test_nonadiabatic_method_invariants.py tests/test_nrpmd_contraction.py
```

The restricted validation environment loaded unmodified official pytest 8.3.5,
pluggy 1.5.0 and iniconfig 2.0.0 tagged sources through `PYTHONPATH`, with
third-party plugin autoload disabled. Repository code and test assertions used
the normal pytest runner. Missing test helper modules were fetched from the
pinned repository revision before the final run.

Checks cover analytic model values and derivatives, Hermitian/SOC matrices,
independent harmonic covariance, electronic phase orientation, signed MMST
statistics, independent-trajectory SEM, EDC norm and rollback, malformed input,
and complete CLI integration. Checkpoint tests include interrupted propagation,
changing process/block settings, exact serial/process RNG consistency,
integrity resealing followed by logical corruption, and state/history agreement.

The full repository suite, MPI jobs and Sphinx builds were **not run locally**;
h5py, MPI and Sphinx were unavailable here. The existing GitHub workflows are
the appropriate gates for those checks; no remote CI outcome is asserted by
this local record.

## Executed CLI examples

Both input files passed `validate --check-pes`, then `run`,
`analyze --recompute`, and `resume --workers 1 --block-steps 64`:

| Input | Initial workers | Trajectories / frames | Final fixed-diabatic populations |
| --- | ---: | ---: | --- |
| `examples/nonadiabatic/population.inp` | 2 | 16 / 21 | 0.9074376072, 0.0925623928 |
| `examples/nonadiabatic/rpsh_population.inp` | 1 | 16 / 21 | 0.7336069011, 0.2663930989 |

Reanalysis and completed resume returned exactly the same population arrays.
The independent interrupted-run tests also compare every checkpoint payload
array and the hopping RNG with uninterrupted propagation. The two examples
have different models and initial electronic bases; their populations are not
an NRPMD-versus-RPSH accuracy comparison.

## Independent quantum reference

```bash
PYTHONPATH=src python benchmarks/nonadiabatic/population_dynamics.py \
  --smoke --seed 1729 --output population_smoke.json
```

Two runs generated identical complete JSON output; the saved JSON matched
standard output. The retained data are
[validation-smoke-2026-11-01.json](validation-smoke-2026-11-01.json).

| Diagnostic | Observed value |
| --- | ---: |
| Independent trajectories / beads / final atomic time | 16 / 8 / 2.0 |
| Final state-1 quantum population | 0.0779513104 |
| Final state-1 NRPMD population | 0.0522260134 |
| NRPMD pointwise trajectory SEM | 0.0250816035 |
| Reference timestep refinement, maximum population difference | 1.2479e-7 |
| Reference doubled-grid sensitivity | 4.0190e-14 |
| Reference doubled-box sensitivity at fixed spacing | 3.8303e-14 |
| Maximum quantum component norm error | 1.21e-13 |
| Maximum outer-edge population | 6.32e-17 |

The reference is separately checked against analytic Rabi oscillations, free
Gaussian propagation and an independently built full Hamiltonian matrix
exponential. The population difference above contains sampling, finite-bead,
numerical and approximate-method contributions. This small ensemble does not
establish NRPMD accuracy or convergence. `physical_converged` remains false.

## Packaging

`python setup.py build_ext --inplace` and `python setup.py bdist_wheel`
completed. Inspection of the CPython 3.12 wheel confirmed all six new Python
modules and both existing native extensions were included. `MANIFEST.in`
includes the new benchmark scripts, documentation and JSON in source archives.
No wheel or platform-specific build product is committed.
