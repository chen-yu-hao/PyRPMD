# PySCF / Caracal PES integration validation — 2026-10-31

## Tested sources and scope

Base PyRPMD: `76dd6ad24adc6c244f775857987cd2f0e119845f`.
Caracal: `7f95ce9818b7f6d81d7cb3b634f360373c090820`, with the archived charge-input
and workspace-allocation safety patches produced by `build_native`.
Real PySCF 2.14.0, Python 3.13, compiled PyRPMD C++ extension, native Linux
Caracal binaries. Single BLAS/OpenMP thread.

The complete local suite passed **1,572 tests and 153 subtests**, with 27 optional
skips and no failures. This snapshot preceded addition of one extra MPI-only
end-to-end EVB test; it does not claim that test ran locally. The original EVB
optimizer test was skipped locally because the complete MPI runtime installation
was unavailable. The dedicated GitHub workflow enables both with
`PYRPMD_PESGEN_FULL=1`; its result must be read separately, not inferred from
these local numbers. Local Sphinx was unavailable. A native wheel built and
contained all three required packaged C/C++/Fortran interface resources.

## Real numerical checks

- Actual water and H2 minimum optimization, PySCF Hessian/Hirshfeld/Mayer data,
  original QMDFF generation, native batch evaluation and completed-run resume.
  Water endpoint energy error is below 1e-9 hartree and force RMSE below 1e-5
  hartree/bohr; these are local fitting checks, not held-out transferability.
- RHF analytic forces/Hessians versus independent finite differences;
  UHF/RKS/UKS forces, charge integration, SCF failure rejection, checksum corruption.
- Pinned original dE-EVB and CF3O+CH4 TREQ examples evaluated through two isolated
  contexts. Reversed batch order gives identical energies and forces; explicit
  energy differences check forces.
- Thirty continuous C++ dynamics steps on the real water QMDFF invoke
  `CABIPotential.evaluate` **zero times**, verified using Python profiling.
  This is a dispatch check, not a wall-clock speedup claim.
- Actual nine-frame PySCF NH3 inversion scan generates two QMDFFs and TREQ.
  A local pilot had training energy RMSE 0.00172447682922528 hartree and force
  RMSE 0.0037219376474005674 hartree/bohr. The supplied constrained scan is not
  an optimized IRC, has no held-out test, and produces no validated reaction rate.

## TREQ qualification

The pinned CF3O+CH4 model's original analytical force at one tested component was
0.004272571219902494 hartree/bohr, while its energy difference was approximately
0.00230114949. The explicit fourth-order native numerical force was approximately
0.00230063857 with a 1e-4 bohr step. This is why public TREQ generation selects
numerical energy derivatives and marks the model experimental rather than silently
accepting the upstream analytical gradient. Cost is 1+12N upstream evaluations per
structure. Step-size, timestep, energy conservation and physical PES accuracy still
need system-specific checks.

## Reproduce

See `docs/manual/pes-generation.md` for dependencies and native build commands.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYRPMD_PESGEN_BUILD="$PWD/runs/caracal-native"
export PYRPMD_CARACAL_SOURCE="$PWD/external/Caracal"
export PYRPMD_PESGEN_FULL=1
python -m pytest -q tests/test_pesgen.py tests/test_pesgen_native.py
python -m pytest -q
```

The CI artifact retains build commands, hashes, source patch, reference JSON,
fit logs, summaries and JUnit results. Bundled numerical checks establish software
contracts, not a universal QMDFF/EVB/TREQ accuracy guarantee or converged RPMD rate.
