# Tier 2 methods: numerical and integration validation

Date: 2026-11-01. Base: `2f907ba96d22cb134198715e398b1d9f8d83d929` on
`main`. This record concerns the implementations and explicit boundaries in
[the coverage guide](../../tier2-methods.md); it does not certify complete
i-PI/DL_POLY Quantum parity or a converged molecular application.

## Environment and results

Linux x86-64, Python 3.12.14, NumPy 2.5.4, SciPy 1.17.0, h5py 3.16.0,
pytest 9.2.0 and PyTorch 2.15.0+cpu. The C++ extension was compiled in place
and reported OpenMP plus Cayley, exact normal-mode and velocity-Verlet
integrators. `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` were used.

| Check | Result |
| --- | --- |
| Full repository `python -m pytest -q` | **4591 passed, 138 skipped, 357 subtests passed**, 20 warnings, 288.85 s |
| Final 19-file Tier 2 + frontend regression run after cross-review fixes | **464 passed, 1 skipped, 4 subtests passed**, 17 warnings, 12.71 s |
| English and Chinese Sphinx HTML builds, warnings as errors | Passed |
| English and Chinese Sphinx single-page HTML builds, warnings as errors | Passed |
| Source distribution and wheel via the declared setuptools PEP 517 backend | Passed; all 125 runtime Python modules and both native extensions included |
| Fresh native wheel rebuilt from the extracted source distribution | Passed; installed offline into an isolated directory |
| Outside-checkout installed-wheel and console-entrypoint smoke | All 47 new public exports, native/thermal extension loading, CLI info and packaged spectroscopy preflight passed |
| Public Tier 2 exports | Importable without importing Torch or initializing external providers |
| Suzuki–Chin, bosonic and PIGS example prepare/run/resume | Each completed and retained 20 saved samples |
| Committee and delta/RPC input examples | Both completed two replicas, 21 production samples per replica |
| Delta/RPC MTS example | 100 outer steps, 401 fast and 101 slow force evaluations |
| Direct-data QD-NRPMD, RPSH and MASH analytic example | Executed each method successfully |
| Adaptive OPES explore example | Executed and restarted successfully |
| `git diff --check` | Passed |

The two pytest rows overlap and must not be added together. The full run was
followed by the focused run after the last nested-provider validation and
independent review corrections. The raw outputs are
[`pytest-full.txt`](pytest-full.txt) and
[`pytest-tier2-final.txt`](pytest-tier2-final.txt).
Packaging receipts are [`package-report.json`](package-report.json),
[`archive-check.json`](archive-check.json) and [`wheel-smoke.json`](wheel-smoke.json).

The final focused command includes:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_shared_metadynamics.py \
  tests/test_adiabatic_on_the_fly.py tests/test_mash.py \
  tests/test_nonadiabatic_rate_cli_workflow.py \
  tests/test_tier2_sampling_review.py tests/test_advanced_cv.py \
  tests/test_advanced_sampling.py tests/test_opes_variants.py tests/test_opes.py \
  tests/test_pimd_methods.py tests/test_pimd_method_workflows.py \
  tests/test_tier2_spectroscopy.py tests/test_spectroscopy_workflow.py \
  tests/test_plumed_workflow.py tests/test_plumed.py \
  tests/test_ml_lifecycle.py tests/test_trpmd_spectroscopy.py \
  tests/test_input_cli.py tests/test_input_preflight.py
```

Use an editable installation or `PYTHONPATH=src` with the required dependencies
installed. The optional TorchScript tests use an actual CPU Torch runtime and
autograd. Their 17 warnings are TorchScript deprecation notices in that runtime.

## Independent scientific checks

- **Spectroscopy:** fictitious-mass normal-mode motion and thermal covariance,
  harmonic force matching, tensor invariants, causal SFG conventions and known
  spectral frequencies; numerical property preflight and failed-provider cleanup.
- **Suzuki–Chin:** nonlinear potential force differences and fourth-order
  even-bead variance convergence from an independent Gaussian integral.
- **Bosons:** explicit small-system representative-cycle sums versus the
  quadratic recurrence, endpoint connection normalization, force differences,
  stable large spring offsets and no double-counting of ordinary ring springs.
- **PIGS:** independent open-chain action/force checks and midpoint convergence
  to the harmonic ground-state variance.
- **CVs and sampling:** complete-ring finite-difference Jacobians, actual
  TorchScript autograd, periodic seams and alignment singularities; explicit
  variable-width KDE normalization; the full temperature-dependent exchange
  action and kinetic-energy-preserving momentum rescaling.
- **Shared MetaD:** simultaneous hills see the same previous bias; independently
  reconstructed sampling bias; every state and RNG rolls back on a failed
  walker or failed deposition; resumed checkpoints match uninterrupted results.
- **Biased kinetics / RETIS:** known acceleration clocks, censored exponential
  likelihoods and integer-overflow boundaries; valid positive-ensemble path
  exchanges without changing interface measures.
- **Nonadiabatic dynamics:** local adiabatic versus fixed-diabatic MInt evolution
  agrees over a complete trajectory, including independent phase/sign changes;
  energy/gradient/NAC/overlap validation, hop energy conservation and failed-step
  rollback; the rate CLI calls the actual signed thermal sampler and estimator.
- **ML lifecycle:** independent population-variance calculations, returned/cache
  array ownership, nested model-file hashes and strict fields, candidate caps
  and replay identity, complete-resume auditing, exact RPC chain-rule forces
  and clearly identified contracted-Hamiltonian estimators.

## Unavailable external checks and remaining scope

The 138 full-suite skips are for optional ASE, PySCF/tblite, native ONNX/JaxPIP,
source-pinned BKMP2/BrHCl/EAM/Caracal builds, a separate reference solver and
optional external/MPI environments. The sole skip in the focused run is the
real PLUMED Python wrapper/kernel test. The actual bridge command protocol,
units, update ordering and frontend failure handling ran against a controlled
mock kernel. A real PLUMED installation and a real quantum-chemistry package
were not newly validated here.

QCMD/h-CMD, a native OPES-flooding exclusion bias, the RETIS minus ensemble,
portable QC-wavefunction restart and a thermal on-the-fly QD rate workflow
remain unimplemented. Quadratic f-CMD, reference adaptive bandwidths, serial
exchange APIs and experimental endpoint MASH retain their documented limits.
Active-learning collection saves unlabeled candidates and predictions; it
does not perform reference labeling or automatic retraining. Passing these
tests does not establish physical convergence of spectra, free energies or
chemical rates.
