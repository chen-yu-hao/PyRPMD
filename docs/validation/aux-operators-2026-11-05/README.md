# T26 auxiliary-operator validation — 2026-11-05

## Scope

This record covers the Python auxiliary-degree-of-freedom protocol and the NRPMD adapter. It does not claim physical convergence, production accuracy, GPU performance, NPT support, or validation of other nonadiabatic methods.

## Environment

- Host: Linux x86_64 container, kernel 6.18.57
- CPU presented to the container: 112 logical CPUs
- Python: 3.11.5
- NumPy/SciPy/pytest: 1.24.3 / 1.15.3 / 7.4.0
- The host NumPy is below the package declaration (`numpy>=1.26`); no dependency
  upgrade was possible through the available setuptools 68 installation path.
- Validation thread settings: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- Baseline commit: `801953dc9e2ca620f36069eb4551c41c529c432a`

## Independent numerical checks

`tests/test_auxiliary_operators.py` contains the following references:

1. **Legacy map reference:** 10,000 NRPMD steps are propagated twice from identical arrays. One path uses the new auxiliary core; the other spells out the former `_mint(dt/2)`, free-ring, `_mint(dt/2)` loop. Nuclear coordinates, momenta, both mapping arrays, clock, and final total energy are required to be bitwise equal.
2. **Periodic force reference:** a periodic analytic two-state Hamiltonian is compared with a centered finite difference of the auxiliary energy and with a full lattice translation.
3. **RPC/MTS reference:** an analytic harmonic split potential with one-bead contraction is propagated for 3,000 steps and checked for bounded total-energy error.
4. **Preparation reference:** Andersen momenta are compared with the analytic variance \(mP/\beta\); Andersen and PILE-L both verify bitwise-frozen mapping arrays.
5. **Persistence reference:** version-2 operator state restores RNG and mapping arrays; a legacy RNG-only state remains readable.
6. **Fail-fast reference:** NPT is rejected before PES calls, provider cell updates, or output-directory creation.

## Commands and observed results

The source checkout was tested with `PYTHONPATH=src` because this host's
setuptools 68 rejects the repository's PEP 621 license string before the
editable install can start. The focused T26 checks passed:

```bash
PYTHONPATH=src python -m pytest -q tests/test_auxiliary_operators.py tests/test_auxiliary_workflow_checkpoint.py
# 12 passed in 25.93s

PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_nonadiabatic*.py tests/test_nrpmd*.py tests/test_complex_nonadiabatic.py \
  tests/test_thermal_nrpmd*.py tests/test_electrochemistry*.py \
  tests/test_constant_potential_nonadiabatic.py tests/test_adaptive_nrpmd.py \
  tests/test_adaptive_constant_potential_example.py tests/test_adaptive_nrpmd_reference.py \
  tests/test_adaptive_nrpmd_study.py tests/test_published_nrpmd_benchmark.py
# 918 passed, 5 skipped, 2 failed in 124.73s (0:02:04)

PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q -m "not campaign"
# 48 failed, 8318 passed, 879 skipped, 1 deselected, 48 warnings, 35 errors in 1768.93s (0:29:28)
```

The two focused-suite failures are the pre-existing serial/process bitwise
comparison for `rpsh` and a dRPMD bootstrap covariance comparison at sub-
`1e-18` scale; the `rpsh` difference was below `8e-16`. The two Andersen
batch cases that fail without the optional native extension are likewise
baseline environment failures. The required full-suite
command ran against this source checkout but not the repository's certified
install: setuptools 68 rejected the PEP 621 license field, the native
extensions were unavailable, the installed PySCF module had no `__version__`,
and several optional socket/fixture tests were host-dependent. No failure was
reported from the T26 auxiliary tests or the T26 focused numerical checks. No
GPU, native-kernel performance, or physical-convergence claim is made here.

## Interpretation

Passing these tests establishes implementation-level equivalence and the stated analytic checks. It does not establish convergence in bead count, timestep, trajectory count, electronic-state truncation, or model fidelity. Any result produced by these paths must retain conservative `physical_converged=false` semantics unless a separate convergence study supports a stronger claim.
