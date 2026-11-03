# T40 excited-state interface validation — 2026-11-03

## Scope and provenance

- Repository: `chen-yu-hao/PyRPMD`
- Baseline inspected before implementation: `main@ceee59cd5f702165b56c5e541a503d8810fb9111`
- Branch: `agent/t40-excited-state-interface`
- Thread controls for all reported tests and benchmarks:
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`
- This record validates software contracts and small reference systems. It does
  **not** establish production electronic-structure accuracy or physical
  convergence.

## Environment

```text
Linux ecdb61e85fe3 6.18.57 #1 SMP Tue Oct 20 21:56:04 UTC 2026 x86_64 GNU/Linux
CPU allocation: 5 logical CPUs
CPU model: AMD EPYC 9V74 80-Core Processor
Python 3.13.5 (GCC 14.2.0)
NumPy 2.3.5
SciPy 1.17.0
pytest 9.0.2
```

## Focused dependency-free tests

Command:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q \
  tests/test_excited_state_provider.py \
  tests/test_excited_state_backends.py \
  tests/test_excited_state_external_fixtures.py
```

Raw summary:

```text
......................................                                   [100%]
38 passed in 3.59s
```

The same focused set plus the public-API RPSH test produced:

```text
.......................................                                  [100%]
39 passed in 3.39s
```

Coverage includes:

- independent two-state rotation references for HST and NPI connections;
- exact line-integral verification of the minimum-norm path NAC;
- maximum-overlap assignment, phase alignment, near-degeneracy diagnostics,
  transition-property phase transformation, and retained-subspace rejection;
- centered finite differences against analytic toy energies, gradients, and NAC;
- one scratch directory per bead, cleanup policies, status mapping, identity,
  and checksummed context round trips;
- recorded normal, SCF-failure, and state-shortage outputs for ORCA, Q-Chem,
  and OpenMolcas, including noncontiguous-root rejection;
- mock subprocess invocation with `shell=False`, previous-wavefunction
  restoration, overlap-helper execution, environment redaction, exact staging
  of hashed basis/ECP snapshots, and external finite-difference initial NAC.

Bytecode compilation also completed successfully:

```text
compileall: success
```

## Public-API RPSH test

Command:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q tests/test_excited_state_rpsh.py -vv
```

Raw summary:

```text
============================= test session starts ==============================
collected 1 item

tests/test_excited_state_rpsh.py::test_excited_state_provider_runs_short_rpsh_with_bounded_numerical_drift PASSED [100%]

============================== 1 passed in 0.36s ===============================
```

The test runs eight steps through `ExcitedStateProvider` ->
`AdiabaticPythonPotential` -> `AdiabaticRPSHDynamics`, checks normalized
populations, zero provider failures, and an absolute total-energy drift below
`2e-10` hartree for the analytic model.

## Executable example

Command:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python examples/excited_state/toy_rpsh.py --steps 20 --dt 0.02
```

Raw output:

```json
{
  "active_surface": 0,
  "completed": true,
  "energy_drift_hartree": -2.3522850334245504e-15,
  "final_energy_hartree": -0.010457269322791451,
  "initial_energy_hartree": -0.0104572693227891,
  "model": "analytic two-state avoided crossing",
  "physical_converged": false,
  "populations": [
    0.999999964858533,
    3.514145186949212e-08
  ],
  "provider_statistics": {
    "backend_seconds": 0.0025741899994500272,
    "call_count": 21,
    "failure_count": 0,
    "frame_count": 21,
    "status_counts": {
      "SUCCESS": 21
    },
    "wall_seconds": 0.012952662000316195
  },
  "steps": 20,
  "time": 0.4000000000000001,
  "units": "atomic"
}
```

## PySCF reference tests

The optional test module contains:

- H2/STO-3G RHF/TDA analytic gradients versus centered finite differences;
- the state-aligned overlap derivative versus the backend's full Cartesian
  finite-difference initial NAC;
- phase and transition-dipole continuity over a three-frame LiH trajectory;
- a two-step H2 `AdiabaticRPSHDynamics` trajectory using the real PySCF backend.

Local command and raw summary:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q tests/test_excited_state_pyscf.py
```

```text
sss                                                                      [100%]
3 skipped in 0.35s
```

PySCF was not available in the local container and the environment had no
package-index network path, so these cases were skipped locally rather than
reported as passed. The existing `electronic-structure` CI job installs the
`pyscf` optional dependency; this change adds the module to that job.

## Microbenchmark

Workload: 2,000 sequential one-bead frames of the analytic avoided-crossing
backend, NPI tracking, and a newly created and removed scratch call directory
per frame. Five repeats, one OpenMP thread and one OpenBLAS thread.

The timed loop excludes construction and the first electronic frame. It moves
one Cartesian coordinate by `1e-5` bohr per frame and passes the explicit prior
geometry/context on every call.

Raw measurements:

```text
repeat=1 frames=2000 elapsed_seconds=1.318373889 frames_per_second=1517.020 failures=0
repeat=2 frames=2000 elapsed_seconds=1.378158347 frames_per_second=1451.212 failures=0
repeat=3 frames=2000 elapsed_seconds=1.432922887 frames_per_second=1395.749 failures=0
repeat=4 frames=2000 elapsed_seconds=1.579774350 frames_per_second=1266.004 failures=0
repeat=5 frames=2000 elapsed_seconds=1.414805717 frames_per_second=1413.622 failures=0
median_elapsed_seconds=1.414805717
median_frames_per_second=1413.622
min_frames_per_second=1266.004
max_frames_per_second=1517.020
```

This is framework/scratch overhead on a toy backend, not a prediction of
throughput for PySCF or an external quantum-chemistry program.

## Full-suite and documentation status

Repository-wide commands required by the project are:

```bash
python -m pip install -e '.[test]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The development container contains a focused reconstructed workspace rather
than a complete checkout, so no local repository-wide or Sphinx result is
claimed. The pull-request test and documentation workflows are the authoritative
full-checkout results and must be read separately from this local record.

## External-program boundary

ORCA, OpenMolcas, and Q-Chem executables were not installed or run. Their
adapters are validated with recorded outputs and mock subprocesses only. Each
backend is therefore documented as **executable untested**. A production wrapper
must emit the documented all-root gradient/property markers, preserve atomic
units, and validate its own wavefunction/overlap tool against an independent
reference.

## Known limitations

- The overlap-derived Cartesian NAC is the minimum-norm vector along the actual
  displacement; perpendicular components are not inferable from two frames.
- Full TDDFT uses an auxiliary `X+Y` determinant overlap, not an analytic TDDFT
  derivative coupling.
- The PySCF backend is restricted to closed-shell RHF/RKS references and PySCF
  2.x; no open-shell, SOC, or QM/MM support is claimed.
- External default templates are starting points and do not by themselves
  guarantee all-root gradients from an unmodified program executable.
- Short toy/PySCF trajectories test integration and numerical drift only.
  `physical_converged` remains false.
