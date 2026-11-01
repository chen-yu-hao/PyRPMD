# Executed NaH₂ molecular checks

Local execution date: 2026-11-01. Environment: Python 3.12.14, NumPy 2.3.5,
PyRPMD 0.2.0; native ring kernels disabled for the recorded runs.

## Local tests and actual trajectories

```bash
PYTHONPATH=/workspace/scratch/3e6096a7ca4a/test-deps:src \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_multistate_external.py -q

PYTHONPATH=src python benchmarks/molecular_nonadiabatic/run_molecular.py \
  --method both --steps 100 --trajectories 8 --beads 4 \
  --output benchmarks/molecular_nonadiabatic/results
```

The targeted suite passed **38 tests**. It covers complete matrix derivatives,
all atom permutations, rotation/translation covariance, force and torque sums,
asymptotic energies, immutable provenance, original source and license hashes,
optional-adapter units and invalid input, and actual nuclear/electronic steps.

Both full smoke examples completed. Each used 8 trajectories, 4 beads, 100
steps at dt=0.2 atomic time, a total of approximately 0.484 fs. The explicit Na/H
masses, geometry, momenta, initial matrix and electronic preparation are retained
in the result JSON files.

| Diagnostic | NRPMD | RPSH |
| --- | --- | --- |
| Completed trajectories | 8/8 | 8/8 |
| Maximum saved-frame sum-bead energy drift, hartree | 5.8826599592e-8 | 4.7163919881e-8 |
| Initial-geometry matrix-gradient finite-difference error, hartree/bohr | 4.9228399135e-11 | 4.9228399135e-11 |
| Physical convergence established | No | No |

The NRPMD signed population mean for state 0 was approximately -0.0159 with
pointwise SEM 0.0368; it was retained without clipping. This small demonstration
ensemble does not establish a transition probability. The RPSH input starts in
a different, centroid-adiabatic electronic state, so its population is not a
matched-preparation method comparison.

## Actual CLI preparation, propagation, analysis and restart

Both checked-in `.inp` examples also passed `validate --check-pes`, `run`,
`analyze --recompute` and `resume` locally with 8 trajectories and 100 steps.
The saved CLI stream layout differs from the direct Python demonstration, so
the small NRPMD ensemble has a different final sample mean. This is not a
cross-method accuracy test. Analysis and resume of completed trajectories left
all checkpoint byte hashes unchanged.

The first CLI run exposed a provenance issue: the generic `System` default had
recorded spin multiplicity 1. The NaH2 provider and resolver now record neutral
doublet provenance, charge 0 and multiplicity 2. This metadata correction does
not change the PES or trajectories. A fresh short cycle for **each method**
used 2 trajectories and 3 steps, with sample stride 1, in new output directories:

```text
phase2-molecular-cli-doublet-nrpmd
phase2-molecular-cli-doublet-rpsh
```

Both cycles passed all four CLI operations and explicitly checked the persisted
`System` charge and multiplicity. Their complete commands, input hashes,
physics fingerprints and checkpoint hashes are in
`results/cli_doublet_validation.json`. The earlier partial command record is
retained as `results/cli_validation.json`; it predates the provenance correction.
These checks establish executable workflow and completed-restart behavior,
without extending the physical claims of the trajectory smoke examples.

## Independent original-source reference

`compare_fortran.py` authenticates and compiles the pinned original kernel,
compares 64 reproducible geometries and checks the port's gradients against
five-point differences of the original energy. Its JSON result contains the
compiler, exact source identity, thresholds and measured errors.

The local environment used above has no gfortran. The independent comparison
was subsequently **executed and passed in GitHub Actions** on 2026-11-01:
[molecular-dpem job 124801235314](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41290725504/job/124801235314).
It used GNU Fortran 15.2.0 on Ubuntu 26.04.1, with 64 deterministic molecular
geometries and a Cartesian finite-difference step of `2e-4` bohr.

| Quantity | Maximum absolute difference | Acceptance threshold |
| --- | ---: | ---: |
| Complete diabatic matrix, hartree | 1.510944147575799e-15 | 3e-12 |
| Adiabatic eigenvalues, hartree | 1.4432899320127035e-15 | 3e-12 |
| Cartesian matrix gradients, hartree/bohr | 3.4329694035412084e-12 | 3e-8 |

The original JSON emitted by the comparison, supplemented with exact CI
provenance, is retained in `results/original_fortran_ci.json`. The
[uploaded artifact](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41290725504/artifacts/13076588517)
also contains the CI result. The tested PR head was
`c53fd1dff83917d4b3137abaebf03135880d9b68`; Actions checked out its merge commit
`a04d478f0b52d938fafe45d427d4906703bc12c6` against the base branch. The
[comparison to head 4cc1049](https://github.com/chen-yu-hao/PyRPMD/compare/c53fd1dff83917d4b3137abaebf03135880d9b68...4cc1049109a9b05a8fae0139b7eb67306040f81e)
changes only workflow concurrency and API documentation, leaving the physical
implementation and original-source comparison unchanged. This records the
commit actually tested rather than attributing the completed job to a later
commit. The same molecular job also completed the NaH2 NRPMD and RPSH CLI runs.

These checks distinguish source equivalence, numerical integration and physical
model accuracy. None certifies a converged chemical prediction.
