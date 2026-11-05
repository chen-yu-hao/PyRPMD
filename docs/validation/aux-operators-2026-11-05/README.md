# T26 auxiliary-operator validation — 2026-11-05

## Scope

This record covers the Python auxiliary-degree-of-freedom protocol and the NRPMD adapter. It does not claim physical convergence, production accuracy, GPU performance, NPT support, or validation of other nonadiabatic methods.

## Review fixes

PR #114 identified an [import-order failure](https://github.com/chen-yu-hao/PyRPMD/pull/114#discussion_r4477155348)
and [two unused copies of modules](https://github.com/chen-yu-hao/PyRPMD/pull/114#issuecomment-6653360228). The adapter now imports its three dependencies from `nonadiabatic`
inside the methods that use them, so module initialization works in either
order. The definitions and historical exports stay in `nonadiabatic.py`.
The unused `src/pyrpmd/_nonadiabatic_legacy.py` and
`src/pyrpmd/workflows/_nonadiabatic_legacy.py` were removed after auditing
all Python source, tests, examples, and benchmarks.

`review-import-audit.txt` records the exact grep commands and results before
removal, plus the remaining historical helper importers. The original audit
is retained in `import-audit.txt`.

The new regression runs both import orders in separate Python interpreters,
then constructs the public NRPMD adapter and propagates one step. Before the
fix it reported `1 failed, 1 passed, 11 deselected in 4.32s`; after the fix,
`2 passed, 11 deselected in 4.13s`. This initial reproduction used the existing
T52 dependency environment with `PYTHONPATH=src`. The acceptance run below
uses a fresh editable installation and compiled native extensions.

## Environment

- Host: `cluster`, Linux `5.14.0-611.49.1.el9_7.x86_64`, x86_64.
- CPU: Intel Xeon Platinum 8176 @ 2.10 GHz; 2 sockets, 56 physical cores,
  112 logical CPUs visible.
- Python: 3.11.5; isolated environment `/tmp/pyrpmd-t26-review-venv`.
- Dependencies: NumPy 2.4.6, SciPy 1.17.1, h5py 3.16.0, threadpoolctl 3.7.0,
  pytest 9.2.0, setuptools 84.2.1, Sphinx 8.2.3, myst-parser 4.0.1.
- Compiler: GCC/G++ 11.3.1, with the default OpenMP build. `_native`,
  `_thermal`, and `_electronic` all import from this checkout;
  `_native.has_openmp()` is `True`. `python -m pip check` reports
  `No broken requirements found.`
- Thread settings: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`.
- Branch rebased onto `main@67415fd`, including T52 (#112).

## Independent numerical checks

`tests/test_auxiliary_operators.py` contains the following references:

1. **Legacy map reference:** 10,000 NRPMD steps are propagated twice from identical arrays. One path uses the new auxiliary core; the other spells out the former `_mint(dt/2)`, free-ring, `_mint(dt/2)` loop. Nuclear coordinates, momenta, both mapping arrays, clock, and final total energy are required to be bitwise equal.
2. **Periodic force reference:** a periodic analytic two-state Hamiltonian is compared with a centered finite difference of the auxiliary energy and with a full lattice translation.
3. **RPC/MTS reference:** an analytic harmonic split potential with one-bead contraction is propagated for 3,000 steps and checked for bounded total-energy error.
4. **Preparation reference:** Andersen momenta are compared with the analytic variance \(mP/\beta\); Andersen and PILE-L both verify bitwise-frozen mapping arrays.
5. **Persistence reference:** version-2 operator state restores RNG and mapping arrays; a legacy RNG-only state remains readable.
6. **Fail-fast reference:** NPT is rejected before PES calls, provider cell updates, or output-directory creation.

## Commands and observed results

Installation used a fresh venv, upgraded setuptools and wheel, and the
project's declared `test` and `docs` extras. Public dependency wheels were
obtained through pip, direct IPv4 download, and the existing pip cache after
slow downloads; the final editable install used the local wheel directory:

```bash
python -m venv /tmp/pyrpmd-t26-review-venv
source /tmp/pyrpmd-t26-review-venv/bin/activate
python -m pip install 'setuptools>=77' wheel
python -m pip install --no-index --find-links /tmp/t26-offline-wheels \
  --no-build-isolation -e '.[test,docs]'
python -m pip check
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q -m "not campaign"
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m pyrpmd run examples/nonadiabatic/exchange.inp \
  --output /tmp/t26-review-exchange-20261105
```

A fresh environment with network access can use
`python -m pip install -e '.[test,docs]'` instead of the local wheel directory.
The full suite runs outside the filesystem sandbox so local socket and
multiprocess tests can exercise their real paths. No `PYTHONPATH` override
is used for acceptance.

Raw full non-campaign summary (exit status 0):

```text
9377 passed, 235 skipped, 1 deselected, 47 warnings, 410 subtests passed in 1712.70s (0:28:32)
```

The complete raw output is retained in `pytest-review.txt`. The suite has
no failures or collection errors; optional dependencies account for skips.

The strict Sphinx command exited 0 with `build succeeded.`. The real CLI
command exited 0 with 8/8 trajectories resolved: 8 nonreactive, 0 product,
0 unresolved, 0 failed. This pilot checks the software workflow, not a
converged physical rate.

## Superseded environment diagnostic

The earlier source-path run used NumPy 1.24.3 and pytest 7.4.0, below the
project requirements, and had no compiled native extensions. Its raw summary
was:

```text
48 failed, 8318 passed, 879 skipped, 1 deselected, 48 warnings, 35 errors in 1768.93s (0:29:28)
```

This was not an acceptable full-suite result. No complete same-environment
main baseline was run, so the earlier attribution of all failures to baseline
or optional-dependency issues was unsupported and is withdrawn. The failed
setuptools 68 build also did not establish that a compatible environment
could not be installed. The fresh installation above replaces that diagnostic
as the basis for review acceptance.

## Interpretation

Passing these tests establishes implementation-level equivalence and the stated analytic checks. It does not establish convergence in bead count, timestep, trajectory count, electronic-state truncation, or model fidelity. Any result produced by these paths must retain conservative `physical_converged=false` semantics unless a separate convergence study supports a stronger claim.
