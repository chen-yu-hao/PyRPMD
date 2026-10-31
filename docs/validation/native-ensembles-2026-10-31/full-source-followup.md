---
orphan: true
---
# Full-source follow-up and portability fix

The source distribution from PR #32's package workflow was downloaded and built
locally using the real repository headers and Python modules, rather than the
initial source-subset harness. Two fixes/tests were then applied:

- Use native POSIX byte-string filenames on Linux/macOS rather than the C++17
  filesystem path overload. The latter needs macOS 10.15 whereas the package
  workflow builds for 10.13. Windows retains the UTF-8-to-wide filesystem path.
- In stochastic velocity rescaling, the exactly-zero initial kinetic-energy
  transition now draws momenta with variance `m*theta*(1-exp(-dt/tau))`, not
  full equilibrium variance. A 60,000-degree-of-freedom test checks this limit.

Rebuilt with `python -m pip install -e . --no-build-isolation --no-deps`.

```text
python -m pytest -q tests/test_native_ensembles.py \
  tests/test_native_ensemble_portability.py --disable-warnings
124 passed in 19.37s
```

The eight comparisons with existing PyRPMD integrators now run and pass. This
replaces the earlier 114-pass/8-skip limitation for the new feature's test suite.
The raw output is retained as `pytest-full-source-targeted.txt`.

For the first PR commit `5e4f4e6c0fc7eedb1295333f8ccd7360dfe4c2c3`, GitHub
Tests run `41188586794` completed successfully for Python 3.11 and 3.12, including
the complete pytest suite, four-rank MPI integration/workflow/materials/surface
checks, package build, and a separate Torch CPU job. Package run `41188586785`
identified the macOS filesystem deployment-target failure fixed here. These
recorded CI results belong to that first commit; the follow-up commit must run
its own checks, rather than inheriting a pass claim.

A local attempt to run *all* tests from the downloaded sdist stopped at collection
because the existing manifest omits `benchmarks/hho2/channels.py` and `scan.py`.
That is distinct from the complete-repository CI result and does not affect the
124 targeted tests. No claim of a complete local pytest run is made.

The scientific and workflow limits in `docs/native-ensembles.md` remain: this
is an opt-in native driver, no Suzuki-Chin NPT or calibrated PIGLET dataset is
included, and real-material production validation is still required.
