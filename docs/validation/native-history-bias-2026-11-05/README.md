# NP4 native history-bias validation

This record covers NP4 (`native history bias`) on branch `agent/np4-history-bias`,
based on main commit `981f74ecbb2b3b951f1445fa88e201da0ae61783`. The coordinated
scope is the one recorded in `claude/awesome-heisenberg-vas3wy` at commit
`365d9c6`: frozen wall/hills/grid descriptors, one snapshot check after native
`advance`, and tagged C++ support in `cv_core.hpp`.

## Build and native API marker

The editable install completed with:

```text
python -m pip install -e . --no-build-isolation
```

The OpenMP-enabled build reported:

```text
_native.has_openmp() == True
_native.NATIVE_HISTORY_BIAS_API == 1
```

The OpenMP-disabled build was also compiled with `PYRPMD_NO_OPENMP=1` and
reported:

```text
_native.has_openmp() == False
_native.NATIVE_HISTORY_BIAS_API == 1
```

## NP4 and native regression tests

With OpenMP enabled, the requested NP4 filter completed as:

```text
OMP_NUM_THREADS=1 PYTHONPATH=/workspace/scratch/08c77586af73/deps:src \
  python -m pytest -q tests -k "metadynamics or wall or bias or native"
1988 passed, 73 skipped, 9555 deselected, 33 warnings, 57 subtests passed in 154.73s (0:02:34)
```

The focused history-bias and native propagation tests completed in both build
modes. The OpenMP-enabled run also included the native compatibility and grid
review suites:

```text
OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_native_history_bias.py tests/test_native_propagation.py \
  tests/test_native_block_compatibility.py tests/test_grid_bias_review.py
68 passed in 6.57s

PYRPMD_NO_OPENMP=1 python -m pytest -q \
  tests/test_native_history_bias.py tests/test_native_propagation.py
26 passed in 4.18s
```

The tests cover wall, standard and well-tempered periodic hills, tensor
Hermite grids, composite flattening, descriptor detachment, stale snapshot
rejection after native return, missing-API fallback/strict failure, and Python
subclass fallback. Direct parity checks used the same frozen descriptors as
the Python implementations; the observed energy/force differences were within
the NP4 `1e-12` numerical-equivalence target.

## Full non-campaign gate

The required full gate was run as:

```text
PYTHONPATH=/workspace/scratch/08c77586af73/deps:src \
  python -m pytest -q -m "not campaign"
9412 passed, 231 skipped, 1 deselected, 11 failed, 47 warnings, 410 subtests passed in 1925.71s (0:32:05)
```

The 11 failures are environmental rather than NP4 failures:

* Ten `tests/test_campaign_marker.py` cases launch nested subprocesses with a
  temporary-only `PYTHONPATH`; those subprocesses could not import the runtime's
  pytest installation. Running that test module after exposing the runtime
  userbase completed with `13 passed`.
* `tests/test_ipi_clients.py::test_existing_unix_address_is_not_removed_on_bind_failure`
  requires an AF_UNIX bind. The managed sandbox rejects that operation with
  `PermissionError: [Errno 1] Operation not permitted`.

All other tests in that gate passed. No code or test was weakened to bypass
either environment restriction.

## Documentation gate

The strict Sphinx build completed after installing the repository's documented
Sphinx extras:

```text
PYTHONUSERBASE=.../userbase PYTHONPATH=src \
  python -m sphinx -W --keep-going -b html docs docs/_build/html
build succeeded. The HTML pages are in docs/_build/html.
```

No performance or GPU claim is made here; this record is limited to the
executed build, parity, regression, and documentation checks.
