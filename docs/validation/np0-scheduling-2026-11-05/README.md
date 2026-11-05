# NP0 scheduling validation (2026-11-05)

This record covers the E0 scheduling changes for RPMD recrossing children and
the legacy materials Andersen path. The reference path disables fusion through
`Dynamics.can_fuse()`; the candidate path uses the native block entrypoint.
Both runs use the same NumPy generator seed and the same deterministic harmonic
C ABI v1 PES.

## Environment

- Source branch: `agent/np0-scheduling`, based on `main@981f74e`.
- Python 3.12, NumPy 2.5.4, SciPy 1.17.0, PyRPMD native extension built with
  `python setup.py build_ext --inplace` and OpenMP enabled.
- Linux 6.18.57, x86_64, 9 visible CPUs. The executor does not expose a CPU
  model through `/proc/cpuinfo`; no hardware model is inferred.
- The benchmark PES is the repository's
  `native/examples/harmonic_pes.c`, compiled locally as a C ABI v1 shared
  library with an evaluation counter. BKMP2 was not built or measured.

## 10,000-step Andersen PIMD microbenchmark

Command: compile the harmonic C ABI example, then run
`_sample_replica()` twice with `PYTHONPATH=src .venv/bin/python`; the reference
run uses `Dynamics.can_fuse = False`, while the candidate run uses the checked
out implementation. Configuration was `P=4`, `block_steps=256`,
`andersen_interval=10`, `sample_interval=1000`, `checkpoint_steps=10000`, and
`production_steps=10000`. The harness counted C ABI evaluations and wrapped
`pyrpmd._native.advance` to count native block calls.

| path | wall time (s) | native block calls | block sizes | C ABI PES evaluations | native steps | Python steps |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| reference one-step | 1.266496 | 0 (`run_block`: 10,000 one-step calls) | 1 | 20,011 | 10,000 | 0 |
| NP0 fused | 0.257904 | 1,000 | 10 (first/last three all `10`) | 11,011 | 10,000 | 0 |

The candidate output arrays, state `q`/`p`, clock, checkpoint data and final RNG
state were compared with `numpy.testing.assert_array_equal`; all comparisons
passed. The measured PES count is for this harmonic C ABI harness and is not a
claim about BKMP2 performance.

## Regression commands

```bash
python setup.py build_ext --inplace
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_np0_scheduling.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_materials_native_blocks.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_fused_propagation.py tests/test_native_propagation.py tests/test_batched_recrossing.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests -k "recross or materials"
PYTHONPATH=src .venv/bin/python -m pytest -q -m "not campaign"
```

The NP0-specific regression summary was `5 passed`; the updated materials
native-block suite was `12 passed`; the rate and batching subset was `37 passed`
before the full-suite run. The full `-m "not campaign"` run reached
`9424 passed, 228 skipped, 1 failed`; the only failure was the unrelated
`tests/test_ipi_clients.py::test_existing_unix_address_is_not_removed_on_bind_failure`
case, where this sandbox rejects `AF_UNIX` socket creation with `EPERM`. No
BKMP2 build was available in this executor.
