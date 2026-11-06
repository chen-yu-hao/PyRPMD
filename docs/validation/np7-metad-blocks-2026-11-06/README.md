# NP7 serial MetaD blocks (2026-11-06)

This record covers the NP7 implementation on agent/np7-metad-blocks from
main@bebaed447c80e0adcb13224977af73a52bd1ce8f. The validation uses the
built-in one-atom harmonic potential and a Cartesian centroid CV. The native
extension was built by python -m pip install -e . --no-build-isolation.

## Reproduction

    PYTHONPATH=.pytest_vendor:src python -m pytest -q tests/test_native_history_bias_blocks.py
    PYTHONPATH=.pytest_vendor:src python -m pytest -q tests/test_metadynamics.py tests/test_grid_metadynamics.py tests/test_enhanced_sampling_composition.py tests/test_native_history_bias.py

The observed summaries were 4 passed in 3.84s and 128 passed, 17 warnings in 9.59s.

The focused tests compare Andersen, grid MetaD and PILE-L block execution with the scalar history path. For the 27-step run, the block path reported native_steps=27, python_steps=0, and an empty fallback_reasons mapping. The final q/p arrays are compared at 1e-11 relative tolerance; samples, sample clocks, sample biases, energies, hills and the NumPy RNG state matched. The run checkpoint contains native_path_contract=1. Removing that field and reading the checkpoint returns the legacy contract (0), so an old checkpoint does not silently switch scheduling on its next save.

No H3/BKMP2 benchmark was run in this environment. These results establish implementation parity for the built-in toy system only; they do not establish physical convergence or production performance for a molecular PES.

## Environment

The checkout is bebaed447c80e0adcb13224977af73a52bd1ce8f plus the NP7 working tree. The runtime is Python 3.12 from the Codex primary runtime, NumPy 2.3.5, and PyRPMD 0.2.0 editable. The test host is a Linux x86_64 managed container; no GPU or BKMP2 library was used. The native extension was built with OpenMP enabled for the focused run.

The card-wide filter was also run after installing the test-only dependencies:

    PYTHONPATH=.pytest_vendor:src python -m pytest -q tests -k 'metadynamics or metad or grid or biased or opes'

It produced 744 passed, 39 skipped, 10859 deselected, 35 warnings, 19 subtests passed in 86.37s.

The required full non-campaign gate was run with OpenMP enabled:

    PYTHONPATH=.pytest_vendor:src OMP_NUM_THREADS=1 python -m pytest -q -m 'not campaign'

It produced 9441 passed, 228 skipped, 1 deselected, 11 failed, 47 warnings, 410 subtests passed in 1278.43s (0:21:18). The 11 failures are environment-only: ten campaign-marker nested subprocesses cannot see the temporary pytest vendor path, and the existing AF_UNIX bind test is denied by the managed sandbox (PermissionError: [Errno 1] Operation not permitted). No NP7-focused test failed.
