# JaxPIP GPU T4 validation

Date: 2026-11-14
Hardware: Google Colab Tesla T4, 15 GiB, driver 580.82.07
Runtime: Python 3.13.16, PyTorch 2.14.1+cu130, JAX 0.11.1, CUDA backend

The test used the current `a18c8cca1e795202194d4082919ebd71722df125` JaxPIP
commit and the current working tree. The JaxPIP models were real fitted-model
objects constructed from the JaxPIP linear and neural-network classes; the
rate stage used a real three-atom JaxPIP model with a zero coefficient vector
to isolate execution and workflow mechanics.

## Results

- JaxPIP linear and NN batch force evaluation on `cuda:0`: passed.
- JAX/Torch coordinate DLPack pointer roundtrip: passed (`data_ptr` identical).
- Energy and force comparison between the direct JAX provider and Torch bridge:
  `0.0` maximum absolute error for both models.
- GPU `TorchDynamics` with umbrella bias and 3 steps: finite state and energy,
  passed for both models.
- Fixed umbrella sampling with GPU q/p state: 2 saved samples, passed.
- Flux-metric rate umbrella sampling: passed on the second clean execution.
- AUS pilot and frozen production: one window, two pilot samples, target met,
  frozen plan written, GPU engine retained.
- Constrained parent chain: 2 snapshots, 3 GPU propagation steps, passed.
- Paired recrossing children: 2 child pairs, 3 steps each, finite signed
  numerators and endpoint energy drift around `1e-15`, passed.
- Existing Torch regression subset: `389 passed`.
- New JaxPIP GPU input contract tests: `3 passed`.

The detailed machine-readable records are in
[`jaxpip-gpu-real-report.json`](jaxpip-gpu-real-report.json) and
[`gpu-validation-summary.json`](gpu-validation-summary.json). The complete
Torch test output is in [`torch-tests.txt`](torch-tests.txt).

The rate checks are execution validation with a deliberately simple synthetic
three-atom model. They verify the production data path, device ownership,
checkpointing and US/AUS/recrossing integration; they do not establish a
chemical rate or convergence of a fitted PES.
