# Native ONNX CUDA float64 ScatterND validation — 2026-11-15

Validation record for issue #202 (`libpyrpmd_onnx.so` custom `float64` CUDA
`ScatterND` operator and dynamic-batch shape helper canonicalization for
exported `JaxPIP` models). Machine-readable environment versions, model SHA256
digests, execution-provider node counts, numerical parity and timing splits are
recorded in `t4_validation_summary.json`. Binary `.onnx`, `.eqx` and `.so`
artifacts are not committed to Git.

## Hardware and software environment

- **Hardware:** NVIDIA Tesla T4 (`sm_75`, 15,360 MiB VRAM), 2-vCPU host VM,
  NVIDIA Driver `580.82.07`, Linux `6.6.159+ x86_64`.
- **Native toolchain:** GNU C++ `13.3.0`, CUDA Toolkit `13.0.88` (`nvcc`),
  CMake `3.31.10`, `nlohmann_json 3.11.3`, Python `3.11.17`.
- **ONNX Runtime C/C++ SDKs:** `onnxruntime-linux-x64-gpu-1.32.0` (CUDA 13) for
  GPU builds; `onnxruntime-linux-x64-1.24.1` for strict CPU-only CI parity.
- **Pinned offline export stack:** `jaxpip 1.0.0`
  (`a18c8cca1e795202194d4082919ebd71722df125`), `jax 0.9.0`, `equinox 0.13.2`,
  `jax2onnx 0.17.1`, `onnx 1.20.1`, `onnx_ir 1.1.0`, `onnxruntime 1.24.1`.

## CUDA ExecutionProvider partitioning (`allow_cpu_fallback = false`)

Queried directly via ONNX Runtime C API `Session_GetEpGraphAssignmentInfo` with
`session.disable_cpu_ep_fallback = "1"` on `device = "cuda:0"`:

| Exported model | SHA256 | `CUDAExecutionProvider` nodes | `CPUExecutionProvider` nodes | ORT warnings |
| --- | --- | ---: | ---: | ---: |
| 3-atom `PolynomialLinearModel` (`morse`, deg 3) | `ed0e08c1714fd4d39ed85f19cd658ac697efa95b2115a139aadd31233bb4ae70` | 87 | 0 | 0 |
| 3-atom `PolynomialNeuralNetwork` (`tanh`, `morse`, deg 3) | `5d87b84239ba3b80e91056aa385a01ddb5420a8f3e3f5aa5a2d8dd13a9fedbb9` | 112 | 0 | 0 |
| 4-atom `PolynomialNeuralNetwork` (`isru`, `reciprocal`, `[8, 6]` MLP) | `fac9bda7c09b9383695f19dde779f6f3b0675b079432393f45d5cde912747b87` | 139 | 0 | 0 |

## Numerical parity on `cuda:0` (`B` in `{1, 8, 32, 128}`)

Maximum absolute differences across batches `1, 8, 32, 128` (`allow_cpu_fallback = false`,
`capabilities.float64_cuda_scatter_nd = true`), well within the `< 1e-12`
hartree and `< 1e-12` hartree/bohr acceptance thresholds:

| Model | Max `|dE|` vs JAX (Ha) | Max `|dF|` vs JAX (Ha/Bohr) | Max `|dE|` vs CPU ORT (Ha) | Max `|dF|` vs CPU ORT (Ha/Bohr) |
| --- | ---: | ---: | ---: | ---: |
| 3-atom `PolynomialLinearModel` | `4.34e-19` | `2.71e-20` | `4.34e-19` | `5.42e-20` |
| 3-atom `PolynomialNeuralNetwork` (`tanh` + `morse`) | `2.93e-18` | `2.08e-17` | `2.14e-18` | `2.39e-17` |
| 4-atom `PolynomialNeuralNetwork` (`isru` + `reciprocal`) | `6.08e-17` | `1.21e-15` | `1.22e-16` | `2.04e-15` |

## Build and test matrix

- **GPU build (`-DPYRPMD_ONNX_CUDA_KERNELS=ON`, default & `-mfma`):**
  - `pytest -v tests/test_onnx_native.py`: `45 passed, 1 skipped` (covers
    `none`, omitted-attribute `none`, `add`, `mul`, `max`, `min` reductions,
    slice and element updates, negative indices, mixed `float32` + `float64`
    `ScatterND` domains, out-of-bounds index rejection on CUDA without host/GPU
    corruption, and end-to-end `pyrpmd.runner.run` on both `"cpu"` and `"cuda:0"`).
  - `pytest -v tests/test_cabi_lifecycle.py tests/test_native_propagation.py`:
    `24 passed` (and passes under `PYRPMD_NO_NATIVE=1`).
- **GPU SDK build with `-DPYRPMD_ONNX_CUDA_KERNELS=OFF`:** `ldd` confirms zero
  `libcudart` linkage; `capabilities.float64_cuda_scatter_nd` reports `false`;
  loading a float64 `ScatterND` model on `cuda:0` with
  `allow_cpu_fallback=false` fails explicitly with actionable rebuild guidance.
- **Strict CPU-only CI build (`ONNXRUNTIME_ROOT` CPU SDK `1.24.1`, CUDA stripped
  from `PATH`, OpenMP ON & OFF):** `ldd` confirms zero CUDA linkage;
  `pytest -q tests/test_onnx_native.py`: `46 passed`;
  `tests/onnx/jaxpip_roundtrip.py` CPU roundtrip passes.
- **Documentation & packaging checks:** `sphinx -W --keep-going` passed with
  zero warnings for English and Chinese `html` and `singlehtml`; `python -m
  build --sdist` verified `native/onnx/scatter_nd_cuda.cu`,
  `native/onnx/scatter_nd_cuda.h` and `native/onnx/onnx_proto_rewrite.h` in the
  source distribution.
- **Repository-wide non-campaign suite (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest -q -m "not campaign"`):**
  `10462 passed, 240 skipped, 1 deselected, 436 subtests passed` on the 2-vCPU
  Colab VM (`2 failed` only in untouched `tests/test_worker_thread_limits.py`,
  where OpenBLAS caps `OPENBLAS_NUM_THREADS=4` at `2` on a 2-vCPU host at
  process startup).

## Reproduce

```bash
# 1. Build CUDA-enabled ONNX PES plugin
cmake -S native/onnx -B build/onnx-cuda -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu \
  -DPYRPMD_ONNX_CUDA_KERNELS=ON -DPYRPMD_ONNX_CONTEXT_TEST=ON
cmake --build build/onnx-cuda --parallel 2

# 2. Run unit, contract and public runner end-to-end tests
PYRPMD_REQUIRE_ONNX_TESTS=1 \
PYRPMD_ONNX_LIBRARY="$PWD/build/onnx-cuda/libpyrpmd_onnx.so" \
PYRPMD_ONNX_CONTEXT_TEST="$PWD/build/onnx-cuda/onnx_context_test" \
python -m pytest -q tests/test_onnx_native.py tests/test_cabi_lifecycle.py tests/test_native_propagation.py

# 3. Run offline JaxPIP export + strict cuda:0 parity roundtrip (B = 1, 8, 32, 128)
PYRPMD_ONNX_LIBRARY="$PWD/build/onnx-cuda/libpyrpmd_onnx.so" \
python tests/onnx/jaxpip_roundtrip.py --device cuda:0 \
  --exporter-python /path/to/export-venv/bin/python \
  --output build/onnx-cuda/jaxpip-validation
```

## Scope and limitations

- `pyrpmd_pes_api_v1` transfers `[B, N, 3]` coordinates and forces through host
  memory; this change removes CPU EP operator fallback inside ONNX Runtime on
  `cuda:<id>`, not host/device buffer transfers across the C ABI boundary.
- Validation models in `jaxpip_roundtrip.py` and `test_onnx_native.py` are small
  deterministic test fixtures, not fitted production potential energy surfaces;
  no chemical accuracy, rate convergence or production GPU speedup factor is
  claimed from these fixture timings.
- Implementation, documentation and automated hardware verification were
  prepared with AI assistance (Antigravity) and verified through the automated
  test suites and Tesla T4 hardware checks recorded above.
