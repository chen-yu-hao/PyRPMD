# JaxPIP GPU fixes: T4 validation

Date: 2026-11-15
Hardware: Google Colab Tesla T4, 15 GiB
Runtime: PyTorch 2.14.1+cu130, JAX 0.11.1, JaxPIP 1.0.0, float64

This run validates the fixes for the previous report's runtime gaps. The model
directory was a checked `pyrpmd.jaxpip-fit/1` fixture with a real JaxPIP linear
model. It exercises the same loader, checksum gate, JAX GPU graph, DLPack bridge
and TorchDynamics block used by a user fitted model.

## Fixes validated

- Model directory checksum and test-bound metadata are checked before loading
  JAX. A deliberately altered manifest was rejected with the production gate;
  `allow_unvalidated=true` remains an explicit pilot escape hatch.
- JAX compilation is warmed up before the production window/parent/child loop.
  The benchmark reports warmup separately (`compile_seconds = 0.474 s`).
- `max_frames_per_call=32` split a 128-frame request into bounded device
  batches. All runs remained finite and the T4 completed the long block.
- Actual GPU rate window sampling through `model_directory` completed with two
  saved samples.
- The combined Torch/JaxPIP regression subset completed with `394 passed`.

## Long-block measurements

The benchmark used 8 beads, trajectories K = 1, 4, 8, 16, 256 production
steps and 16 warmup steps. Results are in
[`jaxpip-fix-benchmark.json`](jaxpip-fix-benchmark.json).

| K | Frames | Block time | Frames/s | Torch reserved peak |
|---:|---:|---:|---:|---:|
| 1 | 8 | 0.476 s | 4,302 | 2.0 MiB |
| 4 | 32 | 0.454 s | 18,038 | 2.0 MiB |
| 8 | 64 | 0.704 s | 23,258 | 2.0 MiB |
| 16 | 128 | 1.109 s | 29,553 | 2.0 MiB |

The fitted fixture is intentionally small, so its VRAM numbers are not a
chemical-model forecast. `torch_reserved_peak` measures PyTorch's allocator;
the JSON also records before/after `nvidia-smi` usage so JAX's allocator is not
silently omitted. The useful result is that the memory cap and batch split are
observable and the force loop remains finite over a long block.

The remaining production limitation is the absence of the user's actual
chemical fitted model in this workspace. Chemical test-set error, domain
coverage, rate convergence and model-specific VRAM still require running this
same benchmark against that model directory.
