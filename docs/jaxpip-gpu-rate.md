# JAX DLPack potentials and Torch GPU rate workflows

PyRPMD provides a first-class zero-copy JAX DLPack potential adapter
(`provider = "jax-dlpack"`, {class}`~pyrpmd.potentials.JAXDLPackPotential`)
alongside the checked `pyrpmd.jaxpip-fit/1` wrapper (`provider = "jaxpip"`,
{class}`~pyrpmd.potentials.JaxPIPPotential`). Both expose the standard NumPy
`.evaluate()` contract as well as a device-resident `.as_torch_pes()` /
`make_workflow_torch_pes()` bridge for `TorchDynamics` and
`sampling.engine = "torch"` workflows.

## First-class `jax-dlpack` provider

Any stateless, batch-independent `float64` JAX energy or force model—such as a
raw JAX potential energy function, an Equinox module, or a custom neural-network
/ PIP force field—can be used directly without ONNX export or host copies:

```toml
task = "rate"
temperature = "300 K"
beads = 32

[potential]
provider = "jax-dlpack"
model = "pes_module.py:energy_fn"
device = "cuda:0"
position_unit = "bohr"
energy_unit = "hartree"
# Optional VRAM guard for large K*P batches.
max_frames_per_call = 256

[sampling]
engine = "torch"
device = "cuda:0"
thermostat = "andersen"

[dynamics]
timestep = "0.05 fs"
integrator = "cayley-bcb"

[recrossing]
child_batch_pairs = 8
```

### Supported JAX callable contracts

`JAXDLPackPotential` enforces `jax_enable_x64 = True` and rejects `float32`
inputs, parameters, or outputs without silent precision or device fallback:

- **Single-frame callables (`batched = "auto"` or `false`)**: accept `[N, 3]`
  coordinates (or `(positions, cell, pbc)` / `(positions, *, cell, pbc)` when
  `periodic = true`) and return:
  - scalar potential energy $E(q)$, automatically wrapped with
    `jax.jit(jax.vmap(jax.value_and_grad(...)))` to compute $F = -\nabla_q E$,
  - `(energy, forces)` (or `(energy, gradient)` with `derivative = "gradient"`), or
  - a mapping containing `"energy"` / `"energies"` and optional `"forces"`,
    `"virial"`, or `"stress"`.
- **Batched callables (`batched = "auto"` or `true`)**: accept `[F, N, 3]`
  batches directly and return `[F]` energies with `[F, N, 3]` forces (or let
  `jax.value_and_grad` differentiate the frame-summed energy in one XLA pass).
- **Periodic virials**: when `periodic = true` and `virial = "auto"` or
  `"autograd"`, affine strain virials are differentiated inside XLA alongside
  the coordinate forces and passed to `TorchPES` via DLPack without requiring
  PyTorch autograd through external DLPack buffers.

### CUDA stream synchronization

When consumed via `.as_torch_pes()` or `make_workflow_torch_pes()`, PyTorch and
JAX exchange `float64` device buffers through the Python Array API DLPack
protocol (`__dlpack__(stream=...)`). Active non-default `torch.cuda.Stream`s
are ordered explicitly across PyTorch and XLA, input tensors invoke
`record_stream` on the caller stream, stream switches record cross-stream CUDA
events, and XLA output buffers remain retained across dispatches so neither
PyTorch's caching allocator nor XLA's `BFCAllocator` aliases in-flight memory.

## Checked JaxPIP GPU rate workflow

`JaxPIPPotential` (`provider = "jaxpip"`) delegates its NumPy evaluation and
DLPack `TorchPES` bridge to `JAXDLPackPotential` while preserving full
`pyrpmd.jaxpip-fit/1` manifest, checksum, and test-bounds verification. The
ONNX/C ABI route remains available for portable host-memory production; on a
CUDA GPU (such as a Colab T4), use the device-resident route:

```toml
task = "rate"
temperature = "200 K"
beads = 32

[potential]
provider = "jaxpip"
model_directory = "/content/model-fit"
device = "cuda:0"
max_frames_per_call = 256

[sampling]
engine = "torch"
device = "cuda:0"
thermostat = "andersen"

[dynamics]
timestep = "0.05 fs"
integrator = "cayley-bcb"
loop_backend = "auto"
block_steps = 256

[recrossing]
child_batch_pairs = 8
```

`model-fit` is a checked `pyrpmd.jaxpip-fit/1` directory containing `model.json`,
the JaxPIP basis and `model.eqx`. A direct configuration may provide `basis`,
`checkpoint`, `kind`, `position_unit`, `energy_unit` and the ordered
`atomic_numbers` instead. A fit outside its requested test bounds is rejected
for production; `allow_unvalidated = true` is reserved for an explicitly
labelled pilot.

## Supported `sampling.engine = "torch"` rate providers

In addition to `jaxpip` and `jax-dlpack`, `task = "rate"` with
`sampling.engine = "torch"` accepts every stateless, batch-independent
`float64` tensor workflow provider supported by `make_workflow_torch_pes`:

- Analytic built-ins: `free`, `harmonic`, `periodic_cosine`
- Direct tensor / DLPack bridges: `jax-dlpack`, `jaxpip`, `torch`
- Batched ML potentials (`dtype = "float64"`): `mace-batched`,
  `nequip-batched`, `fairchem-batched`, `deepmd-batched`, `metatomic`

Host-only providers (such as `cabi`, `python`, `ase`, `lammps`, or `openmm`)
and `float32` configurations are rejected at input preflight before allocating
output directories or starting sampling.

`TorchDynamics` keeps `[trajectories, beads, atoms, 3]` `q`/`p` tensors on the
selected CPU or CUDA device while the umbrella, RPMDrate coordinate, flux
metric, constrained parent RATTLE, and opposite-momentum child batches execute
on device tensors. Only checkpoints, reduced CV/energy observations, and final
rate arrays are transferred to the host.

## Colab setup and benchmarking

In Colab, install the repository and the pinned GPU environment, then verify
the runtime before starting a long job:

```bash
!nvidia-smi
!pip install -e .
!pip install -r tools/requirements-jaxpip-gpu.txt
```

For a fitted JaxPIP model, measure compile time separately from steady-state
throughput and memory:

```bash
PYTHONPATH=src python benchmarks/jaxpip_gpu_production.py /content/model-fit \
  --beads 32 --trajectories 1 4 8 16 --steps 4096 \
  --output runs/jaxpip-t4-benchmark.json
```
