# JaxPIP GPU rate workflow

The ONNX/C ABI route remains available for portable host-memory production. A
Colab T4 with a real JaxPIP model should use the device-resident route instead:

```toml
task = "rate"
temperature = "200 K"
beads = 32

[potential]
provider = "jaxpip"
model_directory = "/content/model-fit"
device = "cuda:0"
# Optional VRAM guard for large K*P batches.
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
`atomic_numbers` instead.

The model directory is checksum-verified at load time. A fit outside its
requested test bounds is rejected for production; `allow_unvalidated = true`
is reserved for an explicitly labelled pilot. `max_frames_per_call` splits a
large `K*P` force request into device batches to cap JAX workspace memory. It
preserves frame order and results, while reducing peak memory at the cost of
additional launches.

The route loads JaxPIP once on the selected JAX device and exposes its compiled
energy/force function as a Torch module. Coordinates and forces cross the
JAX/Torch boundary through DLPack on the same CUDA device. `TorchDynamics` keeps
`[trajectories, beads, atoms, 3]` q/p tensors on the GPU, and the GPU umbrella,
RPMDrate coordinate, flux metric, constrained parent RATTLE and opposite-
momentum child batches operate on those tensors. Existing US/AUS window
planning, MBAR/UI analysis, rate checkpoints and recrossing statistics remain
the workflow contract.

The runtime path uses the C++/CUDA kernels supplied by the selected JAX and
PyTorch CUDA wheels for the force graph and ring maps. Python remains at the
workflow boundary for window scheduling, checkpoint commits and reduced
observations; the old host-memory ONNX C ABI is not used by this route.

Only checkpoints, reduced CV/energy observations and final rate arrays are
copied to the host. The GPU route currently requires a fixed nonperiodic system,
the built-in RPMDrate/Distance/Cartesian CV family and Andersen parent/window
sampling. Unsupported custom CVs or thermostats fail during validation. The
route is explicitly opt-in through `sampling.engine = "torch"`; it never changes
legacy or ONNX execution.

In Colab, install the repository and the pinned JaxPIP exporter environment,
then verify the runtime before starting a long job:

```bash
!nvidia-smi
!pip install -e .
!pip install -r tools/requirements-jaxpip-gpu.txt
```

Use float64 throughout. The model's training/test acceptance report is a model
quality check; it does not replace rate convergence in beads, timestep,
umbrella overlap, parent sampling or recrossing plateau length.

For a real fitted model, measure compile time separately from steady-state
throughput and memory:

```bash
PYTHONPATH=src python benchmarks/jaxpip_gpu_production.py /content/model-fit \
  --beads 32 --trajectories 1 4 8 16 --steps 4096 \
  --output runs/jaxpip-t4-benchmark.json
```
