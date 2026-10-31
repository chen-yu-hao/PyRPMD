# Native batched ONNX potentials

This optional C++17 library connects a **validated JaxPIP ONNX model** to the
existing PyRPMD PES C ABI v1. JAX, Equinox and jax2onnx are offline export tools;
there are no Python/JAX calls inside the native force evaluation or supported
C++ propagation blocks. No change to PyRPMD's integrator mathematics is needed.

## Scope

- A session is loaded once per context. Different contexts may hold different
  models. Input/output host buffers are reused when the batch size is unchanged.
- One synchronous `Session::Run` evaluates the whole `[B,N,3]` batch, returning
  energy `[B]` and negative Cartesian gradient `[B,N,3]`, all float64.
- The ABI accepts bohr/hartree; models can use angstrom/bohr and eV/hartree.
  The plugin converts positions, energies and forces together, without a `1/P`
  factor. Conversion constants travel with the exported model. For an eV/angstrom
  model, `q_model=q_AU*BOHR_ANGSTROM`, `E_AU=E_model*EV_TO_AU`, and
  `F_AU=F_model*EV_TO_AU*BOHR_ANGSTROM`.
- CPU is the default. `device="cuda:0"` explicitly requests a CUDA-enabled ORT
  SDK and its installed CUDA/cuDNN dependencies. Missing CUDA is an error.
  `allow_cpu_fallback=false` (default) rejects unsupported GPU graph partitioning;
  setting it true permits mixed CPU/GPU execution. It does not turn a missing
  CUDA provider into a silent CPU-only run.
- ABI v1 is host-memory, fixed-composition, nonperiodic, single-surface. CUDA
  still transfers coordinates/results between host and device each call. This
  is **not** a GPU-resident integrator or a device-pointer/zero-copy ABI.

## Build separately from PyRPMD

Install the ONNX Runtime **C/C++ SDK**, not only its Python wheel. The ordinary
PyRPMD installation is unchanged and does not depend on ONNX or nlohmann/json.
On Debian/Ubuntu install `nlohmann-json3-dev`, then:

```sh
cmake -S native/onnx -B build/onnx -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime-sdk
cmake --build build/onnx --parallel 2
```

CMake locates `onnxruntime_cxx_api.h` and `libonnxruntime` below that directory.
Keep the matching runtime libraries installed and on the dynamic-loader search
path. Linux CPU CI targets SDK/Python runtime 1.24.1, verifies the publisher's
SDK archive SHA256, and saves the Python dependency environment as an artifact.
GPU execution is not covered by hosted CPU CI. OpenVINO is not implemented.

## Export a real model once

The fitted checkpoint alone is insufficient: keep its matching PIP/FI basis,
atom ordering, electronic state, units and representative nondegenerate
geometries. Units are declared explicitly, never inferred from a file name.
This exporter supports JaxPIP linear and neural-network checkpoints. Install the
pinned offline environment separately from a production dynamics environment:

```sh
python -m pip install -r tools/requirements-jaxpip-export.txt
python tools/export_jaxpip_onnx.py \
  --basis /absolute/path/basis.json.gz --checkpoint /absolute/path/model.eqx \
  --kind nn --geometries /absolute/path/validation.npy \
  --atomic-numbers 1 1 8 --charge 0 --multiplicity 1 \
  --position-unit angstrom --energy-unit eV \
  --output /absolute/path/model.onnx > export-report.json
```

The pinned stack includes JaxPIP commit
`a18c8cca1e795202194d4082919ebd71722df125`, JAX 0.9.0, Equinox 0.13.2,
jax2onnx 0.17.1, ONNX 1.20.1 and ONNX Runtime 1.24.1. Earlier testing with
jax2onnx 0.11.2 found a force-graph reshape failure when the dynamic batch grew
from 1 to 3. A successful single-frame export is **not** sufficient validation.

`validation.npy` is a numeric array `[M>=2,N,3]` in the model's length unit.
The example atomic numbers specify H,H,O **in that order**, not a generic model
for every molecule. Use the trained model's actual ordered elements and state.
The exporter checks dynamic float64 I/O, JAX/ORT energy+force agreement at batch
sizes 1/3/32, batch-versus-single independence on those samples, and force
finite differences. It writes the model atomically only after validation. Such
checks verify the interface, not the PES's chemical training domain or accuracy.

## Use the existing PyRPMD provider

Replace the potential section of an otherwise valid gas-phase job:

```toml
[potential]
provider = "cabi"
library = "/absolute/path/PyRPMD/build/onnx/libpyrpmd_onnx.so"

[potential.params]
model = "/absolute/path/model.onnx"
device = "cpu"
intra_op_threads = 1

[dynamics]
loop_backend = "native"
block_steps = 256
```

The model path is passed literally to the native runtime: use an absolute path
for reliable process/MPI/restart behavior. PyRPMD passes atom count, element
order and available charge/multiplicity; the plugin checks them against model
metadata. `loop_backend="native"` requests the native route explicitly and must
not be used with unsupported custom Python CVs/biases expecting a fallback.
Each process constructs its own context. Avoid nested oversubscription: start
with one ORT intra-op thread per independently scheduled trajectory worker.

## Model contract and errors

ONNX custom metadata `pyrpmd.onnx` contains JSON with schema `pyrpmd.onnx/1`,
`batch_independent=true`, ordered `atomic_numbers`, `state.charge`,
`state.multiplicity`, `position_unit`, `energy_unit`, `force_unit` and conversion
`constants`. The ordered outputs are energy and **forces**, not gradients.
The input is `[B,N,3]`, energy `[B]`, forces `[B,N,3]`; B is symbolic/dynamic,
N and 3 are fixed and every I/O tensor is float64. The exporter also records
source-file SHA256 digests and package versions. Only load trusted models:
metadata is a declaration, not a mathematical proof of a graph's independence.

Every exported C entry point contains exceptions. Invalid dimensions, buffer
counts, config, dtype or model metadata fail explicitly. Nonfinite outputs
produce failed frame status and NaN data; they are not replaced by zero forces.
A nonzero global return invalidates the entire batch, as required by ABI v1.

## Batching and concurrency: implemented versus future work

`B=P` evaluates all beads of one trajectory at once. The PES also accepts
`B=K*P` when a caller supplies K independent trajectories. **This PR does not
add a cross-trajectory integrator/scheduler that packs K trajectories for you.**
The existing workflow trajectory/process scheduling and native integrators are
reused, including their existing OpenMP kernels.

Each C++ context has its own ORT session, reusable buffers and mutex. Separate
contexts can run concurrently from a native/OpenMP caller; same-context calls
serialize. `context_test.cpp` checks that property using two models, eight
contexts and four OpenMP threads. It is a PES concurrency test, **not** a new
OpenMP multi-trajectory RPMD driver. The existing Python CABIPotential lock
remains conservative and serializes one shared library within each process.
Context destruction must not race evaluation, as specified by ABI v1.

## Validation

The `Native ONNX PES` workflow builds the real C++ SDK plugin (no mock runtime)
and runs shape/unit/force/batch/error/lifecycle tests, all three integrators at
1/8/32 beads, a profiler assertion of zero Python PES callbacks inside the
native block, and the native OpenMP context test. A separate step exports real
JaxPIP linear and PIP-NN *test* models and compares native forces/trajectories
with JAX and the Python propagation reference. Reports and timings are retained
as workflow artifacts. These small test models are not production chemical PESs.
