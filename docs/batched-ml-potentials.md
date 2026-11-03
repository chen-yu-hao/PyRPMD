# Batched machine-learning potentials

The opt-in providers `mace-batched`, `nequip-batched`, `fairchem-batched`,
`deepmd-batched` and `metatomic` evaluate all physical configurations in one
model batch. For `K` independent trajectories with `P` beads, the potential
receives `[K*P,N,3]`. Graphs are disconnected between configurations; neither
the physical forces nor the energies acquire a bead scaling factor.

The existing `mace`, `nequip` and `fairchem` providers continue to use their
existing ASE calculators. The existing `deepmd` provider already passes the
complete NumPy batch to `DeepPot.eval`; it is unchanged. Separate new provider
names keep previous model selection, behavior and run identities intact. The
new `deepmd-batched` path adds a direct Torch interface and sparse graph
preparation for its supported PyTorch energy models.

## Install one framework environment

Install the matching optional extra in the environment that will run the
simulation. The adapters import optional packages only when constructed.
Tests create small random models locally; no pretrained weights are downloaded.

```bash
python -m pip install -e '.[mace-batched,test]'
# In separate environments, as appropriate:
python -m pip install -e '.[nequip-batched,test]'
# For the tested NequIP AOTInductor toolchain:
python -m pip install -e '.[nequip-batched-aoti,test]'
python -m pip install -e '.[fairchem-batched,test]'
python -m pip install -e '.[deepmd-batched,test]'
python -m pip install -e '.[metatomic,test]'
```

The APIs used here were checked against these versions:

| Provider | Framework | CPU validation runtime | Model input |
| --- | --- | --- | --- |
| `mace-batched` | MACE 0.3.17, Vesin 0.6.2 | Torch 2.7.1, e3nn 0.4.4 | Local `.model`, or energy `MACE` / `ScaleShiftMACE` module |
| `nequip-batched` | NequIP 0.19.1 | Torch 2.7.1 eager/TorchScript; Torch 2.13.0 eager/AOTI; e3nn 0.6.0 | Local `.nequip.pth`, batch-target `.nequip.pt2`, or `GraphModel` |
| `fairchem-batched` | fairchem-core 2.25.0 | Torch 2.13.0, e3nn 0.6.0 | Local `MLIPInferenceCheckpoint`, or configured `MLIPPredictUnit` |
| `deepmd-batched` | DeePMD-kit 3.2.0, Vesin 0.6.2 | Torch 2.11.0 | Supported local PyTorch energy export, or initialized energy model |
| `metatomic` | metatomic-torch 0.1.19, Vesin 0.6.2 | Torch 2.7.1, metatensor-torch 0.10.8 | Exported `AtomisticModel` file, or initialized `AtomisticModel` |

Use separate environments for MACE and current NequIP/FAIR-Chem/DeePMD:
their e3nn and Torch requirements differ. In particular, MACE 0.3.17 pins
e3nn 0.4.4 while NequIP 0.19.1 requires e3nn 0.6. DeePMD's Torch extra pins
the Torch ABI used by its native extension; a different Torch build is not a
supported substitution. Select a CPU or CUDA Torch installation appropriate
to the framework and machine before installing the extra. No provider changes
devices when a requested device is unavailable.

For CUDA NequIP, also install its native batched neighbor implementation:

```bash
python -m pip install -e '.[nequip-batched-cuda]'
```

The CPU benchmark and validation recorded below do not exercise CUDA.

## Configure a workflow

```toml
[potential]
provider = "mace-batched"
model = "models/local.model"
device = "cuda:0"                 # use "cpu" on a CPU machine
dtype = "float64"                 # must match the saved model
virial = false                    # enable when a physical virial is needed

[parallel]
trajectory_batch_size = 4
bead_workers = 1

[dynamics]
loop_backend = "auto"
```

This combines independent workflow requests using the existing
[trajectory batching contract](batched-dynamics.md). All new providers declare
`native_batch=true`, `batch_independent=true`, `trajectory_batch_safe=true`
and `stateful=false`. They are serialized within a provider context; these
declarations do not permit concurrent mutation of a shared model. A changing
batch size or frame order must not change a frame's physical result, apart
from floating-point reduction rounding.

`trajectory_batch_size` controls the number of independent tasks sharing a
call. Increase it only after measuring throughput and memory on the actual
model. Do not reduce the physical bead count merely to fit a performance
target. There is no automatic microbatching, per-frame inference retry or
CPU fallback after an out-of-memory or model error.

The complete example input is
[`examples/potentials/batched_ml.inp`](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/potentials/batched_ml.inp).
Supply a model trained for its chemistry before using that input scientifically.
For a local interface demonstration with untrained weights, run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python examples/potentials/batched_ml_tensor.py
```

### Framework-specific options

**MACE.** Only the conservative energy `MACE` and `ScaleShiftMACE` architectures
are supported. A multi-head model without a `Default` head requires an explicit
`potential.params.head`. Polar, magnetic, dipole-only and extra-input embedding
models are rejected. Standard MACE inputs use angstrom/eV, as in the default ASE
adapter; this subset does not accept custom MACE unit scales.
Topology assembly uses Vesin's combined pair output and takes ownership while
applying each frame's index offset. Periodic shifts are copied directly into
the model dtype before Vesin reuses its buffers. This reduces intermediate
copies while preserving the exact disconnected graph and one model call.
Vesin 0.6.2 launches CUDA topology kernels on the legacy default stream. For
a caller using another CUDA stream, the adapter orders graph preparation on
the default stream, waits before returning it to the caller stream, and
records tensor lifetimes on the consuming stream. Model inference remains on
the caller's stream. This also orders float32-to-float64 topology promotion and
copies from Vesin's reused buffers; no extra per-frame host synchronization is
introduced by the adapter. Vesin's own completion synchronizations remain.

**NequIP.** Eager, TorchScript and native AOTInductor routes are tested with
locally initialized small NequIP models. AOTI requires Torch >=2.13, with
2.13.0 the tested version. Recompile `.nequip.pt2` files with that toolchain and
`--target batch`; an ASE-only artifact does not carry the batch contract.
The official artifact metadata does not expose a reliable compiler Torch
version, so the provider cannot verify the compiler's provenance. It checks
the current runtime, input/output fields, precision and requested device.
CUDA loading selects the requested device index explicitly even when the
artifact records only the device type. `params.energy_units_to_eV` and
`params.length_units_to_A` have the same meanings as in `NequIPCalculator`.
`params.chemical_species_to_atom_type_map` can map elements to trained type
names. CPU graph construction defaults to `matscipy`; CUDA requires
`neighborlist_backend="alchemiops"`. Requesting CPU graph construction for a
CUDA tensor evaluation raises an error instead of copying coordinates to host.

Set `AOTI_RUNTIME_CHECK_INPUTS=1` in the process environment before loading an
AOTI model and retain it throughout evaluation. The provider verifies the
setting on every call; the native runtime enforces the artifact's compiled
frame/node/edge shape, stride, dtype and device constraints before kernels run.
Out-of-range inputs raise an error without padding, silent fallback or an
inference retry. A concrete Torch 2.7.1 CPU compile specialized the node count
despite dynamic export metadata; this is why older AOTI runtimes are rejected
and native guards are mandatory. Eager/TorchScript inference remains available
on its supported older runtime.

**FAIR-Chem.** Set `potential.params.task_name` to the trained chemistry domain.
The provider uses `MLIPPredictUnit.predict` on one `AtomicData` graph batch,
including its trained normalization and element references. It requires
conservative forces, compatible derivative normalizers and the general
execution mode. Direct-force/direct-stress models, Nitro, MOLE merging,
automatic compilation/precision fallback and single-atom lookup substitutions
are not part of this subset. Its strict inference settings are explicit in the
provider; the old `fairchem` path is unaffected.

**DeePMD.** The new Torch route supports conservative float32 and float64
energy models without changing learned parameter precision. It preserves
DeePMD's high-precision interface: internal geometry, total energy and virial
are float64, while forces have the requested input precision. This requires
the native `DP_INTERFACE_PREC=high` convention; alternative global precisions
are rejected. It uses device-local sparse neighbors and the native atomic
energy core, then differentiates the whole batch once, avoiding dense neighbor
intermediates. Periodic positions are normalized in float64 before descriptor
evaluation, as in native DeePMD, while the caller's unwrapped positions remain
unchanged. The trained neighbor selection remains part of the model;
unsupported models or ambiguous truncation are rejected. Full or no PBC is
supported. The existing NumPy `deepmd` provider retains its wider backend support.

**Metatomic.** The minimum supported interface is the standard differentiable
`energy` output of metatomic-torch 0.1.19. The adapter honors exported atomic
types, units, dtype, devices and requested neighbor lists. A list of `System`
objects is submitted in one `AtomisticModel` call. System-level `charge` and
`spin_multiplicity` inputs are supported when the model requests them. Arbitrary
additional inputs, energy variants, nonconservative force heads and exports
that provide only explicit gradients without autograd connectivity are
rejected. A metatomic model may internally loop over systems: one interface
call alone does not prove a particular model has fused GPU kernels.

## Units, electronic state and cell semantics

Both public interfaces use bohr for positions and cells, hartree for energy
and configurational virial, and hartree/bohr for forces. Forces are
`-dE/dq`. The returned energy is the energy differentiated to produce the
forces, including model normalization. It is not an extrapolated energy with
a different derivative. No bead, trajectory or temperature factor is applied.

`D` and `T` map to hydrogen atomic number/type for PES evaluation. Their distinct
masses remain in the bound PyRPMD `System` and the integrator. Composition,
charge and spin multiplicity cannot change during `bind_system`. Cell and PBC
rebinding invalidate cached geometry tensors, while preserving loaded weights.
Coordinates stay unwrapped at the public boundary; any internal periodic graph
normalization does not change the integrator's coordinates.

| Provider | Charge and spin selection | Boundary conditions |
| --- | --- | --- |
| MACE energy subset | Neutral singlet; nondefault state is rejected | None, full, partial, triclinic cell |
| NequIP subset | Neutral singlet; no charge/spin input is invented | None, full, partial; CUDA backend must support the requested geometry |
| FAIR-Chem `omol` | Charge integer and spin equal to multiplicity, as in ASE | Input PBC is honored |
| Other FAIR-Chem tasks | System charge 0 / multiplicity 1; model embeddings charge 0 / spin 0 | Input PBC is honored |
| DeePMD subset | Neutral singlet; state-conditioned models are unsupported | None or full; partial PBC rejected |
| Metatomic | Pass requested `charge` / `spin_multiplicity`; reject unsupported nondefault state | None, full, partial; nonperiodic lattice rows are zeroed for metatomic |

With `virial=true`, the result is the physical configurational tensor
`W = -dE/dstrain = -volume * stress`, in hartree. It excludes kinetic and
ring-spring terms and is not itself a quantum pressure estimator. For a
nonperiodic molecule, the configurational strain derivative can be nonzero
even when an upstream ASE calculator reports zero stress for nonperiodic
directions. Those cases are checked against an independent affine energy
finite difference, rather than equating a zero ASE stress to a zero virial.

## Device tensor interface

```python
import torch
from pyrpmd.potentials import MACEBatchedPotential

pes = MACEBatchedPotential(
    ["O", "D", "T"], "models/local.model",
    device="cuda:0", dtype="float64", virial=True,
)
q = torch.tensor([[[0., 0., 0.], [1.8, 0., 0.], [-0.6, 1.7, 0.]]],
                 device=pes.device, dtype=torch.float64)  # bohr; shape [B,N,3]
result = pes.evaluate_tensor(q)
# result.energy[B], result.forces[B,N,3], optional result.virial[B,3,3]
```

`evaluate_tensor` returns the existing `torch_backend.TensorPESResult` type.
It checks shape, finite values, dtype and exact device, and preserves the
requested model precision and device. CUDA neighbor preparation and model
evaluation keep coordinates on the selected device. On CPU, NequIP uses its
upstream NumPy/matscipy neighbor constructor; its public result remains Torch
tensors. Finite-value validation can synchronize scalar status with the host;
this adapter does not
claim a completely synchronization-free propagation loop. MACE and metatomic
neighbor APIs construct per-system topology before the one model batch; this
is distinct from doing per-frame model inference.

`torch.no_grad()` around the caller is supported: coordinate derivatives are
enabled internally. `inference_mode`, active autocast and implicit CUDA TF32
are rejected. `dtype="float32"` is an explicit model choice, with correspondingly
looser numerical tolerances. Native reduction precision is recorded separately
in `metadata["energy_dtype"]` and `metadata["virial_dtype"]`; DeePMD also records
its internal `geometry_dtype`. For a float32 model, the output contract is:

| Provider | Energy | Forces | Optional virial |
| --- | --- | --- | --- |
| MACE / metatomic | float32 | float32 | float32 |
| NequIP / FAIR-Chem | float64 | float32 | float32 |
| DeePMD | float64 | float32 | float64 |

Float64 models return all three fields in float64. Model weights are not cast
to obtain these native reduction dtypes. DeePMD explicitly promotes geometry
to its float64 interface precision before model preprocessing.
The NumPy `evaluate` boundary always returns ordinary float64 `PESResult`
arrays, after the explicit host/device transfer.

The current Torch dynamics implementation requires float64 and performs a
`TorchPES` type check. A float64 provider can be used without changing that file:

```python
from pyrpmd.torch_backend import TorchDynamics

engine = TorchDynamics(pes.as_torch_pes(), masses, beta, nbeads, dt)
```

The adapter borrows the provider; keep it open while the engine uses it.
`as_torch_pes()` refuses float32 providers. T06 can also consume the public
`evaluate_tensor` contract directly. This task does not change
`torch_backend.py` or introduce a new device checkpoint format.

## Run identity and restart

New provider names, model paths and all model-specific parameters remain in
the scientific configuration. Precision, head, type mapping, unit scales,
electronic state and cell choices therefore participate in the existing run
identity. The adapter does not strip physical options from the fingerprint.
The behavior and identities of existing provider inputs are unchanged.

The providers retain no dynamical/RNG state or coordinate history between
calls. Cached composition layouts and geometry tensors are reconstructible;
they do not require checkpoint fields. The existing checkpoint writer,
atomic replacement and RNG serialization are unchanged. Passing an adapter
regression does not establish trajectory convergence or a trustworthy trained
potential for a particular chemistry.

The public `pyrpmd.run` regression exercises a locally exported random MACE
model with P=8, three O/D/T trajectories, a PILE-L thermostat, and full/no PBC.
It checks K=2 batches (16 frames plus an 8-frame tail), then interrupts a K=1
run and resumes with K=3 (24 frames). Final coordinates, momenta and recorded
observables agree with the uninterrupted reference at `rtol=1e-10`,
`atol=2e-12`; clocks and per-trajectory RNG states match exactly. Thus changing
the execution grouping is checked through actual inference and checkpoint
orchestration as well as the individual provider interface.

## Reproduce correctness and performance checks

Run the optional framework tests in their matching environments:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_batched_ml_contract.py tests/test_batched_ml_mace.py \
  tests/test_batched_ml_metatomic.py tests/test_batched_ml_workflow.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_batched_ml_nequip.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_batched_ml_fairchem.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_batched_ml_deepmd.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_batched_ml_metatomic.py
```

For the real NequIP AOTI checks, use the `nequip-batched-aoti` extra, a working
C++ compiler and the native runtime's temporary directory (including `/tmp`
on Linux):

```bash
AOTI_RUNTIME_CHECK_INPUTS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q tests/test_batched_ml_nequip.py
```

The test compiles one 54-parameter random float64 model locally on CPU and
checks B=1/2/3/5, full/no/partial PBC, D/T aliases, and energy/force/virial
against eager single-frame results. It also checks that B=6 and a zero-edge graph are
rejected by the fixture's native bounds. Those bounds belong to that test
artifact, not to every NequIP model. Missing optional compiler/runtime support
is skipped; export, compilation, loading or numerical failures in a supported
environment fail the test.
Float32 eager inference retains its separate tolerance tests; float32 and
CUDA AOTI execution have not been validated here.

For singleton batches, the fixture uses Torch 2.13's experimental compile-time
`torch.fx.experimental._config.patch(backed_size_oblivious=True)` while keeping
`aot_inductor.check_lowerbound=True`. Without that opt-in, the tested compiler
specializes the native frame lower bound to 2 even with an explicit export
`Dim(min=1)`. Production providers never change compiler flags or environment
variables. Users must compile for their required shape ranges and validate
their own artifact; a valid eager input is not automatically within compiled
bounds.

The tests use `pytest.importorskip`, random initialized small models, local
export roundtrips, batch sizes 1/2/5, frame permutations, periodic/nonperiodic
systems, and H/D/T. Float64 frame/batch comparisons use `rtol=1e-10`; explicit
float32 tests document their tolerance. Independent force/strain finite
differences and the metatomic analytic reference check scientific signs and
units separately from batch equivalence. Shape, precision, device, electronic
state, closed-provider and failure-without-retry checks guard the public API.

The float32 acceptance tolerances are explicit below. Absolute tolerances
use the corresponding atomic unit (hartree, hartree/bohr or hartree for virial).

| Provider and checked quantity | Relative tolerance | Absolute tolerance |
| --- | ---: | ---: |
| MACE energy, forces and periodic virial | `3e-5` | `3e-9` |
| MACE virial in the additional large periodic-image test | `3e-5` | `3e-8` |
| NequIP / FAIR-Chem energy and forces | `3e-5` | `3e-7` |
| DeePMD energy, forces and virial | `3e-5` | `3e-8` |
| Metatomic energy | `3e-6` | `3e-8` |
| Metatomic forces | `3e-5` | `3e-8` |

The independent float32 DeePMD finite-difference check uses a larger step
(`3e-3` bohr or strain) and `rtol=3e-3`, `atol=2e-6` to resolve derivatives
above network quantization noise. This does not relax the frame/batch
consistency tolerance in the table.

The additional MACE periodic-image case has near-zero virial components.
Its maximum absolute difference from ASE is `2.24e-8` hartree. The published
and refined graph builders produce bit-identical energy, forces and virial
on that case; the larger absolute tolerance describes existing float32
rounding, not a precision change in the copy optimization.

The latest refinement validation on 2026-11-02 produced these summaries.
Each row uses its matching isolated framework environment listed above;
NequIP is deliberately checked on two Torch versions.

| Test files | Observed pytest summary |
| --- | --- |
| Common/MACE/metatomic/public workflow plus the six existing files below | `210 passed, 12 skipped, 98 warnings in 38.56s` |
| `test_batched_ml_nequip.py`, Torch 2.7.1 | `37 passed, 5 skipped in 9.52s` |
| `test_batched_ml_nequip.py`, Torch 2.13.0, real AOTI | `40 passed, 2 skipped, 16 warnings in 31.19s` |
| `test_batched_ml_fairchem.py` | `14 passed, 1 skipped, 1 warning in 9.07s` |
| `test_batched_ml_deepmd.py` | `45 passed, 2 skipped, 61 warnings in 7.67s` |

The six existing files in the first row are `test_potential_adapters.py`,
`test_potential_input_v2.py`, `test_potential_energy_identity.py`,
`test_trajectory_batch.py`, `test_ase_trajectory_batch.py` and
`test_python_potential_rebinding.py`. Its skips are five CUDA cases and seven
PySCF/tblite cases. NequIP skips four native AOTI cases on Torch 2.7.1, or one
upstream-unsupported TorchScript compile case on Torch 2.13.0, plus CUDA on both.
The FAIR-Chem skip requires two GPUs; DeePMD skips both CUDA precision cases.
Framework/TorchScript deprecation warnings remain visible. No CUDA execution
was available for this validation. The graph benchmark CLI also passed a
separate smoke run after being packaged as a reusable example.

### Existing-suite regression record

Before these refinements, the full repository suite ran in separate baseline
and T04 worktrees with
the same CPU environment and each worktree's own `src` on `PYTHONPATH`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
```

| Source snapshot | Raw pytest summary |
| --- | --- |
| Baseline `8cc402f` | `14 failed, 8763 passed, 205 skipped, 48 warnings, 486 subtests passed in 2540.40s` |
| T04 implementation/tests `9956bc6` | `14 failed, 8838 passed, 210 skipped, 135 warnings, 486 subtests passed in 2110.09s` |

The failed-node sets were identical. All 14 failures occurred in existing
i-PI socket tests because the restricted environment rejected socket creation
with `PermissionError: [Errno 1] Operation not permitted`. With local socket
access enabled, the entire `tests/test_ipi_clients.py` file passed in both
worktrees (baseline `19 passed in 3.73s`, T04 `19 passed in 3.70s`). The two
other affected nodes, `PeriodicPotentialTests::test_ipi_real_cell_inverse_and_virial_wire`
and `SocketTests::test_ipi_protocol`, also passed (baseline `2 passed in 1.33s`,
T04 `2 passed in 1.21s`). Socket implementation and test files were unchanged.
The restricted full runs therefore are recorded with their actual failures;
the targeted reruns establish the environment cause without hiding it.

The six existing adapter/input/identity/trajectory-batching test files also
returned `170 passed, 7 skipped` before and after the implementation.
Missing optional packages and CUDA cases remain skipped, not counted as passes.

At `ef7c468`, the random-model helper was adjusted to seed only the CPU
generator, preserving accelerator RNG state. Independent checks for float64
and float32 confirmed all 75 model state tensors and 1,968 parameters were
bit-identical to the recorded implementation, while CPU RNG/default dtype were
preserved and no accelerator seed entry point was called. Thus the recorded
CPU model and benchmark data are unchanged by that helper refinement.

### Original whole-step throughput benchmark

The original real MACE throughput benchmark includes graph construction,
energy, forces, virial and the NumPy boundary on both paths. The recorded
numbers below predate the graph-copy refinement; they remain the original
acceptance record, rather than a new whole-step measurement:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python examples/potentials/benchmark_batched_ml.py \
  --device cpu --output batched-ml-timing.json
```

It uses an untrained two-interaction `ScaleShiftMACE` with 1,968 parameters,
float64, full PBC, two warmups, three repetitions and one Torch/BLAS/OpenMP
thread. It checks that the reference makes `P` model calls and the batched
provider makes exactly one. The following are median milliseconds per PES
step measured in the shared x86_64 Linux CPU environment (nine visible logical
CPUs; physical CPU model was not exposed). No GPU was available.

| Atoms N | Beads/frames P | ASE per frame, ms | Whole batch, ms | Reference / batch |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 8 | 341.283 | 120.286 | 2.837 |
| 64 | 32 | 948.267 | 335.927 | 2.823 |
| 64 | 64 | 3978.790 | 1418.272 | 2.805 |
| 512 | 8 | 729.590 | 693.928 | 1.051 |
| 512 | 32 | 3956.579 | 5984.433 | 0.661 |
| 512 | 64 | 18782.014 | 17632.729 | 1.065 |

Batching was slower for N=512/P=32 in this measurement. Large cases also show
substantial timing variability; the recorded JSON retains every repetition.
The largest relative norm differences were about `1.5e-16` for energy,
`8.7e-15` for force and `1.2e-15` for virial. These results establish numerical
agreement for the tested random model; they do not measure a trained production
model, CUDA throughput, physical convergence or a universal speedup.

The complete hardware/model/settings/timings record is
[`batched_ml_cpu_benchmark.json`](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/potentials/batched_ml_cpu_benchmark.json).
Use `--device cuda` on a suitable CUDA installation to measure that machine;
the script synchronizes CUDA when timing it and reports the GPU model.

### Graph-construction refinement benchmark

The additional CPU benchmark isolates graph construction and compares the
published `ef7c468` operations with the refined provider. It checks every graph
field for bit equality and requires the final `edge_index` to be contiguous.
The reference graph operations are included in the script for reproduction:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python examples/potentials/benchmark_mace_graph.py --output mace-graph.json
```

This uses the same random model and periodic geometry generator as the
whole-step benchmark, one CPU thread, three warmups and 15 repetitions in
alternating order. Other task agents paused CPU-heavy work for this final
matrix; shared-host timing noise remains possible. Values are median
milliseconds for the graph stage, excluding inference, autograd and public
boundary validation:

| Precision | Atoms N | Frames B | Published graph, ms | Refined graph, ms | Published / refined |
| --- | ---: | ---: | ---: | ---: | ---: |
| float64 | 64 | 8 | 2.172 | 2.107 | 1.031 |
| float64 | 64 | 32 | 8.414 | 8.084 | 1.041 |
| float64 | 64 | 64 | 23.999 | 22.544 | 1.065 |
| float64 | 512 | 8 | 12.978 | 12.598 | 1.030 |
| float64 | 512 | 32 | 48.766 | 49.764 | 0.980 |
| float64 | 512 | 64 | 103.474 | 105.922 | 0.977 |
| float32 | 64 | 8 | 2.750 | 2.571 | 1.069 |
| float32 | 64 | 32 | 10.674 | 9.930 | 1.075 |
| float32 | 64 | 64 | 20.463 | 19.264 | 1.062 |
| float32 | 512 | 8 | 12.710 | 12.412 | 1.024 |
| float32 | 512 | 32 | 55.014 | 53.021 | 1.038 |
| float32 | 512 | 64 | 109.966 | 106.134 | 1.036 |

The N=512 float64 cases at B=32 and B=64 are 2.0% and 2.4% slower in this
final measurement. The other ten cases have ratios of 1.024–1.075. Timing
varies across runs on the shared host; these are the final script's complete
results, including both slower cases.

At B=32, `aten::clone` calls fall from 33 to 1 and `aten::stack` calls from
32 to 0. The sum of positive `self_cpu_memory_usage` over Torch profiler
operator groups falls by about 11.6% for float64 and 24.5–24.6% for float32 at both atom
counts. This allocation metric excludes Vesin's native allocations and does
not measure peak process memory. All 12 graph comparisons are bit-identical.
The allocation counts establish reduced intermediate storage and copy calls;
the timings do not establish a uniform graph-stage, end-to-end or GPU speedup.

All repetitions, edge counts, allocation metrics and version details are in
[`mace_graph_cpu_benchmark.json`](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/potentials/mace_graph_cpu_benchmark.json).

### T19 CUDA validation matrix (2026-11-03)

CUDA performance is **not yet measured**. The T19 environment has no NVIDIA
device or CUDA runtime; the later T04 stream changes and PR #84 were also
absent from the main revision used for this work. The existing CPU table
above is not evidence for those CUDA changes.

The new [T19 measurement driver](../benchmarks/efficiency/t19_cuda.py) covers
all five named providers at every combination of `P=8,32,64` and
`N=64,512,4096`. Supply local model files with their SHA256 hashes, explicit
float64 precision and either representative NPZ structures or an explicitly
labeled generated lattice. See the script's `--help` for the manifest schema.
It never downloads weights or changes the requested model/device/precision.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/efficiency/t19_cuda.py --device cuda:0 --suite ml \
  --providers mace-batched --manifest models.json \
  --output t19-mace-cuda.json --markdown t19-mace-cuda.md
```

Use an explicit `--providers` subset when frameworks require incompatible
environments, and collect one report per environment. A requested subset
can pass without marking the complete five-provider T19 matrix as complete.

Each configured model is compared with its CPU float64 evaluation on the
same coordinates and cell. The driver records energy/force/virial errors,
frame-permutation and changing-batch checks, nondefault CUDA stream checks,
median complete NumPy-boundary and resident-tensor timings, all raw
repetitions, peak device memory, framework versions and the actual imported
source files. Device-specific artifacts that cannot supply the CPU reference
are explicit unsupported cases. GPU synchronization surrounds every timed
CUDA evaluation. The resident-tensor timing excludes the initial coordinate upload;
the NumPy timing includes the public boundary transfers.

No model, missing CUDA, unsupported precision and out-of-memory errors leave
explicit rows and a nonzero exit code. A partial matrix cannot be reported
as complete T19 CUDA acceptance. The [T19 validation record](t19-performance.md)
describes the available hardware, missing measurements, workflow comparison
and CPU force-engine results. Short correctness/performance checks keep
`physical_converged=false`.

## API references

- [MACE batch construction and model evaluation](https://github.com/ACEsuit/mace/blob/v0.3.17/mace/tools/multihead_tools.py)
- [NequIP compilation and target contracts](https://nequip.readthedocs.io/en/latest/guide/getting-started/workflow.html)
- [NequIP Torch version compatibility](https://nequip.readthedocs.io/en/latest/guide/reference/pytorch_compatibility.html)
- [FAIR-Chem native batch inference](https://facebookresearch.github.io/fairchem/batch-inference/)
- [DeePMD PyTorch model API](https://docs.deepmodeling.com/projects/deepmd/en/v3.2.0/autoapi/deepmd/pt/model/model/index.html)
- [DeePMD native interface and network precision](https://docs.deepmodeling.com/projects/deepmd/en/v3.2.0/model/precision.html)
- [Metatomic exported model interface](https://docs.metatensor.org/metatomic/latest/torch/reference/models/export.html)
- [Metatomic energy and strain-gradient conventions](https://docs.metatensor.org/metatomic/latest/quantities/energy.html)
- [Metatomic's recommended scientific citation](https://docs.metatensor.org/metatomic/latest/cite.html)
- [Vesin device neighbor lists](https://luthaf.fr/vesin/latest/torch-api.html)
- [Vesin 0.6.2 CUDA kernels and synchronization order](https://github.com/Luthaf/vesin/blob/v0.6.2/vesin/src/vesin_cuda.cpp)
- [Vesin's pinned gpulite stream defaults](https://github.com/metatensor/gpu-lite/blob/6c221167aeca7e113ea6f27e9a2e12391bdb189a/gpulite/gpulite.hpp)
- [CUDA default and nonblocking stream ordering](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html)
