---
orphan: true
---

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
python -m pip install -e '.[fairchem-batched,test]'
python -m pip install -e '.[deepmd-batched,test]'
python -m pip install -e '.[metatomic,test]'
```

The APIs used here were checked against these versions:

| Provider | Framework | CPU validation runtime | Model input |
| --- | --- | --- | --- |
| `mace-batched` | MACE 0.3.17, Vesin 0.6.2 | Torch 2.7.1, e3nn 0.4.4 | Local `.model`, or energy `MACE` / `ScaleShiftMACE` module |
| `nequip-batched` | NequIP 0.19.1 | Torch 2.7.1, e3nn 0.6.0 | Local `.nequip.pth`, batch-target `.nequip.pt2`, or `GraphModel` |
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

**NequIP.** The eager and TorchScript routes are tested with a locally initialized
small NequIP model. AOTInductor files must be compiled with `--target batch`;
an ASE-only artifact does not carry the required batch input contract. The AOTI
loader validates input/output fields, precision and the requested device, but
AOTI compilation and execution were not measured in the CPU acceptance run.
CUDA loading selects the requested device index explicitly even when the
artifact records only the device type. `params.energy_units_to_eV` and
`params.length_units_to_A` have the same meanings as in `NequIPCalculator`.
`params.chemical_species_to_atom_type_map` can map elements to trained type
names. CPU graph construction defaults to `matscipy`; CUDA requires
`neighborlist_backend="alchemiops"`. Requesting CPU graph construction for a
CUDA tensor evaluation raises an error instead of copying coordinates to host.

**FAIR-Chem.** Set `potential.params.task_name` to the trained chemistry domain.
The provider uses `MLIPPredictUnit.predict` on one `AtomicData` graph batch,
including its trained normalization and element references. It requires
conservative forces, compatible derivative normalizers and the general
execution mode. Direct-force/direct-stress models, Nitro, MOLE merging,
automatic compilation/precision fallback and single-atom lookup substitutions
are not part of this subset. Its strict inference settings are explicit in the
provider; the old `fairchem` path is unaffected.

**DeePMD.** The new Torch route supports a conservative, homogeneous float64
energy-model subset. It uses device-local sparse neighbors and the native
atomic energy core, then differentiates the whole batch once. This avoids the
dense neighbor-preparation intermediates in the standard Torch forward path.
The trained neighbor selection remains part of the model; unsupported models
or ambiguous truncation are rejected. Full or no PBC is supported. The
existing NumPy `deepmd` provider remains available for its wider backend support.

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
looser numerical tolerances. NequIP and FAIR-Chem natively accumulate total energy in float64
even for a float32 model; `metadata["energy_dtype"]` records that fact and the
provider preserves it. Forces remain in the model's declared precision.
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

## Reproduce correctness and performance checks

Run the optional framework tests in their matching environments:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_batched_ml_contract.py tests/test_batched_ml_mace.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_batched_ml_nequip.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_batched_ml_fairchem.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_batched_ml_deepmd.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_batched_ml_metatomic.py
```

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
| NequIP / FAIR-Chem energy and forces | `3e-5` | `3e-7` |
| Metatomic energy | `3e-6` | `3e-8` |
| Metatomic forces | `3e-5` | `3e-8` |

The new DeePMD subset explicitly rejects float32; it does not promote a
float32 checkpoint silently.

The CPU validation on 2026-11-02 produced the following summaries. Each row
uses its matching isolated framework environment listed above.

| Test files | Observed pytest summary |
| --- | --- |
| `test_batched_ml_contract.py`, `test_batched_ml_mace.py`, `test_batched_ml_metatomic.py` | `75 passed, 2 skipped, 84 warnings in 90.81s` |
| `test_batched_ml_nequip.py` | `31 passed, 1 skipped in 43.32s` |
| `test_batched_ml_fairchem.py` | `14 passed, 1 skipped, 1 warning in 38.50s` |
| `test_batched_ml_deepmd.py` | `24 passed, 1 skipped, 58 warnings in 53.42s` |

The five skipped cases require CUDA hardware; the FAIR-Chem device-selection
case requires two GPUs. Framework/TorchScript deprecation warnings remain
visible. The NequIP AOTI cases exercise metadata and layout guards with a mock
loader; they do not compile or execute an AOTI artifact.

### Existing-suite regression record

The full repository suite ran in separate baseline and T04 worktrees with
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

A final refinement of the random-model helper seeds only the CPU generator,
preserving accelerator RNG state. Independent checks for float64
and float32 confirmed all 75 model state tensors and 1,968 parameters were
bit-identical to the recorded implementation, while CPU RNG/default dtype were
preserved and no accelerator seed entry point was called. Thus the recorded
CPU model and benchmark data are unchanged by that helper refinement.

### Throughput benchmark

The real MACE throughput benchmark includes graph construction, energy,
forces, virial and the NumPy boundary on both paths:

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

## API references

- [MACE batch construction and model evaluation](https://github.com/ACEsuit/mace/blob/v0.3.17/mace/tools/multihead_tools.py)
- [NequIP compilation and target contracts](https://nequip.readthedocs.io/en/latest/guide/getting-started/workflow.html)
- [FAIR-Chem native batch inference](https://facebookresearch.github.io/fairchem/batch-inference/)
- [DeePMD PyTorch model API](https://docs.deepmodeling.com/projects/deepmd/en/v3.2.0/autoapi/deepmd/pt/model/model/index.html)
- [Metatomic exported model interface](https://docs.metatensor.org/metatomic/latest/torch/reference/models/export.html)
- [Metatomic energy and strain-gradient conventions](https://docs.metatensor.org/metatomic/latest/quantities/energy.html)
- [Metatomic's recommended scientific citation](https://docs.metatensor.org/metatomic/latest/cite.html)
- [Vesin device neighbor lists](https://luthaf.fr/vesin/latest/torch-api.html)
