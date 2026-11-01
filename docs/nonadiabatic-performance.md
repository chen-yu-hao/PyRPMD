# Nonadiabatic electronic kernels and trajectory batches

PyRPMD exposes three backends for the exact fixed-nuclear-coordinate mapping
flow, plus a Python NRPMD driver that batches independent trajectories and
beads together. The backend choice and trajectory batching are separate
decisions. Measure the complete integration step for the intended molecular
PES: a faster electronic contraction alone need not make a trajectory faster.

The reproducible
[benchmark script](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/nonadiabatic/electronic_backends.py)
checks numerical agreement before reporting time. Its cases include a synthetic
complex Hermitian four-state model and the published
[NaH2 5G molecular PES](molecular-nonadiabatic.md). The initial arrays are copied
identically into each compared path; this is a numerical implementation and
performance comparison, without a claim of physical convergence.

## What each backend executes

| Path | Electronic work | Array location and transfers | Current scope |
| --- | --- | --- | --- |
| `electronic_backend="numpy"` | NumPy Hermitian diagonalization, phases and integrated force contractions | Host NumPy arrays | CPU reference with float64/complex128 |
| `electronic_backend="native"` | NumPy Hermitian diagonalization followed by C++/OpenMP phases and integrated force contractions | Host NumPy arrays | Native contraction kernel; the eigensolver and PES are separate |
| `electronic_backend="torch"` through `electronic_mapping_flow` or the dynamics driver | Torch Hermitian diagonalization and tensor contractions | NumPy inputs are converted to tensors and results returned to NumPy; CUDA includes transfers in both directions | Optional CPU/CUDA electronic substep inside a host-based driver |
| `torch_electronic_mapping_flow` | The same Torch tensor flow | Complex128 tensors supplied and returned on one CPU/CUDA device | A resident electronic primitive for callers that already hold the inputs on that device |

Every path propagates both the complex mapping vector and the integrated
nuclear momentum impulse. Propagating only the electronic phases would not be
an equivalent NRPMD step. The public contract is `z[..., S]`, the complete
Hermitian matrix `V[..., S, S]`, and its positive Cartesian derivative
`G[..., S, S, N, 3]`. The outputs are `z_new[..., S]` and
`delta_p[..., N, 3]`. Leading dimensions may represent trajectories and beads.
Degenerate eigenvalues use the analytic sinc limit of the integrated flow.

`native` for the nuclear free-ring propagator and `electronic_backend="native"`
are independent settings. The latter requires the compiled `_electronic`
extension. A requested unavailable backend raises an error in dynamics; it is
never silently replaced by NumPy. Check the installation explicitly:

```python
from pyrpmd.electronic_backends import electronic_backend_capabilities

print(electronic_backend_capabilities())
```

For an NRPMD CLI input, the electronic choice is explicit:

```toml
[dynamics]
dt = "0.2 autime"
native = false
electronic_backend = "native"
electronic_device = "cpu"
```

Torch is optional. The project's `gpu` extra declares the Torch dependency;
the installed Torch build and hardware must also support the requested device.
`electronic_device="cuda:0"` requires `electronic_backend="torch"` and an
available CUDA device. There is no GPU performance measurement in the local
result reported below.

### Resident tensors and the complete dynamics boundary

`torch_electronic_mapping_flow` requires complex128 tensors on one device and
returns tensors on that device. Its validation reductions can synchronize
CUDA; the benchmark also synchronizes explicitly around every timed sample.
Input conversion and final copying to CPU are outside the resident-kernel
timing, and are included in the host-array CUDA timing.

The current NaH2 provider evaluates the molecular matrix and derivatives with
NumPy on the CPU. The scalar and batched NRPMD state, free-ring step and their
transactional state copies are also host-based. Selecting a CUDA electronic
substep therefore does not make this complete molecular workflow resident on
the GPU. The benchmark reports the resident primitive separately from the
complete integration path so that transfer costs remain visible. It measures
float64/complex128 arithmetic throughout; reduced precision and autograd are
outside this comparison.

## Batching independent trajectories

The Python `BatchedNRPMDDynamics` API accepts a `BatchedNRPMDState` with nuclear
arrays `[K, P, N, 3]` and mapping arrays `[K, P, S]`, where `K` is the number of
independent trajectories. `BatchedNRPMDState.from_states(states)` creates it
from ordinary `NRPMDState` objects, and `to_states()` restores separate states.
Constructors take the same molecular provider, atomic masses, inverse
temperature, bead count, timestep and electronic state count as scalar NRPMD.

At each force evaluation, the provider sees `K*P` independent molecular frames.
The free-ring propagator retains each trajectory's own bead ring. A provider
must explicitly declare `metadata["trajectory_batch_safe"]=True`; NaH2 5G and
the standard analytic model library satisfy this contract. All trajectories in
one driver share model and integration parameters, while their phase-space
coordinates and clocks are separate. A failed lane prevents the entire batch
step from committing.

This batch API covers deterministic NRPMD propagation. Trajectory checkpointing,
sampling, process scheduling and statistical analysis remain separate workflow
operations. The timings below exercise the Python batch driver directly. The
CLI's independent-trajectory execution is described in the
[workflow guide](nonadiabatic-workflow.md), and the statistical meaning of the
observations is described in the [ensemble guide](nonadiabatic-ensembles.md).

## Reproduce the benchmark

From a checkout with its normal dependencies and compiled extension installed:

```bash
PYTHONPATH=src python benchmarks/nonadiabatic/electronic_backends.py \
  --backends numpy native torch --torch-device cpu \
  --batch 16 --beads 8 --states 4 --steps 8 --threads 1 \
  --warmups 2 --repeats 5 --kernel-inner 10 \
  --output results/electronic_backends.json
```

The default benchmark uses the Python nuclear free-ring path for both drivers,
which isolates the changes from batching and the electronic backend. To measure
the compiled free-ring path, add `--ring-native`. Multiple threads require this
flag and an OpenMP-capable build. In all cases `threadpoolctl` limits the loaded
BLAS/OpenMP pools, and the requested native thread count is recorded. Torch's
intra-op thread count is set and recorded when it is available. On a
CUDA-equipped system, replace the Torch device with `cuda:0`; unavailable
backends are recorded as unavailable in the benchmark result, with no substitute
timing.

The report has two measurement levels:

1. **Electronic flow:** the complete public mapping-flow call, including its
   validation, diagonalization, phase/force contractions and output allocation.
   PES evaluation and preparation of the fixed input matrices are outside this
   timing. A sample averages ten calls by default.
2. **Integration:** eight consecutive steps of all sixteen trajectories,
   including two PES/gradient evaluations per step, both electronic flows,
   free-ring propagation and the driver's transactional copies. State reset,
   driver construction, observations, checkpoint I/O and sampling are outside
   the timed interval. The scalar path loops over the same individual initial
   states that constitute the batched path.

Both levels warm up twice and report the median and every raw sample from five
timed repeats. The JSON includes `K/P/S/N`, atomic masses, timestep, inverse
temperature, seed, initial-array hashes, exact model identities, source and
compiled-extension hashes, Python/NumPy/Torch versions, loaded BLAS/OpenMP
pools, requested threads, device and backend availability. The benchmark fails
if its relevant source or compiled extension changes while it is running.

Before accepting a timing, the script compares the complex vector and impulse
against the NumPy flow. Complete integration additionally compares nuclear
positions and momenta, both mapping arrays, signed populations, complex density
matrices, energies, step indices and times against scalar NumPy propagation.
The tolerance is `atol=5e-10`, `rtol=2e-11`; all differences are retained in the
JSON. Speedup is never a pass/fail criterion. These checks test implementation
agreement for the recorded examples, not the accuracy of the NRPMD approximation.

## Recorded local measurements

The checked-in
[2026-11-01 result](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/nonadiabatic/electronic_backends_local_2026-11-01.json)
was actually executed on Linux x86-64, Python 3.12.14, NumPy 2.3.5 and OpenBLAS
0.3.30. The environment exposed nine logical CPUs, but the measurement requested
one thread and recorded one thread in each loaded BLAS/OpenMP pool. A specific
CPU model was not exposed. Torch and CUDA were unavailable in this environment.
The nuclear free-ring path was Python for every row.

All rows use `K=16`, `P=8`; the synthetic complex LVC case has `S=4`, `N=1`, and
the molecular NaH2 case has `S=2`, `N=3`. Each row below is the median wall time
for the stated measurement, not time per electronic state or per trajectory.

| Electronic flow, 128 frames | NumPy | Native contraction |
| --- | ---: | ---: |
| Synthetic complex LVC | 0.729 ms | 0.616 ms |
| Published NaH2 5G matrices/gradients | 0.255 ms | 0.198 ms |

| Eight integration steps, all 16 trajectories | Scalar NumPy | Batched NumPy | Scalar native electronic | Batched native electronic |
| --- | ---: | ---: | ---: | ---: |
| Synthetic complex LVC | 87.59 ms | 25.57 ms | 109.35 ms | 17.95 ms |
| Published NaH2 5G | 420.85 ms | 41.22 ms | 416.34 ms | 37.72 ms |

For this NaH2 workload, batching with NumPy took approximately 10.2 times less
time than the scalar NumPy loop; batching with the native electronic contraction
took approximately 11.2 times less time. Batching reduces the number of Python
PES calls per integration step from 32 to 2 while evaluating the same 256
molecular frames. This makes substantial use of the portable NaH2 provider's
vectorized matrix and derivative evaluation.

The four-state synthetic case also demonstrates why a kernel-only measurement
is insufficient: its native scalar integration was slower than the scalar
NumPy integration in this run. Dispatch, validation, small-array work and batch
size affect the complete cost. These measurements support choosing a measured
path for the intended workload; they do not establish a universal backend
ranking. Raw samples are retained because process scheduling and shared CPU
load can change timing ratios between runs.

The largest absolute difference across the checked complete-integration
observables was `5.83e-15`; the NumPy scalar and batched results were identical
in these cases. This is evidence of numerical equivalence for the recorded
initial conditions and steps. It does not certify long-time sampling,
chemical convergence, GPU performance or superiority of one nonadiabatic
dynamics approximation.
