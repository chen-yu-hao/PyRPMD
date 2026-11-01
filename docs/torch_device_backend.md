# Device-resident Torch dynamics: experimental Python API

This is an opt-in backend for the sum-beads Hamiltonian with real masses and
atomic units. It is **not yet connected to `rate`, `drpmd`, or `pimd` input
files**, existing workflow checkpoints, constrained recrossing, or collective
variable/bias evaluation. Those workflows continue to use their existing
NumPy/native paths. Do not add unsupported input keys expecting GPU execution.

The backend requires the existing `pyrpmd[gpu]` extra. CPU Torch is useful for
correctness testing; request `device="cuda"` explicitly for GPU execution.
The implementation supports float64 on CPU and CUDA. It does not silently
reduce precision, choose a different device, or fall back to ASE.

## What runs on the device

`TorchState` owns `q` and `p` tensors with shape `[B,P,N,3]`: trajectories,
beads, atoms, Cartesian components. The positions, momenta, real-to-complex
normal-mode transforms, ring propagation, PILE-L/TRPMD noise and momentum
updates, RPC projections, and force accumulation stay on the selected device.

Each additive potential component receives a single batch of `[B*Pc,Nc,3]`
coordinates, where `Pc` is that component's contracted bead count and `Nc` is
its atom count. There is no trajectory/bead loop around the model. The model
must promise independent frames. This does not mean interactions between
atoms within a frame must be independent.

The backend itself performs no per-step `.cpu()`, `.numpy()`, or `.item()`.
It accumulates finite-status checks on the device and reads one scalar per
`propagate` block. Initial validation, RNG state capture/rollback at block
boundaries, and explicit snapshots are outside that guarantee. A user model
can still introduce its own host transfers; its adapter must also remain
on-device. Use appropriately sized blocks rather than repeatedly calling
`step` when a scalar synchronization per step would be costly.

## Minimal example

```python
import torch
from pyrpmd.torch_backend import TorchPES, TorchState, TorchDynamics

class ExampleEnergy(torch.nn.Module):
    metadata = {"batch_independent": True}

    def forward(self, q):
        # Replace with a model accepting independent [frames, atoms, 3].
        return 0.5 * q.square().sum(dim=(1, 2))

pes = TorchPES(ExampleEnergy(), device="cuda")
# All numbers below are in atomic units, including masses, beta, and time.
dynamics = TorchDynamics(
    pes, masses=[1836.15, 3672.3], beta=1000.0, nbeads=32, dt=1.0,
    integrator="cayley-bcocb", thermostat="pile-l", gamma_centroid=0.001, seed=17,
)
q = torch.zeros((16, 32, 2, 3), dtype=torch.float64, device=pes.device)
state = TorchState(q, dynamics.sample_momenta(16))
dynamics.propagate(state, 100)
energy_on_device = dynamics.energy(state)  # One sum-beads energy per trajectory.
output = state.snapshot()                 # Explicit host-copy boundary.
```

`TorchState.from_arrays` uploads existing arrays once and accepts either a
single `[P,N,3]` ring or `[B,P,N,3]`. `snapshot()` returns independent NumPy
arrays plus the clock. It is not a workflow checkpoint: it does not include
model/configuration identity or the thermostat generator state. Exact
stochastic restarts need `dynamics.generator.get_state()` / `set_state()` in
addition to a state snapshot, plus reconstruction of identical dynamics.

`propagate` commits the entire block atomically. A model exception or any
nonfinite energy/force/state rolls back coordinates, momenta, clock, and the
engine RNG. Diagnostic evaluation counters record attempted calls and are
not rolled back. No force cache survives the block, so external coordinate
changes cannot leave a stale endpoint force. Providers must be deterministic
position-only models; arbitrary internal model state cannot be rolled back. State construction owns
contiguous tensors; replacing live q/p with overlapping or noncontiguous views
is rejected before model evaluation, RNG consumption, or state mutation.

## Explicit BCOCB sampling

Use `integrator="cayley-bcocb", thermostat="pile-l"` to select the BCOCB
splitting. The existing defaults and `normal-mode-bab` / `cayley-bcb` retain
their previous outer-thermostat behavior; no old input is silently reinterpreted.
For the full step `dt`, the new map is

```
B(dt/2) sqrt(cay(dt*A)) O(dt) sqrt(cay(dt*A)) B(dt/2).
```
The principal square root is Eq. (17) of Korol et al., JCP **152**, 104102
(2020), DOI: 10.1063/1.5134810. It is **not** `cay(dt*A/2)`.
The middle O step uses one independent real bead-space Gaussian field and
full-step OU coefficients before transforming back. State and transforms
stay on the selected device. PILE-L damps the centroid; TRPMD does not.

At a stable physical-force step, BCOCB has the exact harmonic **position**
marginal for every bead count. For a harmonic stiffness `k` and mass `m`,
`dt*sqrt(k/m) < 2` is still required. Arbitrary nonlinear potentials need
explicit timestep-convergence and equilibration checks; no automatic Hessian
bound or universal stability guarantee is provided.

**Endpoint momenta are not an exact Maxwell distribution.** In the scalar
harmonic case their stationary variance is
`m*P/beta * (1-dt**2*k/(4*m))`. Use position-based primitive or centroid-virial
quantum estimators, not `p**2/(2*m)`, for thermodynamic kinetic energy.
`energy()` still returns the mechanical sum-beads Hamiltonian, not a quantum
thermal-energy estimator. Before launching a physical RPMD trajectory from
sampled positions, independently refresh real-mass Maxwell momenta with
`sample_momenta`, and use an unthermostatted engine. For example:

```python
# sampling_state.q has been equilibrated with BCOCB/PILE-L and checked for
# timestep and bead convergence. Use a separate RNG/engine for the launch.
production = TorchDynamics(pes, masses, beta, P, dt, integrator="cayley-bcb",
                           thermostat=None, seed=23)
production_state = TorchState(sampling_state.q,
                              production.sample_momenta(sampling_state.q.shape[0]))
```

RPC is supported at `inner_steps=1`: both force groups appear in the same B
half-kicks and refer to the explicit **contracted** Hamiltonian. This does not
undo contraction bias. `cayley-bcocb` rejects `inner_steps>1` rather than claiming
a validated BCOCB/MTS composition. With `thermostat=None`, its two roots
collapse to the existing full BCB step, without extra FFTs or RNG consumption.

See [independent verification and repeated observables](torch_bcocb_validation.md).
The new splitting is available in the Torch Python API only; workflow input,
checkpoint, constraints, CUDA and trained-chemistry acceptance gates remain.

## Model contract and existing Torch providers

A model returns energies `[F]`/`[F,1]`, `(energies, forces)`, or a mapping with
`energy`/`energies` and optional `forces`. Energy-only models use coordinate
autograd; direct forces must be **negative** energy gradients. Returned tensors
must retain float64 and the input device. Do not use `torch.inference_mode()`:
coordinate derivatives are needed, even when model parameters are not trained.
`torch.no_grad()` around propagation is supported.

For models using angstrom/eV, set `length_factor` to bohr per angstrom and
`energy_factor` to hartree per eV. Model inputs are divided by `length_factor`;
energies are multiplied by `energy_factor`, and forces by
`energy_factor / length_factor`. Existing unit factors can be reused without
restating constants:

```python
from pyrpmd.torch_backend import TorchPES

# legacy_provider is an existing module-backed optional.TorchPotential.
# Construct once, then pass this object to TorchDynamics.
pes = TorchPES.from_provider(legacy_provider, batch_independent=True)
```

The bridge rejects NumPy factory delegates and closed providers. Explicit
independence cannot override a declared coupled/stateful model. The original
model is shared and moved into eval mode; do not concurrently train it or
mutate the original provider's cell. Direct tensor evaluation validates
shape/device/precision; block propagation provides finite-value validation.
A direct `evaluate_tensor` caller must check result finiteness itself.

A fixed periodic cell is supported with `periodic=True`, a cell in bohr, and
three boolean PBC flags. The model signature becomes `forward(q, cell, pbc)`
with cells `[F,3,3]` in model length units and flags `[F,3]`. `virial="provided"`
accepts a model virial or tensile-positive stress; `virial="autograd"`
differentiates frame-local strain. RPC returns no pressure/virial estimator.
There is no barostat or variable-cell dynamics in this backend.

**ASE calculators are not tensor models.** MACE, NequIP, and other graph-based
models need model-specific wrappers that build batched neighbor graphs on the
device, preserving periodic-image and graph boundaries, energy conventions,
and units. This change does not provide those wrappers and does not make the
existing ASE adapters batched. It provides the device-native contract they
can target without returning to NumPy.

## RPC, regional components, and MTS

An RPC term uses the same low-frequency projection convention as the existing
`RingPolymerContraction`, including the cosine terminal mode for even `Pc<P`.
For a projection `T` from `P` to `Pc`, its Hamiltonian contribution is

```
U(q) = (P/Pc) * sum_k V((T q)_k)
F(q) = (P/Pc) * T.T @ F_model(T q)
```

The force lift follows the derivative of the approximated Hamiltonian. For
`Pc=P`, coordinates and forces take an exact identity path. For `Pc<P`, only
the sum of the returned bead energies is meaningful; contracted energy is
distributed uniformly over `P` beads.

Regional terms require **an additive energy decomposition**. `atoms` selects
all coordinates on which that component depends. It does not mask a full
model's forces after evaluation. For example:

```python
from pyrpmd.torch_backend import TorchRPCTerm, TorchSplitPotential, TorchDynamics

P = 32
# h_energy, metal_energy, and coupling_energy are distinct TorchPES objects.
# They must sum to the intended Hamiltonian; the names alone do not establish
# such a decomposition. Coupling remains full-system and full-resolution.
fast = [
    TorchRPCTerm(h_energy, P, P, atoms=[0]),
    TorchRPCTerm(coupling_energy, P, P),
]
slow = [TorchRPCTerm(metal_energy, P, 4, atoms=[1, 2, 3])]
split = TorchSplitPotential(fast, slow)
dynamics = TorchDynamics(split, masses, beta, P, dt,
                         inner_steps=4, thermostat=None)
```

This reduces the **potential evaluation resolution** of the metal component;
all atoms still have `P` dynamical beads and real-mass ring springs. It is not
a general mixed-per-atom-bead discretization of an arbitrary many-body ML PES.
Interactions with excluded atoms must remain in another component. When using
a cheap full-system reference, the slow model must be the correction
`V_target - V_reference`, not another full target energy added on top.

`dt` is the outer timestep. With `M=inner_steps` and `h=dt/M`, the reversible
map is `Bslow(dt/2) [Bfast(h/2) A(h) Bfast(h/2)]^M Bslow(dt/2)`.
An `S`-outer-step block uses `S*M+1` fast-group evaluations and `S+1` slow-group
evaluations. Every term in a group receives its own single flattened batch.
`inner_steps>1` requires an explicit slow group; no forces are silently
classified. The two MTS-compatible free-ring maps are `normal-mode-bab` and
`cayley-bcb`. The separate `cayley-bcocb` sampling option requires `inner_steps=1`.

No thermostat is applied by default. `pile-l` thermalizes all normal modes,
with the requested centroid friction; `trpmd` leaves the centroid undamped.
For `normal-mode-bab` and `cayley-bcb`, symmetric OU half-steps bracket the
deterministic outer step. BCOCB instead puts one full OU step in the middle. Noise is sampled
in real bead space and transformed, retaining the correct centroid/Nyquist
variance. The OU bath and `sample_momenta()` use variance `m*P/beta`; this is
not a claim that finite-step BCOCB endpoint momenta are Maxwellian. Sampling steps are not a substitute
for unthermostatted rate children. Converge outer/inner timestep, contracted
beads, and the decomposition against full-resolution observables: RPC is a
Hamiltonian approximation and MTS can introduce resonance/accuracy problems.

## Validation and benchmark

Run:

```sh
PYTHONPATH=src pytest -q tests/test_torch_backend.py tests/test_torch_bcocb.py
PYTHONPATH=src python benchmarks/torch_bcocb_validation.py --section all --output bcocb.json
PYTHONPATH=src python benchmarks/torch_device_backend.py --device cpu
PYTHONPATH=src python benchmarks/torch_device_backend.py --device cuda
```

Tests cover independent NumPy normal-mode parity, odd/even/classical bead
limits, reversibility, timestep convergence, finite-difference RPC gradients,
regional force scattering, batching equivalence, model call counts, thermal
variance/covariance, finite-P harmonic configurational sampling, RNG rollback, and absence of hot-loop host conversions.
CUDA parity tests skip when CUDA is unavailable; they are not a GPU benchmark.

The benchmark uses a synthetic dense network, not a trained chemistry model.
It verifies batched/frame-loop trajectory agreement and records synchronized
timing, expensive-model call counts, and evaluated-frame counts. Its frame
loop is not a benchmark of ASE itself. The RPC/MTS case uses a different,
explicitly marked approximate Hamiltonian; its trajectory is not expected to
match the uncontracted case. Report hardware and real-model measurements
before making an application-level speedup claim.

## Remaining integration work

Before exposing this through production inputs, connect validated acceleration
settings to the actual rate/dRPMD/PIMD engine factories; implement device-aware
biases/constraints/observations, thermostat-free rate-child routing, checkpoint
identity/RNG persistence, and reject unsupported combinations before execution.
Model-specific graph batching, full-workflow CPU/GPU parity, real-model GPU
benchmarks, and scientific RPC/MTS convergence studies remain necessary.

## Implementation references

- Existing PyRPMD `dynamics.CayleyIntegrator`, `potentials.contraction`, and
  `multitime.MTSDynamics` at base commit
  `2f907ba96d22cb134198715e398b1d9f8d83d929` establish the mass, transform,
  contraction, and splitting conventions used here.
- PyTorch real FFT normalization and inverse-length convention:
  https://docs.pytorch.org/docs/2.9/generated/torch.fft.rfft.html
- PyTorch coordinate-gradient API:
  https://docs.pytorch.org/docs/2.9/generated/torch.autograd.grad.html
- i-PI component-wise contraction overview:
  https://docs.ipi-code.org/introduction.html


## RPMDrate normal-mode compatibility

Normal modes are a coordinate representation; BCOCB is a time splitting.
They are compatible, not alternative algorithms. The `cayley` in the
integrator name identifies the free-ring map, **not** bead-space integration.
Use `normal_mode_layout="rpmdrate"` independently of the integrator to select
RPMDrate's real FFTW-halfcomplex layout. With BCOCB, the C/O/C substeps then
remain in normal-mode space without intermediate bead-space round trips.
The default `normal_mode_layout="rfft"` retains the previous arithmetic.

See [the compatibility report](rpmdrate_normal_modes.md) for the pinned
upstream source, normalization, mode/noise metric, CPU/native comparisons,
limitations, and the runnable [sampling-to-production example](../examples/ensembles/torch_normal_modes.py).
The new layout is a Torch Python API option, not a new `.inp` routing key.
