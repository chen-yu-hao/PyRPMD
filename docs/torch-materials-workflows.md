---
orphan: true
---

# Tensor PIMD and RPMD transport workflows

`sampling.engine="torch"` connects the existing float64, device-resident
integrator to `pyrpmd validate`, `run`, `resume`, and diffusion `analyze`.
It is an explicit, fixed-cell workflow with an atomic checkpoint for the whole
trajectory batch. Existing inputs continue to select their original engines
and retain their scientific identities.

Install the optional `gpu` extra with the appropriate CPU or CUDA PyTorch
distribution. CPU Torch is supported for verification. CUDA execution is an
explicit request and fails if the requested device is unavailable; the
workflow never switches to CPU, float32, NumPy, or ASE after a failure.

## Running the examples

The repository includes two short CPU examples:

```bash
pyrpmd validate examples/materials/torch_pimd.inp --check-pes
pyrpmd run examples/materials/torch_pimd.inp --output runs/torch-pimd
pyrpmd resume runs/torch-pimd

pyrpmd run examples/materials/torch_diffusion.inp --output runs/torch-diffusion
pyrpmd analyze runs/torch-diffusion --recompute
```

The PIMD example samples a harmonic potential. The diffusion example is a
periodic free-particle **negative control**: its centroids move ballistically,
so the analysis withholds a diffusion coefficient. Neither short example
certifies equilibration, bead convergence, timestep convergence, or a material
transport property.

The engine-specific input is small:

```toml
[sampling]
engine = "torch"
device = "cpu"
normal_mode_layout = "rfft"
thermostat = "pile-l"
integrator = "bcocb"
replicas = 16
equilibration_steps = 1000
production_steps = 10000
sample_interval = 10

[dynamics]
integrator = "cayley-bcb"
timestep = "0.1 fs"
block_steps = 256

[output]
trajectory_storage = "stream"
chunk_frames = 128
checkpoint_steps = 1000
```

`normal_mode_layout` accepts `rfft` or the existing `rpmdrate` real packing.
Both stay on the selected device. The layout is part of the new input's
scientific fingerprint because it selects a floating-point execution path.
The public workflow retains the existing requirement that sampling and
production timesteps match.

For a module-backed Torch potential, set the provider's own `device` to the
same device as `sampling.device`. The binding checks that agreement and does
not move an already configured provider. Analytic potentials use
`sampling.device` directly. `cuda` selects the current CUDA device and records
its resolved index; use `cuda:0`, `cuda:1`, etc. to remove that ambiguity.

## Supported models and combinations

| Choice | Support |
| --- | --- |
| Free, harmonic, fixed-lattice periodic cosine | Explicit tensor implementations; energy and force parity with the existing NumPy providers is tested |
| Harmonic parameters | Scalar, Cartesian and atom-axis broadcasting; no frame-dependent parameters |
| Periodic cosine | Triclinic cells, partial PBC, atom subsets, repeats and nonperiodic confinement |
| Module-backed `TorchPotential` | Requires `batch_independent=True`, a deterministic position-only model, float64 and matching device |
| T04 batched ML providers, after PR #81 is integrated | The `as_torch_pes()` bridge is borrowed without host fallback; each provider retains its own model and dependency restrictions |
| PIMD | All-mobile Cartesian atoms, fixed cell, PILE-L with BCOCB |
| Diffusion | Same sampling route, followed by Cayley BCB or exact normal-mode BAB with no thermostat |
| NPT, fixed atoms, bond constraints, exchange/PIGS/high-order actions, friction, active learning, bias/CVs | Explicitly rejected by this workflow |
| MPI, bead workers, trajectory worker pools, legacy `trajectory_batch_size` | Explicitly rejected; the workflow owns one device batch |
| Precision and devices | float64 on CPU or explicitly requested CUDA; no automatic conversion |

The named T04 provider factories (`mace-batched`, `nequip-batched`,
`fairchem-batched`, `deepmd-batched` and `metatomic`) belong to the separate,
unmerged [PR #81](https://github.com/chen-yu-hao/PyRPMD/pull/81). This main-based
change supplies compatible bindings, not those factories. The combined source
has been tested with a real T04 base-class fixture; public inputs using those
provider names require the T04 integration as well.

Accepting a tensor bridge is not evidence that a particular trained model is
physically accurate. A provider's own strict checks may add device
synchronizations. In particular, T04's tensor interface on PR #81 checks
finite values with scalar reads; this workflow does not claim to remove those
provider-side synchronizations.

`sampling.replicas=B` fixes the `[B,P,N,3]` state and the shape of every noise
draw. Each PES invocation receives `[B*P,N,3]` in one call. Changing `B` changes
the reproducible random stream and therefore requires a new run. It is not a
scheduler tuning parameter. `parallel.threads_per_rank` controls CPU Torch
threads, and the saved runtime must match when continuing an incomplete run.

## Sampling, launch momenta and quantum estimators

The sampling map is BCOCB with PILE-L. It uses the principal square root of the
Cayley ring map around the thermostat, as in Korol et al., *J. Chem. Phys.*
**152**, 104102 (2020), [DOI: 10.1063/1.5134810](https://doi.org/10.1063/1.5134810).
`normal-mode-bab` in `[dynamics]` selects the deterministic **production** map;
it does not relabel Torch's outer-thermostat map as BAOAB.

BCOCB's harmonic position marginal is exact for the finite-bead distribution
within its physical-force stability limit. Its endpoint momenta have a
finite-step bias. Therefore, at the diffusion launch the workflow draws a
fresh real-mass Maxwell field with variance `m*P/beta` from a separate owned
generator. The new momenta and frame zero are committed atomically. A restart
cannot omit or repeat this draw. PIMD production continues with the sampling
thermostat; PIMD time is not a physical transport time.

All internal coordinates, energies, masses and clocks use atomic units and
the sum-beads Hamiltonian at inverse temperature `beta/P`. The saved
thermodynamic estimators are

\[
V_P=\frac1P\sum_j V(q_j),\qquad
K_{\rm prim}=\frac{3NP}{2\beta}-\frac{H_{\rm spring}}P,
\]

\[
K_{\rm cv}=\frac{3N}{2\beta}
-\frac1{2P}\sum_j(q_j-\bar q)\cdot F(q_j),\qquad
E_{\rm cv}=V_P+K_{\rm cv}.
\]

The mechanical ring Hamiltonian is a separate integration diagnostic. The
workflow never substitutes fictitious momentum kinetic energy for a quantum
thermal-energy estimate. Reduced centroids, centroid velocities, these five
energy fields and per-atom radii of gyration are calculated on the device,
packed and transferred once per recorded frame. Full ring coordinates and
momenta are transferred only at checkpoint boundaries.

## Atomic restart contract

`checkpoints/materials_torch.npz` is the sole authoritative commit for the
batch. It stores float64 `q/p`, an exact phase clock, the launch flag, both
generator states, the scientific fingerprint, explicit source hashes, a
runtime description, and every replica's immutable trajectory descriptor.
Trajectory chunks prepared before a failed checkpoint replacement remain
uncommitted. Resume follows only the saved descriptors and ignores orphans.

The integrity envelope verifies array checksums; the reader also validates
the phase, shapes, dtypes, frame counts, sample clocks, source files and
runtime metadata. An incomplete run additionally requires matching PyTorch
version, device, platform, thread configuration and normal-mode layout.
Deterministic algorithms are enabled during propagation and the caller's
prior settings are restored on exit. A model operation without a deterministic
implementation fails explicitly.

This is a same-environment restart contract. PyTorch does not promise bitwise
reproducibility across releases, platforms or CPU/CUDA implementations; see
its [reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
and [Generator state API](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html).
Arbitrary stateful or random model internals are outside the deterministic
position-only provider contract.

Changing execution block, checkpoint or chunk sizes through the resolved
configuration/API preserves the trajectory exactly. The original input file
itself is source-hashed under the project's existing policy: editing that
file is not the same operation as applying an execution-only override.
Device, layout, replica count, thermostat, timestep, PES and other scientific
settings remain immutable for a run.

A completed replay reconstructs derived exports without constructing a PES,
consuming RNG or rewriting the checkpoint. Diffusion reanalysis reuses the
existing analysis implementation and reads verified chunks once, retaining
only selected coordinate histories unless full return data are explicitly
requested. Legacy and native checkpoints keep their original readers; this
feature does not reinterpret or migrate them into Torch checkpoints.

## Verification and reproducible measurements

The regression suite covers both layouts, PIMD and diffusion, interruptions
before and after phase commits, failed multi-replica chunk publication,
changed execution sizes, corrupted payloads with otherwise valid checksum
envelopes, model-source changes, completed replay, public CLI execution, and
unchanged legacy scientific fingerprints.

Independent physical checks include ballistic free-particle centroid
transport, finite differences of the path action, tensor/NumPy force and
estimator comparisons, and actual PIMD workflow statistics against the exact
finite-bead harmonic distribution. For physical frequency `omega`, one
Cartesian coordinate has variance

\[
\langle q^2\rangle_P=\frac1{\beta m}
\sum_{k=0}^{P-1}\frac1{\omega^2+[2P\sin(\pi k/P)/\beta]^2}.
\]

Run the end-to-end statistical and CPU timing experiment with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
  examples/materials/torch_workflow_validation.py \
  --output torch-materials-validation.json --hardware "your CPU model"
```

The committed machine-readable results are in
`docs/validation/torch-materials-2026-11-03.json`. Timings include provider
construction, trajectory/checkpoint I/O and analysis after import warmup.
The comparison uses the existing serial legacy workflow on the same revision,
with the same scientific settings and recording schedule. Different engines
use different RNG algorithms; individual trajectories are not expected to
coincide.

The recorded CPU experiment uses an AMD EPYC 9V74 shared host with nine
available vCPUs, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and PyTorch
2.15.0+cpu. Each Torch, OpenMP and BLAS pool uses one thread. For 32 beads,
three atoms and 320 steps per replica, the medians of three alternating-order
measurements are:

| Replicas | Legacy wall / process CPU (s) | Torch wall / process CPU (s) | Wall speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.1651 / 0.1480 | 0.2021 / 0.1891 | 0.82x |
| 8 | 1.1641 / 1.0939 | 0.3858 / 0.3260 | 3.02x |
| 32 | 4.6250 / 4.3826 | 0.9840 / 0.7773 | 4.70x |

The single-replica tensor run is slower. Batching amortizes tensor and
workflow overhead in this small harmonic workload; a costly molecular model
can have a different balance. The shared host also ran verification jobs,
and these measurements are not a portable performance guarantee.

The separate statistical experiment uses 32 replicas, 2000 equilibration
steps and 8000 production steps at each bead count, with `m=2`, `k=2`,
`beta=4` and `dt=0.5` in atomic units:

| Beads | Exact finite-bead total energy (hartree) | Measured total energy (hartree) | Largest relative error among V, Kcv and Ecv |
| ---: | ---: | ---: | ---: |
| 8 | 1.51180472 | 1.51412930 | 0.259% |
| 16 | 1.54454505 | 1.54562897 | 0.111% |
| 32 | 1.55309008 | 1.55451861 | 0.137% |

Each estimate is within six between-replica standard errors of its
finite-bead reference. The continuum quantum total energy is 1.55597208
hartree; it is a distinct reference and is not substituted for the finite
discretization when assessing the sampling map.

CPU measurements do not establish a GPU speedup, and harmonic validation does
not establish convergence for a molecular PES. All workflow summaries retain
`converged=false` and `physical_converged=false` pending independent numerical,
sampling and physical checks.
