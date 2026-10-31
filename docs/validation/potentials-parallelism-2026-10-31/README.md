# Potentials and parallelism: implementation and verification

Date: 2026-10-31. Base: `76dd6ad24adc6c244f775857987cd2f0e119845f`.
The branch is `feat/potentials-parallelism-20261031`. Numerical example outputs,
runtime versions and SHA256 hashes of the reviewed source/test/example files are
in [`evidence.json`](evidence.json).

## What changed

- C ABI v2 adds row-vector periodic cells, boolean PBC flags and optional physical
  virials to one-call batched compiled PES evaluation. V1 retains its original
  symbols and contract. The new example compiles as strict C11 and the header is
  included in both source and wheel distributions.
- Torch accepts an explicit `forward(q, cell, pbc)` contract and computes virials
  from joint coordinate/cell strain derivatives or model stress/virial output.
- MACE, NequIP and fairchem reuse their official ASE calculators. DeePMD uses
  native `DeepPot` batches. These adapters preserve backend precision rather
  than claiming all models run in float64.
- tblite provides on-the-fly GFN1/GFN2/IPEA1-xTB, including periodic geometry and
  its physical virial conversion. PySCF provides molecular RHF/UHF/RKS/UKS with
  analytic forces, optional density fitting and explicit SCF convergence checks.
- One i-PI server accepts a configurable persistent pool of clients, dispatches
  beads concurrently and returns complete ordered results. Failed exchanges
  poison and close the pool. Close/accept and rebind/evaluate races are covered.
- Dynamic MPI uses one coordinator and refills whichever trajectory group is
  free. dRPMD samples persistent parent chains first, then queues individual
  collision trajectories with their existing global IDs and checkpoints.

## Verification scope

### Compiled ABI and Torch

`tests/test_cabi_v2.py` compiles actual libraries and exercises all nine strain
components, force differences, triclinic and partial-PBC cells, heterogeneous
batch cells, frame ordering, optional virials, malformed metadata, per-frame and
batch failures, v1 fallback, identity validation and context lifetime. A v2
function pointer cannot be exported as a v1 native descriptor.

`tests/test_torch_periodic.py` uses actual CPU PyTorch. It checks force and strain
finite differences, periodic-image invariance, partial PBC, asymmetric tensor
orientation, stress units, direct outputs and disconnected autograd graphs.

Both the periodic Torch example and the compiled v2 example completed a short
four-bead PIMD calculation with 21 recorded frames and a completed restart. The
v2 restart left every saved NPZ byte-for-byte unchanged. These are interface
checks: both runs retain `converged=false` and `physical_converged=false`.

### Real electronic-structure backends

Run the two supplied examples with one BLAS/OpenMP thread:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python examples/potentials/on_the_fly.py tblite
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python examples/potentials/on_the_fly.py pyscf
```

They evaluate H2 at 1.4 and 1.47 bohr, then compare forces against central energy
finite differences with a 1e-4-bohr step. The recorded maximum errors across both
geometries are approximately **4.80e-7 hartree/bohr** for tblite GFN2-xTB and
**2.57e-9 hartree/bohr** for PySCF RHF/STO-3G. These are numerical derivative
checks; they do not compare either approximate PES with exact chemistry.

The adapter tests also cover actual UHF H, RKS H2, tblite periodic strain virial,
SCF failures, engine reuse, per-slot warm starts and independent cold starts.
PySCF requires an explicit fresh density: `reset()` alone retains orbitals, and
`kernel(dm0=None)` can otherwise reuse the preceding frame's state. DFT gradients
include grid response by default. tblite's reported tensile strain derivative
is negated to match PyRPMD's `W=-V*stress`.

MACE, NequIP, DeePMD and fairchem loader tests use dependency stand-ins with real
ASE conversion. They verify model arguments, species order, charge/spin inputs,
cell/PBC, energy/force/stress conversions, batch calls and lifecycle. **No trained
ML checkpoint was run, and no actual GPU was tested.** The fairchem tests cover
its distinct molecular multiplicity and nonmolecular zero-spin model inputs.

### Actual sockets and MPI

The i-PI tests use real loopback clients. A two-client barrier proves that bead
evaluations overlap; a rendezvous proves that the faster client gets additional
work before the slower one completes. Tests also cover fragmented payloads,
ordered non-symmetric virials, repeated INIT, client timeout/disconnection,
malformed replies, persistent connections and closure/rebinding races. No
wall-clock speedup is inferred from these correctness tests.

```bash
mpiexec -n 5 python tests/mpi_dynamic_integration.py
mpiexec -n 4 python tests/mpi_integration.py
```

Both scripts passed on actual local MPI ranks. The five-rank script proves
first-free-group dispatch, ordered bead forces, empty/underfilled queues,
collective error draining, initialization failures, unpicklable results,
exactly-once cleanup, mock GPU placement, and exact free-PES dRPMD/static
launch/trajectory/checkpoint arrays. Completed restart and dynamic-to-static
restart also pass with a fixed parent count. MPI process crashes, stalled
engines and network failures remain launcher-level concerns.

Local MPICH required `MPIR_CVAR_CH4_CMA_ENABLE=0 UCX_TLS=tcp,self` because the
container does not implement CMA's process-memory operation. These settings are
only in the local test command; neither runtime source nor the OpenMPI CI gate
changes MPI transport settings. The local socket and PySCF runs used normal
socket/process-metadata access, rather than the restricted test sandbox.

### Regression, docs and packaging

**Final full suite: 1,643 passed, 17 skipped, 153 subtests passed**, with three
existing TorchScript deprecation warnings. Skips comprise ten unbuilt BKMP2
checks, one unconfigured EAM check, one optional independent solver and five
BLAS-runtime capability checks. Full details are in `evidence.json`. The additional final
socket invocation passed all 20 cases, including two INIT/PBC contradiction
regressions added after the full suite had collected its cases. The focused
input and legacy periodic-contract tests passed after their old blanket C-ABI
rejection expectation was narrowed to explicitly selected v1.

English and Chinese Sphinx HTML builds pass with warnings treated as errors.
Source and native wheel builds pass, including the new adapter module, v2
header and compiled example. CI now adds the actual five-rank dynamic-MPI script,
periodic Torch tests and a real tblite/PySCF job.

## Boundaries

- C ABI v2 PES batches are compiled; the outer propagator uses the Python path.
  Fused C++ v2 propagation is not implemented.
- ASE-backed MACE/NequIP/fairchem evaluate frames sequentially. Generic Torch and
  DeePMD have native batch calls. DeePMD device choice belongs to its backend;
  PyRPMD does not pretend to assign its GPU through an unsupported generic API.
- The PySCF adapter is molecular CPU HF/DFT. Periodic PySCF, post-HF, excited-state
  and GPU4PySCF calculations are not part of this change.
- i-PI clients must implement the same PES independently of task assignment.
  All configured clients must connect before the first batch. Different
  PyRPMD trajectory slots need distinct socket addresses; no port-offset policy
  is invented.
- Dynamic MPI consumes one coordinator rank and separates parent sampling from
  collision propagation. It does not overlap these phases or certify a
  speedup on a particular molecular workload.
- No trained-model quality, physical material/reaction convergence, multi-node
  scaling or GPU performance claim follows from these checks.
