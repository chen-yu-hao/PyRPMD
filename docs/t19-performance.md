# CUDA and large-system performance validation

<a id="t19-cuda-and-large-system-performance-validation"></a>

## Acceptance status and source revisions

**CUDA acceptance is incomplete.** The 2026-11-03 execution environment has no
NVIDIA device, driver utility, CUDA-enabled Torch runtime, or OpenMM CUDA
platform. GPU model, driver version, CUDA version, errors and timings are
therefore unavailable. CPU measurements below are CPU measurements only.
The machine-readable [environment record](validation/t19-performance/environment.json)
retains this distinction and sets `gpu_acceptance_complete=false` and
`physical_converged=false`.

This work starts at
[`main@f877051`](https://github.com/chen-yu-hao/PyRPMD/commit/f87705115c7496821b283a0966c991abd541231c).
At inspection time, [PR #84](https://github.com/chen-yu-hao/PyRPMD/pull/84),
head `a17a69774b2284f6345c92ac44b021999b07556a`, was still open. The T04
follow-up branch was at `c7f05464d1724c58863e18494e0d11e24e062571`; its MACE
graph-copy and nondefault CUDA stream changes are absent from this main
revision. A benchmark report must identify the tested source and cannot
certify those changes merely because a CPU test passed.

The absent `docs/torch-materials-workflows.md` belongs to PR #84. Its CUDA
measurement procedure and pending status are recorded here to avoid creating
a competing copy of that unmerged page.

## Measurement contract

The CUDA matrix is every combination of beads/frames `P=8,32,64` and atoms
`N=64,512,4096`. A potential evaluation contains a complete `[R*P,N,3]`
batch, with `R=1` replica by default; the force-engine tables below use `[P,N,3]`.
A workflow evaluation includes its actual propagation, observation,
checkpoint and output costs, with the trajectory count reported separately.
OpenMM reports construction and cold-call costs separately. ML provider
timings exclude construction, while workflow timings include it.

All coordinates, energies and forces at the PyRPMD boundary use atomic units.
CPU float64 references use the same model, coordinates, cell and chemical
state. No model is silently cast, moved to another device, split into smaller
batches or replaced after an out-of-memory error. Unsupported cases remain
explicit records rather than disappearing from the matrix. Trained model
files and representative structures must be supplied for production ML
measurements; a synthetic harmonic or Lennard-Jones probe is not an ML model
performance claim.

Errors report absolute maxima and reference-scaled norms, with separate
energy, force and (where provided) configurational virial fields. Warmed
timings retain individual repetitions and their median. CUDA timings must
synchronize the selected device around each measured operation: GPU work is
asynchronous, and timing Python enqueue operations alone is invalid. See
[PyTorch CUDA timing and stream semantics](https://docs.pytorch.org/docs/main/notes/cuda.html#asynchronous-execution).

A stochastic PIMD path on CUDA is not expected to reproduce a CPU random
path from the same seed. The workflow comparison instead checks the same
saved positions/momenta against an independent NumPy float64 calculation;
free-particle diffusion also admits an analytic ballistic-centroid check.
This separates numerical correctness from stochastic sampling uncertainty.
See [PyTorch's reproducibility boundary](https://docs.pytorch.org/docs/main/notes/randomness.html).
Short timing runs never establish physical convergence.

The ML/workflow driver's default numerical gate is the elementwise condition
`abs(actual-reference) <= 1e-10 + 1e-9*abs(reference)` in the reported atomic
units. Relative L2 errors are diagnostic fields, not a second acceptance
criterion. Exit code zero means only that all requested cases passed; it
does not by itself mean the complete CUDA matrix or T19 acceptance passed.

## Available hardware

| Item | Observed value |
| --- | --- |
| CPU | AMD EPYC 9V74 80-Core Processor, CPUID brand string |
| Affinity-visible logical CPUs | 9; cgroup quota and memory limit are not exposed |
| Python / NumPy / SciPy | 3.12.14 / 2.3.5 / 1.17.0 |
| Torch | 2.15.0+cpu; CUDA unavailable |
| OpenMM | 8.6.1; Reference and CPU platforms |
| LAMMPS / MPI | 2025.7.22.4.0 / MPICH 5.0.2 |
| Thread policy | One OpenMP, BLAS and OpenMM CPU thread per engine; four host workers for the Context-pool mode |
| GPU / driver / CUDA runtime | Unavailable; no GPU measurement |

LAMMPS runs use one MPI rank and `UCX_TLS=self` because this environment does
not provide the interprocess/network transports needed by default UCX
initialization. This is a single-rank evaluation, not an MPI scaling result.
All times are shared-host measurements, not dedicated-node performance.
The separate Torch CPU workflow smoke uses one intra-op thread; its recorded
inter-op pool has nine threads. That smoke is numerical/API evidence only.

## OpenMM: persistent RPMD copy storage

The explicit `batch_mode="rpmd"` route stores every input row in one
`RPMDIntegrator` copy bank and one Context. It never calls `step()` or
`getTotalEnergy()`, never applies ring-polymer contraction, and returns only
the physical System potential energy and forces. In particular, OpenMM's
ring springs and thermostat never enter a PyRPMD force evaluation.

The bank grows to the largest batch seen and is retained for later smaller
batches. This reduces the number of persistent Contexts, but the public
OpenMM API still uploads each row and retrieves each row separately. Device
residency alone does not establish a speedup. The serial evaluator remains
the default, and its metadata and existing run identities are unchanged.
OpenMM's platform and copy-count restrictions still apply; unsupported
allocations fail explicitly without changing device, precision or batch size.

This new mode requires a fixed cell; changed-cell binding and `set_cell`
requests are explicitly rejected before mutation. This prevents a grown or
recreated Context from selecting a different automatic PME grid after a cell
change. Systems with OpenMM constraints are also explicitly rejected.
Use the existing serial or concurrent-Context routes for their established
constraint contract. Virtual sites are recomputed with OpenMM's own kernels,
and their physical parent forces are independently checked by finite
differences on the Reference platform. The virtual-site path verifies the selected physical positions
because it relies on the tested RPMD copy-to-Context behavior. That extra
position read is included in timings.

Upstream contracts: [RPMDIntegrator copy API](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.RPMDIntegrator.html)
and [the installed OpenMM 8.6.1 RPMD implementation](https://github.com/openmm/openmm/blob/b399af4725573963b46d6c1083fdcf7a37615857/plugins/rpmd/openmmapi/src/RPMDIntegrator.cpp).
Its [common device kernel](https://github.com/openmm/openmm/blob/b399af4725573963b46d6c1083fdcf7a37615857/plugins/rpmd/platforms/common/src/CommonRpmdKernels.cpp)
downloads the Context position/charge buffer before each copy upload. This
is another reason to measure rather than assume that copy residency reduces
host/device transfer time.

## LAMMPS: diagnosis and conservative optimization

The T05 boundary-aligned FCC case rebuilt neighbor lists for 288 of 320
individual frames at 2048 atoms. Its uniformly translated case reused lists
for 288 of 320 frames but still showed little wall-time benefit. Both
observations matter: frequent rebuilding explains one cost, while removing
that cost alone does not guarantee faster complete evaluations. The original
measurements remain in the [T05 report](https://github.com/chen-yu-hao/PyRPMD/blob/f87705115c7496821b283a0966c991abd541231c/docs/potential-adapters.md#measured-cpu-results-2026-11-02).

This change removes repeated sorting and per-atom Python symbol construction
from the strict atom-identity validation. The replacement uses linear-time
ID counts and vectorized element comparisons while preserving checks before
and after force evaluation. The existing periodic wrapping, neighbor-skin
criterion, force refresh commands and metadata remain unchanged.

A continuous-image optimization was investigated but is not enabled: using
different periodic images across a resumed provider can introduce floating
point differences unless the required neighbor/image state is checkpointed.
The existing provider checkpoint contract does not save that state. The
performance investigation does not relax exact-resume requirements to obtain
a better timing.

The profiling script separates complete uninstrumented batch timings from
diagnostic timings of atom validation, box handling, LAMMPS commands and
other Python/array work. Profile timings must not be substituted for the
uninstrumented medians.

The force refresh still uses `run 1 pre no post no` without integration
fixes on a reused neighbor list. `run 0 pre no` does not refresh the physical
force. On normal non-rebuild steps LAMMPS forwards ghost coordinates; on
rebuild steps it remaps periodic images and rebuilds ownership/neighbors.
See [LAMMPS timestep execution](https://docs.lammps.org/Developer_flow.html).


## Measured CPU results

The following are **force-evaluation** costs in milliseconds per complete
batch, not full dynamics steps. Measurements ran sequentially without the
full regression suite running alongside them. The host is shared, so small
positive or negative changes should not be interpreted as stable machine
performance guarantees. Raw samples, CPU process times where available,
construction/cold costs, source hashes and model parameters remain in the
linked JSON records.

### OpenMM: Lennard-Jones, nine shapes

[Raw OpenMM CPU record](validation/t19-performance/openmm-cpu.json).
Each mode has three rounds, one warmup and three timed calls per round;
mode order is reversed in alternating rounds. The reported median uses all
nine timed samples. Each Context has one CPU engine thread; the concurrent
mode uses four host workers and retains one Context per row, while RPMD uses
one Context with a copy bank. These CPU resource budgets differ.

| N | P | Serial ms | Contexts, 4 workers ms | RPMD copies ms | Contexts speedup | RPMD speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 8 | 1.184 | 1.820 | 1.347 | 0.650 | 0.879 |
| 64 | 32 | 6.218 | 5.812 | 5.845 | 1.070 | 1.064 |
| 64 | 64 | 7.722 | 8.294 | 8.937 | 0.931 | 0.864 |
| 512 | 8 | 3.954 | 2.866 | 3.670 | 1.379 | 1.077 |
| 512 | 32 | 16.601 | 10.279 | 10.177 | 1.615 | 1.631 |
| 512 | 64 | 29.287 | 21.109 | 31.908 | 1.387 | 0.918 |
| 4096 | 8 | 23.459 | 9.290 | 19.689 | 2.525 | 1.191 |
| 4096 | 32 | 96.298 | 40.836 | 95.138 | 2.358 | 1.012 |
| 4096 | 64 | 175.931 | 90.673 | 179.578 | 1.940 | 0.980 |

Speedups divide the serial median by the corresponding mode's median.
At N=4096 the Context pool is 1.94–2.53 times faster, while RPMD copies are
0.98–1.19 times as fast. Small N=64 calls often become slower. The RPMD bank
still performs row-by-row public-API uploads and force/result queries; it
does not turn them into a native batched force kernel or execute the rows
concurrently. Extra call and copy overhead can outweigh savings. These
timings do not identify an exact hardware-level breakdown; no GPU transfers
were measured here.

The float64 oracle is OpenMM Reference. The timed [CPU nonbonded kernels](https://github.com/openmm/openmm/blob/b399af4725573963b46d6c1083fdcf7a37615857/platforms/cpu/src/CpuNonbondedForce.cpp) use
float operations internally; float64 arrays at the Python boundary do not
make those kernels double precision. The explicit CPU comparison gate is
`abs(error) <= 1e-7 + 3e-6*abs(reference)` in atomic units; the maximum
observed scaled error is 0.02146. CUDA LJ/harmonic measurements instead
request `Precision=double` and use `atol=1e-12, rtol=1e-10`.
PME water has its separately recorded reciprocal-space comparison tolerance.

Each error row below is the maximum over all three modes, configurations and
rounds for that shape. Relative L-infinity divides the largest absolute force
error by the largest absolute reference force. The nearly symmetric LJ grid
has small net forces, so relative force errors reach 0.001136 even though
absolute errors remain below 1e-9 Ha/bohr. Passing the combined gate does not
mean bitwise agreement or a relative error below the rtol alone.

| N | P | Max energy error (Ha) | Max force error (Ha/bohr) | Force relative L-infinity error |
| --- | --- | --- | --- | --- |
| 64 | 8 | 3.373189e-09 | 2.222166e-10 | 3.596872e-04 |
| 64 | 32 | 4.132272e-09 | 2.766064e-10 | 3.651160e-04 |
| 64 | 64 | 3.153490e-09 | 2.766064e-10 | 3.651160e-04 |
| 512 | 8 | 1.842231e-08 | 4.959259e-10 | 6.217113e-04 |
| 512 | 32 | 2.057596e-08 | 5.546212e-10 | 6.740362e-04 |
| 512 | 64 | 2.063164e-08 | 5.498911e-10 | 6.675118e-04 |
| 4096 | 8 | 1.215548e-07 | 9.911057e-10 | 1.136210e-03 |
| 4096 | 32 | 1.296625e-07 | 9.972327e-10 | 1.097439e-03 |
| 4096 | 64 | 1.371340e-07 | 9.993977e-10 | 1.057662e-03 |

The force oracle also has independent analytic and finite-difference tests;
virtual-site and PME restart checks run on Reference. CUDA-specific tests
are skipped when that platform is absent.

### LAMMPS: pinned old implementation versus current implementation

[Raw LAMMPS CPU record](validation/t19-performance/lammps-cpu.json).
All rows use P=32, three rounds, two warmups and seven timed batches per
round, with alternating mode order. The statistic is the median of the
three per-round medians. The baseline adapter is loaded directly from
`f87705115c7496821b283a0966c991abd541231c`; both versions evaluate the same
coordinate sequence and model.

Offset is the translation as a fraction of the FCC lattice constant.
Offset 0 places grid sites at periodic faces; 0.173 shifts them away.
All ratios greater than one indicate faster execution.

| N | Offset | Old serial ms | New serial ms | Old persistent ms | New persistent ms | New serial / persistent | Old / new persistent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 0.0 | 44.36 | 47.45 | 52.54 | 49.04 | 0.968 | 1.071 |
| 256 | 0.173 | 38.22 | 36.86 | 22.81 | 23.42 | 1.574 | 0.974 |
| 2048 | 0.0 | 261.09 | 272.46 | 277.71 | 247.49 | 1.101 | 1.122 |
| 2048 | 0.173 | 180.71 | 116.83 | 108.01 | 91.30 | 1.280 | 1.183 |
| 4096 | 0.0 | 523.50 | 497.70 | 490.30 | 471.05 | 1.057 | 1.041 |
| 4096 | 0.173 | 247.38 | 216.20 | 165.29 | 94.07 | 2.298 | 1.757 |

The optimized persistent mode is still slower than current serial for the
256-atom boundary case. It is also about 2.6% slower than the old persistent
mode for the translated 256-atom case. No uniform acceleration is claimed.
At 4096 atoms the translated case improves from 165.29 to 94.07 ms in
persistent mode, but the boundary case only improves from 490.30 to
471.05 ms. Process CPU samples are also retained: for example, N=2048,
offset 0 serial process time drops from 258.51 to 240.92 ms even though
its wall-clock median increases. This is a reason to retain scheduling
variability rather than infer a deterministic regression from one wall-time
ratio.

Correctness uses the old adapter's full-setup result as the campaign
reference, with additional independent LJ and finite-difference unit tests.
Below are maxima over both current modes, all measured configurations and
rounds. The comparison gate is `atol=1e-12, rtol=2e-10` in atomic units.

| N | Offset | Max energy error (Ha) | Max force error (Ha/bohr) | Max virial error (Ha) |
| --- | --- | --- | --- | --- |
| 256 | 0.0 | 1.221245e-15 | 3.019249e-18 | 1.179612e-16 |
| 256 | 0.173 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 2048 | 0.0 | 1.509903e-14 | 2.524158e-19 | 8.881784e-16 |
| 2048 | 0.173 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 4096 | 0.0 | 1.776357e-14 | 2.361634e-19 | 2.109424e-15 |
| 4096 | 0.173 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |

Each timed round contains 7*32=224 frame evaluations. The persistent
rebuild/reuse counts are identical across the three rounds:

| N | Offset | Rebuilds per timed round | Reuses per timed round |
| --- | --- | --- | --- |
| 256 | 0.0 | 216 | 8 |
| 256 | 0.173 | 0 | 224 |
| 2048 | 0.0 | 224 | 0 |
| 2048 | 0.173 | 0 | 224 |
| 4096 | 0.0 | 224 | 0 |
| 4096 | 0.173 | 0 | 224 |

At N=2048 and 4096, offset 0, **none of the timed frames reuses a neighbor
list**. Differences between serial and persistent wall time in these rows
cannot be attributed to neighbor reuse. Both modes still execute native
setup/force work for every frame. The translated rows reuse all timed lists,
but must still refresh ghost coordinates and calculate the current force.

Separate instrumented passes diagnose the Python/native split. These
numbers are not summed into or used as denominators for the uninstrumented
timings above:

| N | Offset | Old atom checks ms | New atom checks ms | New native commands ms | Native fraction |
| --- | --- | --- | --- | --- | --- |
| 256 | 0.0 | 6.11 | 5.65 | 30.68 | 58.8% |
| 256 | 0.173 | 4.83 | 5.03 | 13.81 | 41.1% |
| 2048 | 0.0 | 57.06 | 7.12 | 205.13 | 86.3% |
| 2048 | 0.173 | 32.40 | 7.98 | 56.04 | 60.4% |
| 4096 | 0.0 | 55.53 | 9.01 | 416.63 | 89.8% |
| 4096 | 0.173 | 45.96 | 11.13 | 109.27 | 66.6% |

The atom-check optimization matters at large N. After it, native LAMMPS
commands account for approximately 86–90% of the new persistent profile in
the two largest boundary cases. Persistent mode retains one instance per
frame but still calls them sequentially; the serial baseline already retains
a single instance, so this comparison is not process startup versus no
startup. Cache/memory pressure from multiple instances remains an unmeasured
hypothesis, not an established cause.

The original T05 count of 320 frames includes cold and warmup calls and
uses offset 0.25 for its translated experiment. It must not be compared
directly to this campaign's 224 timed frames per round and offset 0.173.
The rejected continuous-image experiment is preserved as a
[diagnostic record](validation/t19-performance/lammps-continuous-image-rejected.json):
its tiny but nonzero post-recreation force difference was sufficient to
reject a claim of bitwise checkpoint equivalence.

## CUDA execution status and PR #84 CPU smoke

| Scope | Requested shapes | Result | GPU timing/error evidence |
| --- | --- | --- | --- |
| Five ML providers | 5 * 9 = 45 | Blocked: no CUDA and no supplied local models | None |
| Torch PIMD and diffusion | 2 * 9 = 18 | Blocked: no CUDA; PR #84 absent from the task base | None |
| OpenMM serial / Contexts / RPMD | 9 shapes | CUDA platform unavailable | None |

The [default ML/workflow report](validation/t19-performance/cuda-blocked.json)
retains all 63 blocked cases and exits 2. The
[OpenMM CUDA report](validation/t19-performance/openmm-cuda-blocked.json)
retains all nine unmeasured shapes and also exits 2. A completed enumeration
of blocked cases does not mean the requested measurements passed.

For interface verification only, a separate worktree applied PR #84's
runtime/source/test/example patch to main@f877051, without merging the PR or
its documentation into this branch. The
[CPU workflow smoke](validation/t19-performance/pr84-cpu-smoke-final.json)
uses P=8, N=64 and one replica; each call completes two equilibration and
six production steps. Each task executes CPU and target-CPU paths, one warmup
and three measured calls per path. The actual imported modules and their
hashes are recorded separately from the benchmark script's source.

| Independent check | PIMD maximum absolute error | Free diffusion maximum absolute error | Unit |
| --- | --- | --- | --- |
| Final energy observables | 1.4210854715202004e-14 | 0 | Ha |
| Final centroid position / velocity / squared gyration | 0 | 0 | Corresponding atomic units |
| Analytic ballistic centroid position | Not applicable | 1.1102230246251565e-15 | bohr |
| Centroid velocity conservation | Not applicable | 1.3322676295501878e-15 | bohr / atomic time |

Both cases pass, but this small CPU smoke ran alongside regression tests.
Its recorded durations and CPU/CPU ratios are **not performance evidence**.
Neither physical convergence nor CUDA acceptance follows from it. The
workflows use harmonic/free probes and do not load the ML manifest.

## Running the remaining CUDA campaign

First integrate the T04 follow-up and PR #84, then record the resulting
commit. The five ML frameworks may need separate compatible environments.
An explicit provider subset can pass its requested cases without claiming
that all five providers and both workflows have passed. Review the individual
reports together; the driver does not merge campaigns from different runs.

For one local float64 model, a manifest has the following form. Replace the
path and SHA256 with the actual trusted model and supply representative
structures for each requested size. Generated lattices are also supported,
but are labeled as performance probes.

```json
{
  "schema": "pyrpmd.t19-models.v1",
  "models": [{
    "name": "local-mace",
    "provider": "mace-batched",
    "model": "models/local.model",
    "sha256": "<actual 64-character lowercase SHA256>",
    "model_kind": "trained",
    "dtype": "float64",
    "params": {"virial": true},
    "structures": {
      "64": "structures/n64.npz",
      "512": "structures/n512.npz",
      "4096": "structures/n4096.npz"
    }
  }]
}
```

Each NPZ contains float64 `positions[N,3]` and optional `cell[3,3]` in bohr,
string `symbols[N]` and optional boolean `pbc[3]`. Periodic structures require
a cell. Paths are relative to the manifest.
The supplied hash is checked before loading; no weights are downloaded.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/efficiency/t19_cuda.py --device cuda:0 --suite ml \
  --providers mace-batched --manifest models.json \
  --output t19-mace-cuda.json --markdown t19-mace-cuda.md

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/efficiency/t19_cuda.py --device cuda:0 --suite workflows \
  --beads 8 32 64 --atoms 64 512 4096 \
  --output t19-workflows-cuda.json --markdown t19-workflows-cuda.md

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENMM_CPU_THREADS=1 \
  python benchmarks/efficiency/t19_openmm.py --device cuda:0 \
  --model lj --beads 8 32 64 --atoms 64 512 4096 --workers 4 \
  --rounds 3 --output t19-openmm-cuda.json
```

Use `nequip-batched`, `fairchem-batched`, `deepmd-batched` or `metatomic` in
their corresponding environments and supply each framework's required
parameters. Unsupported CPU float64 model formats are explicit failures to
provide the requested reference. Neither OOM nor an unsupported combination
triggers a fallback.

### PR #84 workflow measurement subsection to integrate

After `docs/torch-materials-workflows.md` lands, link its measurement section
to this record and the workflow command above. The driver calls the public
`run_materials` entry point for harmonic PIMD and free-particle diffusion.
It verifies the requested replicas, completion status, total steps, elapsed
simulation time, trajectory frames, sample clock, shapes and float64 dtype
before it reports milliseconds per step. Complete workflow timing includes
construction, propagation, checkpoint/trajectory I/O and analysis; it is not
an isolated integrator-kernel timing.

Each final snapshot is checked with independent NumPy formulas; diffusion
also checks the free centroid's analytic motion. GPU and CPU random paths
are not compared as if they were identical. This suite validates synthetic
workflow plumbing, not trained-ML end-to-end performance or convergence of
material observables. Keep `physical_converged=false` and the pending CUDA
status until actual evidence supports a narrower, explicit claim.

## Regression verification and reproducibility

The [validation index](validation/t19-performance/validation.json) contains
the executed commands, working directories, exit codes, source revisions,
dependency-integration recipe and SHA256 hashes of the evidence files.
The measured code is published at
[`6dc3b527`](https://github.com/chen-yu-hao/PyRPMD/commit/6dc3b527531369f1f9fae7ea5f6dfa3800ef9af1).
The LAMMPS run began at local commit `e983daaa`; the
[commit mapping](validation/t19-performance/commit-mapping.json) verifies
that it has the identical Git tree. Publication changes commit metadata,
not the tested file contents.

| Verification | Original result | Raw log |
| --- | --- | --- |
| Full unchanged main | 3 failed, 10898 passed, 171 skipped, 68 warnings, 493 subtests passed in 1485.02s | [Baseline](validation/t19-performance/baseline-full.log) |
| Full candidate | 3 failed, 10974 passed, 182 skipped, 68 warnings, 493 subtests passed in 1012.21s | [Candidate](validation/t19-performance/candidate-full.log) |
| LAMMPS provider, batch and new performance regressions | 48 passed in 5.57s | [LAMMPS](validation/t19-performance/lammps-focused.log) |
| OpenMM provider, batch, resident and benchmark regressions | 54 passed, 7 skipped in 36.45s | [OpenMM](validation/t19-performance/openmm-focused.log) |
| CUDA campaign harness | 42 passed, 4 skipped in 6.99s | [Harness](validation/t19-performance/cuda-harness-tests.log) |
| PR #84 isolated public-workflow CPU harness | 1 passed in 12.62s | [Workflow](validation/t19-performance/pr84-cpu-harness-test.log) |

The candidate adds 76 passes and 11 skips with **no new failed test IDs**.
Both full runs exit 1 because the same environment-limited tests fail:

- `tests/test_engine_subcommunicators.py::test_actual_lammps_private_engine_groups[static-4]`
- `tests/test_engine_subcommunicators.py::test_actual_lammps_private_engine_groups[dynamic-5]`
- `tests/test_ipi_clients.py::test_existing_unix_address_is_not_removed_on_bind_failure`

The two MPI cases fail during MPICH/UCX initialization with denied
`process_vm_readv` access; the socket case fails on AF_UNIX socket creation
with `PermissionError`. These failures were preserved, not skipped or
reclassified. Both worktrees use byte-identical copies of the three native
extension libraries. Full-suite wall times are not performance comparisons.

The full candidate command, from the task checkout, was:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENMM_CPU_THREADS=1 \
LD_LIBRARY_PATH=/workspace/scratch/998803105056/.venv/lib UCX_TLS=self \
TMPDIR=/workspace/scratch/998803105056/tmp \
PYTHONPATH=/workspace/scratch/998803105056/PyRPMD/src \
  ../.venv/bin/python -m pytest -q \
  > ../t19-validation/candidate-full.log 2>&1
```

The baseline used the same environment and interpreter from the
`PyRPMD-baseline` worktree, with `PYTHONPATH` set to that worktree's `src`
and output written to `baseline-full.log`. The index gives both complete
commands. Focused tests additionally verify analytic forces, finite
differences, virtual-site parent forces, failure recovery, unchanged default
metadata and exact stochastic checkpoint continuation.

The new documentation page is included by the single corresponding addition
to `docs/conf.py`. The existing source-distribution manifest already covers
the new benchmark scripts, tests, Markdown, JSON and logs. README files,
release/reference catalogs and package `__init__.py` files are unchanged.
