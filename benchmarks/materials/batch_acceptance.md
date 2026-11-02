# Real materials trajectory-batch acceptance

`batch_acceptance.py` validates the fixed-cell `pimd` and `diffusion` workflows
with the checksum-pinned Angelo–Moody–Baskes Ni–Al–H EAM table already used by
[`ni_h_eam.py`](ni_h_eam.py). This is an executable real-potential matrix, not
a harmonic stand-in or a GPU performance claim. The potential file remains
outside Git and is checked against the existing byte count and SHA-256.

## Run the implementation gate

From an installed source checkout:

```sh
python -m pip install -e '.[test,ase,mpi]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/materials/batch_acceptance.py --quick \
  --model /absolute/path/NiAlH_jea.eam.alloy --download \
  --output /new/path/materials-batch-acceptance \
  --modes serial,process,batch,process_batch,mpi,mpi_batch,mpi_dynamic,mpi_dynamic_batch \
  --mpi-command 'mpiexec -n 2' \
  --mpi-dynamic-command 'mpiexec -n 3'
```

Use `--download` only to explicitly obtain a missing table. A supplied file with
different bytes is rejected. The executable is the outer coordinator: do not
launch that coordinator itself under MPI. It starts fresh Python interpreters
and the declared MPI launchers for individual cases. Adapt only launcher
arguments to the cluster's allocated resources. The two-rank modes have
`G=2, B=1`; dynamic modes have `C=1, G=2, B=1` and require three ranks.

Without MPI, omit the launcher options and `--modes`; the default covers serial,
two local processes, task batching, and processes plus task batching. The CI
workflow explicitly installs Open MPI and runs all eight modes. MPI is not
silently replaced with a local executor, and a missing dependency is a failure
in the explicitly requested matrix.

Each task has five independent replica RNG streams, four beads, 28 atoms,
a temperature of 600 K, four preparation steps and sixteen production steps
at 0.1 fs. A two-lane
batch therefore has a short final group, and three groups exceed the two MPI
worker groups. These small trajectories validate the numerical workflow;
they are too short to determine an equilibrium material property or diffusivity.

The following checks must all pass:

| Check | Evidence recorded |
| --- | --- |
| Real potential and units | Direct ASE energy/force/stress conversion; Cartesian energy finite differences |
| Physical virial | All nine homogeneous cell/coordinate strain components; no `sum(r*f)` replacement |
| Periodicity and frame independence | Lateral lattice shifts, unchanged unwrapped input, distinct reordered frames |
| Actual task batching | Histogram of shapes passed to the real `ASEPotential.evaluate`; `[K*P,N,3]` calls must occur and provider-call count must decrease |
| Conserved work | Equal total physical PES frame count in scalar and packed execution |
| Serial/process/MPI agreement | Exact same-machine final bead `q/p`, RNG, physical clocks, all saved stream arrays, propagation counters and reported statistics |
| Interruption and continuation | A real integration-step exception after accepted progress; incomplete checkpoints resumed with two-to-three-lane repacking and a changed block length |
| Durable prefixes | Every pre-interruption immutable chunk remains byte-for-byte unchanged |
| Completed restart | Zero PES evaluations, zero advanced physical steps and unchanged checkpoint bytes |
| Complete evidence | Every registered replica checkpoint is required; missing/corrupt data cannot become a pass |
| Stable runtime | Before/after hashes of the Python source tree, native sources, executed native extension and benchmark sources |

The exception is deliberately injected after an actual integrator step. The
workflow's ordinary rollback and checkpoint path handles it; the harness does
not synthesize, truncate or edit a checkpoint to manufacture a continuation.
Failure is propagated after in-flight cooperative lanes have drained. This
tests controlled interruption, not recovery from a destroyed filesystem or an
unresponsive external process.

The provider-call probe exists only in isolated benchmark workers. It wraps
the real ASE adapter and accumulates a bounded shape histogram. It does not
replace the calculator, force, energy, virial, RNG or integrator. This EAM's
explicit `potential.trajectory_batch_safe=true` declaration is specific to its
history-independent frame semantics; it is not permission to mark every ASE
calculator safe.

### Performance and resource interpretation

`report.json` preserves per-mode wall time, CPU time, PES calls and frames,
committed replica steps per second, shape histograms and individual process
peak-RSS receipts. Startup, checkpoint I/O and the small counting probe are
included. Resume throughput counts only newly committed steps; an already
completed restart advances zero steps. MPI receipts distinguish rank-local
self peaks from the largest terminated child peak; their sum is not described
as simultaneous process-tree RSS.

Timing is a single finite-run observation per mode. There is **no required
speedup**. ASE EAM evaluates its frames sequentially, so a packed API call
does not by itself reduce EAM force work or provide native GPU batching. Use
representative longer runs and repeated order-balanced measurements before
selecting production resources. The script does not set CPU affinity or
allocate hardware. The thread environment is fixed at one numerical thread
per worker; the MPI launcher and scheduler still own placement.
Record other host workloads with any published timing. The initial local
acceptance run shared its host with regression tests, so those observations
must not be used to claim parallel speedup.

All input requests, stdout/stderr and resource/call receipts remain under
`evidence/`. A failed gate leaves `implementation_checks_passed=false` plus its
reason in the report and exits nonzero. `physical_converged=false` is retained
even if every implementation comparison passes.

## Run a longer equilibrium convergence study

`batch_convergence.py` supplies a separate executable seven-arm PIMD study.
It writes the entire protocol before starting any potential evaluation and
uses distinct predeclared root seeds in every arm. It does not adapt the
matrix after looking at a point estimate.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/materials/batch_convergence.py \
  --model /absolute/path/NiAlH_jea.eam.alloy \
  --output /new/path/materials-equilibrium-convergence \
  --replicas 4 --beads 4 --dt-fs 0.1 \
  --equilibration-steps 2000 --production-steps 10000
```

The seven arms are the base run, doubled beads, halved timestep, larger lateral
cell, doubled preparation, doubled production, and doubled replica count.
Halving the timestep doubles preparation, production and recording step
counts, preserving the corresponding physical durations. The cell control
repeats the complete Ni slab plus H adsorbate in both lateral directions:
28 atoms become 112 atoms and the coverage and number of substrate layers stay
fixed. It does not conflate a cell-size change with a coverage change.

The reported equilibrium targets are potential energy per atom, centroid-virial
internal energy per atom, and H ring radius squared. All recorded frames enter
each replica mean. Independent replica means determine the Student-t and Welch
intervals; molecules, beads, time origins and saved frames are not counted as
independent replicas. Broad overlapping intervals do not establish equivalence.

Before execution, the protocol declares absolute margins of 0.005 eV/atom for
the two energy observables and 0.002 bohr² for H ring radius squared. Every
control's entire 95% difference interval must fit inside its margin. All arms
also require independent-replica precision, production-half stationarity and
at least twenty estimated effective samples with four complete correlation
blocks per replica and observable. These are explicit finite-study screens,
not universal physical accuracy tolerances. Revise the protocol before a new
experiment if a scientific question needs different bounds; do not tune them
to make an existing result pass.

An inconclusive or failed convergence study keeps every arm and exits **2**.
An execution/corruption failure exits nonzero with its error recorded. A
complete comparison can set `convergence_checks_passed=true`, while
`physical_converged=false` remains: potential adequacy, material-specific
equilibrium, longer length/time scales and scientific interpretation need
review. PIMD time is not real transport time. The short diffusion acceptance
matrix and these equilibrium controls do not certify a diffusion coefficient,
an isotope enhancement or a reproduction of the Ni(100) literature model.

The default study is substantially more expensive than the quick matrix,
especially its larger cell and bead arms. Smaller explicit step budgets can
exercise the complete analysis path, but retain their expected insufficient-
sampling outcome. Neither shorter settings nor a failed gate are silently
promoted to a production result.

For a short end-to-end exercise of all seven analysis arms, explicitly use
`--replicas 4 --beads 2 --dt-fs 0.1 --equilibration-steps 16
--production-steps 64`. Its base production length is only 6.4 fs, and the
unchanged effective-sampling, precision and stationarity gates must still run.
This diagnostic protocol checks that a complete but insufficient study is
reported as such; it is not a shorter route to a converged material property.
