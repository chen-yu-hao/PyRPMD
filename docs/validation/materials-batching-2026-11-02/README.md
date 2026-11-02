# Materials trajectory batching: implementation and real Ni–H acceptance

## Implementation result

The fixed-cell legacy `pimd` and `diffusion` workflows now combine independent
replicas' physical PES requests when `parallel.trajectory_batch_size > 1`.
The completed real EAM matrix passed **16 of 16 execution cases** and both
controlled interruption/recovery checks. Every case preserved the scalar
reference's final bead q/p, clocks, RNG, saved physical arrays, accepted-step
counters and reported statistics exactly on this CPU runtime.

This record validates the implementation and its declared execution scope.
It does not establish an equilibrium material property or a converged
diffusion coefficient. Both the implementation report and the separate
convergence diagnostic preserve `physical_converged=false`.
The quick diffusion workflow itself reports `analysis_status=insufficient_data`
and `diffusion.status=insufficient_data`.

- [Complete real-potential report](acceptance-report.json)
- [Runtime and regression record](validation.json)
- [Full pytest output](pytest-full.txt)
- [Completed seven-arm diagnostic](convergence-report.json) and [predeclared protocol](convergence-protocol.json)
- [Executable example and acceptance rules](../../../benchmarks/materials/batch_acceptance.md)

The implementation is commit `de910aa0a4445b40b0669aa1e97a89cd0428e6f7`, based
on main `cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d`. The reports additionally
record before/after SHA-256 identities for 207 Python/native source and loaded
native-extension entries; all identities agreed during the real EAM matrix.
Later documentation/evidence-only commits do not change those scientific
sources.

## Physical input and independent oracle

The fixture is the existing 28-atom Ni(100)+H slab, at 600 K with lateral PBC
and a nonperiodic vacuum direction. It uses the Angelo–Moody–Baskes Ni–Al–H
EAM table from the repository's pinned LAMMPS source, with SHA-256
`545b0f79207ae519b7f355451812acd999bd6ed095f40a6bcfa9aa5258a38388`.
The table is downloaded only on an explicit request and is not redistributed
in this PR. Model URL, upstream commit, byte count and digest are preserved
in the report.

Each task has five independent replicas, four beads, four preparation steps
and sixteen production steps at 0.1 fs. Production lasts only 1.6 fs. The
Python workflow uses the installed compiled ring kernels; the exact loaded
extension hash is included. PIMD uses its normal thermostat, and diffusion
uses the existing unthermostatted production path.

| Independent check | Measured maximum error |
| --- | ---: |
| Adapter energy versus direct ASE | 4.44 × 10⁻¹⁶ hartree |
| Adapter forces versus direct ASE | 1.43 × 10⁻¹⁷ hartree/bohr |
| Physical configurational virial versus direct ASE stress | 4.44 × 10⁻¹⁶ hartree |
| Reordered physical frames | 0 in recorded atomic-unit values |
| Cartesian energy finite difference versus force | 4.43 × 10⁻¹⁰ eV/Å |
| Nine cell/coordinate strain derivatives versus virial | 8.68 × 10⁻⁷ eV |

The lattice-shift checks passed and the unwrapped input coordinates were not
modified. Strain tests deform both the cell and positions using the documented
row-cell convention. These checks validate adapter semantics and derivatives;
they are not an independent assessment of EAM's accuracy for real nickel.

## Actual batch work

Both PIMD and diffusion produced the following counts. The counter wraps the
real `ASEPotential.evaluate` in isolated benchmark workers; it does not
replace the calculator, force, energy or integrator.

| Execution | Trajectory processes / MPI ranks | Batch size | Actual PES calls | Physical frames |
| --- | --- | ---: | ---: | ---: |
| Serial | 1 process | 1 | 200 | 800 |
| Process | 2 workers | 1 | 200 | 800 |
| Batch | 1 process | 2 | 120 | 800 |
| Process + batch | 2 workers | 2 | 120 | 800 |
| Static MPI | 2 ranks, G=2/B=1 | 1 | 200 | 800 |
| Static MPI + batch | 2 ranks, G=2/B=1 | 2 | 120 | 800 |
| Dynamic MPI | 3 ranks, C=1/G=2/B=1 | 1 | 200 | 800 |
| Dynamic MPI + batch | 3 ranks, C=1/G=2/B=1 | 2 | 120 | 800 |

Every batched mode issued 80 requests of shape `[8,28,3]` and 40 requests of
shape `[4,28,3]`. This is the expected two-pair plus single-replica tail for
five replicas and four beads: 40% fewer provider API calls, with the same
800 physical frames. The nondivisible tail and the dynamic queue with more
tasks than worker groups are exercised by the real model.

ASE EAM still evaluates frames sequentially (`native_batch=false`). This
demonstrates actual cross-trajectory packing, not fused GPU inference.
Measured wall times were 20.07–36.26 seconds per case, including startup,
checkpoint I/O and the counting probe. The shared host also ran regression
tests, there was no CPU affinity isolation, and each mode has one observation.
These measurements are resource records and do not support a speedup claim.
The report retains separate rank/process peak-RSS receipts, not an invented
simultaneous process-tree peak.

## Recovery and collective ownership

For each real workflow, the harness raised an exception after an actual
integration step 9. The ordinary transaction path retained two partial
replicas at step 8. Continuation changed batch size 2 to 3, one local worker
to two processes, and block length 3 to 7. All four previously committed
trajectory chunks remained byte-identical, and the completed result matched
the uninterrupted scalar reference exactly. A second, terminal restart made
zero PES calls, advanced zero steps and did not rewrite checkpoint bytes.

The separate `tests/mpi_materials_batch_integration.py` regression ran with
both two-rank static and three-rank dynamic MPI. It started with replicas
0/2 complete at step 21, replica 1 interrupted and rolled back to step 6,
and replicas 3/4 not started. Both layouts preserved stable replica ownership,
completed checkpoints and exact scalar results. A fully completed restart
succeeded with every PES factory replaced by a function that raises, proving
zero model construction. A corrupted disposable checkpoint produced the same
collective error on every rank rather than leaving peers in different MPI
collectives.

## Complete convergence diagnostic: insufficient sampling

The separate seven-arm study was also executed in full, using a predeclared
short budget to exercise both the numerical and statistical reporting paths.
The base used four replicas, two beads, 16 preparation steps and 64 production
steps at 0.1 fs: only 6.4 fs of production. Independent controls doubled bead
count, halved timestep while preserving physical durations, repeated the
whole slab/adsorbate cell laterally to 112 atoms at the same coverage, doubled
preparation, doubled production, and doubled replica count. Every arm used
its own predeclared seed.

All seven arms and all 32 replica trajectories finished, evaluating 14,080
physical PES frames. Their summed worker wall time was 385.822 seconds. The
207 source/native identities matched both before/after this study and the
successful implementation matrix. There was no execution or data-integrity
failure. The study correctly exited **2** because its scientific gates did
not pass:

| Gate | Passed | Required for a complete pass |
| --- | ---: | --- |
| Independent-replica precision | 10 / 21 | All three observables in all seven arms |
| Production-half stationarity | 10 / 21 | All three observables in all seven arms |
| Effective sampling and complete blocks | 0 / 21 | Each replica has ESS ≥ 20 and at least four complete blocks |
| Control/base equivalence intervals | 10 / 18 | All three observables in all six comparisons |

All H radius-squared precision and stationarity checks failed. The doubled
bead and doubled production arms failed those gates for all three observables.
Each arm's worst estimated ESS was only 2.34–2.50. All six H radius-squared
equivalence comparisons failed, and the doubled-production arm's two energy
difference intervals also extended outside the predeclared ±0.005 eV/atom
margin. Broad uncertainty was not treated as evidence of equivalence.

The report retains `all_arms_complete=true`,
`convergence_checks_passed=false` and `physical_converged=false`. This short
study verifies that an inadequate scientific calculation is reported as
inadequate; it does not validate a thermodynamic limit, long-time transport,
isotope enhancement or the potential's material accuracy. The example also
provides a substantially longer default protocol, whose eventual convergence
must still be assessed from its results.

Reproduce the recorded diagnostic with the same verified table:

```sh
python benchmarks/materials/batch_convergence.py \
  --model /absolute/path/NiAlH_jea.eam.alloy \
  --output /new/path/convergence-diagnostic \
  --replicas 4 --beads 2 --dt-fs 0.1 \
  --equilibration-steps 16 --production-steps 64
```

## Regression and reproduction

The complete local test invocation finished with **7,624 passed, 224 skipped,
52 warnings and 478 subtests passed**, in 567.78 seconds. Optional engines and
explicitly enabled external-model studies account for skips; the real EAM
harness was separately run with its model environment variable and finished
with **4 passed**. Focused materials, provider and parallel groups passed as
recorded in `validation.json`; those overlapping counts are not added to the
full-suite total.

English and Chinese HTML and single-page manuals all passed strict Sphinx
builds with warnings treated as errors. The first CI run identified two
references to Markdown documents outside the curated Sphinx source set;
the references were changed to repository URLs and all four builds were
repeated successfully. [Build receipts](sphinx-builds.json) preserve the
exit codes and log digests.

The existing four-rank MPI integration and expanded five-rank dynamic MPI
integration also passed. Dynamic lifecycle checks cover reuse, opt-out,
stateful providers, task errors, serialization errors, unclosed MPI bead
proxies and close-error propagation. GPU placement checks use a mock provider;
no physical CUDA result or multi-node scaling measurement is claimed.

The local runtime was Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, ASE 3.29.0,
mpi4py 4.1.2 and MPICH 5.0.2. Numerical thread limits were one per worker.
This container required `MPIR_CVAR_CH4_CMA_ENABLE=0` because its kernel does
not provide MPICH's cross-process memory-read transport. That local setting
does not change the library or the Open MPI configuration used by CI.

From an installed checkout with ASE and MPI:

```sh
python benchmarks/materials/batch_acceptance.py --quick \
  --model /absolute/path/NiAlH_jea.eam.alloy --download \
  --output /new/path/acceptance \
  --modes serial,process,batch,process_batch,mpi,mpi_batch,mpi_dynamic,mpi_dynamic_batch \
  --mpi-command 'mpiexec -n 2' --mpi-dynamic-command 'mpiexec -n 3'
```

The model checksum, all registered replicas, oracle checks, call packing,
physical frame conservation, numerical equivalence, restart durability and
unchanged runtime identities are required gates. An execution error retains
the partial report and exits nonzero. CI runs this matrix and uploads the
entire output/evidence directory, including failed runs.
