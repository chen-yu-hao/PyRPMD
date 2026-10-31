# Materials extensions validation — 2026-10-31

This record covers isotropic BZP NPT PIMD, fixed atoms in the materials and
metadynamics workflows, a one/two-dimensional metadynamics grid, and streaming
materials trajectories. The implementation starts from main commit
`76dd6ad24adc6c244f775857987cd2f0e119845f`.

## Software checks

| Check | Result |
| --- | --- |
| Full local regression suite with the compiled extension and ASE | **1,693 passed**, **153 subtests passed**, 14 skipped, 2 deselected; 227.28 s |
| The two socket tests, run separately with localhost access | **2 passed**, 17 deselected |
| Focused grid, metadynamics and fixed-surface suite | **100 passed** |
| Independent grid review regressions after the final checkpoint-clock guard | **10 passed** |
| English and Chinese Sphinx HTML builds, warnings as errors | Passed |
| English and Chinese Sphinx single-page HTML builds, warnings as errors | Passed |
| Source distribution and compiled wheel | Built successfully |
| Installed-wheel smoke test outside the checkout | Native extension imported; five production modules matched source SHA256; public NPT run produced five frames |

The focused suites overlap the full regression run; these counts must not be
added. The final independent grid run includes the last hill-timestamp
validation change. The full regression command was:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q \
  -k 'not test_ipi_protocol and not test_ipi_real_cell_inverse_and_virial_wire'
```

The local environment used Python 3.12.14, NumPy 2.5.4, SciPy 1.17.0 and pytest
9.2.0. Additional dependency directories were included in the local PYTHONPATH.
The skips were optional real BKMP2 checks (10), the opt-in real Ni-Al-H EAM
check (1), an optional independent 2D PMF reference solver (1), and optional
PyTorch checks (2). MPI was not installed locally; the updated MPI materials
smoke test is exercised by GitHub CI. ASE issued upstream NumPy shape-assignment
deprecation warnings, with no failing tests.

## What the tests establish

### Constant pressure

`tests/test_barostat.py` checks the physical virial and pressure against finite
differences under centroid/cell dilation; bead normalization at 1, 4 and 16
beads; reversibility with the supported deterministic particle integrators;
second-order timestep convergence of the extended invariant; and exact
restart, cell propagation, provider unit conversion and failure rollback.
Separate ideal-gas tests at one and four beads check the known NPT volume mean,
variance and mean pressure after equilibration. Both tests use 26,000 steps,
discard the first 2,000, and sample every 10 steps. This also checks the
unconstrained-center-of-mass volume-measure correction.

The executable [NPT ideal-gas input](../../../examples/materials/npt_free.inp)
passed `validate --check-pes`, ran two replicas with 401 saved frames each, and
resumed successfully. The completed restart preserved checkpoint bytes. This
short pilot is separate from the statistical ideal-gas regression tests.

### Fixed atoms

Tests check fixed coordinates on every bead, zero fixed momenta, correct mobile
degrees of freedom in kinetic estimators, zero fixed-atom ring radius, diffusion
selection/reference handling, and rejection of corrupt completed checkpoints.
Metadynamics tests combine fixed substrates, partial PBC and all three
deterministic particle integrators. The
[fixed-slab PIMD input](../../../examples/materials/fixed_slab.inp) passed
validation, two-replica execution (81 frames each), and restart.

### Grid metadynamics

The [grid report](grid-metadynamics.md) records numerical errors, restart and
failure tests, benchmark methodology and limitations. The
[raw benchmark](grid-bias-benchmark.json) retains repeat timings, source hashes
and runtime details. At 10,000 hills, the measured bias-only process-CPU speedup
was 40.99 in one dimension and 20.10 in two dimensions. The two-dimensional
grid was slower at 100 hills. These results do not measure whole-simulation
speedup.

The [fixed-slab grid input](../../../examples/metadynamics/grid_fixed_slab.inp)
passed input/PES validation, 500 steps (20 hills and 80 saved samples), and
restart. Numerical tests compare the interpolated energy and analytic gradient
against direct hill sums and independent finite differences, including periodic
seams, nonlinear CVs, well-tempered deposition, signed interpolation undershoot,
cache corruption and history-only reconstruction.

### Streaming storage

Tests compare memory/stream trajectories and summaries, exact interrupted
continuation, both legacy-format migration directions, compact worker results,
atomic chunk/checkpoint failure recovery, corrupt or truncated chunks, orphan
chunks and completed restarts that do not construct the PES. The stress case
stores 5,000 frames for 128 atoms with a 32-frame buffer; approximately 15 MB of
logical trajectory data stays below 4 MiB of traced allocation during storage.

The storage descriptor has fixed size and uses immutable hash-linked chunks.
The storage path does not hold the full trajectory in memory. Summary analysis
still retains scalar time series (and selected atoms for diffusion), so total
analysis memory is not independent of the number of saved frames. Explicit
`return_data=True` materializes complete histories.

## Pilot outputs and reproducibility

[Materials smoke summaries](materials-smoke-summaries.json) contain the observed
NPT and fixed-slab results, checkpoint sizes, base revision and production-source
hashes. All examples use analytic teaching potentials. Their outputs retain
`physical_converged = false` and do not establish production material properties.

Typical reproduction commands from an installed checkout are:

```bash
pyrpmd validate examples/materials/npt_free.inp --check-pes
pyrpmd run examples/materials/npt_free.inp --output runs/materials-npt-free
pyrpmd resume runs/materials-npt-free
pyrpmd validate examples/materials/fixed_slab.inp --check-pes
pyrpmd run examples/materials/fixed_slab.inp --output runs/materials-fixed-slab
pyrpmd resume runs/materials-fixed-slab
pyrpmd validate examples/metadynamics/grid_fixed_slab.inp --check-pes
pyrpmd run examples/metadynamics/grid_fixed_slab.inp --output runs/meta-grid-fixed-slab
pyrpmd resume runs/meta-grid-fixed-slab
```

## Scientific and operational scope

- BZP is isotropic equilibrium PIMD with full three-dimensional PBC, all atoms
  mobile, an explicit variable-cell provider and physical per-bead virial.
  Independent trajectory processes are supported; bead pools, MPI NPT,
  anisotropic cells, fixed-layer NPT and transport under NPT are not implemented.
- Fixed atoms are supported in fixed-cell PIMD, diffusion and metadynamics.
- The optional metadynamics grid has one or two dimensions. Nonperiodic bounds
  are hard validity limits; crossing them raises an error and restores the last
  accepted step. Grid spacing requires its own accuracy check.
- Trajectory streaming is the default for materials PIMD and diffusion.
  Metadynamics hill history and its existing checkpoints still grow with time.
- Numerical regression and analytic-model sampling checks do not replace bead,
  timestep, cell-size, potential or statistical convergence for a real material.

Algorithm details and references are in [constant-pressure PIMD](../../materials-npt.md),
[grid metadynamics](../../grid-metadynamics.md), and
[streaming trajectories](../../streaming-trajectories.md).
