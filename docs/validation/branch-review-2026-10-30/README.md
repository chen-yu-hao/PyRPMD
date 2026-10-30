# Branch integration and application review

Reviewed all 17 remote branches and PRs #1–#14. All branches except
`perf/audit-fixes-2026-10-30` were ancestors of main `6fde29d`; PRs #1–#13
were already merged. This follow-up completes draft PR #14 at `a7bb586`.

## Changes

- Remove the duplicate periodic-provider validation inside the private factory;
  its sole public caller still validates before constructing a provider.
- Avoid copying a validated force buffer when no bias or fixed atoms require
  mutation. Preserve provider ownership when composing forces. Regression tests
  use reused, read-only buffers and verify actual two-step trajectories.
- Preserve class-level Python overrides in both native propagation APIs. The
  previous dispatch could silently bypass custom Dynamics, integrator, CV and
  bias mathematics. Shared method checks now compare against original methods
  captured in their defining modules; auto falls back, and strict native rejects
  before mutating state. Scientific validation and native buffer checks remain.
- Exercise installed native PILE-L sampling, RNG parity, large-ring normal-mode
  propagation, and block energy checking from both wheel and rebuilt sdist.
  Require the FFT header and new regression suites in package CI.

## Validation

- Full suite: **1396 passed, 13 skipped, 153 subtests passed**; exit status 0.
  The 13 skips are optional BKMP2 (10), external EAM data (1), and Torch (2).
  ASE emits 103 NumPy deprecation warnings.
- Dispatch/CV/sampling tests: 317 passed, 1 optional BKMP2 skip. After adding
  four nested class-override cases, the final dispatch file passed all 32 tests.
- Four actual four-rank MPI suites passed, including FES/rate, materials,
  surface, offline analysis and resume. Surface serial/MPI comparisons matched
  27 arrays/checkpoint files exactly.
- PCET, PIMD, material diffusion, NRPMD and the full surface example each passed
  validate/run/resume/analyze, preserving completed summaries exactly. PCET
  rates match the pre-integration baseline exactly; material coordinate
  differences stay below 3.3e-13 bohr. Diffusion correctly withholds D for its
  ballistic negative control. The surface example propagated 64 child
  trajectories and preserved fixed substrate positions.
- Fresh wheel and source-distribution installations outside the checkout both
  passed package smoke: network (four paths), adaptive FES, standard and
  well-tempered metadynamics, PCET, and dRPMD with two integrators. New PILE-L
  and FFT comparisons cover 5/32/256 beads and matching RNG state.
- The wheel's 52 Python source files were compared byte-for-byte with the final
  production tree. Application tests ran against `a7bb586` plus the follow-up
  edits; accompanying reports record paths and working-tree state, not an
  invented clean-commit identity.

Commands: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q`;
the four `tests/mpi_*` scripts under `mpiexec -n 4`; and
`tests/package_smoke.py --source-root <checkout> --output <fresh-directory>`
using each independent installed environment from a directory outside checkout.

These are finite workflow and numerical consistency checks, not physical
convergence claims. CUDA and optional external potentials were not run locally.
The pre-follow-up PR head's Linux/Torch/real-PES CI checks passed; its macOS
runner was still queued at review time. New-head remote CI is separate from the
local evidence above.

## Further opportunity

Completed PCET resume still solves base/refined/extended proton spectra before
reading completed scan checkpoints. A fingerprinted spectrum cache could remove
that cost, but needs its own invalidation and integrity design; this review does
not replace those safeguards with an unchecked early return.
