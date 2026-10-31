# Initial-state-selected scattering implementation checks

This record covers the first approximate vibrational NE-RPMD / harmonic QCT
initial-ensemble workflow, its native observations, fixed-energy analysis and
controlled unresolved-trajectory continuation. The branch was updated to
`main` commit `13bc43ddc03a775812023ecaf90f7bf3975ba2f2`, including the PES
diagnostic-buffer ownership fix, before the full regression run.

## Local results

Environment: Python 3.12.14, Linux, a rebuilt C++17/OpenMP extension, NumPy/SciPy
and pytest 9.2.0. The real BKMP2 Fortran PES, MPI runtime, ASE, PyTorch and the
opt-in material potential were not installed in this environment.

| Check | Result |
| --- | --- |
| Full regression suite on the updated base | 1,707 passed; 23 optional-dependency/reference checks skipped; 223 subtests passed |
| Final new-feature suite, including single-sample uncertainty reporting | 114 passed; 70 subtests passed |
| Native extension build | `python setup.py build_ext --inplace` succeeded |
| Both example inputs | CLI validation succeeded |
| CLI workflow | `validate → state-check → run NE → extend → resume → run QCT` succeeded |
| Patch whitespace | `git diff --check` clean |

The full regression initially encountered sandbox restrictions on two existing
loopback i-PI socket tests. Both passed when run with local sockets enabled;
the final full regression used that environment. The final focused run also
covers the subsequent correction that one sample cannot estimate a sample
standard deviation or standard error (`null`, not zero).

Reproduce the focused suite from a checkout with the test dependencies and a
built native extension:

```bash
python -m pytest -q \
  tests/test_initial_states.py \
  tests/test_scattering_analysis.py \
  tests/test_state_selected_native.py \
  tests/test_state_selected_input.py \
  tests/test_state_selected_workflow.py \
  tests/test_state_selected_check.py \
  tests/test_drpmd_extend.py
```

## Scientific and implementation checks

- Analytic harmonic coordinate/momentum covariances, finite-bead energy
  convergence and free fragment-COM ring spread.
- `sqrt(P)` centroid normalization: NE excitation adds `v*omega`; QCT uses
  `(v+1/2)*omega`. Primitive mode energies use the negative spring term.
- Actual isotope masses, projected rigid modes, linear/nonlinear references,
  zero centroid angular momentum and rigid-transform invariance.
- Native and Python observations agree for P=1, 3 and 32. Energy decomposition
  sums to unbiased `H_P/P`. Native collision and state-check tests execute
  compiled C PES fixtures rather than merely asserting backend flags.
- The independent native PES counter reports ten calls for nine force-reusing
  propagation steps, both with and without the new observations. No Python PES
  callbacks occur inside the native block.
- Initial-state and collision draws depend on trajectory ID, not worker/group
  allocation. The selected branch never thermalizes a parent.
- Checkpoint restart preserves the physical state, diagnostics and classifier
  persistence. Invalid prepared reactants and PES failures remain explicit
  failed launches, without rejection-resampling or denominator changes.
- Extensions retain original input/model identity, initial launches and parent
  checkpoints. Tests cover larger/repeated budgets, a crash after checkpoint
  commit, automatic pending-extension recovery and changing worker allocation
  before a later extension. Completed and failed launches are not rerun.
- Fixed-energy channel and initial-state grouping, all-launched denominators,
  unknown-outcome bounds, independent Wilson intervals and eligible parent
  cluster bootstrap. Zero-event cluster bootstrap is not reported as a
  misleading zero-width interval.
- State-check subtracts the common full-PES reference zero; adding a constant
  PES offset does not change the switching-error diagnostic. Mode projections
  occur at sparse observations after C++ blocks return.

## Scope

These are analytic, implementation and compatibility checks. They do not
establish chemical cross-section accuracy for a realistic PES, stationary pure
rovibrational states, final quantum state-to-state assignments, convergence of
32 beads or a universal trajectory duration. The toy Morse examples have no
interfragment reaction potential. A production study still needs reference/PES
consistency, beta/bead/timestep convergence, incoming-state survival, separation,
impact cutoff, classification and finite-time tests. See
[the method guide](../../state-selected.md) for literature and assumptions.
