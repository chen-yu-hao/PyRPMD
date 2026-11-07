# Exact quantum references and controlled RPMD correction

PyRPMD separates two tasks that are often conflated:

1. an **exact-reference calculation** for a small quantum system, where finite
   grid, time step, bath expansion and hierarchy-depth errors are independently
   converged; and
2. a **controlled hybrid correction** that transfers a low-dimensional
   quantum-minus-RPMD response defect to a full RPMD calculation.

The first task is implemented by `DVRSystemHEOM`, `SpinBosonHEOM`, and the
finite-grid solvers in `pyrpmd.analysis.quantum_reference`. The second task is
implemented by `run_quantum_correction`.

## Response-level estimator

For a selected observable (or a matched matrix of observables), compute three
Kubo correlations on the same time grid:

```python
import numpy as np
from pyrpmd.analysis import run_quantum_correction

time = np.linspace(0.0, 40.0, 4001)
frequency = np.linspace(0.02, 6.0, 600)

result = run_quantum_correction(
    time,
    frequency,
    rpmd_full_correlation=C_full_rpmd,
    rpmd_sub_correlation=C_sub_rpmd,
    quantum_sub_correlation=C_sub_exact,
    beta=4.0,
    damping=0.02,
    inverse_ridge=0.0,
    output_ridge=0.0,
    provenance={
        "full_reference": "32-bead RPMD",
        "subspace": ["x", "b1"],
        "quantum_reference": "DVR/HEOM",
    },
)

corrected_spectrum = result.response
print(result.diagnostics)
```

The function first forms the causal response

\[
\chi(t)=-\beta\,\frac{dC(t)}{dt},
\]

and then uses the finite-window convention

\[
\widetilde\chi(\omega)=\int_0^{t_{\max}}
e^{(i\omega-\eta)t}\chi(t)\,dt.
\]

The t=0 point has half trapezoid weight. The hybrid response is assembled as

\[
\widetilde\chi_{\rm mix}^{-1}(\omega)=
\widetilde\chi_{\rm RP,H}^{-1}(\omega)+
\left[\widetilde\chi_{\rm Q,A}^{-1}(\omega)-
\widetilde\chi_{\rm RP,A}^{-1}(\omega)\right].
\]

`inverse_ridge` and `output_ridge` are optional Tikhonov stabilizers. They are
zero by default so that the algebraic zero-defect and full-subspace limits are
preserved for nonsingular responses. A positive ridge is part of the estimator
and must be scanned as a numerical parameter.

## What is and is not exact

The exact-reference layer is deterministic for its finite numerical model. It
does not make the finite grid, hierarchy, exponential bath fit, or time window
physically converged automatically. `DVRHEOMResult` and
`QuantumCorrectionResult` therefore expose diagnostics and keep
`physical_converged=False` until independent gates have passed.

The hybrid response is a transfer approximation. It becomes the full quantum
response in the algebraic test where the selected subspace is the full system,
and it reduces to RPMD when the inverse defect is zero. Those limits do not
prove accuracy at an intermediate subspace dimension, and the estimator is not
a general exact reaction-rate identity. Reaction rates still require a
flux-side or population observable, reactant normalization, plateau checks and
their own exact reference.

## Required convergence record

Before interpreting a corrected spectrum or rate, record at least:

- the common time and frequency grids, damping and ridge values;
- RPMD beads, trajectory count and block/bootstrap uncertainty;
- quantum grid, propagation step, hierarchy depth, exponential count and
  omitted-weight or boundary diagnostics;
- condition-number and effective-rank diagnostics for all three inverses;
- subspace labels and the exact observable definition;
- the zero-defect, full-subspace and separable-limit checks.

The workflow intentionally does not perform automatic smoothing, spectral
clipping, normalization, or Hermitization. Such operations would hide whether
the approximation or the numerical reference is failing.

## Three-dimensional stress test

`examples/three_d_quantum_correction.py` runs a coupled three-dimensional
harmonic oscillator in an `8^3` normal-mode Fock basis. It compares a finite
spectral Kubo quantum reference with the analytic classical quadratic
correlation, transfers an `x^2` defect to `x^2+y^2+z^2`, and checks the
full-subspace algebraic limit. The reproducibility record is stored in
`validation/2026-11-07-qe-rpmd-3d/README.md`.
