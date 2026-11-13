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
`QuantumCorrectionResult` therefore exposes diagnostics; independent gates
must be checked before interpreting the result as physically validated.

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

## Online RPMD trajectory with a quantum wavefunction

`OnlineQuantumRPMD` provides the online version of the small-system reference
workflow. At every time step it propagates an RPMD ring polymer and a
normalized one-dimensional wavefunction on the same clock. The quantum
potential is evaluated at the ring-polymer centroid. The ring polymer uses the
standard RPMD Hamiltonian sampled at `beta/P` (full physical force on every
bead), and the centroid-coupled quantum energy enters it once per bead, so every
bead receives the full Ehrenfest force `-<dV_q/dq_c>`; `total_energy` is the
conserved `H_P` including `P*<H_q>`:

```python
from pyrpmd.analysis import OnlineQuantumRPMD

runner = OnlineQuantumRPMD(
    grid, quantum_mass=1.0, beta=4.0, dt=0.002,
    positions=q_beads, momenta=p_beads, bath_masses=masses,
    quantum_potential=lambda x, centroid: V_q(x, centroid),
    quantum_gradient=lambda x, centroid: dV_dcentroid(x, centroid),
    bath_force=lambda q: F_bath(q),
    wavefunction=psi0,
)
trajectory = runner.propagate(10_000, sample_stride=10)
```

`positions` and `momenta` are `[bead, coordinate]` arrays. `trajectory` keeps
the synchronized positions, momenta, wavefunctions, quantum force, energy
diagnostics, and norm history. Each step is transactional: a callback failure
restores the complete classical and quantum state. The implementation is a
Python reference driver with centroid coupling and a frozen-potential
Fourier--Strang step inside each mixed-dynamics step. It is therefore an
online mixed quantum--classical closure, not an exact entangled
system-plus-bath propagator. Bead, grid, time-step, and ensemble checks must
be performed independently.

## Automatic quantum-subspace selection

`select_quantum_subspace` ranks coordinates using mass-weighted force features
that include linear terms, squares, and pair products. It combines

- non-affine force strength;
- linear and nonlinear force coupling to residual coordinates; and
- correlations of target observables with the same nonlinear features.

`joint_coordinates=[(i, j), ...]` declares coordinates that must enter the
quantum subspace together. The returned `QuantumSubspaceSelection` contains
the selected basis, each score component, the force-model residual, the
coupling matrix, and an explicit provenance record:

```python
selection = select_quantum_subspace(
    positions, forces, max_dimension=2,
    joint_coordinates=[(0, 1)],
    target_observables=target_features,
)
print(selection.selected_indices)
print(selection.diagnostics["coupling_matrix"])
```

The selector is an auditable ranking heuristic. Its score is not an error
bound and it does not prove that a low-dimensional inverse-response defect is
transferable. A nested scan in which the selected subspace is enlarged and the
target defect is recomputed remains required. The earlier `select_subspace`
and `memory_embed` functions remain available for the experimental
harmonic-memory closure; that closure is restricted to a stable linear
residual bath.

## Flux--side and population-rate correction

For a rate calculation, first form a matched flux--side object from independent
replicas:

```python
from pyrpmd.analysis import flux_side_rate, hybrid_flux_side_correction

rpmd_full = flux_side_rate(time, F_full, h_full, reactant_population,
                           bootstrap=1000, rng=rng)
rpmd_sub = flux_side_rate(time, F_sub, h_sub, reactant_population,
                          bootstrap=1000, rng=rng)
quantum_sub = flux_side_rate(time, F_q, h_q, reactant_population,
                             bootstrap=1000, rng=rng)
corrected = hybrid_flux_side_correction(time, rpmd_full, rpmd_sub, quantum_sub)
```

`flux_side_rate` keeps the reactant population explicit, resamples complete
replicas rather than time frames, and records plateau diagnostics. The
correction transfers an additive rate defect, a regularized inverse defect, or
(`mode="ratio"`) a multiplicative defect `k_full * k_Q,sub / k_RP,sub`. Prefer the
ratio for rates: it is invariant to a prefactor shared by the two subspace
rates, whereas the additive form assumes the subspace model and the full system
have the same absolute rate. For thermal rate constants with an exact 1-3D
subspace calculation, see [Subspace rate correction](subspace-correction.md). It has the same algebraic limits as the response workflow: zero defect
returns the full RPMD curve, and passing the full system in all three inputs
returns the quantum curve exactly. These are identities of the estimator, not
proofs of physical exactness.

There is no generally exact flux--side rate correction for an arbitrary coupled
system that can be obtained from a lower-dimensional quantum calculation. A
physical rate still needs a converged dividing-surface measure, reactant
normalization, a real-time plateau, independent-chain uncertainty, and an
independent full-system quantum benchmark. The API therefore exposes
`exact_general_rate_claim=False` instead of silently promoting the transfer
approximation to a theorem.
