# Adaptive string and restricted electrode/NRPMD integration

This pilot now has executable numerical checks, but it is **not** a validated
end-to-end constant-potential nonadiabatic rate workflow. The mean-force string,
restricted electrode/matrix adapter and literature-linked PCET regression have
separate evidence and must not be conflated.

## String geometry

`pyrpmd.adaptive_string.adaptive_mean_force_string(images, gradient)` relaxes
interior images perpendicular to the tangent, redistributes by arc length and
inserts images at bends. Endpoints are fixed. It requires a consistent gradient
of free energy in scaled Cartesian or constant-Euclidean-metric CVs. This API
does not implement the position-dependent metric of general nonlinear CVs.
A PES gradient gives an MEP, not a nuclear quantum free-energy path.

`iterations` now counts completed updates; `history` has the initial residual
plus one entry per update. The reported residual ALWAYS belongs to the returned
images. Previously, exhausting the budget returned an updated path with the
previous path's residual and could miss convergence on the last update. Both
cases now have regressions. Invalid boolean/complex controls, complex gradients,
degenerate tangents and degenerate reparameterization are explicitly rejected.
The convergence flag is geometric, not a statistical sampling guarantee.

## Derived restricted coupling

The new `pyrpmd.electrochemistry.diabatic` adapter assumes the model

\[
\mathbf H(R,N)=\mathbf V_{\mathrm{rel}}(R)+E_{\mathrm{el}}(R,N)\mathbf I.
\]

Here the charging coordinate is **independent of the local electronic state**;
`V_rel` must exclude all electrode energy and reservoir work. With a fast,
adiabatic, convex electrode, elimination at fixed chemical potential gives

\[
\Omega_{\mathrm{el}}(R;\mu)=\min_N[E_{\mathrm{el}}(R,N)-\mu N],\qquad
\mathbf V_{\mathrm{eff}}=\mathbf V_{\mathrm{rel}}+\Omega_{\mathrm{el}}\mathbf I.
\]

The matrix gradient adds `-electrode_forces` on the diagonal, with no extra
`dN*/dR` term at an interior stationary point. Existing per-frame electrode
minimization runs independently for each bead, preserving the PES batch contract.
For this stated decomposition the scalar term is not double counted. Applying
the adapter to an arbitrary full matrix already containing electrode terms is
incorrect and cannot be detected from numerical arrays alone.

```python
from pyrpmd.electrochemistry.diabatic import StateIndependentElectrodeDiabaticPotential
from pyrpmd.thermal_nrpmd import ThermalNRPMDSampler, ThermalNRPMDDynamics

# relative_matrix_pes contains NO electrode terms.
# electrode is an existing AdiabaticElectrodePotential with a calibrated reservoir.
potential = StateIndependentElectrodeDiabaticPotential(relative_matrix_pes, electrode)
sampler = ThermalNRPMDSampler(potential, masses, beta, q_initial, nstates=2, seed=41)
parents = sampler.sample(100, burn_in=1000, stride=10)
dynamics = ThermalNRPMDDynamics(potential, masses, beta, len(q_initial), dt,
                                nstates=2)
# Same provider instance; thermal scalar/traceless propagation is essential.
child = parents.states[0].copy()
dynamics.step(child)
```

This is not state-dependent charging, a wide-band metal reservoir, a charge
exchange dynamics model, shared-N grand-canonical sampling or extended TP-RPMD.
The example sample counts are argument illustrations, not convergence advice.
The adapter is for finite Cartesian systems, owns neither provider, and does not
add a new CLI task. Complex local matrices can be composed, but the existing
thermal sampler still rejects complex phase sampling.

## Run the evidence

From a source checkout with PyRPMD and pytest installed:

```bash
python -m pytest tests/test_adaptive_string.py tests/test_electrode_diabatic.py -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/electrochemical_string/validate.py --output runs/electrode-string.json
```

The benchmark records source SHA256 hashes and separates:

* Exact convex charging and complete matrix-gradient finite differences.
* Coupled four-bead trajectories versus independently coded Hamilton equations
  integrated with DOP853. This is an integrator reference, NOT exact quantum dynamics.
* An analytic two-level coherent-population limit with constant electronic
  matrix and a moving nuclear ring. Its finite phase quadrature is deterministic;
  small errors here do not validate dissipative kinetics or nuclear-electronic coupling.
* A separate analytic Euclidean string test, NOT a sampled electrochemical PMF.
* Fresh golden-rule PCET rates versus archived outputs from the official pyPCET
  commit `031a679eab50e210aa7f76698ef5f302de31a460`. The reference is NOT rerun by
  this offline script, and these rates do NOT come from the new NRPMD adapter.

The PCET regression uses the same 300 K harmonic model, H/D masses, potential
conventions, 0.7 eV reorganization energy and 0.001 eV electronic coupling as
[the archived comparison](validation/pcet-model-2026-10-30/README.md). It is a
paper-author-code comparison, not a reproduction of a paper's experimental table.
Results and execution qualifications are in the
[continuation record](validation/electrochemical-string-2026-11-01/README.md).

## Remaining acceptance gates

A sampled quantum mean-force oracle with correlated errors, metric-correct CVs,
and independent adaptive/frozen production stages is still needed. For actual
electrochemical charge-transfer kinetics, the electrode/state coupling and
reservoir dynamics must be derived explicitly, then compared with matched
HEOM/MCTDH or other numerically exact dynamics. Independent seeds, bead/timestep
convergence, signed normalization and a justified kinetic plateau are required
before reporting a rate. The evidence retains `physical_converged=false` and
`combined_rate_available=false`.

## Primary references and scope

Richardson and Thoss, J. Chem. Phys. 139, 031102 (2013),
DOI 10.1063/1.4816124: NRPMD framework, not validation of this electrode model.

Maragliano et al., J. Chem. Phys. 125, 024106 (2006),
DOI 10.1063/1.2212942: collective-variable string methods; a general nonlinear
metric is outside this pilot.

Reed, Lanning and Madden, J. Chem. Phys. 126, 084704 (2007),
DOI 10.1063/1.2464084: constant-potential electrode methodology; its ionic-liquid
simulation is not reproduced here.

Hutchison, Cui, Zhong and Hammes-Schiffer, J. Chem. Phys. 163, 091501 (2025),
DOI 10.1063/5.0284337: nonadiabatic PCET tutorial and associated pyPCET package.
