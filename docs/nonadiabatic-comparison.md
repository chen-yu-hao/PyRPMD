# Nonadiabatic implementation comparison and completed gaps

Audit date: 2026-11-01. The starting PyRPMD revision was
`2f907ba96d22cb134198715e398b1d9f8d83d929`. This comparison distinguishes
implemented numerical methods, usable workflows and physical validation.

## What was already present

PyRPMD already implemented focused MMST NRPMD with integrated MInt nuclear
impulses, complex Hermitian SOC matrices, signed SEO thermal preparation,
thermal correlation estimators, MV-RPMD, the original primitive KC-RPMD,
two-state isomorphic-Ehrenfest dynamics, centroid RPSH and electronic friction.
Its state-selected collision workflow already had restart, failed/unresolved
trajectory accounting and signed state-resolved statistics. The audit did not
establish a reason to replace those propagators or claim their features were
absent. See the prior [validation record](validation/nonadiabatic-2026-10-31/README.md).

In particular, NQCDynamics' current NRPMD documentation describes separable
nonequilibrium initial-state generation and does not provide its equilibrium
mapping-variable sampling workflow. PyRPMD's existing signed thermal sampler
therefore should not be described as a missing feature just because it is not
part of the focused population workflow.

## Comparison at the workflow level

| Capability | Official comparison evidence | Starting PyRPMD | Completed in this change |
| --- | --- | --- | --- |
| Standard reusable diabatic models | NQCModels provides an analytic model catalog; SHARP provides Tully/spin-boson examples | Teaching/example-specific matrix providers | Tully I–III, general Hermitian LVC, explicit finite-mode spin-boson; immutable parameters, analytic gradients and strict serialized identities |
| General time-dependent population calculations | NQCDynamics has a distribution/factory to ensemble/reduction/output interface; SQD has model initializers and population output | NRPMD/RPSH kernels and collision/thermal-specific workflows; users wrote model trajectory loops | Independent NRPMD/RPSH ensemble API and `task="nonadiabatic"` with the existing CLI cycle |
| Electronic observables | NQCDynamics distinguishes population estimates and discrete active states | Selected population accessors and mostly final-state example diagnostics | Fixed-diabatic complex density estimates/coherences; RPSH adiabatic amplitude populations and active fractions remain distinct |
| Nuclear preparation for reduced models | NQCDynamics documents several initial distributions; SHARP has PIMD sampling | Hand-built model states plus other task-specific samplers | Exact finite-P harmonic Gaussian nuclear parents with explicit active coordinates and fixed spectators |
| Uncertainty | Ensemble reductions vary by implementation and do not establish independence automatically | Strong collision/thermal analysis, no general model time-series interface | Per-trajectory histories, independent-trajectory pointwise SEM, real/imaginary coherence errors; one trajectory yields no SEM |
| RPSH decoherence | SHARP has a decoherence option; method names must still be checked | Coherent centroid RPSH without a decoherence option | Optional Granucci–Persico EDC using explicitly documented centroid quantities; default remains `none` |
| Restart | Competitor object-saving facilities are not evidence of a complete identical restart contract | Complete collision checkpoints; RPSH API required separate state and RNG handling | Model workflow atomically saves state, RNG, history and identity, with integrity and logical validation |
| Quantum reference comparisons | NQCDynamics' spin-boson example points to literature exact data; this is not evidence of an included exact solver | Extensive local numerical invariants, no standard time-dependent model benchmark | Independent finite-box 1D multistate Fourier reference; analytic and full-Hamiltonian tests, plus a runnable population comparison |

The [workflow guide](nonadiabatic-workflow.md), [models](nonadiabatic-models.md)
and [ensemble API](nonadiabatic-ensembles.md) specify the new contracts. All
algorithms use the original complete matrix/positive-gradient provider contract.

The [executed validation record](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/nonadiabatic/VALIDATION.md)
lists targeted regression results, CLI runs and the independent quantum smoke
comparison, with explicit local environment and physical-convergence limits.

## Why these changes were prioritized

A reusable model and well-defined preparation allow two methods to be compared
on the same problem. Complete time histories reveal electronic coherence and
population transfer that a final-state-only output cannot show. Retaining
individual trajectory contributions makes errors and independent-seed studies
auditable. Atomic state/RNG/history restart prevents a resumed calculation from
silently sampling a different stochastic process.

The new quantum reference is independent of the NRPMD/RPSH propagators. It has a
finite periodic box, finite spatial grid and finite timestep. Its numerical
sensitivity is reported separately from Monte Carlo uncertainty and approximate
method differences. Agreement with a reference on one model is not a universal
accuracy guarantee; numerical identities and physical approximation errors have
different acceptance criteria.

## Remaining research and engineering scope

1. **General on-the-fly electronic structure.** A scalar ASE/PySCF/i-PI adapter
   does not supply a complete diabatic Hamiltonian and all matrix gradients.
   Real-system use still needs a supported multistate provider, a consistent
   electronic basis, and any derivatives introduced by coordinate-dependent
   basis transformations. The new CLI uses the declared analytic model catalog;
   the Python ensemble API accepts custom matrix providers.
2. **Converged physical benchmarks.** The short examples and finite model study
   do not certify excited-state lifetimes, PCET/ET rates, continuum-bath limits
   or realistic conical-intersection dynamics. Beads, timesteps, ensemble size,
   preparation and domain must be converged for the desired observable.
3. **Other electronic formulations.** SM-NRPMD, RP-MASH, RPSH-DM, bead-approximation
   RPSH, DBOC and complex-SOC surface hopping are distinct methods. EDC is not
   a substitute for them. Adding them requires their own equations, initial
   distributions, estimators and independent references.
4. **General correlated or signed ensembles.** The new workflow is unweighted
   and independently initialized. Arbitrary mixed/coherent MMST preparation,
   correlated nuclear/electronic parents and multi-chain signed thermal
   inference remain separate extensions. Existing thermal SEO/jackknife APIs
   retain their own normalization and sign diagnostics.
5. **Electronic acceleration.** Local trajectory processes are supported and
   the free ring can use native kernels. This change does not add MPI bead
   splitting, fused native electronic propagation or GPU matrix trajectories.

These are explicit scope boundaries, not features emulated by placeholders.

## Primary sources

- [NQCDynamics NRPMD implementation and initial-distribution scope](https://nqcd.github.io/NQCDynamics.jl/stable/dynamicssimulations/dynamicsmethods/nrpmd/).
- [NQCModels analytic catalog](https://nqcd.github.io/NQCDynamics.jl/stable/NQCModels/analyticmodels/).
- [NQCDynamics ensembles and reductions](https://nqcd.github.io/NQCDynamics.jl/stable/ensemble_simulations/).
- [NQCDynamics output definitions](https://nqcd.github.io/NQCDynamics.jl/stable/api/NQCDynamics/dynamicsoutputs/).
- [NQCDynamics spin-boson example](https://nqcd.github.io/NQCDynamics.jl/stable/examples/spinboson/).
- [Mandal group SQD](https://github.com/mandalgrouptamu/SQD).
- [SHARP Pack features](https://sharppack.readthedocs.io/en/latest/intro.html) and [2.0 source](https://github.com/fashakib/SHARP_pack_2).
- [Libra releases](https://github.com/Quantum-Dynamics-Hub/libra-code/releases): KC-RPMD is a distinct supported development, not a synonym for mapping NRPMD. The release statement alone does not establish a complete multi-bead atomistic workflow.
- J. C. Tully, *J. Chem. Phys.* **93**, 1061 (1990), [doi:10.1063/1.459170](https://doi.org/10.1063/1.459170), Eqs. 21, 23 and 24 for the three model potentials.
- G. Granucci and M. Persico, *J. Chem. Phys.* **126**, 134114 (2007), [doi:10.1063/1.2715585](https://doi.org/10.1063/1.2715585), Eq. 17 for EDC; the centroid RPSH adaptation is specified separately in the guide.
