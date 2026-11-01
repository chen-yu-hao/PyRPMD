# Nonadiabatic implementation comparison and completed gaps

Audit date: 2026-11-01. The original audit started from PyRPMD revision
`2f907ba96d22cb134198715e398b1d9f8d83d929`. The current column includes the
subsequent workflow, spin-method, hopping, molecular-PES and electronic-backend
extensions. This comparison distinguishes implemented numerical methods,
usable workflows and physical validation.

## What was already present

PyRPMD already implemented focused MMST NRPMD with integrated MInt nuclear
impulses, complex Hermitian SOC matrices, signed SEO thermal preparation,
thermal correlation estimators, MV-RPMD, the original primitive KC-RPMD,
two-state isomorphic-Ehrenfest dynamics, centroid RPSH and electronic friction.
Its state-selected collision workflow already had restart, failed/unresolved
trajectory accounting and signed state-resolved statistics. The audit did not
establish a reason to replace those propagators or claim their features were
absent. See the prior [validation record](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/nonadiabatic-2026-10-31/README.md).

In particular, NQCDynamics' current NRPMD documentation describes separable
nonequilibrium initial-state generation and does not provide its equilibrium
mapping-variable sampling workflow. PyRPMD's existing signed thermal sampler
therefore should not be described as a missing feature just because it is not
part of the focused population workflow.

## Comparison at the workflow level

| Capability | Official comparison evidence | Original audit baseline | Current PyRPMD |
| --- | --- | --- | --- |
| Standard reusable diabatic models | NQCModels provides an analytic model catalog; SHARP provides Tully/spin-boson examples | Teaching/example-specific matrix providers | Tully I–III, general Hermitian LVC, explicit finite-mode spin-boson; immutable parameters, analytic gradients and strict serialized identities |
| Published molecular PES | POTLIB/ChemPotPy supplies the historical NaH₂ 5G matrix surface | A custom matrix provider was needed for a chemically specified molecule | Portable `NaH2LEPS5G` with full Cartesian matrix gradients, a pinned optional ChemPotPy bridge, and the CLI model `nah2_leps_5g_1992` |
| General time-dependent population calculations | NQCDynamics has a distribution/factory to ensemble/reduction/output interface; SQD has model initializers and population output | NRPMD/RPSH kernels and collision/thermal-specific workflows; users wrote model trajectory loops | `task="nonadiabatic"` covers the eight explicit method selectors below, with time series and restart; the original NRPMD/RPSH ensemble API remains available |
| Electronic observables | NQCDynamics distinguishes population estimates and discrete active states; MASH defines operator-pair weights | Selected population accessors and mostly final-state example diagnostics | Signed MMST/W-spin estimates and hopping amplitude densities; RPSH active fractions are separate; MASH CLI reports adiabatic active populations and does not label its propagation spin as a density estimator |
| Nuclear preparation for reduced models | NQCDynamics documents several initial distributions; SHARP has PIMD sampling | Hand-built model states plus other task-specific samplers | Exact finite-P harmonic Gaussian nuclear parents with explicit active coordinates and fixed spectators |
| Uncertainty | Ensemble reductions vary by implementation and do not establish independence automatically | Strong collision/thermal analysis, no general model time-series interface | Per-trajectory histories, independent-trajectory pointwise SEM, real/imaginary coherence errors; one trajectory yields no SEM |
| RPSH formulations | Original BA/DM papers specify different electronic states and hopping currents | Coherent centroid RPSH without EDC | Real BA; bead-wavefunction DM with a common centroid density; optional EDC for CA/BA; consistent DBOC energy/gradient and a stated complex-Hermitian CA/DM extension |
| Spin formulations | SM-NRPMD and RP-MASH have distinct Hamiltonians, sampling and estimators | No dedicated spin-method driver | Two-state W-spin SM-NRPMD with focused preparation; two-state CA-RP-MASH with population importance sampling, resolved equator events and a separate general-correlation API |
| Restart | Competitor object-saving facilities are not evidence of a complete identical restart contract | Complete collision checkpoints; RPSH API required separate state and RNG handling | Workflow saves state, RNG, history and identity, including BA transported frames and MASH sampling labels, with integrity and logical validation |
| Quantum reference comparisons | NQCDynamics' spin-boson example points to literature exact data; this is not evidence of an included exact solver | Extensive local numerical invariants, no standard time-dependent model benchmark | Independent finite-box 1D multistate Fourier reference; analytic and full-Hamiltonian tests, plus a runnable population comparison |
| Electronic acceleration | A GPU label alone does not establish the accelerated operation or a speedup | NumPy electronic propagation and optional native nuclear free-ring steps | Exact fixed-coordinate mapping flow and integrated nuclear impulse through NumPy, native C++/OpenMP or optional Torch; joint trajectory/bead NRPMD batch API; no CUDA hardware result claimed |

The [workflow guide](nonadiabatic-workflow.md), [models](nonadiabatic-models.md)
and [ensemble API](nonadiabatic-ensembles.md) specify the new contracts. All
algorithms use the original complete matrix/positive-gradient provider contract.
The spin methods and advanced hopping use their own states and restartable CLI
workflow; they are not silently accepted by the original two-method
`run_nonadiabatic_ensemble` API.

## Exact CLI method scope

`task="nonadiabatic"` accepts the following `method` values. These are explicit
selectors, not interchangeable names for the same approximation.

| `method` | Electronic propagation and restrictions |
| --- | --- |
| `nrpmd` | Focused MMST mapping, one mapping vector per bead; complex Hermitian matrices |
| `sm-nrpmd` | Two-state W-spin mapping, one spin per bead; complex Hermitian matrices |
| `rp-mash` | Two-state **centroid-approximation** RP-MASH; real matrices, one spin and active surface for the whole ring; adaptive CPU event integration |
| `rpsh` | Original real centroid RPSH, retaining its existing centroid/ring rescaling conventions and optional EDC |
| `rpsh-ca` | Extended centroid RPSH; nondegenerate Hermitian matrices, full-ring hop energy accounting and optional DBOC/Berry force |
| `rpsh-ba` | One common adiabatic coefficient vector driven by bead-averaged energies and `v·d`; real matrices and transported bead frames |
| `rpsh-dm` | One wavefunction per bead, averaged in the common centroid basis for hopping; nondegenerate Hermitian matrices; no additional EDC |
| `soc-rpsh` | Explicit selector for the same extended CA driver as `rpsh-ca`, with complex-Hermitian SOC and the documented local Berry-force/rescaling prescription |

`dboc=true` is an option on the four extended RPSH selectors, not an additional
method name. It corrects both energies and forces and uses the included electronic
states only. The complex prescription is the maximum-real-NAC direction and
nondegenerate Abelian Berry-force ansatz, not a degenerate-subspace or phase-space
surface-hopping implementation. All RPSH variants keep one shared active surface.

Definitions, initialization and observable contracts are in the
[spin-method guide](nonadiabatic-spin-methods.md) and
[hopping-extension guide](nonadiabatic-hopping-extensions.md). For MASH, general
diabatic populations/coherences require the full-sphere operator-correlation API
and its weights; the CLI's population-importance ensemble records only the
corresponding adiabatic active populations. A pure propagation-spin outer product
is not substituted for a MASH density estimator.

## Molecular and acceleration scope

The [molecular guide](molecular-nonadiabatic.md) describes the historical coupled
two-doublet NaH₂ 5G fit for Na + H₂ quenching/reaction, atom order, source identity
and complete Cartesian gradients. This supplies a real molecular matrix PES,
not an on-the-fly electronic-structure service or support for arbitrary
compositions. It contains neither SOC nor higher electronic states. Comparison
with the independently compiled original Fortran establishes implementation
equivalence; it does not revalidate the fit against new electronic-structure
data or certify a converged reaction observable.

The [performance guide](nonadiabatic-performance.md) separates the nuclear and
electronic backends. `dynamics.electronic_backend="numpy"|"native"|"torch"` is
available to the `nrpmd` and `sm-nrpmd` population workflows. Torch can place the
exact fixed-coordinate mapping/impulse kernel on `cpu` or `cuda[:index]` when
available; the host drivers still evaluate molecular PESs and propagate nuclear
coordinates on CPU. `BatchedNRPMDDynamics` separately batches trajectories and
beads in the Python API. It does not turn every CLI method into a batched GPU
trajectory workflow. There is no GPU hopping/MASH kernel, end-to-end GPU claim,
or reported CUDA hardware speedup in the current validation record.

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
   Beyond the supported NaH₂ fit, a system still needs a verified multistate
   provider, a consistent electronic basis, and any derivatives introduced by
   coordinate-dependent basis transformations. The CLI uses a declared model/PES
   registry; method-level Python APIs accept complete custom matrix providers.
2. **Converged physical benchmarks.** The short examples and finite model study
   do not certify excited-state lifetimes, PCET/ET rates, continuum-bath limits
   or realistic conical-intersection dynamics. Beads, timesteps, ensemble size,
   preparation and domain must be converged for the desired observable.
3. **Further formulations and preparation.** SU(N) SM-NRPMD, the SM electronic
   thermal trace and its phase reweighting, bead-approximation RP-MASH, multistate
   or complex-SOC MASH, non-Abelian degenerate-state hopping and phase-space
   surface hopping remain outside the implementations. The implemented DBOC
   and Berry-force combinations retain their stated approximation limits.
4. **General correlated or signed ensembles.** The new workflow is unweighted
   and independently initialized. Arbitrary mixed/coherent MMST preparation,
   correlated nuclear/electronic parents and multi-chain signed thermal
   inference remain separate extensions. Existing thermal SEO/jackknife APIs
   retain their own normalization and sign diagnostics. The MASH full-sphere
   operator-correlation API is implemented separately; it is not the CLI's
   unweighted population output.
5. **Further acceleration and validation.** MPI bead splitting, GPU PES/nuclear
   propagation and GPU hopping/event kernels remain unavailable. The native
   mapping contraction is not a fused native full-trajectory loop. Optional
   Torch/CUDA support needs its dependencies and hardware; CUDA accuracy and
   performance have not been established by the local CPU validation.

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
- D. Bossion, S. N. Chowdhury and P. Huo, *J. Chem. Phys.* **154**, 184106 (2021), [doi:10.1063/5.0051456](https://doi.org/10.1063/5.0051456), W-spin SM-NRPMD.
- Z. Cao, V. A. Suarez and J. S. Kretchmer, [doi:10.26434/chemrxiv-2025-xps2g](https://doi.org/10.26434/chemrxiv-2025-xps2g), 2025 RP-MASH preprint; the implemented variant is CA-RP-MASH.
- P. Shakya, R. Welsch, L. Inhester and R. Santra, *Phys. Rev. A* **107**, 062810 (2023), [doi:10.1103/PhysRevA.107.062810](https://doi.org/10.1103/PhysRevA.107.062810), BA definitions and RPSH-DM.
- D. K. Limbu et al., *J. Chem. Phys.* **163**, 234102 (2025), [doi:10.1063/5.0301249](https://doi.org/10.1063/5.0301249), DBOC in RPSH.
- G. Miao, N. Bellonzi and J. E. Subotnik, *J. Chem. Phys.* **150**, 124101 (2019), [doi:10.1063/1.5088770](https://doi.org/10.1063/1.5088770), the stated complex-NAC and Berry-force prescription.
- P. Halvick and D. G. Truhlar, *J. Chem. Phys.* **96**, 2895–2909 (1992), [doi:10.1063/1.461986](https://doi.org/10.1063/1.461986), with [1994 erratum](https://doi.org/10.1063/1.467282), the NaH₂ fit.
