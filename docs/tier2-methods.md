# Tier 2 method coverage and boundaries

This extension adds force-based sampling actions, spectroscopy workflows,
collective variables, shared bias histories, direct electronic-structure data,
and ML model monitoring. It is a tested numerical reference implementation
with explicit scope. It does **not** claim complete method parity with i-PI or
DL_POLY Quantum, or application-level validation on a molecular material.

## Coverage

| Area | Implemented entry point | Current boundary |
| --- | --- | --- |
| PA-CMD | `task="spectroscopy"`, `method="pa-cmd"`; `PACMDDynamics` | Serial fixed-cell implementation with fictitious normal-mode masses; converge adiabatic separation |
| f-CMD | Centroid force dataset conversion, `fit_quadratic_centroid_pmf`, and `method="f-cmd"` | Explicit quadratic Cartesian centroid PMF, tagged with its fitting temperature, propagated with P=1 |
| QCMD / h-CMD | `ibi_update` is a general distribution-matching helper only | Constrained quasicentroid dynamics, molecular reconstruction and a complete h-CMD/f-QCMD fitting workflow remain unimplemented; these method names are rejected |
| Raman / SFG | Relative Raman invariants and causal complex tensor-component SFG in the spectroscopy workflow | No absolute experimental intensity, optical/Fresnel factors or automatic interface alignment |
| ML property models | Python and TorchScript dipole/polarizability contract with unit conversion and chunked trajectory evaluation | User supplies the trained model, atom order and physical property convention |
| Suzuki–Chin | `task="pimd"` + `[path_integral] method="suzuki_chin"` | Force-bearing finite-difference action; CLI alpha=0 and even-bead potential observable; no SC total-energy/pressure estimator |
| Bosonic exchange | `method="bosonic"`; `bosonic_exchange` | Quadratic Feldman–Hirshberg recurrence for identical groups, no periodic winding sectors |
| PIGS | `method="pigs"`; `PIGSPotential` | Positive endpoint trial, primitive open chain, midpoint potential observable; no fermion nodes or combined exchange/projection action |
| Extra CVs | Dihedral, coordination, aligned RMSD, path `(s,z)`, TorchScript via `[[cvs]]` and Python | Analytic/autograd complete-ring Jacobians; singular geometries fail explicitly |
| PLUMED input | `task="plumed"` + `[plumed] file=...` or inline input | One serial trajectory; MD snapshot does not restore opaque PLUMED kernel state, so automatic resume is rejected |
| Shared MetaD | `task="metadynamics"`, `shared_walkers=true`, `sampling.replicas>=2` | Serial synchronized walkers, one history and one atomic ensemble checkpoint; no asynchronous multi-process file sharing |
| OPES variants | `[opes] mode="metad"|"explore"`, adaptive widths | Explicit Welford/Scott adaptation, not PLUMED's diffusion-window adaptation; uncompressed reference KDE |
| Temperature/Hamiltonian exchange | `ReplicaExchange` Python API | Correct temperature-dependent ring action and momentum rescaling; serial, no new standalone CLI/restart format |
| Biased kinetics | `rescale_biased_time`, `estimate_biased_rate` | Requires audited per-step bias, an unbiased transition region and independent first-passage experiments; no native flooding-exclusion generator |
| RETIS | `ReplicaExchangeTIS` Python API | Shooting and path exchanges between positive interface ensembles; independent flux remains external, no minus ensemble |
| On-the-fly nonadiabatic data | `AdiabaticPythonPotential`, `QDNRPMDDynamics`, `AdiabaticRPSHDynamics` | Callback receives previous geometry/context and returns energies, gradients, NACs and overlaps; no bundled validated QC-package bridge |
| MASH | `MASHDynamics` Python API | Two-state fixed-step endpoint switching; P>1 is an experimental centroid extension, not validated RP-MASH; no event-localized reversible or multistate implementation |
| Nonadiabatic rates | `pyrpmd nonadiabatic-rate INPUT --output DIR` | Signed thermal NRPMD population-flux workflow with its own atomic-unit input schema and diabatic provider; no automatic rate from focused populations |
| ML lifecycle | Nested committee and delta providers; `[active_learning]` for ordinary PIMD/diffusion | Disagreement monitoring and unlabeled candidate collection; training and external reference labeling remain separate |
| Delta learning / RPC | `DeltaLearningPotential.split(P,Pc)` and nested `provider="delta"` input | Slow model is an additive correction; Pc<P samples an explicitly approximate contracted Hamiltonian |

The main guides give the equations, units, checkpoint contracts and examples:

- [Spectroscopy and property surfaces](tier2-spectroscopy.md)
- [Main-engine path-integral methods](tier2-path-integral.md)
- [Collective variables, OPES, exchange and kinetics](tier2-sampling.md)
- [Shared-walker metadynamics](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/enhanced-sampling.md#shared-multiple-walkers)
- [PLUMED input workflow](plumed.md)
- [Direct nonadiabatic data and the rate command](tier2-nonadiabatic.md)
- [Committee monitoring, data collection and delta learning](tier2-ml-lifecycle.md)

## First runs

These short commands exercise analytic providers; none is a converged physical
prediction. Give every new run a different output directory.

```bash
# Fictitious-mass CMD and relative property spectra.
pyrpmd validate examples/spectroscopy/pacmd.inp --check-pes
pyrpmd run examples/spectroscopy/pacmd.inp --output runs/pacmd
pyrpmd analyze runs/pacmd --recompute

# Main-engine force-based Suzuki–Chin, exchange and ground-state projection.
pyrpmd run examples/path_integral/suzuki_chin.inp --output runs/sc
pyrpmd run examples/path_integral/bosonic.inp --output runs/bosons
pyrpmd run examples/path_integral/pigs.inp --output runs/pigs

# Adaptive explore-mode OPES.
pyrpmd run examples/opes/adaptive_explore.inp --output runs/opes-explore

# Independent thermal nonadiabatic workflow; see its distinct input schema.
pyrpmd nonadiabatic-rate examples/nonadiabatic/thermal_rate.inp --output runs/thermal-rate
```

PA-CMD and path-integral sampling are different uses of an extended classical
system. The first supplies a deliberately chosen approximate real-time
centroid dynamics; the new SC/bosonic/PIGS actions only sample equilibrium or
projected distributions. The workflow dispatch keeps these choices separate.

## Persistence and external models

Spectroscopy and the new path-integral actions save full state, clocks, RNG
and the accumulated samples in integrity-checked atomic checkpoints. Shared
MetaD commits the whole walker ensemble together, so restoring only some of
the trajectories cannot change the next common hill batch. Input model files,
including nested potential and TorchScript CV/property files, participate in
the run identity. Changing a source or model file is a new scientific run.

Large-scale streaming, MPI orchestration and all combinations of these
methods are not implied by their availability as separate APIs. In
particular, the new path-integral actions retain their saved prefix in memory.
PLUMED owns history and output side effects outside PyRPMD's MD snapshot;
its input workflow explicitly refuses automatic restart.

ML uncertainty is committee disagreement, not a calibrated error estimate.
Collected configurations retain their predicted quantities and an explicit
unlabeled status. They do not silently become reference training labels.
Likewise, electronic overlaps refer to exactly the previous frame provided
to a QC callback; hiding a package's changing phase convention or state
truncation can change the dynamics.

## What validation establishes

The [extended literature validation report](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/tier2-literature-2026-11-01/README.md)
adds independent quantum and statistical references for published models,
actual seeded trajectories, numerical convergence, uncertainty, and real
optional runtimes. It records the OPES initialization defect exposed by direct
PLUMED comparison and its versioned-checkpoint correction. It also preserves
short-run uncertainty and model discrepancies instead of treating a successful
software test as physical convergence.

The tests include independent nonlinear force differences, Gaussian action
integrals and convergence orders, explicit small-system bosonic recurrences,
harmonic spectra, phase/gauge changes, exact adiabatic/diabatic integrator
equivalence, known bias clocks and censored likelihoods. They also exercise
actual TorchScript autograd, CLI paths, interrupted stochastic restart and
provider ownership. PLUMED command-protocol tests use a controlled mock
kernel when the optional real library is absent.

These tests check numerical and software contracts. A research application
still needs convergence of the imaginary-time discretization, real/sampling
timestep, trajectory duration, statistical uncertainty, potential/property
models and the approximation specific to the chosen method. Missing physical
convergence is not inferred from a passing unit test or a successful CLI run.
