# Capabilities and validation boundaries

The [documentation index](../index.md) is the starting point for a calculation.
This page collects the detailed capability, limitation and validation material
previously presented in the repository README. The linked guides remain the
source of equations, accepted settings, examples and checkpoint contracts.
Working software, a successful pilot and a physically converged prediction are
different milestones; follow [Scientific conventions](science.md) before using
an estimate in a research conclusion.

## Choose the appropriate workflow

| Calculation | What the workflow provides | Where to continue |
| --- | --- | --- |
| Thermal gas-phase rates | Transition-state or endpoint preparation, reaction-channel checks, umbrella/frozen-bias/external PMFs, constrained parent sampling and paired recrossing children | [Reaction workflows](workflows.md), [PMF selection](../rate-pmf-selection.md), [transition-state search](../transition-state-search.md) |
| Rate precision | Separate pilot and frozen independent production, prespecified checks, statistical/numerical acceptance and an enforced PES-frame budget | [Precision controller and CLI](../rate_precision.md) |
| Scattering and reaction networks | dRPMD collisions, approximate vibrational-state NE-RPMD and P=1 QCT, unresolved-trajectory continuation, entrance/channel/TS rate matrices | [Reaction workflows](workflows.md), [state-selected scattering](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/state-selected.md) |
| Free energies and reaction paths | 1D–3D umbrella reconstruction, overlap/ESS diagnostics, adaptive windows, metadynamics, OPES, REUS, ABF/eABF, driven work and restrained CV strings | [Enhanced sampling](../enhanced-sampling.md), [extended CVs](../tier2-sampling.md), [finite-temperature strings](../finite-temperature-string.md) |
| Rare events | Fixed-ring committor shots; TIS/RETIS with a minus ensemble and internal flux, REPPTIS and direct FFS | [Committor analysis](../committor.md), [path sampling](../transition-interface-sampling.md) |
| Materials and surfaces | Equilibrium PIMD, supported NPT ensembles, diffusion, fixed layers, site/surface escape rates, estimator and trajectory output | [Materials and surfaces](periodic.md), [condensed-phase NQE](../condensed-phase-nqe.md) |
| Spectroscopy | PA-CMD, quadratic f-CMD, experimental f-QCMD for bent triatomics, TRPMD and relative IR/Raman/SFG using supplied property models | [Property spectra](../tier2-spectroscopy.md), [f-QCMD](../qcmd.md), [TRPMD](../trpmd-spectroscopy.md) |
| Low-temperature quantum effects | Molecular instanton optimization, rate/splitting convergence scans, Hessian reuse, Gaussian fluctuation prefactors and normal-regime golden-rule instantons | [Instantons](../instanton.md), [low-temperature methods](../low-temperature.md), [quantum estimators](../quantum-statistics.md) |
| Electronic dynamics and electrochemistry | NRPMD/spin mapping/hopping, classical IESH/multistate MASH, SA-CASSCF data, spin–boson HEOM, local PCET and effective electrode models | [Nonadiabatic methods](nonadiabatic.md), [IESH/MASH](../nonadiabatic-iesh-multistate.md), [electronic data/HEOM](../tier2-nonadiabatic.md), [PCET](pcet.md), [electrode models](electrochemistry.md) |
| Model construction | PySCF reference data, PES fitting/export, committee uncertainty, delta corrections and restartable sample–label–train loops | [PES generation](pes-generation.md), [ML lifecycle](../tier2-ml-lifecycle.md) |

Ordinary input workflows use `validate`, `run`, `analyze` and `resume`.
`precision`, `instanton`, `nonadiabatic-rate` and the Python-only methods have
their own contracts. A Python API's presence does not imply a TOML task, an MPI
implementation or support for every thermostat and bias combination.

## Engines, parallelism and restart

Existing inputs keep their original engine and scientific identity. The default
materials workflow, the explicitly selected native PIMD engine and the Torch
workflow still have distinct supported combinations and checkpoint readers.
Do not reinterpret a checkpoint as another engine's state.

The [native ensemble API](../native-ensembles.md) owns persistent C++ physical
state, thermostat/barostat auxiliaries, force caches and RNG state. Native
bindings for [built-in analytic potentials](../builtin-native-potentials.md)
let those models use fused C++ loops. A Python PES bridge remains a Python force
evaluation even when the integration operators are native. The
[common propagation stack](../propagation-stack.md) documents block-atomic
versus accepted-prefix updates and the compatible backend changes.

The [Torch workflow](../torch-materials-workflows.md) keeps a float64
`[replicas, beads, atoms, 3]` state on the chosen CPU/CUDA device and commits the
whole batch atomically. It supports all-mobile, fixed-cell PIMD and diffusion.
It rejects NPT, fixed atoms, constraints, exchange/high-order actions, friction,
biases and MPI/worker-pool combinations. A requested unavailable device or
incompatible model fails explicitly; it is not a request to change precision
or select a host fallback. Exact continuation is a same-environment contract.

[Execution](execution.md) describes process pools, bead workers, static/dynamic
MPI groups and OpenMP controls. [Batched ML potentials](../batched-ml-potentials.md)
and [force-engine adapters](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/potential-adapters.md)
state their model, cell, thread and device requirements separately. OpenMM's
copy-bank route evaluates copies; it does not replace PyRPMD's dynamics with an
OpenMM trajectory. LAMMPS persistent contexts do not guarantee a speedup.

Checkpoints contain complete workflow state and RNG data, with atomic publication
and integrity checks. The scientific fingerprint rejects changed model/source
files and changed physics; execution tuning is permitted only where the specific
reader supports it. See [execution and restart](execution.md),
[portable publication](../portable-persistence.md) and the
[Windows identity policy](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/development.md#windows-runs-with-missing-historical-source-hashes).
Historical Windows runs with missing drive-path source hashes are refused
before migration and must be rerun in a new directory.

For PLUMED, an MD snapshot alone does not restore opaque action history.
Automatic continuation requires the verified `[plumed] restart="files"` profile
selected before the first run; see [PLUMED restart](../plumed.md).

## Scientific limits to check

**Rates and precision.** Equivalent channels, source-basin normalization,
recrossing plateaus, energy conservation and uncertainty must all match the
physical question. A connected projected FES is not a rate or a branching ratio.
The precision executor currently requires fixed local umbrella designs, explicit
Python propagation and serial execution; unsupported sampling/parallel choices
are rejected. A statistical target alone is not acceptance, and unrequested
numerical comparisons are not inferred to have passed.

**Quantum sampling.** Suzuki–Chin, exchange and PIGS sample equilibrium or
projected distributions, not physical real-time trajectories. Fermionic results
use signed reweighting of positive bosonic sampling and require adequate average
sign. Periodic exchange chooses converged Gaussian image sums or an explicit
minimum-image approximation. Its isotropic NPT path uses an all-bead BZP
propagator with physical PES virials and cell updates; it is incompatible with
the centroid BZP exchange combination. There is no superfluid estimator. See
[path-integral actions](../tier2-path-integral.md).

**GLE/PIGLET.** [Parameter import and generation](../condensed-phase-nqe.md#generating-and-importing-glepiglet-parameters)
record frequency ranges, targets, residuals and provenance. A successful harmonic
fit does not certify anharmonic observables. The archived P=4 PIGLET calibration
fails the anharmonic-pressure check; that negative result remains in the
[condensed-phase validation](../condensed-phase-nqe.md).

**Spectra.** Experimental f-QCMD uses fixed-topology bent triatomics, separable
IBI corrections and P=1 production. Its fixed-quasi-centroid PIMD sampler is an
equilibrium tool. Real-time adiabatic QCMD and h-CMD remain unimplemented, and
the requested converged f-QCMD spectral evidence is still outstanding. Supplied
dipole/polarizability models define relative intensities; no absolute optical
intensity or automatic Fresnel correction is implied.

**Instantons.** An optimized action is not by itself a multidimensional rate.
Reactant partition functions and symmetry/rigid-mode conventions must be supplied.
Dense prefactors are limited to 4096 active bead coordinates. Golden-rule support
uses constant weak coupling, two bound/constrained diabatic surfaces and the
normal regime. Numerical stationarity and finite prefactors retain
`physical_converged=false`; test bead count, imaginary-time duration, PES and
semiclassical accuracy separately.

**Electronic methods and model validity.** IESH and multistate MASH currently
require P=1, real Hamiltonians and nondegenerate adiabatic levels. The PySCF
bridge implements SA-CASSCF; the HEOM reference implements its documented
spin–boson bath and hierarchy. Neither implies general electronic-structure
or open-system coverage. PCET assumes a local single-proton complex, a wide-band
metal and a classical Gaussian environment. Committee disagreement is not a
calibrated model error, and a successful training loop does not validate a PES.

## Archived evidence and performance

These records preserve the evidence previously linked from the README. They are
dated model-specific studies, not new measurements performed by the documentation
update. The [literature inventory](../literature-validation.md) and
[reproduction checklist](reproduction.md) identify remaining requirements.

| Record | Evidence and practical limit |
| --- | --- |
| [H + H₂/BKMP2 PMF refinement](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/h3-pmf-refinement-2026-10-29/README.md) | Four independent seeded repeats at 1500 K and P=16 used 848,000 samples and 1,024 new recrossing trajectories. Paired UI/MBAR barrier intervals contained zero; joint relative rate standard error was 20–23%. This is finite-sampling evidence for that setup. |
| [State-selected BKMP2 study](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/h3/STATE_PREPARATION_STUDY.md) | Separates centroid conditioning, full-PES action QCT and experimental smooth preparation; retains unresolved trajectories and a low-energy discrepancy. |
| [PCET harmonic comparison](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/pcet-model-2026-10-30/README.md) | Twelve forward/reverse rates agree within 0.062% with a pinned pyPCET implementation for the specified harmonic model. |
| [PMF numerical checks](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/pmf-accuracy-2026-10-30/README.md) | Independent analytic integrals and PyMBAR test bead counts, integrators, periodic CVs and correlated errors. |
| [H + BrCl model](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/brhcl/README.md) and [adaptive-umbrella audit](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/umbrella-reliability-2026-10-31/README.md) | Record hidden-coordinate mixing, overlap/uncertainty repairs and the distinction between pilot thresholds and precision. |
| [Ni–H EAM pilot](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/materials/README.md) and [water pilot](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/water/README.md) | Source-pinned material models and short model checks; not complete converged reproductions of all observables. |
| [Condensed-phase NQE](../condensed-phase-nqe.md), [instantons](../instanton.md) and [nonadiabatic comparisons](../nonadiabatic-iesh-multistate.md) | Independent references, finite-bead/step limits and reported uncertainties remain attached to their respective models. |

CPU timing evidence, model sizes, hardware and thread counts are recorded with
the [native-potential benchmarks](../builtin-native-potentials.md),
[batched-ML measurements](../batched-ml-potentials.md) and
[large-system force-engine report](../t19-performance.md). CUDA measurement
drivers are available, but **no GPU timings or GPU speedups have been established
by those reports**. Do not transfer a cheap analytic-kernel speedup to an expensive
force model or turn CPU throughput into a CUDA claim.

The [periodic-materials audit](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/periodic-materials-audit.md),
[implementation review](../implementation-review-2026-11-03.md) and
[development roadmap](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/roadmap.md)
retain longer engineering assessments and unfinished work.

## Inputs, extensions and attribution

Use [Input and CLI](input.md) for minimal rate/scattering inputs and the
[examples](examples.md) for complete runnable jobs. Python inputs are trusted
executable code; place reusable factories in importable modules. A PES accepts
atomic-unit `float64[B,N,3]` coordinates and returns `[B]` energies and
`[B,N,3]` forces, with `forces = -dV/dq` and no bead scaling. A custom CV supplies
its full ring Jacobian. See [potential interfaces](potentials.md) and the
[Python API](api.md).

[Output and analysis](results.md) covers manifests, samples, diagnostics,
trajectory formats and reanalysis. [Development](development.md) covers tests,
builds and contribution rules. Source revisions, upstream attribution and
third-party notices remain in
[THIRD_PARTY_NOTICES.md](https://github.com/chen-yu-hao/PyRPMD/blob/main/THIRD_PARTY_NOTICES.md).
Cite the actual method, model, parameters, upstream software and PyRPMD commit;
use the [method catalog](references.md) and [reproduction checklist](reproduction.md)
to identify the appropriate sources.
