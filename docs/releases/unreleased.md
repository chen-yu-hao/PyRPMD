# Next release (unreleased)

These notes collect changes integrated after the original
[0.2.0 release notes](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/releases/0.2.0.md),
through `main@ceee59c` on 2026-11-03, together with the documentation changes
described below. The installed package version remains 0.2.0. A release number,
tag and publication date have not been assigned by these notes.

## Finding the right entry point

The English and Chinese READMEs now focus on installation, a runnable first
calculation, a capability table and documentation links. Detailed restrictions
and evidence are collected in the matching
[capabilities chapter](../manual/capabilities.md). The
[manual index](../index.md) links the recently integrated workflows, including
precision control, tensor PIMD, experimental f-QCMD and batched ML potentials.
The [reference catalog](../manual/references.md) distinguishes implemented
methods from related research and records the original papers.

The precision guide remains part of Sphinx and is now reachable from its
navigation. Documentation CI checks the README size and section correspondence,
the English/Chinese chapter sets, local destinations and fragments, and external
links. Existing guide filenames remain valid; internal planning labels are
removed from the public entry points.

## Dynamics, potentials and quantum statistics

- **Native propagation:** free, harmonic, Morse-pair, double-well and compatible
  fixed-cell periodic-cosine potentials have native bindings. Shared native
  CV/bias/metric kernels and analytic Hessian–vector products remove duplicate
  implementations and dense metric work. See
  [built-in potentials](../builtin-native-potentials.md) and the
  [propagation stack](../propagation-stack.md).
- **Tensor workflows:** `sampling.engine="torch"` runs fixed-cell `pimd` and
  `diffusion` in one float64 device batch with atomic state/RNG checkpoints.
  Unsupported NPT, constraint, exchange and parallel-worker combinations fail
  explicitly; see [the workflow contract](../torch-materials-workflows.md).
- **Potential providers:** batched MACE, NequIP, fairchem, DeePMD and metatomic
  providers expose batch and tensor routes. OpenMM has persistent Contexts and
  an RPMD copy-bank evaluation route; LAMMPS has persistent bead-affine instances
  and optional private MPI communicators. Provider constraints, framework
  versions and measurements are in [batched ML](../batched-ml-potentials.md) and
  [force-engine performance](../t19-performance.md).
- **Exchange:** fermionic observables use sign reweighting of bosonic sampling
  with average-sign diagnostics. Periodic exchange supports Gaussian image sums
  (`full_winding`) or the explicitly approximate `minimum_image` mode; isotropic
  NPT uses the separate all-bead BZP route. This does not supply a superfluid
  estimator or solve the sign problem. See
  [path-integral actions](../tier2-path-integral.md).
- **GLE/PIGLET:** import gle4md-style parameters with units/provenance or generate
  the supported harmonic-target parameter families; fit targets and residuals
  enter identity. [Condensed-phase quantum statistics](../condensed-phase-nqe.md)
  documents the estimators, pressure routes and independent checks.

## Rates, sampling and spectroscopy

- **Precision-controlled rates:** `pyrpmd precision plan`, `run` and `status`
  execute frozen thermal rate designs with PES-frame budgets, authenticated
  receipts and interruption/replay handling. Statistical precision, numerical
  equivalence and physical validity are separate conclusions. The first executor
  supports local serial fixed-umbrella rates; see
  [precision control](../rate_precision.md).
- **Instantons:** Hessian providers, Bofill optimization and `pyrpmd instanton`
  support rate and splitting convergence studies. Normal-regime golden-rule
  rates use two bound/constrained diabatic surfaces and constant weak coupling.
  The prefactor remains dense with a 4096-active-coordinate guard. See
  [instantons](../instanton.md), [normalization](../low-temperature.md) and
  [phase projection](../tunnelling-phase-projection.md).
- **Path sampling:** RETIS adds a minus ensemble and internally estimated flux;
  REPPTIS and direct FFS are available as Python APIs. A general TOML workflow
  and infinity-RETIS are outside this delivery. See
  [transition-interface sampling](../transition-interface-sampling.md).
- **Enhanced sampling:** ABF/eABF, steered work/free-energy differences,
  finite-temperature strings, learned CV export and declared PLUMED file restart
  are documented in [enhanced sampling](../enhanced-sampling.md),
  [strings](../finite-temperature-string.md), [mlcolvar](../mlcolvar.md) and
  [PLUMED](../plumed.md).
- **Experimental f-QCMD:** bent triatomic molecular quasi-centroids, conditional
  sampling, IBI corrections and one-bead production extend the existing
  spectroscopy workflow. The selected implementation and its model limits are
  in [f-QCMD](../qcmd.md). Paired physical spectral validation and h-CMD are still
  pending; these notes do not claim that the curvature problem is resolved.
- **Selected-state QCT:** the explicit `j0_energy_policy="preserve_actions"`
  policy preserves quantized actions while removing angular momentum; the
  historical energy policy remains available. See
  [state-selected preparation](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/state-selected.md).

## Nonadiabatic dynamics and model development

Classical **IESH** and the implemented **multistate MASH** formulation support
single-bead workflows, checkpoints and independent small-system comparisons;
ring-polymer variants are not implied. See
[IESH and multistate MASH](../nonadiabatic-iesh-multistate.md).

The **PySCF SA-CASSCF** bridge supplies energies, gradients, NACs and overlaps;
**spin–boson HEOM** supplies a reference method with independent bath, hierarchy
and time-step convergence checks. The
[electronic-structure guide](../tier2-nonadiabatic.md) specifies their scope and
optional dependencies. A resumable **active-learning loop** connects sampling,
reference labelling and training, with an optional CPU MACE reference trainer;
see [the model lifecycle](../tier2-ml-lifecycle.md).

## Compatibility and verification

Existing default engine choices and scientific identities are retained. Explicit
engine selection has its own state and restart contract; legacy, native and Torch
checkpoints are not interchangeable. The legacy/native engine convergence task
must merge before this final documentation consolidation can be considered closed.

Portable native publication, authenticated archive paths and bounded Windows
JSON replacement retries have been integrated. Old Windows runs whose
drive-letter plugin/model sources were never hashed are rejected before resume;
the missing historical source identity cannot be reconstructed safely from
today's files. See [portable persistence](../portable-persistence.md) and the
[Windows compatibility policy](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/development.md).
Some Windows skip registrations still await their separate cleanup.

The integration also fixes variance-cancellation acceptance, native double-well
FMA cancellation, PLUMED HILLS formatting and timing assumptions in process tests.
The [integration roadmap](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/roadmap.md)
records source revisions and exact test results. The documentation pass changes
no numerical kernels or checkpoint formats.

## Evidence still required

Published CPU timings apply only to their recorded hardware, models and shapes.
There are **no recorded CUDA measurements** for the integrated performance
campaign; [its report](../t19-performance.md) supplies runnable drivers and states
that limitation. LAMMPS persistent instances are not uniformly faster.

Real-time QCMD, h-CMD, infinity-RETIS, VES, ring-polymer IESH/multistate MASH and
TDDFT NACs remain separate work. Passing unit tests or a small analytic example
does not establish a converged molecular rate, spectrum or material property.
