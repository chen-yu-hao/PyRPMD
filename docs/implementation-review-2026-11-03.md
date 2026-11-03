# Implementation review and follow-up boundaries

**Historical snapshot.** The review below records the revisions available before
the integration rounds. Its statements about unmerged PRs and missing features
are not the current support matrix. For the integrated state, use the
[capability guide](manual/capabilities.md), [next-release notes](releases/unreleased.md)
and [roadmap](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/roadmap.md).

This review distinguishes current `main@8cc402f` from the separate integration
branch in [PR #81](https://github.com/chen-yu-hao/PyRPMD/pull/81), pinned at
`8fd652a` for the full integration regression. The integration branch's roadmap uses "merged" to mean merged
into that branch; it does not mean that those changes are already on `main`.
The review covered public input and execution routes, numerical and potential
backends, checkpoint and artifact ownership, analysis, tests and CI, and the
method-specific additions in that integration branch.

The most useful next work connects implemented numerical components to
reproducible workflows and supplies missing physical or hardware evidence.
Adding another method name is not sufficient to establish either property.

## Three independent changes delivered in this review

| Pull request | Problem addressed | Deliberate boundary |
| --- | --- | --- |
| [#82: phase projection](https://github.com/chen-yu-hao/PyRPMD/pull/82) | A complete QR complement and two dense products precede the splitting eigensolve. A Householder reflection and symmetric rank-two update remove that cubic projection work. | The final eigensolve remains dense, and the 4096-coordinate guard remains. Local timing is not a total molecular splitting speedup. |
| [#83: portable persistence](https://github.com/chen-yu-hao/PyRPMD/pull/83) | Writable-descriptor fsync, memory-map/hard-link publication ordering, portable authenticated archive references and bounded Windows JSON-replace retries. | Three T18 defect categories are addressed. Drive-letter source hashing is deferred because changing the existing hash set changes old scientific identities. Actual Windows CI remains necessary. |
| [#84: tensor materials workflows](https://github.com/chen-yu-hao/PyRPMD/pull/84) | Public fixed-cell PIMD/diffusion execution, all-replica checkpoint commits, owned generators, a once-only Maxwell launch and verified stream replay. | float64, all-mobile Cartesian atoms and a single device batch. No NPT, constraint, rate, exchange or automatic precision/device fallback. T04 provider factories still require #81. |

Each branch starts from `main`; none merges another open PR. The three source
changes were also assembled together on `main` and on that pinned #81
revision for integration testing. The #81 splitting application needs one
import reconciliation: retain its `_CountingPotential` import while removing
the obsolete `_complement` import. Exact validation commands, source revisions
and raw pytest summary lines are recorded in
`validation/implementation-review-2026-11-03.json`.

While validation was running, #81 advanced to `15aafbb`, fixing cancellation
in the native double-well confinement energy on FMA-enabled compilers. That
upstream change was added to the integration verification tree, its native
extension was rebuilt, and the relevant built-in/native suites passed again
(455 passed, one optional-PES skip). The 11100-pass full-suite result remains
attributed to the preceding `8fd652a` base; it is not relabeled as a new full
run on `15aafbb`. This upstream fix is not part of any of the three PRs above.

See the [tensor workflow guide](torch-materials-workflows.md) for its complete
contract and reproducible measurements. The other two PRs contain their own
method and persistence guides. Existing scientific inputs and legacy/native
checkpoint readers retain their identities and formats.

## Status of the implementation areas

The following table describes incremental features, rather than claiming that
an entire method family is absent from `main`. The original task labels come
from [the integration roadmap](https://github.com/chen-yu-hao/PyRPMD/blob/8fd652a18d4f9990e2079c550d1e27a8b8cb77da/docs/roadmap.md).

| Area | Observed implementation boundary | Useful continuation |
| --- | --- | --- |
| T01 engineering and CI | Cross-platform library jobs, optional-dependency guards and the campaign marker are in #81. | Remove only Windows skips whose specific defect has passed actual Windows validation. Preserve the source-hash skip until a migration design exists. |
| T02 constraint metric | Analytic Hessian-vector products replace dense CV Hessians in #81. | Unify the duplicate native CV/bias implementations under that contract. |
| T03 analytic native PES | Scalar built-in C ABI bindings are in #80 and #81. | Design a separate matrix/diabatic ABI; scalar energy/force bindings do not provide it. |
| T04 batched ML PES | Named batched providers and their tensor interface are in #81. | Measure actual CUDA throughput, graph construction and scalar synchronization before optimizing. |
| T05 force engines | Persistent OpenMM contexts and LAMMPS instances/MPI subcommunicators are in #81. | Establish GPU residency and diagnose the lack of a consistent large-system LAMMPS benefit. |
| T06 tensor workflows | The bottom-level Torch integrator is on `main`; #84 adds the public materials route. | Expand supported physics only with separate state, identity and estimator contracts. |
| T07 f-QCMD | #81 contains experimental conditional sampling, IBI and one-bead production for fixed-topology bent triatomic molecules. | Transfer the available alternative-branch water validation to the selected implementation before claiming spectral improvement. |
| T08 exchange | Fermionic sign reweighting and periodic minimum-image support are in #81. | Winding sums and exchange with NPT remain separate scientific work. |
| T09 instanton | `main` has the foundation; production solving, Hessian providers and the CLI are in #81. | Scalable prefactors beyond the dense limit need independent determinant and zero-mode validation. |
| T10 path sampling | Minus-ensemble RETIS, REPPTIS and direct FFS additions are in #81. | Add a public TOML/CLI workflow and durable sampler state before infinity-RETIS scheduling. |
| T11 enhanced sampling | ABF/eABF, steered work, string methods and PLUMED file restart additions are in #81. | VES remains unimplemented; steered free-energy differences are not a general equilibrium FES. |
| T12 nonadiabatic methods | IESH and multistate MASH additions in #81 are single-bead methods. | Ring-polymer variants require a new scientific derivation and validation scope. |
| T13 electronic structure/HEOM | SA-CASSCF multistate bridging and HEOM are in #81. | TDDFT NAC and file-backed full-hierarchy HEOM recovery remain useful work. |
| T14 active learning | #81 contains a resumable sampling/labeling/training loop. | Replace repeated accumulated-dataset snapshots with immutable blocks and a commit manifest, preserving old readers and provenance. |
| T15 GLE/PIGLET | Parameter import/generation tools are in #81. | Validate nonharmonic systems and broader frequency ranges independently of harmonic parameter checks. |
| T16 shared implementation/state | Native duplication and multiple engine persistence contracts remain. | First unify mathematical kernels, then specify explicit cross-engine migration semantics. |
| T17 documentation | Integration and public workflow changes have not all reached `main`. | Consolidate README/references/releases after the interfaces and integration order are settled. |
| T18 Windows persistence | #83 addresses three defect categories with platform-contract tests. | Confirm them on Windows and design a versioned source-hash migration. |
| T19 GPU/large systems | CPU evidence does not establish GPU correctness or throughput. | Run model, propagation and checkpoint measurements on identified hardware. |
| T20 f-QCMD/h-CMD evidence | Paired spectra and h-CMD remain incomplete. | Establish comparable spectra and uncertainty first; separate h-CMD partition/coupling work. |
| T21 campaign scheduling | Two recorded tests depend on timing-sensitive process behavior. | Repair readiness/exit synchronization and child-process ownership; do not merely increase timeouts. |
| T22 precision controller | Planner, result import and ledger are in #79/#81; public execution integration is missing. | Bind persistent execution IDs to real jobs and test interruption, replay and budget accounting end to end. |

## Recommended next implementation sequence

1. **T22: one real precision-controlled local workflow.** Connect
   `precision_jobs.py`, `precision_ledger.py` and `precision_artifacts.py` to
   `cli.py`/`runner.py`. Limit the first delivery to local serial execution.
   Verify plan, launch, interruption, recovery, result import and replanning
   without duplicate launch, duplicate budget charges or pilot/production
   data reuse masquerading as independent evidence.

2. **T16: shared numerical contracts before checkpoint migration.** Start with
   `native/fused.hpp`, `native/propagation.h` and `cv.py`. Then align phase
   clocks, RNG ownership and committed trajectory prefixes across
   `workflows/native_pimd.py` and `workflows/torch_materials.py`. Starting a new
   engine from the same phase-space point does not guarantee bitwise
   continuation; migrations must record their semantics and provenance.

3. **T19: real CUDA and force-engine measurements.** Inspect
   `_batched_ml_base.py`, the per-provider graph builders and `torch_backend.py`.
   T04 finite-value scalar reads can synchronize the device. Report model
   calls, complete propagation, checkpoint boundaries, peak memory and
   correctness together. Evaluate OpenMM residency and LAMMPS large-system
   behavior as separate workloads.

4. **T21: campaign process synchronization.** Repair the recorded failure
   admission/drain and isolated CPU-limit tests through explicit process
   readiness and ownership in `benchmarks/h3/p256_parent_campaign.py` and
   `quantum_isolated_continuation.py`. Repetition is an acceptance check, not
   a substitute for identifying the synchronization defect.

5. **T20: paired f-QCMD spectroscopy.** Use the selected `qcmd.py`,
   `_qcmd_geometry.py`, `_qcmd_sampling.py` and water example. Hold the PES,
   temperature, training data, timestep and spectral processing fixed when
   comparing CMD and f-QCMD. Assess peak positions, widths, uncertainty and
   training/sampling-length dependence. Keep h-CMD as a distinct method task.

6. **T10: a durable public path-sampling workflow.** Choose one of `retis.py`,
   `pptis.py` or `ffs.py` first. Add strict input validation, interface-chain
   state and RNG checkpointing, run/analyze commands and a short interrupted
   analytic-double-well comparison before adding parallel scheduling.

T14 dataset blocks and T13 full-ADO checkpoints are smaller independent
follow-ups with clear persistence benefits. Neither should weaken provenance
or drop hidden state merely to make recovery faster.

## Evidence and limits

Force finite differences, analytic free transport, harmonic statistics and
exact stochastic continuation test distinct implementation properties. They
do not certify a molecular PES, a long-time diffusion regime, bead/timestep
convergence or a physically reliable spectrum. A minimum-image exchange path
does not implement winding sums, and a small average-sign example does not
resolve the fermion sign problem. Single-bead nonadiabatic examples cannot
establish ring-polymer variants; HEOM also needs hierarchy and bath-expansion
convergence.

The two performance experiments measure different scopes. The splitting
benchmark includes phase projection and the same final dense eigensolve. The
Torch benchmark includes the PIMD provider, propagation, immutable
trajectory/checkpoint I/O and analysis. At one CPU thread per pool, the latter
is slower for one replica and faster for eight or 32 replicas in the recorded
small harmonic problem. Neither is a general GPU or molecular speedup claim.

README files, release notes, reference catalogs and package-root exports were
reserved and left untouched. A later integration update should document the
explicit engine choice, strict restart boundary, measured batch-size tradeoff
and unsupported combinations in both languages. Reuse the existing BCOCB
reference, add official PyTorch reproducibility/API links if needed, and do
not introduce a new scientific-method citation for a standard Householder
linear-algebra refactor.
