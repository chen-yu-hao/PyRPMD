# Tier 2: literature comparisons and extended validation

Date: 2026-11-01. This study follows PR #63 at original head
`4da679023408bccfcd3173bfc81f961c48d1d0ff`. It adds reproducible published-model
comparisons, actual trajectories, independent reference solvers, timestep and
bead convergence, seed uncertainty, real optional runtimes, and explicit
negative results. It also found and corrected an OPES initialization defect.

**The evidence supports specific numerical contracts and model benchmarks;
it does not establish complete i-PI/DL_POLY Quantum parity or converged
molecular predictions.** A passed test, a consistent confidence interval, and
a reproduced publication are different levels of evidence.

## What was compared

| Area | Published source / independent reference | Result and limit |
| --- | --- | --- |
| PA-CMD | Cao–Voth quartic oscillator, JCP 99, 10070 (1993), Eq. 18; independent basis/real-space quantum calculation | Ground-energy increase 11.8293% agrees with the paper's rounded 12%. Baseline identically windowed IR peak differs by 0.171%; the ensemble Kubo correlation still has 5.93% normalized RMSE over t=0–15. |
| Quadratic f-CMD | Held-out centroid forces for that same model | Fitted frequency 1.211071 ± 0.000276 SEM versus exact fundamental 1.210356. A quadratic fit does not represent a general anharmonic centroid PMF. |
| Suzuki–Chin | Kapil et al., JCP 145, 234103 (2016); independent harmonic Gaussian action | Fitted convergence orders 3.94–4.00, versus 1.97–2.00 for the primitive action. Nonlinear force, NVE drift and reversibility checked. No water/ice reproduction. |
| Bosonic exchange | Feldman–Hirshberg, JCP 159, 154107 (2023), Appendix B2/Fig. 6; authors' pinned 2D trap inputs | Exact P=36 discretization bias at beta=6 is about -0.345%. Actual three-seed MD is statistically consistent with finite-P references, but its uncertainty is wider than the paper's 0.5% envelope. |
| PIGS | Sarsa et al., JCP 113, 1366 (2000); exact projected oscillator and open-chain integral | Separately resolves finite projection and O(tau²) discretization. Harmonic stationary integrator covariance agrees with the finite action below 8e-14 relative error. No interacting-helium reproduction. |
| OPES reference | Invernizzi–Parrinello, JCTC 18, 3988 (2022), SI Sec. S3 | Independent quadrature gives basin differences 5.7125917412 and 5.6905167844 kBT; the paper prints 5.712(6) and 5.690(5). Parentheses are preserved literally without assuming a statistical meaning. |
| OPES implementation | Real PLUMED 2.9.4, identical deposited centers and probe points | After initialization correction: fixed OPES energy/force differences at most 7.11e-15/9.24e-14; explore 2.75e-14/9.95e-14. This comparison disables compression and matches literal kernel widths. |
| OPES finite trajectories | Published Müller model, 32 main histories plus four sensitivity histories | 4.56 million MD steps in total. Undefined basin estimates, missing crossings and large seed/width sensitivity prevent a free-energy convergence claim. |
| Direct-data FSSH and endpoint MASH | Tully I, JCP 93, 1061 (1990), Eq. 21; independent stationary quantum scattering | 4,608 actual trajectories, three momenta, four main seed blocks and dt=8/4/2. All main outgoing-channel references lie in pointwise 95% intervals, whose finite width precludes claiming exact dynamics. One finite-time censored trajectory is retained. |

The detailed derivations, parameter differences, sources, complete results and
figures are in [spectroscopy](spectroscopy/README.md),
[path integrals](path_integral/README.md),
[nonadiabatic dynamics](nonadiabatic/README.md), and
[enhanced sampling / ML](sampling/README.md).

## Defects and corrections exposed by testing

### OPES empty-history bias affects the first statistical weight

With no kernels, the OPES equation gives
`V=prefactor*log(epsilon)=-barrier`, with zero force. The previous implementation
returned zero. Although an initial constant has no force, it changes the first
METAD weight from `exp(-beta*barrier)` to one, while later weights use the
equation's convention. Thus it cannot be discarded independently of later
updates.

An 80-deposition, 21-probe-per-deposition comparison against a source-pinned
PLUMED kernel exposed maximum pre-fix differences of 11.2253 in bias and
54.7848 in force for the selected reduced-unit history. The corrected
implementation agrees at floating-point precision. This is a **direct bias
and force comparison**, without the spline acceleration used in long sampling
experiments. It does not establish equivalence of the different adaptive
bandwidth rules or compressed kernels.

New histories use schema 2 and persist `initial_bias=-barrier`. Recognized
schema-1 histories retain their original convention and weights when restored;
analysis records their version and a legacy warning. Unknown versions and
inconsistent schema-2 initialization are rejected. Pre-fix results are kept
separate from corrected `opes_v2` results.

A partially completed job whose saved histories are all schema 1 also starts
its missing replicas with schema 1, preserving whole-job continuation. Missing
replicas in a job with mixed saved schemas are rejected before constructing
providers; completed existing histories retain their own declared versions.

### MPI CI launcher compatibility

The previous PR head's Python 3.11 CI run passed **4,618 tests** before its
first `mpiexec` command failed during process allocation. Its ORTE-era
`OMPI_MCA_rmaps_base_oversubscribe` setting did not enable oversubscription in
the newer PRRTE launcher. The workflow now uses explicit
`mpiexec --oversubscribe`, documented by
[Open MPI](https://docs.open-mpi.org/en/v5.0.7/man-openmpi/man1/mpirun.1.html).
The eight real local MPI gates exercise ordered/dynamic task scheduling,
FES/workflow restart, adaptive sampling, OPES, materials, and surface workflows.
Local runtime versions and logs are retained in `software/`.

### Reproducibility and two-state evaluation cost

PI cache preflight now checks parameters, seeds and source provenance before
using either a completed result or partial checkpoint, and fails before
rewriting a manifest. The exact original driver used for the committed PI
trajectories is archived and its identity is preserved; adding guards does
not relabel those trajectories as newly computed.

The two-state overlap logarithm and Hermitian propagator use algebraically
exact SU(2)/Pauli expressions, verified against general SciPy matrix functions.
This makes thousands of actual public-engine trajectories practical without
introducing a physical approximation. Larger electronic spaces retain the
general routines.

## Interpretation of the numerical comparisons

### Spectral processing must be matched

The quartic model's unbroadened fundamental is **1.2103563168**. With the
finite record, Hann taper and IR processing used in this experiment, the
exact-reference peak is **1.2347835036**. The PA-CMD result is
**1.2368960859 ± 0.0021099688 SEM**, from eight independent seed groups.
Comparing a windowed peak directly to an unbroadened transition would mix
record-length effects with dynamics error. Both quantities are retained.

The paper's Fig. 1 reports the real quantum correlation; this study compares
Kubo-transformed correlations calculated for the same published Hamiltonian.
It is a published-model validation, not digitization/reproduction of that
particular figure. Remaining correlation error is not hidden behind the
close peak position. Raman/SFG tensor conventions and actual TorchScript
property evaluation have software tests; no liquid/interfacial Raman or SFG
spectrum was reproduced here.

### Discretization and statistical agreement are separate

For 16 ideal 2D bosons at beta=6, the continuum total energy is 16.00500687,
the exact P=36 action gives 15.94975672, and the short main-engine trajectories
give **15.88987612 ± 0.49239761** (three-seed Student-t 95% half-width).
The confidence interval contains both references, but it is much too wide
to certify the original paper's 0.5% maximum discrepancy. No seeds were
discarded and no budget was extended until a result passed.

For PIGS at alpha=0.25 and projection time 8, the finite-projection continuum
variance is 0.5002013181, the P=33 action gives 0.4963437199, and the true
ground-state value is 0.5. At P=129 the value 0.4999575725 benefits from partial
error cancellation. A small error relative to 0.5 alone would not prove both
projection and imaginary-time convergence.

### Finite scattering ensembles and boundary censoring remain visible

The main calculations use 512 trajectories per method/momentum at dt=4.
The table reports the most diagnostic outgoing channel, with pointwise 95%
Wilson intervals:

| Incident momentum / channel | Independent monochromatic quantum | FSSH | Endpoint MASH |
| --- | ---: | --- | --- |
| 8.8 / lower reflection | 0.067252 | 0.080078 [0.05955, 0.10685] | 0.050781 [0.03489, 0.07337] |
| 15 / upper transmission | 0.322988 | 0.359375 [0.31900, 0.40184] | 0.355469 [0.31523, 0.39787] |
| 25 / upper transmission | 0.623678 | 0.642578 [0.60015, 0.68289] | 0.644531 [0.60213, 0.68477] |

Near threshold, averaging the quantum reference over the paper-inspired
5% Gaussian momentum spread changes the p=8.8 lower reflection to about
0.035284 and opens a nonzero averaged upper channel. That different initial
distribution is **not** substituted for the monochromatic trajectory target.
One MASH trajectory is unresolved at the predeclared main time cutoff; its
separate longer follow-up is preserved without rewriting the original count.
The 128-trajectory timestep subsets cannot resolve a few-percent population
convergence claim. This P=1 scattering experiment does not validate P>1
RP-MASH/RPSH or the separate thermal-rate workflow.

### Sampling diagnostics must include missing support

The corrected OPES campaign contains 32 main histories and four additional
timestep/bandwidth histories, totaling 4.56 million MD steps. These remain
far shorter than the authors' 200-million-step trajectories. In the poor-x-CV
METAD cases, zero interbasin crossings coexist with small globally weighted
divergences because the missed basin has very low equilibrium probability.
Required zero-population basin estimates are marked undefined. Half-timestep
and half-width comparisons expose sensitivity rather than establish
convergence. Neither a low scalar error nor a large weight-only ESS replaces
adequate phase-space coverage.

The auxiliary checks have separate, narrower conclusions:

- Temperature/Hamiltonian exchange uses four seeds and exact finite-bead
  harmonic references. Relative moment errors range from -2.08% to +2.70%;
  17 of 18 pointwise 95% intervals contain their references. The remaining
  +0.80% bead-variance discrepancy is retained, and simultaneous comparisons
  are not presented as independent validation proofs.
- Four-walker shared MetaD reproduces the frozen biased ensemble, with
  reweighted harmonic variance 1.0210 ± 0.0471 (95% interval), versus exact
  one. Simultaneous hill-height error is zero and batch permutation changes
  the bias by at most 4.44e-16. This stationary check begins after hills stop.
- At dt=0.0005, 128 unbiased first passages give mean time 86.655 ± 17.282
  and 128 statically biased paths give rescaled 86.186 ± 13.951, versus an
  independent exact MFPT of 80.7744. The bias is explicitly zero in the
  transition region. This tests the clock-analysis API on an overdamped
  model, not an adaptive infrequent-MetaD/OPES-flooding algorithm.
- Actual trained TorchScript committees leave monitored trajectories
  unchanged and collect 32 unlabeled candidates. Full delta-force agreement
  is 1.11e-16; contracted RPC forces agree with derivatives of their
  approximate Hamiltonian to 1.08e-10. Extrapolation force RMSE 0.40595
  substantially exceeds mean committee spread 0.049643, illustrating why
  disagreement is not itself a calibrated error bound.

## Software evidence and replay

The final full repository suite reports **4,712 passed, 97 skipped, 357
subtests passed, zero failures/errors**, in 302.23 seconds. Subtests and
focused regression runs are not added to the main passed-test count. The
928 warnings are upstream TorchScript and ASE/NumPy deprecations, retained
in the log rather than suppressed. All 345 production/test source files and
the numerical drivers remained unchanged during that run. The saved-data
plotting helper was extended separately; its old/new hashes are identified
in the source verification receipt.

Eight real MPI gates passed with Open MPI 4.1.6/mpi4py 4.1.2. The OPES gate
was repeated after the final collective legacy-state preflight change and
again passed serial/MPI equality and checkpoint/reanalysis checks. An
independent live-PLUMED/OPES review passed 74 tests with no skips. English
and Chinese HTML documentation both built with warnings treated as errors.
The final native wheel was installed offline and exercised outside the
checkout, including new/legacy OPES initialization, the two-state propagator,
all three native integrators and the CLI. See [software receipts](software/README.md)
for commands, package verification and skip categories.

`software/runtime.json` identifies the actual native backend and installed
versions, including PLUMED, CPU TorchScript, ASE, tblite, PySCF and MPI.
The final frozen-source full-suite receipt, live-runtime checks, MPI logs,
documentation and source-package verification are collected in that directory.
Optional source-pinned model and architecture-dependent skips are listed in
the raw pytest output; they are not counted as passes.

An earlier suite overlapped the OPES source correction. Its already-imported
parent analyzed new-schema results from newly spawned children and failed the
process/serial consistency test. The complete intermediate log is retained
as such; it is superseded by a fresh run after the sources were frozen.

Reproduction scripts live in `benchmarks/tier2_literature/`. Every detailed
report supplies its commands, seeds, settings and reference definitions. The
CSV/JSON files contain numerical evidence; figures are derived from that
evidence. Selected enhanced-sampling NPZ files retain unique raw stochastic
time series and ML held-out data needed for saved-data replay. Rebuildable
restart caches and redundant arrays are excluded. The sampling manifest
identifies the curated raw files explicitly. Use a fresh output directory
when changing code or settings.

These experiments do not supply the missing QCMD/h-CMD implementation,
periodic bosonic winding sectors, a RETIS minus ensemble, a validated external
excited-state quantum-chemistry bridge, or real-material ML training data.
Committee disagreement is not automatically a calibrated error bound. The
coverage guide continues to state these boundaries explicitly.
