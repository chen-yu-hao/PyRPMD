# P=256 parent availability and dynamic bead-convergence proposal

**Read-only assessment and fixed-input proposal; no new sampling or child
dynamics has run.** The four frozen P=256 equilibrium controls do not contain
a saved parent sequence. They cannot presently support the same child
experiment as the eight P=128 primary chains. A deterministic replay in new
directories can potentially recover their historical parents without changing
the sampler or adding independent chains. It must pass the complete byte-level
audit below before any parent becomes eligible for children.

The machine-readable [proposal](proposal.json) fixes the replay inputs,
maximum one worker, sequential seed order, a **7,200-second whole-campaign
wall ceiling**, CPU measurements and failure policy. It is for review and
does not launch or authorize the separate child experiment. The
[input audit](input-audit.json) binds every inspected frozen checkpoint,
source, settings file and reference. Its [read-only program](audit_inputs.py)
performed zero PES evaluations, Monte Carlo sweeps, dynamics steps or RNG draws.

## What the frozen P=256 controls actually contain

The registered [control protocol](../../surface-partition/controls/protocol.json)
uses seeds 825601, 825603, 825607 and 825609, 2,000 warmup sweeps and 20,000
production sweeps. It deliberately sets `snapshot_interval=0`. All four
checkpoints pass their integrity envelope, whole-file/report hash, complete
settings fingerprint and raw reference-domain identity checks.

| Saved item | Per-chain shape or value | What it supports |
| --- | --- | --- |
| `endpoint_q` | `(0, 256, 3, 3)` | No stored parent sequence |
| `endpoint_completed_sweeps` | `(0,)` | No historical snapshot indices |
| `radius`, `ratio` | `(9,)` each | Last centroid shape at every lambda |
| `eta_R`, `eta_r` | `(9, 256, 3)` each | Last full Jacobi ring deviations |
| `delta_energy` | `(20000, 9)` | Original BAR and energy-time diagnostics |
| `diagnostics` | `(2000, 9, 8)` | Scalar centroid, energy, minimum-distance and lineage diagnostics |
| RNG and counters | Full final PCG64 state; move, swap, traversal and lineage counters | Exact continuation/replay identity |
| Gradients or Maxwell momenta | Absent | Must be recomputed/drawn by a separately bound child experiment |

The final lambda=1 geometry can be reconstructed once per chain. The audit
found its centroid surface residual below `3.2e-15` bohr and its minimum bead
pair distance between 1.074 and 1.266 bohr. This confirms a usable final state,
not 200 historical parent configurations. The 20,000 scalar energy rows and
2,000 diagnostic rows do not determine the missing ring paths. Other lambda
states also cannot be counted as independent lambda=1 parents.

The existing equilibrium comparison is valid for its stated scope:
P=256 minus P=128 QTST has a mean log change of −0.0056454 and a 95% interval
of [−0.0284227, +0.0171319], within the registered ±log(1.05) margin.
The [completed assessment](../../surface-partition/controls/p256-case-assessment.json)
explicitly does **not** infer recrossing or full-rate bead convergence.

## Recover historical configurations without changing the Markov chain

The frozen `surface_partition.run` already has a snapshot observer. After
a completed production sweep, it constructs the lambda=1 Cartesian path
and stores it at the requested interval. This branch draws no random numbers,
evaluates no PES and does not modify the retained Jacobi arrays. The proposal
therefore uses exactly the original four seeds, run arguments, original
runtime, source hashes and BKMP2 library, in four **new empty directories**,
changing only `snapshot_interval` from 0 to 100.

This would save 200 configurations per chain at completed sweeps 2,100,
2,200, …, 22,000, for 800 configurations total. They remain observations from
the same four equilibrium chains. Neither deterministic replay nor the new
snapshots increases the number of independent chains.

The original completed checkpoint cannot simply be resumed with a different
snapshot interval or a longer production count: its complete settings
fingerprint correctly rejects either change. Nor may the original report,
NPZ or reference-domain file be replaced. A newly sampled continuation would
be a different prospective parent cohort; it is not part of this proposal.

### Required identity checks

For each replay, all 16 old payload arrays other than `fingerprint`,
`endpoint_q` and `endpoint_completed_sweeps` must have identical keys, dtypes,
shapes and C-order bytes. This includes the complete energy and diagnostic
series, final rings, RNG string, domain hash and all move/swap/lineage counters.
Floating-point tolerances do not substitute for this test. The audit records
each old array's dtype, shape and content digest for review.

Only the declared observation interval may differ in the settings dictionary.
The old and new fingerprints are checked against their respective complete
settings. `reference.json` and `reference-domain.json` must remain byte-identical;
their normalization must not silently change. The report's scientific analysis
must match the original. New timestamps, elapsed times, snapshot metadata and
the new checkpoint hash are explicitly separate provenance fields.

Every new parent must be finite, on the same active `r01=r12<=r02` centroid
surface, within the same bead-pair domain, and have the same zero-COM coordinate
convention. The new NPZ files are expected to have new whole-file hashes;
their old-array equality receipt binds them to the immutable original files.
All original inputs are rehashed before and after the operation.

One global manifest may mark the parents eligible only after **all four**
chains pass. A mismatch, timeout, incomplete chain or failed geometry test
remains a failed/incomplete replay. There is no replacement seed and no
selection of whichever chains happen to pass.

## Scheduling and what the historical timing measures

| Seed | Original timed sampling wall seconds |
| --- | ---: |
| 825601 | 550.238860 |
| 825603 | 540.474245 |
| 825607 | 340.400203 |
| 825609 | 305.065255 |
| Total | **1736.178563** |

These are measured `perf_counter` wall durations, approximately 28.94 worker
minutes in total. They are **not measured process CPU times**. The original
controller pinned each worker to one CPU, but reference/domain initialization
and final BAR analysis lie outside the producer's timed region. Shared-host
contention also varied substantially between chains. There is no honest way
to recover an exact whole-run CPU number from these reports.

The proposed replay runs one chain at a time on one recorded CPU, with
BLAS/OpenMP threads set to one, and records process user/system CPU via
`rusage` together with whole-call monotonic wall time. The 7,200-second limit
covers all four sequential calls, setup, writes, verification and analysis,
not 7,200 seconds per seed. A bounded supervisor must preserve any partial
checkpoint, stop admitting work and confirm that its worker has drained before
writing an incomplete receipt or releasing its exclusion lock.

Root must record the actual resource allocation after the existing extra-pair
pool exits normally. This proposal does not raise the MC worker limit or take
an implicitly free CPU. A separate wrapper, bounded synthetic identity and
failure tests, source freeze and review are required before execution.

## An independent P=256 child experiment

The current primary child driver hard-binds the P=128 primary parent protocol
and its complete settings. It should continue to reject these P=256 controls.
A separate reviewed P=256 parent binding is needed; setting `calibration=True`
or editing the active driver would not establish the required provenance.

The prospective experiment retains the same 300 K, hydrogen mass, BKMP2 bytes,
centroid surface, classical reference normalization and native Cayley-BCB
Hamiltonian propagation. It recomputes the analytic surface gradient and PES
forces from each reconstructed parent. Ordinary Maxwell momenta have variance
`m*P/beta`; plus/minus trajectories are antithetic pairs. Both timestep runs
use the same parent, channel and momentum, so the **within-P=256** timestep
comparison is legitimately paired.

The proposed initial costed design is four independent P=256 chains, 200 parents
per chain, four momentum pairs per parent and timesteps 0.1 and 0.05 fs to
200 fs. The 0.05 fs result remains primary. Each chain estimates

\[
\kappa_P(t)=
\frac{\sum (v/f_s)[H(s_+(t))-H(s_-(t))]}
     {\sum |v|/f_s}.
\]

The static parent distribution already contains the flux metric; dividing
both numerator and denominator by `f_s` retains the existing estimator.
Free COM momenta remain decoupled from the relative observable. Initial
static-domain restrictions do not become reflecting walls in child dynamics.
PES/integrator failures and sampled domain crossings must remain in the record.

The same-chain time-plateau and timestep intervals must lie inside ±0.01 in
κ. Full ring energy errors are checked at every physical step and scaled by
`beta/P`; the primary-step maximum must be at most 0.1. Minimum pair distances
are observed at 1 fs boundaries and must not be described as an every-step
minimum. The proposal makes no assumption that P=128 timestep behavior proves
P=256 timestep behavior.

## Four chains versus eight: compare the joint full-rate estimator

P=256 and P=128 parents have different dimensions, different equilibrium seeds
and no matched sampling history. There is **no between-bead pairing**. Matching
parent indices or reusing a seed label would not create one. The 800 P=256
configurations, their momenta, two signs and two timesteps are also not 800 or
more independent uncertainty units.

For each independent chain form the complete log-rate estimate first:

\[
L_{P,i}=\log k_{\rm classical}^{\rm TST}
+\log\frac{Z_{\rm TS,P}}{Z_{\rm TS,1}}\bigg|_i
-\log\frac{Z_{\rm R,P}}{Z_{\rm R,1}}
+\log\kappa_{P,i}.
\]

This retains the covariance of each chain's BAR log partition and log κ.
The finite-P reactant corrections are −7.282252233219134 for P=256 and
−7.27275868577072 for P=128, from the existing converged transfer calculation
bound by the control protocol. They must not be exchanged or replaced by the
continuum value. The common classical reference cancels in the log difference.

For the two independent groups use

\[
\Delta=\bar L_{256}-\bar L_{128},\qquad
\mathrm{SE}^2=s_{256}^2/4+s_{128}^2/8,
\]

with Welch degrees of freedom. Full-rate bead equivalence requires the entire
95% interval for Δ to lie within ±log(1.05), corresponding to the ratio band
[1/1.05, 1.05]. A small point difference, overlapping marginal intervals, or
failure to reject equality is insufficient. The full eight-chain P=128
four-pair result must be complete and hash-bound before this comparison is
launched; no partial extra-pair result was read to choose the proposal.

Report κ separately: four-chain P=256 intervals have three degrees of freedom,
and the eight-chain P=128 intervals have seven. Retain the existing 3% relative
95% halfwidth precision criterion and give an independent between-group κ
interval. A failed κ precision diagnostic cannot be hidden by apparent
agreement of the complete rate.

Using only the immutable **two-pair** P=128 base cohort as a variance
illustration, equal per-chain variance would give a four-chain P=256 κ relative
halfwidth of 6.23%, and a four-versus-eight log-rate difference halfwidth of
0.05567, already larger than log(1.05)=0.04879 even at zero point difference.
Four pairs may reduce momentum noise, but cannot be assumed to remove
parent-chain variance. This is a warning about likely limited precision,
not a prediction, power certificate or an observation of the active batch.
All four chains and any inconclusive outcome must remain reportable.

## Cost of the separate dynamics proposal

| Design | Native physical steps across both dt values | Historical linear-bead wall extrapolation |
| --- | ---: | ---: |
| Four chains × 200 parents × two pairs × two signs | 19,200,000 | 2.61–3.99 worker hours |
| Four chains × 200 parents × four pairs × two signs | 38,400,000 | 5.22–7.98 worker hours |

The lower number uses the later four completed P=128 base seeds and the upper
number uses all eight base seeds. Both multiply per-step cost by two for the
bead increase. They are **unmeasured P=256 extrapolations**, not upper bounds:
normal-mode cost, host contention, IO and failures may change throughput.
The eight base seeds consumed 14,357.406876 recorded worker-wall seconds for
38.4 million P=128 steps. The exact individual timing inputs and extrapolation
are retained in the audit.

The proposed four-pair P=256 raw volume is roughly twice the 325 MB base
P=128 child data under the same schema, plus about 14.75 MB of uncompressed
new parent-coordinate arrays. These are planning estimates; exact byte counts
would be recorded after completion. Replayed parent bytes, failed outcomes
and new child data must have separate immutable identities and archives.

No resource reservation or child launch follows from these estimates. The
next concrete step is review of the fixed replay protocol and a scheduling
wrapper; the original P=128 and P=256 evidence stays unchanged.
