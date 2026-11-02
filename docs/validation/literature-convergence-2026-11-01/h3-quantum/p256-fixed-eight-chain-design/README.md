# Fixed eight-chain P=256 control: proposal before implementation

**No parent sampling or child dynamics has been launched.** This is a new
fixed-cohort proposal, separate from the frozen
[four-chain feasibility assessment](../p256-dynamics-readiness/README.md).
That assessment found that four P=256 chains could leave the full-rate
equivalence interval too wide. The proposed control therefore uses eight
independent P=256 chains from the outset. It will not stop after four chains
and decide whether to add more from a favorable or unfavorable point estimate.

The [machine-readable proposal](proposal.json) binds all eight seeds, complete
expected settings, original-source identities, child definitions, statistics,
resource limits and failure rules. It remains subject to review, separate
implementation, bounded tests, source freeze and a recorded launch decision.
The original four-chain equilibrium result and its raw checkpoint identities
remain unchanged.

## Fixed parent cohort

| Role | Seeds | Required evidence |
| --- | --- | --- |
| Historical deterministic replay | 825601, 825603, 825607, 825609 | Every original non-snapshot payload is byte-identical; add only deterministic parent observations |
| New independent chains | 825613, 825617, 825621, 825623 | Fresh empty directories, the exact frozen settings and complete new raw state/RNG/provenance |

The four new seed labels were fixed before drawing any new random number or
evaluating the PES. An exact seed-record/path search in `benchmarks`, `docs`
and `tests` found no prior registered use. They were not selected from a seed
search, test simulation or comparison with a literature point.

Every chain uses the same 300 K, hydrogen mass, P=256, BKMP2 source and library
bytes, nine-lambda grid, 2,000 warmup sweeps and 20,000 production sweeps. The
proposal retains the existing 0.5 bohr bead-pair lower bound, 8 bohr centroid
upper bound, 131,072 reference-domain draws, move sizes and exchange schedule.
It stores 200 lambda=1 configurations at completed sweeps 2,100, 2,200, …,
22,000. All eight chains thus yield 1,600 parent configurations, with **eight**
independent uncertainty units.

### The historical half must be an exact replay

The four old controls have `snapshot_interval=0` and no stored parent sequence.
In a new directory the unchanged sampler can execute the same seed and settings
with `snapshot_interval=100`. Its existing snapshot observer consumes no RNG
draw and does not alter the sampled state. Acceptance requires identical
dtypes, shapes and bytes for all 16 original non-snapshot arrays, including
the entire energy/diagnostic history, final ring paths, RNG and counters.
Original reference/domain bytes, physical analysis, source and runtime
identities must match as well.

The new NPZ and settings hashes will differ for the explicitly added observation
fields. Those files need new identities linked to the old files by the byte
audit; they do not replace the originals. A replay mismatch is a failure and
cannot be relabeled as a successful new independent seed.

### The new half starts independently

The four new chains use their own normal sampler and reference-domain RNG
streams. They do not copy a historical final state. Each must preserve its
complete checkpoint, settings, domain normalization, 20,000 energy records per
lambda, scalar diagnostics and all 200 parent configurations, including any
interrupted or failed outcome.

For every chain, the existing parent-sampling flags still require minimum
BAR overlap 0.2, minimum diagnostic energy ESS 200, and absolute first/second
half log-partition difference at most 0.2. Every parent must also satisfy the
finite geometry, active centroid surface, static bead domain and zero-COM
checks. None of these tests establishes 200 independent parents; the chain
remains the uncertainty unit.

One final parent-eligibility manifest must cover **all eight** chains and all
their source/settings/reference/domain/report/checkpoint identities. A partial
four-chain estimate cannot enable children or change the remaining budget.
Missing, failed or poorly sampled chains are retained and prevent a successful
whole-cohort gate; they are not dropped or replaced.

## Parent scheduling and measurement

The proposed allocation, only after root records that the existing H3
extra-pair pool has drained normally, is two umbrella-MC workers, two P=256
parent workers and four water workers. This is eight project worker slots;
the proposal does not also increase the MC pool to four.

The parent coordinator admits the fixed eight-seed order with at most two
separate processes. Each worker uses one recorded CPU and one BLAS/OpenMP
thread; the Fortran potential is not shared between concurrent threads.
The **7,200-second whole-parent-campaign wall ceiling** covers both historical
replay and all four new chains, setup, IO, verification and analysis. It is
not a per-seed allowance.

The original four chains recorded 1,736.178563 seconds of timed sampling
worker-wall duration. Doubling that work gives about 3,472 seconds of worker
time and an idealized 1,736-second elapsed estimate with two workers. This
excludes the original untimed setup/analysis, assumes comparable new-chain
cost and does not predict shared-host contention. Original process CPU was
not measured. The new coordinator must record per-call user/system CPU with
`rusage`, whole-call wall time, actual worker/affinity/thread identities,
peak RSS and the full completion ledger.

A separate tested coordinator must hold an exclusive campaign lease and drain
all admitted workers before releasing it after a timeout, failure or signal.
It must preserve all partial checkpoint bytes and failure receipts, stop new
admissions, and publish an incomplete status rather than a subset success.
No implementation or worker has yet been created by this proposal.

## Prospective child definition

The child experiment remains a separate launch decision after all eight parent
chains have passed. The existing primary child driver binds the P=128 parent
protocol and should keep rejecting these P=256 inputs. A new reviewed parent
binding may reuse the unchanged flux, keyed-momentum and native-propagation
functions; it may not bypass validation with the calibration flag or modify
the active primary driver.

The fixed initial design is:

- Eight independent P=256 chains and all 200 parents from each chain.
- Four independently keyed ordinary Maxwell momentum draws per parent,
  indices 0–3, each propagated with plus and minus signs.
- Both 0.1 and 0.05 fs timesteps to 200 fs; 0.05 fs remains primary.
- The same physical hydrogen mass, 300 K, centroid `s1=0` dividing surface,
  equivalent-channel convention, BKMP2 bytes and native Cayley-BCB Hamiltonian
  propagation, without a thermostat.
- Surface and minimum-distance observations every 1 fs, and full ring energy
  extrema at every physical step.

Maxwell variance is `m*P/beta`. The parent source already contains the flux
metric, so the per-chain transmission estimator remains

\[
\kappa_P(t)=
\frac{\sum (v/f_s)[H(s_+(t))-H(s_-(t))]}
     {\sum |v|/f_s}.
\]

Within P=256, both timestep runs use the same parent, equivalent channel and
momentum. Their comparison is paired. P=256 and P=128 use independent parent
chains of different dimensions and cannot be paired by matching an index or
inventing a common seed history. Antithetic signs are likewise not additional
independent replicas.

The P=128 baseline is the complete eight-chain, four-pair result from pair
batches 0 and 1 on the original 1,600 parents. Its final comparison and full
input hashes must be bound before the P=256 child launch. No partial extra-pair
rate has been read to choose these P=256 settings or the eight-chain count.

## Compare full rates with the covariance intact

First form the complete log-rate estimate within each chain:

\[
L_{P,i}=\log k_{\rm classical}^{\rm TST}
+\log(Z_{\rm TS,P}/Z_{\rm TS,1})_i
-\log(Z_{\rm R,P}/Z_{\rm R,1})
+\log\kappa_{P,i}.
\]

The same chain supplies its BAR surface correction and its child parents,
so their covariance must be retained before computing the group variance.
The finite-P reactant log corrections remain −7.282252233219134 for P=256
and −7.27275868577072 for P=128, bound to the existing transfer reference.
The common classical reference cancels from the between-group log difference.

Use the independent eight-versus-eight Welch comparison

\[
\Delta=\bar L_{256}-\bar L_{128},\qquad
\mathrm{SE}^2=s_{256}^2/8+s_{128}^2/8,
\]

with the Welch degrees of freedom calculated from the two measured chain
variances. The complete two-sided 95% interval must lie within
**±log(1.05)**, the same convention as the existing equilibrium control.
Its rate-ratio band is [1/1.05, 1.05]. A small point difference, overlapping
marginal intervals or an insignificant difference is not an equivalence pass.

κ retains its separate relative 95% halfwidth gate of 3%. Each group's
eight-chain Student-t interval has seven degrees of freedom. Report independent
between-group κ difference and log-ratio intervals as well. A nonpositive
plateau invalidates a log-rate calculation and must remain a reported failure;
it cannot be omitted to make the remaining seven chains analyzable.

Within P=256, the paired 95% intervals for the 80–100 versus 180–200 fs plateau
and for the two timestep results must lie inside ±0.01 in κ. The primary
0.05 fs maximum ring energy error, scaled by `beta/P`, must be at most 0.1.
The minimum-distance diagnostic is only at 1 fs boundaries, not an every-step
bound. Initial equilibrium-domain restrictions do not become reflecting walls
in the child dynamics.

With the completed two-pair P=128 base variance as an illustration, eight
versus eight chains at equal variance would have a 95% log-difference halfwidth
of about 0.04000, compared with 0.05567 for the unlaunched four-versus-eight
option. This motivates a larger fixed independent cohort. It does not predict
the P=256 point shift, actual variance, κ precision or successful equivalence.
Even the eight-chain experiment may be inconclusive. Extra momenta may reduce
momentum noise without curing parent-chain variance.

All eight chains, all gates and any failure or wide interval remain in the
final report. No favorable-result stopping, parameter adjustment or validated
flag is permitted. Any later extension needs its own fixed protocol and
resource decision, based on precision or numerical diagnostics rather than
proximity to a paper number.

## Child budget: substantial and still unmeasured at P=256

The fixed eight-chain design requires **76.8 million native physical steps**
across both timesteps, or 19.6608 billion bead steps. Linear bead-cost
extrapolation from the immutable P=128 base timings gives **10.44–15.95
worker hours**. The lower estimate uses the later four base seeds; the upper
uses all eight. This is approximately 5.2–8.0 hours elapsed on two workers
under ideal division of work, excluding setup, queues, failures and changed
host contention. P=256 throughput has not been measured and normal-mode
cost need not scale exactly linearly.

The raw child schema suggests roughly 1.3 GB for this design. The 1,600 parent
coordinate arrays alone contain exactly 29,491,200 uncompressed bytes; actual
compressed archive sizes would be measured after completion. Outputs must
preserve exact byte identities and failures separately from the old four-chain
and P=128 archives. No raw upload or Git action is part of this proposal.

The next step is parent review of these fixed inputs, then a small separately
tested coordinator and parent binding. Nothing in this document starts either
the eight-parent campaign or the longer child calculation.
