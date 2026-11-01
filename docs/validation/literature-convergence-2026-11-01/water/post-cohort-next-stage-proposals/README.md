# Proposed next water stages after the complete original cohort

**These are two new, unadopted proposals. No registration, preparation extension,
production run, saved-RNG draw or lease acquisition was performed.** The paired
timestep control was reported active when these documents were prepared; its
normal completion and released leases remain explicit admission conditions.
The integration worktree and its running full-suite gate were not changed.

The immediate proposal is the previously reviewed **additional 400 ps of
preparation for all eight chains**. A separate prospective design then asks
whether diffusion estimates stabilize over longer correlation times. Additional
preparation and longer production address different failures, and require
separate adoption decisions.

## 1. Evidence that must stay visible

The complete original experiment retained eight seeds, 100 ps of preparation
per seed, and all 25 × 4 ps NVE segments per seed. Its
[original report](../original-fixed-cohort/analysis/report.json) remains identified
by SHA256 `f544d5de9202a73f7013ee5191fed4e53e330a9eeb41c87770bfaaa3f7d4d1f8`.
The normal exit, released original leases, raw-only archive and actual
[archive extraction/replay](../original-cohort-archive-replay/README.md) are
already recorded separately.

Three chains failed the original potential-energy preparation screen:

| Seed | U change, 75–100 minus 50–75 ps (kcal/mol/water) | Original absolute limit |
|---|---:|---:|
| 1102 | −0.157442 | 0.05 |
| 1105 | +0.056427 | 0.05 |
| 1106 | −0.108600 | 0.05 |

Every original temperature, OH-length and HOH-angle screen passed. These facts
do not make the failed potential-energy tests disappear. All eight chains,
including the five whose original U screen passed, receive the same proposed
additional preparation.

The original 0.5–1 ps MSD slope also exceeded the 1–2 ps slope by
**0.017118 Å²/ps**, with nominal paired-chain 95% interval
**[0.011973, 0.022264] Å²/ps**. The original compatibility/equivalence failures
and null formal literature comparison remain unchanged. The later
[VACF–MSD identity diagnostic](../vacf-msd-identity/README.md) explains this shift
through transient correlation behavior. Longer preparation cannot be assumed
to eliminate that physical short-time curvature.

## 2. Concrete preparation proposal: the same fixed additional 400 ps

[preparation-proposal.json](preparation-proposal.json) binds the unchanged
[version 2 proposal](../equilibration-extension-proposal-v2.json), the complete
original assessment and the frozen preparation runner/coordinator. All source,
model, environment, test/review and prior-evidence identities are listed in
[source-and-evidence-bindings.json](source-and-evidence-bindings.json).

The proposed execution uses the original frozen checkout and OpenMM CPU
environment. The source hashes, rather than a current branch name, determine
the eligible implementation. The native reference source is `92867b1d…`, the
preparation runner is `7442d986…`, and the coordinator is `f51c63e1…`. Full hashes
are in the binding file. Their existing real Reference/CPU boundary tests and
orchestration tests remain the evidence for those implementations; no duplicate
test execution is claimed here.

### Exact source states and RNG phase

[starting-state-bindings.json](starting-state-bindings.json) binds all eight
original `chunk-000099.npz` states at **absolute step 400000 / 100 ps**, with
`active_segment = -1`. It includes the complete checkpoint and manifest hashes,
canonical q/p hashes, raw engine state hashes, serialized model/integrator
identities and saved RNG hashes. The data come from the already sealed all-eight
selection receipt; this proposal did not reopen the live original checkpoints
or acquire any currently held lease.

| Seed | Exact checkpoint SHA256 prefix |
|---|---|
| 1101 | `0b9ed7afffa6286b` |
| 1102 | `0b151153c8223824` |
| 1103 | `b7df140f0bbee88d` |
| 1104 | `ca9c20aed155f396` |
| 1105 | `e900bfcaec5455ee` |
| 1106 | `58fdb04cd5c44c35` |
| 1107 | `e1ed293fba161258` |
| 1108 | `86e63aa2de12c48d` |

At later registration, the frozen coordinator must rehash the actual selected
files while holding all original exclusion leases. A filename or the old
selection receipt alone is insufficient for execution.

Each branch restores its exact 100 ps state and saved RNG, performs exactly the
pending preparation Maxwell draw with **no COM projection**, and continues the
unchanged refresh schedule every 400 steps / 100 fs. The 0.25 fs time step,
216-molecule cell, density, PME, cutoff, masses and force field remain fixed.
This deliberately retains the original periodic thermostat; it does not claim
that small method tests establish ergodicity or equilibrium.

Every seed then advances exactly **1,600,000 additional steps**, reaching
**absolute step 2000000 / 500 ps**. It stops before any terminal preparation
refresh or NVE momentum draw. The saved additional refresh count is 4,000.
The original NVE trajectory and the paired timestep trajectory are never used
as selected replacement endpoints.

The version 2 CPU checkpoint contract remains unchanged: raw positions and
half-step velocities must match exactly, independently bounded force-cache
differences must explain the canonical momentum difference, and future CPU
trajectories are not described as bitwise replays.

### Fixed late-window assessment

All eight chains must complete the full budget before the final preparation
assessment. The two windows are **(300, 400] ps and (400, 500] ps**; there are
1,000 scalar observations per window per seed at the unchanged 100 fs cadence.
No latest good window, favorable subset or early pass is selected.

The original per-seed absolute change thresholds stay fixed:

| Observable | Maximum absolute change between the fixed windows |
|---|---:|
| Potential energy per water | 0.05 kcal/mol |
| Temperature | 5 K |
| Mean OH length | 0.001 Å |
| Mean HOH angle | 0.1° |

All 32 individual scalar screens must be reported. An eight-chain average
cannot hide a failed individual screen. Fixed 5, 10 and 20 ps block summaries
and scalar correlation curves through 50 ps are also retained. The block sizes
are reported together; a favorable one is not chosen after seeing the data.
Slow unresolved correlation prevents an equilibrium claim even when the point
thresholds pass.

The proposed structural policy retains every seed's complete OO RDF and fixed
3.5 Å coordination statistic in both windows, sampled every 0.5 ps. OO bins are
0.02 Å wide, with exact shell normalization and an explicit final partial bin
if needed at half the stored box length. OH and HOH distributions use every
1 ps checkpoint and fixed bins, including explicit underflow/overflow counts.
The machine-readable proposal gives their complete definitions.

RDF and coordination thresholds remain **descriptive, unadopted proposals**.
The reporting policy uses per-seed fixed-block diagnostics and pointwise
nominal intervals across the eight chain differences. It does not claim
simultaneous coverage, equivalence from an interval crossing zero, or a
calibrated liquid false-alarm probability. The offline analysis definitions
and implementation require independent review and freezing before any long
preparation begins. Preparation completion never launches production
automatically.

### Admission after the paired timestep control

The paired-control registration is already bound by SHA256
`0d73545ac4751b84bdf64a99d693dee23a7f683bb3cd38e2cff5b7d8055e8365`.
No partial control outcome was read or used to design these stages.

Before adopting the unchanged 0.25 fs preparation, the later root decision must
bind the complete paired-control report, require all eight pairs and both
numerical equivalence checks plus the fixed energy checks, and observe the
entire control launcher exiting normally. A failed or inconclusive control
does not trigger a favorable-seed selection or automatic preparation launch.

The control launcher must be retired and its own lease released. The frozen
preparation registration then acquires **all eleven original leases itself**,
checks every immutable source state, and publishes its new registration last.
Do not hold those same eleven locks through a separate external acquisition
while invoking registration. The future preparation coordinator reacquires
the original and new leases, retains committed prefixes, and drains existing
children normally on an admission stop or handled failure.

The frozen coordinator predates the new paired-control outcome condition. It
enforces original-cohort identity and writer exclusion; the later root
admission/launcher must additionally verify and hash-bind the paired-control
result, normal exit and cumulative resource budget. That root adoption and
launcher have not been created here.

[future-root-adoption.template.json](future-root-adoption.template.json) fills
the now-known original assessment and state identities but deliberately retains
an invalid pending decision, null review time and unresolved control-exit/result
fields. A later adopted record must be a separate document, preserving this
draft.

## 3. Separate prospective longer-lag production design

[long-lag-production-design.json](long-lag-production-design.json) is a new
proposal for testing asymptotic behavior on **new data**. The old version 2
preparation document remains unchanged; its unadopted suggestion of another
25 × 4 ps cohort is not adopted by this design.

| Property | Proposed fixed value |
|---|---|
| Chain identities | All eight original seeds, 1101–1108 |
| Starting state | Exact, assessed 500 ps endpoint of every preparation branch |
| NVE segments per chain | 20 |
| Segment duration | 20 ps |
| NVE time per chain / all chains | 400 ps / 3,200 ps |
| Integration time step | 0.25 fs, conditional on the numerical control |
| Observation cadence | 5 fs; 4,001 frames per segment |
| Checkpoints | Every 1 ps |
| Maximum correlation lag | 10 ps; 2,001 time origins per molecule at that lag |
| Primary VACF integral cutoff | 5 ps |
| Primary MSD fit interval | 5–10 ps |

Each segment starts with exactly one COM-projected Maxwell momentum draw from
the continued saved RNG. The thermostat is disabled during the entire segment.
Checkpoint/restart must preserve the absolute and segment clocks, pending-draw
phase and RNG, without an extra draw at resumption or after the final segment.
The output is a separate OpenMM classical external reference, retaining the
same Hamiltonian and finite cell.

The design compares fixed VACF cutoffs of 2, 7.5 and 10 ps with the primary
5 ps cutoff, and fixed MSD windows of 2–5, 5–7.5 and 7.5–10 ps with 5–10 ps.
The old 0.5–1 and 1–2 ps windows remain descriptive outputs on the new data,
while their original failed comparison stays unchanged in the old evidence.
There is no fitted-tail switch, automatic window search or assumed zero tail
beyond the measured correlation range.

The point estimates retain all molecules and lag-specific origins. Each chain
averages all 20 segments equally; the final estimate averages eight chain means
equally. Standard errors and nominal df7 intervals come from the eight chains.
Every estimator/cutoff/window contrast is formed within chain before calculating
its uncertainty. Segments 0–9 versus 10–19 provide a fixed paired production-time
check, while all 20 remain in the primary estimate. Old and new cohorts share
ancestry and are never counted as sixteen independent chains.

The proposed new finite-lag screens require both primary 95% interval halfwidths
to be at most **0.01 Å²/ps**, and the complete paired 95% interval for each fixed
contrast to lie inside **±0.01 Å²/ps**. The sampled total-energy envelope remains
0.01 kcal/mol/water and the absolute linear drift limit remains
0.001 kcal/mol/water/ps, applied to each new 20 ps segment. These are prospective
screens for the new study. Failure or insufficient precision retains every
record and yields an unresolved result; it cannot trigger deletion, a longer
budget, or a more favorable estimator.

This production stage requires a **new, separately reviewed production runner
and general-lag analyzer**. The frozen original analyzer has a 4 ps / 2 ps
contract and must not be silently repurposed. The new source and actual 500 ps
endpoint hashes do not yet exist. The design lists the necessary phase/restart,
normalization, membership, tamper and lifecycle checks before adoption.

## 4. Finite resources and incomplete-run policy

[resource-estimates.json](resource-estimates.json) records the arithmetic and
its source evidence. The retained original 67 ps invocation measured
0.005103 CPU seconds per integration step and 0.006749 logged wall seconds per
step under its observed CPU allocation.

| Proposed stage | Added steps across all eight chains | Forecast CPU time | Forecast wall at four workers under the measured allocation |
|---|---:|---:|---:|
| Additional 400 ps preparation | 12,800,000 | 18.14 h | 6.00 h |
| Separate 20 × 20 ps production | 12,800,000 | 18.14 h | 6.00 h |

These are forecasts from an NVE invocation, not guaranteed upper bounds;
preparation cadence and shared-host contention can differ. Each stage proposes
a separate maximum of **32 single-CPU worker-admission hours**. For preparation,
this means at most four full 2 h coordinator invocations with four workers, or
the equivalent cumulative charge with one or two workers. Each native call is
at most 1,800 s of nominal admission. No CPU is reserved by this proposal.

The frozen runner checks its budget at committed 1 ps boundaries. Startup,
hashing, checkpoint publication and normal drain can exceed nominal admission
durations. The budget is therefore explicitly an admission budget, with no new
OS hard kill of a water integration child. A later launcher must record its
cumulative charge. Exhausting the fixed budget stops new invocations and
retains incomplete prefixes; it does not shorten the physical stage or
automatically authorize more compute.

The proposed water allocation is at most four one-thread workers with 2 GiB
total planned integration memory. Offline analysis processes one segment at a
time, with a separate one-process budget of 1,800 CPU seconds / 3,600 wall
seconds and 2 GiB planned memory. Actual headroom must be checked before root
adoption; no current resource limit or other process was modified.

Original stored chunk sizes project approximately **0.60 GB** of new preparation
chunks and **4.99 GB** of new production chunks. The combined storage proposal
reserves 20 GiB for raw data, raw-only archives, independent replay extractions
and metadata. These are storage forecasts; raw checkpoints will remain outside
Git, with source code identified by Git/hash references. No upload or additional
archive was created here.

## 5. Scientific scope and source context

The original q-TIP4P/F paper obtains diffusion from 2 ps velocity correlations
over 100 consecutive 4 ps NVE segments with Maxwell momentum refreshes,
repeating that procedure twenty times. The present longer-lag proposal asks a
separate time-regime question motivated by the complete original failure; it
does not silently relabel a changed protocol as the published one. See
[Habershon, Markland and Manolopoulos, Sec. III.B](https://arxiv.org/pdf/1011.1047).

Correlations between MSD values and short-time nonlinearity require care in
uncertainty estimation. This design retains the existing fixed-window OLS point
estimate for comparability, while obtaining uncertainty across complete chain
means instead of treating fitted lag points as independent. The general issue
is discussed by [Bullerjahn, von Bülow and Hummer](https://arxiv.org/abs/2003.09193).
The full source metadata and scope of these citations are recorded in
[literature-context.json](literature-context.json).

The same finite 216-molecule box and model are retained; no finite-size
correction is chosen to improve reference proximity. Existing source-parameter,
Table-I rounding and CPU-rounding qualifications remain. These proposed stages
provide no quantum, IR, GPU or PyRPMD-production validation result. Even a future
pass of all proposed finite-lag screens would require a separate scientific
assessment before claiming paper agreement.

[proposal-consistency.json](proposal-consistency.json) records the performed
metadata/source checks. They verify complete seed membership, exact source-state
references, unchanged scalar thresholds, duration/sampling arithmetic, preserved
evidence hashes and the invalid pending adoption. They contain **zero dynamics,
zero PES evaluations, zero Context creation and zero saved-RNG draws**.
