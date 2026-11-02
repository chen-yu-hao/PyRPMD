# Independent review of the complete 200 fs H₃ child cohort

The independently replayed raw arrays reproduce the reported fixed-
128-bead quantum rate, its eight-chain uncertainty, and the reason for the
registered precision extension. All eight seeds and all 1,600 immutable
static parents are present. The original two momentum pairs give 6,400
children at each of the paired 0.1 fs and 0.05 fs timesteps. The only failed
mandatory gate remains the relative 95% half-width of κ: **3.2722% against
the registered 3% limit**. This review does not change that result.

## What was checked

[replay.py](replay.py) reads the original parent and child checkpoints without
calling the producer, analyzer, PES, or dynamics. It verifies all 400 child
chunk hashes and integrity envelopes, all eight parent checkpoint envelopes,
and the frozen protocol, model, source, settings, and report identities. It
reconstructs 6,400 keyed Maxwell arrays and 3,200 keyed channel selections
(including both timesteps), and checks that each timestep uses identical
parents, momenta, and channel choices. All 483 bound input files remained
unchanged during the review.

The explicit Cartesian surface gradient independently gives the stored flux
metric and surface velocity exactly in this run. Every saved signed
numerator and absolute-flux denominator replays bit for bit. Negative or zero
contributions are retained. Per-chain summation and Student-t statistics are
then rebuilt with an independent implementation, preserving the covariance
between each parent's log partition estimate and its log κ. The independent
uncertainty unit is the **whole parent chain: eight units, seven degrees of
freedom**. Children, antithetic partners, timesteps, and time origins are not
counted as additional independent chains.

The read-only replay passed in 4.79 seconds on one CPU. Its original output
is [replay.log](replay.log); numerical results are in
[raw-replay.json](raw-replay.json), and exact input identities are in
[input-file-hashes.json](input-file-hashes.json). The review plan was written
before the raw replay and endpoint calculations: [review-plan.json](review-plan.json).
Workspace input paths in the hash list identify local raw files; they are
not promises that raw child trajectories are included in this publication.

## Reproduced statistical results

| Quantity | Independent replay |
| --- | --- |
| Full rate, geometric chain estimate | 1.9719603515 × 10⁻¹⁶ cm³ molecule⁻¹ s⁻¹ |
| Nominal 95% chain interval | [1.9114198197, 2.0344183878] × 10⁻¹⁶ cm³ molecule⁻¹ s⁻¹ |
| κ, arithmetic chain estimate | 0.7730772916 |
| κ nominal 95% interval | [0.7477806569, 0.7983739263] |
| κ relative 95% half-width | 0.03272199947; registered limit 0.03 fails |
| Paired coarse-minus-fine κ difference | 0.0008108683; nominal 95% interval [−0.0008366822, 0.0024584187] |
| Paired late-minus-early κ difference | Exactly zero in each of the eight observed chains |
| Second-minus-first parent-half κ difference | 0.0221070857; nominal 95% interval [−0.0205559376, 0.0647701090] |
| Mean denominator diagnostic | 2.0121506132; nominal 95% interval [1.9503523676, 2.0739488588] |

The mean denominator is consistent with the analytical value 2 for the
ordinary-Maxwell, antithetic, flux-weighted-parent estimator. Both numerator
and denominator divide by the parent flux metric; the replay confirms this
normalization and the signed product-side convention. The exact finite-P H₂
normalization and independent classical TST factor are bound to the same
temperature, mass, model, and bead count used by the children.

The cross-covariance of the chain means is −2.5248830648 × 10⁻⁵ in log units.
Including it gives a log-rate variance of 0.0001738916514; incorrectly
discarding it would give 0.0002243893127. The published central rate is the
geometric mean of chain rate estimates. It is not the product of an
arithmetic κ mean and a separately averaged QTST estimate.

The complete report's reference rate is 1.95 × 10⁻¹⁶ cm³ molecule⁻¹ s⁻¹. The
point estimate is 1.126% higher. In this particular completed cohort, the
nominal rate interval both contains that point and lies wholly inside its
linear ±5% band. These observations are separate from the registered
point-within-5% gates and precision gates; their conjunction is not, in
general, a whole-interval equivalence test. The failed supplementary
parent-half equivalence check (margin ±0.03) remains visible and is not
reclassified as a pass.

## Endpoint diagnostics and their limits

The endpoint quantities below were declared in the review plan before the
raw pass. They are supplementary diagnostics, not new acceptance gates.
For each final configuration, the closest centroid pair is identified and
the third atom's displacement and velocity are measured relative to that
pair's center of mass.

| 0.05 fs primary endpoint diagnostic | Result over all 6,400 children |
| --- | --- |
| Closest centroid-pair distance | 1.103–1.689 bohr; median 1.388 bohr |
| Third-atom separation from pair center | 19.716–48.722 bohr; median 35.610 bohr |
| Outward relative radial velocity | 0.116264–0.243808 bohr/fs; positive in every child |
| Separation at least 20 bohr | 6,399 of 6,400 children |
| Observed surface-sign changes from 80 to 200 fs | None at the saved 1 fs observation points |

These observations support an outgoing, widely separated endpoint ensemble.
The closest pair is a geometric diagnostic, not an energy-based proof of a
bound molecule. The observed late-time difference and its degenerate
Student-t interval `[0, 0]` do **not** bound arbitrarily rare later recrossings.
No extra propagation was performed for this review.

The minimum recorded bead-pair distance is 0.5588503650 bohr, above the 0.5
bohr static domain boundary. These distances are observed at 1 fs boundaries,
not at every physical integration step. The largest primary 0.05 fs energy
deviation is 0.0722056927 kBT per bead and passes the registered 0.1 stability
screen. The coarser timestep reaches 0.2711910820; its separate paired rate
comparison passes. Neither energy screen is a rigorous rate-bias bound.

## Extension decision and chronology

The completed all-eight-seed report was produced before the immutable
precision-extension decision, which binds the full report and execution
hashes. The decision preserves every original gate and requests only the
already registered addition of momentum-pair indices 2 and 3 for **every**
same parent and seed, at both timesteps and the same 200 fs horizon. Plateau
and timestep gates passed; κ precision alone triggered the extension.
Paper-point agreement did not trigger or stop it.

The root's separate post-launch receipt states that the immutable decision
was written in a preceding sequential command, and that the resulting
running session was observed no later than 18:17:51 UTC. The exact launch
timestamp was not captured. This review verifies the decision, launch-receipt,
and schedule identities and their agreement; it does not reinterpret the
observation bound or a filesystem timestamp as an exact launch time.
[review-receipt.json](review-receipt.json) records those hashes and limits.
No partial results from the additional momentum-pair batch were read.

## Statistical scope

The eight-chain Student-t intervals are nominal approximations. Nonlinear
ratio and log estimates, finite-chain geometric-mean bias, and the
conditional precision extension do not acquire exact or sequential
coverage merely by replaying the arrays. Additional momenta improve the
conditional child estimate; they do not supply new independent parent
equilibrium chains or resolve the supplementary parent-half result.

This review checks the frozen fixed-P, fixed-domain result. Separate static
bead/domain controls do not automatically validate κ at other bead counts
or cutoffs. The exact reactant and classical numerical references are
treated as fixed numerical inputs here. No production-umbrella convergence
claim, new physical equivalence gate, or general literature-validation
claim is added.

The seven new text/code files are listed in
[publication-inventory.json](publication-inventory.json). Existing frozen
review inventories, sources, protocols, reports, checkpoints, and Git state
were not edited.
