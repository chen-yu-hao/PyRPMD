# P256 child binding and complete-cohort analysis

These additive tools implement the
[fixed eight-chain design](../p256-fixed-eight-chain-design/README.md).
**No P256 child dynamics or bead-convergence result is supplied by this
software gate.** The separate parent campaign must first finish all eight
chains and publish its eligible manifest. The child coordinator, its
resource adoption and its final all-branch completion manifest are separate.

`benchmarks/h3/p256_surface_recrossing.py` imports the unchanged P128
surface, channel, Maxwell-momentum and checkpoint functions. Independent
AST review verifies that its native propagation, signed-flux accumulation,
momentum keys and child-checkpoint body are identical to the frozen P128
implementation. The adapter resolves each historical/new parent directory
from the bound proposal, checks every parent file identity, and rejects a
calibration or subset bypass. The registered protocol also binds both
analysis source files before any children run.

Each of eight independent chains supplies 200 parents. Every parent uses
four ordinary Maxwell momentum pairs, with both signs at both 0.1 and
0.05 fs for 200 fs. The source includes the configuration flux metric;
child weights divide by that metric. The Maxwell variance is `mP/beta`,
so the centroid velocity variance contains no extra factor of P. The
expected flux denominator remains 2. A branch report's `complete` flag
does not imply completion of the whole experiment.

`benchmarks/h3/p256_recrossing_analyze.py` requires all eight seeds, both
timesteps and both two-pair batches before reading primary flux statistics.
It retains the same-chain covariance of log partition and log κ. P256
uses its own previously computed finite-P reactant log ratio
`-7.282252233219134`; the P128 denominator is not reused. The full-rate
comparison uses independent eight-versus-eight Welch intervals and the
predeclared whole-interval margin ±log(1.05). Children and parent snapshots
are not treated as independent chains.

The analyzer preserves the original 3% κ precision, timestep, plateau,
energy and sampled-domain checks. Parent-time-half equivalence is reported
separately, including failures. Nonpositive finite κ retains every chain
but disables log-rate inference. Nonfinite complete-cohort outcomes yield
an explicit failed report, retaining every row and each invalid field's
location/representation; no κ curve or passing conclusion is generated.

## Executed checks and review history

The final combined local gate passed **58 tests** on Python 3.12:
30 parent/branch contracts and 28 statistical/analysis checks. These use
synthetic parent data, inert metadata-only models and independent SciPy
statistical oracles. They perform **zero PES evaluations and zero physical
dynamics steps**. Cross-version integration checks are separate evidence.

- The initial 28-case binding gate passed and remains in
  [its original receipt](initial-28-case-gate/receipt.json).
- Independent review found and resolved the original nested-directory
  routing error and a missing connection between checkpoint summary and
  actual file inventory. The intermediate 29-case gate passed after this
  fix; its exact test output is retained separately.
- The initial analysis gate had 17 passes and five native-boolean contract
  failures. Its [source and failure receipt](initial-analysis-gate/receipt.json)
  remain unchanged. Numeric Welch comparisons already agreed with SciPy;
  the helper's result type was corrected.
- Additional full-analyzer fixtures cover omitted seed/timestep/batch,
  stale helper identity, zero κ and nonfinite outcomes. Fixture baseline
  data is synthetic and portable; the tests do not require live experiment
  directories outside the checkout.
- [Independent adapter review](independent-adapter-review.json) and
  [additive analysis review](independent-analysis-review.json) preserve all
  findings, source identities and their resolutions. The second review
  covers the additional analysis-source binding without replacing the first.
- [Final combined gate](final-gate.json) records exact source/test and output
  identities. The final test inventory is counted once, independently of
  the earlier runs and repeated review checks.

This software gate establishes input, estimator and reporting contracts.
It does not supply a physical P256 trajectory, convergence result, revised
PES, paper fit or validation of the production MD umbrella workflow.
