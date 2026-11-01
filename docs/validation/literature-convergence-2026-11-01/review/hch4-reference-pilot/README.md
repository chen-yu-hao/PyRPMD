# Excluded H + CH4 finite-domain reference pilot

## Result and scope

The single evaluated pilot completed its fixed 768 target draws, including 641
successful CBE/PES-2008 evaluations and 127 geometric zero contributions. Energy
evaluation was inexpensive in this run. The observed importance weights were
severely concentrated, so this pilot does **not establish a usable precision
reference or a reaction rate**. No additional draws or changes to the proposal
were made after inspecting the result.

| Target | Fixed draws | Nonzero geometric weights | Empirical ESS | Largest normalized weight |
| --- | ---: | ---: | ---: | ---: |
| Classical surface, alpha = 0.999 | 256 | 214 | 4.3285 | 27.647% |
| Classical surface at alpha = 1.007 | 256 | 214 | 7.1830 | 22.008% |
| Reactant entry surface at 30 bohr | 256 | 213 | 6.1429 | 32.999% |

The four individual abstraction-channel estimates each use 64 unconditional
draws. Their empirical ESS values range from 1.1255 to 3.9219; one draw carries
94.222% of its channel's observed total weight. These estimates cannot exclude
important unseen tails. The two classical surfaces also share latent proposals,
and all observations belonging to a diagnostic seed are retained together.
Channels and paired surfaces do not create additional independent replicates.

The pooled, dimensionless log flux-ratio diagnostics are retained in the raw
report, including the poorly determined paired surface difference. Their
presence is not an accuracy claim. In particular, the alpha = 1.007 calculation
still has **one bead**; it is a classical reference at a surface used in the
published quantum calculation. No quantum correction, transmission coefficient,
converted thermal rate, confidence interval or literature-agreement flag was
produced.

See [execution summary](execution-summary.json),
[complete report](completed-run/report.json), and
[independent output replay](independent-output-replay.json).

## Fixed design

The [measure and support derivation](../hch4-reference-feasibility/README.md) and
[original pilot design](../hch4-reference-feasibility/pilot-design/README.md)
remain unchanged. The exact [proposal JSON](../hch4-reference-feasibility/pilot-design/proposal.json)
was frozen before implementation. Its historical draft status is preserved;
adoption and the actual execution are recorded separately in the
[namespace-safe execution plan](execution-plan-namespace-v2.json).

The eight fixed diagnostic seeds are 2026111201 through 2026111208. Each seed
has eight latent draws for each of four abstraction channels, evaluated at both
alpha values, plus 32 independent entry-surface draws: 96 target geometries per
seed and at most 768 PES evaluations in total. The full
[latent ledger](completed-run/ledger.json) was written before any PES call and
has SHA256 `8c1018119828753931319f08bf10304ee3f089af7f1f79cf8449ab81c47f8eb7`.

The deliberately provisional molecular domain fixes all four methane C-H radii
to [1.4, 4.5] bohr and all six-atom pair distances to at least 0.7 bohr. Transition
draws also require a positive radial root, the designated active maximum
channel, and the connected reaction-coordinate domain. The 127 zero
contributions comprise 114 methane-radius exclusions, five minimum-pair
exclusions and eight inactive-channel exclusions. They remain in unconditional
means and were never replaced. There was no PES exception or nonfinite result.

The exact full shape mixture has a uniform finite-box component and two
untruncated reflected Gaussian components. The uniform component supplies
support over the entire declared finite molecular domain. The full mixture
density is used in every denominator. Four separately sampled transition-channel
means are summed once, and divided by the single entry-chart mean. The central
ratio uses pooled integral means, rather than averaging noisy seed ratios.

Full proposal support is a mathematical property. It does not validate the
finite molecular domain, omitted tails, PES reliability there, separation at
30 bohr, absolute normalization against a numerical oracle, or estimator
precision. The observed weight concentration leaves all of those scientific
questions open. Any later proposal-efficiency or tail study needs its own
independent design and authorization; this pilot contains no continuation rule.

## Cost and resource accounting

The evaluated attempt used one CPU affinity and one thread for BLAS/OpenMP.
The supervisor's wall interval was 2.0601 seconds. Measured CPU was 1.0400
seconds for the entire worker, 0.7150 seconds for preparation before its final
registration write, and 0.6734 seconds for the supervisor before its final
receipt. The latter two measurements exclude their remaining finalization;
their enforced reserves bound that cost.

The three whole-process hard CPU allocations were 2 seconds for preparation,
3 seconds for supervision and 55 seconds for the worker, summing to the fixed
60-second ceiling. The worker's 55-second soft admission check charges both
earlier process caps. The outer supervisor ceiling was 120 seconds. Runtime
receipts distinguish requested allocations from actual installed limits and
measured usage. No budget extension or physical dynamics occurred.

The 641 recorded evaluator intervals total 0.08526 CPU seconds, with a median
of 0.11260 milliseconds and a maximum of 2.19780 milliseconds. These intervals
include the pre-call attempt record and fsync, plus the energy/force adapter
evaluation. They are not pure energy-only kernel timings and cannot be directly
equated with complete umbrella or quantum-trajectory cost.

## Preserved preflight failure and repair

The first attempt stopped before Popen and before any PES call because two
separate managed execution namespaces both assigned PID 4. Bare PID inequality
therefore could not establish separate interpreter lifetimes. Its original
source, registration, ledger, one-attempt marker, log and
[failure receipt](attempt-1-preflight-failure/failure.json) are retained.
The original budget file's `hard_caps_enforced` field described intended
allocation before the guard failed; no supervisor or worker limit had yet been
installed. Only the completed preparation's two-second cap was installed.

Root explicitly authorized the necessary process-identity repair and one new
bounded attempt. Invocation UUIDs now come from OS entropy, independently of
every scientific RNG. Same-interpreter preparation and supervision are refused;
distinct invocations may have the same namespace PID. The worker still checks
its actual parent PID and the supervisor token, and both one-attempt markers
remain exclusive. The second attempt used a new directory. Its entire latent
ledger is byte-identical to the preserved first attempt, including every zero.
The original one-attempt marker was not removed or reset.

Only lifecycle and receipt functions changed in this repair. All scientific
function bodies are AST-identical to the original reviewed implementation.
The full [original implementation review](implementation-review.json) and
[additive namespace review](namespace-v2-implementation-review.json) remain
separate. The latter includes an independent reviewer-run test receipt.

## Validation and reproducibility

The initial complete synthetic gate passed 28 tests without loading the external
PES. It covered normalized mixture densities, rotational measure, analytic
four-channel and entry flux integrals, Cartesian metric and coarea derivatives,
unconditional zeros, pooled ratios, fixed paired draws, failure retention and
actual isolated-process resource limits.

The namespace revision passed eight focused lifecycle/resource tests, with 22
unchanged cases deselected. Two cases were new; the real resource fixture was
strengthened to use equal namespace PIDs. Together these receipts cover 30
distinct test cases across the two source revisions, not 36 distinct cases on
the final source. The independent reviewer separately reran three of the
namespace tests. No physics was evaluated by those test runs.

[replay.py](replay.py) imports only the standard library. It checks every saved
attempt/success pair, reconstructs all weights from recorded energies and the
pinned beta, retains all zeros, independently recomputes all channel/pooled/seed
statistics, and verifies runtime identities and hashes. It replayed all 641
successful evaluations without calling the PES. The largest absolute difference
over the saved scalar comparisons was `1.0658141036401503e-14`. All 157 guarded
source, review, model and validation files matched before and after the run.

The strict local replay command is:

```bash
python docs/validation/literature-convergence-2026-11-01/review/hch4-reference-pilot/replay.py \
  /workspace/scratch/a5e4820daedd/literature-convergence-hch4/reference-pilot-768-namespace-v2 \
  /path/to/new-replay-receipt.json
```

This command verifies the recorded absolute provenance paths, including the
external library hash; it therefore requires those exact inputs to remain
available. Saved JSON/JSONL data and script source are included in this bundle,
but the external PES library, its upstream coefficients, and the runtime are
not redistributed. [publication-inventory.json](publication-inventory.json)
lists the exact publishable files. Execution source snapshots distinguish all
historical and final implementation revisions.
