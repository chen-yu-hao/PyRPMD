# H + CH₄ proposal-efficiency implementation

This bundle implements the separately reviewed, excluded classical proposal-efficiency design. **No new CBE/PES2008 evaluation, molecular dynamics, actual pilot registration, or actual pilot execution has occurred for this implementation.** Its results so far are synthetic software and integration checks. They do not establish an efficient physical proposal or a reference reaction rate.

The unchanged scientific definition is [the frozen proposal](../hch4-reference-efficiency-design/proposal.json), SHA256 `8c9c576f37714cead7beb6a788a1262f88b7a1f34914029f041643774b78d962`. Its [independent mathematical review](../hch4-reference-efficiency-design/independent-method-review.json) remains a design review. This new bundle records implementation testing and separate implementation review; it does not replace that earlier inventory or the unsuccessful first proposal pilot.

## Implementation and fixed accounting

The new isolated module is `benchmarks/hch4/efficiency_pilot.py`, with tests in `tests/test_hch4_efficiency_pilot.py`. It imports the original coarea and geometry helper read-only and pins its exact SHA256. It does not change the CBE adapter, potential library, active H₃ campaigns, water campaign, or production dynamics.

| Stage | Fixed maximum | Required ordering |
| --- | ---: | --- |
| Initial energy-only Hessian calibration | 711 geometries | Write the entire deterministic ledger before its first evaluation. |
| Shifted centers, unit probes and reflections | 89 geometries | Construct from the first stage and write its entire ledger before evaluation. |
| Main independent diagnostic cohort | 3,072 geometries | Freeze the calibrated proposal before drawing any of the eight new seed vectors. |
| Matched entry-separation control | 128 geometries | Reevaluate the same preselected 16 entry draws per seed at 60 bohr. |
| **Total attempted PES ceiling** | **4,000 geometries** | Count training and failed attempts; never replace a draw or increase the ceiling. |

The main cohort preserves the original finite target and its four abstraction charts. The proposal combines a positive uniform support floor, both reflected local Gaussian families and Student distributions. Every importance denominator uses the complete mixture density with the stereographic area Jacobian. The broader outer domain is diagnostic; points outside the original target retain zero weight in its unconditional integral.

The two surface locations, 0.999 and 1.007, are both classical P1 diagnostics. The latter is not a quantum ensemble. The implementation sums four channel means for each surface and uses one entry-chart mean. Its central flux-ratio diagnostic is the ratio of pooled integral means. The eight independently seeded vectors, including all paired observations within each seed, are the uncertainty units. The reported delta-method Student interval is nominal and conditional; it is neither a tail guarantee nor a physical-rate interval.

The entry-separation control compares only the same 16 selected draws at 30 and 60 bohr after removing each separation's squared radial factor. The main entry denominator retains all 128 draws per seed. This control cannot establish convergence of the full finite-separation reaction-rate normalization.

## Resource admission and failure retention

Actual use requires a new explicit root adoption, exact implementation-review and test receipts, the pinned model/build identities, runtime and source inventories, and a separate workspace output directory. This bundle supplies no adopted execution plan. Receipt hashes alone are insufficient: the review must explicitly approve the current source, tests and frozen design without remaining blockers, and the gate must report successful unchanged-source testing with positive case counts and zero failures, errors or skips. The transitive package inventory also refuses to replace any protected design hash with newly observed bytes.

Three distinct interpreter invocations have fixed whole-process CPU ceilings: preparation 2 seconds, supervision 3 seconds and the private worker 55 seconds. Their nominal OS-enforced ceilings sum to 60 seconds. CPU time already used by interpreter startup and imports is included. The worker stops admitting new evaluations at 55 combined CPU seconds after charging both prior ceilings in full. The outer supervisor's wall ceiling is 120 seconds; the private worker has its own remaining wall timer. As with OS resource limits generally, enforcement is subject to kernel timer/accounting granularity.

The supervisor preserves a persistent one-attempt claim and owns one separately grouped worker process. Handled interrupt or termination signals stop further work and trigger termination/reaping of that owned worker, with a bounded forced-kill fallback. This cleanup is not described as natural completion of the physical pilot. A hard process kill can prevent a final receipt; persistent claims and the worker's own resource limits remain, and automatic recovery is forbidden.

Before each actual PES call, the worker durably appends its attempt identity. It retains successful energies, failures, raw ledgers and completed-prefix counts. An invalid calibration, nonfinite energy, exhausted resource admission, source change or incomplete cohort withholds aggregate diagnostics. It cannot silently refit, clip a Hessian, substitute a proposal, discard a failed draw, retry, resume or extend the budget. Any later repair or adoption would require separate preserved evidence and authorization.

## Synthetic validation

The final [85-case gate](admission-guard-gate/gate.json) passed with no skips in 10.82 seconds on one allowed CPU and one library thread. The [JUnit result](admission-guard-gate/pytest.xml) and [test log](admission-guard-gate/pytest.log) bind the executed source and tests. They exercise analytic densities and integrals, exact cohort accounting, and inert process fixtures. Synthetic random draws and synthetic encoded coordinates are used; no external PES is loaded or called.

The checks include:

- Gaussian and Student density oracles, numerical sphere-area integration, full-mixture stratified integration, positive support and reflection symmetry.
- Four disjoint active-channel surfaces versus a single entry normalization, unconditional geometric zeros and unresolved zero integrals.
- Exact quadratic Hessians, scaled-coordinate covariance and eigenaxis back-transforms, and rejection of indefinite, ill-conditioned, inconsistent or excessive-displacement calibrations.
- The complete 800 + 3,072 + 128 synthetic pipeline, ledger-before-evaluation ordering, fixed eight-seed membership, and correct partial-stage retention.
- Ratio-of-pooled-means uncertainty and a nonconstant-weight check proving that the separation diagnostic uses the matched 16 draws rather than all 128 entry draws.
- Resource admission before scientific imports, namespace-safe invocation identity, fixed source/model/runtime/adoption guards, and independent CPU-cap inheritance.
- Actual OS SIGINT and SIGTERM delivered to an isolated supervisor with a bounded inert child, plus the TERM-ignoring child case that requires forced kill and reaping.
- Semantically invalid review or gate receipts whose file hashes and adopted inventories have been deliberately updated, proving that rejection depends on their contents rather than a stale byte hash.
- A changed protected package file in an isolated fake project, proving that a refreshed transitive inventory cannot silently replace its frozen design identity.

The earlier [30-case mathematical gate](initial-math/source-snapshot.json), [63-case lifecycle gate](initial-lifecycle-cpu5/gate.json), [71-case signal and matched-control gate](signal-and-matched-control-gate/gate.json), and their exact source snapshots remain available. The first lifecycle launch failed before pytest because the requested CPU 9 was outside the available mask; its [zero-test launch receipt](initial-lifecycle/gate.json) and [error log](initial-lifecycle/pytest.log) are preserved. The following gate used an observed allowed CPU and changed no scientific inputs or expectations.

The independent lifecycle review then identified the receipt-semantics and protected-hash issues described above. The [exact admission-only patch](admission-guard-only.patch) and [supersession proof](admission-guard-supersession.json) preserve that history. Two existing admission functions changed, one receipt-validation function was added, and a `TESTS` path constant was added to bind the test-file identity. All 25 independently reviewed scientific function ASTs, and all nine protected design source identities, are unchanged.

## Future execution contract

The new command-line interface accepts a read-only `preflight`, a dedicated `prepare` invocation, and a distinct `run` supervisor invocation. The private `_worker` requires the supervisor's exact process/token/deadline identity and an exclusive worker claim; it is not a separate public sampling route.

A future adopted plan must bind the following concrete inputs before `prepare` can create its new directory:

- The final script/test identities, frozen design and method-review identities, complete protected/transitive Python source inventory, fixed CBE library and its existing build receipt.
- The observed interpreter executable/hash, Python/NumPy/SciPy versions and platform; one actually available CPU; and a separate resource-adoption receipt explicitly authorizing at most 4,000 geometries on one CPU.
- A successful exact-source synthetic gate and an implementation-review receipt with `approved_for_single_excluded_efficiency_pilot: true`, `remaining_blockers: []`, and exact `design_sha256`, `source_sha256` and `tests_sha256` fields. This review decision is separate from root execution adoption.
- The unchanged eight seeds, staged counts, resource ceilings, output-field set and false automatic-retry/automatic-extension flags.

`prepare` creates the immutable registration and preparation claim without scientific RNG draws or PES loading. `run` rechecks the adoption, receipts, sources and runtime, creates the permanent one-attempt claim, and starts the single private worker. A failed attempt must remain in place; this interface offers no resume or retry operation.

## Scientific limits and adoption status

The real energy-only Hessians have not been evaluated. They may fail the fixed positive-definiteness, finite-difference, displacement, probe or reflection conditions. Even a successful calibration and high empirical effective sample size would not prove negligible tails, correct physical-domain truncation or an absolute literature-normalized rate. All such failures and limitations must remain visible.

Independent implementation review is complete. The [mathematical review](independent-method-review.json) and [narrow final-source identity update](independent-method-identity-update.json) found no method blocker. The separate [lifecycle review](independent-lifecycle-review.json) found no remaining blocker after the two admission-guard corrections, and explicitly retains hard-kill and missing-final-receipt limitations.

These are implementation-review decisions for the exact excluded design. No root execution adoption, actual registration, or new physical sampling is implied by this README or the passed synthetic checks. The report schema intentionally contains no converted physical rate or paper-agreement flag.
