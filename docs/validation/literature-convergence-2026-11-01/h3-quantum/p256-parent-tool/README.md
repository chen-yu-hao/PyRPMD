# Reviewed execution tool for the fixed eight P=256 parent chains

This directory records the scheduling and integrity checks for the [fixed eight-chain design](../p256-fixed-eight-chain-design/README.md). The tool is `benchmarks/h3/p256_parent_campaign.py`; its standard test module is `tests/test_h3_p256_parent_campaign.py`. The scientific producer remains the frozen `benchmarks.h3.surface_partition.run`. No production parent simulation or recrossing trajectory was launched to prepare this package.

The final synthetic gate passed 56 tests in 3.95 seconds, exercising the existing producer with a small analytic P=4 harmonic-bond model and inert subprocesses. It does not evaluate BKMP2. The separate read-only preflight checks 39 source, runtime, model and historical-state guards against the actual workspace. It does not construct the external potential or create the production output root. The exact commands, hashes, CPU measurements and test counts are in `final/synthetic-gate.json` and `final/preflight-receipt.json`. The first gate and its source snapshots remain preserved separately; only the final gate binds the current driver.

## Fixed cohort and output identity

The campaign admits seeds in this order:

| Role | Seeds | Required result |
|---|---|---|
| Historical replay | 825601, 825603, 825607, 825609 | Bitwise equality of all 16 old non-observation checkpoint arrays |
| New independent chains | 825613, 825617, 825621, 825623 | The same fixed scientific settings, with distinct predeclared seeds |

Every chain has 2,000 warmup sweeps, 20,000 production sweeps, nine coupling states, and exactly 200 TS observations at completed sweeps 2,100, 2,200, …, 22,000. All eight complete chains are required. The four historical chains are not an interim stopping option.

The only output root permitted by the frozen design is:

```
/workspace/scratch/a5e4820daedd/literature-convergence-quantum/p256-fixed-eight-chain-v1/parents
```

Under each `seed-<seed>` directory, the unmodified producer writes `settings.json`, `report.json`, `reference.json`, `reference-domain.json` and `checkpoint.npz` (and its usual progress files). The global `parents-complete.json` is published once only if all eight identities, geometries, counts and predeclared sampling flags pass. It is never overwritten. The exact consumer contract is in `parent-output-contract.json`.

For the historical chains, only the settings fingerprint and two new endpoint observation arrays may differ. The old and new versions of `domain_sha256`, `radius`, `ratio`, `eta_R`, `eta_r`, `completed`, `delta_energy`, `diagnostics`, `accepted`, `proposed`, `swap_attempts`, `swap_accepts`, `lineage`, `last_extreme`, `endpoint_traversals` and `rng` must match in dtype, shape and all bytes. Signed-zero differences count as differences. Both reference JSON files must also match their original bytes. Scientific report values must match, excluding only elapsed time, creation time, whole-checkpoint hash and the two endpoint observation fields.

Each new endpoint array has shape `(200, 256, 3, 3)`, in bohr, with beadwise zero total COM and the same TS chart as the original producer. The check verifies the centroid surface relation and domain, finite coordinates, and all bead-pair cutoffs. These are static flux-weighted configurations, not trajectories or saved Maxwell momenta. No claim of independent parent configurations is made: the independent uncertainty units are the eight equilibrium chains.

## Authorization and resource record

The CLI requires a separately written, reviewed execution authorization. `execution-authorization.schema.json` describes its JSON shape; the program additionally verifies the whole bytes of every referenced receipt and source. A schema-valid object alone does not authorize execution.

The authorization must bind the frozen design SHA, the exact new driver SHA, the passing synthetic gate, an independent source review, and root's resource decision. The resource decision must explicitly confirm one or two available worker CPUs, a total of at most eight project worker slots, and actual queue-drain evidence. A prediction of when water or child workers might finish is insufficient. Each worker is pinned to one named CPU and all native/BLAS thread variables are set to one. This check does not inspect or terminate jobs in any other workspace.

The whole campaign has a 7,200-second wall ceiling, including setup and final verification. Worker admission and work stop 15 seconds before that ceiling to reserve time for draining and final receipts. The coordinator sends SIGTERM and, if necessary, SIGKILL only to its own process groups, then reaps them. Time and CPU usage remain observed quantities: the receipts separately record whole worker process user/system CPU, worker call wall time, peak RSS, aggregate child CPU, and coordinator CPU. If operating-system or filesystem cleanup cannot be confirmed, the campaign remains failed/incomplete and the exclusive lease is retained; it cannot publish eligible parents.

An exclusive sibling directory lock prevents another invocation of this coordinator. SIGINT/SIGTERM deferral remains installed through worker draining, including after a journal error. No stale lock is removed automatically. The unchanged legacy producer can be called directly and does not know this lock, so no other dispatcher or direct producer may write these output directories concurrently.

Each invocation creates a new `attempts/<uuid>/` containing the immutable plan, journal, per-seed logs, worker claims, execution receipts and a result. Any failed or incomplete attempt remains present. The driver stops new admissions on failure and drains all admitted workers; it does not replace a seed. Completed producer files are verified and reused without recompressing or rerunning them on an explicitly authorized `--resume`. An interrupted producer resumes through its existing checkpoint/RNG path. A resume uses the same source, settings, runtime and output root, plus a separately reviewed resource authorization; it does not automatically extend a budget.

## Commands

From the original repository, the read-only check is:

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python benchmarks/h3/p256_parent_campaign.py preflight
```

After the independent review and root's actual resource decision, root may supply a concrete authorization file:

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python benchmarks/h3/p256_parent_campaign.py run \
  --authorization /absolute/path/to/the-reviewed-execution-authorization.json
```

The tool has no child-launch function. The private `_worker` entrypoint requires the owning coordinator and an exclusive per-attempt worker claim; it is not an alternative launch interface. This fixed replay is Linux-specific because the recorded runtime, CPU affinity and process-group controls are part of its identity. The test module skips before importing these controls on unsupported platforms.

## Gates and interpretation

The synthetic tests check exact observer/RNG neutrality, interrupted replay, valid-envelope payload tampering, domain-byte changes, complete-state reuse, geometric checks, all-eight completion, fixed admission order, two-worker maximum, source guards, CPU/resource authorization, exclusive leases, deadlines, process failure and repeated signals during draining. `before-finalization-reserve/` preserves the first 56-test source and receipts before the final scheduling-only 15-second cleanup reserve was added; no physics or random draw changed.

Per-chain sampling flags are the frozen design's minimum BAR overlap 0.2, minimum diagnostic energy ESS 200, and absolute half-production log-partition difference at most 0.2. A complete chain with a failed flag remains recorded as complete but ineligible. The eligible manifest requires every chain to pass, and does not state that dynamical bead convergence has been demonstrated.

Only a separately reviewed and authorized child campaign can measure the P=256 transmission coefficient. That later experiment must compare all eight independent P=256 chains against the complete eight P=128 chains, without inventing cross-bead pairing, using the already registered full-rate log-equivalence and precision criteria. Parent availability or an equilibrium P=128→256 comparison cannot substitute for that dynamic comparison.
