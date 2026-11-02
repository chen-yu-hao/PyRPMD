# Root review guide: one excluded HCH₄ efficiency pilot

**Procedure proposal only. No execution plan, resource adoption, registration or sampling has been created.** The frozen implementation inventory is `83d73d906ff8e345ee39761029ad36b889426c7c2c01b7b2ae7781718040511f`; all 36 members were rehashed and still match. Full input paths, byte sizes, schemas and SHA256 values are in [input-identities.json](input-identities.json). Nothing in the implementation bundle has changed.

## 1. Adopt the exact inputs before calling `prepare`

There is no `adopt` CLI command. Root must first review the concrete available CPU and create two new immutable documents outside this guide: a resource-adoption receipt and the execution plan. `prepare` requires an already adopted plan; it cannot precede adoption.

The execution-plan schema is `pyrpmd.hch4-efficiency-execution-plan.v1`. Its required contents are defined by `load_plan` in the frozen script:

| Input | Exact contract |
| --- | --- |
| Approval | `parent_adoption: true`; fixed design and design-method-review SHA256 values. |
| Cohort | Seeds 2026112201 through 2026112208, in that order. |
| Counts | Initial calibration 711; follow-up 89; main 3,072; control 128; total maximum 4,000. |
| Resources | One CPU; preparation/supervisor/worker CPU caps 2/3/55 seconds; sum 60; combined soft admission 55; outer supervisor wall 120 seconds. |
| Runtime | Exact `environment_identity()` result: resolved interpreter/hash, Python/NumPy/SciPy versions and platform; one available integer `cpu_affinity`. |
| File inventory | Exact `required_sources(design)` map, plus the library, build receipt, synthetic gate, implementation review and resource-adoption receipt. Do not replace protected design hashes with current hashes. |
| File identities | Each bound library/receipt uses `{path, bytes, sha256}`. Library path is the pinned `libcbe2008.so`; build receipt must be its sibling `libcbe2008.build.json`. |
| Output | A new canonical absolute directory beneath `/workspace/scratch/a5e4820daedd`, outside the original PyRPMD checkout; it must not exist when `prepare` starts. |
| Other fixed values | Exact `OUTPUT_FIELDS`; `automatic_retry: false`; `automatic_extension: false`. |

The resource-adoption receipt must explicitly contain `authorize_single_bounded_pilot: true`, `available_CPU_count: 1`, and `maximum_PES_geometries: 4000`. The current validator does **not** require a particular schema string for that receipt; its exact bytes and these fields are enforced. Record the actual resource decision and timestamp rather than treating the guide as that decision.

Use these existing receipts in the future plan:

| Plan field or supporting input | Schema / SHA256 |
| --- | --- |
| `synthetic_gate` | `pyrpmd.hch4.efficiency-pilot.synthetic-gate.v1`; `7d084215e7cb4b83cfda6f2c0b8e44d227107c529bcbfcbba03eec5f7c77ed6a` |
| `implementation_review` | `pyrpmd.hch4-efficiency-independent-lifecycle-review.v1`; `6fcbaa6f5ddaf4cdebe7989408dc8e6abc16901e122a93d8a14562358e087a07` |
| Final mathematical identity review | `pyrpmd.hch4-efficiency.method-review-source-identity-update.v1`; `c79ddaea13e71c4a2bfea3ac48621a7f8ee7cb959a763a1bf3c461bc47f4cf6c` |
| Design | `8c9c576f37714cead7beb6a788a1262f88b7a1f34914029f041643774b78d962` |
| Design-method review | `694590d63fb8d153bef43286e90bcd218271d0977e9372f6849d8d70f3247ff7` |
| Script / tests | `8630f0ac4106d826beb4ccb471eb57ab0382443cd9a6bca1fb474400877d91fb` / `3f32570efcef28733db3ec961450cafdf836836bc213d1794d50436c24924dc6` |
| Library / build receipt | `aa33e4f55246d5b99839cd3b08702f1d091a6b3dddac1d6c97d34bb542d95aaf` / `e0f083d451df13d8fa46ead08f0859e2ea852b1bb7fa58fb1d50e60e1af2c84c` |

The implementation-review receipt explicitly approves the current source/test/design with no remaining blockers. The gate contains 85 successful distinct synthetic cases with unchanged source and no errors, failures or skips. Both their semantics and hashes are checked again during preparation and execution.

## 2. Exact future command sequence

These commands are templates for **after** the separate adoption. Replace the three angle-bracket values from that adopted plan. Use fresh interpreter invocations; do not call the functions from a long-lived pytest/notebook process. Each command inherits one CPU and one library thread. Retain the actual tool/shell exit result as well as output files.

```bash
hch4_root=/workspace/scratch/a5e4820daedd/PyRPMD
hch4_python="$hch4_root/.venv/bin/python"
hch4_plan='<absolute path to the adopted execution plan>'
hch4_output='<the adopted new output directory>'
hch4_cpu='<the adopted integer CPU>'

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$hch4_root/src:$hch4_root" \
  taskset -c "$hch4_cpu" "$hch4_python" "$hch4_root/benchmarks/hch4/efficiency_pilot.py" \
  prepare --plan "$hch4_plan" --output "$hch4_output"
```

Only after that invocation exits successfully and its exact registration/claims are retained, invoke the supervisor once in a separate interpreter:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$hch4_root/src:$hch4_root" \
  taskset -c "$hch4_cpu" "$hch4_python" "$hch4_root/benchmarks/hch4/efficiency_pilot.py" \
  run --registration "$hch4_output/registration.json"
```

Do not invoke `_worker` directly. It requires the exact owning supervisor, UUID token, remaining wall deadline and exclusive worker claim.

The optional read-only CLI form is `preflight --plan "$hch4_plan"`, with the same interpreter/environment prefix. It is not required: `prepare` repeats that validation inside its capped process. A standalone `preflight` is an uncapped metadata operation and is outside the three-invocation execution CPU accounting; omit it from the counted execution sequence rather than claiming its overhead is included.

## 3. Resource accounting and retained outputs

| Process | Accounting and stop behavior |
| --- | --- |
| Preparation | Whole-process hard CPU cap 2 seconds, including startup, hashing and imports; no scientific RNG or PES loading. Its own wall alarm is 120 seconds. |
| Supervisor | Whole-process CPU cap 3 seconds, including preflight and finalization; soft limit applies before spawn and the hard limit is lowered after the separately capped child inherits its allocation. |
| Worker | Whole-process hard CPU cap 55 seconds, installed before scientific imports. New evaluations stop when worker CPU plus the full prior 5-second allowance reaches 55 seconds. |
| Wall | The registered **outer supervisor** interval is 120 seconds from its entry. Worker uses the same absolute deadline. New work stops two seconds before it; owned-child cleanup uses bounded TERM/reap and KILL fallback. This is not a single combined prepare-plus-run wall interval. |

Nominal CPU ceilings sum to 60 seconds, subject to ordinary OS accounting/timer granularity. A hard kill can leave a partial file or omit a final receipt; the permanent claims remain and do not authorize recovery.

Preparation creates `PREPARATION_ONCE.json` and `registration.json` (`pyrpmd.hch4-efficiency-registration.v1`). Execution adds `RUN_ONCE.json`, `WORKER_ONCE.json`, `child-started.json`, `worker.log` and `runtime.json`. Each reachable scientific stage then writes:

1. `calibration-initial-ledger.json` before its 711 evaluations.
2. `calibration-followup-ledger.json` before its 89 evaluations.
3. `frozen-proposal.json` before generating the fresh diagnostic draws.
4. `diagnostic-ledger.json` before the 3,072 main and 128 control evaluations.
5. `calls.jsonl` with durable attempt/success/failure entries, and complete-stage `main-results.json` / `control-results.json` when available.
6. `report.json` (`pyrpmd.hch4-reference-efficiency-pilot.v1`) and `supervisor.json` (`pyrpmd.hch4-efficiency-supervisor.v1`) when finalization is possible.

Root should retain actual normal exit evidence and rehash the adopted inputs and produced files. Completion requires supervisor exit zero, confirmed owned-worker drain, unchanged source/ledger/proposal identities, 800 completed calibration evaluations, all 3,072 main and 128 control draws, and at most 4,000 attempted PES geometries. Geometric zeros remain in the unconditional means and can reduce actual calls. A completed pilot may still fail every efficiency screen; exit zero does not establish a useful physical reference.

## 4. Scientific stop rules

Stop and retain the attempt if any original-domain calibration point is invalid; a PES status/energy or weight is nonfinite; the Hessian has minimum eigenvalue ≤10⁻⁶ or condition number >10⁸; diagonal refinement discrepancy exceeds 0.05; Newton Mahalanobis displacement exceeds 3; a unit-probe residual exceeds 5; reflected energy differs by >10⁻¹⁰ hartree or pair distances by >10⁻¹² bohr; or a resource/source/identity guard fails. There is no Hessian clipping, optimizer fallback, refit, replacement draw, retry or automatic extension.

After a complete cohort, preserve all fixed efficiency, mirror, component, nested-domain and matched-entry-control diagnostics, including failures or unresolved zero integrals. The eight seed vectors remain the uncertainty units. Both surface locations are classical P1; no quantum or literature-normalized rate is inferred. Passing empirical efficiency screens does not establish tail, finite-domain or full finite-separation accuracy. Never delete claims, choose favorable seeds, enlarge the budget, or alter a stopping rule in response to a numerical result.
