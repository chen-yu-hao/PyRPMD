# Isolated fixed 300,000-sweep continuation: scheduling tool

**Status: implemented and tested with inert workers; independent review pending; no restore or physical execution.** This new owner calls the frozen `quantum_sampling_campaign.one_seed` once for each of the same eight seeds. It changes process ownership and storage routing only. The earlier sampler, protocol, schedule, 30k/100k prefixes, model, analysis, RNG and reference values remain unchanged.

## Scientific decision and exact ancestry

The complete 100k result has primary log standard error **0.050873748737126245**, above the registered 0.03 limit. That precision failure is the sole continuation trigger. The original 300,000-sweep endpoint remains the final fixed endpoint; point agreement with a paper cannot trigger, stop or select a stage. The original design, cost forecast and implementation policy are in [the proposal directory](../umbrella-stage-300000-proposal/). Their creation-time descriptions remain historical, including the initial unimplemented status and superseded 10 GiB storage forecast.

The required prior archive is `pyrpmd-h3-quantum-umbrella-stage-100000-2026-11-01.tar.gz`, SHA256 `f02c799907993143533a570e1862bbb2772f9b8994344c50982b102b02d5603d`. Root's actual archive execution ended normally, and its exact replay was independently reviewed. This tool verifies those evidence identities and the 116-member whitelist, then creates a new independent byte copy. It never links to, changes permissions on, writes to or retires a lease in either the original campaign or the existing verified extraction.

The new root is `/workspace/scratch/a5e4820daedd/literature-convergence-quantum/umbrella-stage-300000-isolated-v1`. The active copy is `restored/data/campaign-run` beneath it. The valid prior snapshot is the archived **46,138-byte** `data/campaign-run/stage-100000.json`, SHA256 `c102f002d674e347020ecf5003b1a5e4de4bfc7b22d4691b28a5a768727ad317`. Its historical absolute `campaign_directory` remains byte-identical; the new route is registered separately. The compact archive `stage-snapshot.json` is not a scheduler input.

## Admission and ownership

A new root adoption must bind the exact source, current tests, successful inert gate and approving independent receipt. The non-executable [adoption template](root-adoption-template.json) deliberately contains unset resource/review fields and `adopted: false`. It is not launch authorization.

Admission verifies every restored checkpoint, geometry, saved RNG, settings file and previous prefix/state/receipt before any worker is launched. A permanent `RUN_ONCE.json`, exact registration and exclusive new-directory lease prevent automatic retries. Exactly two or four fixed CPU slots are accepted; there is no inherited control file or mid-run resizing. Each private worker requires the owning PID and per-invocation UUID token, a matching registration/plan, one fixed seed and CPU, and an exclusive per-seed claim. UUID generation does not consume the sampler's NumPy RNG.

The owner uses the unchanged, previously reviewed `p256_parent_campaign.dispatch_jobs`. That helper stops admissions on failure, defers interrupt/termination signals while draining, and retains its lease if draining cannot be confirmed. The new layer blocks the owner's timeout signal across `Popen` until the child has been recorded as owned, then restores it. A focused inert regression injects an alarm in exactly that interval. Source/event failure, stop requests and unconfirmed drain are also tested. An unexpected hard kill cannot guarantee a final receipt or cleanup; it leaves the persistent claims/lease for explicit inspection.

Before scientific imports, every worker installs a kernel **21,600 CPU-second** hard limit and a separate real-time alarm bounded by both six hours and the owner's fixed work deadline. The worker unblocks any inherited alarm mask. These caps remain effective if the owner disappears. The owner allows **43,200 seconds** whole wall time with a **120-second** drain/finalization reserve. Preparation has a 900-second wall alarm and a cooperative observed 300 CPU-second budget. The owner process CPU budget is a cooperative observed 600 seconds, rather than an inherited lower hard limit that would prevent children setting their six-hour cap. RSS and disk-space limits are observations, not kernel quotas. Timing/cleanup cannot guarantee interruption of an operating-system uninterruptible I/O state.

## Completion without partial analysis

Every seed must have 305,000 completed sweeps: 5,000 historical warmup plus all 300,000 production sweeps. The final CV array must be `(300000, 48)` and the diagnostic array `(30000, 48, 7)`. Both a final atomic checkpoint **and** its matching 300k state/receipt are required; a cutoff between their writes is incomplete. All old 30k and 100k prefix hashes and state bytes must still match.

Only the owner publishes one `stage-300000.json`, with all eight checked rows, their worker results, exact archive ancestry and confirmed worker drain. Its schema is `pyrpmd.h3.isolated300k-completion.v1`; it is intentionally distinct from the old scheduler's envelope. The separate root actual terminal observation and a future complete-cohort analysis admission are still required. This tool invokes no analyzer, computes no partial rate, produces no κ/full-rate estimate and starts no further refinement. All failed, incomplete and precision-failing outcomes must remain reviewable.

## Space and resource admission

The [phase-by-phase estimates](storage-stage-estimates.json) use only the completed archive member inventory and saved array descriptions; no raw NPZ was decoded to prepare the estimate.

| Incremental phase | Planning occupancy |
| --- | ---: |
| Exact fresh 100k restore | 474,508,661 bytes, observed member total |
| Full 300k working tree | 1.62 GB |
| Sampling with four atomic checkpoint temporaries | 2.50 GB |
| Completed working tree plus one new archive | 3.26 GB |
| Working tree, archive, fresh replay extraction and analysis/metadata | 5.00 GB |

The 5 GB is this task's incremental allowance, excluding existing evidence already deducted from current free space. The **4 GiB global free reserve is counted once**, then the root must add explicit remaining reservations for water and active P256 data/archive growth. No old evidence is deleted. At the root's 25,039,101,952-byte observation, at most 15,744,134,656 bytes remain for all other incremental reservations after the MC allocation and free reserve. Admission refuses an insufficient total.

A metadata-only watchdog observes filesystem free space and RSS for the owner and its own PID/start-time-bound workers. It requests stopping/draining if the remaining reserve plus checkpoint buffers, observed 3 GiB aggregate RSS, or owner CPU budget is exceeded. This is not a cross-task disk or memory quota. Future final archiving must remeasure final sizes and remain sequential: this forecast does not authorize an additional full temporary archive or extra duplicate extraction.

## Test scope and launch interface

The [inert gate](inert-gate/gate.json) records **62 passing cases, zero skips**, on one CPU, including exact-copy rejection, current-source review/gate binding, complete membership/state+receipt requirements, worker parent/token/resource guards, actual isolated kernel cap expiry, signal-safe child capture and failed-drain exclusion. No PES, primary raw checkpoint, actual cohort restore or scientific sampler was used. The protected scientific inputs remained unchanged.

After independent review and a separate root adoption, the root can execute the new module with the original `.venv` Python and `--authorization /absolute/path/to/root-adoption.json`, with all four thread environment variables set to `1`. The wrapper itself pins its selected CPUs. No execution is authorized by this README or its template.
