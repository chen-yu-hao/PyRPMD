# Isolated fixed 300,000-sweep continuation: scheduling tool

**Status: corrected scheduling/resource tool independently approved for separate root adoption; 76 inert cases pass; no restore or physical execution has occurred at this freeze.** This new owner calls the frozen `quantum_sampling_campaign.one_seed` once for each of the same eight seeds. It changes process ownership and storage routing only. The earlier sampler, protocol, schedule, 30k/100k prefixes, model, analysis, RNG and reference values remain unchanged.

## Scientific decision and exact ancestry

The complete 100k result has primary log standard error **0.050873748737126245**, above the registered 0.03 limit. That precision failure is the sole continuation trigger. The original 300,000-sweep endpoint remains the final fixed endpoint; point agreement with a paper cannot trigger, stop or select a stage. The original design, cost forecast and implementation policy are in [the proposal directory](../umbrella-stage-300000-proposal/). Their creation-time descriptions remain historical, including the initial unimplemented status and superseded 10 GiB storage forecast.

The required prior archive is `pyrpmd-h3-quantum-umbrella-stage-100000-2026-11-01.tar.gz`, SHA256 `f02c799907993143533a570e1862bbb2772f9b8994344c50982b102b02d5603d`. Root's actual archive execution ended normally, and its exact replay was independently reviewed. This tool verifies those evidence identities and the 116-member whitelist, then creates a new independent byte copy. It never links to, changes permissions on, writes to or retires a lease in either the original campaign or the existing verified extraction.

The new root is `/workspace/scratch/a5e4820daedd/literature-convergence-quantum/umbrella-stage-300000-isolated-v1`. The active copy is `restored/data/campaign-run` beneath it. The valid prior snapshot is the archived **46,138-byte** `data/campaign-run/stage-100000.json`, SHA256 `c102f002d674e347020ecf5003b1a5e4de4bfc7b22d4691b28a5a768727ad317`. Its historical absolute `campaign_directory` remains byte-identical; the new route is registered separately. The compact archive `stage-snapshot.json` is not a scheduler input.

## Admission and ownership

A new root adoption must bind the exact source, current tests, successful inert gate and approving independent receipt. The non-executable [adoption template](root-adoption-template.json) deliberately contains unset resource/review fields and `adopted: false`. It is not launch authorization.

Admission verifies every restored checkpoint, geometry, saved RNG, settings file and previous prefix/state/receipt before any worker is launched. A permanent `RUN_ONCE.json`, exact registration and exclusive new-directory lease prevent automatic retries. Exactly two or four fixed CPU slots are accepted; there is no inherited control file or mid-run resizing. Each private worker requires the owning PID and per-invocation UUID token, a matching registration/plan, one fixed seed and CPU, and an exclusive per-seed claim. UUID generation does not consume the sampler's NumPy RNG.

The owner uses the unchanged, previously reviewed `p256_parent_campaign.dispatch_jobs`. That helper stops admissions on failure, defers interrupt/termination signals while draining, and retains its lease if draining cannot be confirmed. The new layer blocks the owner's timeout signal across `Popen` until the child has been recorded as owned with a kernel pidfd, then restores it. It also independently confirms/reaps every captured child under alarm masking and deferred stop signals if an alarm interrupts the frozen helper's exceptional drain. Only those captured kernel identities can be signaled; recycled PIDs are never used to choose a target. Focused inert regressions inject alarms both between `Popen` and capture, and during exceptional draining. Source/event failure, stop requests and truly unconfirmed drain are also tested. An unexpected hard kill cannot guarantee a final receipt or cleanup; it leaves the persistent claims/lease for explicit inspection.

Before scientific imports, the owner and every worker install a **1 GiB `RLIMIT_AS` hard cap**. This is virtual address space, not a claim of measured RSS. Four workers plus the owner therefore have a sum of five GiB in independently enforced address-space caps. Every worker also installs a kernel **21,600 CPU-second** hard limit and a separate real-time alarm bounded by both six hours and the owner's fixed work deadline. The worker unblocks any inherited alarm mask. These caps remain effective if the owner disappears. The owner allows **43,200 seconds** whole wall time with a **120-second** drain/finalization reserve. Preparation has a 900-second wall alarm and a cooperative observed 300 CPU-second budget. The owner process CPU budget is a cooperative observed 600 seconds, rather than an inherited lower hard limit that would prevent children setting their six-hour cap. The shared-memory admission adds root-bound water, P256 and diagnostic/runtime reservations to the MC address-space-cap sum and refuses totals above eight GiB. Each external entry states whether it is an enforced cap or a planning estimate. The aggregate three-GiB RSS screen and disk-space screen remain observed planning rules, not kernel quotas. Timing/cleanup cannot guarantee interruption of an operating-system uninterruptible I/O state.

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

A metadata-only watchdog observes filesystem free space and positive **self-reported peak RSS** for the owner and each exact owned worker. It uses `getrusage(RUSAGE_SELF)` and hash/token/PID-bound worker heartbeats; it does not access `/proc`. A subsequent worker cannot be admitted until the preceding worker has written its first valid positive resource report. Missing, stale or invalid live-worker observations stop admission/draining; an unknown value is recorded as `null`, never zero. Only a pidfd-confirmed exited child may be excluded. It requests stopping/draining if the reserve plus checkpoint buffers, observed sum of peak RSS above three GiB, or owner CPU budget is exceeded. This is not a cross-task disk or RSS quota. Future final archiving must remeasure final sizes and remain sequential: this forecast does not authorize an additional full temporary archive or extra duplicate extraction.

## Test scope and launch interface

The [corrected inert gate](inert-gate-final/gate.json) records **76 passing cases, zero skips**, on one CPU, including exact-copy rejection, current-source review/gate binding, complete membership/state+receipt requirements, worker parent/token/resource guards, actual isolated CPU/wall/address-space cap enforcement, positive self-resource reporting, unknown/stale observation rejection, signal-safe child capture, alarm-during-drain recovery and failed-drain exclusion. No PES, primary raw checkpoint, actual cohort restore or scientific sampler was used. The protected scientific inputs remained unchanged.

After independent review and a separate root adoption, the root can execute the new module with the original `.venv` Python and `--authorization /absolute/path/to/root-adoption.json`, with all four thread environment variables set to `1`. The wrapper itself pins its selected CPUs. No execution is authorized by this README or its template.

## Preserved initial findings and resource correction

The initial source `403300a6...`, its 62-case gate, old policy and the independent nonapproval findings remain under `initial-candidate/`, `inert-gate/`, `initial-review/` and the original proposal files. The two concrete issues were interruption of exceptional draining by an alarm and treating unavailable `/proc` RSS as zero. Neither candidate was used for a physical continuation.

The owned-only [observability probe](owned-resource-observability-probe.json) found that both this probe process and its directly spawned inert child had no readable `/proc/<pid>/stat`; pidfd liveness and self `getrusage` worked. The separate [excluded memory-layout/import probe](synthetic-AS-layout-probe.json) installed one-GiB `RLIMIT_AS` before imports, then allocated 196,134,912 bytes of synthetic arrays and a 201,000,000-byte checkpoint surrogate and exercised the largest 115.2-MB hash copy. It completed in 0.917 seconds wall / 0.866 CPU, with 569,126,912-byte measured peak RSS. This supports the proposed cap for the relevant layout; it is not a measurement of a real long sampling run. No PES, primary raw arrays, random draws or scientific propagation were used.

The current [v2 scheduling policy](../umbrella-stage-300000-proposal/implementation-policy-v2.json) supersedes only process/resource handling. The historical 15.3-GB water forecast in its ancestry does **not** enter the actual admission sum: the root must provide the current remaining water and P256 reservations explicitly. The new full-checkpoint, RNG/prefix and fixed scientific endpoint requirements remain unchanged.

## Final review identity

Root adoption must bind [independent-review-final.json](independent-review-final.json), SHA256 `776f9520a6db6212790cd5c7b35f559321b9aebca1754cfb6a6fe15c06a789c0`, and the exact [final 76-case gate](inert-gate-final/gate.json), SHA256 `7d3359bbce96e1988e7a7b943485fc5d530f7f3f5836d8792c2b9569ed81bbbb`. The source SHA is `a7ec140dbe953a3e068cb491b24f0769e395cd4d2c449e9d2daf20aa45b4a2b7`; tests SHA is `69df0a97c06992db5c24e074abee9866474ae4f02a90b8e9f6fe73d1d073ef05`; v2 policy SHA is `a8cbef299f4bda18fbdab495fdabcae26982ea08aca25aca257f83866366c5da`.

The earlier same-source review bound the pre-portability test file, so its original receipt and gate are preserved. The sole final test change removes an unused Unix-only import before the Linux skip guard; no Windows execution or scientific behavior change is claimed. All final declared original scientific inputs were checked unchanged.
