# Four-worker scheduling handoff

This directory records a scheduling change to the existing eight-seed classical water experiment. The physical experiment remains the original [protocol](../protocol.json), SHA256 `288059c14c1b0467c322806fcf41765b69df0ed50af436941e7f0cad3fbcf631`: seeds 1101–1108, 100 ps preparation and 25 fixed 4 ps NVE segments per seed, 0.25 fs integration, the same full PME model, thermostat procedure, random streams and observation/checkpoint boundaries. The trajectories remain an **external OpenMM reference**, not PyRPMD production trajectories.

The new [coordinator](../../../../../benchmarks/water/native_campaign_schedule_v2.py) runs these fixed queues:

| Queue | Fixed seed order |
|---|---|
| a | 1101, 1105 |
| b | 1102, 1106 |
| c | 1103, 1107 |
| d | 1104, 1108 |

Only seeds 1101 and 1102 had started before this handoff. Their committed histories are retained. The remaining six seeds are reassigned before their first invocation, without changing any input or seed. All four queues must finish; a result or diffusion estimate never determines queue order, stopping or seed replacement.

## Exclusion and graceful stopping

One coordinator holds its own exclusive lock, both original odd/even queue locks, and a separate lease for each seed. Every native child inherits the coordinator lock, both original queue locks and its seed lease. If the coordinator unexpectedly exits, those inherited locks remain held until the corresponding native children exit. A second coordinator or an old odd/even scheduler therefore cannot acquire the required leases during that interval.

Before every invocation, the coordinator checks the registered native source, its own source, the original protocol and checkpoint-contract revision, all seed configurations, and the initial checkpoint checksum. It validates serialized engine identities and every committed chunk. Both the migration prefix and the prefix immediately before each invocation must remain unchanged. The registration and the handoff receipt are separately hashed.

Any child failure, provenance failure or changed historical prefix stops new dispatch in every queue. Already-running native calls finish normally. A durable `four-worker-control/FAILURE.json` prevents an automatic restart after an error. The source does not call `terminate()` or `kill()` on an integration child. Post-spawn exception cleanup waits for the child even if writing its start event fails.

An operator can request a scheduling-only stop by creating `four-worker-control/STOP_AFTER_CURRENT`. The coordinator then starts no further native calls and waits for the existing calls to finish. Remove that request only after reviewing the completed invocation. Each coordinator invocation has at most 7200 seconds of scheduling budget; each native call receives at most 1800 seconds. Native checkpoint completion can extend the wall time beyond its nominal limit. `--workers 2` retains the same registered queue partition when resources require a smaller concurrency limit; `--workers 4` is the authorized maximum.

All launches of this campaign must use the coordinator. The frozen standalone native driver predates these leases and does not itself acquire them. The migration receipt explicitly verifies the earlier owned native processes before handing over their output directories.

## Tests and source identity

The [test gate](test-gate.json) and [JUnit record](pytest.xml) report **19 passing, self-contained orchestration tests**. They do not load historical raw checkpoints or run molecular dynamics. The cases include actual inherited-file-descriptor lock survival, original/global/seed lock exclusion, complete fixed queue coverage with two and four workers, immutable historical prefixes, source/input/receipt corruption, durable global failure handling, post-spawn journal failure, and normal completion of other in-flight calls after one failure.

An internal parallel code review identified and corrected the post-spawn cleanup gap, the failure-latch race at dispatch, and incomplete old-queue receipt validation before the source was frozen. This review is an engineering check, not an independent physical benchmark.

| File | SHA256 |
|---|---|
| `benchmarks/water/native_campaign_schedule_v2.py` | `af1258f9118c4de296c90edb18c57aceb07a007d564652b4f5db6e851cece75e` |
| `tests/test_water_native_schedule_v2.py` | `5611446ccb7d9b2ea520dd04c237c723b37f47cd1106c270a9effebe6ddfc11c` |

The original 21 frozen water code/test files, five live baseline sources and two compiled native binaries were checked against the earlier [code-freeze receipt](../code-freeze.json) and remained unchanged. This new scheduling evidence is separate from the previously sealed stable publication manifest.

## Resource check

At 17:28 UTC, `/sys/fs/cgroup/memory.current` was close to its 8 GiB limit. The detailed counters showed approximately 0.940 GB anonymous memory, 6.261 GB file memory, and 1.365 GB reclaimable slab memory; inactive file memory was 2.160 GB. A known running native water process used approximately 88 MB RSS, including 53.5 MB anonymous memory. Thus the apparent pressure was dominated by reclaimable caches, and adding two identical water contexts was estimated to need about 110 MB additional anonymous memory. The recorded OOM-kill counter remained three, unchanged from its earlier historical value. No cache-clearing or unrelated-process operation was performed.

The selected OpenMM CPU force platform and OpenMP/BLAS environment remain limited to one thread per call. Actual process thread counts are recorded separately because runtime helper threads can exist even with that force-platform setting. No GPU resource is used.

## Scientific interpretation

This scheduling change does not establish equilibration or literature agreement. At the first complete preparation audit, seed 1102 failed the predeclared potential-energy stationarity threshold: its 50–75 ps and 75–100 ps preparation means differed by −0.15744 kcal/mol/water, versus the 0.05 threshold. Seed 1101's difference was −0.00901 kcal/mol/water. That failed gate remains visible, and every seed and registered production segment remains in the fixed cohort. Any later equilibration extension requires a separately recorded protocol; the current production is unchanged.

## Completed handoff

The two old native calls exited normally with status zero. Seed 1101 committed 185 ps, including 21 complete 4 ps NVE segments and a partial next segment; seed 1102 committed 148 ps and 12 complete segments. Their entire earlier committed prefixes passed checksum validation. Only after those native exits were observed were the stopped old dispatchers retired. Their deliberate status 143 belongs to the dispatchers, not to the successfully completed native trajectories. No integration child received a signal.

The [migration receipt](four-worker-migration-receipt.json) and [registration](four-worker-registration.json) were sealed at 17:43 UTC. The new coordinator launched seeds 1101, 1102, 1103 and 1104 at **17:44:03 UTC**, with a 7200-second outer budget and 1800-second native call budgets. The [startup proof](four-worker-startup-receipt.json) independently observed the coordinator and all four children holding the coordinator's lock, and recorded the unchanged OOM-kill counter. The [fixed initial event snapshot](startup-events.jsonl) contains only these nine startup events; the continuing runtime journal is not included.

Journal PIDs belong to the sandbox PID namespace. The startup proof records separately verified host identities obtained from this campaign's own lock file. A numeric journal PID must never be used as a signal target from another namespace. The supported scheduling-only stop mechanism is the `STOP_AFTER_CURRENT` file described above.

With the recorded sources, environment and restored campaign data available, the equivalent launch command from the repository root is:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1 \
  .venv/bin/python benchmarks/water/native_campaign_schedule_v2.py \
  --output /path/to/restored/water-convergence-2026-11-01 \
  --registration docs/validation/literature-convergence-2026-11-01/water/scheduling-v2/four-worker-registration.json \
  --workers 4 --wall-seconds 7200 --invocation-seconds 1800
```

The source checks deliberately reject an incompatible checkout or changed initial-state file. The original raw initial checkpoint must be restored at its registered relative path. This command is for the registered experiment, not a generic latest-version benchmark invocation.

All files listed in [publication-inventory.json](publication-inventory.json) are fixed evidence snapshots. Live progress, the continuing runtime journal, trajectories and binary checkpoints are outside this inventory. The separately proposed equilibration extension has not been launched and is not part of this scheduling change.
