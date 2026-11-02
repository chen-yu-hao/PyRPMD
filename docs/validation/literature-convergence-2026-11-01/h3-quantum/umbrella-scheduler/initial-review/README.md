# Complete-cohort umbrella scheduling wrapper

**25 fake-worker tests and the real 30,000-stage preflight passed. No physical
sampling was launched by this validation.** The new
`benchmarks/h3/quantum_sampling_scheduler.py` invokes the unchanged
`quantum_sampling_campaign.one_seed` once for each unfinished registered seed.
It changes process scheduling and completion bookkeeping only. Sampler, driver,
protocol, schedule, mass, temperature, model, observations and RNG are unchanged.

The tested scheduler SHA-256 is
`f429904e2b8eb7b7c35b29f51d0f28a63c55ec84be5caf50719eec0e94c1122e`;
the test file SHA-256 is
`54c4eb94acc49fbe4a873da2e96c98e56290d2535a896dc9a8d34d84757d6c58`.
[execution.json](execution.json) records exact commands, environment, timing and
unchanged source hashes; [tests.log](tests.log), [tests.xml](tests.xml) and
[check-only.log](check-only.log) are the actual outputs.

## Guarantees and scope

- All eight seeds are fixed in registered order. There is no subset-seed CLI.
  Only the next registered prefix, 100,000 or 300,000, is admitted.
- The prior snapshot and continuation decision require explicit expected SHA-256
  values. The decision must cite the complete prior cohort and an observed
  precision failure; closeness to the paper cannot authorize continuation.
- Before any worker is submitted, all eight checkpoints, exact settings,
  earlier sample and diagnostic prefixes, stage-state files and receipts are
  verified. Registered sources, actual imported module locations, model and
  schedule identities are checked. The archived initial stage is verified too.
- One coordinator holds the campaign lock. Each worker calls the frozen
  `one_seed` in its own seed directory. Only the coordinator can atomically
  create `campaign-run/stage-100000.json` or `stage-300000.json`, after all eight
  results and prior prefixes verify. An existing global manifest is never
  overwritten. Sampling completion is separate from statistical convergence.
- An error or invalid control stops new submissions and drains already running
  workers. A separate failure receipt records exact post-drain checkpoints and
  completed seeds; no partial cohort is published as complete. Explicit
  hash-bound recovery reuses completed seed receipts without rewriting them
  and resumes partially progressed chains from their current saved RNG/state.

The lock is cooperative between this wrapper's invocations. Do not run the old
campaign CLI, call `one_seed` independently or edit seed files concurrently. A
hard process/container kill may leave a lock and no drained failure receipt;
the wrapper deliberately refuses to break that lock automatically or trust
unrecorded progress. Inspect owned processes and data before a separate recovery.

## Scheduling-only 2-to-4 worker control

Without `--worker-control`, `--workers` is a fixed limit of one through four.
With a control file, its sole field `active_workers` is the active submission
limit and may range from one through four; the process-pool ceiling remains four.
Raising two to four submits the next registered seeds. Lowering a limit allows
existing workers to finish and constrains subsequent submissions. No worker is
killed and no sampling parameter or RNG is touched. Control changes and their
hashes are recorded in `scheduler-attempts/.../events.jsonl`.

Initialize the control file **once**, after review and before launch:

```python
import json
from pathlib import Path

path = Path("/workspace/scratch/a5e4820daedd/literature-convergence-quantum/umbrella-worker-control.json")
with path.open("x") as stream:
    json.dump({"active_workers": 2}, stream)
    stream.write("\n")
```

After the extra-pair child jobs have finished and capacity is available, update
that separate scheduling file atomically:

```python
import json
import os
from pathlib import Path
import tempfile

path = Path("/workspace/scratch/a5e4820daedd/literature-convergence-quantum/umbrella-worker-control.json")
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
    json.dump({"active_workers": 4}, stream)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
    temporary = stream.name
os.replace(temporary, path)
```

The coordinator reads only this explicit control, never other workloads or
unrelated processes. Do not edit the file nonatomically while it is being read.

## Reviewed launch command

[launch-plan.json](launch-plan.json) supplies the exact argument vector and
environment for root review. From the original checkout:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m benchmarks.h3.quantum_sampling_scheduler \
  --campaign /workspace/scratch/a5e4820daedd/literature-convergence-quantum/campaign-run \
  --library /workspace/scratch/a5e4820daedd/literature-audit-assets/libbkmp2.so \
  --protocol docs/validation/literature-convergence-2026-11-01/h3-quantum/protocol.json \
  --prior-snapshot docs/validation/literature-convergence-2026-11-01/h3-quantum/umbrella-stage-30000/raw-archive/stage-snapshot.json \
  --prior-snapshot-sha256 0ea2cd78af59dafb91ab4f0846786b6b63a63457ecc4434e8489ec1478b6d9a4 \
  --decision docs/validation/literature-convergence-2026-11-01/h3-quantum/umbrella-stage-30000/continuation-decision.json \
  --decision-sha256 c96255a6eea63023f4eefe9450daa8fc4d67399d55a3143769bd83aa41169c16 \
  --stage 100000 --workers 2 \
  --worker-control /workspace/scratch/a5e4820daedd/literature-convergence-quantum/umbrella-worker-control.json
```

Adding `--check-only` performs the no-worker preflight. The recorded successful
preflight used fixed two-worker configuration before the mutable control file
was created. A later 300,000 continuation requires the newly completed 100,000
manifest as prior snapshot and a fresh matching precision decision based on its
complete analysis; this wrapper does not choose that continuation automatically.

For a drained failure, preserve the failure JSON, compute its SHA-256, and add
`--resume-failure PATH --resume-failure-sha256 HASH` to the same launch inputs.
Unrecorded checkpoint changes, changed source/model/settings or modified earlier
receipts are rejected. Retry does not replace a complete global manifest.

## Validation covered

The fake workers exercise full eight-seed completion and exact ordered prefixes,
100,000-to-300,000 continuation, partial-checkpoint/RNG recovery, reuse of already
completed seed receipts, concurrent-run exclusion, global no-overwrite behavior,
source/model/settings/sample/diagnostic/state tampering, runtime source changes,
wrong returned receipts, the two-to-four worker increase, invalid-control draining,
import-path mismatch and rejection of paper-point or incomplete-cohort decisions.
They generate deterministic test arrays and do not load a PES or propagate atoms.

The real preflight verified all eight original checkpoints and the immutable
stage archive with zero workers launched. It did not rerun WHAM or alter any
scientific gate. The [publication inventory](publishable-files.json) lists only
the new scheduling source/tests and compact validation files; add the inventory
itself separately. Raw test directories and real campaign data are excluded.
