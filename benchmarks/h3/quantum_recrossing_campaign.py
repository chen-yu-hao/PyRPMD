"""Two-process, fixed-inventory scheduler for registered H3 quantum children."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.h3.quantum_surface_recrossing import PARENT_PROTOCOL, register, run_seed, sha
from pyrpmd.io import write_json


def execute(job):
    output, library, seed, horizon, pair_batch = job
    parent = PARENT_PROTOCOL.parent / "primary/ts" / f"seed-{seed}" / "report.json"
    started = time.monotonic()
    while not parent.exists() or not json.loads(parent.read_text())["complete"]:
        if time.monotonic() - started > 7200:
            raise TimeoutError(f"registered parent chain {seed} not complete within two hours")
        print(json.dumps({"event": "waiting_for_complete_registered_parent", "seed": seed}), flush=True)
        time.sleep(30)
    reports = []
    for timestep in (.1, .05):
        report = run_seed(output, library, seed, timestep, horizon=horizon, pair_batch=pair_batch)
        if not report["complete"] or not report["source_unchanged"]:
            raise ValueError(f"incomplete or source-changed child run for seed {seed}")
        reports.append(report)
    print(json.dumps({"event": "seed_both_timesteps_complete", "seed": seed,
                      "horizon_fs": horizon, "pair_batch": pair_batch}), flush=True)
    return {"seed": seed, "horizon_fs": horizon, "pair_batch": pair_batch, "complete": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=(200, 400), default=200)
    parser.add_argument("--pair-batch", type=int, choices=(0, 1), default=0)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    protocol = register(args.output, args.library)
    inventory = {"schema": "pyrpmd.h3.quantum-child-schedule.v1", "horizon_fs": args.horizon,
                 "pair_batch": args.pair_batch,
                 "seeds": protocol["seeds"], "timestep_order_fs": [.1, .05],
                 "protocol_sha256": sha(args.output / "protocol.json"),
                 "scheduler_source_sha256": sha(__file__),
                 "rule": "All registered seeds and both timesteps; no rate-dependent selection or early stopping."}
    path = args.output / f"schedule-{args.horizon}-pairs-{args.pair_batch}.json"
    if path.exists() and json.loads(path.read_text()) != inventory:
        raise ValueError("registered schedule differs")
    write_json(path, inventory)
    jobs = [(args.output, args.library, seed, args.horizon, args.pair_batch) for seed in protocol["seeds"]]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        results = list(pool.map(execute, jobs))
    write_json(args.output / f"schedule-{args.horizon}-pairs-{args.pair_batch}-completed.json",
               {"inventory": inventory, "completed": results, "workers": args.workers,
                "created_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
