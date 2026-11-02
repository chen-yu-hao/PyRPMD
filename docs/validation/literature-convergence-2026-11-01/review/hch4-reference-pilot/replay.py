#!/usr/bin/env python3
"""Read-only, standard-library replay of the excluded pilot's saved weights.

No PES, PyRPMD, NumPy or simulator imports. Does not form a physical rate.
Usage: python replay.py RUN_DIRECTORY NEW_RECEIPT.json
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lse(values):
    finite = [x for x in values if math.isfinite(x)]
    if not finite:
        return None
    peak = max(finite)
    return peak + math.log(math.fsum(math.exp(x-peak) for x in finite))


def statistics(values):
    total = lse(values)
    if total is None:
        return {"draws": len(values), "positive_weights": 0, "log_mean": None,
                "empirical_ESS": 0., "maximum_normalized_weight": None}
    probabilities = [math.exp(x-total) for x in values if math.isfinite(x)]
    return {"draws": len(values), "positive_weights": len(probabilities),
            "log_mean": total-math.log(len(values)),
            "empirical_ESS": 1/math.fsum(x*x for x in probabilities),
            "maximum_normalized_weight": max(probabilities)}


def integrals(rows, weights):
    summaries, logs = {}, {}
    for target in ("entry", "alpha_0.999", "alpha_1.007"):
        chosen = [i for i, row in enumerate(rows) if row["target"] == target]
        summary = statistics([weights[i] for i in chosen])
        if target == "entry":
            value = summary["log_mean"]
        else:
            channels = {str(h): statistics([weights[i] for i in chosen if rows[i]["channel"] == h])
                        for h in (0, 2, 3, 4)}
            value = lse([item["log_mean"] for item in channels.values() if item["log_mean"] is not None])
            summary["channel_means"] = channels
        logs[target] = value
        summary["log_surface_integral_without_common_rotation"] = value
        summaries[target] = summary
    ratios = {target: value-logs["entry"] if value is not None and logs["entry"] is not None else None
              for target, value in logs.items() if target != "entry"}
    difference = (logs["alpha_1.007"]-logs["alpha_0.999"]
                  if logs["alpha_1.007"] is not None and logs["alpha_0.999"] is not None else None)
    return summaries, ratios, difference


def compare(actual, expected, residuals):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for name in expected:
            compare(actual[name], expected[name], residuals)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for x, y in zip(actual, expected):
            compare(x, y, residuals)
    elif isinstance(expected, float):
        error = abs(actual-expected)
        residuals.append(error)
        assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-10), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def replay(directory):
    directory = Path(directory).resolve()
    reg = read(directory/"registration.json")
    plan = read(reg["plan_path"])
    assert digest(reg["plan_path"]) == reg["plan_sha256"]
    assert reg["source_sha256"] == plan["source_sha256"]
    for path, expected in reg["source_sha256"].items():
        assert digest(path) == expected, path
    proposal = read(plan["proposal_path"])
    assert digest(plan["proposal_path"]) == plan["proposal_sha256"]
    beta = proposal["physics"]["coordinate_constants"]["beta_hartree_inverse"]
    offset = proposal["estimator"]["common_energy_offset_hartree"]
    assert digest(directory/"ledger.json") == reg["ledger_sha256"]
    rows = read(directory/"ledger.json")["draws"]
    events = [json.loads(line) for line in (directory/"calls.jsonl").read_text().splitlines()]
    report = read(directory/"report.json")
    assert report["status"] == "complete_cost_pilot", "Only complete output can be numerically replayed; retain incomplete output unchanged."
    eligible = [row["index"] for row in rows if row["zero_reason"] is None]
    assert len(rows) == 768 and len(events) == 2*len(eligible)
    weights = [-math.inf]*len(rows)
    residuals = []
    for number, index in enumerate(eligible, 1):
        attempt, success = events[2*number-2:2*number]
        assert attempt["event"] == "attempt" and success["event"] == "success"
        assert attempt["index"] == success["index"] == index and attempt["attempt_number"] == number
        assert attempt["seed"] == rows[index]["seed"]
        assert math.isfinite(success["energy_hartree"])
        weights[index] = rows[index]["log_importance_prefactor"]-beta*(success["energy_hartree"]-offset)
        compare(success["log_weight"], weights[index], residuals)
    assert report["attempted_PES_geometries"] == report["successful_PES_geometries"] == len(eligible)
    assert report["completed_target_draws"] == 768 and report["pending_target_draws"] == 0
    assert report["geometric_zero_draws"] == len(rows)-len(eligible)
    assert report["zero_reasons"] == dict(Counter(row["zero_reason"] for row in rows if row["zero_reason"]))
    assert report["registration_sha256"] == digest(directory/"registration.json")
    summaries, ratios, difference = integrals(rows, weights)
    compare(report["weight_diagnostics"], summaries, residuals)
    compare(report["pooled_log_flux_ratios"], ratios, residuals)
    compare(report["paired_log_surface_difference"], difference, residuals)
    seeds = []
    for seed in range(2026111201, 2026111209):
        chosen = [i for i, row in enumerate(rows) if row["seed"] == seed]
        summary, ratio, delta = integrals([rows[i] for i in chosen], [weights[i] for i in chosen])
        seeds.append({"seed": seed, "weights": summary, "log_flux_ratios": ratio,
                      "paired_log_surface_difference": delta})
    compare(report["seed_summaries"], seeds, residuals)
    budget = read(directory/"CPU-budget.json")
    runtime = read(directory/"runtime.json")
    supervisor = read(directory/"supervisor.json")
    once = read(directory/"RUN_ONCE.json")
    worker = read(directory/"WORKER_ONCE.json")
    assert once["CPU_budget_sha256"] == digest(directory/"CPU-budget.json")
    assert budget["sum_hard_CPU_caps_seconds"] == sum(budget[k+"_hard_CPU_seconds"] for k in ("preparation", "supervisor", "worker")) == 60
    assert runtime["hard_CPU_seconds_in_this_process"] == budget["worker_hard_CPU_seconds"]
    assert runtime["affinity"] == plan["affinity"] and set(runtime["thread_environment"].values()) == {"1"}
    assert reg["prepare_process_token"] != once["supervisor_process_token"] != worker["worker_process_token"]
    assert worker["supervisor_PID"] == once["supervisor_PID"] == budget["supervisor_PID"]
    assert runtime["PID"] == worker["worker_PID"] == supervisor["child_PID"]
    assert supervisor["complete_cost_pilot"] and supervisor["worker_exit_code"] == 0
    for item in runtime["actual_imports"].values():
        assert digest(item["path"]) == item["sha256"] == reg["source_sha256"][item["path"]]
    inputs = {str(path): {"sha256": digest(path), "bytes": path.stat().st_size}
              for path in sorted(directory.iterdir()) if path.is_file()}
    return {"schema": "pyrpmd.hch4-reference-pilot-independent-output-replay.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(), "status": "passed_saved_output_replay",
            "scope": "Bookkeeping and independent standard-library reweighting only; no physical rate or accuracy validation.",
            "inputs": inputs, "source_guards_unchanged": len(reg["source_sha256"]),
            "target_draws": len(rows), "saved_PES_success_records_replayed": len(eligible),
            "PES_evaluations_by_replay": 0, "dynamics_steps": 0, "independent_seeds": 8,
            "scalar_comparisons": len(residuals), "maximum_absolute_replay_difference": max(residuals),
            "statistical_summaries": summaries, "pooled_log_flux_ratios": ratios,
            "paired_log_surface_difference": difference, "resources": supervisor["CPU_accounting"],
            "limitations": report["limitations"]}


if __name__ == "__main__":
    result = replay(sys.argv[1])
    with Path(sys.argv[2]).open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"receipt": sys.argv[2], "sha256": digest(sys.argv[2]),
                      "status": result["status"], "maximum_absolute_difference": result["maximum_absolute_replay_difference"]}))
