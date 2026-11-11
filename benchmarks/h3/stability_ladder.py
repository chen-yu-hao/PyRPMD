"""Opt-in short MetaD/OPES domain checks before a separate production run.

Each bias level starts independent histories in a fresh directory. Input freeze
and reweighting settings are retained, so every requested budget must include
the original freeze, relaxation and retained-sample phases. A final dt/2 stage
checks the strongest bias at the same physical duration and deposition cadence.
Passing this ladder is a finite-budget domain check, not a converged PMF or rate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from pyrpmd.input import JobSpec
from pyrpmd.input_fes import fes_method
from pyrpmd.io import write_json
from pyrpmd.units import parse_quantity
from pyrpmd.workflows.biased_sampling import history_failures


def plan_ladder(config, system, *, factors, production_steps):
    """Validate every stage before starting any trajectories."""
    method = fes_method(config)
    if method not in ("metadynamics", "opes"):
        raise ValueError("stability ladder requires a metadynamics or opes input")
    bias = config[method]
    if (bias.get("shared_walkers") or bias.get("multiple_walkers")
            or bias.get("walkers_mode", "serial") != "serial"):
        raise ValueError("stability ladder currently requires independent walker histories")
    factors = np.asarray(factors, float)
    steps = np.asarray(production_steps)
    if (factors.ndim != 1 or len(factors) < 1 or not np.isfinite(factors).all()
            or np.any(factors <= 0) or np.any(np.diff(factors) <= 0)
            or factors[-1] != 1 or steps.shape != factors.shape
            or steps.dtype.kind not in "iu" or np.any(steps < 1)
            or np.any(np.diff(steps) <= 0)):
        raise ValueError("use increasing positive factors ending at 1 and matching increasing integer step budgets")
    field = "height" if method == "metadynamics" else "barrier"
    strength = parse_quantity(bias.get(field + "_au", bias.get(field)), "energy")
    stages = []
    for factor, budget in zip(factors, steps):
        stage = deepcopy(config)
        stage[method].pop(field, None)
        stage[method][field + "_au"] = float(factor * strength)
        stage["sampling"]["production_steps"] = int(budget)
        stage["sampling"]["monitor_pair_distances"] = True
        for alias in ("production", "equilibration", "interval"):
            stage["sampling"].pop(alias, None)
        stages.append({"label": f"level-{len(stages) + 1:02d}", "factor": float(factor),
                       "timestep_scale": 1.0, "config": stage})
    fine = deepcopy(stages[-1])
    fine.update(label="strongest-dt-half", timestep_scale=.5)
    stage = fine["config"]
    stage["dt_au"] *= .5
    if "dt_au" in stage["sampling"]:
        stage["sampling"]["dt_au"] *= .5
    stage.setdefault("dynamics", {})["timestep"] = stage["dt_au"]
    if "timestep" in stage["sampling"]:
        stage["sampling"]["timestep"] = stage["sampling"].get("dt_au", stage["dt_au"])
    for key, default in (("equilibration_steps", 0), ("production_steps", 1),
                         ("sample_interval", 1), ("andersen_interval", 10)):
        stage["sampling"][key] = 2 * stage["sampling"].get(key, default)
    if "interval_steps" in stage["sampling"]:
        stage["sampling"]["interval_steps"] *= 2
    thermostat = stage["sampling"].get("thermostat_params", {})
    for key in ("interval", "interval_steps"):
        if key in thermostat:
            thermostat[key] *= 2
    for key in ("pace", "freeze_after_steps", "freeze_equilibration_steps"):
        if key in stage[method]:
            stage[method][key] *= 2
    stage[method].setdefault("pace", 200)
    stage["output"]["checkpoint_steps"] = 2 * stage["output"].get("checkpoint_steps", 1000)
    stage["output"].pop("checkpoint_interval", None)
    stages.append(fine)
    from pyrpmd.workflows.metadynamics import validate_metadynamics
    from pyrpmd.workflows.opes import validate_opes
    validate = validate_metadynamics if method == "metadynamics" else validate_opes
    for item in stages:
        validate(item["config"], system)
    return stages


def _run_stage(config, system, output):
    from pyrpmd.workflows.metadynamics import run_metadynamics
    from pyrpmd.workflows.opes import run_opes
    workflow = run_metadynamics if fes_method(config) == "metadynamics" else run_opes
    return workflow(config, system, output, return_data=True)


def _stage_gate(result, config, output, minimum_distance):
    replicas = config["sampling"].get("replicas", 1)
    total = config["sampling"].get("equilibration_steps", 0) + config["sampling"]["production_steps"]
    streams = result["streams"]
    if sorted(s["replica"] for s in streams) != list(range(replicas)):
        raise ValueError("not every expected walker returned a history")
    if any(not s.get("complete", False) or s.get("elapsed") != total for s in streams):
        raise ValueError("an incomplete walker history blocks the next ladder stage")
    failures = history_failures(output)
    if failures:
        raise ValueError("a failed walker history blocks the next ladder stage")
    diagnostics = []
    for stream in sorted(streams, key=lambda s: s["replica"]):
        path = Path(output) / "diagnostics" / f"{fes_method(config)}_{stream['replica']:04d}.pair-distances.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        previous, minimum = 0, float("inf")
        for index, record in enumerate(records):
            value = record["minimum_bead_pair_distance"]
            expected = previous if index == 0 else previous + 1
            if (record["start_step"] != previous or record["end_step"] != expected
                    or not record["finite"] or value is None or not np.isfinite(value)
                    or value < minimum_distance):
                raise ValueError("incomplete or out-of-domain block distance records")
            minimum = min(minimum, value)
            previous = record["end_step"]
        if previous != total:
            raise ValueError("block distance records do not cover the complete walker history")
        diagnostics.append({"replica": stream["replica"], "accepted_steps": len(records) - 1,
                            "minimum_bead_pair_distance": minimum, "path": str(path)})
    return diagnostics


def run_ladder(config, system, output, *, factors=(.1, .3, 1.), production_steps,
               minimum_distance=.2):
    """Stop on the first failed stage; never retry or combine partial histories."""
    if not np.isfinite(minimum_distance) or minimum_distance <= 0:
        raise ValueError("minimum_distance must be a positive distance in bohr")
    stages = plan_ladder(config, system, factors=factors, production_steps=production_steps)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "pyrpmd.stability-ladder.v1", "passed": False,
              "role": "finite_budget_domain_diagnostic", "production_started": False,
              "minimum_distance_bohr": minimum_distance, "stages": [],
              "limitations": "Domain survival at dt and dt/2 is not timestep, PMF or rate convergence."}
    write_json(output / "ladder.json", report)
    for item in stages:
        directory = output / item["label"]
        directory.mkdir()
        stage = item["config"]
        stage["output"]["directory"] = str(directory)
        record = {key: value for key, value in item.items() if key != "config"}
        record.update(directory=str(directory), status="running")
        report["stages"].append(record)
        write_json(directory / "config.json", stage)
        write_json(output / "ladder.json", report)
        print(json.dumps({"stage": item["label"], "status": "running"}), flush=True)
        try:
            result = _run_stage(stage, system, directory)
            record["walkers"] = _stage_gate(result, stage, directory, minimum_distance)
            record["status"] = "passed"
        except Exception as error:
            record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
            report["failed_stage"] = item["label"]
            write_json(output / "ladder.json", report)
            raise
        write_json(output / "ladder.json", report)
        print(json.dumps({"stage": item["label"], "status": "passed"}), flush=True)
    report["passed"] = True
    write_json(output / "ladder.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factors", type=float, nargs="+", default=[.1, .3, 1.])
    parser.add_argument("--production-steps", type=int, nargs="+", required=True)
    parser.add_argument("--minimum-distance", type=float, default=.2, help="PES domain boundary in bohr (BKMP2: 0.2)")
    args = parser.parse_args()
    config, system = JobSpec.from_file(args.input).resolve()
    run_ladder(config, system, args.output, factors=args.factors,
               production_steps=args.production_steps, minimum_distance=args.minimum_distance)


if __name__ == "__main__":
    main()
