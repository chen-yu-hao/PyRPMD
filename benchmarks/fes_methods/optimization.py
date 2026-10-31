"""Run explicit, provenance-recorded sampling optimization cases.

Uses the same independent OGRe potential and bin-integral oracle as the first
validation. Missing bins are retained. Settings are supplied as a JSON override
of ogre_benchmark.config, and each label has an isolated checkpoint directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pyrpmd.system import System
from pyrpmd.workflows.fes import run_fes, write_json
from pyrpmd.workflows.metadynamics import run_metadynamics
from ogre_benchmark import config, oracle, KJ, ROOT
from summarize import rebin


def merge(base, changes):
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value


def run_case(model, method, seed, output, library, label, changes, *, workflow=None, provenance=None):
    out = Path(output).resolve() / f"{model}-{label}-s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    c = config(model, method, seed, Path(library).resolve())
    merge(c, changes)
    system = System(("H",), np.array([1837.152647]), np.zeros((1, 3)))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((ROOT / "src/pyrpmd").rglob("*.py"))}
    write_json(out / "input-resolved.json", {"config": c, "system": system.to_dict(),
                                            "source_sha256": hashes, "workflow_override": provenance})
    start = time.monotonic()
    try:
        runner = workflow or (run_metadynamics if c["task"] == "metadynamics" else run_fes)
        result = runner(
            c, system, out / "run", return_data=True)
    except (RuntimeError, ValueError) as error:
        write_json(out / "failure.json", {"error": str(error), "wall_seconds": time.monotonic()-start})
        raise
    wall = time.monotonic() - start
    data = result["data"]
    is_meta = c["task"] == "metadynamics"
    exact, point = oracle(data["edges"], model, point=is_meta)
    good = ~data["mask"] & np.isfinite(data["free_energy"])
    free = data["free_energy"] * KJ
    residual = free - exact
    residual -= residual[good].mean()
    paper = (free-free[good].min()) - (point-point[good].min())
    windows = result["summary"].get("windows", 1)
    planpath = out / "run/adaptive-plan.json"
    plan = json.loads(planpath.read_text()) if planpath.exists() else {"rounds": []}
    pilot = sum(len(r["centers"]) * (r["pilot"]["equilibration_steps"] + r["pilot"]["production_steps"])
                for r in plan["rounds"])
    steps = windows * (c["sampling"]["equilibration_steps"] + c["sampling"]["production_steps"]) + pilot
    report = {"model": model, "method": method, "label": label, "seed": seed,
              "beads": c["beads"], "wall_seconds": wall, "windows": windows,
              "integration_steps": steps, "pilot_steps": pilot, "pilot_rounds": len(plan["rounds"]),
              "samples": result["summary"]["samples"], "coverage": float(good.mean()),
              "rmse_kjmol": float(np.sqrt(np.mean(residual[good]**2))),
              "paper_point_min_gauge_rmse_kjmol": float(np.sqrt(np.mean(paper[good]**2))),
              "max_abs_error_kjmol": float(np.max(abs(residual[good]))),
              "physical_convergence_claimed": False, "pilot_samples_used_in_production": 0}
    np.savez_compressed(out / "comparison.npz", free_energy_kjmol=free, reference_kjmol=exact,
                        point_reference_kjmol=point, residual_kjmol=residual, mask=~good,
                        edges_0=data["edges"][0], edges_1=data["edges"][1])
    if not is_meta:
        report["paper_grid"] = rebin(out, model)
    if plan["rounds"]:
        last = plan["rounds"][-1]
        report["pilot_min_ess"] = float(min(last["effective_samples_per_window"]))
        report["pilot_coverage"] = last.get("coverage", None)
    write_json(out / "comparison.json", report)
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["double_well", "ackley"], required=True)
    parser.add_argument("--method", choices=["fixed", "adaptive", "unbiased", "standard", "well_tempered"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--override", type=Path, required=True)
    args = parser.parse_args()
    run_case(args.model, args.method, args.seed, args.output, args.library, args.label,
             json.loads(args.override.read_text()))


if __name__ == "__main__":
    main()
