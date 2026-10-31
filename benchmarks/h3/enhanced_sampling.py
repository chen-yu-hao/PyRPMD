"""Compare fixed 96-window and adaptive H3 PMFs; audit unconfined xi MetaD.

This is a sampling diagnostic, not a rate or convergence certification.
Umbrellas use rate_flux; metadynamics deliberately retains cv_density and
cannot be compared numerically to that PMF. analysis.ranges is not a wall.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from pyrpmd.analysis.fes import mbar_rate_pmf
from pyrpmd.workflows.fes import run_fes, write_json
from pyrpmd.workflows.metadynamics import run_metadynamics
from pyrpmd.workflows.rate import _prepare

try:
    from .rpmd import configuration
except ImportError:  # Direct script invocation from the repository root.
    from rpmd import configuration


def sampling_config(method, output, *, workers=4, library=None, profile="production", seed=721062):
    if method not in ("fixed96", "adaptive", "metadynamics"):
        raise ValueError("unknown enhanced-sampling method")
    if profile not in ("short", "production"):
        raise ValueError("profile must be short or production")
    config, system = configuration("rate", output, workers=workers, library=library)
    config["seed"] = seed
    config["sampling"].update(
        equilibration_steps=1000 if profile == "short" else 3000,
        production_steps=2000 if profile == "short" else 10000,
        sample_interval=10, interval_steps=10,
    )
    # Remove the template's human-unit aliases after overriding resolved steps.
    for key in ("equilibration", "production", "interval"):
        config["sampling"].pop(key, None)
    config["umbrella"].update(start=-0.05, stop=1.1, count=96, kappa=400 / config["beta"])
    config["analysis"].update(bootstrap=0, bins=120, ranges=[[-0.15, 1.25]], min_count=5)
    config["output"]["checkpoint_steps"] = 5000
    if method == "adaptive":
        config["umbrella"]["count"] = 6
        config["adaptive_umbrella"] = {"enabled": True}
    return config, system


def run_case(method, output, *, workers=4, library=None, profile="production", seed=721062,
             bootstrap=0, production_steps=None):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a fresh benchmark output directory")
    config, system = sampling_config(method, output, workers=workers, library=library,
                                     profile=profile, seed=seed)
    if production_steps is not None:
        if production_steps < 20:
            raise ValueError("production_steps must include at least two saved samples")
        config["sampling"]["production_steps"] = production_steps
    config = _prepare(config, system, output)
    if method == "metadynamics":
        config["task"] = "metadynamics"
        config.pop("umbrella")
        config["analysis"].update(method="bias_estimate", measure="cv_density")
        config["metadynamics"] = {"sigma": 0.03, "height": 0.2 / config["beta"],
                                  "pace": 100, "bias_factor": 10.0}
    write_json(output / "config.json", config)
    started = time.perf_counter()
    workflow = run_metadynamics if method == "metadynamics" else run_fes
    result = workflow(config, system, output, return_data=True)
    elapsed = time.perf_counter() - started
    summary = result["summary"]
    streams = result["streams"]
    report = {"method": method, "profile": profile, "seed": seed,
              "temperature_K": config["temperature_K"], "beads": config["beads"],
              "measure": summary["measure"], "sampling": config["sampling"],
              "sampling_and_fes_seconds": elapsed, "samples": summary["samples"],
              "windows": summary.get("windows"), "converged": False,
              "scope": "PMF/exploration only; no recrossing or rate calculation",
              "production_integration_steps": len(streams) * (
                  config["sampling"]["equilibration_steps"] + config["sampling"]["production_steps"]),
              "pilot_integration_steps": 0}
    if method == "metadynamics":
        report.update(hills_per_replica=summary["hills_per_replica"],
                      sampling_diagnostics=summary["sampling_diagnostics"],
                      common_support=summary["common_support"],
                      comparable_to_rate_pmf=False, warnings=summary["warnings"])
    else:
        pmf = mbar_rate_pmf(
            [s["samples"] for s in streams], [s["center"] for s in streams],
            [s["stiffness"] for s in streams], config["beta"],
            xi_star=1.00925, reference=0., bins=120, ranges=[[-0.15, 1.25]],
            bootstrap=bootstrap, seed=seed, min_count=5, compare_ui=True,
        )
        report.update(
            overlap_connected=pmf["overlap_connected"], pmf_ready=pmf["ready"],
            minimum_stream_ess=float(np.min(pmf["effective_samples_per_stream"])),
            median_stream_ess=float(np.median(pmf["effective_samples_per_stream"])),
            barrier_hartree=pmf["free_energy_barrier_hartree"],
            barrier_standard_error_hartree=pmf["barrier_standard_error_hartree"],
            bootstrap=pmf["bootstrap"],
            xi=pmf["xi"], free_energy_hartree=pmf["free_energy"], mask=pmf["mask"],
        )
        if method == "adaptive":
            plan = json.loads((output / "adaptive-plan.json").read_text())
            report["adaptive_umbrella"] = summary["adaptive_umbrella"]
            report["pilot_rounds"] = [{"windows": len(r["centers"]), "pilot": r["pilot"],
                                        "spanning_overlap": r["spanning_overlap"],
                                        "minimum_window_ess": min(r["effective_samples_per_window"])}
                                       for r in plan["rounds"]]
            report["pilot_integration_steps"] = sum(
                len(r["centers"]) * config["sampling"]["replicas"] * sum(r["pilot"].values())
                for r in plan["rounds"])
    write_json(output / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("fixed96", "adaptive", "metadynamics"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--profile", choices=("short", "production"), default="production")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=721062)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--production-steps", type=int)
    args = parser.parse_args()
    report = run_case(args.method, args.output, workers=args.workers, library=args.library,
                      profile=args.profile, seed=args.seed, bootstrap=args.bootstrap,
                      production_steps=args.production_steps)
    print(json.dumps({k: report.get(k) for k in ("method", "samples", "windows", "barrier_hartree",
                      "minimum_stream_ess", "common_support", "sampling_and_fes_seconds")}))


if __name__ == "__main__":
    main()
