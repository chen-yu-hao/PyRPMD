"""Controlled thermostat/timestep pilot scan for OGRe model windows.

Each setting uses identical seeds, centers, step counts, stiffness and saved
stride. Changing dt changes physical trajectory length; this is explicitly
recorded and is not interpreted as a same-physical-time comparison. Coordinate
ESS is an autocorrelation diagnostic, not a guarantee of global convergence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from ogre_benchmark import config, potential, BETA, KJ
from pyrpmd.constants import AU_TIME_FS
from pyrpmd.system import System
from pyrpmd.workflows.fes import _sample_window, resolve_umbrellas, write_json
from pyrpmd.analysis.fes import statistical_inefficiency


def reference(center, stiffness, model):
    """Independent dense quadrature for biased-coordinate mean and variance."""
    sigma = np.sqrt(1 / (BETA * stiffness[0, 0]))
    axes = [np.linspace(c - 6 * sigma, c + 6 * sigma, 801) for c in center]
    x, y = np.meshgrid(*axes, indexing="ij")
    energy = potential(x, y, model)
    if model == "ackley":
        energy += 4 * (np.maximum(abs(x) - 8, 0)**4 + np.maximum(abs(y) - 8, 0)**4)
    energy = energy / KJ + stiffness[0, 0] * ((x-center[0])**2 + (y-center[1])**2) / 2
    probability = np.exp(-BETA * (energy - energy.min()))
    probability /= probability.sum()
    mean = np.array([(probability * value).sum() for value in (x, y)])
    variance = np.array([(probability * (value - mu)**2).sum() for value, mu in zip((x, y), mean)])
    return mean, variance


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    library = args.library.resolve()
    c = config(args.model, "fixed", args.seed, library)
    c["dt_au"] = args.dt_fs / AU_TIME_FS
    c["sampling"].update(equilibration_steps=args.equil, production_steps=args.steps,
                         andersen_interval=args.andersen, sample_interval=args.stride)
    if args.all_windows:
        axis = np.linspace(-6, 6, 7) if args.model == "ackley" else None
        if axis is not None:
            c["umbrella"]["centers"] = [[x, y] for x in axis for y in axis]
    else:
        c["umbrella"]["centers"] = ([[-4, 6], [4, 4], [0, -6], [2, 0], [0, 0], [6, 6]]
                                    if args.model == "ackley" else
                                    [[-.5, -1], [-.25, -.5], [0, 0], [.25, .5], [.5, 1]])
    system = System(("H",), np.array([1837.152647]), np.zeros((1, 3)))
    _, specs, centers, stiffness = resolve_umbrellas(c, system)
    write_json(out / "input-resolved.json", {"config": c, "system": system.to_dict()})
    records = []
    start = time.monotonic()
    for i, center in enumerate(centers):
        stream = _sample_window(dict(config=c, system=system, specs=specs, window=i,
                                     replica=0, center=center, stiffness=stiffness, output=out))
        values = stream["samples"]
        mean, variance = reference(center, stiffness, args.model)
        ess = len(values) / statistical_inefficiency(values)
        moment_ess = len(values) / statistical_inefficiency(np.column_stack((values, (values-values.mean(0))**2)))
        records.append(dict(window=i, center=center.tolist(), samples=len(values), ess=ess,
                            mean_and_variance_ess=moment_ess,
                            mean=values.mean(0).tolist(), variance=values.var(0).tolist(),
                            reference_mean=mean.tolist(), reference_variance=variance.tolist(),
                            normalized_mean_error=float(np.sqrt(np.mean((values.mean(0)-mean)**2/variance))),
                            normalized_variance_error=float(np.sqrt(np.mean((values.var(0)/variance-1)**2))),
                            finite=bool(np.isfinite(values).all())))
    result = dict(model=args.model, seed=args.seed, dt_fs=args.dt_fs, andersen_interval=args.andersen,
                  collision_fs=args.dt_fs*args.andersen, production_steps=args.steps,
                  production_time_fs=args.dt_fs*args.steps, equilibration_steps=args.equil,
                  window_count=len(records), ess_min=min(r["ess"] for r in records),
                  ess_median=float(np.median([r["ess"] for r in records])),
                  mean_and_variance_ess_min=min(r["mean_and_variance_ess"] for r in records),
                  mean_and_variance_ess_median=float(np.median([r["mean_and_variance_ess"] for r in records])),
                  normalized_mean_error_rms=float(np.sqrt(np.mean([r["normalized_mean_error"]**2 for r in records]))),
                  normalized_variance_error_rms=float(np.sqrt(np.mean([r["normalized_variance_error"]**2 for r in records]))),
                  wall_seconds=time.monotonic()-start, windows=records)
    write_json(out/"summary.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "windows"}), flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--model", choices=("ackley", "double_well"), default="ackley")
    p.add_argument("--seed", type=int, default=5101)
    p.add_argument("--steps", type=int, default=80000)
    p.add_argument("--equil", type=int, default=10000)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--andersen", type=int, default=100)
    p.add_argument("--dt-fs", type=float, default=.1)
    p.add_argument("--all-windows", action="store_true")
    run(p.parse_args())
