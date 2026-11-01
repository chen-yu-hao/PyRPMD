#!/usr/bin/env python3
"""Bounded OPES timestep/width sensitivity using the unchanged campaign driver.

Two predeclared seeds are compared at equal model time, update interval and
physical sampling interval. This is a sensitivity check, not a convergence
proof. Postanalysis explicitly discards the same 40 model-time units after
freezing in all cases; the main driver's step-based discard is not reused.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


path = Path(__file__).with_name("sampling.py")
spec = importlib.util.spec_from_file_location("sampling_campaign", path)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def harmonized(output, metrics):
    raw = np.load(output/"opes_v2"/metrics["label"]/"trajectory.npz")
    history = raw["history"]
    samples = history[history[:, 1] > 440.]
    inside = ((samples[:, 2] >= -1.3) & (samples[:, 2] <= 1.) &
              (samples[:, 3] >= -.2) & (samples[:, 3] <= 2.))
    samples = samples[inside]
    weights = np.exp(samples[:, 4]-samples[:, 4].max())
    left, right = weights[samples[:, 2] < 0].sum(), weights[samples[:, 2] > 0].sum()
    above = samples[:, 3] > samples[:, 2]+1.3
    a, b = weights[above].sum(), weights[~above].sum()
    bins = np.linspace(-.2, 2., 81)
    weighted = np.histogram(samples[:, 3], bins, weights=weights)[0]
    raw_counts = np.histogram(samples[:, 3], bins)[0]
    exact = raw["exact_fes"]
    density = np.exp(-exact)
    frozen = np.exp(-exact-raw["bias_values"]+raw["bias_values"].min())
    return {"seed": metrics["seed"], "dt": metrics["dt"], "sigma": metrics["initial_sigma"],
            "pace": metrics["pace"], "adaptation_time": 400., "frozen_time": 200.,
            "discarded_frozen_time": 40., "physical_sample_interval": metrics["sample_stride"]*metrics["dt"],
            "retained_frames_in_rectangle": len(samples), "fraction_inside_rectangle": float(inside.mean()),
            "transitions_adapting": metrics["transitions_adapting"],
            "direct_FES_RMSE_kBT": metrics["direct_FES_RMSE_kBT"],
            "frozen_delta_F_x_kBT": -float(np.log(right/left)) if min(left, right) else None,
            "frozen_delta_F_oblique_kBT": -float(np.log(b/a)) if min(a, b) else None,
            "JS_unbiased_reweighted": campaign.js_divergence(weighted, density),
            "JS_sample_to_exact_frozen": campaign.js_divergence(raw_counts, frozen),
            "Kish_ESS": float(weights.sum()**2/np.sum(weights*weights)),
            "spline_max_force_error": metrics["spline_max_force_error"],
            "bias_state_schema_version": metrics["bias_state_schema_version"],
            "trajectory": "opes_v2/"+metrics["label"]+"/trajectory.npz"}


def main(output):
    output = Path(output)
    records = []
    for seed in (57859, 28661):
        base = output/"opes_v2"/f"metad_fixed_y_dt0.005_s0.1_seed{seed}"/"metrics.json"
        if not base.is_file():
            raise RuntimeError("run the declared baseline first")
        records.append(harmonized(output, json.loads(base.read_text())))
    for dt, sigma, steps, freeze, pace, stride in (
            (.0025, .1, 160000, 80000, 1000, 20),
            (.005, .05, 80000, 40000, 500, 10)):
        for seed in (57859, 28661):
            metrics = campaign.opes_run(output, seed=seed, axis=1, mode="metad", adaptive=False,
                steps=steps, freeze=freeze, dt=dt, sigma=sigma, pace=pace,
                sample_stride=stride, grid_points=4097)
            records.append(harmonized(output, metrics))
            campaign.write_csv(output/"sensitivity_harmonized.csv", records)
            campaign.write_json(output/"sensitivity_harmonized.json", records)
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=campaign.DEFAULT_OUTPUT)
    main(parser.parse_args().output)
