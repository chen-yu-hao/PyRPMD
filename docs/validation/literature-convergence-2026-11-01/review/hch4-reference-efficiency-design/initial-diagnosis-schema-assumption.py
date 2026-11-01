#!/usr/bin/env python3
"""Diagnose the complete excluded pilot without a PES import, RNG or new draw.

All moments are descriptive self-normalized importance moments. With the very
small empirical ESS, they are not covariance estimates suitable for fitting a
new proposal without independent calibration and fresh diagnostic data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
PILOT = BASE / "hch4-reference-pilot"
EXPECTED_INVENTORY = "10ce01b004b9870e283a79e746781543edcafa5d7598e6539a21c2bfd008528e"
TARGETS = ("alpha_0.999", "alpha_1.007", "entry")
SEEDS = tuple(range(2026111201, 2026111209))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(logs):
    values = np.asarray(logs, dtype=float)
    maximum = float(np.max(values))
    scaled = np.exp(values-maximum)
    return scaled/scaled.sum()


def folded_coordinates(row):
    """Parity-fold only descriptive coordinates; do not combine original draws."""
    z = np.asarray(row["shape"], dtype=float).copy()
    u = np.asarray(row["direction"], dtype=float).copy()
    if z[4] < 0:
        z[[4, 7]] *= -1
        u[1] *= -1
    if row["target"] == "entry":
        return z
    if not 1+u[2] > 0:
        raise ValueError("a saved draw lies exactly on the excluded stereographic pole")
    return np.r_[z, u[:2]/(1+u[2])]


def describe(rows, success):
    valid = [row for row in rows if row["index"] in success]
    weights = normalized([success[row["index"]]["log_weight"] for row in valid])
    by_index = {row["index"]: float(w) for row, w in zip(valid, weights)}
    sorted_weights = np.sort(weights)[::-1]
    components = {}
    for component in sorted({row["shape_component"] for row in rows}):
        chosen = [row for row in rows if row["shape_component"] == component]
        components[component] = {
            "unconditional_draws": len(chosen),
            "PES_evaluations": sum(row["index"] in success for row in chosen),
            "observed_normalized_weight": sum(by_index.get(row["index"], 0.) for row in chosen),
        }
    x = np.asarray([folded_coordinates(row) for row in valid])
    mean = weights@x
    centered = x-mean
    covariance = (centered*weights[:, None]).T@centered
    standard = np.sqrt(np.maximum(np.diag(covariance), 0))
    correlation = covariance/np.outer(standard, standard)
    loo = []
    for omitted in SEEDS:
        keep = np.asarray([row["seed"] != omitted for row in valid])
        w = weights[keep]/weights[keep].sum()
        average = w@x[keep]
        sd = np.sqrt(w@((x[keep]-average)**2))
        loo.append({"omitted_seed": omitted, "mean": average.tolist(), "standard_deviation": sd.tolist()})
    bounds = []
    for row in valid:
        q = np.asarray(row["coordinates_bohr"], dtype=float)
        radii = np.linalg.norm(q[[0, 2, 3, 4]]-q[1], axis=1)
        bounds.append(bool(np.any(radii < 1.6) or np.any(radii > 4.0) or row["minimum_pair_bohr"] < .9))
    ranks = sorted(range(len(valid)), key=lambda i: -weights[i])
    return {
        "unconditional_draws": len(rows), "geometric_zero_draws": len(rows)-len(valid),
        "positive_geometric_weights": len(valid),
        "empirical_ESS": float(1/(weights@weights)),
        "maximum_normalized_weight": float(weights.max()),
        "top_four_observed_mass": float(sorted_weights[:4].sum()),
        "draws_for_90_percent_observed_mass": int(np.searchsorted(np.cumsum(sorted_weights), .9)+1),
        "shape_component_contributions": components,
        "weighted_mean_direction_z": float(sum(w*row["direction"][2] for row, w in zip(valid, weights))),
        "observed_weight_with_direction_z_above_point9": float(sum(w for row, w in zip(valid, weights) if row["direction"][2] > .9)),
        "observed_boundary_band_weight": float(weights@np.asarray(bounds)),
        "boundary_band_definition": "Any CH radius <1.6 or >4.0 bohr, or any pair <0.9 bohr, within original finite domain.",
        "largest_five_draws": [{"index": valid[i]["index"], "seed": valid[i]["seed"],
                                "channel": valid[i]["channel"], "shape_component": valid[i]["shape_component"],
                                "weight_fraction": float(weights[i])} for i in ranks[:5]],
        "descriptive_folded_moments": {
            "coordinates": ["r", "s", "v", "ax", "ay", "az", "bx", "by", "bz"]
            + ([] if rows[0]["target"] == "entry" else ["stereographic_tx", "stereographic_ty"]),
            "folding": "Reflect ay, by and direction_y only when ay<0; original samples/weights/counts unchanged.",
            "mean": mean.tolist(), "population_covariance": covariance.tolist(),
            "standard_deviation": standard.tolist(),
            "correlation_eigenvalues": np.linalg.eigvalsh(correlation).tolist(),
            "leave_one_seed_out": loo,
            "not_a_calibration_covariance": True,
        },
    }


def run(output):
    inventory_path = PILOT / "publication-inventory.json"
    if sha(inventory_path) != EXPECTED_INVENTORY:
        raise ValueError("the complete frozen excluded-pilot inventory changed")
    inventory = json.loads(inventory_path.read_text())
    root = Path(__file__).resolve().parents[5]
    selected = [PILOT/"completed-run"/name for name in ("ledger.json", "calls.jsonl", "report.json")]
    lookup = {row["path"]: row for row in inventory["files"]}
    for path in selected:
        record = lookup[str(path.relative_to(root))]
        if sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("saved pilot input no longer matches its frozen inventory")
    ledger = json.loads(selected[0].read_text())["draws"]
    events = [json.loads(line) for line in selected[1].read_text().splitlines()]
    success = {row["index"]: row for row in events if row["event"] == "success"}
    attempts = [row["index"] for row in events if row["event"] == "attempt"]
    report = json.loads(selected[2].read_text())
    if (len(ledger) != 768 or [row["index"] for row in ledger] != list(range(768))
            or len(success) != 641 or len(attempts) != 641 or len(set(attempts)) != 641
            or set(attempts) != set(success) or report["status"] != "complete"):
        raise ValueError("expected the complete one-attempt pilot, not a selected subset")
    if sorted({row["seed"] for row in ledger}) != list(SEEDS):
        raise ValueError("all eight original diagnostic seeds are required")
    beta = 1052.5834160132904
    energy_offset = -.7
    replay_max = 0.
    for row in ledger:
        saved = success.get(row["index"])
        if (row["zero_reason"] is None) != (saved is not None):
            raise ValueError("original zero/PES membership differs")
        if saved is not None:
            replay = row["log_importance_prefactor"]-beta*(saved["energy_hartree"]-energy_offset)
            replay_max = max(replay_max, abs(replay-saved["log_weight"]))
    if replay_max > 1e-11:
        raise ValueError("saved importance weights failed an energy/prefactor replay")
    targets = {name: describe([row for row in ledger if row["target"] == name], success) for name in TARGETS}
    for name in TARGETS:
        for key in ("empirical_ESS", "maximum_normalized_weight"):
            if not math.isclose(targets[name][key], report["weight_diagnostics"][name][key], rel_tol=1e-12):
                raise ValueError("weight diagnostic disagrees with the original complete report")
    result = {
        "schema": "pyrpmd.hch4-saved-pilot-proposal-diagnosis.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [{"path": str(path.relative_to(root)), "sha256": sha(path), "bytes": path.stat().st_size}
                   for path in [inventory_path, *selected]],
        "script_sha256": sha(__file__), "all_768_original_draws_retained": True,
        "all_eight_original_seeds_retained": True, "PES_evaluations": 0, "new_random_draws": 0,
        "energy_to_log_weight_max_absolute_replay_error": replay_max,
        "geometric_zero_reasons": dict(Counter(row["zero_reason"] for row in ledger if row["zero_reason"] is not None)),
        "targets": targets,
        "interpretation_limits": [
            "These are descriptive self-normalized importance moments from only about4–7 effective weights per target; they cannot establish a trustworthy11D covariance.",
            "Observed negligible uniform-component or boundary-band weights do not bound unseen or omitted-domain mass.",
            "Reflection folding is used only for geometric summaries; no samples, labels, zero weights or independent-seed units were combined or discarded.",
            "No absolute integral, converted rate, paper-comparison gate or new proposal performance is reported.",
        ],
    }
    with Path(output).open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "sha256": sha(output),
                      "empirical_ESS": {name: targets[name]["empirical_ESS"] for name in TARGETS},
                      "PES_evaluations": 0}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
