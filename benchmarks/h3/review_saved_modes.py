"""Inspect saved far-reactant ring modes without propagating or evaluating a PES.

Each selected checkpoint is checked against the prior audit's SHA-256 receipt.
The center cutoff is a geometric selection, not a selection on observed variance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.constants import AU_TIME_FS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect(campaigns, output, max_center=.1):
    rows, sources = [], {}
    for label, directory, audit in campaigns:
        directory, audit = Path(directory), Path(audit)
        settings_path = directory / "settings.json"
        settings = json.loads(settings_path.read_text())
        config, system = settings["config"], settings["system"]
        authoritative = {Path(item["file"]).name: item["sha256"]
                         for item in json.loads(audit.read_text())["input_checkpoints"]}
        beads, beta, dt = config["beads"], config["beta"], config["dt_au"]
        masses = np.asarray(system["masses"])
        umbrella = config["umbrella"]
        centers = np.linspace(umbrella["start"], umbrella["stop"], umbrella["count"])
        interval = config["sampling"]["andersen_interval"]
        if beads != 128 or config["sampling"].get("andersen_schedule", "fixed") != "fixed":
            raise ValueError("this endpoint review requires the audited fixed-schedule P128 campaigns")
        sources[label] = {"settings_sha256": sha256(settings_path), "prior_audit_sha256": sha256(audit),
                          "beads": beads, "beta": beta, "dt_au": dt, "interval_steps": interval}
        for path in sorted((directory / "run/umbrella/checkpoints").glob("window_*.npz")):
            window = int(path.stem.split("_")[1])
            if centers[window] > max_center + 1e-12:
                continue
            checksum = sha256(path)
            if authoritative.get(path.name) != checksum:
                raise ValueError(f"checkpoint differs from the prior audit receipt: {path}")
            with load_checkpoint(path, require_integrity=True) as saved:
                q, step = saved["q"].copy(), int(saved["step"])
            if step % interval:
                raise ValueError("the analytic endpoint prediction assumes an integral number of refresh intervals")
            ring = q - q.mean(axis=0, keepdims=True)
            monomer = ring[:, 0]
            dimer_com = np.sum(ring[:, 1:] * masses[None, 1:, None], axis=1) / masses[1:].sum()
            row = {"campaign": label, "checkpoint": path.name, "sha256": checksum,
                   "window_center": float(centers[window]), "step": step,
                   "time_fs": step * dt * AU_TIME_FS,
                   "minimum_bead_incoming_H2_distance_bohr": float(np.min(np.linalg.norm(q[:, :1] - q[:, 1:], axis=-1))),
                   "mode_variance_ratio": {}, "free_collapsed_start_prediction": {}}
            for name, coordinates, mass in (("incoming_atom", monomer, masses[0]),
                                             ("H2_center_of_mass", dimer_com, masses[1:].sum())):
                transformed = np.fft.rfft(coordinates, axis=0, norm="ortho")
                row["mode_variance_ratio"][name] = {}
                for mode in (13, 37):
                    omega = 2 * beads / beta * np.sin(np.pi * mode / beads)
                    expected = beads / (beta * mass * omega**2)
                    row["mode_variance_ratio"][name][str(mode)] = float(np.mean(abs(transformed[mode])**2) / expected)
                    angle = 2 * np.arctan(dt * omega / 2)
                    row["free_collapsed_start_prediction"][str(mode)] = float(1 - np.cos(interval*angle)**(2*(step//interval)))
            rows.append(row)
    if not rows:
        raise ValueError("no checkpoints met the preselected entrance-center range")
    aggregate = {}
    for label in [item[0] for item in campaigns] + ["combined"]:
        selected = [row for row in rows if label == "combined" or row["campaign"] == label]
        stats = {}
        for component in ("incoming_atom", "H2_center_of_mass"):
            stats[component] = {}
            for mode in ("13", "37"):
                values = np.array([row["mode_variance_ratio"][component][mode] for row in selected])
                stats[component][mode] = {"mean_variance_ratio": float(values.mean()),
                                          "between_checkpoint_standard_error": float(values.std(ddof=1)/np.sqrt(len(values)))}
        aggregate[label] = {"independent_task_endpoints": len(selected), "mode_statistics": stats,
                            "minimum_incoming_distance_bohr": min(row["minimum_bead_incoming_H2_distance_bohr"] for row in selected)}
    report = {"schema": "pyrpmd.h3.saved-free-mode-review.v1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "maximum_selected_center": max_center, "source_receipts": sources,
              "script_sha256": sha256(__file__), "aggregate": aggregate, "checkpoints": rows,
              "scope": "Endpoint internal modes from previously generated H3 campaigns; no new dynamics, no PES calls, no fitted target rate.",
              "interpretation_limits": ["Free-mode predictions neglect the weak interaction at finite reactant separation.",
                                         "End-point internal-mode deficits do not determine the sign or magnitude of a full PMF/rate error.",
                                         "The reported standard errors describe independent task endpoints within the selected entrance windows."]}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", nargs=3, action="append", required=True, metavar=("LABEL", "DIRECTORY", "PRIOR_AUDIT_JSON"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-center", type=float, default=.1)
    args = parser.parse_args()
    result = inspect(args.campaign, args.output, args.max_center)
    print(json.dumps(result["aggregate"], indent=2))
