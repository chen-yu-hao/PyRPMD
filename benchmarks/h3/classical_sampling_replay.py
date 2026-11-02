"""Materialize an earlier registered prefix for unchanged WHAM analysis.

Later stages retain all observations in one current checkpoint. This tool
recovers the exact registered CV prefix and original RNG/geometry snapshot;
it performs no simulation and does not edit the source campaign. A derived
checkpoint has its own provenance, distinct from the historical zip file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.h3.classical_sampling_campaign import SEEDS, array_hash, file_hash


def prepare(campaign: Path, output: Path, *, stage: int, seeds=SEEDS):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    if campaign == output or campaign in output.parents:
        raise ValueError("derived replay must be outside the source campaign")
    if output.exists() and any(output.iterdir()):
        raise ValueError("use an empty directory for the derived replay")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        source = campaign/f"seed-{seed}"
        receipt_path = source/f"stages/{stage}/stage.json"
        receipt = json.loads(receipt_path.read_text())
        settings = json.loads((source/"settings.json").read_text())
        state_path = receipt_path.parent/"state.npz"
        if file_hash(state_path) != receipt["state_snapshot_sha256"]:
            raise ValueError("registered state snapshot checksum mismatch")
        with np.load(state_path, allow_pickle=False) as saved:
            state = {key: saved[key].copy() for key in saved.files}
        if int(state["completed"]) != stage+settings["warmup"]:
            raise ValueError("registered stage and state counts differ")
        # One open inode remains stable even if a live sampler replaces its path.
        with (source/"checkpoint.npz").open("rb") as handle:
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024*1024), b""):
                digest.update(block)
            checkpoint_hash = digest.hexdigest()
            handle.seek(0)
            with np.load(handle, allow_pickle=False) as saved:
                all_samples = saved["samples"]
                if len(all_samples) < stage:
                    raise ValueError("current checkpoint predates requested stage")
                samples = all_samples[:stage].copy()
                del all_samples
                count = (stage+settings["diagnostic_interval"]-1)//settings["diagnostic_interval"]
                diagnostics = saved["diagnostics"][:count].copy()
        if array_hash(samples) != receipt["samples_prefix_sha256"]:
            raise ValueError("CV prefix differs from the original stage receipt")
        target = output/f"seed-{seed}"
        target.mkdir()
        np.savez_compressed(target/"checkpoint.npz", **state, samples=samples, diagnostics=diagnostics)
        shutil.copyfile(source/"settings.json", target/"settings.json")
        stage_target = target/f"stages/{stage}"
        stage_target.mkdir(parents=True)
        shutil.copyfile(receipt_path, stage_target/"stage.json")
        shutil.copyfile(state_path, stage_target/"state.npz")
        (target/"run-report.json").write_text(json.dumps(receipt["run_report"], indent=2)+"\n")
        rows.append({"seed": seed, "production_sweeps": stage,
                     "current_source_checkpoint_sha256": checkpoint_hash,
                     "historical_checkpoint_sha256": receipt["checkpoint_sha256_at_stage"],
                     "derived_checkpoint_sha256": file_hash(target/"checkpoint.npz"),
                     "samples_sha256": array_hash(samples),
                     "state_snapshot_sha256": receipt["state_snapshot_sha256"],
                     "stage_receipt_sha256": file_hash(receipt_path),
                     "diagnostic_samples": len(diagnostics),
                     "simulation_steps_executed": 0})
    report = {"schema": "pyrpmd.h3.classical-campaign-prefix-replay.v1",
              "stage": stage, "runs": rows,
              "source_sha256": file_hash(Path(__file__)),
              "scope": "Exact registered ordered CV prefixes and saved stage states; new zip-container identities are explicit."}
    (output/"prefix-replay.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.campaign, args.output, stage=args.stage), indent=2))
