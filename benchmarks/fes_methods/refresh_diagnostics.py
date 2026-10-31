"""Reanalyze existing production with moment-aware ESS; verify FES unchanged."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pyrpmd.system import System
from pyrpmd.workflows.fes import reanalyze_fes, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for case in sorted(args.production.iterdir()):
        if not (case / "run/fes.h5").exists():
            continue
        saved = json.loads((case / "input-resolved.json").read_text())
        config = saved["config"]
        if config["task"] != "fes":
            continue
        with np.load(case / "run/fes.npz", allow_pickle=False) as z:
            before = {key: z[key].copy() for key in ("free_energy", "probability_mass", "raw_count", "mask")}
        old_summary = (case / "run/summary.json").read_text()
        result = reanalyze_fes(config, System.from_dict(saved["system"]), case / "run", return_data=True)
        same = {key: np.array_equal(value, result["data"][key], equal_nan=True)
                for key, value in before.items()}
        if not all(same.values()):
            raise AssertionError((case.name, same))
        record = {"case": case.name, "unchanged": same, "new_dynamics_steps": 0,
                  "minimum_production_ess": float(np.min(result["data"]["effective_samples_per_stream"])),
                  "old_summary_sha256": hashlib.sha256(old_summary.encode()).hexdigest(),
                  "note": "Reanalyzed the existing checkpoints with moment-aware ESS and bootstrap block diagnostics; pilot decisions retained."}
        write_json(case / "diagnostic-refresh.json", record)
        records.append(record)
        print(json.dumps(record), flush=True)
    write_json(args.production.parent / "diagnostic-refresh.json", records)


if __name__ == "__main__":
    main()
