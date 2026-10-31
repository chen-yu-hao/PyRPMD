"""Compare original/tuned metadynamics only on matched supported bins.

Coverage and RMSE on each run's own support are retained alongside the
matched-support comparison. Every comparison fits one additive FES constant.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for case in sorted(args.optimized.glob("double_well-*-s*")):
        if not (case / "comparison.json").exists():
            continue
        report = json.loads((case / "comparison.json").read_text())
        mode, seed = report["method"], report["seed"]
        if mode not in ("standard", "well_tempered"):
            continue
        oldpath = args.baseline / f"double_well-{mode}-s{seed}-n1"
        with np.load(oldpath / "comparison.npz") as before, np.load(case / "comparison.npz") as after:
            for key in ("edges_0", "edges_1", "reference_kjmol"):
                np.testing.assert_array_equal(before[key], after[key])
            common = ~before["mask"] & ~after["mask"]
            if not np.any(common):
                raise ValueError(f"No matched support for {case.name}")
            values = []
            for source in (before, after):
                delta = source["free_energy_kjmol"] - source["reference_kjmol"]
                delta -= delta[common].mean()
                values.append(float(np.sqrt(np.mean(delta[common]**2))))
            oldreport = json.loads((oldpath / "comparison.json").read_text())
            records.append(dict(label=report["label"], seed=seed, steps=report["integration_steps"],
                coverage_before=float((~before["mask"]).mean()), coverage_after=float((~after["mask"]).mean()),
                rmse_own_support_before=oldreport["rmse_kjmol"], rmse_own_support_after=report["rmse_kjmol"],
                common_coverage=float(common.mean()), rmse_common_before=values[0], rmse_common_after=values[1]))
    if not records:
        raise ValueError("No completed metadynamics comparisons found")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "common_support.json").write_text(json.dumps(records, indent=2)+"\n")
    with (args.output / "common_support.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
