"""Matched integration budgets, with the original planner loaded from git.

Needs completed pilot-discovery cases; no oracle is used to place windows.
All pilot costs (including discarded rounds) count against each case's budget.
Longer trajectories with the same seed share prefixes, not independent repeats.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

from optimization import run_case

BASELINE = "6fde29d2039c01921f00393bf17830b939af131b"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline-data", type=Path, required=True)
    p.add_argument("--budget", type=int, default=4000000)
    args = p.parse_args()
    source = subprocess.check_output(["git", "show", BASELINE + ":src/pyrpmd/workflows/adaptive_umbrella.py"])
    snapshot = args.output / "legacy_adaptive_umbrella.py"
    snapshot.write_bytes(source)
    spec = importlib.util.spec_from_file_location("pyrpmd.workflows._legacy_adaptive", snapshot)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    provenance = {"original_commit": BASELINE, "original_workflow_sha256": hashlib.sha256(source).hexdigest(),
                  "note": "Original adaptive workflow loaded as a separate Python module; unchanged shared FES integrator/MBAR."}
    production = args.output / "production"
    library = args.baseline_data / "ogre/build/double_well.so"
    for seed in (4101, 4102, 4103):
        for legacy in (True, False):
            prior = (args.baseline_data / f"ogre/double_well-adaptive-s{seed}-n1" if legacy else
                     production / f"double_well-coverage-tuned-s{seed}")
            d = json.loads((prior / "comparison.json").read_text())
            nprod = ((args.budget-d["pilot_steps"])//d["windows"]-10000)//20*20
            if nprod <= 0:
                raise ValueError("pilot cost already exhausts requested budget")
            changes = {"sampling": {"production_steps": nprod}}
            if not legacy:
                changes["sampling"]["andersen_interval"] = 50
                changes["adaptive_umbrella"] = {"coverage_bins": [4, 8],
                    "coverage_ranges": [[-.5, .5], [-1, 1]], "min_bin_effective_samples": 5,
                    "target_coverage": .95, "max_windows": 45}
            label = ("legacy-budget" if legacy else "coverage-budget") + str(args.budget)
            report = run_case("double_well", "adaptive", seed, production, library, label, changes,
                              workflow=old.run_adaptive_fes if legacy else None,
                              provenance=provenance if legacy else None)
            assert args.budget-20*report["windows"] < report["integration_steps"] <= args.budget


if __name__ == "__main__":
    main()
