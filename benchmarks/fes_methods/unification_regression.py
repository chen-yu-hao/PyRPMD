"""Compare unchanged examples and real legacy-checkpoint continuation.

Run with the same Python/dependencies and native build in both checkouts::

    python benchmarks/fes_methods/unification_regression.py \
        --baseline /path/to/pre-refactor-checkout --output runs/fes-regression

The baseline checkout is read-only apart from its usual Python import caches.
All simulation products and the machine-readable report go below --output.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


EXAMPLES = {
    "umbrella_2d": "examples/fes/harmonic2d.inp",
    "umbrella_adaptive": "examples/fes/adaptive_harmonic.inp",
    "umbrella_exchange": "examples/fes/replica_exchange_2d.inp",
    "metadynamics_standard": "examples/metadynamics/standard.inp",
    "metadynamics_well_tempered": "examples/metadynamics/well_tempered_2d.inp",
    "metadynamics_grid": "examples/metadynamics/grid_fixed_slab.inp",
    "opes": "examples/opes/harmonic_2d.inp",
}
RESUME_EXAMPLES = ("umbrella_2d", "metadynamics_standard", "opes")

WORKER = r'''
import sys
from pathlib import Path
import pyrpmd
from pyrpmd.runner import run, resume_run
from pyrpmd.dynamics import Dynamics

assert Path(pyrpmd.__file__).resolve().parent == Path.cwd() / "src" / "pyrpmd"
example, output, mode = sys.argv[1:]
if mode == "resume":
    resume_run(output)
elif mode == "interrupt":
    original_step = Dynamics.step
    original_block = Dynamics.propagate_block
    def step(self, state, *args, **kwargs):
        if state.step >= 200:
            raise RuntimeError("intentional legacy-regression interruption")
        return original_step(self, state, *args, **kwargs)
    def block(self, state, *args, **kwargs):
        if state.step >= 200:
            raise RuntimeError("intentional legacy-regression interruption")
        return original_block(self, state, *args, **kwargs)
    Dynamics.step, Dynamics.propagate_block = step, block
    try:
        run(example, output=output)
    except RuntimeError as error:
        if str(error) != "intentional legacy-regression interruption":
            raise
    else:
        raise AssertionError("baseline run did not reach the interruption")
else:
    run(example, output=output)
'''


def execute(checkout, example, output, mode="run"):
    environment = dict(os.environ, PYTHONPATH=str(checkout / "src"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".log").open("w") as log:
        subprocess.run([sys.executable, "-c", WORKER,
                        str(checkout / example), str(output), mode],
                       cwd=checkout, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    print(f"{mode}: {output.name}", flush=True)


def compare(reference, candidate):
    """Require exact scientific arrays, including RNG and bias histories."""
    arrays, checkpoints, differences = 0, 0, []
    expected_files = [reference / "fes.npz", *sorted(reference.rglob("checkpoints/*.npz"))]
    for source in expected_files:
        relative = source.relative_to(reference)
        target = candidate / relative
        if not target.is_file():
            differences.append(f"missing {relative}")
            continue
        checkpoints += relative.name != "fes.npz"
        with np.load(source, allow_pickle=False) as before, np.load(target, allow_pickle=False) as after:
            for key in before.files:
                if key in ("fingerprint", "_pyrpmd_checkpoint"):
                    continue
                if key not in after:
                    differences.append(f"missing {relative}:{key}")
                    continue
                left, right = before[key], after[key]
                # File locations and producer metadata are intentionally different.
                if relative.name == "fes.npz" and left.dtype.kind in "US":
                    continue
                arrays += 1
                equal = left.shape == right.shape and np.array_equal(
                    left, right, equal_nan=left.dtype.kind in "fc" and right.dtype.kind in "fc")
                if not equal:
                    differences.append(f"changed {relative}:{key}")
    return {"exact": not differences, "arrays_compared": arrays,
            "checkpoints_compared": checkpoints, "differences": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("all", "baseline", "current"), default="all")
    args = parser.parse_args()
    baseline, output = args.baseline.resolve(), args.output.resolve()
    current = Path(__file__).resolve().parents[2]
    output.mkdir(parents=True, exist_ok=True)
    if args.mode in ("all", "baseline"):
        for name, example in EXAMPLES.items():
            execute(baseline, example, output / "baseline" / name)
        for name in RESUME_EXAMPLES:
            execute(baseline, EXAMPLES[name], output / "legacy_resume" / name, "interrupt")
    if args.mode == "baseline":
        return
    comparisons = {}
    for name, example in EXAMPLES.items():
        execute(current, example, output / "current" / name)
        comparisons[name] = compare(output / "baseline" / name, output / "current" / name)
    continuation = {}
    for name in RESUME_EXAMPLES:
        execute(current, EXAMPLES[name], output / "legacy_resume" / name, "resume")
        continuation[name] = compare(output / "baseline" / name, output / "legacy_resume" / name)
    report = {"baseline_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=baseline, text=True).strip(),
              "examples": comparisons, "legacy_continuation": continuation,
              "all_exact": all(row["exact"] for row in [*comparisons.values(), *continuation.values()])}
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
