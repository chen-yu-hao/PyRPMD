"""Steepest-descent factors of the reduction skeletons (deterministic).

    python run_skeleton.py results/qm.json results/skeleton.json

The skeleton of a reduction is the same path and mass with the bare potential.
The full-dimensional instanton is a saddle of the bare ring-polymer potential,
so its steepest-descent factor has to be compared with that of the skeleton,
not with that of the reduction dressed with a transverse free energy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.subspace_instanton.instanton import steepest_descent_factor  # noqa: E402
from benchmarks.subspace_instanton.models import family  # noqa: E402
from benchmarks.subspace_instanton.reductions import minimum_energy_path, reductions  # noqa: E402
from benchmarks.subspace_instanton.run_qm import tabulated_instanton  # noqa: E402

SKELETON = {"mep": "mep", "fe": "mep", "fe+sct": "mep+sct", "bare": "bare"}


def main() -> int:
    qm = json.load(open(sys.argv[1]))
    out = {"skeleton_of": SKELETON, "models": {}}
    for name, valley in family().items():
        path = minimum_energy_path(valley)
        record = {}
        for ratio, entry in qm["models"][name]["temperatures"].items():
            beta = entry["beta"]
            reduced, _ = reductions(valley, beta, path)
            item = {}
            for key in ("mep", "mep+sct", "bare"):
                x, potential = reduced[key]
                instanton, frequency = tabulated_instanton(x, potential, beta, 256)
                item[key] = steepest_descent_factor(beta, frequency, instanton)
            item["full"] = entry["ln_eps_sd"]
            record[ratio] = item
        out["models"][name] = record
        print(name, {r: {k: round(v, 4) for k, v in item.items()} for r, item in record.items() if float(r) <= 1.0})
    with open(sys.argv[2], "w") as handle:
        json.dump(out, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
